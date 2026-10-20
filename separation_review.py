"""Speaker Review extension for any number of stems and separate human decisions."""
import argparse
import copy
import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from audio_mixer import read_json, sha256_file
from voice_profiler import overlap_regions
from language.cache import atomic_json, fingerprint

ISSUES={'MIXED','UNKNOWN','ASR_ERROR','SEPARATION_FAILED','OTHER'}


def write_review(folder,bundle):
    folder=Path(folder);items=[];cards=[];esc=html.escape
    for job in bundle['jobs']:
        body=f'<h2>{esc(job["job_id"])} · {job["core_start"]:.3f}–{job["core_end"]:.3f}秒</h2><p>{esc(job["status"])}</p><p>元のASR: {esc(job.get("original_text",""))}</p>'
        if job.get('original_audio_file'):
            url=quote(os.path.relpath(job['original_audio_file'],folder));body+=f'<h3>元音声（前後の文脈を含む）</h3><audio controls preload="none" src="{url}"></audio>'
        body+='<pre>'+esc(json.dumps(job['overlap_status'],ensure_ascii=False,indent=2))+'</pre>'
        for s in job['stems']:
            item={'stem_id':s['stem_id'],'audio_sha256':s['audio_sha256'],'audio_file':s['audio_file']};items.append(item)
            url=quote(os.path.relpath(s['audio_file'],folder));index=len(items)-1
            options='<option value="">未確認</option>'+''.join(f'<option>{esc(v)}</option>' for v in [*bundle['detected_speakers'],*sorted(ISSUES)])
            evidence={k:s[k] for k in ('speaker_id','speaker_candidates','speaker_confidence_metrics','review_reasons','features')}
            text=' / '.join(r['speaker_id']+': '+r['text'] for r in s['transcriptions'])
            body+=f'<section><h3>{esc(s["stem_id"])}</h3><audio controls preload="none" src="{url}"></audio><p>分離後／区間ASR: {esc(text) or "認識された単語なし"}</p><select data-index="{index}">{options}</select><label><input type="checkbox" data-text="{index}">文字起こし・欠落を聴取確認済み</label><textarea data-note="{index}" placeholder="問題・修正候補（ここでは単語時刻を捏造せず、下流出力を保留します）"></textarea><details><summary>類似度・品質・判定根拠</summary><pre>{esc(json.dumps(evidence,ensure_ascii=False,indent=2))}</pre></details></section>'
        if job.get('error'):body+='<p>'+esc(job['error'])+'</p>'
        cards.append('<article>'+body+'</article>')
    review={'format_version':1,'kind':'separation_review','run_key':bundle['run_key'],'bundle_fingerprint':fingerprint(bundle),'items':items}
    review['review_key']=fingerprint(review);atomic_json(folder/'separation_review.json',review)
    template={'format_version':1,'review_key':review['review_key'],'reviewed_by':'','source_coverage_confirmed':False,'decisions':[]}
    atomic_json(folder/'review_decisions.template.json',template)
    payload=json.dumps({'template':template,'items':items},ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Speaker Review · 音声分離</title><style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;background:#f5f7fa;color:#172333}article{background:white;padding:24px;margin:22px 0;border:1px solid #cad2df;border-radius:12px}section{border-top:1px solid #ddd;padding:18px 0}audio{display:block;width:100%;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}textarea{display:block;width:95%;margin-top:12px}select,input,button{font:inherit;margin:8px}label{display:block}</style><h1>Speaker Review · 分離音声の比較</h1><p>モデルの推定は未確認です。Stemの順番は話者IDではありません。元音声と聞き比べてください。UNKNOWN・混在・認識誤りは下流への確定出力を保留します。</p><label>確認者<input id="reviewer"></label><label><input id="coverage" type="checkbox">全区間を聴取し、元の発言の欠落・追加がないことを確認した</label><button id="save">判断を別JSONへ保存</button>'''+''.join(cards)+f'<script id="payload" type="application/json">{payload}</script>'+'''
<script>const data=JSON.parse(document.getElementById('payload').textContent);document.getElementById('save').onclick=()=>{const out=structuredClone(data.template);out.reviewed_by=document.getElementById('reviewer').value.trim();if(!out.reviewed_by){alert('確認者を入力してください');return;}out.source_coverage_confirmed=document.getElementById('coverage').checked;document.querySelectorAll('select[data-index]').forEach(el=>{if(el.value){const i=el.dataset.index;out.decisions.push({...data.items[i],label:el.value,text_confirmed:document.querySelector(`[data-text="${i}"]`).checked,note:document.querySelector(`[data-note="${i}"]`).value});}});const url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='separation_review_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script></html>'''
    (folder/'review.html').write_text(page,encoding='utf-8')


def validate_decisions(bundle,review,decisions):
    expected={k:v for k,v in review.items() if k!='review_key'}
    if fingerprint(expected)!=review.get('review_key') or review.get('bundle_fingerprint')!=fingerprint(bundle):raise ValueError('Review bundle changed.')
    if decisions.get('format_version')!=1 or decisions.get('review_key')!=review['review_key'] or not str(decisions.get('reviewed_by','')).strip():raise ValueError('Human review identity missing or stale.')
    items={i['stem_id']:i for i in review['items']};labels={}
    for d in decisions.get('decisions',[]):
        sid=d.get('stem_id')
        if sid not in items or sid in labels:raise ValueError('Unknown/duplicate stem decision.')
        item=items[sid]
        if d.get('audio_sha256')!=item['audio_sha256'] or sha256_file(item['audio_file'])!=item['audio_sha256']:raise ValueError('Reviewed audio changed.')
        if d.get('label') not in set(bundle['detected_speakers'])|ISSUES:raise ValueError('Invalid speaker/issue label.')
        if type(d.get('text_confirmed')) is not bool:raise ValueError('text_confirmed must be boolean.')
        labels[sid]=copy.deepcopy(d)
    return labels


def export_reviewed(bundle,review,decisions):
    """Strict adapter: no unreviewed partial replacement, no fabricated translations/word times."""
    labels=validate_decisions(bundle,review,decisions)
    if decisions.get('source_coverage_confirmed') is not True:raise ValueError('Human source coverage confirmation required.')
    rows=[];source=bundle['source_utterances'];next_id=max([r['id'] for r in source]+[0])+1
    a,b=bundle['range']['relative_start'],bundle['range']['relative_end']
    if any(u['relative_start']<a<u['relative_end'] or u['relative_start']<b<u['relative_end'] for u in source):raise ValueError('Processing range cuts a source utterance; expand range before exporting.')
    for job in bundle['jobs']:
        if job['status'] in ('execution_failed','unseparated_capacity'):raise ValueError('Failed/unseparated job requires resolution.')
        for stem in job['stems']:
            d=labels.get(stem['stem_id'])
            if not d or d['label'] in ISSUES-{'UNKNOWN'} or not d['text_confirmed'] or d.get('note','').strip():raise ValueError('Every stem needs resolved speaker/text review without outstanding corrections.')
            for r in stem['transcriptions']:
                speaker=None if d['label']=='UNKNOWN' else d['label'];words=copy.deepcopy(r['words'])
                for w in words:w.update(speaker=speaker,assignment_method='human_review',review_required=speaker is None)
                rows.append({'id':next_id,'speaker':speaker,'text_en':r['text'],'words':words,'word_count':len(words),
                             **{k:r[k] for k in ('start','end','relative_start','relative_end')},
                             'source_utterance_ids':r['source_utterance_ids'],'transcript_id':r['transcript_id'],
                             'review_required':speaker is None,'review_reasons':['unknown_speaker'] if speaker is None else [],
                             'human_confirmation':d,'automatic_speaker':r['speaker_id']});next_id+=1
    if not rows:raise ValueError('No reviewed speech; empty replacement prohibited.')
    covered={uid for r in rows for uid in r['source_utterance_ids']}
    required={u['id'] for u in source if u['relative_start']<b and u['relative_end']>a}
    if not required<=covered:raise ValueError('Source utterance has no replacement; speech loss requires review.')
    rows += [copy.deepcopy(u) for u in source if u['id'] not in required]
    rows.sort(key=lambda r:(r['relative_start'],r['id']))
    # Preserve overlaps; report same-speaker collisions for the existing optimizer/mixer.
    collisions=[{'first_id':x['id'],'second_id':y['id'],'speaker':x['speaker']} for i,x in enumerate(rows) for y in rows[i+1:] if x['speaker'] is not None and x['speaker']==y['speaker'] and x['relative_start']<y['relative_end'] and y['relative_start']<x['relative_end']]
    return {'format_version':3,'video_id':bundle['video_id'],'clip':bundle['clip'],'speakers':bundle['detected_speakers'],
            'utterances':rows,'source_word_count':sum(len(r.get('words',[])) for r in rows),'separation_run_key':bundle['run_key'],
            'human_review_fingerprint':fingerprint(decisions),'same_speaker_collisions':collisions,
            'downstream':{'translation_required':True,'regenerate_dubbing_script':True,'regenerate_tts':True,
                          'old_script_and_feedback_reuse_allowed':False,'preserve_different_speaker_overlap':True}}


def mixer_evidence(source):
    """Existing Audio Mixer --speaker-analysis accepts this compatibility envelope."""
    turns=[{'speaker':u['speaker'],**{k:u[k] for k in ('start','end','relative_start','relative_end')}} for u in source['utterances'] if u['speaker'] is not None]
    return {'format_version':1,'kind':'speaker_analysis','analysis_key':source['human_review_fingerprint'],
            'video_id':source['video_id'],'clip':source['clip'],'utterances_file':'utterances.json',
            'speakers':source['speakers'],'turns':turns,'source_overlap_regions':overlap_regions(turns),
            'provenance':{'separation_run_key':source['separation_run_key'],'human_review_fingerprint':source['human_review_fingerprint']}}


def comparison_report(bundle):
    lines=['# 音声分離・比較レポート','', '波形生成は分離品質の確認ではありません。実音声の正解ラベル・聴取確認は未取得です。', '',json.dumps(bundle['metrics'],ensure_ascii=False,indent=2),'','|区間|処理|元のASR|音声別ASRと話者候補|','|---|---|---|---|']
    for j in bundle['jobs']:
        texts=' / '.join(' ; '.join(t['speaker_id']+': '+t['text'] for t in s['transcriptions']) or s['speaker_id']+': [ASR empty]' for s in j['stems'])
        lines.append(f'|{j["core_start"]:.3f}–{j["core_end"]:.3f}|{j["status"]}|{j.get("original_text", "").replace("|","/")}|{texts.replace("|","/")}|')
    lines+=['','## 元発話との対応（範囲内）','','|ID|元発話|元の話者|処理区間・候補|','|---|---|---|---|']
    for u in bundle['source_utterances']:
        js=[j for j in bundle['jobs'] if u['id'] in j['source_utterance_ids']]
        if js:lines.append(f'|{u["id"]}|{u["text_en"].replace("|","/")}|{u["speaker"] or "UNKNOWN"}|'+ '; '.join(j['job_id']+': '+j['status']+' ('+', '.join(sorted({r['speaker_id'] for s in j['stems'] for r in s['transcriptions']}))+')' for j in js)+'|')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bundle=read_json(a.run/'speaker_separation.json');review=read_json(a.run/'separation_review.json');decisions=read_json(a.decisions)
    result=export_reviewed(bundle,review,decisions);a.output.mkdir(parents=True,exist_ok=True)
    target=Path(tempfile.mkdtemp(prefix='reviewed-',dir=a.output));atomic_json(target/'utterances.json',result,overwrite=False)
    atomic_json(target/'speaker_analysis.json',mixer_evidence(result),overwrite=False)
    atomic_json(target/'human_decisions.json',decisions,overwrite=False);atomic_json(target/'selection.json',bundle['clip'],overwrite=False)
    print(target)

if __name__=='__main__':main()
