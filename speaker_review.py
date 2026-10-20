"""Local listening review and explicit human decisions in separate immutable outputs."""
import argparse
import copy
import html
import json
from pathlib import Path
import re
import tempfile
import shutil
import os

from audio_mixer import read_json, sha256_file
from language.cache import atomic_json, fingerprint

FOCUS=['Dario','famously, what',"You're building humanoid robots.",'Yes, he','like,']


def build_review(folder,analysis,utterances,profiles,samples,original):
    from speaker_refinement import crop,save_audio
    folder=Path(folder);(folder/'review_audio').mkdir()
    records=[]
    for row in utterances:
        if not row['review_required'] and row['speaker'] is not None: continue
        a,b=row['relative_start'],row['relative_end'];left=max(0,a-.65);right=min(len(samples)/16000,b+.65)
        name=f"review_audio/utterance_{row['id']:03d}.wav"
        save_audio(folder/name,crop(samples,left,right))
        exact=f"review_audio/utterance_{row['id']:03d}_exact.wav"
        save_audio(folder/exact,crop(samples,a,b))
        records.append({'utterance_id':row['id'],'text_en':row['text_en'],'relative_start':a,'relative_end':b,
                        'original_speaker':row.get('original_speaker',row['speaker']),'speaker':row['speaker'],
                        'proposed_speaker':row.get('proposed_speaker'),'candidates':row.get('acoustic_candidates',[]),
                        'temporal_candidates':row.get('temporal_candidates',[]),
                        'reason':row['review_reasons'],'words':row['words'],'audio_file':name,'exact_audio_file':exact,
                        'context_start':left,'context_end':right,'audio_sha256':sha256_file(folder/name),
                        'focus':row['text_en'].strip().lower() in [s.lower() for s in FOCUS],
                        'acoustic_evidence_unavailable_reason':row.get('acoustic_evidence_unavailable_reason'),
                        **({'attribution':row['attribution']} if 'attribution' in row else {})})
    bundle={'format_version':1,'kind':'speaker_review','analysis_key':analysis['analysis_key'],
            'video_id':analysis['video_id'],'clip':analysis['clip'],'speakers':analysis['speakers'],
            'original_utterances_fingerprint':fingerprint(original),'items':records,
            'reference_candidates':[r for p in profiles['profiles'] for r in p['reference_candidates'] if r.get('audio_file')]}
    bundle['review_key']=fingerprint(bundle)
    atomic_json(folder/'speaker_review.json',bundle,overwrite=False)
    atomic_json(folder/'review_decisions.template.json',{'format_version':1,'analysis_key':analysis['analysis_key'],
                'review_key':bundle['review_key'],'reviewed_by':'','decisions':[],'reference_edits':[]},overwrite=False)
    cards=[]
    for row in sorted(records,key=lambda r:(not r['focus'],r['original_speaker'] is not None,r['relative_start'])):
        uid=row['utterance_id'];esc=html.escape
        options='<option value="">未確認・変更しない</option><option value="UNKNOWN">UNKNOWN</option><option value="MIXED">複数話者の混在</option>'+''.join(f'<option>{esc(s)}</option>' for s in analysis['speakers'])
        scores=', '.join(f"{c['speaker_id']}: {c['cosine_similarity']:.3f}" for c in row['candidates']) or '独立した埋め込み候補なし（短すぎる／重なりなど）'
        temporal=', '.join(f"{c['speaker']}: {c['coverage']:.1%} / {c['overlap_seconds']:.3f}秒" for c in row['temporal_candidates']) or '対応する話者区間なし'
        word_evidence='\n'.join(f"{w['word']}: {w.get('speaker')} | {w.get('assignment_method')} | {', '.join(w.get('review_reasons',[]))}" for w in row['words'])
        attribution_details=''
        if row.get('attribution'):
            evidence=row['attribution']
            auto=evidence['automatic']; human=evidence['human_confirmation']
            auto_label=auto['speaker_id'] or ('複数話者の混在（モデル推定）' if auto['status']=='mixed' else 'UNKNOWN')
            attribution_details='<p><strong>新しい自動判定: '+esc(auto_label)+'</strong> / '+esc(auto['method'])+'</p>'
            attribution_details+='<p>人間の確認: '+esc(json.dumps(human,ensure_ascii=False) if human is not None else '未確認')+'</p>'
            attribution_details+='<p>同時発話・境界の見立て: '+esc(', '.join(auto['overlap']['hypotheses']) or '強い混在根拠なし')+'</p>'
            attribution_details+='<p>次の値は未校正です。混在票は同じpyannoteモデルの推定であり、同時発話の正解確率ではありません。</p><table><tr><th>音響窓</th><th>時刻</th><th>候補の類似度</th><th>混在票</th></tr>'
            for w in evidence['windows']:
                candidates=', '.join(f"{c['speaker_id']}: {c['cosine_similarity']:.3f}" for c in w['candidates']) or '短すぎるため取得せず'
                vote=w['activity']['overlap_vote']
                attribution_details+=f"<tr><td>{esc(w['scope'])}</td><td>{w['relative_start']:.2f}–{w['relative_end']:.2f}</td><td>{esc(candidates)}</td><td>{vote:.3f}</td></tr>" if vote is not None else f"<tr><td>{esc(w['scope'])}</td><td>音響情報不足</td><td>{esc(candidates)}</td><td>不明</td></tr>"
            attribution_details+='</table>'
            concise={k:v for k,v in evidence.items() if k not in ('duration_reference_checks','windows')}
            concise['windows']=[{k:v for k,v in w.items() if k!='embedding'} for w in evidence['windows']]
            attribution_details+='<details><summary>特徴・しきい値・前後の会話・判断根拠</summary><pre>'+esc(json.dumps(concise,ensure_ascii=False,indent=2))+'</pre></details>'
        cards.append(f'''<article class="{'focus' if row['focus'] else ''}"><small>発話 {uid} · {row['relative_start']:.2f}–{row['relative_end']:.2f}秒 {'· 重点確認' if row['focus'] else ''}</small>
<h2>{esc(row['text_en'])}</h2><p>以前: {esc(str(row['original_speaker'] or 'UNKNOWN'))} / 現在: {esc(str(row['speaker'] or 'UNKNOWN'))} / 単語からの候補: {esc(str(row['proposed_speaker'] or 'なし'))}</p>
<p>対象そのもの</p><audio controls preload="none" src="{row['exact_audio_file']}"></audio><p>前後0.65秒の文脈を含む音声（別話者を含む可能性あり）</p><audio controls preload="none" src="{row['audio_file']}"></audio>
<p>時間の重なりによる候補（確率ではありません）: {esc(temporal)}</p><p>音響類似度（確率ではありません）: {esc(scores)}</p><p class="reason">{esc(', '.join(row['reason']))}</p>
<details><summary>単語ごとの根拠と矛盾</summary><pre>{esc(word_evidence)}</pre></details>
{attribution_details}
<label>確認結果 <select data-uid="{uid}">{options}</select></label><label>判断の根拠 <input data-note="{uid}" placeholder="聞き比べた特徴や残る疑問"></label></article>''')
    refs=[]
    for r in bundle['reference_candidates']:
        rid=r['reference_id']
        refs.append(f'''<article><h3>{html.escape(r['speaker_id'])} · {rid}</h3><p>{r['relative_start']:.2f}–{r['relative_end']:.2f}秒。検出済み重なりなし・未聴取確認・クローン許可なし。</p>
<audio controls preload="none" src="{r['audio_file']}"></audio><p>自動文字起こし: {html.escape(r['transcript']['asr_text'])}</p>
<label>修正文字起こし（変更時のみ）<textarea data-ref="{rid}"></textarea></label></article>''')
    payload=json.dumps(bundle,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TransClip 話者確認</title>
<style>body{font:16px/1.6 system-ui;background:#f3f5f7;color:#15283a;max-width:1100px;margin:auto;padding:30px}h1{font-size:32px}article{background:white;border:1px solid #dce2e8;border-radius:14px;padding:22px;margin:18px 0}article.focus{border-left:6px solid #217d8d}audio{width:100%;max-width:600px}small,.reason{color:#536471}label{display:block;margin-top:16px}input,select,textarea,button{font:inherit;padding:10px;border:1px solid #9caebb;border-radius:6px}input,textarea{box-sizing:border-box;width:100%}pre{white-space:pre-wrap}button{background:#175b6a;color:white;cursor:pointer}header{background:#e4f0f1;padding:22px;border-radius:14px}summary{cursor:pointer}</style>
<header><h1>話者を聞き比べて確認</h1><p>音声の特徴から人物の実名は特定しません。類似度はモデルの推定であり、正解率ではありません。短い発話は前後の別話者に引っ張られることがあります。判断できなければUNKNOWNのままにしてください。</p>
<label>確認者の名前または識別名<input id="reviewer" placeholder="確認結果に保存されます"></label><p><button id="download">確認結果を別JSONへ保存</button> <span id="status" role="status"></span></p><p>元データは変更しません。未確認項目は保存対象外。保存したJSONはSpeaker Review CLIで取り込めます。</p></header>
<h2>比較用の参考音声</h2>'''+''.join(refs)+'<h2>確認対象</h2>'+''.join(cards)+'''<script>
const data='''+payload+''';
document.querySelector('#download').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){document.querySelector('#status').textContent='確認者を入力してください';return;}
 const decisions=[...document.querySelectorAll('select[data-uid]')].filter(x=>x.value).map(x=>{const id=Number(x.dataset.uid),item=data.items.find(r=>r.utterance_id===id);return {utterance_id:id,speaker:['UNKNOWN','MIXED'].includes(x.value)?null:x.value,attribution_status:x.value==='MIXED'?'mixed':x.value==='UNKNOWN'?'unknown':'single',audio_sha256:item.audio_sha256,note:document.querySelector(`[data-note="${id}"]`).value.trim()};});
 const reference_edits=[...document.querySelectorAll('textarea[data-ref]')].filter(x=>x.value.trim()).map(x=>{const ref=data.reference_candidates.find(r=>r.reference_id===x.dataset.ref);return {reference_id:ref.reference_id,audio_sha256:ref.audio_sha256,corrected_text:x.value.trim()};});
 const result={format_version:1,analysis_key:data.analysis_key,review_key:data.review_key,reviewed_by:reviewer,reviewed_at:new Date().toISOString(),decisions,reference_edits};
 const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='speaker_review_decisions.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.querySelector('#status').textContent=`${decisions.length}件の判断・${reference_edits.length}件の文字起こし修正を保存`;
};</script></html>'''
    (folder/'speaker_review.html').write_text(page,encoding='utf-8')
    return bundle


def validate_decisions(bundle,decisions):
    expected=fingerprint({k:v for k,v in bundle.items() if k!='review_key'})
    if expected!=bundle['review_key']: raise ValueError('Review bundle content hash mismatch.')
    if (decisions.get('format_version')!=1 or decisions.get('analysis_key')!=bundle['analysis_key']
            or decisions.get('review_key')!=bundle['review_key']): raise ValueError('Stale or unrelated review decisions.')
    if not isinstance(decisions.get('reviewed_by'),str) or not decisions['reviewed_by'].strip():
        raise ValueError('Explicit human reviewer is required.')
    items={r['utterance_id']:r for r in bundle['items']};seen=set()
    for d in decisions.get('decisions',[]):
        uid=d.get('utterance_id')
        if uid in seen or uid not in items: raise ValueError('Duplicate/unknown review utterance ID.')
        seen.add(uid)
        if 'speaker' not in d: raise ValueError('Reviewed speaker must be explicit.')
        if d.get('speaker') is not None and d['speaker'] not in bundle['speakers']: raise ValueError('Invalid reviewed speaker ID.')
        status=d.get('attribution_status', 'unknown' if d['speaker'] is None else 'single')
        if status not in ('single','mixed','unknown') or (status=='single') != (d['speaker'] is not None):
            raise ValueError('Inconsistent human attribution status.')
        if d.get('audio_sha256')!=items[uid]['audio_sha256']: raise ValueError('Review audio hash mismatch.')
        if not isinstance(d.get('note'),str): raise ValueError('Review note must be text.')
    refs={r['reference_id']:r for r in bundle['reference_candidates']};seen=set()
    for d in decisions.get('reference_edits',[]):
        rid=d.get('reference_id')
        if rid in seen or rid not in refs: raise ValueError('Duplicate/unknown reference ID.')
        seen.add(rid)
        if d.get('audio_sha256')!=refs[rid]['audio_sha256'] or not isinstance(d.get('corrected_text'),str) or not d['corrected_text'].strip():
            raise ValueError('Invalid reference transcript correction.')
    return decisions


def apply_review(analysis_dir,decisions_path,input_dir,output_dir):
    folder=Path(analysis_dir).resolve();inputs=Path(input_dir).resolve();target=Path(output_dir).resolve()
    bundle=read_json(folder/'speaker_review.json');decisions=validate_decisions(bundle,read_json(decisions_path))
    analysis_path=folder/('speaker_timeline.json' if (folder/'speaker_attribution.json').exists() else 'diarization.enhanced.json')
    analysis=read_json(analysis_path)
    source=read_json(inputs/'utterances.json');translated=read_json(inputs/'utterance_translations.json')
    selection=read_json(inputs/'selection.json')
    if (fingerprint(source)!=bundle['original_utterances_fingerprint'] or fingerprint(translated)!=analysis['input_fingerprints']['utterance_translations']
            or fingerprint(selection)!=analysis['input_fingerprints']['selection']): raise ValueError('Original inputs changed since review.')
    for r in bundle['items']+bundle['reference_candidates']:
        path=(folder/r['audio_file']).resolve()
        if not path.is_relative_to(folder) or sha256_file(path)!=r['audio_sha256']: raise ValueError('Review audio changed.')
    updates={d['utterance_id']:d for d in decisions.get('decisions',[])}
    # Preserve Phase 5 automatic results for items not explicitly reviewed.
    if analysis.get('attribution_file'):
        from speaker_attribution import verify_bundle
        verify_bundle(folder)
        source=read_json(folder/'utterances.json');translated=read_json(folder/'utterance_translations.json')
    # Source utterance IDs/text/timing stay unchanged. Translation is not regenerated.
    for row in source['utterances']:
        if row['id'] not in updates: continue
        d=updates[row['id']];row['speaker']=d['speaker']
        row['attribution_status']=d.get('attribution_status','unknown' if d['speaker'] is None else 'single')
        row['attribution_method']='human_review'
        row['human_review']={'reviewed_by':decisions['reviewed_by'],'note':d['note'],'analysis_key':bundle['analysis_key'],'attribution_status':row['attribution_status']}
        row['review_required']=True  # Meaning, timing and other inherited concerns remain unresolved.
        row['review_reasons']=sorted(set(row.get('review_reasons',[])+['human_speaker_review_recorded']))
        for word in row['words']:
            word['speaker']=d['speaker'];word['assignment_method']='human_review';word['review_required']=True
    for row in translated['utterances']:
        if row['id'] in updates:
            d=updates[row['id']];row['speaker']=d['speaker'];row['review_required']=True
            row['attribution_status']=d.get('attribution_status','unknown' if d['speaker'] is None else 'single')
            row['attribution_method']='human_review'
    source['cache_key']=fingerprint(source['utterances']);translated['cache_key']=fingerprint(translated['utterances'])
    profiles=read_json(folder/'speaker_profiles.json');edits={d['reference_id']:d for d in decisions.get('reference_edits',[])}
    for p in profiles['profiles']:
        for r in p['reference_candidates']:
            if r.get('audio_file'): r['audio_file']=str(folder/r['audio_file'])
            if r['reference_id'] in edits:
                r['transcript'].update(corrected_text=edits[r['reference_id']]['corrected_text'],reviewed=True,reviewed_by=decisions['reviewed_by'])
    timeline={**analysis,'utterances_file':'utterances.json','human_review_decisions_fingerprint':fingerprint(decisions),
              'note':'Turns retain model evidence; utterance-level human decisions are in utterances.json.'}
    output={'review_decisions.json':decisions,'speaker_timeline.json':timeline,'utterances.json':source,'utterance_translations.json':translated,
            'selection.json':selection,'speaker_profiles.json':profiles}
    if target.exists():
        if all((target/n).is_file() and read_json(target/n)==v for n,v in output.items()): return target
        raise ValueError('Review output exists with different contents; choose a new path.')
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.pending-review-',dir=target.parent))
    try:
        for n,v in output.items(): atomic_json(temp/n,v,overwrite=False)
        os.rename(temp,target)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return target


def build_separation_review(folder, bundle):
    """Phase 6 review supports arbitrary stem counts in a separate decision schema."""
    from separation_review import write_review
    return write_review(folder, bundle)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Apply explicit listening decisions without modifying original inputs.')
    parser.add_argument('--analysis-dir',type=Path,required=True);parser.add_argument('--decisions',type=Path,required=True)
    parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args(argv)
    try:
        print(apply_review(a.analysis_dir,a.decisions,a.input_dir,a.output_dir));return 0
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'Error: {error}');return 1


if __name__=='__main__': raise SystemExit(main())
