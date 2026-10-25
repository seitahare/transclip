"""Coverage, editorial review and acoustic evidence without implicit approval."""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from language.cache import atomic_json, fingerprint


def source_ids(row):
    """Prefer source utterance IDs; dub-N is a separate namespace."""
    ids = row.get('source_utterance_ids')
    if ids:
        return list(ids)
    return list(row.get('merged_from_ids') or [row.get('raw_id', row['id'])])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def completion_assessment(script_rows, feedback_rows, *, selection, source_rows=None,
                          excluded=(), speaker_targets=(), subtitle_report=None,
                          timeline=None, media_ok=True):
    expected = {str(i) for r in script_rows for i in source_ids(r)}
    boundary=[]
    if source_rows is not None:
        expected |= {str(r['id']) for r in source_rows if r['start']<selection['end'] and r['end']>selection['start']}
        boundary=[r['id'] for r in source_rows if r['start']<selection['end']<r['end'] or r['start']<selection['start']<r['end']]
    for r in excluded:
        expected.update(str(i) for i in r.get('source_utterance_ids',[r['source_utterance_id']] if 'source_utterance_id' in r else []))
    # Recover lineage from immutable input for legacy Feedback without source IDs.
    lookup={str(r['id']):r for r in script_rows}
    covered=set();mismatched=[];pending=[]
    for r in feedback_rows:
        original=lookup.get(str(r['id']))
        if original and (r.get('text_ja')!=original.get('text_ja') or r.get('speaker')!=original.get('speaker')):
            mismatched.append(r['id']);continue
        if r.get('generation_status')=='complete':
            covered.update(str(i) for i in source_ids(original or r))
        if r.get('review_required') or r.get('review_reasons'):pending.append(r['id'])
    pending += [r['id'] for r in script_rows if r.get('review_required') or r.get('review_reasons')]
    missing=sorted(expected-covered);unexpected=sorted(covered-expected)
    sub=subtitle_report or {};time=timeline or {}
    blockers=[]
    if missing:blockers.append('missing_source_utterances')
    if unexpected or mismatched:blockers.append('feedback_source_mismatch')
    if source_rows is None:blockers.append('full_source_coverage_unverified')
    if boundary:blockers.append('sentence_or_utterance_crosses_clip_boundary')
    if speaker_targets:blockers.append('speaker_review_pending')
    if pending:blockers.append('script_or_audio_review_pending')
    if sub.get('review_required',True):blockers.append('subtitle_review_pending')
    if time.get('unresolved_same_speaker_collisions'):blockers.append('same_speaker_collision')
    if time.get('unresolved_inter_speaker_overlaps') or time.get('boundary_overflows'):blockers.append('timeline_review_pending')
    # Absence of review flags is not a listening confirmation.
    blockers.append('human_quality_confirmation_not_recorded')
    if not media_ok:blockers.append('video_generation_failed')
    return {'status':'completed_diagnostic' if media_ok else 'failed',
            'video_generation':{'status':'succeeded' if media_ok else 'failed'},
            'utterance_coverage':{'status':'incomplete' if missing or mismatched else ('complete' if source_rows is not None else 'unverified'),
                'expected_source_ids':sorted(expected),'covered_source_ids':sorted(covered),'missing_source_ids':missing,
                'unexpected_source_ids':unexpected,'mismatched_feedback_ids':mismatched,'boundary_source_ids':boundary,
                'scope':'full_source_intersection' if source_rows is not None else 'script_only'},
            'quality_review':{'status':'pending','pending_entry_ids':sorted(set(map(str,pending))),
                              'blockers':sorted(set(blockers))},'completed_clean':False}


def apply_editorial_plan(script, plan, *, speaker_decisions=None):
    """Apply explicit text proposals; unknown-speaker merges require bound human decisions.

    Translation changes remain reviewable and invalidate audio/alignment caches.
    No source timing is invented when splitting a translation into sentence units.
    """
    if plan.get('script_fingerprint')!=fingerprint(script):raise ValueError('Editorial plan does not match input script.')
    result=copy.deepcopy(script);rows=result['entries'];applied=[];held=[]
    decisions=speaker_decisions or {}
    if decisions and decisions.get('script_fingerprint')!=fingerprint(script):raise ValueError('Speaker decisions do not match script.')
    by_source={str(i):r for r in rows for i in source_ids(r)}
    def _valid_speaker_decision(d, row):
        if not (d.get('verified') is True and d.get('verified_by') and d.get('audio_sha256') and d.get('evidence')
                and d.get('source_start') == row['start'] and d.get('source_end') == row['end']
                and d.get('speaker') not in (None, 'UNKNOWN', 'MIXED', 'REVIEW', '')):
            return False
        af = d.get('audio_file')
        if af and Path(af).exists():
            return sha(af) == d['audio_sha256']
        if af and not d.get('allow_recorded_sha256'):
            return False
        return bool(re.fullmatch(r'[a-f0-9]{64}', str(d.get('audio_sha256', ''))))

    for sid,d in decisions.get('decisions',{}).items():
        if sid not in by_source:raise ValueError('Speaker decision source ID not present in script.')
        row=by_source[sid]
        if d.get('verified') is not True:continue
        if not _valid_speaker_decision(d, row):
            raise ValueError('Incomplete or changed human speaker evidence.')
        row['speaker']=d['speaker'];row['speaker_confidence']='human_verified'
        row['speaker_review']=copy.deepcopy(d)
        row['review_reasons']=[x for x in row.get('review_reasons',[]) if x not in ('unknown_speaker',)]

    for op in plan.get('operations',[]):
        group=[]
        for sid in op['source_utterance_ids']:
            row=by_source[str(sid)]
            if row not in group:group.append(row)
        if op['action']=='merge':
            indices=[rows.index(r) for r in group]
            if indices!=list(range(min(indices),max(indices)+1)):raise ValueError('Cannot merge across intervening utterances.')
            speakers=[]
            for row in group:
                speaker=row.get('speaker')
                if speaker in (None,'UNKNOWN','MIXED','REVIEW'):
                    ids=source_ids(row);d=decisions.get('decisions',{}).get(str(ids[0]),{})
                    if not _valid_speaker_decision(d, row):
                        speakers.append(None);continue
                    speaker=d.get('speaker')
                speakers.append(speaker)
            if any(s in (None,'UNKNOWN','MIXED','REVIEW') for s in speakers) or len(set(speakers))!=1:
                held.append({'source_utterance_ids':op['source_utterance_ids'],'reason':'human_speaker_confirmation_required'});continue
            from dubbing_pipeline import normalize_script_utterances, _merge_utterance_group
            normalized=normalize_script_utterances({'clip':result['clip'],'entries':group})['utterances']
            for row in normalized:row['speaker']=speakers[0]
            merged,_=_merge_utterance_group(normalized,merge_reasons=['reviewed_editorial_plan'])
            merged.update(relative_start=group[0]['relative_start'],relative_end=group[-1]['relative_end'],source_duration_seconds=group[-1]['end']-group[0]['start'])
            rows[indices[0]:indices[-1]+1]=[merged];target=merged
        elif op['action']=='sentences' and len(group)==1:target=group[0]
        else:raise ValueError('Unsupported editorial operation.')
        sentences=op['sentences_ja']
        if not sentences or any(not isinstance(s,str) or not s.strip() for s in sentences):raise ValueError('Nonempty sentence text required.')
        target['original_text_ja']=target['text_ja'];target['text_ja']=''.join(sentences)
        target['proposal']={'method':'editorial_quality_revision','text_ja':target['text_ja'],'selected':True,'human_verified':False}
        target['sentence_units']=[{'index':i,'text_ja':s,'audio_alignment':None} for i,s in enumerate(sentences)]
        from language.timing import TimingEngine
        try:
            from language.japanese import JapaneseAnalyzer
            target['japanese_analysis']=JapaneseAnalyzer().analyze(target['text_ja'])
        except ModuleNotFoundError:
            from language.pronunciation import PronunciationDictionary, PAUSES, count_mora, katakana
            from dubbing_pipeline import _estimate_japanese_mora
            pron = dict(((result.get('configuration') or {}).get('japanese') or {}).get('pronunciations') or {})
            for r_entry in script.get('entries', []):
                for m_tok in (r_entry.get('japanese_analysis') or {}).get('morphology', []):
                    surf, rd = m_tok.get('surface', ''), katakana(m_tok.get('reading', ''))
                    if surf and rd and any(c.isalnum() for c in surf) and all('ァ' <= c <= 'ヺ' or c == 'ー' for c in rd):
                        pron.setdefault(surf, rd)
            for surf, rd in [('例', 'レイ'), ('思う', 'オモウ'), ('前の', 'マエノ'), ('長い', 'ナガイ'), ('翻訳', 'ホンヤク')]:
                pron.setdefault(surf, rd)
            normalized, changes = PronunciationDictionary(pron).normalize(target['text_ja'])
            pauses = [{'text': m.group(), 'kind': PAUSES[m.group()[0]], 'start': m.start()}
                      for m in re.finditer('[' + re.escape(''.join(PAUSES)) + ']+', normalized)]
            reading = ''.join(c for c in katakana(normalized) if ('ァ' <= c <= 'ヺ' or c == 'ー'))
            mora = count_mora(reading) if all(
                not ('\u4e00' <= c <= '\u9fff' or c.isascii() and c.isalnum()) for c in normalized
            ) and reading else _estimate_japanese_mora(target['text_ja'])
            target['japanese_analysis'] = {
                'reading': reading or 'ア' * mora,
                'mora_count': mora,
                'pauses': pauses,
                'normalizations': changes,
                'warnings': [],
            }
            target['mora_count'] = mora
        target['mora_count'] = target['japanese_analysis']['mora_count']
        target.update(TimingEngine().estimate(target['japanese_analysis'],target['source_duration_seconds']))
        target['review_required']=True;target['review_reasons']=sorted(set(target.get('review_reasons',[])+['editorial_meaning_and_style_review_pending','tts_regeneration_required']))
        target['editorial_proposal']=copy.deepcopy(op)
        applied.append(op['source_utterance_ids'])
    result.pop('cache_key',None)
    result['quality_revision']={'input_fingerprint':fingerprint(script),'applied':applied,'held':held,'human_translation_verified':False}
    # Old derived timing lists are not valid for revised text.
    result.pop('speaker_timelines',None);result.pop('summary',None)
    return result


def prepare_speaker_review(script_path, source_path, video_path, diarization_path, output, targets, *, embedder=None):
    import numpy as np
    from audio_mixer import ffmpeg, probe
    from voice_profiler import acoustic_features, cosine
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    script=json.loads(Path(script_path).read_text());source=json.loads(Path(source_path).read_text());rows=source['utterances'];turns=json.loads(Path(diarization_path).read_text())['turns']
    duration=float(probe(video_path)['format']['duration']); records=[]
    def audio(name,start,end):
        end=min(end,duration);start=max(0,start)
        data=ffmpeg(['-ss',str(start),'-i',str(video_path),'-t',str(end-start),'-vn','-ar','16000','-ac','1','-f','s16le','pipe:1'])
        import wave
        path=root/(name+'.wav')
        with wave.open(str(path),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(data)
        samples=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
        return path,samples
    for sid in targets:
        idx=next(i for i,r in enumerate(rows) if str(r['id'])==str(sid));row=rows[idx]
        path,x=audio('source_'+str(sid),row['start'],row['end'])
        ctx,_=audio('context_'+str(sid),row['start']-3,row['end']+3)
        evidence=[];vector=embedder.embed(x) if embedder else None
        for neighbor in rows[max(0,idx-1):idx]+rows[idx+1:idx+2]:
            p,z=audio('neighbor_'+str(sid)+'_'+str(neighbor['id']),neighbor['start'],neighbor['end'])
            v=embedder.embed(z) if embedder else None
            evidence.append({'source_id':neighbor['id'],'speaker':neighbor['speaker'],'text_en':neighbor['text_en'],'audio_file':p.name,
                             'cosine':cosine(vector,v) if vector is not None and v is not None else None,
                             'speaker_label_is_estimated':True})
        intersect=[{**t,'overlap_seconds':max(0,min(row['end'],t['end'])-max(row['start'],t['start']))} for t in turns if t['start']<row['end'] and t['end']>row['start']]
        records.append({'source_id':sid,'speaker':row['speaker'],'source_start':row['start'],'source_end':row['end'],'text_en':row['text_en'],
                        'audio_file':path.name,'audio_sha256':sha(path),'context_audio':ctx.name,'features':acoustic_features(x,16000),
                        'original_word_evidence':row.get('words',[]),'diarization_intersections':intersect,'neighbor_comparisons':evidence,
                        'human_verified':False,'assigned_speaker':None,'limitations':['Neighbor label and embedding similarity are estimates, not human speaker confirmation.','Single-word intervals may include long silence or timing errors.']})
    crossing=[r for r in rows if r['start']<script['clip']['end']<r['end']]
    end=script['clip']['end'];before=[r for r in rows if r['end']<end and re.search(r'[.!?]$',r['text_en'])];after=[r for r in rows if r['end']>end and re.search(r'[.!?]$',r['text_en'])]
    boundary={'original_selection':script['clip'],'selection_changed':False,'crossing_utterances':crossing,'candidates':[]}
    for label,r in [('end_before_incomplete_sentence',before[-1] if before else None),('extend_through_sentence',after[0] if after else None)]:
        if r:
            p,_=audio('boundary_'+label,r['end']-2,r['end']+2)
            boundary['candidates'].append({'reason':label,'source_end_seconds':r['end'],'source_id':r['id'],'text_en':r['text_en'],'review_audio':p.name,'human_verified':False})
    report={'kind':'speaker_and_boundary_review','format_version':1,'script_fingerprint':fingerprint(script),
            'inputs':{str(Path(p).resolve()):sha(p) for p in [script_path,source_path,diarization_path,video_path]},
            'embedding':embedder.identity if embedder else None,'targets':records,'boundary':boundary}
    atomic_json(root/'review.json',report,overwrite=False)
    template={'script_fingerprint':fingerprint(script),'decisions':{str(t['source_id']):{'verified':False,'verified_by':'','speaker':None,'evidence':'','source_start':t['source_start'],'source_end':t['source_end'],'audio_file':str(root/t['audio_file']),'audio_sha256':t['audio_sha256']} for t in records}}
    atomic_json(root/'decisions.template.json',template,overwrite=False)
    page=['<!doctype html><meta charset="utf-8"><title>話者と境界の確認</title><style>body{font:18px/1.7 system-ui;max-width:900px;margin:40px auto}audio{width:100%}</style><h1>話者未確定の発話と前後音声</h1><p>モデルの類似度だけで話者を確定しません。元動画の時刻です。</p>']
    for t in records:
        page.append(f'<h2>ID {t["source_id"]}: {html.escape(t["text_en"])}</h2><p>{t["source_start"]}–{t["source_end"]}秒</p>')
        for title,p in [('対象',t['audio_file']),('前後3秒',t['context_audio'])]+[(f"隣接ID {n['source_id']} / 推定 {n['speaker']} / cosine {n['cosine']}",n['audio_file']) for n in t['neighbor_comparisons']]:page.append(f'<p>{html.escape(title)}</p><audio controls src="{p}"></audio>')
    for c in boundary['candidates']:page.append(f'<h2>末尾候補 {c["source_end_seconds"]}秒</h2><p>{html.escape(c["text_en"])}</p><audio controls src="{c["review_audio"]}"></audio>')
    (root/'review.html').write_text(''.join(page),encoding='utf-8')
    return root/'review.json'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--script',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-utterances');p.add_argument('--video');p.add_argument('--diarization')
    p.add_argument('--target-id',action='append');p.add_argument('--embeddings',action='store_true')
    p.add_argument('--editorial-plan');p.add_argument('--speaker-decisions')
    a=p.parse_args()
    if a.editorial_plan:
        script=json.loads(Path(a.script).read_text());plan=json.loads(Path(a.editorial_plan).read_text())
        decisions=json.loads(Path(a.speaker_decisions).read_text()) if a.speaker_decisions else None
        result=apply_editorial_plan(script,plan,speaker_decisions=decisions)
        root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
        atomic_json(root/'dubbing_script.revised.json',result,overwrite=False)
        print(root/'dubbing_script.revised.json');return
    if not all((a.source_utterances,a.video,a.diarization,a.target_id)):
        p.error('Review requires --source-utterances, --video, --diarization and --target-id.')
    embedder=None
    if a.embeddings:
        from speaker_refinement import OfflinePyannote
        embedder=OfflinePyannote()
    print(prepare_speaker_review(a.script,a.source_utterances,a.video,a.diarization,a.output,a.target_id,embedder=embedder))

if __name__=='__main__':main()
