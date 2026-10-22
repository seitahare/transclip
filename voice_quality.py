"""Reference/preset/clone listening comparisons; scores never approve a voice.

Reuses TRANS voice_qa's reference and within-speaker comparisons with existing
TransClip acoustic/embedding adapters. Handles one or many speakers without
inventing a small-sample acceptance threshold.
"""
import html
import itertools
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from language.cache import atomic_json, fingerprint
from language.timing import measure_wav
import voice_profiler
from voice_profiler import acoustic_features, cosine


def compare_voices(manifest_path, output, *, embedder=None):
    from audio_mixer import ffmpeg
    from voice_cloning import digest
    source=Path(manifest_path).resolve();request=json.loads(source.read_text())
    if request.get('kind')!='voice_quality_request' or request.get('format_version')!=1:
        raise ValueError('Expected voice_quality_request v1.')
    refs=request.get('references',{});samples=request.get('samples',[])
    if not refs or not samples:raise ValueError('References and samples are required.')
    if any(s in ('UNKNOWN','MIXED','REVIEW','',None) for s in refs):raise ValueError('Known speaker labels required.')
    ids=[r['id'] for r in samples]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample IDs.')
    files={}
    def resolve(value):
        path=(source.parent/value).resolve()
        if str(path) not in files:files[str(path)]=measure_wav(path)
        return str(path)
    references={s:resolve(p) for s,p in refs.items()}
    normalized=[]
    for row in samples:
        if row['speaker'] not in references or not isinstance(row.get('text_ja'),str) or not row['text_ja'].strip():
            raise ValueError('Missing speaker/reference or Japanese text.')
        r={**row}
        if not any(row.get(mode+'_audio') for mode in ('preset','clone')):raise ValueError('At least one generated WAV required.')
        for mode in ('preset','clone'):
            r[mode+'_audio']=resolve(row[mode+'_audio']) if row.get(mode+'_audio') else None
        normalized.append(r)
    identity={'request':request,'files':files,'implementation':digest(__file__),
              'dsp_implementation':digest(voice_profiler.__file__),
              'embedding':embedder.identity if embedder else None}
    key=fingerprint(identity);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);target=root/key[:20]
    if target.exists():
        saved=json.loads((target/'quality.json').read_text())
        if saved.get('identity')!=identity:raise ValueError('Quality cache identity mismatch.')
        for name,sha in saved['artifacts'].items():
            if digest(target/name)!=sha:raise ValueError('Quality artifact changed.')
        return target/'quality.json'
    vectors={};features={}
    for path in files:
        data=ffmpeg(['-i',path,'-vn','-ar','16000','-ac','1','-f','s16le','pipe:1'])
        x=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
        features[path]=acoustic_features(x,16000)
        features[path].pop('asr_estimates',None)
        if embedder:
            vectors[path]=embedder.embed(x)
            if vectors[path] is not None:cosine(vectors[path],vectors[path])
    def similarity(a,b):
        return cosine(vectors[a],vectors[b]) if embedder and vectors.get(a) is not None and vectors.get(b) is not None else None
    rows=[]
    for sample in normalized:
        for mode in ('preset','clone'):
            path=sample[mode+'_audio']
            if not path:continue
            reference=references[sample['speaker']]
            other={s:similarity(path,p) for s,p in references.items() if s!=sample['speaker']}
            rows.append({'id':sample['id'],'speaker':sample['speaker'],'mode':mode,'text_ja':sample['text_ja'],
                         'audio_file':path,'measurement':files[path],'features':features[path],
                         'reference_cosine':similarity(reference,path),'other_reference_cosines':other,
                         'script_characters_per_audio_second':len(''.join(sample['text_ja'].split()))/files[path]['audio_duration'],
                         'generation_wall_seconds':sample.get(mode+'_generation_wall_seconds'),
                         'pronunciation_verified':False,'perceptual_similarity_verified':False})
    consistency=[]
    for speaker in references:
        for mode in ('preset','clone'):
            group=[r for r in rows if r['speaker']==speaker and r['mode']==mode]
            consistency.append({'speaker':speaker,'mode':mode,'sample_count':len(group),
                'pairs':[{'ids':[a['id'],b['id']],'cosine':similarity(a['audio_file'],b['audio_file'])} for a,b in itertools.combinations(group,2)],
                'human_consistency_verified':False,'acceptance_threshold':None})
    result={'kind':'voice_quality_comparison','format_version':1,'identity':identity,
            'references':{s:{'audio_file':p,'measurement':files[p],'features':features[p]} for s,p in references.items()},
            'rows':rows,'consistency':consistency,'embedding_identity':embedder.identity if embedder else None,
            'review_required':True,'is_authorization_record':False,
            'limitations':['Analysis is not an authorization record.','Cosine similarity is not proof of identity or pronunciation.',
                          'No acceptance threshold is fitted to these samples.','Cross-language scores need separate listening interpretation.',
                          'F0/activity are DSP estimates; characters/second is a script-based rate proxy.']}
    tmp=Path(tempfile.mkdtemp(prefix='.quality-',dir=root))
    try:
        copied={}
        for i,path in enumerate(files):
            name=f'audio-{i}.wav';shutil.copyfile(path,tmp/name)
            if digest(tmp/name)!=files[path]['sha256']:raise ValueError('Audio changed during comparison.')
            copied[path]=name
        esc=html.escape
        page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>Voice quality comparison</title>',
              '<style>body{font:16px/1.7 system-ui;max-width:900px;margin:32px auto;padding:20px}article{border:1px solid #ccc;padding:16px;margin:16px 0}audio{width:100%}</style>',
              '<h1>参考 / Preset / Clone 比較</h1><p>発音・声質・一貫性は未確認。スコアは承認ではありません。</p>']
        for speaker,path in references.items():
            page.append(f'<h2>{esc(speaker)} · Reference</h2><audio controls src="{copied[path]}"></audio>')
        for sample in normalized:
            page.append(f'<article><h2>{esc(sample["id"])} / {esc(sample["speaker"])}</h2><p>{esc(sample["text_ja"])}</p>')
            for mode in ('preset','clone'):
                path=sample[mode+'_audio']
                if not path:page.append(f'<p>{mode}: 未生成・比較保留</p>');continue
                row=next(r for r in rows if r['id']==sample['id'] and r['mode']==mode)
                pitch=row['features']['dsp_estimates']['f0_hz_median'];score=row['reference_cosine']
                page.append(f'<h3>{mode}</h3><audio controls src="{copied[path]}"></audio><p>実測 {files[path]["audio_duration"]:.3f}秒 / F0推定 {pitch} Hz / 参考とのcosine {score}</p>')
            page.append('</article>')
        page.append('<p>同じ文章の発音・欠落・音色・イントネーション・話速と、文章間の声質を実際に聞いて確認してください。</p></html>')
        (tmp/'review.html').write_text(''.join(page),encoding='utf-8')
        result['artifacts']={p.name:digest(p) for p in tmp.iterdir() if p.is_file()}
        atomic_json(tmp/'quality.json',result,overwrite=False);tmp.rename(target)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return target/'quality.json'
