"""Bounded, non-destructive overlap separation with independent speaker evidence."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import time
import wave

import numpy as np
from audio_mixer import read_json, sha256_file
from language.cache import atomic_json, fingerprint
from separation_backends import ConvTasNetSeparator, NoASR, NoSeparator, TorchScriptSeparator, WhisperProcess, peak_memory_bytes
from speaker_transcription import align_segments, compare_transcript
from voice_profiler import acoustic_features, centroid, cosine, rank_speakers

VERSION = 1
UNKNOWN = {None, '', 'UNKNOWN', 'MIXED'}


def wav_info(path):
    with wave.open(str(path), 'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':
            raise ValueError('Source must be mono PCM16 WAV; extract selected clip with ffmpeg first.')
        rate=w.getframerate(); frames=w.getnframes()
        if rate!=16000 or frames<=0: raise ValueError('Source must be nonempty 16 kHz WAV.')
        return {'sample_rate':rate,'sample_frames':frames,'duration':frames/rate}


def read_window(path, a, b):
    info=wav_info(path); rate=info['sample_rate']
    if not 0<=a<b<=info['duration']+.0001: raise ValueError('Audio window outside source.')
    with wave.open(str(path),'rb') as w:
        start=round(a*rate); count=round(b*rate)-start;w.setpos(start); raw=w.readframes(count)
    if len(raw)!=count*2: raise ValueError('Truncated WAV.')
    return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768


def write_wav(path, samples, rate=16000, gain=1.):
    x=np.asarray(samples)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid waveform.')
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((np.clip(x*gain,-1,1-1/32768)*32768).astype('<i2').tobytes())


def active_speakers(turns,a,b):
    return sorted({t['speaker'] for t in turns if t['speaker'] not in UNKNOWN and t['relative_start']<b and t['relative_end']>a})


def max_concurrency(turns,a,b):
    edges=sorted({a,b,*[max(a,min(b,t[k])) for t in turns for k in ('relative_start','relative_end')]})
    return max((len(active_speakers(turns,x,y)) for x,y in zip(edges,edges[1:])),default=0)


def plan_windows(turns, start, end, *, activity=None, max_window=8., padding=.35):
    """Turn boundaries retain tiny interjections; frame votes are hypotheses, not truth."""
    if not all(math.isfinite(v) for v in (start,end,max_window,padding)) or not 0<=start<end or not 0<=padding<max_window/2:
        raise ValueError('Invalid processing range/window.')
    for t in turns:
        if not 0<=t['relative_start']<t['relative_end'] or not math.isfinite(t['relative_end']):raise ValueError('Invalid turn.')
    edges=sorted({start,end,*[t[k] for t in turns for k in ('relative_start','relative_end') if start<t[k]<end]})
    regions=[]
    for a,b in zip(edges,edges[1:]):
        speakers=active_speakers(turns,a,b)
        if regions and regions[-1]['candidate_speakers']==speakers: regions[-1]['end']=b
        else: regions.append({'start':a,'end':b,'candidate_speakers':speakers})
    output=[]
    for region in regions:
        a=region['start']
        while a<region['end']-1e-8:
            b=min(region['end'],a+max_window-2*padding); x=max(start,a-padding);y=min(end,b+padding)
            evidence=None
            if activity:
                from speaker_attribution import activity_between
                evidence=activity_between(activity,a,b)
            count=len(region['candidate_speakers']); vote=(evidence or {}).get('overlap_vote')
            supported=count>=2 and vote is not None and vote>=.35 and evidence['coverage']>=.5
            hypotheses=['A_model_supported_simultaneous_voices'] if supported else ['B_boundary_or_D_word_timing_or_F_unresolved'] if count>=2 else ['F_no_confirmed_overlap']
            if b-a<.8:hypotheses.append('C_short_interjection_possible')
            if evidence and evidence['speech_vote'] is not None and evidence['speech_vote']<.5:hypotheses.append('D_timing_or_E_noise_music_possible')
            output.append({'job_id':f'window-{len(output)+1:05d}','core_start':a,'core_end':b,'window_start':x,'window_end':y,
                           'candidate_speakers':region['candidate_speakers'],'simultaneous_speaker_count':count,
                           'context_max_concurrency':max_concurrency(turns,x,y),
                           'route':'separate' if supported else 'original',
                           'overlap_status':{'model_supported':supported,'human_verified':False,'frame_evidence':evidence,'hypotheses':hypotheses}})
            a=b
    return output


def assess_stems(original, stems, expected):
    if len(stems)!=expected:raise ValueError('Wrong number of output stems.')
    x=np.asarray(original); arrays=[np.asarray(s,dtype=np.float32) for s in stems]
    if any(s.shape!=x.shape or not np.isfinite(s).all() for s in arrays):raise ValueError('Invalid or missing stem samples.')
    rms=lambda s:float(np.sqrt(np.mean(np.square(s,dtype=np.float64))))
    base=max(rms(x),1e-8);reasons=[]; levels=[rms(s) for s in arrays]
    if any(r/base<.015 for r in levels):reasons.append('missing_or_nearly_silent_stem')
    for i,s in enumerate(arrays):
        for t in arrays[:i]:
            if rms(s)>1e-8 and rms(t)>1e-8 and abs(float(np.dot(s,t)/(np.linalg.norm(s)*np.linalg.norm(t))))>.98:reasons.append('duplicate_or_leaking_stems')
    mixture=np.sum(arrays,axis=0,dtype=np.float64)
    residual=rms(x-mixture)/base
    denominator=float(np.dot(mixture,mixture))
    gain=float(np.dot(x,mixture)/denominator) if denominator>1e-16 else 1.
    scaled_residual=rms(x-gain*mixture)/base
    if scaled_residual>.5:reasons.append('large_scale_aligned_mixture_residual')
    peak=max(float(np.max(np.abs(s))) for s in arrays)
    gain=math.copysign(min(abs(gain),.99/max(peak,1e-8)),gain)
    return {'stem_rms':levels,'residual_rms_ratio':residual,'scale_aligned_residual_rms_ratio':scaled_residual,
            'gain_method':'one signed least-squares mixture gain for all stems, peak bounded; no per-stem normalization',
            'common_export_gain':gain,
            'review_reasons':sorted(set(reasons)),'clean_separation_verified':False}


class SpeakerRegistry:
    """Embedding matches are hypotheses; new IDs require independent repeated support."""
    def __init__(self, known_ids=()):self.known=set(known_ids);self.pending=[]

    def identify(self, vector, anchors, a, b, *, eligible=True):
        centers={s:centroid([r['embedding'] for r in rs]) for s,rs in anchors.items() if rs}
        ranked=rank_speakers(vector,centers);margin=ranked[0]['cosine_similarity']-ranked[1]['cosine_similarity'] if len(ranked)>1 else None
        result={'speaker_id':'UNKNOWN','speaker_candidates':ranked,'speaker_confidence_metrics':{'margin':margin,'calibrated':False},'review_required':True}
        if not eligible or vector is None or b-a<.8:return result
        if ranked and ranked[0]['cosine_similarity']>=.72 and (margin is None or margin>=.12):
            result['speaker_id']=ranked[0]['speaker_id'];return result
        # No forced new identity when existing speaker is plausible or references are absent.
        if not centers or (ranked and ranked[0]['cosine_similarity']>.45) or b-a<1.5:return result
        for p in self.pending:
            if cosine(vector,p['embedding'])>=.85 and all(b<=x or a>=y for x,y in p['windows']):
                p['windows'].append((a,b))
                if p['speaker_id'] is None:
                    n=1
                    while f'SPEAKER_NEW_{n:03d}' in self.known:n+=1
                    p['speaker_id']=f'SPEAKER_NEW_{n:03d}';self.known.add(p['speaker_id'])
                result['speaker_id']=p['speaker_id'];result['provisional_new_cluster']=True;return result
        self.pending.append({'embedding':vector,'windows':[(a,b)],'speaker_id':None})
        result['new_cluster_pending']=True;return result


def reject_colliding_matches(stems):
    ids=[s['speaker_id'] for s in stems if s['speaker_id'] not in UNKNOWN]
    for s in stems:
        if s['speaker_id'] in ids and ids.count(s['speaker_id'])>1:
            s['speaker_id']='UNKNOWN';s.setdefault('review_reasons',[]).append('same_speaker_claimed_by_multiple_stems')


def cached_job(root, identity, compute):
    key=fingerprint(identity);folder=Path(root)/key
    if (folder/'receipt.json').exists():
        receipt=read_json(folder/'receipt.json')
        valid=fingerprint(receipt['identity'])==fingerprint(identity) and all((folder/n).is_file() and sha256_file(folder/n)==h for n,h in receipt['hashes'].items())
        if valid:return read_json(folder/'result.json'),folder,True
        # Never overwrite damaged or old audio; build a new attempt.
    folder.mkdir(parents=True,exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=folder))
    result=compute(attempt);atomic_json(attempt/'result.json',result)
    receipt={'identity':identity,'hashes':{p.name:sha256_file(p) for p in attempt.iterdir() if p.is_file()}}
    atomic_json(attempt/'receipt.json',receipt)
    # Cache pointer contains immutable artifact directory, allowing interrupted attempts to remain untouched.
    atomic_json(folder/'pointer.json',{'directory':attempt.name})
    return result,attempt,False


def load_cached_job(root, identity, compute):
    folder=Path(root)/fingerprint(identity);pointer=folder/'pointer.json'
    if pointer.exists():
        attempt=folder/read_json(pointer)['directory']
        if attempt.parent!=folder:raise ValueError('Invalid cache pointer.')
        receipt=read_json(attempt/'receipt.json')
        if fingerprint(receipt['identity'])==fingerprint(identity) and all((attempt/n).is_file() and sha256_file(attempt/n)==h for n,h in receipt['hashes'].items()):
            return read_json(attempt/'result.json'),attempt,True
    return cached_job(root,identity,compute)


def execute(*,video_id,clip,audio,turns,utterances,source_words,output,separator,asr,embedder=None,
            profiles=None,analysis=None,activity=None,captions=(),start=0.,end=30.,dry_run=False):
    started=time.perf_counter();info=wav_info(audio);duration=clip['end']-clip['start']
    fingerprint([clip,turns,utterances,source_words,profiles,analysis,activity,captions])
    if profiles and analysis and profiles.get('analysis_key')!=analysis.get('analysis_key'):raise ValueError('Profile/analysis identity mismatch.')
    if abs(duration-info['duration'])>.1 or not 0<=start<end<=duration+.0001:raise ValueError('Clip/audio/range mismatch.')
    end=min(end,info['duration']);audio_hash=sha256_file(audio)
    plan=plan_windows(turns,start,end,activity=activity,max_window=separator.spec.max_window_seconds)
    identity={'version':VERSION,'video_id':video_id,'clip':clip,'source_audio_hash':audio_hash,
              'inputs_hash':fingerprint([turns,utterances,source_words,profiles,analysis,activity,captions]),
              'range':[start,end],'separator':separator.identity,'asr':asr.identity,
              'embedding':getattr(embedder,'identity',None),
              'implementation':{p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('separation_backends.py'),Path(__file__).with_name('speaker_transcription.py')]}}
    if dry_run:return {'identity':identity,'plan':plan}
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix=f'run-{fingerprint(identity)[:12]}-',dir=root))
    known=sorted({t['speaker'] for t in turns if t['speaker'] not in UNKNOWN});registry=SpeakerRegistry(known)
    jobs=[];transcripts=[];hits=0
    try:
        for job in plan:
            related=[u for u in utterances if u['relative_start']<job['core_end'] and u['relative_end']>job['core_start']]
            job_identity={'run':identity,'job':job}
            def compute(folder):
                tick=time.perf_counter();x=read_window(audio,job['window_start'],job['window_end']);write_wav(folder/'original.wav',x)
                original=asr.transcribe(folder/'original.wav');stems=[];status='original_retained';quality=None;separation_seconds=0.
                if job['route']=='separate':
                    count=max(job['simultaneous_speaker_count'],job['context_max_concurrency'])
                    if count not in separator.spec.simultaneous_speaker_counts:status='unseparated_capacity'
                    else:
                        sep_tick=time.perf_counter();ys=separator.separate(x,16000,count);separation_seconds=time.perf_counter()-sep_tick
                        quality=assess_stems(x,ys,count)
                        status='quality_flagged' if quality['review_reasons'] else 'waveforms_generated_unverified'
                        for i,y in enumerate(ys):
                            name=f'stem-{i+1}.wav';write_wav(folder/name,y,gain=quality['common_export_gain'])
                            # Match target audio only; padded neighboring voices never decide a short target.
                            core=(y*quality['common_export_gain'])[round((job['core_start']-job['window_start'])*16000):round((job['core_end']-job['window_start'])*16000)]
                            features=acoustic_features(core)
                            vector=embedder.embed(core) if embedder and len(core)>=12800 else None
                            stems.append({'stem_id':f'{job["job_id"]}/stem-{i+1}','audio_file':name,'embedding':vector,'features':features,'asr':asr.transcribe(folder/name)})
                if not stems:
                    core=x[round((job['core_start']-job['window_start'])*16000):round((job['core_end']-job['window_start'])*16000)]
                    vector=embedder.embed(core) if embedder and len(core)>=12800 and job['simultaneous_speaker_count']==1 else None
                    stems=[{'stem_id':f'{job["job_id"]}/original','audio_file':'original.wav','embedding':vector,'features':acoustic_features(core),'asr':original}]
                return {'status':status,'quality':quality,'stems':stems,'original_asr':original,'elapsed_seconds':time.perf_counter()-tick,'separation_seconds':separation_seconds,'peak_memory_bytes':peak_memory_bytes()}
            try:
                data,folder,hit=load_cached_job(root/'cache',job_identity,compute);hits+=int(hit)
                anchors={}
                if analysis and profiles:
                    from speaker_attribution import independent_anchors
                    anchors=independent_anchors(analysis,profiles,job['window_start'],job['window_end'])
                stems=[]
                for raw in data['stems']:
                    stem=copy.deepcopy(raw);quality=data['quality'] or {}
                    from speaker_attribution import quality_reasons
                    reasons=quality_reasons(stem['features'])+quality.get('review_reasons',[])
                    eligible=not reasons and (job['route']=='separate' and data['status']!='unseparated_capacity' or job['simultaneous_speaker_count']==1)
                    stem.update(registry.identify(stem['embedding'],anchors,job['core_start'],job['core_end'],eligible=eligible))
                    stem['review_reasons']=reasons+['automatic_match_not_identity_confirmation'];stem['audio_file']=str((folder/stem['audio_file']).resolve())
                    stem['audio_sha256']=sha256_file(stem['audio_file']);stems.append(stem)
                reject_colliding_matches(stems)
                original_rows=align_segments(data['original_asr'],clip_start=clip['start'],window_start=job['window_start'],window_end=job['window_end'],core_start=job['core_start'],core_end=job['core_end'],speaker_id='UNKNOWN',source='original_window_asr',separation_source=None)
                comparison_words=source_words+[w for r in original_rows for w in r['words']]
                for stem in stems:
                    separated=stem['audio_file'].endswith('.wav') and Path(stem['audio_file']).name!='original.wav'
                    rows=align_segments(stem['asr'],clip_start=clip['start'],window_start=job['window_start'],window_end=job['window_end'],core_start=job['core_start'],core_end=job['core_end'],speaker_id=stem['speaker_id'],source=asr.identity,separation_source=separator.identity if separated else None)
                    for i,row in enumerate(rows):
                        row.update(transcript_id=f'{stem["stem_id"]}/{i}',stem_id=stem['stem_id'],source_utterance_ids=[u['id'] for u in related],audio_sha256=stem['audio_sha256'])
                        short_unknown=[u for u in related if u['speaker'] in UNKNOWN and u['relative_end']-u['relative_start']<.8 and u['relative_start']<row['relative_end'] and u['relative_end']>row['relative_start']]
                        if short_unknown:
                            row['speaker_id']='UNKNOWN'
                            for word in row['words']:word['speaker']='UNKNOWN'
                            row['review_reasons'].append('short_unknown_target_not_resolved_by_surrounding_voice')
                        row['comparison']=compare_transcript(row,comparison_words,captions);row['review_reasons']+=row['comparison']['review_reasons']+stem['review_reasons']
                    stem['transcriptions']=rows;transcripts+=rows
                    stem['asr_empty']=not rows
                jobs.append({**job,**{k:v for k,v in data.items() if k!='stems'},'stems':stems,'cache_hit':hit,
                             'original_audio_file':str((folder/'original.wav').resolve()),'original_text':' '.join(u['text_en'] for u in related),
                             'source_utterance_ids':[u['id'] for u in related],
                             'original_start':clip['start']+job['core_start'],'original_end':clip['start']+job['core_end'],
                             'refined_start':None,'refined_end':None,'review_required':True})
            except Exception as error:
                jobs.append({**job,'status':'execution_failed','error':f'{type(error).__name__}: {error}','source_utterance_ids':[u['id'] for u in related],'stems':[],'review_required':True})
            print(f'{job["job_id"]}: {jobs[-1]["status"]}',flush=True)
    finally:asr.close()
    bundle={'format_version':VERSION,'kind':'speaker_separation','run_key':fingerprint(identity),'video_id':video_id,'clip':clip,
            'range':{'relative_start':start,'relative_end':end},'sample_rate':16000,'source_audio_hash':audio_hash,
            'detected_speakers':sorted(registry.known),'global_speaker_count':len(registry.known),'provenance':identity,
            'separation_model':separator.identity,'jobs':jobs,'source_utterances':utterances,
            'metrics':{'elapsed_seconds':time.perf_counter()-started,'process_peak_memory_bytes':peak_memory_bytes(),'cache_hits':hits,
                       'asr_worker_peak_memory_bytes':max([s['asr'].get('peak_memory_bytes',0) for j in jobs for s in j['stems']]+[0]),
                       'waveform_generated_jobs':sum(j['status'] in ('quality_flagged','waveforms_generated_unverified') for j in jobs),
                       'human_verified_separation_successes':0,'failed_jobs':sum(j['status']=='execution_failed' for j in jobs)}}
    atomic_json(run/'speaker_separation.json',bundle)
    atomic_json(run/'speaker_transcriptions.json',{'format_version':1,'video_id':video_id,'clip':clip,'run_key':bundle['run_key'],'utterances':transcripts})
    from separation_review import comparison_report
    from speaker_review import build_separation_review
    build_separation_review(run,bundle);(run/'comparison.md').write_text(comparison_report(bundle),encoding='utf-8')
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('video_id');p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=30)
    p.add_argument('--profile-dir',type=Path);p.add_argument('--activity',type=Path);p.add_argument('--captions',type=Path)
    p.add_argument('--backend',choices=['none','convtasnet','torchscript'],default='none');p.add_argument('--weights',type=Path);p.add_argument('--model-config',type=Path)
    p.add_argument('--asr-python',type=Path);p.add_argument('--asr-model',type=Path);p.add_argument('--dry-run',action='store_true');p.add_argument('--no-embedding',action='store_true')
    a=p.parse_args();source=read_json(a.input_dir/'utterances.json');clip=read_json(a.input_dir/'selection.json');diar=read_json(a.input_dir/'diarization.json')
    if source['video_id']!=a.video_id or source['clip']!=clip or diar['clip']!=clip:p.error('Input video/clip mismatch.')
    profiles=analysis=None
    if a.profile_dir:
        profiles=read_json(a.profile_dir/'speaker_profiles.json');analysis=read_json(a.profile_dir/'diarization.enhanced.json')
        if profiles['video_id']!=a.video_id or profiles['clip']!=clip or analysis['clip']!=clip:p.error('Profile clip mismatch.')
        diar=analysis
    separator=NoSeparator()
    if a.backend!='none' and not a.weights:p.error('--weights is required; this CLI never downloads models.')
    if a.backend=='convtasnet':separator=ConvTasNetSeparator(a.weights)
    if a.backend=='torchscript':
        if not a.model_config:p.error('--model-config required.')
        separator=TorchScriptSeparator(a.weights,read_json(a.model_config))
    asr=NoASR()
    if bool(a.asr_python)!=bool(a.asr_model):p.error('Supply both ASR options.')
    if a.asr_python:asr=WhisperProcess(a.asr_python,a.asr_model)
    embedder=None
    if not a.no_embedding and not a.dry_run and profiles:
        from speaker_refinement import OfflinePyannote
        embedder=OfflinePyannote()
    result=execute(video_id=a.video_id,clip=clip,audio=a.audio,turns=diar['turns'],utterances=source['utterances'],source_words=read_json(a.input_dir/'word_timestamps.json')['words'],output=a.output,separator=separator,asr=asr,embedder=embedder,profiles=profiles,analysis=analysis,activity=read_json(a.activity) if a.activity else None,captions=read_json(a.captions) if a.captions else (),start=a.start,end=a.end,dry_run=a.dry_run)
    print(json.dumps(result,ensure_ascii=False,indent=2) if isinstance(result,dict) else result)


if __name__=='__main__':main()
