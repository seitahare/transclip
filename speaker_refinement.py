"""Offline full-clip speaker evidence, profiles and non-destructive review assets."""
import argparse
import copy
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import wave
import numpy as np

from audio_mixer import ffmpeg, read_json, sha256_file
from language.cache import fingerprint, atomic_json
from word_speaker_alignment import assign_speaker, covered_duration
from voice_profiler import (RATE, acoustic_features, centroid, cosine, overlap_regions,
                            rank_speakers, reference_quality, reference_windows, union)

BASE_DIR=Path(__file__).resolve().parent
MODEL='pyannote/speaker-diarization-community-1'
UNKNOWN=(None,'UNKNOWN','REVIEW')
VERSION=1


class OfflinePyannote:
    """Use existing local assets only; no fixed speaker count and no telemetry."""
    def __init__(self):
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYANNOTE_METRICS_ENABLED='0')
        from huggingface_hub import snapshot_download
        self.snapshot=Path(snapshot_download(MODEL,local_files_only=True))
        names=['config.yaml','embedding/pytorch_model.bin','segmentation/pytorch_model.bin','plda/plda.npz','plda/xvec_transform.npz']
        self.identity={'model':MODEL,'revision':self.snapshot.name,'files':{n:sha256_file(self.snapshot/n) for n in names},
                       'versions':{n:version(n) for n in ['pyannote.audio','torch','numpy']},
                       'device':'cpu','speaker_count':'automatic','overlap_excluded_from_embeddings':True}
        self.pipeline=None

    def load(self):
        if self.pipeline is None:
            import torch
            from pyannote.audio import Pipeline
            torch.set_num_threads(4)
            self.pipeline=Pipeline.from_pretrained(self.snapshot).to(torch.device('cpu'))
        return self.pipeline

    def diarize(self,samples,clip_start):
        import torch
        from diarizer import collect_turns
        def hook(step,artifact,**kwargs):
            done,total=kwargs.get('completed'),kwargs.get('total')
            if done is None or done==total: print(f'pyannote {step}: {done}/{total}',flush=True)
        result=self.load()({'waveform':torch.from_numpy(samples.copy()).float().unsqueeze(0),'sample_rate':RATE},hook=hook)
        return {'turns':collect_turns(result.speaker_diarization,clip_start),
                'exclusive_turns':collect_turns(result.exclusive_speaker_diarization,clip_start)}

    def embed(self,samples):
        import torch
        if len(samples)<RATE*.8: return None
        result=self.load()._embedding(torch.from_numpy(samples.copy()).float()[None,None,:])[0]
        if not np.isfinite(result).all() or np.linalg.norm(result)<1e-8: return None
        return (result/np.linalg.norm(result)).tolist()


def read_audio(path):
    with wave.open(str(path),'rb') as f:
        n=f.getnframes()
        if f.getframerate()!=RATE or f.getnchannels()!=1 or f.getsampwidth()!=2 or n<=0:
            raise ValueError('Expected nonempty mono 16kHz PCM16.')
        data=f.readframes(n)
        if len(data)!=n*2: raise ValueError('Truncated source WAV.')
    return np.frombuffer(data,dtype='<i2').astype(np.float32)/32768


def save_audio(path,samples):
    if not len(samples): raise ValueError('Cannot export empty reference/review audio.')
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,RATE,0,'NONE','not compressed'))
        f.writeframes(np.clip(np.rint(samples*32768),-32768,32767).astype('<i2').tobytes())


def crop(samples,a,b):
    if not 0<=a<b<=len(samples)/RATE+.001: raise ValueError('Audio crop outside selected clip.')
    return samples[round(a*RATE):min(len(samples),round(b*RATE))]


def validate_inputs(directory,video_id):
    names=['selection','diarization','word_timestamps','word_speakers','utterances','utterance_translations']
    data={n:read_json(directory/(n+'.json')) for n in names}
    selection=data['selection']
    a,b=selection.get('start'),selection.get('end')
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in (a,b)) or not 0<=a<b:
        raise ValueError('Invalid selection.')
    clip={'start':a,'end':b}
    for name,d in data.items():
        fingerprint(d)
        if d.get('video_id',video_id)!=video_id: raise ValueError(f'{name}: video ID mismatch.')
        if name!='selection' and d.get('clip')!=clip: raise ValueError(f'{name}: clip mismatch.')
    for name,key in [('diarization','turns'),('word_timestamps','words'),('word_speakers','words'),('utterances','utterances')]:
        items=data[name].get(key)
        if not isinstance(items,list) or not items: raise ValueError(f'{name}: missing {key}.')
        for row in items:
            x,y=row.get('relative_start'),row.get('relative_end')
            # pyannote's final frame can extend slightly beyond the decoded clip.
            allowance=.20 if name=='diarization' else .01
            if any(type(t) not in (int,float) or not math.isfinite(t) for t in (x,y)) or not 0<=x<y<=b-a+allowance:
                raise ValueError(f'{name}: invalid interval.')
            for k,v in [('start',x),('end',y)]:
                if type(row.get(k)) not in (int,float) or abs(row[k]-a-v)>.011: raise ValueError(f'{name}: absolute/relative mismatch.')
            if name=='diarization' and (not isinstance(row.get('speaker'),str) or not re.fullmatch(r'[A-Za-z0-9_-]+',row['speaker'])):
                raise ValueError('Invalid diarization speaker.')
    old_words=data['word_speakers']['words'];words=data['word_timestamps']['words']
    if len(old_words)!=len(words) or any(any(x.get(k)!=y.get(k) for k in ['word','start','end','relative_start','relative_end']) for x,y in zip(old_words,words)):
        raise ValueError('ASR/word-speaker content mismatch.')
    ids=[u['id'] for u in data['utterances']['utterances']]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate utterance IDs.')
    translations={u['id']:u for u in data['utterance_translations']['utterances']}
    if set(ids)!=set(translations): raise ValueError('Translation IDs differ.')
    for u in data['utterances']['utterances']:
        if any(u.get(k)!=translations[u['id']].get(k) for k in ['speaker','start','end','text_en']): raise ValueError('Stale utterance translation.')
        for w in u['words']:
            wid=w.get('word_id')
            if type(wid) is not int or not 1<=wid<=len(words): raise ValueError('Invalid source word ID.')
            if any(w.get(k)!=words[wid-1].get(k) for k in ['word','start','end']): raise ValueError('Utterance word provenance mismatch.')
    if read_json(directory/'source.info.json').get('id')!=video_id: raise ValueError('Source metadata video mismatch.')
    return data


def bound_turns(turns,clip):
    result=[];adjustments=[];duration=clip['end']-clip['start']
    for original in turns:
        row=copy.deepcopy(original)
        if row['relative_end']>duration:
            if row['relative_end']-duration>.20: raise ValueError('Diarization exceeds source by more than one boundary tolerance.')
            adjustments.append({'speaker':row['speaker'],'original_end':row['relative_end'],'bounded_end':duration,
                                'reason':'model frame extends beyond selected source; metadata clipped, no audio truncated'})
            row['relative_end']=duration;row['end']=clip['end']
        if row['relative_start']<row['relative_end']: result.append(row)
    return result,adjustments


def stable_mapping(new_centers,old_centers,new_turns,old_turns):
    """Mutual unique embedding matches; new/ambiguous clusters get new IDs."""
    mapping={};evidence=[];claimed=set();new_index=0
    rankings={s:rank_speakers(v,old_centers) for s,v in new_centers.items()}
    reverse={s:rank_speakers(v,new_centers) for s,v in old_centers.items()}
    for label in sorted({t['speaker'] for t in new_turns},key=lambda s:min(t['relative_start'] for t in new_turns if t['speaker']==s)):
        ranked=rankings.get(label,[]); best=ranked[0] if ranked else None
        margin=best['cosine_similarity']-ranked[1]['cosine_similarity'] if len(ranked)>1 else (best['cosine_similarity'] if best else 0)
        target=best['speaker_id'] if best else None
        mutual=bool(target and reverse.get(target) and reverse[target][0]['speaker_id']==label)
        confident=bool(best and best['cosine_similarity']>=.65 and margin>=.10 and mutual and target not in claimed)
        temporal={s:sum(covered_duration([t for t in old_turns if t['speaker']==s],r['relative_start'],r['relative_end'])
                        for r in new_turns if r['speaker']==label) for s in old_centers}
        if confident:
            mapping[label]=target;claimed.add(target)
        else:
            while f'SPEAKER_NEW_{new_index:02d}' in old_centers or f'SPEAKER_NEW_{new_index:02d}' in claimed: new_index+=1
            mapping[label]=f'SPEAKER_NEW_{new_index:02d}';claimed.add(mapping[label]);new_index+=1
        evidence.append({'raw_speaker_id':label,'speaker_id':mapping[label],'matched_previous_id':confident,
                         'candidates':ranked,'margin':margin,'temporal_overlap_seconds':temporal,
                         'method':'mutual cosine match to non-overlap previous-turn anchors',
                         'uncertainty':'model evidence, not human-confirmed identity'})
    return mapping,evidence


def word_evidence(words,old_words,turns,window_records,centers):
    result=[]
    for index,(word,old) in enumerate(zip(words,old_words),1):
        row=copy.deepcopy(word);base=assign_speaker(word,turns);a,b=word['relative_start'],word['relative_end']
        enclosing=[r for r in window_records if r['relative_start']<=a and r['relative_end']>=b and r.get('embedding') is not None]
        window=min(enclosing,key=lambda r:r['relative_end']-r['relative_start']) if enclosing else None
        ranked=rank_speakers(window['embedding'],centers) if window else []
        best=ranked[0] if ranked else None
        margin=best['cosine_similarity']-ranked[1]['cosine_similarity'] if len(ranked)>1 else (best['cosine_similarity'] if best else None)
        reasons=[];speaker=base['speaker']
        near=[{'speaker':t['speaker'],'boundary':k,'time':t[k],'distance_seconds':round(abs(v-t[k]),6)}
              for t in turns for k,v in [('relative_start',a),('relative_end',b)] if abs(v-t[k])<.12]
        if base['overlapping_speech']: reasons.append('overlapping_speech')
        if b-a<.20: reasons.append('short_word')
        if near: reasons.append('near_speaker_boundary')
        if old.get('overlapping_speech') and not base['overlapping_speech']:
            reasons.append('old_new_overlap_disagreement');speaker=None
        if best and base['speaker'] and (best['speaker_id']!=base['speaker'] or best['cosine_similarity']<.55 or (margin or 0)<.1):
            reasons.append('embedding_temporal_disagreement');speaker=None
        if old.get('speaker') and speaker and old['speaker']!=speaker:
            reasons.append('previous_new_speaker_disagreement');speaker=None
        # Unknown recovery requires independent full-clip temporal + acoustic-context agreement.
        # Never resolve formerly overlapping/ambiguous or boundary-near words automatically.
        if old.get('speaker') in UNKNOWN:
            if not (speaker and best and best['speaker_id']==speaker and best['cosine_similarity']>=.65 and (margin or 0)>=.15
                    and base['speaker_candidates'][0]['coverage']>=.9 and not near and b-a>=.08
                    and old.get('status') not in ('overlap','ambiguous') and not base['overlapping_speech']):
                speaker=None
            else:
                reasons.append('new_contextual_assignment_needs_review')
        if speaker is None: reasons.append('unresolved_speaker')
        if not window: reasons.append('no_clean_embedding_window')
        row.update(base)
        if speaker is None and row['status']=='single': row['status']='uncertain'
        row.update({'word_id':index,'speaker':speaker,'original_speaker':old.get('speaker'),
                    'original_status':old.get('status'),'review_required':bool(reasons),'review_reasons':sorted(set(reasons)),
                    'assignment_method':'temporal_and_embedding_context' if window and speaker else 'temporal_only' if speaker else 'unresolved',
                    'embedding_candidates':ranked,'embedding_margin':margin,
                    'embedding_scope':{'relative_start':window['relative_start'],'relative_end':window['relative_end'],
                                       'kind':'contextual window, not isolated-word identity'} if window else None,
                    'boundary_checks':near,'diarization_asr_conflict':not base['speaker_candidates'] or bool(base['overlapping_speech'])})
        result.append(row)
    return result


def refine_utterances(source,words,turns,centers,backend,samples):
    result=[]
    for original in source:
        row=copy.deepcopy(original);selected=[words[w['word_id']-1] for w in original['words']]
        row['words']=selected;votes={}
        for w in selected:
            if w['speaker'] not in UNKNOWN: votes[w['speaker']]=votes.get(w['speaker'],0)+(w['relative_end']-w['relative_start'])
        ranked=sorted(votes,key=lambda s:-votes[s])
        candidate=ranked[0] if ranked else None
        a,b=row['relative_start'],row['relative_end']
        temporal=assign_speaker(row,turns)['speaker_candidates']
        is_overlap=any(max(a,l)<min(b,r) for l,r in overlap_regions(turns))
        vector=backend.embed(crop(samples,a,b)) if b-a>=.8 and not is_overlap else None
        acoustic=rank_speakers(vector,centers)
        reasons=list(row.get('review_reasons',[]))
        if original['speaker'] in UNKNOWN: reasons.append('human_confirmation_required_for_unknown_utterance')
        if candidate and original['speaker'] not in UNKNOWN and candidate!=original['speaker']:
            row['speaker']=None;reasons.append('previous_new_speaker_disagreement')
        if acoustic and original['speaker'] not in UNKNOWN and acoustic[0]['speaker_id']!=original['speaker']:
            row['speaker']=None;reasons.append('utterance_embedding_disagreement')
        if len(votes)>1: reasons.append('multiple_word_speaker_candidates')
        if is_overlap: reasons.append('overlapping_source_speech')
        if any(w['review_required'] for w in selected): reasons.append('word_assignment_needs_review')
        row.update(original_speaker=original['speaker'],proposed_speaker=candidate,
                   acoustic_candidates=acoustic,acoustic_evidence_unavailable_reason='overlap' if is_overlap else 'shorter_than_0.8s' if b-a<.8 else None,
                   temporal_candidates=temporal,
                   review_reasons=sorted(set(reasons)),review_required=bool(reasons),
                   speaker_evidence={'weighted_word_seconds':votes,'human_confirmed':False})
        result.append(row)
    return result


def audit_boundaries(turns,words,samples):
    word_edges=sorted({w[k] for w in words for k in ('relative_start','relative_end')})
    records=[]
    for t in turns:
        for side in ('relative_start','relative_end'):
            time=t[side];nearest=min(word_edges,key=lambda x:abs(x-time))
            before=samples[max(0,round((time-.08)*RATE)):min(len(samples),round(time*RATE))]
            after=samples[max(0,round(time*RATE)):min(len(samples),round((time+.08)*RATE))]
            records.append({'speaker':t['speaker'],'boundary':side,'relative_time':time,
                            'nearest_asr_boundary':nearest,'distance_seconds':abs(nearest-time),
                            'before_rms':float(np.sqrt(np.mean(before**2))) if len(before) else None,
                            'after_rms':float(np.sqrt(np.mean(after**2))) if len(after) else None,
                            'review_required':abs(nearest-time)>.20,
                            'interpretation':'ASR and diarization are estimates; energy is not a speaker label.'})
    return records


def run_analysis(video_id,input_dir,output_dir,backend):
    directory=Path(input_dir).resolve();root=Path(output_dir).resolve()
    data=validate_inputs(directory,video_id);clip={k:data['selection'][k] for k in ('start','end')}
    source=directory/'source.mp4'
    provenance={n:fingerprint(d) for n,d in data.items()}
    identity={'format_version':VERSION,'video_id':video_id,'clip':clip,'inputs':provenance,
              'source_video_sha256':sha256_file(source),'backend':backend.identity,
              'implementation':{n:sha256_file(Path(__file__).parent/n) for n in ['speaker_refinement.py','voice_profiler.py','speaker_review.py']}}
    key=fingerprint(identity);root.mkdir(parents=True,exist_ok=True);target=root/key[:16]
    with (root/'.analysis.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another speaker analysis is running.') from error
        if target.exists():
            manifest=read_json(target/'manifest.json')
            if manifest.get('identity')!=identity: raise ValueError('Existing analysis identity mismatch.')
            for n,digest in manifest['artifacts'].items():
                path=(target/n).resolve()
                if not path.is_relative_to(target) or sha256_file(path)!=digest: raise ValueError('Existing analysis changed; use a new output directory.')
            return target,True
        temp=Path(tempfile.mkdtemp(prefix='.pending-analysis-',dir=root))
        try:
            print('Extracting selected source audio...',flush=True)
            ffmpeg(['-ss',str(clip['start']),'-i',source,'-t',str(clip['end']-clip['start']),'-vn','-ac','1','-ar',str(RATE),'-c:a','pcm_s16le',temp/'selected_source.wav'])
            samples=read_audio(temp/'selected_source.wav')
            if abs(len(samples)/RATE-(clip['end']-clip['start']))>.01: raise ValueError('Source duration mismatch.')
            print('Analyzing full selected clip; speaker count automatic...',flush=True)
            raw=backend.diarize(samples,clip['start'])
            if not raw['turns']: raise ValueError('No speakers detected.')
            raw['turns'],new_boundary_adjustments=bound_turns(raw['turns'],clip)
            old_turns,old_boundary_adjustments=bound_turns(data['diarization']['turns'],clip)
            all_overlap=union([*overlap_regions(data['diarization']['turns']),*overlap_regions(raw['turns'])])
            def embed_windows(turns):
                records=reference_windows(turns,excluded=all_overlap)
                for i,r in enumerate(records):
                    r['embedding']=backend.embed(crop(samples,r['relative_start'],r['relative_end']))
                    if (i+1)%20==0: print(f'Embedding windows {i+1}/{len(records)}',flush=True)
                centers={s:centroid([r['embedding'] for r in records if r['speaker_id']==s and r['embedding'] is not None]) for s in sorted({t['speaker'] for t in turns})}
                return records,centers
            print('Building full-clip embedding anchors...',flush=True)
            old_records,old_centers=embed_windows(old_turns)
            records,raw_centers=embed_windows(raw['turns'])
            mapping,mapping_evidence=stable_mapping(raw_centers,old_centers,raw['turns'],data['diarization']['turns'])
            centers={mapping[s]:v for s,v in raw_centers.items()}
            turns=copy.deepcopy(raw['turns'])
            for t in turns: t['raw_speaker']=t['speaker'];t['speaker']=mapping[t['speaker']]
            for r in records: r['speaker_id']=mapping[r['speaker_id']]
            print('Measuring voice features and reference quality...',flush=True)
            references=[];profiles=[];(temp/'references').mkdir()
            for i,r in enumerate(records):
                a,b=r['relative_start'],r['relative_end'];speaker=r['speaker_id']
                words=[w for w in data['word_timestamps']['words'] if a<=w['relative_start'] and w['relative_end']<=b]
                r['features']=acoustic_features(crop(samples,a,b),words=words)
                # Leave overlapping windows out of own centroid to avoid self-confirmation.
                peers=[q['embedding'] for q in records if q['speaker_id']==speaker and q['embedding'] is not None
                       and (q['relative_end']<=a or q['relative_start']>=b)]
                peer_center=centroid(peers)
                own=cosine(r['embedding'],peer_center) if r['embedding'] is not None and peer_center is not None else None
                alternatives=[cosine(r['embedding'],v) for s,v in centers.items() if s!=speaker and v is not None] if r['embedding'] is not None else []
                margin=own-max(alternatives) if own is not None and alternatives else own
                stable=next(m['matched_previous_id'] for m in mapping_evidence if m['speaker_id']==speaker)
                quality=reference_quality(r['features'],own,margin,stable_identity=stable)
                r.update(reference_id=f'ref-{i+1:03d}',speaker_similarity=own,speaker_margin=margin,
                         independent_anchor_count=len(peers),quality=quality,
                         transcript={'asr_text':' '.join(w['word'] for w in words),'corrected_text':None,'reviewed':False,
                                     'word_ids':[j+1 for j,w in enumerate(data['word_timestamps']['words']) if a<=w['relative_start'] and w['relative_end']<=b],
                                     'boundary_warning':'Only fully contained ASR words; audio edges may include partial words.'})
                references.append(r)
            for speaker,c in centers.items():
                eligible=sorted([r for r in references if r['speaker_id']==speaker and r['quality']['eligible']],key=lambda r:-r['quality']['ranking_score'])
                selected=[]
                for r in eligible:
                    if any(max(r['relative_start'],s['relative_start'])<min(r['relative_end'],s['relative_end']) for s in selected): continue
                    selected.append(r)
                    if len(selected)==3: break
                for r in selected:
                    name=f"references/{speaker}_{r['reference_id']}.wav"
                    save_audio(temp/name,crop(samples,r['relative_start'],r['relative_end']))
                    r['audio_file']=name;r['audio_sha256']=sha256_file(temp/name)
                all_rows=[r for r in references if r['speaker_id']==speaker]
                f0=[r['features']['dsp_estimates']['f0_hz_median'] for r in selected if r['features']['dsp_estimates']['f0_hz_median'] is not None]
                profiles.append({'speaker_id':speaker,'identity_status':'model_cluster_not_real_person_identity',
                                 'analyzed_intervals':[{k:r[k] for k in ['relative_start','relative_end','reference_id','features','quality']} for r in all_rows],
                                 'model_estimates':{'speaker_embedding':c,'dimension':len(c) if c else 0,'window_count':len(all_rows),
                                                    'embedding_model':backend.identity,'similarity_is_probability':False},
                                 'summary_dsp_estimates':{'median_reference_f0_hz':float(np.median(f0)) if f0 else None,
                                                          'aggregation':'median of selected reference medians; inspect per-window features'},
                                 'reference_candidates':[{k:v for k,v in r.items() if k not in ('embedding','features')} for r in all_rows],
                                 'selected_reference_ids':[r['reference_id'] for r in selected],
                                 'clone_authorization':{'verified':False,'consent_reference':None},
                                 'quality':{'reference_count':len(selected),'review_required':True,'intelligibility_verified':False}})
            words=word_evidence(data['word_timestamps']['words'],data['word_speakers']['words'],turns,records,centers)
            print('Comparing utterances and exporting review audio...',flush=True)
            utterances=refine_utterances(data['utterances']['utterances'],words,turns,centers,backend,samples)
            comparisons=[{'speakers':[a,b],'cosine_similarity':cosine(centers[a],centers[b]),
                          'confusion_review':cosine(centers[a],centers[b])>.65}
                         for i,a in enumerate(centers) for b in list(centers)[i+1:] if centers[a] is not None and centers[b] is not None]
            enhanced={'format_version':1,'kind':'speaker_analysis','analysis_key':key,'video_id':video_id,'clip':clip,
                      'speakers':sorted(centers),'turns':turns,'raw_model_turns':raw['turns'],
                      'source_overlap_regions':overlap_regions(turns),'conservative_reference_exclusions':all_overlap,
                      'identity_mapping':mapping_evidence,'speaker_similarity':comparisons,'model':backend.identity,
                      'model_end_boundary_adjustments':{'previous':old_boundary_adjustments,'new':new_boundary_adjustments},
                      'input_fingerprints':provenance,'unknown_policy':'No automatic utterance UNKNOWN confirmation.',
                      'utterances_file':'utterances.enhanced.json',
                      'turn_boundary_audit':audit_boundaries(turns,data['word_timestamps']['words'],samples),
                      'boundary_checks':[{'word_id':w['word_id'],'checks':w['boundary_checks']} for w in words if w['boundary_checks']],
                      'embedding_window_evidence':[{k:r[k] for k in ['speaker_id','relative_start','relative_end','embedding','speaker_similarity','speaker_margin','independent_anchor_count']} for r in records]}
            profile_data={'format_version':1,'kind':'speaker_profiles','analysis_key':key,'video_id':video_id,'clip':clip,
                          'profiles':profiles,'features_are_not_clone_permission':True,'model':backend.identity}
            improved_words={'video_id':video_id,'clip':clip,'format_version':2,'analysis_key':key,'speakers':sorted(centers),'words':words}
            improved_utterances={**data['utterances'],'analysis_key':key,'cache_key':fingerprint(utterances),'utterances':utterances}
            for name,content in [('diarization.enhanced.json',enhanced),('speaker_profiles.json',profile_data),
                                 ('word_speakers.enhanced.json',improved_words),('utterances.enhanced.json',improved_utterances)]:
                atomic_json(temp/name,content,overwrite=False)
            summary={'accuracy_improvement_verified':False,'human_ground_truth_available':False,
                     'previous_speaker_count':len(data['diarization']['speakers']),'new_speaker_count':len(centers),
                     'previous_turn_count':len(data['diarization']['turns']),'new_turn_count':len(turns),
                     'short_turn_count_under_half_second':sum(t['relative_end']-t['relative_start']<.5 for t in turns),
                     'previous_unknown_words':sum(w.get('speaker') in UNKNOWN for w in data['word_speakers']['words']),
                     'previous_reconstructed_unknown_words':sum(w.get('speaker') in UNKNOWN for u in data['utterances']['utterances'] for w in u['words']),
                     'new_unknown_words':sum(w['speaker'] in UNKNOWN for w in words),
                     'previous_unknown_utterances':sum(u['speaker'] in UNKNOWN for u in data['utterances']['utterances']),
                     'new_unknown_utterances':sum(u['speaker'] in UNKNOWN for u in utterances),
                     'previous_overlap_seconds':sum(b-a for a,b in overlap_regions(data['diarization']['turns'])),
                     'new_overlap_seconds':sum(b-a for a,b in overlap_regions(turns)),
                     'boundary_review_word_count':sum(bool(w['boundary_checks']) for w in words),
                     'asr_diarization_conflict_word_count':sum(w['diarization_asr_conflict'] for w in words),
                     'changed_word_assignments':[w['word_id'] for w in words if w['speaker']!=w['original_speaker']],
                     'reference_candidate_count':len(references),'selected_reference_count':sum(len(p['selected_reference_ids']) for p in profiles),
                     'speaker_similarity':comparisons}
            atomic_json(temp/'comparison.json',summary,overwrite=False)
            from speaker_review import build_review
            build_review(temp,enhanced,utterances,profile_data,samples,data['utterances'])
            if sha256_file(source)!=identity['source_video_sha256']: raise ValueError('Source changed during analysis.')
            for name,digest in provenance.items():
                if fingerprint(read_json(directory/(name+'.json')))!=digest: raise ValueError('Input changed during analysis.')
            atomic_json(temp/'manifest.json',{'identity':identity,'analysis_key':key,
                        'artifacts':{str(p.relative_to(temp)):sha256_file(p) for p in temp.rglob('*') if p.is_file()}},overwrite=False)
            os.rename(temp,target)
        finally:
            if temp.exists(): shutil.rmtree(temp)
    return target,False


def main(argv=None):
    parser=argparse.ArgumentParser(description='Offline speaker refinement, voice profiling and listening review.')
    parser.add_argument('video_id');parser.add_argument('--input-dir',type=Path);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.video_id): raise ValueError('Invalid video ID.')
        path,reused=run_analysis(args.video_id,args.input_dir or BASE_DIR/'input'/args.video_id,
                                 args.output_dir or BASE_DIR/'output'/args.video_id/'speaker_analysis',OfflinePyannote())
        print(('Reused: ' if reused else 'Saved: ')+str(path),flush=True)
        return 0
    except (ValueError,OSError,RuntimeError) as error:
        print(f'Error: {error}',flush=True);return 1


if __name__=='__main__': raise SystemExit(main())
