"""Local approved-reference synthesis, immutable registration and preset comparison CLI."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

from language.cache import atomic_json, fingerprint
from language.timing import measure_wav
from language.voice import VoiceProfile, validate_clone_permission

DEFAULT_BASE = 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit'
UNKNOWN = {None, '', 'UNKNOWN', 'MIXED', 'REVIEW'}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def text_digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_reference(profile):
    """Fail closed before cache lookup as well as generation; never infer approval."""
    if profile.mode not in ('approved_clone','approved_reference'):
        raise ValueError('Expected approved_clone profile.')
    if profile.speaker in UNKNOWN or not re.fullmatch(r'[A-Za-z0-9_-]+',profile.speaker):
        raise ValueError('Explicit safe speaker ID required; UNKNOWN/MIXED cannot receive a clone.')
    if not isinstance(profile.reference_text,str) or not profile.reference_text.strip():
        raise ValueError('Exact reference text is required.')
    consent=validate_clone_permission(profile)
    measured=measure_wav(profile.reference_audio)
    audio_hash=digest(profile.reference_audio)
    if profile.reference_audio_sha256 != audio_hash:raise ValueError('Reference hash changed.')
    if measured['channels']!=1 or not 2.5<=measured['audio_duration']<=30:
        raise ValueError('Reference must be mono PCM WAV, 2.5–30 seconds.')
    if measured['clipped_sample_fraction']>.001 or measured['rms_amplitude']<.005:
        raise ValueError('Reference clipping or insufficient volume requires review.')
    review=profile.reference_text_review or {}
    if (review.get('status')!='verified' or not isinstance(review.get('verified_by'),str) or not review['verified_by'].strip()
            or review.get('reference_audio_sha256')!=audio_hash
            or review.get('reference_text_sha256')!=text_digest(profile.reference_text)):
        raise ValueError('Reference transcript needs explicit verification for this exact audio/text pair.')
    quality=profile.source_audio_quality or {}
    required=('speaker_identity_verified','single_speaker_verified','intelligibility_verified','boundaries_verified')
    if any(quality.get(k) is not True for k in required) or quality.get('known_overlap_seconds')!=0:
        raise ValueError('Reference identity/overlap/intelligibility/boundaries require explicit review.')
    if type(quality.get('known_overlap_seconds')) not in (int,float):raise ValueError('Numeric overlap review required.')
    if quality.get('unresolved_issues')!=[]:raise ValueError('Unresolved reference quality issues.')
    if (quality.get('reference_audio_sha256')!=audio_hash or not isinstance(quality.get('verified_by'),str)
            or not quality['verified_by'].strip()):
        raise ValueError('Quality review is missing or belongs to another waveform.')
    if quality.get('source_kind')=='separated' and quality.get('separation_artifacts_reviewed') is not True:
        raise ValueError('Separated reference needs explicit artifact review; prefer clean original speech.')
    return {'profile':asdict(profile),'consent_record':consent,'reference_measurement':measured,
            'reference_audio_sha256':audio_hash,'reference_text_sha256':text_digest(profile.reference_text)}


def profile_snapshot(profile):
    record=validate_reference(profile)
    record['profile_fingerprint']=fingerprint(record)
    return record


def validate_snapshot(snapshot, speaker, *, live=True):
    if not isinstance(snapshot,dict) or snapshot.get('profile',{}).get('speaker')!=speaker:
        raise ValueError('Speaker and Voice Profile do not match.')
    content={k:v for k,v in snapshot.items() if k!='profile_fingerprint'}
    if fingerprint(content)!=snapshot.get('profile_fingerprint'):raise ValueError('Voice Profile fingerprint mismatch.')
    profile=VoiceProfile(**snapshot['profile'])
    if profile.mode not in ('approved_clone','approved_reference'):raise ValueError('Clone mode/profile mismatch.')
    if live and profile_snapshot(profile)!=snapshot:raise ValueError('Reference, transcript or authorization changed; re-register/review required.')
    return profile


def load_profiles(path):
    path=Path(path).resolve();doc=json.loads(path.read_text())
    if doc.get('kind')!='synthesis_voice_profiles' or doc.get('format_version')!=1:raise ValueError('Expected synthesis_voice_profiles v1.')
    profiles={}
    for row in doc.get('profiles',[]):
        args=dict(row)
        for key in ('reference_audio','consent_reference'):
            if args.get(key):args[key]=str((path.parent/args[key]).resolve())
        profile=VoiceProfile(**args)
        if profile.speaker in profiles:raise ValueError('Duplicate speaker profile.')
        validate_reference(profile);profiles[profile.speaker]=profile
    if not profiles:raise ValueError('No voice profiles registered.')
    return profiles


def register_profile(*,speaker,audio,text_path,consent_path,review_path,output,characteristics=None):
    """Registration copies exact supplied evidence; never produces consent or verification."""
    audio=Path(audio).resolve();text=Path(text_path).read_text(encoding='utf-8')
    review=json.loads(Path(review_path).read_text());consent_path=Path(consent_path).resolve()
    p=VoiceProfile(speaker,mode='approved_clone',reference_audio=str(audio),reference_text=text,
                   reference_audio_sha256=digest(audio),consent_reference=str(consent_path),
                   reference_text_review=review.get('transcript'),source_audio_quality=review.get('quality'),
                   voice_characteristics=characteristics)
    snap=profile_snapshot(p);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    target=root/f'{speaker}-{fingerprint(snap)[:16]}'
    if target.exists():
        old=load_profiles(target/'voice_profiles.json')[speaker]
        old_data={**asdict(old),'reference_audio':str(audio),'consent_reference':str(consent_path)}
        if old_data!=asdict(p) or validate_clone_permission(old)!=snap['consent_record']:raise ValueError('Existing registration differs.')
        return target/'voice_profiles.json'
    temp=Path(tempfile.mkdtemp(prefix='.registration-',dir=root))
    try:
        shutil.copyfile(audio,temp/'reference.wav');shutil.copyfile(consent_path,temp/'consent.json')
        (temp/'reference.txt').write_text(text,encoding='utf-8');atomic_json(temp/'review.json',review)
        stored={**asdict(p),'reference_audio':'reference.wav','consent_reference':'consent.json'}
        atomic_json(temp/'voice_profiles.json',{'format_version':1,'kind':'synthesis_voice_profiles','profiles':[stored]})
        # Validate copied bytes before publication (also detects concurrent source changes).
        copied=load_profiles(temp/'voice_profiles.json')[speaker]
        if validate_clone_permission(copied)!=snap['consent_record']:raise ValueError('Authorization changed during registration.')
        temp.rename(target)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return target/'voice_profiles.json'


def prepare_review_template(audio,text_path,speaker,output):
    """Unapproved template only. Creating it grants neither consent nor quality approval."""
    audio_hash=digest(audio);text=Path(text_path).read_text(encoding='utf-8');measurement=measure_wav(audio)
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'consent.template.json',{'authorized':False,'scope':'voice_cloning','speaker_id':speaker,
        'reference_audio_sha256':audio_hash,'evidence':'','verified_by':''},overwrite=False)
    atomic_json(folder/'review.template.json',{'transcript':{'status':'pending','verified_by':'','reference_audio_sha256':audio_hash,
        'reference_text_sha256':text_digest(text)},'quality':{'reference_audio_sha256':audio_hash,'verified_by':'',
        'speaker_identity_verified':False,'single_speaker_verified':False,'intelligibility_verified':False,
        'boundaries_verified':False,'known_overlap_seconds':None,'unresolved_issues':['not_reviewed'],
        'source_kind':'original'},'measured':measurement},overwrite=False)
    return folder


def phase4_candidates(document, speaker, directory):
    """Reuse Phase 4 ranking/rejections; no candidate implies cloning authorization."""
    if document.get('kind')!='speaker_profiles' or document.get('format_version')!=1:raise ValueError('Expected Phase 4 profiles.')
    matches=[p for p in document['profiles'] if p['speaker_id']==speaker]
    if len(matches)!=1 or speaker in UNKNOWN:raise ValueError('Known unique speaker required.')
    candidates=[]
    for ref in matches[0]['reference_candidates']:
        if not ref.get('audio_file'):continue
        p=(Path(directory)/ref['audio_file']).resolve()
        if digest(p)!=ref['audio_sha256']:raise ValueError('Phase 4 reference changed.')
        quality=ref['quality'];candidates.append({'reference_id':ref['reference_id'],'audio_file':str(p),
            'reference_audio_sha256':ref['audio_sha256'],'phase4_quality':quality,'features':ref.get('features'),
            'text':ref.get('text_en',ref.get('transcript')),'transcript_evidence':ref.get('transcript',{}),'eligible_for_clone':False,
            'review_required':True,'reasons':['authorization_and_exact_transcript_review_required'],
            'source':'Phase 4 candidates; original clean single-speaker references preferred'})
    return sorted(candidates,key=lambda x:(not x['phase4_quality']['eligible'],-x['phase4_quality']['ranking_score']))


class QwenBaseSynthesizer:
    """Local Base model; reusable conditioning isolated by cryptographic profile identity."""
    supported_voice_modes={'approved_clone'}

    def __init__(self,model_id=DEFAULT_BASE,*,seed=0):
        from tts_generator import QwenPresetSynthesizer
        base=QwenPresetSynthesizer(model_id,seed=seed,expected_model_type='base')
        self.snapshot=base.snapshot;self.settings={**base.settings};self._model=None;self._conditioning={}
        package=Path(find_spec('mlx_audio').origin).parent
        self.identity={**base.identity,'adapter_version':1,'voice_mode':'approved_clone',
            'model_quantization':json.loads((self.snapshot/'config.json').read_text()).get('quantization'),
            'conditioning_cache':'in-process profile-isolated native ICL cache; no persistent embedding pickle',
            'adapter_sha256':digest(__file__),'voice_contract_sha256':digest(Path(__file__).parent/'language/voice.py'),
            'native_source_sha256':digest(package/'tts/models/qwen3_tts/qwen3_tts.py')}

    def synthesize(self,*,text,profile,output_path):
        import numpy as np
        import mlx.core as mx
        import soundfile as sf
        from qwen_tts_mlx import MLXQwen3TTS
        started=time.perf_counter();snap=profile_snapshot(profile);key=snap['profile_fingerprint']
        if not isinstance(text,str) or not text.strip():raise ValueError('Japanese text required.')
        if self._model is None:self._model=MLXQwen3TTS.from_pretrained(str(self.snapshot))
        reused=key in self._conditioning
        if not reused:
            # Use loaded arrays, not mutable paths, for repeated conditioning.
            wav,rate=sf.read(profile.reference_audio,dtype='float32')
            if digest(profile.reference_audio)!=snap['reference_audio_sha256']:raise ValueError('Reference changed during load.')
            prompt=self._model.create_voice_clone_prompt(ref_audio=(wav,rate),ref_text=profile.reference_text,x_vector_only_mode=False)
            self._conditioning[key]={'prompt':prompt,'icl_cache':{}}
        condition=self._conditioning[key];native=self._model.model
        if not hasattr(native,'_icl_cache'):raise RuntimeError('Installed mlx-audio lacks the inspected ICL cache contract.')
        # Upstream cache key uses size/sum; isolate by SHA-bound profile to prevent same-size/sum cross-speaker collision.
        native._icl_cache=condition['icl_cache'];mx.random.seed(self.settings['seed'])
        options={k:v for k,v in self.settings.items() if k not in ('seed','wav_subtype','language','instruct')}
        prompt=condition['prompt']
        try:
            results=list(native.generate(text=text,lang_code='Japanese',ref_audio=prompt.ref_audio,ref_text=prompt.ref_text,**options))
        finally:
            condition['icl_cache']=native._icl_cache;native._icl_cache={}
        if not results or any(r.token_count>=self.settings['max_tokens'] for r in results):raise RuntimeError('No audio or token limit reached; incomplete speech rejected.')
        rates={int(r.sample_rate) for r in results}
        if len(rates)!=1 or next(iter(rates))<=0:raise ValueError('Invalid output sample rate.')
        samples=np.concatenate([np.asarray(r.audio) for r in results])
        if samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all():raise ValueError('Invalid generated waveform.')
        if np.max(np.abs(samples))>1:raise ValueError('Generated amplitude outside PCM range; review required, no silent clipping.')
        if profile_snapshot(profile)!=snap:raise ValueError('Profile/authorization changed during synthesis.')
        rate=next(iter(rates));sf.write(output_path,samples,rate,subtype='PCM_16',format='WAV')
        return {'sample_rate':rate,'sample_frames':len(samples),'token_count':sum(r.token_count for r in results),
            'termination':'eos_before_limit','processing_seconds':time.perf_counter()-started,'conditioning_reused':reused,
            'profile_fingerprint':key,'reference_audio_sha256':snap['reference_audio_sha256'],
            'reference_text_sha256':snap['reference_text_sha256'],'peak_memory_bytes':mx.get_peak_memory(),
            'pronunciation_verified':False,'voice_similarity_verified':False,'review_required':True}


def compare_feedback(preset_path, clone_path, output):
    """Create an immutable listening comparison, never infer pronunciation or similarity success."""
    import html
    from audio_mixer import validate_feedback
    from urllib.parse import quote
    def load(path):
        doc=json.loads(Path(path).read_text())
        with tempfile.TemporaryDirectory() as folder:
            selection=Path(folder)/'selection.json';atomic_json(selection,doc['clip'])
            return validate_feedback(path,selection,doc['video_id'])[:2]
    preset,pr=load(preset_path);clone,cr=load(clone_path)
    if preset['video_id']!=clone['video_id'] or preset['clip']!=clone['clip']:raise ValueError('Comparison video/clip mismatch.')
    a={r['id']:r for r in pr if r['generation_status']=='complete'}
    b={r['id']:r for r in cr if r['generation_status']=='complete'}
    if not a or a.keys()!=b.keys():raise ValueError('Comparison requires identical complete utterance IDs.')
    rows=[]
    for uid,x in a.items():
        y=b[uid]
        if any(x[k]!=y[k] for k in ('speaker','text_ja','text_en','relative_start','relative_end')):raise ValueError('Comparison texts/speakers/times differ.')
        if x.get('voice_mode','preset')!='preset' or y.get('voice_mode')!='approved_clone':raise ValueError('Expected preset versus approved_clone.')
        rows.append({'id':uid,'speaker_id':x['speaker'],'text_ja':x['text_ja'],
            'reference_audio':y['voice_profile']['profile']['reference_audio'],
            'reference_audio_sha256':y['voice_profile']['reference_audio_sha256'],
            'reference_text':y['voice_profile']['profile']['reference_text'],
            'preset_audio':x['audio_file'],'clone_audio':y['audio_file'],
            'source_seconds':x['source_duration_seconds'],'estimated_seconds':x['estimated_duration_seconds'],
            'preset_measurement':x['wav_measurement'],'clone_measurement':y['wav_measurement'],
            'clone_minus_preset_seconds':y['actual_tts_duration_seconds']-x['actual_tts_duration_seconds'],
            'voice_profile_fingerprint':y['voice_profile']['profile_fingerprint'],
            'pronunciation_verified':False,'similarity_verified':False,'review_required':True})
    speakers={s:{'utterance_count':sum(r['speaker_id']==s for r in rows),
                 'same_profile_used':len({r['voice_profile_fingerprint'] for r in rows if r['speaker_id']==s})==1,
                 'audible_consistency_verified':False} for s in sorted({r['speaker_id'] for r in rows})}
    result={'format_version':1,'kind':'voice_clone_comparison','video_id':preset['video_id'],'clip':preset['clip'],
            'preset_tts':preset['tts'],'clone_tts':clone['tts'],'rows':rows,'speaker_consistency':speakers,
            'limitations':['Matching reference/profile is not perceptual voice preservation.',
                          'Pronunciation, reference-text agreement and cross-lingual similarity require listening.']}
    root=Path(output);root.mkdir(parents=True,exist_ok=True);target=root/fingerprint(result)[:16]
    if target.exists():
        if json.loads((target/'comparison.json').read_text())!=result:raise ValueError('Existing comparison changed.')
        for i,row in enumerate(rows):
            for field in ('reference_audio','preset_audio','clone_audio'):
                copied=target/f'{i}-{field}.wav'
                expected=row['reference_audio_sha256'] if field=='reference_audio' else row['preset_measurement' if field=='preset_audio' else 'clone_measurement']['sha256']
                if not copied.is_file() or digest(copied)!=expected:
                    raise ValueError('Existing comparison audio is missing or changed.')
        return target
    tmp=Path(tempfile.mkdtemp(prefix='.comparison-',dir=root))
    try:
        atomic_json(tmp/'comparison.json',result);esc=html.escape;cards=[]
        lines=['# Preset / Approved Clone comparison','','Unverified listening comparison; no accuracy claim.','','|ID|Speaker|Japanese|Source s|Estimate s|Preset s|Clone s|','|---|---|---|---:|---:|---:|---:|']
        # Copies make this review portable; source WAVs remain unchanged.
        for i,r in enumerate(rows):
            body=f'<article><h2>{esc(r["id"])} / {esc(r["speaker_id"])}</h2><p>{esc(r["text_ja"])}</p><p>Reference: {esc(r["reference_text"])}</p>'
            for field,label in [('reference_audio','Reference English'),('preset_audio','Preset Japanese'),('clone_audio','Approved Clone Japanese')]:
                name=f'{i}-{field}.wav';shutil.copyfile(r[field],tmp/name)
                expected=r['reference_audio_sha256'] if field=='reference_audio' else r['preset_measurement' if field=='preset_audio' else 'clone_measurement']['sha256']
                if digest(tmp/name)!=expected:raise ValueError('Audio changed while preparing comparison.')
                body+=f'<h3>{label}</h3><audio controls preload="none" src="{quote(name)}"></audio>'
            body+='<p>日本語の発音・欠落、声質の近さ、文章間の安定性を聞き比べてください。いずれも未確認です。</p></article>';cards.append(body)
            lines.append('|'+ '|'.join(str(v).replace('|','/') for v in [r['id'],r['speaker_id'],r['text_ja'],r['source_seconds'],r['estimated_seconds'],r['preset_measurement']['audio_duration'],r['clone_measurement']['audio_duration']])+'|')
        (tmp/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (tmp/'review.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>Voice comparison</title><style>body{font:16px system-ui;max-width:900px;margin:30px auto;padding:20px}article{border:1px solid #ccc;padding:20px;margin-bottom:20px}audio{width:100%}</style><h1>Preset / Approved Clone</h1><p>声質・発音は未確認です。実在人物の承認記録は配布せず、試聴用ファイルを適切に管理してください。</p>'+''.join(cards)+'</html>',encoding='utf-8')
        tmp.rename(target)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return target


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('template','register'):
        c=sub.add_parser(name);c.add_argument('--speaker',required=True);c.add_argument('--reference-audio',type=Path,required=True)
        c.add_argument('--reference-text',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
        if name=='register':
            c.add_argument('--consent',type=Path,required=True);c.add_argument('--review',type=Path,required=True)
            c.add_argument('--phase4-profiles',type=Path);c.add_argument('--reference-id')
    c=sub.add_parser('candidates');c.add_argument('--phase4-profiles',type=Path,required=True);c.add_argument('--speaker',required=True)
    c=sub.add_parser('compare');c.add_argument('--preset-feedback',type=Path,required=True);c.add_argument('--clone-feedback',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('generate');c.add_argument('video_id');c.add_argument('--script',type=Path,required=True)
    c.add_argument('--mode',choices=('preset','approved_clone'),required=True);c.add_argument('--profiles',type=Path,action='append')
    c.add_argument('--voice-map',type=Path);c.add_argument('--model');c.add_argument('--output',type=Path,required=True)
    c.add_argument('--seconds',type=float,default=30);c.add_argument('--max-jobs',type=int);c.add_argument('--entry-id',action='append')
    c.add_argument('--seed',type=int,default=0);c.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.command=='compare':print(compare_feedback(a.preset_feedback,a.clone_feedback,a.output));return
    if a.command=='template':print(prepare_review_template(a.reference_audio,a.reference_text,a.speaker,a.output));return
    if a.command=='register':
        characteristics=None
        if a.phase4_profiles:
            if not a.reference_id:p.error('--reference-id required with --phase4-profiles.')
            document=json.loads(a.phase4_profiles.read_text())
            choices=phase4_candidates(document,a.speaker,a.phase4_profiles.parent)
            refs=[r for r in choices if r['reference_id']==a.reference_id]
            if len(refs)!=1 or refs[0]['reference_audio_sha256']!=digest(a.reference_audio) or not refs[0]['phase4_quality']['eligible']:
                p.error('Selected Phase 4 reference is ineligible or does not match supplied audio.')
            source=next(r for r in document['profiles'] if r['speaker_id']==a.speaker)
            characteristics={'analysis_key':document.get('analysis_key'),'model':document.get('model'),
                'summary_dsp_estimates':source.get('summary_dsp_estimates'),'model_estimates':source.get('model_estimates'),
                'selected_reference':refs[0]}
        elif a.reference_id:p.error('--reference-id requires --phase4-profiles.')
        print(register_profile(speaker=a.speaker,audio=a.reference_audio,text_path=a.reference_text,consent_path=a.consent,review_path=a.review,output=a.output,characteristics=characteristics));return
    if a.command=='candidates':print(json.dumps(phase4_candidates(json.loads(a.phase4_profiles.read_text()),a.speaker,a.phase4_profiles.parent),ensure_ascii=False,indent=2));return
    from tts_feedback import plan_script,run_feedback
    from tts_generator import QwenPresetSynthesizer,DEFAULT_MODEL
    script=json.loads(a.script.read_text())
    if script['video_id']!=a.video_id:p.error('Script video ID mismatch.')
    profiles={}
    if a.mode=='approved_clone':
        if not a.profiles or a.voice_map:p.error('Clone requires --profiles and cannot accept --voice-map.')
        for path in a.profiles:
            entries=load_profiles(path)
            if profiles.keys() & entries.keys():p.error('Duplicate speaker across profile documents.')
            profiles.update(entries)
        voices={s:'approved_clone:'+profile_snapshot(v)['profile_fingerprint'] for s,v in profiles.items()}
    else:
        if not a.voice_map or a.profiles:p.error('Preset requires --voice-map and cannot accept clone profiles.')
        voices=json.loads(a.voice_map.read_text())
    if a.dry_run:
        rows=plan_script(script,voices,seconds=a.seconds,entry_ids=a.entry_id)
        print(json.dumps({'mode':a.mode,'model':a.model or (DEFAULT_BASE if profiles else DEFAULT_MODEL),
            'profiles':{s:profile_snapshot(v) for s,v in profiles.items()},'entries':rows},ensure_ascii=False,indent=2));return
    backend=QwenBaseSynthesizer(a.model or DEFAULT_BASE,seed=a.seed) if profiles else QwenPresetSynthesizer(a.model or DEFAULT_MODEL,seed=a.seed)
    print(run_feedback(a.script,a.output,backend,voices,seconds=a.seconds,max_jobs=a.max_jobs,entry_ids=a.entry_id,voice_profiles=profiles or None))


if __name__=='__main__':main()
