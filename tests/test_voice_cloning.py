"""No models/network/real identity: consent and audio below are synthetic test fixtures."""
import copy
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from language.cache import atomic_json, fingerprint
from language.voice import VoiceProfile
from speech_separation import write_wav
from tts_feedback import run_feedback, audio_request
from voice_cloning import (QwenBaseSynthesizer, compare_feedback, digest, load_profiles, main,
                           phase4_candidates, prepare_review_template, profile_snapshot,
                           register_profile, text_digest, validate_reference, validate_snapshot)
from test_audio_mixer import fixture_script


class CloneMock:
    supported_voice_modes={'approved_clone'}
    def __init__(self,fail_at=None):self.identity={'model':'mock-base','backend':'synthetic-test','settings':{'seed':0},'model_revision':'r1'};self.calls=[];self.fail_at=fail_at
    def synthesize(self,*,text,profile,output_path):
        self.calls.append((text,asdict(profile)))
        if len(self.calls)==self.fail_at:raise RuntimeError('injected failure')
        write_wav(output_path,np.sin(np.arange(12800)*.1)*.1)
        return {'termination':'fixture','review_required':True}


class CloneTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.audio=self.root/'reference.wav';write_wav(self.audio,np.sin(np.arange(48000)*.1)*.1)
        self.text='This is a synthetic test fixture.';self.text_path=self.root/'reference.txt';self.text_path.write_text(self.text)
        self.profiles={}
        for sid in ('A','B','C'):self.profiles[sid]=self.make_profile(sid)
        self.profile=self.profiles['A'];self.script=fixture_script();self.script_path=self.root/'script.json';atomic_json(self.script_path,self.script)
        self.selection=self.root/'selection.json';atomic_json(self.selection,self.script['clip'])
    def make_profile(self,speaker):
        h=digest(self.audio);cp=self.root/f'{speaker}-consent.json'
        atomic_json(cp,{'authorized':True,'speaker_id':speaker,'reference_audio_sha256':h,'scope':'voice_cloning','evidence':'unit test fixture only; no real voice','verified_by':'test fixture'})
        review={'status':'verified','verified_by':'test fixture','reference_audio_sha256':h,'reference_text_sha256':text_digest(self.text)}
        quality={'reference_audio_sha256':h,'verified_by':'test fixture','speaker_identity_verified':True,'single_speaker_verified':True,
                 'intelligibility_verified':True,'boundaries_verified':True,'known_overlap_seconds':0,'unresolved_issues':[],'source_kind':'original'}
        return VoiceProfile(speaker,mode='approved_clone',reference_audio=str(self.audio),consent_reference=str(cp),reference_text=self.text,reference_audio_sha256=h,reference_text_review=review,source_audio_quality=quality)
    def generate(self,backend=None,profiles=None,**kwargs):
        profiles=profiles or self.profiles;backend=backend or CloneMock();voices={s:'approved_clone:'+profile_snapshot(v)['profile_fingerprint'] for s,v in profiles.items()}
        atomic_json(self.script_path,self.script)
        with redirect_stdout(io.StringIO()):path=run_feedback(self.script_path,self.root/'tts',backend,voices,seconds=4,voice_profiles=profiles,**kwargs)
        return path,json.loads(path.read_text()),backend
    def test_valid_reference_measured(self):
        r=validate_reference(self.profile);self.assertEqual(r['reference_measurement']['audio_duration'],3);self.assertEqual(r['reference_audio_sha256'],digest(self.audio))
    def test_permission_bound_to_speaker(self):
        with self.assertRaises(ValueError):validate_reference(replace(self.profile,speaker='other'))
    def test_permission_required(self):
        p=Path(self.profile.consent_reference);d=json.loads(p.read_text());d['authorized']=False;atomic_json(p,d)
        with self.assertRaises(ValueError):validate_reference(self.profile)
    def test_reference_hash_changed(self):
        write_wav(self.audio,np.sin(np.arange(48000)*.2)*.1)
        with self.assertRaises(ValueError):validate_reference(self.profile)
    def test_transcript_hash_changed(self):
        with self.assertRaises(ValueError):validate_reference(replace(self.profile,reference_text='Other words.'))
    def test_transcript_unverified(self):
        for review in ({},{**self.profile.reference_text_review,'status':'pending'},*({**self.profile.reference_text_review,'verified_by':v} for v in ('',None,False,123))):
            with self.assertRaises(ValueError):validate_reference(replace(self.profile,reference_text_review=review))
    def test_bad_quality_rejected(self):
        for change in ({'known_overlap_seconds':.1},{'known_overlap_seconds':False},{'boundaries_verified':False},{'unresolved_issues':['cut_word']},{'single_speaker_verified':False},{'reference_audio_sha256':'stale'},{'source_kind':'separated'},*({'verified_by':v} for v in ('',None,False,123))):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_reference(replace(self.profile,source_audio_quality={**self.profile.source_audio_quality,**change}))
    def test_unknown_mixed_profiles_rejected(self):
        for speaker in ('UNKNOWN','REVIEW','MIXED',None):
            with self.assertRaises(ValueError):replace(self.profile,speaker=speaker)
    def test_bad_wav(self):
        self.audio.write_bytes(b'invalid')
        with self.assertRaises(ValueError):validate_reference(self.profile)
    def test_silent_short_clipped_wav(self):
        for x in (np.zeros(48000),np.ones(1000)*.1,np.ones(48000)):
            write_wav(self.audio,x);p=self.make_profile('A')
            with self.assertRaises(ValueError):validate_reference(p)
    def test_template_does_not_grant_permission(self):
        folder=prepare_review_template(self.audio,self.text_path,'A',self.root/'templates');d=json.loads((folder/'consent.template.json').read_text());self.assertIs(d['authorized'],False)
        with self.assertRaises(FileExistsError):prepare_review_template(self.audio,self.text_path,'A',folder)
    def test_registration_copies_without_modifying_sources(self):
        review=self.root/'review.json';atomic_json(review,{'transcript':self.profile.reference_text_review,'quality':self.profile.source_audio_quality});before=self.audio.read_bytes()
        path=register_profile(speaker='A',audio=self.audio,text_path=self.text_path,consent_path=self.profile.consent_reference,review_path=review,output=self.root/'registered')
        p=load_profiles(path)['A'];self.assertEqual(p.reference_text,self.text);self.assertNotEqual(p.reference_audio,str(self.audio));self.assertEqual(self.audio.read_bytes(),before)
        again=register_profile(speaker='A',audio=self.audio,text_path=self.text_path,consent_path=self.profile.consent_reference,review_path=review,output=self.root/'registered');self.assertEqual(again,path)
    def test_duplicate_profiles(self):
        p=self.root/'profiles.json';atomic_json(p,{'kind':'synthesis_voice_profiles','format_version':1,'profiles':[asdict(self.profile)]*2})
        with self.assertRaises(ValueError):load_profiles(p)
    def test_snapshot_integrity_and_speaker(self):
        s=profile_snapshot(self.profile)
        with self.assertRaises(ValueError):validate_snapshot(s,'B')
        s['profile']['reference_text']='tampered'
        with self.assertRaises(ValueError):validate_snapshot(s,'A')
    def test_one_two_three_speaker_mapping(self):
        for n in (1,2,3):
            self.script=fixture_script();self.script['entries']=self.script['entries'][:n]
            for row,speaker in zip(self.script['entries'],('A','B','C')):row['speaker']=speaker
            path,data,backend=self.generate();self.assertEqual(len(data['entries']),n)
            for row in data['entries']:self.assertEqual(row['speaker'],row['voice_profile']['profile']['speaker'])
    def test_unknown_utterance_preserved(self):
        _,data,backend=self.generate();unknown=data['entries'][-1];self.assertIsNone(unknown['speaker']);self.assertEqual(unknown['generation_status'],'skipped');self.assertIsNone(unknown['audio_file'])
    def test_profile_map_mismatch(self):
        with self.assertRaises(ValueError):self.generate(profiles={'A':self.profiles['B']})
    def test_cached_no_regeneration(self):
        b=CloneMock();path,data,_=self.generate(b);self.generate(b);self.assertEqual(len(b.calls),3)
    def test_permission_revoked_before_cache(self):
        self.generate();p=Path(self.profile.consent_reference);d=json.loads(p.read_text());d['authorized']=False;atomic_json(p,d)
        with self.assertRaises(ValueError):self.generate()
    def test_resume_failure(self):
        with self.assertRaises(RuntimeError):self.generate(CloneMock(fail_at=2))
        path,data,b=self.generate();self.assertEqual(len(b.calls),2);self.assertTrue(data['entries'][0]['cache_reused'])
    def test_model_change_invalidates(self):
        self.generate();b=CloneMock();b.identity['model_revision']='r2';self.generate(b);self.assertEqual(len(b.calls),3)
    def test_same_reference_across_utterances(self):
        _,d,b=self.generate();a=[p for text,p in b.calls if p['speaker']=='A'];self.assertEqual(a[0],a[1]);self.assertEqual(d['entries'][0]['voice_profile'],d['entries'][2]['voice_profile'])
    def test_transcript_verified_change_invalidates(self):
        self.generate();old=self.profile;newtext='Different verified test text.';p=replace(old,reference_text=newtext,reference_text_review={**old.reference_text_review,'reference_text_sha256':text_digest(newtext)})
        _,d,b=self.generate(profiles={**self.profiles,'A':p});self.assertEqual(len(b.calls),2);self.assertTrue(d['entries'][1]['cache_reused'])
    def test_profile_characteristics_change_invalidates(self):
        self.generate();_,d,b=self.generate(profiles={**self.profiles,'A':replace(self.profile,voice_characteristics={'note':'verified test'})});self.assertEqual(len(b.calls),2)
    def test_source_json_and_reference_non_destructive(self):
        before=self.script_path.read_bytes();wav=self.audio.read_bytes();self.generate();self.assertEqual(self.script_path.read_bytes(),before);self.assertEqual(self.audio.read_bytes(),wav)
    def test_preset_request_unchanged(self):
        row={'voice':'Ryan','text_ja':'例','speaker':'A'};self.assertEqual(audio_request(row,{}),{'format_version':1,'tts':{},'voice':'Ryan','text_ja':'例','mode':'preset'})
    def test_max_jobs_and_target_id(self):
        _,d,b=self.generate(max_jobs=1);self.assertEqual(len(b.calls),1);self.assertEqual(d['summary']['pending'],2)
        _,d,b=self.generate(entry_ids=['dub-2']);self.assertEqual(len(b.calls),1);self.assertEqual(d['summary']['complete'],1)
    def test_feedback_mixer_contract(self):
        from audio_mixer import validate_feedback
        p,d,_=self.generate();report,rows,_,_=validate_feedback(p,self.selection,'fixture');self.assertEqual(rows[0]['voice_mode'],'approved_clone');self.assertTrue(rows[0]['actual_same_speaker_conflicts']);self.assertTrue(rows[0]['actual_other_speaker_overlaps'])
        d['entries'][0]['voice_profile']=d['entries'][1]['voice_profile'];atomic_json(p,d)
        with self.assertRaises(ValueError):validate_feedback(p,self.selection,'fixture')
    def test_mixer_collision_blocked_even_diagnostic(self):
        from audio_mixer import build_plan
        p,d,_=self.generate();source=self.root/'source.mp4';source.write_bytes(b'test');info=self.root/'source.info.json';atomic_json(info,{'id':'fixture'})
        with patch('audio_mixer.probe',return_value={'format':{'duration':'10'},'streams':[{'codec_type':'video'}]}):
            plan=build_plan(p,self.selection,source,info,'fixture',seconds=4,diagnostic=True)
        self.assertIn('clone_same_speaker_collision_requires_resolution',plan['blockers'])
    def test_comparison_and_portable_audio(self):
        clone,_,_=self.generate()
        class Preset(CloneMock):supported_voice_modes={'preset'}
        with redirect_stdout(io.StringIO()):preset=run_feedback(self.script_path,self.root/'preset',Preset(),{'A':'Ryan','B':'Aiden'},seconds=4)
        target=compare_feedback(preset,clone,self.root/'compare');data=json.loads((target/'comparison.json').read_text());self.assertEqual(len(data['rows']),3);self.assertEqual(len(list(target.glob('*.wav'))),9);self.assertFalse(data['rows'][0]['similarity_verified'])
        self.assertEqual(compare_feedback(preset,clone,self.root/'compare'),target)
        (target/'0-clone_audio.wav').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):compare_feedback(preset,clone,self.root/'compare')
    def test_dry_run_no_model(self):
        p=self.root/'profiles.json';atomic_json(p,{'kind':'synthesis_voice_profiles','format_version':1,'profiles':[asdict(self.profile)]})
        with patch('voice_cloning.QwenBaseSynthesizer') as model,redirect_stdout(io.StringIO()):main(['generate','fixture','--script',str(self.script_path),'--mode','approved_clone','--profiles',str(p),'--output',str(self.root/'dry'),'--dry-run'])
        model.assert_not_called();self.assertFalse((self.root/'dry').exists())
    def test_new_audio_with_updated_authorization_invalidates(self):
        self.generate();old=self.audio;self.audio=self.root/'new-reference.wav';write_wav(self.audio,np.sin(np.arange(48000)*.17)*.1)
        new=self.make_profile('A');self.audio=old
        _,d,b=self.generate(profiles={**self.profiles,'A':new});self.assertEqual(len(b.calls),2);self.assertTrue(d['entries'][1]['cache_reused'])
    def test_settings_and_text_invalidate(self):
        self.generate();b=CloneMock();b.identity['settings']['seed']=4;self.generate(b);self.assertEqual(len(b.calls),3)
        self.script['entries'][0]['text_ja']='変更した文章です。';_,d,b=self.generate();self.assertEqual(len(b.calls),1)
    def test_stereo_reference_rejected(self):
        import wave
        with wave.open(str(self.audio),'wb') as w:
            w.setparams((2,2,16000,0,'NONE','not compressed'));w.writeframes((np.tile([1000,-1000],48000).astype('<i2')).tobytes())
        p=self.make_profile('A')
        with self.assertRaises(ValueError):validate_reference(p)
    def test_permission_rechecked_before_render(self):
        from audio_mixer import render_plan
        _,d,_=self.generate();record=Path(self.profile.consent_reference);content=json.loads(record.read_text());content['authorized']=False;atomic_json(record,content)
        with self.assertRaises(ValueError):render_plan({'blockers':[],'placements':[d['entries'][0]]},self.root/'mix')
        self.assertFalse((self.root/'mix').exists())
    def test_phase4_candidates_never_authorize(self):
        doc={'kind':'speaker_profiles','format_version':1,'profiles':[{'speaker_id':'A','reference_candidates':[{'reference_id':'r','audio_file':str(self.audio),'audio_sha256':digest(self.audio),'quality':{'eligible':True,'ranking_score':1}}]}]}
        rows=phase4_candidates(doc,'A',self.root);self.assertFalse(rows[0]['eligible_for_clone'])


class NativeAdapterTests(CloneTests):
    # Only native-specific methods are discovered below; shared fixture methods inherited are excluded later.
    def make_backend(self):
        b=object.__new__(QwenBaseSynthesizer);b.snapshot=self.root/'model';b.settings={'seed':0,'max_tokens':32,'temperature':.9,'stream':False,'verbose':False};b._conditioning={}
        self.arrays=[]
        owner=self
        class Native:
            def __init__(self):self._icl_cache={};self.calls=[]
            def generate(self,**kwargs):
                self.calls.append(kwargs);self._icl_cache['shared-size-sum']='encoded'
                yield types.SimpleNamespace(audio=np.sin(np.arange(16000)*.1)*.1,sample_rate=16000,token_count=10)
        native=Native()
        class Facade:
            model=native
            def create_voice_clone_prompt(self,**kwargs):
                owner.arrays.append(kwargs);return types.SimpleNamespace(ref_audio=kwargs['ref_audio'][0],ref_text=kwargs['ref_text'])
        b._model=Facade();self.native=native
        return b
    def fake_modules(self):
        core=types.ModuleType('mlx.core');core.random=types.SimpleNamespace(seed=lambda n:None);core.get_peak_memory=lambda:1024
        mlx=types.ModuleType('mlx');mlx.core=core
        sf=types.ModuleType('soundfile');sf.read=lambda *a,**k:(np.sin(np.arange(48000)*.1)*.1,16000);sf.write=lambda path,x,rate,**kw:write_wav(path,x,rate)
        return {'mlx':mlx,'mlx.core':core,'soundfile':sf,'qwen_tts_mlx':types.SimpleNamespace(MLXQwen3TTS=object)}
    def test_native_prompt_reused_and_isolated(self):
        b=self.make_backend()
        with patch.dict('sys.modules',self.fake_modules()):
            for speaker in ('A','B','A'):b.synthesize(text='テスト',profile=self.profiles[speaker],output_path=self.root/f'{speaker}.wav')
        self.assertEqual(len(self.arrays),2);self.assertEqual(len(b._conditioning),2);self.assertEqual(self.native._icl_cache,{})
        self.assertEqual(self.native.calls[0]['lang_code'],'Japanese')
    def test_native_token_limit_rejected(self):
        b=self.make_backend();b.settings['max_tokens']=10
        with patch.dict('sys.modules',self.fake_modules()),self.assertRaises(RuntimeError):b.synthesize(text='テスト',profile=self.profile,output_path=self.root/'out.wav')
        self.assertFalse((self.root/'out.wav').exists())
    def test_native_bad_samples_rejected(self):
        b=self.make_backend();self.native.generate=lambda **k:iter([types.SimpleNamespace(audio=np.array([np.nan]),sample_rate=16000,token_count=1)])
        with patch.dict('sys.modules',self.fake_modules()),self.assertRaises(ValueError):b.synthesize(text='テスト',profile=self.profile,output_path=self.root/'out.wav')

# Share fixtures without rerunning inherited tests.
for name in dir(CloneTests):
    if name.startswith('test_') and name not in NativeAdapterTests.__dict__:setattr(NativeAdapterTests,name,None)

if __name__=='__main__':unittest.main()
