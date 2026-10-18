import copy
from contextlib import redirect_stdout
import io
import json
import math
import os
import types
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from audio_mixer import attach_speaker_evidence,ffmpeg
from language.cache import atomic_json,fingerprint
from language.voice import (VoiceProfile,ProfiledPresetSynthesizer,profile_for_synthesis,synthesize_profile)
from speaker_refinement import (stable_mapping,word_evidence,refine_utterances,bound_turns,
                                run_analysis,validate_inputs,OfflinePyannote,read_audio)
from speaker_review import validate_decisions,apply_review,build_review
from voice_profiler import (acoustic_features,reference_windows,reference_quality,overlap_regions,
                            subtract,centroid,cosine)


def tone(hz=200,seconds=3):
    return (.2*np.sin(2*np.pi*hz*np.arange(round(seconds*16000))/16000)).astype(np.float32)


def turn(s,a,b):
    return {'speaker':s,'relative_start':a,'relative_end':b,'start':1+a,'end':1+b}


def word(text,a,b):
    return {'word':text,'relative_start':a,'relative_end':b,'start':1+a,'end':1+b}


def profile_document():
    return {'kind':'speaker_profiles','format_version':1,'analysis_key':'a'*64,'video_id':'fixture','clip':{'start':1,'end':19},
            'profiles':[{'speaker_id':'A','summary_dsp_estimates':{'median_reference_f0_hz':200},'model_estimates':{'speaker_embedding':[1,0]},
                         'quality':{'review_required':True},'reference_candidates':[]}]}


class AcousticTests(unittest.TestCase):
    def test_f0_known_tone_and_measured_duration(self):
        r=acoustic_features(tone())
        self.assertEqual(r['measured']['sample_frames'],48000)
        self.assertAlmostEqual(r['measured']['duration_seconds'],3)
        self.assertAlmostEqual(r['dsp_estimates']['f0_hz_median'],200,delta=2)
        self.assertNotIn('f0_hz_median',r['measured'])

    def test_noise_proxy_rejects_unvoiced_reference(self):
        samples=np.random.default_rng(0).normal(0,.08,48000)
        r=acoustic_features(samples)
        self.assertFalse(reference_quality(r,.9,.6)['eligible'])
        self.assertIn('noise_like_spectral_flatness_proxy',reference_quality(r,.9,.6))

    def test_pitch_variation(self):
        r=acoustic_features(np.concatenate([tone(110,1),tone(220,1),tone(330,1)]))
        self.assertGreater(r['dsp_estimates']['f0_hz_p90']-r['dsp_estimates']['f0_hz_p10'],150)
        self.assertGreater(r['dsp_estimates']['f0_hz_std'],50)

    def test_pause_and_asr_rate_are_estimates(self):
        x=np.concatenate([tone(200,1),np.zeros(8000),tone(200,1)])
        r=acoustic_features(x,words=[word('yes',0,.2),word('hello',1.6,2)])
        self.assertAlmostEqual(r['dsp_estimates']['pause_seconds'],.5,delta=.06)
        self.assertAlmostEqual(r['asr_estimates']['words_per_second'],.8)
        self.assertGreater(r['asr_estimates']['word_gap_seconds'][0],1)

    def test_silent_and_invalid_waveforms(self):
        r=acoustic_features(np.zeros(16000));self.assertIsNone(r['dsp_estimates']['f0_hz_median'])
        self.assertFalse(reference_quality(r,.9,.8)['eligible'])
        for x in [[],[float('nan')],[2.0],[[.1]]]:
            with self.subTest(x=x),self.assertRaises(ValueError):acoustic_features(x)

    def test_reference_quality_rejects_overlap_clipping_short_and_uncertain(self):
        r=acoustic_features(tone())
        self.assertTrue(reference_quality(r,.9,.5)['eligible'])
        for kwargs in [{'overlap_seconds':.01},{'stable_identity':False}]:
            self.assertFalse(reference_quality(r,.9,.5,**kwargs)['eligible'])
        self.assertFalse(reference_quality(r,.5,.05)['eligible'])
        self.assertFalse(reference_quality(acoustic_features(tone(seconds=1)),.9,.5)['eligible'])
        r['measured']['clipped_fraction']=.1;self.assertFalse(reference_quality(r,.9,.5)['eligible'])

    def test_long_turn_multiple_windows_no_overlap(self):
        turns=[turn('A',0,30),turn('B',10,14)]
        windows=reference_windows(turns)
        self.assertGreater(len(windows),3)
        for r in windows:
            self.assertFalse(r['relative_start']<14 and r['relative_end']>10)
        self.assertEqual(overlap_regions(turns),[(10,14)])

    def test_excluded_old_overlap_never_selected(self):
        windows=reference_windows([turn('A',0,20)],excluded=[(4,8)])
        self.assertTrue(windows)
        self.assertTrue(all(not(r['relative_start']<8 and r['relative_end']>4) for r in windows))
        self.assertEqual(subtract([(0,10)],[(2,4),(3,6)]),[(0,2),(6,10)])

    def test_embedding_validation_and_centroid(self):
        self.assertAlmostEqual(cosine([1,0],[0,1]),0)
        self.assertAlmostEqual(cosine(centroid([[1,0],[2,0]]),[1,0]),1)
        for v in [[0,0],[float('nan'),1],[]]:
            with self.assertRaises(ValueError):cosine(v,[1,0])

    def test_global_mapping_three_speakers_and_swapped_labels(self):
        old={'A':[1,0,0],'B':[0,1,0],'C':[0,0,1]}
        new={'x':[0,1,0],'y':[0,0,1],'z':[1,0,0]}
        mapping,evidence=stable_mapping(new,old,[turn('x',0,3),turn('y',3,6),turn('z',6,9)],[])
        self.assertEqual(mapping,{'x':'B','y':'C','z':'A'})
        self.assertTrue(all(r['matched_previous_id'] for r in evidence))

    def test_ambiguous_mapping_does_not_merge_or_force_identity(self):
        mapping,evidence=stable_mapping({'x':[1,1]},{'A':[1,0],'B':[0,1]},[turn('x',0,3)],[])
        self.assertTrue(mapping['x'].startswith('SPEAKER_NEW'))
        self.assertFalse(evidence[0]['matched_previous_id'])

    def test_pyannote_final_frame_overshoot_explicit(self):
        t=turn('A',0,18.098);bounded,notes=bound_turns([t],{'start':1,'end':19})
        self.assertEqual(bounded[0]['relative_end'],18);self.assertEqual(t['relative_end'],18.098)
        self.assertTrue(notes)
        with self.assertRaises(ValueError):bound_turns([turn('A',0,19)],{'start':1,'end':19})


class AlignmentTests(unittest.TestCase):
    def test_unknown_recovery_requires_temporal_and_embedding_agreement(self):
        w=word('hello',2,2.3);old={**w,'speaker':None,'status':'uncertain','overlapping_speech':False}
        windows=[{'relative_start':1,'relative_end':4,'embedding':[1,0],'speaker_id':'A'}]
        r=word_evidence([w],[old],[turn('A',0,5)],windows,{'A':[1,0],'B':[0,1]})[0]
        self.assertEqual(r['speaker'],'A');self.assertTrue(r['review_required'])
        self.assertIn('contextual',r['embedding_scope']['kind'])
        r=word_evidence([w],[old],[turn('A',0,5)],[],{'A':[1,0]})[0]
        self.assertIsNone(r['speaker'])

    def test_overlap_and_short_words_not_forced(self):
        for duration,turns in [(.05,[turn('A',0,5)]),(.3,[turn('A',0,5),turn('B',1,4)])]:
            w=word('yes',2,2+duration);old={**w,'speaker':None,'status':'overlap','overlapping_speech':True}
            r=word_evidence([w],[old],turns,[],{})[0]
            self.assertIsNone(r['speaker']);self.assertTrue(r['review_required'])

    def test_embedding_contradiction_becomes_review(self):
        w=word('hello',2,2.3);old={**w,'speaker':'A','status':'single'}
        r=word_evidence([w],[old],[turn('A',0,5)],[{'relative_start':1,'relative_end':4,'embedding':[0,1]}],{'A':[1,0],'B':[0,1]})[0]
        self.assertIsNone(r['speaker']);self.assertIn('embedding_temporal_disagreement',r['review_reasons'])

    def test_boundary_and_no_speech_conflict_saved(self):
        w=word('yes',.1,.4);old={**w,'speaker':None,'status':'uncertain'}
        r=word_evidence([w],[old],[turn('A',.15,1)],[],{})[0]
        self.assertTrue(r['boundary_checks'])
        r=word_evidence([w],[old],[],[],{})[0];self.assertTrue(r['diarization_asr_conflict'])

    def test_unknown_utterance_never_automatically_confirmed(self):
        w={**word('hello',1,2),'word_id':1,'speaker':'A','review_required':False}
        u={**turn(None,1,2),'id':1,'text_en':'hello','words':[w],'review_reasons':[]}
        class Backend:
            def embed(self,x):return [1,0]
        result=refine_utterances([u],[w],[turn('A',0,4)],{'A':[1,0]},Backend(),tone(seconds=4))[0]
        self.assertIsNone(result['speaker']);self.assertEqual(result['proposed_speaker'],'A')
        self.assertTrue(result['temporal_candidates']);self.assertTrue(result['review_required'])


class VoiceInterfaceTests(unittest.TestCase):
    def test_preset_analysis_passed_without_reference_audio(self):
        class Backend:
            identity={'model':'fake'}
            def synthesize(self,**kwargs):return kwargs['profile']
        d=profile_document();backend=ProfiledPresetSynthesizer(Backend(),d,video_id='fixture',clip=d['clip'])
        p=backend.synthesize(text='日本語',profile=VoiceProfile('A',voice='Ryan'),output_path='unused')
        self.assertEqual(p.voice,'Ryan');self.assertIsNone(p.reference_audio)
        self.assertEqual(p.analysis['summary_dsp_estimates']['median_reference_f0_hz'],200)
        self.assertIn('speaker_profiles_fingerprint',backend.identity)

    def test_voice_design_explicit_unsupported_backend_fails(self):
        p=profile_for_synthesis(profile_document(),'A',mode='voice_design',design_description='Synthetic voice with moderate pitch variation')
        self.assertIsNone(p.reference_audio)
        with self.assertRaisesRegex(ValueError,'does not implement'):synthesize_profile(object(),text='test',profile=p,output_path='unused')
        with self.assertRaises(ValueError):VoiceProfile('A',mode='voice_design')

    def test_clone_permission_bound_to_exact_audio_and_speaker(self):
        with tempfile.TemporaryDirectory() as temp:
            audio=Path(temp)/'reference.wav';audio.write_bytes(b'fixture')
            consent=Path(temp)/'consent.json'
            p=VoiceProfile('A',mode='approved_clone',reference_audio=str(audio),consent_reference=str(consent))
            class Backend:
                supported_voice_modes={'approved_clone'}
                def synthesize(self,**kwargs):return {'mock':True}
            atomic_json(consent,{'authorized':False})
            with self.assertRaises(ValueError):synthesize_profile(Backend(),text='test',profile=p,output_path='unused')
            from audio_mixer import sha256_file
            atomic_json(consent,{'authorized':True,'speaker_id':'A','reference_audio_sha256':sha256_file(audio),'scope':'voice_cloning','evidence':'synthetic test permission','verified_by':'fixture'})
            self.assertEqual(synthesize_profile(Backend(),text='test',profile=p,output_path='unused'),{'mock':True})
            audio.write_bytes(b'changed')
            with self.assertRaises(ValueError):synthesize_profile(Backend(),text='test',profile=p,output_path='unused')

    def test_offline_backend_only_uses_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for n in ['config.yaml','embedding/pytorch_model.bin','segmentation/pytorch_model.bin','plda/plda.npz','plda/xvec_transform.npz']:
                (p/n).parent.mkdir(parents=True,exist_ok=True);(p/n).write_bytes(b'fixture')
            calls=[]
            def snapshot(name,**kwargs):calls.append((name,kwargs));return str(p)
            module=types.SimpleNamespace(snapshot_download=snapshot)
            with patch.dict('sys.modules',{'huggingface_hub':module}),patch('speaker_refinement.version',return_value='fixture'),patch.dict(os.environ,{},clear=False):
                backend=OfflinePyannote()
                self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
                self.assertEqual(os.environ['PYANNOTE_METRICS_ENABLED'],'0')
                self.assertIsNone(backend.pipeline)
            self.assertEqual(calls[0][1],{'local_files_only':True})
            self.assertEqual(backend.identity['speaker_count'],'automatic')

    def test_unknown_and_mixed_modes_rejected(self):
        for kwargs in [{'speaker':'UNKNOWN'},{'speaker':'A','reference_audio':'x'},{'speaker':'A','mode':'voice_design','design_description':'x','reference_audio':'x'}]:
            with self.assertRaises(ValueError):VoiceProfile(**kwargs)
        with self.assertRaises(ValueError):profile_for_synthesis(profile_document(),'missing',preset='Ryan')


class ReviewValidationTests(unittest.TestCase):
    def bundle(self):
        b={'analysis_key':'a','speakers':['A','B'],'items':[{'utterance_id':1,'audio_sha256':'h'}],'reference_candidates':[]}
        b['review_key']=fingerprint(b);return b

    def test_explicit_unknown_and_human_confirmation(self):
        b=self.bundle();d={'format_version':1,'analysis_key':'a','review_key':b['review_key'],'reviewed_by':'tester',
                          'decisions':[{'utterance_id':1,'speaker':None,'audio_sha256':'h','note':'unclear'}]}
        self.assertEqual(validate_decisions(b,d),d)
        original=copy.deepcopy(d)
        for key,value in [('reviewed_by',''),('review_key','stale')]:
            d=copy.deepcopy(original);d[key]=value
            with self.assertRaises(ValueError):validate_decisions(b,d)

    def test_invalid_or_duplicate_decision_rejected(self):
        b=self.bundle()
        for updates in [[{'utterance_id':1,'speaker':'invalid','audio_sha256':'h','note':''}],
                        [{'utterance_id':1,'speaker':'A','audio_sha256':'other','note':''}],
                        [{'utterance_id':1,'speaker':'A','audio_sha256':'h','note':''}]*2]:
            with self.assertRaises(ValueError):validate_decisions(b,{'format_version':1,'analysis_key':'a','review_key':b['review_key'],'reviewed_by':'tester','decisions':updates})

    def test_mixer_blocks_changed_assignments_in_diagnostic_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);clip={'start':1,'end':19}
            atomic_json(p/'analysis.json',{'format_version':1,'kind':'speaker_analysis','video_id':'fixture','clip':clip,'analysis_key':'a'})
            atomic_json(p/'utterances.enhanced.json',{'video_id':'fixture','clip':clip,'utterances':[{'id':1,'speaker':'B','relative_start':0,'relative_end':1}]})
            plan={'video_id':'fixture','clip':clip,'placements':[{'id':'dub-1','speaker':'A','source_utterance_ids':[1],'relative_start':0,'relative_end':1}], 'inputs':{},'blockers':[],'diagnostic':True}
            attach_speaker_evidence(plan,p/'analysis.json')
            self.assertIn('speaker_assignment_changed_regenerate_script_and_feedback',plan['blockers'])
            self.assertEqual(plan['placements'][0]['speaker'],'A')


@unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required for small media fixture')
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.inputs=self.root/'input';self.inputs.mkdir();self.output=self.root/'output'
        clip={'start':1,'end':19};turns=[turn('A',0,6),turn('B',6,12),turn('C',12,18)]
        words=[word('yes',1,1.4),word('hello',7,7.5),word('okay',13,13.5)]
        self.data={'selection':clip,'diarization':{'clip':clip,'speakers':['A','B','C'],'turns':turns},
                   'word_timestamps':{'clip':clip,'words':words},
                   'word_speakers':{'clip':clip,'words':[{**w,'speaker':s,'status':'single','overlapping_speech':False} for w,s in zip(words,['A',None,'C'])]}}
        utterances=[]
        for i,(w,s) in enumerate(zip(words,['A',None,'C']),1):
            utterances.append({**w,'id':i,'speaker':s,'text_en':w['word'],'text_ja':'例','review_required':s is None,
                               'review_reasons':['unknown'] if s is None else [],'words':[{**w,'word_id':i,'speaker':s}]})
        self.data['utterances']={'video_id':'fixture','clip':clip,'format_version':3,'utterances':utterances}
        self.data['utterance_translations']={'video_id':'fixture','clip':clip,'utterances':copy.deepcopy(utterances)}
        for n,v in self.data.items():atomic_json(self.inputs/(n+'.json'),v)
        atomic_json(self.inputs/'source.info.json',{'id':'fixture'})
        ffmpeg(['-f','lavfi','-i','color=c=blue:s=32x32:r=10:d=19','-f','lavfi','-i','sine=frequency=200:sample_rate=16000:duration=19',
                '-c:v','libx264','-preset','ultrafast','-c:a','aac','-shortest',self.inputs/'source.mp4'])
        self.turns=turns
        parent=self
        class Backend:
            identity={'backend':'mock','model':'fixture'}
            def diarize(self,samples,clip_start):return {'turns':copy.deepcopy(parent.turns),'exclusive_turns':[]}
            def embed(self,samples):return [1,0,0]
        self.backend=Backend()

    def run_analysis(self):
        with redirect_stdout(io.StringIO()):return run_analysis('fixture',self.inputs,self.output,self.backend)

    def test_full_profile_review_non_destructive_and_cache(self):
        before={p.name:p.read_bytes() for p in self.inputs.iterdir()}
        folder,reused=self.run_analysis();self.assertFalse(reused)
        profiles=json.loads((folder/'speaker_profiles.json').read_text());self.assertEqual(len(profiles['profiles']),3)
        self.assertTrue((folder/'speaker_review.html').is_file())
        self.assertFalse(any(p['clone_authorization']['verified'] for p in profiles['profiles']))
        second,reused=self.run_analysis();self.assertEqual(folder,second);self.assertTrue(reused)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.inputs.iterdir()})
        (folder/'comparison.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):self.run_analysis()

    def test_review_import_preserves_original_and_changes_translation_speaker(self):
        folder,_=self.run_analysis();b=json.loads((folder/'speaker_review.json').read_text());item=next(r for r in b['items'] if r['utterance_id']==2)
        decisions={'format_version':1,'analysis_key':b['analysis_key'],'review_key':b['review_key'],'reviewed_by':'test human',
                   'decisions':[{'utterance_id':2,'speaker':b['speakers'][0],'audio_sha256':item['audio_sha256'],'note':'synthetic fixture'}]}
        path=self.root/'decisions.json';atomic_json(path,decisions)
        original=(self.inputs/'utterances.json').read_bytes()
        target=apply_review(folder,path,self.inputs,self.root/'reviewed')
        self.assertEqual(original,(self.inputs/'utterances.json').read_bytes())
        u=json.loads((target/'utterances.json').read_text());t=json.loads((target/'utterance_translations.json').read_text())
        self.assertEqual(u['utterances'][1]['speaker'],b['speakers'][0]);self.assertEqual(t['utterances'][1]['speaker'],b['speakers'][0])
        self.assertEqual(apply_review(folder,path,self.inputs,target),target)
        self.assertTrue((target/'speaker_timeline.json').is_file())

    def test_input_video_clip_and_word_mismatch(self):
        for name,key,value in [('diarization','clip',{'start':2,'end':19}),('utterances','video_id','other')]:
            original=(self.inputs/(name+'.json')).read_bytes();d=json.loads(original);d[key]=value;atomic_json(self.inputs/(name+'.json'),d)
            with self.assertRaises(ValueError):validate_inputs(self.inputs,'fixture')
            (self.inputs/(name+'.json')).write_bytes(original)

    def test_backend_failure_never_publishes_partial_run(self):
        with patch.object(self.backend,'diarize',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):self.run_analysis()
        self.assertEqual([p.name for p in self.output.iterdir()],['.analysis.lock'])


if __name__=='__main__':unittest.main()
