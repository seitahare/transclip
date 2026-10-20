"""Synthetic fixtures have known routing/identity truth; they do not measure real separation accuracy."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np
from separation_backends import NoASR, NoSeparator, SeparationSpec, WhisperProcess
from speech_separation import (SpeakerRegistry, assess_stems, execute, load_cached_job, max_concurrency,
                               plan_windows, read_window, reject_colliding_matches, wav_info, write_wav)
from speaker_transcription import align_segments, compare_transcript
from separation_review import export_reviewed, validate_decisions, mixer_evidence
from language.cache import fingerprint


def turn(s,a,b):return {'speaker':s,'relative_start':a,'relative_end':b}
def activity(n=400):return {'frame_start':0.,'frame_step':.01,'frame_duration':.01,'votes':[[1.,1.]]*n}
def anchors():return {'alice':[{'embedding':[1.,0.,0.]}],'bob':[{'embedding':[0.,1.,0.]}]}


class MockSeparator:
    spec=SeparationSpec('synthetic','1',(2,3),16000,None,max_window_seconds=8.)
    identity=asdict(spec)
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    def separate(self,x,rate,count):
        self.calls+=1
        if self.fail:self.fail=False;raise RuntimeError('injected failure')
        t=np.arange(len(x))/rate
        return [(np.sin(2*np.pi*(180+i*100)*t)*.05).astype(np.float32) for i in range(count)]


class MockASR(NoASR):
    identity={'backend':'synthetic_known_text'}
    def transcribe(self,path):
        with wave.open(str(path),'rb') as w:d=w.getnframes()/w.getframerate()
        return {'segments':[{'words':[{'word':'hello','start':d*.3,'end':d*.7,'probability':.9}], 'avg_logprob':-.2,'no_speech_prob':.01}]}


class RoutingTests(unittest.TestCase):
    def test_one_speaker_original(self):
        self.assertTrue(all(j['route']=='original' for j in plan_windows([turn('solo',0,3)],0,3,activity=activity())))
    def test_two_overlap(self):
        p=plan_windows([turn('a',0,3),turn('b',1,2)],0,3,activity=activity());self.assertEqual([j['simultaneous_speaker_count'] for j in p],[1,2,1]);self.assertEqual(p[1]['route'],'separate')
    def test_global_three_simultaneous_two(self):
        ts=[turn('a',0,2),turn('b',1,3),turn('late',3,4)];p=plan_windows(ts,0,4,activity=activity());self.assertEqual(max_concurrency(ts,0,4),2);self.assertEqual(p[-1]['candidate_speakers'],['late'])
    def test_three_simultaneous(self):
        p=plan_windows([turn(s,0,2) for s in ('a','b','c')],0,2,activity=activity());self.assertEqual(p[0]['context_max_concurrency'],3)
    def test_short_backchannel_retained(self):
        p=plan_windows([turn('a',0,2),turn('b',1,1.06)],0,2,activity=activity());self.assertAlmostEqual(p[1]['core_end']-p[1]['core_start'],.06);self.assertIn('C_short_interjection_possible',p[1]['overlap_status']['hypotheses'])
    def test_no_frame_evidence_not_proof(self):
        p=plan_windows([turn('a',0,2),turn('b',0,2)],0,2);self.assertEqual(p[0]['route'],'original');self.assertFalse(p[0]['overlap_status']['human_verified'])
    def test_long_clip_chunked(self):
        p=plan_windows([turn('solo',0,3600)],0,3600);self.assertTrue(all(j['window_end']-j['window_start']<=8.00001 for j in p));self.assertAlmostEqual(sum(j['core_end']-j['core_start'] for j in p),3600)
    def test_padding_capacity(self):
        p=plan_windows([turn('a',0,3),turn('b',1,2),turn('c',2,2.2)],0,3,activity=activity());self.assertEqual(max(j['context_max_concurrency'] for j in p),2)
    def test_invalid_times(self):
        for args in [(1,0),(0,float('nan'))]:
            with self.assertRaises(ValueError):plan_windows([], *args)


class IdentityTests(unittest.TestCase):
    def test_permutation(self):
        r=SpeakerRegistry(anchors());first=[r.identify(v,anchors(),0,2)['speaker_id'] for v in ([1,0,0],[0,1,0])];second=[r.identify(v,anchors(),3,5)['speaker_id'] for v in ([0,1,0],[1,0,0])];self.assertEqual(first,second[::-1])
    def test_short_unknown(self):self.assertEqual(SpeakerRegistry().identify([1,0,0],anchors(),0,.3)['speaker_id'],'UNKNOWN')
    def test_ambiguous_unknown(self):self.assertEqual(SpeakerRegistry().identify([1,1,0],anchors(),0,2)['speaker_id'],'UNKNOWN')
    def test_new_speaker_requires_repetition(self):
        r=SpeakerRegistry(['alice','bob']);self.assertEqual(r.identify([0,0,1],anchors(),0,2)['speaker_id'],'UNKNOWN');new=r.identify([0,0,1],anchors(),3,5);self.assertTrue(new['speaker_id'].startswith('SPEAKER_NEW_'));self.assertIn(new['speaker_id'],r.known)
    def test_overlapping_context_not_independent(self):
        r=SpeakerRegistry();r.identify([0,0,1],anchors(),0,2);self.assertEqual(r.identify([0,0,1],anchors(),1,3)['speaker_id'],'UNKNOWN')
    def test_no_reference_no_new_identity(self):self.assertEqual(SpeakerRegistry().identify([0,0,1],{},0,2)['speaker_id'],'UNKNOWN')
    def test_collision(self):
        rows=[{'speaker_id':'a'},{'speaker_id':'a'},{'speaker_id':'b'}];reject_colliding_matches(rows);self.assertEqual([r['speaker_id'] for r in rows],['UNKNOWN','UNKNOWN','b'])


class AudioAndASRTests(unittest.TestCase):
    def test_wav_exact(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.wav';write_wav(p,np.ones(16001)*.1);self.assertEqual(wav_info(p)['duration'],16001/16000);self.assertEqual(len(read_window(p,0,.5)),8000)
    def test_truncated_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.wav';write_wav(p,np.ones(16000)*.1);p.write_bytes(p.read_bytes()[:-20])
            with self.assertRaises(ValueError):read_window(p,0,1)
    def test_invalid_waveform(self):
        for x in ([float('nan')],[],[[1,2]]):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):write_wav(Path(d)/'x.wav',x)
    def test_missing_stem(self):
        x=np.sin(np.arange(16000)*.2);q=assess_stems(x,[x,np.zeros_like(x)],2);self.assertIn('missing_or_nearly_silent_stem',q['review_reasons'])
    def test_leakage(self):
        x=np.sin(np.arange(16000)*.2);self.assertIn('duplicate_or_leaking_stems',assess_stems(x,[x/2,x/2],2)['review_reasons'])
    def test_wrong_length_and_nan(self):
        for ys in ([np.ones(10),np.ones(9)],[np.ones(10),np.full(10,np.nan)]):
            with self.assertRaises(ValueError):assess_stems(np.ones(10),ys,2)
    def test_word_absolute_times(self):
        rows=align_segments({'segments':[{'words':[{'word':'hello','start':.1,'end':.5}]}]},clip_start=100,window_start=2,window_end=3,core_start=2,core_end=3,speaker_id='who',source='mock',separation_source={});self.assertEqual(rows[0]['words'][0]['start'],102.1)
    def test_word_boundary_ownership(self):
        result={'segments':[{'words':[{'word':'hi','start':.4,'end':.6}]}]};base=dict(clip_start=0,window_start=0,window_end=1,speaker_id='x',source='mock',separation_source=None)
        self.assertEqual(align_segments(result,core_start=0,core_end=.5,**base),[]);self.assertEqual(len(align_segments(result,core_start=.5,core_end=1,**base)),1)
    def test_invalid_word_times_not_fabricated(self):
        r=align_segments({'segments':[{'words':[{'word':'x','start':0,'end':20}]}]},clip_start=0,window_start=0,window_end=1,core_start=0,core_end=1,speaker_id='UNKNOWN',source='mock',separation_source=None);self.assertEqual(r,[])
    def test_hallucination_flag(self):
        r={'text':'invented','words':[{'word':'invented','relative_start':0,'relative_end':1}],'quality':{'no_speech_prob':.9}};c=compare_transcript(r,[]);self.assertEqual(c['unsupported_words'],['invented']);self.assertFalse(c['automatic_confirmation_allowed'])
    def test_python_venv_symlink_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'model.bin').write_bytes(b'test');(p/'python').symlink_to('/usr/bin/python3');
            with patch('separation_backends.subprocess.check_output',return_value='{}'):
                w=WhisperProcess(p/'python',p);self.assertEqual(w.python,p/'python')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.audio=self.root/'source.wav';write_wav(self.audio,(np.sin(np.arange(32000)*.1)*.1).astype(np.float32))
        self.source=[{'id':1,'speaker':None,'text_en':'hello','relative_start':0,'relative_end':2,'start':100,'end':102,'words':[]}]
    def run_pipeline(self,separator=None,**kw):
        args=dict(video_id='generic',clip={'start':100,'end':102},audio=self.audio,turns=[turn('alpha',0,2),turn('beta',0,2)],utterances=self.source,source_words=[],output=self.root/'out',separator=separator or MockSeparator(),asr=MockASR(),activity=activity(),end=2)
        args.update(kw);run=execute(**args);return run,json.loads((run/'speaker_separation.json').read_text())
    def test_cache_reuse_and_nonmutation(self):
        s=MockSeparator();before=copy.deepcopy(self.source);r,b=self.run_pipeline(s);r2,b2=self.run_pipeline(s);self.assertEqual(s.calls,1);self.assertEqual(b2['metrics']['cache_hits'],1);self.assertEqual(self.source,before);self.assertNotEqual(r,r2)
    def test_resume_after_failure(self):
        s=MockSeparator(fail=True);_,b=self.run_pipeline(s);self.assertEqual(b['metrics']['failed_jobs'],1);_,b=self.run_pipeline(s);self.assertEqual(b['metrics']['failed_jobs'],0);self.assertEqual(s.calls,2)
    def test_corrupt_cache_regenerated(self):
        s=MockSeparator();_,b=self.run_pipeline(s);p=Path(b['jobs'][0]['stems'][0]['audio_file']);p.write_bytes(b'broken');_,b2=self.run_pipeline(s);self.assertEqual(s.calls,2);self.assertEqual(p.read_bytes(),b'broken');self.assertNotEqual(b2['jobs'][0]['stems'][0]['audio_file'],str(p))
    def test_input_change_invalidates(self):
        s=MockSeparator();self.run_pipeline(s);self.source[0]['text_en']='different';self.run_pipeline(s);self.assertEqual(s.calls,2)
    def test_capacity_not_success(self):
        _,b=self.run_pipeline(NoSeparator());self.assertEqual(b['jobs'][0]['status'],'unseparated_capacity');self.assertEqual(b['metrics']['waveform_generated_jobs'],0)
    def test_three_stems_review(self):
        run,b=self.run_pipeline(turns=[turn(s,0,2) for s in ('a','b','c')]);self.assertEqual(len(b['jobs'][0]['stems']),3);review=json.loads((run/'separation_review.json').read_text());self.assertEqual(len(review['items']),3)
    def test_different_speaker_parallel_times(self):
        _,b=self.run_pipeline();stems=b['jobs'][0]['stems'];self.assertEqual(stems[0]['transcriptions'][0]['start'],stems[1]['transcriptions'][0]['start'])
    def test_human_priority_and_downstream_schema(self):
        run,b=self.run_pipeline();review=json.loads((run/'separation_review.json').read_text());d={'format_version':1,'review_key':review['review_key'],'reviewed_by':'synthetic test','source_coverage_confirmed':True,'decisions':[{**r,'label':('alpha' if i==0 else 'beta'),'text_confirmed':True} for i,r in enumerate(review['items'])]}
        result=export_reviewed(b,review,d);self.assertEqual([u['speaker'] for u in result['utterances']],['alpha','beta']);self.assertEqual(result['same_speaker_collisions'],[]);self.assertFalse(result['downstream']['old_script_and_feedback_reuse_allowed'])
        from dubbing_script import validate_inputs
        translations={**result,'format_version':1,'utterances':[{**u,'text_ja':'こんにちは'} for u in result['utterances']]};validate_inputs(result,translations)
        d['decisions'][1]['label']='alpha';self.assertEqual(len(export_reviewed(b,review,d)['same_speaker_collisions']),1)
    def test_mixer_rejects_old_source_ids(self):
        from audio_mixer import attach_speaker_evidence
        run,b=self.run_pipeline();review=json.loads((run/'separation_review.json').read_text())
        d={'format_version':1,'review_key':review['review_key'],'reviewed_by':'test','source_coverage_confirmed':True,'decisions':[{**r,'label':'alpha','text_confirmed':True} for r in review['items']]}
        result=export_reviewed(b,review,d);(run/'utterances.json').write_text(json.dumps(result));(run/'speaker_analysis.json').write_text(json.dumps(mixer_evidence(result)))
        plan={'video_id':'generic','clip':b['clip'],'placements':[{'id':'old','speaker':'beta','source_utterance_ids':[1],'relative_start':0,'relative_end':2}],'inputs':{},'blockers':[]}
        with self.assertRaisesRegex(ValueError,'Missing source'):attach_speaker_evidence(plan,run/'speaker_analysis.json')
        plan['placements'][0]['source_utterance_ids']=[result['utterances'][0]['id']]
        attach_speaker_evidence(plan,run/'speaker_analysis.json');self.assertIn('speaker_assignment_changed_regenerate_script_and_feedback',plan['blockers'])
    def test_common_signed_scale_preserves_stem_ratio(self):
        x=np.sin(np.arange(100)*.1);ys=[-1000*x*.25,-1000*x*.75];q=assess_stems(x,ys,2)
        self.assertAlmostEqual(q['common_export_gain'],-.001,places=5);self.assertLess(q['scale_aligned_residual_rms_ratio'],1e-6)
    def test_tts_changed_speaker_voice_never_relabels_old_wav(self):
        from tts_feedback import cached_audio
        class TTS:
            identity={'model':'synthetic'}
            calls=0
            def synthesize(self,*,text,profile,output_path):
                self.calls+=1;write_wav(output_path,np.sin(np.arange(16000)*.1)*.1);return {'termination':'eos_before_limit'}
        backend=TTS();row={'speaker':'alpha','voice':'Ryan','text_ja':'こんにちは'}
        first=cached_audio(self.root,backend,row);second=cached_audio(self.root,backend,{**row,'speaker':'beta','voice':'Aiden'})
        self.assertEqual(backend.calls,2);self.assertNotEqual(first[0],second[0]);self.assertTrue(first[0].is_file())
    def test_stale_human_review_rejected(self):
        run,b=self.run_pipeline();r=json.loads((run/'separation_review.json').read_text())
        with self.assertRaises(ValueError):validate_decisions(b,r,{'format_version':1,'review_key':'wrong','reviewed_by':'test'})
    def test_unreviewed_export_blocked(self):
        run,b=self.run_pipeline();r=json.loads((run/'separation_review.json').read_text());d={'format_version':1,'review_key':r['review_key'],'reviewed_by':'test','decisions':[]}
        with self.assertRaises(ValueError):export_reviewed(b,r,d)

if __name__=='__main__':unittest.main()
