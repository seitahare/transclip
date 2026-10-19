import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from audio_mixer import attach_speaker_evidence, sha256_file
from language.cache import atomic_json, fingerprint
from speaker_attribution import (SETTINGS, activity_between, windows_for, independent_anchors,
                                 compare_embedding, quality_reasons, best_supported, decide,
                                 conversation_context, apply_human_labels, evaluate,
                                 downstream_documents, run_attribution, verified_human_labels)
from speaker_attribution import reference_duration_checks
from speaker_refinement import save_audio
from speaker_review import validate_decisions, apply_review
from voice_profiler import acoustic_features


def tone(seconds=2, frequency=180):
    return .15*np.sin(2*np.pi*frequency*np.arange(round(seconds*16000))/16000)


def activity(seconds=30, overlap=0, speech=1):
    return dict(frame_start=0, frame_step=.01, frame_duration=.01,
                votes=[[speech, overlap]]*round(seconds*100))


def anchors():
    return {s: [dict(speaker_id=s, relative_start=10+3*i, relative_end=12+3*i, embedding=v)
                for i in range(2)] for s, v in [('A', [1,0,0]), ('B', [0,1,0]), ('C', [0,0,1])]}


def window(scope='exact', seconds=2, vector=None, overlap=0):
    return dict(scope=scope, relative_start=0, relative_end=seconds,
                features=acoustic_features(tone(seconds)), quality_reasons=[],
                candidates=compare_embedding(vector or [1,0,0], anchors()),
                activity=activity_between(activity(overlap=overlap), 0, seconds))


def row(seconds=2, speaker=None, uid=1):
    return dict(id=uid, speaker=speaker, relative_start=0, relative_end=seconds,
                start=1, end=1+seconds, text_en='Yes.', words=[], review_reasons=[])


class AttributionTests(unittest.TestCase):
    def test_same_and_different_and_three_speakers(self):
        for s, vector in zip('ABC', np.eye(3).tolist()):
            result=decide(row(), [window(vector=vector), window('expanded', vector=vector)], [])
            self.assertEqual(result['speaker_id'], s)
            self.assertEqual(result['score_calibration'], 'uncalibrated_rules_not_probability')

    def test_similarity_margin_is_required(self):
        result=decide(row(), [window(vector=[1,1,0]), window('expanded', vector=[1,1,0])], [])
        self.assertIsNone(result['speaker_id'])

    def test_duration_check_is_reference_consistency_not_human_calibration(self):
        refs=anchors()
        for s,rs in refs.items(): rs.append({**rs[0],'relative_start':17,'relative_end':20})
        class Backend:
            def embed(self,x): return [1,0,0]
        checks=reference_duration_checks(refs,tone(25),Backend(),{})
        for s,groups in checks.items():
            for check in groups.values():
                self.assertFalse(check['human_calibrated'])
                for probe in check['probes']:
                    self.assertEqual(probe['peer_reference_count'],2)
                    self.assertGreaterEqual(probe['relative_start'],probe['parent_reference'][0])
        # Indistinguishable references must not create a permissive threshold.
        self.assertGreaterEqual(checks['B']['1.0']['required_similarity'],1.)

    def test_duration_threshold_does_not_remove_margin_guard(self):
        w=window(vector=[1,1,0])
        w['candidates'][0]['duration_reference_check']={'usable_probe_count':6,'required_similarity':.5}
        self.assertIsNone(best_supported(w))

    def test_repeated_views_not_independent_reference_claim(self):
        w=window(); w['candidates'][0]['reference_count']=1
        self.assertIsNone(best_supported(w))

    def test_reference_exclusion_all_context_and_deduplication(self):
        rs=[dict(speaker_id='A', relative_start=a, relative_end=b, embedding=[1,0], speaker_similarity=.9)
            for a,b in [(0,3),(2,5),(5,8),(9,12),(10,13),(15,18)]]
        profiles={'profiles':[{'speaker_id':'A','reference_candidates':[{**r,'quality':{'eligible':True}} for r in rs]}]}
        chosen=independent_anchors({'embedding_window_evidence':rs},profiles,4,9)
        self.assertEqual([(r['relative_start'],r['relative_end']) for r in chosen['A']],[(0,3),(10,13),(15,18)])

    def test_short_continuity_with_three_views(self):
        w=[window(seconds=.3)]+[window(scope) for scope in ('left_context','right_context','center_context')]
        w[0]['candidates']=[]
        result=decide(row(.3),w,[{'speaker':'A','overlap_seconds':.3}])
        self.assertEqual(result['speaker_id'],'A')
        self.assertEqual(result['method'],'short_acoustic_continuity')

    def test_short_at_speaker_boundary_not_assigned(self):
        w=[window(seconds=.3),window('left_context'),window('right_context',vector=[0,1,0]),window('center_context')]
        self.assertIsNone(decide(row(.3),w,[])['speaker_id'])

    def test_tiny_word_not_assigned_by_surrounding_voice(self):
        w=[window(seconds=.06)]+[window(scope) for scope in ('left_context','right_context','center_context')]
        self.assertIsNone(decide(row(.06),w,[])['speaker_id'])

    def test_short_pitch_disagreement(self):
        w=[window(seconds=.3)]+[window(scope) for scope in ('left_context','right_context','center_context')]
        w[0]['features']['dsp_estimates']['f0_hz_median']=400
        self.assertIsNone(decide(row(.3),w,[])['speaker_id'])

    def test_true_overlap_not_collapsed_to_dominant_embedding(self):
        result=decide(row(),[window(overlap=.9),window('expanded',overlap=.9)],[])
        self.assertEqual(result['status'],'mixed');self.assertIsNone(result['speaker_id'])
        self.assertFalse(result['overlap']['human_verified'])

    def test_temporal_overlap_alone_not_treated_as_true_mixture(self):
        result=decide(row(),[window(),window('expanded')],
                      [{'speaker':'A','overlap_seconds':2},{'speaker':'B','overlap_seconds':1.5}])
        self.assertEqual(result['speaker_id'],'A')
        self.assertIn('B_apparent_turn_overlap_not_supported_by_frame_audio',result['overlap']['hypotheses'])

    def test_local_speaker_change_not_merged(self):
        result=decide(row(),[window(),window('expanded'),window('local',vector=[0,1,0])],[])
        self.assertIsNone(result['speaker_id'])
        self.assertIn('local_windows_disagree_possible_speaker_change',result['reasons'])

    def test_silence_noise_clipping(self):
        for x in [np.zeros(32000),np.ones(32000),np.random.default_rng(2).normal(0,.1,32000)]:
            self.assertTrue(quality_reasons(acoustic_features(x)))
        w=window();w['quality_reasons']=['noise_like_spectrum']
        self.assertIsNone(decide(row(),[w,window('expanded')],[])['speaker_id'])

    def test_missing_activity_is_unknown(self):
        w=window();w['activity']=activity_between(activity(seconds=1),2,4)
        self.assertIsNone(best_supported(w))

    def test_pauses_do_not_require_half_the_span_to_contain_speech(self):
        w=window(seconds=3);w['activity']['speech_vote']=.3
        self.assertEqual(best_supported(w),'A')
        w['activity']['speech_vote']=.1
        self.assertIsNone(best_supported(w))

    def test_asr_timing_hypothesis_not_rewrite(self):
        source=row();before=copy.deepcopy(source);w=window();w['activity']['speech_vote']=.1
        result=decide(source,[w,window('expanded')],[])
        self.assertIn('D_asr_timing_or_non_speech_possible',result['overlap']['hypotheses'])
        self.assertEqual(before,source)

    def test_conversation_does_not_define_questioner(self):
        rows=[{**row(speaker='C'),'text_en':'Building robots?'},row(speaker=None,uid=2)]
        context=conversation_context(rows,1)
        self.assertTrue(context['hypotheses']);self.assertFalse(context['used_to_assign_without_audio'])
        self.assertNotIn('speaker_id',context)

    def test_windows_in_clip_and_exact_preserved(self):
        for a,b in [(0,.2),(9.9,10),(1,8)]:
            windows=windows_for(a,b,10)
            self.assertEqual(windows[0]['relative_start'],a)
            self.assertTrue(all(0<=r['relative_start']<r['relative_end']<=10 for r in windows))

    def test_human_unknown_and_known_override_automatic(self):
        for human in [None,'B']:
            auto=decide(row(),[window(),window('expanded')],[])
            result={'utterance_id':1,'automatic':auto,'speaker_id':'A','status':'single','method':'auto',
                    'human_confirmation':None,'previous_speaker':None}
            apply_human_labels([result],[row()],{1:{'speaker':human,'reviewed_by':'test human'}})
            self.assertEqual(result['speaker_id'],human)
            self.assertEqual(result['automatic']['speaker_id'],'A')
            self.assertEqual(result['method'],'human_review')

    def test_existing_human_label_never_overwritten(self):
        result={'utterance_id':1,'speaker_id':'A','status':'single'}
        apply_human_labels([result],[{**row(speaker='B'),'human_review':{'reviewed_by':'test'}}],{})
        self.assertEqual(result['speaker_id'],'B')

    def test_existing_unverified_label_not_erased_by_missing_evidence(self):
        r={'utterance_id':1,'speaker_id':None,'status':'unknown'}
        apply_human_labels([r],[row(speaker='C')],{})
        self.assertEqual(r['speaker_id'],'C');self.assertEqual(r['method'],'previous_assignment_unverified')

    def test_evaluation_only_human_labels_and_abstentions(self):
        result=[dict(previous_speaker='A',automatic={'speaker_id':'B'},human_confirmation={'speaker':'B'}),
                dict(previous_speaker=None,automatic={'speaker_id':None},human_confirmation={'speaker':'C'}),
                dict(previous_speaker=None,automatic={'speaker_id':'A'},human_confirmation=None)]
        e=evaluate(result)
        self.assertEqual(e['known_speaker_label_count'],2);self.assertEqual(e['selective_accuracy'],1)
        self.assertEqual(e['coverage_on_known_labels'],.5);self.assertFalse(e['thresholds_calibrated'])
        self.assertIsNone(evaluate([])['selective_accuracy'])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.parent=self.root/'phase4';self.parent.mkdir();self.inputs=self.root/'input';self.inputs.mkdir()
        clip={'start':1,'end':21};u=row(2,uid=1);u['words']=[{'word':'Yes.','word_id':1,'speaker':None,'start':1,'end':3,'relative_start':0,'relative_end':2}]
        w={k:v for k,v in u['words'][0].items() if k not in ('word_id','speaker')}
        turns=[{'speaker':'A','start':1,'end':11,'relative_start':0,'relative_end':10},
               {'speaker':'B','start':11,'end':21,'relative_start':10,'relative_end':20}]
        self.data={'selection':clip,'diarization':{'clip':clip,'speakers':['A','B'],'turns':turns},
                   'word_timestamps':{'clip':clip,'words':[w]},'word_speakers':{'clip':clip,'words':[{**w,'speaker':None}]},
                   'utterances':{'format_version':3,'video_id':'fixture','clip':clip,'utterances':[u]},
                   'utterance_translations':{'format_version':1,'video_id':'fixture','clip':clip,'utterances':[{**copy.deepcopy(u),'text_ja':'はい。'}]}}
        for n,d in self.data.items():atomic_json(self.inputs/(n+'.json'),d)
        atomic_json(self.inputs/'source.info.json',{'id':'fixture'});(self.inputs/'source.mp4').write_bytes(b'mocked-video-not-decoded')
        save_audio(self.parent/'selected_source.wav',tone(20))
        class Backend:
            identity={'model':'mock'}
            def activity(self,samples):return activity(20)
            def embed(self,samples):return [1,0,0]
        self.backend=Backend()
        refs=[];profiles=[]
        for s,v,starts in [('A',[1,0,0],[4,7]),('B',[0,1,0],[11,15])]:
            records=[dict(speaker_id=s,relative_start=a,relative_end=a+2.5,embedding=v,speaker_similarity=.9) for a in starts]
            refs.extend(records)
            profiles.append({'speaker_id':s,'reference_candidates':[{**r,'reference_id':f'{s}{i}','quality':{'eligible':True}} for i,r in enumerate(records)],
                             'analyzed_intervals':[{**r,'features':acoustic_features(tone(2.5))} for r in records]})
        self.analysis={'kind':'speaker_analysis','format_version':1,'analysis_key':'old','video_id':'fixture','clip':clip,'speakers':['A','B'],
                       'model':self.backend.identity,'turns':turns,'source_overlap_regions':[],
                       'input_fingerprints':{k:fingerprint(v) for k,v in self.data.items()},'embedding_window_evidence':refs}
        atomic_json(self.parent/'diarization.enhanced.json',self.analysis)
        atomic_json(self.parent/'speaker_profiles.json',{'profiles':profiles});(self.parent/'references').mkdir()
        manifest={'identity':{'source_video_sha256':sha256_file(self.inputs/'source.mp4')},
                  'artifacts':{p.name:sha256_file(p) for p in self.parent.iterdir() if p.is_file()}}
        atomic_json(self.parent/'manifest.json',manifest)

    def run_it(self,**kwargs):
        with redirect_stdout(io.StringIO()):
            return run_attribution('fixture',self.inputs,self.parent,self.root/'out',self.backend,**kwargs)

    def test_real_file_pipeline_non_destructive_cache_and_adapter(self):
        before={p.name:p.read_bytes() for p in self.inputs.iterdir()}
        folder,reused=self.run_it();self.assertFalse(reused)
        u=json.loads((folder/'utterances.json').read_text());self.assertEqual(u['utterances'][0]['speaker'],'A')
        from dubbing_script import validate_inputs
        validate_inputs(u,json.loads((folder/'utterance_translations.json').read_text()),self.data['selection'])
        plan={'video_id':'fixture','clip':self.analysis['clip'],'placements':[{'id':'dub-1','speaker':'B','source_utterance_ids':[1],
              'relative_start':0,'relative_end':2}],'inputs':{},'blockers':[]}
        attach_speaker_evidence(plan,folder/'speaker_timeline.json')
        self.assertTrue(plan['blockers'])
        self.assertEqual(self.run_it(),(folder,True))
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.inputs.iterdir()})
        self.assertEqual((folder/'human_evaluation.jsonl').read_text(),'')
        bundle=json.loads((folder/'speaker_review.json').read_text())
        self.assertEqual(bundle['items'][0]['attribution']['automatic']['speaker_id'],'A')
        (folder/'comparison.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):self.run_it()

    def test_failure_then_retry_no_partial_success(self):
        with patch.object(self.backend,'embed',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):self.run_it()
        self.assertEqual([p.name for p in (self.root/'out').iterdir()],['.attribution.lock'])
        self.assertFalse(self.run_it()[1])

    def test_human_review_evaluation_and_mixed_schema(self):
        folder,_=self.run_it();bundle=json.loads((folder/'speaker_review.json').read_text())
        decisions={'format_version':1,'analysis_key':bundle['analysis_key'],'review_key':bundle['review_key'],
                   'reviewed_by':'fixture human','decisions':[{'utterance_id':1,'speaker':'B','note':'synthetic only',
                   'audio_sha256':bundle['items'][0]['audio_sha256']}]}
        path=self.root/'decisions.json';atomic_json(path,decisions)
        result,_=self.run_it(decisions_path=path,review_dir=folder)
        u=json.loads((result/'utterances.json').read_text())['utterances'][0]
        self.assertEqual(u['speaker'],'B')
        e=json.loads((result/'evaluation.json').read_text());self.assertEqual(e['selective_accuracy'],0)
        self.assertEqual(len((result/'human_evaluation.jsonl').read_text().splitlines()),1)
        decisions['decisions'][0].update(speaker=None,attribution_status='mixed');atomic_json(path,decisions)
        validate_decisions(bundle,decisions)
        target=apply_review(folder,path,self.inputs,self.root/'reviewed')
        self.assertEqual(json.loads((target/'utterances.json').read_text())['utterances'][0]['attribution_status'],'mixed')
        translated=json.loads((target/'utterance_translations.json').read_text())['utterances'][0]
        self.assertEqual(translated['attribution_status'],'mixed');self.assertEqual(translated['attribution_method'],'human_review')
        decisions['decisions'][0]['speaker']='A'
        with self.assertRaises(ValueError):validate_decisions(bundle,decisions)

    def test_model_and_source_provenance_mismatch(self):
        self.backend.identity={'model':'different'}
        with self.assertRaisesRegex(ValueError,'models differ'):self.run_it()
        self.backend.identity={'model':'mock'};(self.inputs/'source.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'video changed'):self.run_it()


if __name__=='__main__':unittest.main()
