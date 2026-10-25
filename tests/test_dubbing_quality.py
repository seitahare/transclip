import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language.cache import fingerprint
from dubbing_quality import completion_assessment, apply_editorial_plan, sha
from japanese_subtitles import align_entry_cues, SubtitleStyleConfig
import dubbing_pipeline as dp
from test_audio_mixer import tone


def entry(uid=1,speaker='A',text='例です。',start=0,end=1):
    return {'id':'dub-'+str(uid),'source_utterance_ids':[uid],'speaker':speaker,'text_en':'Example.',
            'text_ja':text,'start':start,'end':end,'relative_start':start,'relative_end':end,
            'source_duration_seconds':end-start,'review_required':False,'review_reasons':[]}


class QualityTests(unittest.TestCase):
    def test_unknown_and_missing_cannot_be_clean(self):
        rows=[entry(),entry(2,None,start=1,end=2)]
        fb=[{**rows[0],'generation_status':'complete'}]
        result=completion_assessment(rows,fb,selection={'start':0,'end':3},speaker_targets=[{'id':2}])
        self.assertEqual(result['video_generation']['status'],'succeeded')
        self.assertEqual(result['utterance_coverage']['missing_source_ids'],['2'])
        self.assertFalse(result['completed_clean']);self.assertIn('speaker_review_pending',result['quality_review']['blockers'])
    def test_source_crossing_boundary_not_in_script_is_missing(self):
        rows=[entry()];fb=[{**rows[0],'generation_status':'complete'}]
        source=[{'id':1,'start':0,'end':1},{'id':70,'start':2.5,'end':5}]
        result=completion_assessment(rows,fb,selection={'start':0,'end':3},source_rows=source)
        self.assertEqual(result['utterance_coverage']['missing_source_ids'],['70'])
        self.assertEqual(result['utterance_coverage']['boundary_source_ids'],[70])
    def test_media_success_does_not_approve_quality(self):
        rows=[entry()];fb=[{**rows[0],'generation_status':'complete'}]
        r=completion_assessment(rows,fb,selection={'start':0,'end':2},source_rows=[{'id':1,'start':0,'end':1}],subtitle_report={'review_required':False})
        self.assertEqual(r['utterance_coverage']['status'],'complete');self.assertEqual(r['quality_review']['status'],'pending')
        self.assertNotEqual(r['status'],'completed_clean')
    def test_review_and_mismatched_feedback_block(self):
        a=entry();a['review_required']=True;b={**a,'text_ja':'違う台本。','generation_status':'complete'}
        r=completion_assessment([a],[b],selection={'start':0,'end':2})
        self.assertEqual(r['utterance_coverage']['missing_source_ids'],['1'])
        self.assertIn('feedback_source_mismatch',r['quality_review']['blockers'])
    def test_merged_lineage_covers_all_source_ids(self):
        a=entry();a['source_utterance_ids']=[1,2,3]
        r=completion_assessment([a],[{**a,'generation_status':'complete'}],selection={'start':0,'end':2})
        self.assertEqual(r['utterance_coverage']['covered_source_ids'],['1','2','3'])
    def test_editorial_merge_held_until_bound_human_review(self):
        s={'clip':{'start':0,'end':3},'entries':[entry(64),entry(65,None,start=1,end=2),entry(66,start=2,end=3)]}
        before=copy.deepcopy(s)
        p={'script_fingerprint':fingerprint(s),'operations':[{'action':'merge','source_utterance_ids':[64,65,66],'sentences_ja':['たぶん、いい例だと思う。']}]}
        r=apply_editorial_plan(s,p);self.assertEqual(len(r['entries']),3);self.assertEqual(len(r['quality_revision']['held']),1)
        with tempfile.TemporaryDirectory() as folder:
            audio=Path(folder)/'source.wav';tone(audio)
            review={'script_fingerprint':fingerprint(s),'decisions':{'65':{'verified':True,'verified_by':'fixture human','speaker':'A','evidence':'listened with neighbors','source_start':1,'source_end':2,'audio_file':str(audio),'audio_sha256':sha(audio)}}}
            r=apply_editorial_plan(s,p,speaker_decisions=review)
            self.assertEqual(len(r['entries']),1);self.assertEqual(r['entries'][0]['source_utterance_ids'],[64,65,66]);self.assertTrue(r['entries'][0]['review_required'])
            audio.write_bytes(b'changed')
            with self.assertRaises(ValueError):apply_editorial_plan(s,p,speaker_decisions=review)
        self.assertEqual(s,before)
    def test_sentence_units_preserve_time_and_input(self):
        a=entry(68,text='前の長い翻訳。',end=10);s={'clip':{'start':0,'end':10},'entries':[a]};before=copy.deepcopy(s)
        p={'script_fingerprint':fingerprint(s),'operations':[{'action':'sentences','source_utterance_ids':[68],'sentences_ja':['育てる必要はない。','それでも育てることはできる。']}]}
        r=apply_editorial_plan(s,p);x=r['entries'][0]
        self.assertEqual(x['end'],10);self.assertEqual(len(x['sentence_units']),2);self.assertIsNone(x['sentence_units'][0]['audio_alignment'])
        self.assertEqual(s,before);self.assertIn('tts_regeneration_required',x['review_reasons'])
    def test_assignment_and_merge_match_source_not_dub_number(self):
        s={'clip':{'start':0,'end':3},'entries':[entry(1,start=0,end=1),entry(2,None,start=1,end=2),entry(3,start=2,end=3)]}
        for r,sid in zip(s['entries'],[64,65,66]):r['source_utterance_ids']=[sid]
        rows=dp.normalize_script_utterances(s)['utterances']
        part=dp.partition_and_manage_null_speakers(rows,explicit_speaker_assignments={'65':'A'})
        merged=dp.detect_and_merge_split_utterances(part['assigned_utterances'],enabled=False,explicit_merge_groups=['64-66'])
        self.assertEqual(merged['utterances'][0]['source_utterance_ids'],[64,65,66])
        self.assertFalse(dp._utterance_matches_merge_token(rows[1],'2'))
    def test_existing_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'keep.txt';p.write_text('keep')
            with self.assertRaisesRegex(ValueError,'empty'):dp.run_phase11_dubbing_pipeline(output_dir=folder)
            self.assertEqual(p.read_text(),'keep')

    def test_mixer_refuses_missing_or_truncated_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);wav=root/'speech.wav';tone(wav,seconds=2)
            row={'id':'x','speaker':'A','generation_status':'complete','audio_file':str(wav)}
            place={'x':{'clip_relative_start_seconds':0,'active_audio_file':str(wav)}}
            for missing in (False,True):
                if missing:wav.unlink()
                with self.assertRaises(ValueError):
                    dp._fallback_mix_audio_and_video(video_path='unused',feedback_doc={'entries':[row]},placements_map=place,selection_start=0,clip_duration=1,output_dir=root/str(missing))
                self.assertFalse((root/str(missing)/'dubbed_clip.mp4').exists())
    def test_phase12_revised_script_and_boundary_70_coverage(self):
        repo=Path(__file__).resolve().parent.parent
        orig=json.loads((repo/'output/phase11_1885_1915_inputs/dubbing_script.json').read_text(encoding='utf-8'))
        decisions=json.loads((repo/'output/phase12_1885_1915_quality/speaker_decisions.json').read_text(encoding='utf-8'))
        plan=json.loads((repo/'output/phase12_1885_1915_quality/editorial_plan.json').read_text(encoding='utf-8'))
        revised=json.loads((repo/'output/phase12_1885_1915_quality/dubbing_script.revised.json').read_text(encoding='utf-8'))
        recomputed=apply_editorial_plan(orig,plan,speaker_decisions=decisions)
        self.assertEqual([e['id'] for e in revised['entries']],['dub-001-003','dub-004','dub-005','dub-006'])
        self.assertEqual(revised['entries'][0]['source_utterance_ids'],[64,65,66])
        self.assertEqual(revised['entries'][0]['text_ja'],'たぶん、ガーデニングがいい例になると思う。')
        self.assertEqual(len(revised['entries'][2]['sentence_units']),4)
        self.assertEqual(revised['entries'][3]['speaker'],'SPEAKER_01')
        self.assertEqual([e['text_ja'] for e in recomputed['entries']],[e['text_ja'] for e in revised['entries']])
        norm=dp.normalize_script_utterances(revised)
        merged=dp.detect_and_merge_split_utterances(norm['utterances'],enabled=not bool(revised.get('quality_revision')))
        self.assertEqual(len(merged['utterances']),4)
        cues,warns=align_entry_cues({**revised['entries'][2],'placement_seconds':13.246,'actual_tts_duration_seconds':13.6},SubtitleStyleConfig())
        self.assertEqual(len(cues),4)
        self.assertIn('proportional_timing_not_audio_alignment',warns)
        source_rows=json.loads((repo/'input/XuoqKYxDHVc/utterances.json').read_text(encoding='utf-8'))['utterances']
        fb=[{**u,'generation_status':'complete'} for u in merged['utterances']]
        comp=completion_assessment(merged['utterances'],fb,selection={'start':1885.0,'end':1915.0},source_rows=source_rows)
        self.assertEqual(comp['utterance_coverage']['covered_source_ids'],['64','65','66','67','68','69'])
        self.assertEqual(comp['utterance_coverage']['missing_source_ids'],['70'])
        self.assertEqual(comp['utterance_coverage']['boundary_source_ids'],[70])
        self.assertIn('sentence_or_utterance_crosses_clip_boundary',comp['quality_review']['blockers'])
        self.assertFalse(comp['completed_clean'])


class AlignedSubtitleTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);self.wav=self.root/'speech.wav';tone(self.wav,seconds=4)
        self.row={**entry(text='最初です。次です。',end=4),'audio_file':str(self.wav),'actual_tts_duration_seconds':4,'placement_seconds':10}
        self.align={'audio_sha256':sha(self.wav),'time_basis':'wav_relative','words':[{'word':'最初です。','start':.3,'end':1.2},{'word':'次です。','start':2.1,'end':3.4}]}
    def test_real_word_timing_preserves_pause_and_text(self):
        cues,warnings=align_entry_cues(self.row,SubtitleStyleConfig(),self.align)
        self.assertEqual(len(cues),2);self.assertEqual([(c['start'],c['end']) for c in cues],[(10.3,11.2),(12.1,13.4)])
        self.assertEqual(''.join(c['text_ja'] for c in cues),self.row['text_ja']);self.assertIn('asr_word_timing_requires_listening_review',warnings)
    def test_different_wav_and_non_monotonic_words_are_not_aligned(self):
        for wrong in ('hash','order','nan'):
            a=copy.deepcopy(self.align)
            if wrong=='hash':a['audio_sha256']='bad'
            if wrong=='order':a['words'].reverse()
            if wrong=='nan':a['words'][0]['start']=float('nan')
            c,w=align_entry_cues(self.row,SubtitleStyleConfig(),a)
            self.assertEqual(c[0]['timing_source'],'tts_measured_proportional');self.assertTrue(any(x.startswith('word_alignment_rejected') for x in w))
    def test_missing_alignment_is_review_not_measured_cues(self):
        c,w=align_entry_cues(self.row,SubtitleStyleConfig())
        self.assertEqual(len(c),2);self.assertIn('proportional_timing_not_audio_alignment',w)
    def test_pipeline_uses_explicit_alignment_adapter(self):
        file=self.root/'align.json';file.write_text(json.dumps({'dub-1':self.align}))
        with patch.object(dp,'_fallback_generate_subtitles_and_mp4',return_value=('path',{})) as render:
            dp._invoke_japanese_subtitles(rel_feedback_path='unused',rel_feedback_doc={'entries':[self.row]},mix_report_path=None,dubbed_video_path='unused',output_dir=self.root/'new',clip_duration=30,alignment_json_path=file)
        self.assertEqual(render.call_args.kwargs['alignment_data']['dub-1'],self.align)
    def test_no_swallowed_invalid_alignment_file(self):
        file=self.root/'bad.json';file.write_text('{bad')
        with self.assertRaises(ValueError):dp._invoke_japanese_subtitles(rel_feedback_path='unused',rel_feedback_doc={},mix_report_path=None,dubbed_video_path='unused',output_dir=self.root/'new',clip_duration=30,alignment_json_path=file)
