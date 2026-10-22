import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language.cache import atomic_json
from timing_adjustment import bounded_placements, diagnose_collision_budgets
from tts_feedback import run_feedback
from test_audio_mixer import fixture_script, Backend
import audio_mixer


def row(uid,speaker,start,end,duration,status='complete',mora=None,pause=0.0):
    r={'id':uid,'speaker':speaker,'relative_start':start,'relative_end':end,
       'actual_tts_duration_seconds':duration,'generation_status':status,'pause_seconds':pause}
    if mora is not None:
        r['mora_count']=mora
    return r


class TimingAdjustmentTests(unittest.TestCase):
    def setUp(self):self.rows=[row('a','A',0,.5,.8),row('b','A',.6,1,.3)]
    def test_bounded_delay_no_waveform_edit(self):
        before=copy.deepcopy(self.rows);p=bounded_placements(self.rows,seconds=2)
        self.assertAlmostEqual(p['starts']['b'],.84);self.assertEqual(p['playback_rate'],1)
        self.assertEqual(self.rows,before);self.assertFalse(p['waveform_modified'])
    def test_limit_keeps_unresolved_originals(self):
        p=bounded_placements(self.rows,seconds=2,max_shift=.1)
        self.assertEqual(p['starts']['b'],.6);self.assertEqual(p['events'][0]['status'],'unresolved')
    def test_preview_boundary(self):
        p=bounded_placements(self.rows,seconds=1)
        self.assertIn('preview_boundary',p['events'][0]['reasons']);self.assertEqual(p['starts']['b'],.6)
    def test_other_speaker_and_unknown_are_barriers(self):
        for speaker in ('B',None,'UNKNOWN'):
            rows=self.rows+[row('c',speaker,1.1,1.3,.2,'complete' if speaker=='B' else 'skipped')]
            p=bounded_placements(rows,seconds=2)
            self.assertEqual(p['starts']['b'],.6);self.assertIn('intervening_or_other_speaker_boundary',p['events'][0]['reasons'])
    def test_natural_simultaneous_speech_preserved(self):
        rows=self.rows+[row('c','B',.1,.4,.3)]
        p=bounded_placements(rows,seconds=2)
        self.assertEqual(p['starts'],{'a':0,'c':.1,'b':.6});self.assertIn('a',p['source_overlap_anchors'])
    def test_source_overlap_anchor_inside_adjacent_group(self):
        rows=[row('a','A',0,.4,.8),row('b','A',.5,1,.4),row('c','B',.7,1.2,.5)]
        p=bounded_placements(rows,seconds=2)
        self.assertIn('source_overlap_anchor',p['events'][0]['reasons']);self.assertEqual(p['starts']['b'],.5)
    def test_multiple_speakers_and_interleaving(self):
        rows=[row(str(i),s,i,i+.5,.4) for i,s in enumerate(['A','B','C','A','B','C'])]
        p=bounded_placements(rows,seconds=7);self.assertEqual(len(p['starts']),6);self.assertFalse(p['events'])
    def test_invalid_options_and_unknown_generated(self):
        for value in (True,float('nan'),-1,3):
            with self.assertRaises(ValueError):bounded_placements(self.rows,seconds=2,max_shift=value)
        with self.assertRaises(ValueError):bounded_placements([row('x',None,0,1,.8)],seconds=2)
    def test_mixer_uses_measured_placements_preserves_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);script=fixture_script();script['entries']=[script['entries'][0],script['entries'][2]]
            path=root/'script.json';atomic_json(path,script);selection=root/'selection.json';atomic_json(selection,script['clip'])
            with redirect_stdout(io.StringIO()):feedback=run_feedback(path,root/'tts',Backend(),{'A':'Ryan'},seconds=4)
            before=feedback.read_bytes();info=root/'source.info.json';atomic_json(info,{'id':'fixture'});video=root/'source.mp4';video.write_bytes(b'fixture')
            with patch('audio_mixer.probe',return_value={'format':{'duration':'8'},'streams':[{'codec_type':'video'}]}):
                plan=audio_mixer.build_plan(feedback,selection,video,info,'fixture',seconds=4,diagnostic=True,max_timing_shift=.5)
                blocked=audio_mixer.build_plan(feedback,selection,video,info,'fixture',seconds=4,diagnostic=True,max_timing_shift=.1)
            self.assertEqual(plan['same_speaker_conflicts'],[]);self.assertEqual(plan['blockers'],[])
            self.assertAlmostEqual(plan['placements'][1]['placement_seconds'],1.34)
            self.assertEqual(plan['placements'][1]['relative_start'],1)
            self.assertEqual(plan['placements'][1]['end_sample']-plan['placements'][1]['placement_sample'],38400)
            self.assertIn('bounded_timing_unresolved_same_speaker_collision',blocked['blockers'])
            self.assertEqual(feedback.read_bytes(),before)
    def test_diagnose_collision_budgets_computes_exact_reduction_and_mora_target(self):
        # Simulate the real 11.26s English -> 16.56s Japanese TTS overrun colliding with next same-speaker turn at 13.50s
        rows=[
            row('dub-002','SPEAKER_00',1.90,13.16,16.56,mora=105,pause=0.75),
            row('dub-004','SPEAKER_01',5.00,6.20,1.10,mora=8,pause=0.0),  # Natural cross-speaker overlap!
            row('dub-006','SPEAKER_00',13.50,16.00,2.40,mora=16,pause=0.25),
        ]
        before=copy.deepcopy(rows)
        diag=diagnose_collision_budgets(rows,seconds=30.0,max_shift=0.5,minimum_pause=0.04)
        self.assertEqual(rows,before)
        self.assertFalse(diag['waveform_modified'])
        self.assertTrue(diag['requires_regeneration'])
        self.assertEqual(len(diag['unresolved_same_speaker_collisions']),1)
        collision=diag['unresolved_same_speaker_collisions'][0]
        self.assertEqual(collision['speaker'],'SPEAKER_00')
        self.assertEqual(collision['entry_ids'],['dub-002','dub-006'])
        b_map={b['id']:b for b in diag['entry_budgets']}
        b2=b_map['dub-002']
        self.assertTrue(b2['requires_script_shortening'])
        # Because dub-002 overlaps with SPEAKER_01 in source, it is anchored; dub-006 starts at 13.50s (minus 0.04s pause = 13.46s ceiling -> 11.56s max duration)
        self.assertAlmostEqual(b2['max_allowable_duration_seconds'],11.56,places=2)
        self.assertAlmostEqual(b2['required_reduction_seconds'],5.00,places=2)
        self.assertLess(b2['target_mora_budget'],b2['current_mora_count'])
        self.assertGreater(b2['mora_reduction_needed'],20)
        # Cross-speaker overlapping turn dub-004 has no same-speaker collision and needs 0 reduction.
        self.assertFalse(b_map['dub-004']['requires_script_shortening'])
        self.assertEqual(b_map['dub-004']['required_reduction_seconds'],0.0)


if __name__ == '__main__':
    unittest.main()
