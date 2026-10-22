import json
from pathlib import Path
import shutil
import tempfile
import unittest

from language.cache import atomic_json
from voice_quality import compare_voices
from test_audio_mixer import tone


class Embedder:
    identity={'model':'mock-embedding'}
    def __init__(self):self.calls=0
    def embed(self,x):self.calls+=1;return [1.,0.,0.]


@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
class VoiceQualityTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        for n in ('ref','preset','clone'):tone(self.root/(n+'.wav'),seconds=1)
        self.path=self.root/'request.json';self.doc={'kind':'voice_quality_request','format_version':1,
            'references':{'A':'ref.wav'},'samples':[{'id':'one','speaker':'A','text_ja':'テストです。','preset_audio':'preset.wav','clone_audio':'clone.wav'},
            {'id':'two','speaker':'A','text_ja':'次のテストです。','preset_audio':'preset.wav','clone_audio':None}]}
    def run_quality(self,embedder=None):
        atomic_json(self.path,self.doc)
        return compare_voices(self.path,self.root/'quality',embedder=embedder)
    def test_single_speaker_no_automatic_acceptance(self):
        p=self.run_quality(Embedder());d=json.loads(p.read_text())
        self.assertFalse(d['is_authorization_record']);self.assertTrue(d['review_required'])
        self.assertEqual(d['rows'][0]['reference_cosine'],1);self.assertEqual(d['rows'][0]['other_reference_cosines'],{})
        self.assertIsNone(d['consistency'][0]['acceptance_threshold']);self.assertFalse(d['rows'][0]['pronunciation_verified'])
    def test_one_two_three_speakers(self):
        for n in (1,2,3):
            self.doc['references']={s:'ref.wav' for s in 'ABC'[:n]}
            self.doc['samples']=[{'id':s,'speaker':s,'text_ja':'テストです。','preset_audio':'preset.wav'} for s in 'ABC'[:n]]
            p=self.run_quality(Embedder());d=json.loads(p.read_text());self.assertEqual(len(d['references']),n)
    def test_no_embedding_is_explicitly_unmeasured(self):
        d=json.loads(self.run_quality().read_text());self.assertIsNone(d['embedding_identity'])
        self.assertIsNone(d['rows'][0]['reference_cosine'])
    def test_cache_reuses_analysis_and_rejects_corrupt_copy(self):
        b=Embedder();p=self.run_quality(b);before=b.calls;self.assertEqual(self.run_quality(b),p);self.assertEqual(b.calls,before)
        (p.parent/'audio-0.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.run_quality(b)
    def test_sources_not_modified_and_no_missing_profile(self):
        before=(self.root/'ref.wav').read_bytes();self.run_quality();self.assertEqual((self.root/'ref.wav').read_bytes(),before)
        self.doc['samples'][0]['speaker']='OTHER'
        with self.assertRaises(ValueError):self.run_quality()
    def test_duplicate_ids_and_unknown(self):
        self.doc['samples'][1]['id']='one'
        with self.assertRaises(ValueError):self.run_quality()
        self.doc['references']={'UNKNOWN':'ref.wav'}
        with self.assertRaises(ValueError):self.run_quality()
