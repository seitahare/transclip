import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from language.cache import atomic_json
from language.voice import VoiceProfile
from synthetic_reference import generate_reference, validate_generation_record, QwenDesignSynthesizer
from voice_cloning import register_profile
from test_audio_mixer import tone


class Designer:
    identity={'backend':'mock','model':'mock-design','settings':{'seed':7,'language':'Japanese'}}
    def __init__(self):self.calls=0;self.fail=False
    def synthesize(self,*,text,profile,output_path):
        self.calls+=1
        if self.fail:raise RuntimeError('fixture failure')
        tone(output_path,seconds=3)
        return {'fixture':True}


class SyntheticReferenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name);self.backend=Designer()
    def generate(self,**kwargs):
        return generate_reference(self.backend,**{'speaker':'SYNTH_A','text':'これは検証用の文章です。',
            'instruction':'A fictional voice, no real person imitation.','output':self.root,'fictional':True,**kwargs})
    def test_generation_never_approves(self):
        path=self.generate();r=validate_generation_record(path)
        self.assertFalse(r['authorized']);self.assertFalse(r['human_verified']);self.assertEqual(r['measurement']['audio_duration'],3)
        c=json.loads((path.parent/'pending_review/consent.template.json').read_text());self.assertIs(c['authorized'],False)
    def test_cache_reuse_and_corruption(self):
        p=self.generate();self.assertEqual(p,self.generate());self.assertEqual(self.backend.calls,1)
        (p.parent/'reference.wav').write_bytes(b'bad')
        with self.assertRaises(ValueError):self.generate()
        self.assertEqual(self.backend.calls,1)
    def test_instruction_text_model_language_changes(self):
        paths={self.generate(),self.generate(text='別の文章です。'),self.generate(instruction='Another fictional voice.')}
        self.backend.identity=copy.deepcopy(self.backend.identity);self.backend.identity['settings']['language']='English'
        paths.add(self.generate());self.backend.identity['model']='another-model';paths.add(self.generate())
        self.assertEqual(len(paths),5)
    def test_failure_then_resume(self):
        self.backend.fail=True
        with self.assertRaises(RuntimeError):self.generate()
        self.assertFalse(list(self.root.glob('*/generation.json')))
        self.backend.fail=False;p=self.generate();self.assertTrue(p.is_file());self.assertFalse(list(self.root.glob('.reference-*')))
    def test_unknown_and_explicit_request(self):
        for sid in ('UNKNOWN','MIXED','REVIEW','../A'):
            with self.subTest(sid=sid),self.assertRaises(ValueError):self.generate(speaker=sid)
        with self.assertRaises(ValueError):self.generate(fictional=False)
        self.assertEqual(self.backend.calls,0)
    def test_generation_record_binding(self):
        p=self.generate()
        for kw in ({'speaker':'OTHER'},{'text':'wrong'}):
            with self.assertRaises(ValueError):validate_generation_record(p,**kw)
        r=json.loads(p.read_text());r['authorized']=True;atomic_json(p,r)
        with self.assertRaises(ValueError):validate_generation_record(p)
    def test_pending_registration_rejected_non_destructively(self):
        p=self.generate();before=(p.parent/'reference.wav').read_bytes()
        with self.assertRaises(ValueError):register_profile(speaker='SYNTH_A',audio=p.parent/'reference.wav',
            text_path=p.parent/'reference.txt',consent_path=p.parent/'pending_review/consent.template.json',
            review_path=p.parent/'pending_review/review.template.json',output=self.root/'registry')
        self.assertEqual((p.parent/'reference.wav').read_bytes(),before)
        self.assertFalse((self.root/'registry').exists())
    def test_native_design_api_and_wave_validation(self):
        b=QwenDesignSynthesizer.__new__(QwenDesignSynthesizer)
        b.settings={'seed':7,'language':'English','max_tokens':100,'temperature':.9,'wav_subtype':'PCM_16','instruct':None}
        calls=[]
        result=types.SimpleNamespace(audio=np.sin(np.arange(24000)*.1)*.1,sample_rate=24000,token_count=20)
        def native(**kw):calls.append(kw);return iter([result])
        b._model=types.SimpleNamespace(model=types.SimpleNamespace(generate_voice_design=native))
        mx=types.ModuleType('mlx.core');mx.random=types.SimpleNamespace(seed=lambda x:None);mx.get_peak_memory=lambda:0
        mlx=types.ModuleType('mlx');mlx.core=mx
        facade=types.ModuleType('qwen_tts_mlx');facade.MLXQwen3TTS=object
        sf=types.ModuleType('soundfile');sf.write=lambda path,samples,rate,**kw:tone(path,seconds=len(samples)/rate,rate=rate)
        p=VoiceProfile('A',mode='voice_design',design_description='Fictional calm voice.')
        with patch.dict('sys.modules',{'mlx':mlx,'mlx.core':mx,'qwen_tts_mlx':facade,'soundfile':sf}):
            out=self.root/'native.wav';b.synthesize(text='Test.',profile=p,output_path=out)
            self.assertEqual(calls[0]['language'],'English');self.assertEqual(calls[0]['instruct'],p.design_description)
            result.audio=np.array([float('nan')])
            with self.assertRaises(ValueError):b.synthesize(text='Test.',profile=p,output_path=self.root/'bad.wav')
            result.token_count=100
            with self.assertRaises(RuntimeError):b.synthesize(text='Test.',profile=p,output_path=self.root/'limit.wav')
