"""Auditable fictional VoiceDesign references; generation never grants clone approval.

Reuses TRANS's explicit VoiceDesign -> reference -> Base workflow, keeping the
reference generation step separate from human verification and clone registration.
"""
import fcntl
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

from language.cache import atomic_json, fingerprint
from language.timing import measure_wav
from language.voice import VoiceProfile

DEFAULT_DESIGN = 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16'


class QwenDesignSynthesizer:
    supported_voice_modes = {'voice_design'}

    def __init__(self, model_id=DEFAULT_DESIGN, *, seed=0, language='Japanese'):
        from tts_generator import QwenPresetSynthesizer
        from voice_cloning import digest
        from importlib.util import find_spec
        if language not in ('Japanese', 'English'):
            raise ValueError('Reference language must be Japanese or English.')
        base = QwenPresetSynthesizer(model_id, seed=seed, expected_model_type='voice_design')
        self.snapshot, self._model = base.snapshot, None
        self.settings = {**base.settings, 'language': language}
        native = Path(find_spec('mlx_audio').origin).parent / 'tts/models/qwen3_tts/qwen3_tts.py'
        self.identity = {**base.identity, 'voice_mode': 'voice_design', 'settings': self.settings.copy(),
                         'adapter_sha256': digest(__file__), 'native_source_sha256': digest(native),
                         'model_quantization': json.loads((self.snapshot/'config.json').read_text()).get('quantization')}

    def synthesize(self, *, text, profile, output_path):
        import numpy as np
        import mlx.core as mx
        import soundfile as sf
        from qwen_tts_mlx import MLXQwen3TTS
        if profile.mode != 'voice_design' or profile.reference_audio or not text.strip():
            raise ValueError('Explicit voice design and nonempty text required.')
        started = time.perf_counter()
        if self._model is None:
            self._model = MLXQwen3TTS.from_pretrained(str(self.snapshot))
        mx.random.seed(self.settings['seed'])
        options = {k:v for k,v in self.settings.items() if k not in ('seed','wav_subtype','instruct')}
        results = list(self._model.model.generate_voice_design(
            text=text, instruct=profile.design_description, **options))
        if not results or any(r.token_count >= self.settings['max_tokens'] for r in results):
            raise RuntimeError('Empty output or token ceiling; reference not published.')
        rates = {r.sample_rate for r in results}
        if len(rates) != 1 or next(iter(rates)) <= 0:
            raise ValueError('Invalid output sample rates.')
        samples = np.concatenate([np.asarray(r.audio) for r in results])
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
            raise ValueError('Invalid/out-of-range waveform; no clipping or truncation applied.')
        rate = int(next(iter(rates)))
        sf.write(output_path, samples, rate, subtype='PCM_16', format='WAV')
        return {'sample_rate':rate, 'sample_frames':len(samples),
                'processing_seconds':time.perf_counter()-started,
                'token_count':sum(r.token_count for r in results), 'termination':'eos_before_limit',
                'peak_memory_bytes':mx.get_peak_memory()}


def generate_reference(backend, *, speaker, text, instruction, output, fictional=False):
    from voice_cloning import digest, prepare_review_template
    if fictional is not True:
        raise ValueError('Explicit fictional design request required; this does not grant clone permission.')
    if not isinstance(speaker,str) or speaker in ('UNKNOWN','MIXED','REVIEW') or not re.fullmatch(r'[A-Za-z0-9_-]+',speaker):
        raise ValueError('Known safe synthetic speaker label required.')
    if not isinstance(text,str) or not text.strip() or not isinstance(instruction,str) or not instruction.strip():
        raise ValueError('Exact text and voice instruction are required.')
    profile = VoiceProfile(speaker,mode='voice_design',design_description=instruction)
    request = {'format_version':1,'tts':backend.identity,'speaker_id':speaker,'text':text,
               'instruction':instruction,'fictional_design_requested':True}
    key = fingerprint(request)
    root = Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    target = root / key[:20]
    with (root/'.reference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            saved = validate_generation_record(target/'generation.json')
            if saved['request'] != request:
                raise ValueError('Reference cache request mismatch.')
            for name, value in saved['artifacts'].items():
                if digest(target/name) != value:
                    raise ValueError('Reference artifact changed; retained unchanged.')
            return target/'generation.json'
        tmp = Path(tempfile.mkdtemp(prefix='.reference-',dir=root))
        try:
            started = time.perf_counter()
            result = backend.synthesize(text=text,profile=profile,output_path=str(tmp/'reference.wav'))
            measured = measure_wav(tmp/'reference.wav')
            (tmp/'reference.txt').write_text(text,encoding='utf-8')
            reasons = ['authorization_pending','transcript_listening_pending','quality_listening_pending']
            if measured['channels'] != 1 or not 2.5 <= measured['audio_duration'] <= 30:
                reasons.append('reference_format_or_duration_outside_registration_limits')
            if measured['clipped_sample_fraction'] > .001 or measured['rms_amplitude'] < .005:
                reasons.append('reference_volume_requires_review')
            prepare_review_template(tmp/'reference.wav',tmp/'reference.txt',speaker,tmp/'pending_review')
            (tmp/'review.html').write_text(
                '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Reference review</title>'
                '<style>body{font:18px/1.7 system-ui;max-width:850px;margin:40px auto;padding:20px}audio{width:100%}</style>'
                '<h1>架空の参考音声：未承認</h1><audio controls src="reference.wav"></audio>'
                '<h2>生成テキスト（発声との一致は未確認）</h2><p>'+html.escape(text)+'</p>'
                '<h2>声質の指示</h2><p>'+html.escape(instruction)+'</p>'
                '<p>利用権限、発声内容、他声の混入、明瞭さ、語頭・語尾を人が確認してください。'
                '自動測定や再生では承認されません。確認した内容はテンプレートを別名へコピーして記入します。</p></html>',encoding='utf-8')
            record = {'kind':'synthetic_voice_reference','format_version':1,'request':request,'cache_key':key,
                      'audio_file':'reference.wav','text_file':'reference.txt','measurement':measured,
                      'backend_result':result,'wall_seconds':time.perf_counter()-started,
                      'authorized':False,'human_verified':False,'review_required':True,'review_reasons':reasons,
                      'artifacts':{str(p.relative_to(tmp)):digest(p) for p in tmp.rglob('*') if p.is_file()}}
            atomic_json(tmp/'generation.json',record,overwrite=False)
            tmp.rename(target)
        finally:
            if tmp.exists():shutil.rmtree(tmp)
    return target/'generation.json'


def validate_generation_record(path, *, audio=None, text=None, speaker=None):
    from voice_cloning import digest
    path=Path(path).resolve();record=json.loads(path.read_text())
    if record.get('kind')!='synthetic_voice_reference' or record.get('format_version')!=1:
        raise ValueError('Expected synthetic reference generation record.')
    request=record['request']
    if record['cache_key']!=fingerprint(request):raise ValueError('Generation request changed.')
    if record.get('authorized') is not False or record.get('human_verified') is not False:
        raise ValueError('Generation record cannot grant human verification or clone permission.')
    wav=path.parent/'reference.wav';txt=path.parent/'reference.txt'
    if measure_wav(wav)!=record['measurement'] or txt.read_text(encoding='utf-8')!=request['text']:
        raise ValueError('Generated audio/text changed.')
    if audio is not None and digest(audio)!=record['measurement']['sha256']:
        raise ValueError('Registered WAV differs from generation record.')
    if text is not None and text!=request['text']:raise ValueError('Registered text differs from generation record.')
    if speaker is not None and speaker!=request['speaker_id']:raise ValueError('Synthetic speaker mismatch.')
    return record
