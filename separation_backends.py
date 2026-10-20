"""Local-only replaceable separation and ASR backends. No download entry point."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time
from typing import Protocol

import numpy as np
from audio_mixer import sha256_file


def peak_memory_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


@dataclass(frozen=True)
class SeparationSpec:
    model_name: str
    revision: str
    simultaneous_speaker_counts: tuple
    sample_rate: int
    output_channels: int | None
    devices: tuple = ('cpu',)
    required_memory_bytes: int | None = None
    license: str = 'not specified'
    max_window_seconds: float = 8.


class SpeechSeparator(Protocol):
    spec: SeparationSpec
    identity: dict

    def separate(self, samples: np.ndarray, sample_rate: int, count: int) -> list[np.ndarray]: ...


class NoSeparator:
    spec = SeparationSpec('unavailable', 'none', (), 16000, None)
    identity = asdict(spec)

    def separate(self, samples, sample_rate, count):
        raise RuntimeError('No separation model selected.')


class ConvTasNetSeparator:
    """Official local checkpoint, CPU inference; never calls bundle.get_model/download."""
    def __init__(self, weights, *, threads=4):
        self.weights = Path(weights).resolve()
        digest = sha256_file(self.weights)
        self.spec = SeparationSpec('torchaudio/CONVTASNET_BASE_LIBRI2MIX', digest, (2,), 8000, 2,
                                  license='Torchaudio code: BSD-2-Clause; checkpoint-specific grant not separately stated')
        self.identity = {**asdict(self.spec), 'weights_sha256':digest,
                         'torch_version':version('torch'), 'torchaudio_version':version('torchaudio'),
                         'settings':{'device':'cpu','threads':threads,'dtype':'float32'}}
        self.model = None; self.threads = threads

    def separate(self, samples, sample_rate, count):
        import torch
        import torchaudio
        if count != 2: raise ValueError('Conv-TasNet checkpoint supports exactly 2 simultaneous voices.')
        if len(samples)/sample_rate > self.spec.max_window_seconds+.001:
            raise ValueError('Model window exceeds memory-bounded limit.')
        if self.model is None:
            torch.set_num_threads(self.threads)
            self.model = torchaudio.models.conv_tasnet_base(num_sources=2)
            self.model.load_state_dict(torch.load(self.weights, map_location='cpu', weights_only=True))
            self.model.eval()
        x = torch.from_numpy(np.asarray(samples,dtype=np.float32).copy())
        if sample_rate != 8000: x = torchaudio.functional.resample(x, sample_rate, 8000)
        with torch.inference_mode(): y = self.model(x[None,None,:])[0]
        if sample_rate != 8000: y = torchaudio.functional.resample(y, 8000, sample_rate)
        # Resampling rounding is at most a sample; no speech-length correction.
        if abs(y.shape[-1]-len(samples)) > 1: raise ValueError('Unexpected model output duration.')
        if y.shape[-1] > len(samples): y = y[:,:len(samples)]
        if y.shape[-1] < len(samples): y = torch.nn.functional.pad(y,(0,len(samples)-y.shape[-1]))
        return [row.cpu().numpy().copy() for row in y]


class TorchScriptSeparator:
    """Optional local user-supplied backend, [1,1,T] -> [1,stems,T]."""
    def __init__(self, weights, config):
        self.weights = Path(weights).resolve(); self.spec = SeparationSpec(**config)
        if 'cpu' not in self.spec.devices or self.spec.max_window_seconds <= 0:
            raise ValueError('Invalid CPU backend metadata.')
        self.identity = {**asdict(self.spec), 'weights_sha256':sha256_file(self.weights),
                         'torch_version':version('torch'), 'device':'cpu'}
        self.model = None

    def separate(self, samples, sample_rate, count):
        import torch
        if sample_rate != self.spec.sample_rate or count not in self.spec.simultaneous_speaker_counts:
            raise ValueError('Unsupported sample rate or simultaneous speaker count.')
        if len(samples)/sample_rate > self.spec.max_window_seconds+.001: raise ValueError('Window too long.')
        if self.model is None: self.model = torch.jit.load(str(self.weights),map_location='cpu').eval()
        with torch.inference_mode(): y = self.model(torch.from_numpy(samples.copy()).float()[None,None,:])[0]
        return [row.cpu().numpy().copy() for row in y]


class WhisperProcess:
    """Reuse the existing faster-whisper environment in a persistent local worker."""
    def __init__(self, python, model_dir, *, language='en', timeout=180):
        self.python = Path(python).absolute(); self.model_dir = Path(model_dir).resolve()
        if not self.python.is_file() or not (self.model_dir/'model.bin').is_file():
            raise ValueError('Existing ASR Python and complete local model directory required.')
        package_versions=json.loads(subprocess.check_output([str(self.python),'-c',
            'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ("faster-whisper","ctranslate2","numpy")}))'],text=True))
        self.identity = {'versions':package_versions,'backend':'faster-whisper','model_revision':self.model_dir.name,
                         'model_files':{p.name:sha256_file(p) for p in self.model_dir.iterdir() if p.is_file()},
                         'python':str(self.python),'settings':{'language':language,'compute_type':'int8',
                         'word_timestamps':True,'vad_filter':False,'beam_size':3,'condition_on_previous_text':False}}
        self.language=language; self.timeout=timeout; self.process=None

    def transcribe(self,path):
        if self.process is None:
            env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
            self.process=subprocess.Popen([str(self.python),str(Path(__file__).with_name('speaker_transcription.py')),
                                           '--worker','--model-dir',str(self.model_dir),'--language',self.language],
                                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                          text=True,bufsize=1,env=env)
        self.process.stdin.write(json.dumps({'audio_file':str(Path(path).resolve())})+'\n');self.process.stdin.flush()
        if not select.select([self.process.stdout],[],[],self.timeout)[0]:
            self.close(); raise RuntimeError('ASR worker timed out; successful jobs remain cached.')
        line=self.process.stdout.readline()
        if not line: self.close(); raise RuntimeError('ASR worker stopped without a result.')
        result=json.loads(line)
        if result.get('error'): raise RuntimeError('ASR worker: '+result['error'])
        return result

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)
            self.process=None


class NoASR:
    identity={'backend':'unavailable'}
    def transcribe(self,path): return {'segments':[], 'unavailable':True,'elapsed_seconds':0.,'peak_memory_bytes':0}
    def close(self): pass
