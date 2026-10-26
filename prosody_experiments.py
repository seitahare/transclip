"""Cross-lingual Voice Cloning Japanese Prosody Improvement & Comparison Engine.

Implements:
- Experiment A: Baseline ICL Voice Cloning (`x_vector_only_mode=False`)
- Experiment B: Speaker-Embedding-Only (X-vector) Voice Cloning (`x_vector_only_mode=True`, `ref_text=None`)
- Experiment C: Controlled single-variable sampling & reference-boundary experiments (`temperature`, `top_k`, `top_p`, `repetition_penalty`, `seed`, `ref_trailing_silence_seconds`)
- Japanese prosody & rhythm acoustic metrics (F0 semitone range, sentence-final F0 contour delta, frame-to-frame pitch smoothness, mora duration ms, pause structure)
- Side-by-side listening bundle (`output/prosody_experiments/{reference,preset,icl,xvector,comparison.json,comparison.md,review.html}`)
"""
import argparse
import html
import inspect
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import quote
import wave

import numpy as np

DEFAULT_BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
DEFAULT_PRESET_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
DEFAULT_SPEAKER_ID = "SYNTH_EN_01"
EXPECTED_EN_REF_SHA256 = "4a19ef1c00d3ee87d3afbb707f59166f830fe6a99c196a2df7d7fecdd7f73696"

CORE_SAMPLES = [
    {
        "id": "sample_01",
        "dub_id": "dub-001",
        "sentence_type": "declarative",
        "text_en": "New technology is gradually changing the way we live.",
        "text_ja": "新しい技術が、私たちの暮らしを少しずつ変えています。",
        "mora_count": 26,
    },
    {
        "id": "sample_02",
        "dub_id": "dub-002",
        "sentence_type": "numeric_declarative",
        "text_en": "This computer performs 144 trillion calculations per second.",
        "text_ja": "このコンピューターは、1秒間に144兆回の計算を行います。",
        "mora_count": 32,
    },
    {
        "id": "sample_03",
        "dub_id": "dub-003",
        "sentence_type": "interrogative",
        "text_en": "So what should we verify in the next experiment?",
        "text_ja": "それでは、次の実験で何を確かめればよいのでしょうか。",
        "mora_count": 27,
    },
]

EXTENDED_PROSODY_SAMPLES = [
    {
        "id": "sample_04",
        "dub_id": "dub-004",
        "sentence_type": "expressive_contrast",
        "text_en": "Wait, is that really true? That is a remarkable result!",
        "text_ja": "えっ、それは本当ですか？驚くほど素晴らしい結果ですね。",
        "mora_count": 26,
    },
]

BASELINE_CLONE_SETTINGS = {
    "seed": 0,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "ref_trailing_silence_seconds": 0.0,
}


def build_experiment_c_specs(base_method="xvector"):
    """Return single-variable Experiment C configurations (varying one parameter at a time)."""
    if base_method not in ("xvector", "icl"):
        raise ValueError("base_method must be 'xvector' or 'icl'.")
    specs = [
        {
            "mode_name": f"{base_method}_temp07",
            "clone_method": base_method,
            "changed_parameter": "temperature",
            "rationale": "Lower temperature (0.7 vs 0.9) reduces high-entropy pitch/codec jumps and stabilizes Japanese pitch accent.",
            "settings": {**BASELINE_CLONE_SETTINGS, "temperature": 0.7},
        },
        {
            "mode_name": f"{base_method}_topp09",
            "clone_method": base_method,
            "changed_parameter": "top_p",
            "rationale": "Nucleus sampling (top_p=0.9 vs 1.0) prunes low-probability acoustic tokens that cause prosodic wobble.",
            "settings": {**BASELINE_CLONE_SETTINGS, "top_p": 0.9},
        },
        {
            "mode_name": f"{base_method}_topk25",
            "clone_method": base_method,
            "changed_parameter": "top_k",
            "rationale": "Tighter top_k (25 vs 50) restricts first-codebook prosody token candidates to high-confidence Japanese contours.",
            "settings": {**BASELINE_CLONE_SETTINGS, "top_k": 25},
        },
        {
            "mode_name": f"{base_method}_rep102",
            "clone_method": base_method,
            "changed_parameter": "repetition_penalty",
            "rationale": "Gentler repetition_penalty (1.02 vs 1.05) avoids penalizing sustained Japanese long vowels (長音) and steady mora timing.",
            "settings": {**BASELINE_CLONE_SETTINGS, "repetition_penalty": 1.02},
        },
        {
            "mode_name": f"{base_method}_seed42",
            "clone_method": base_method,
            "changed_parameter": "seed",
            "rationale": "Alternative seed (42 vs 0) evaluates stochastic variation in sentence-level intonation.",
            "settings": {**BASELINE_CLONE_SETTINGS, "seed": 42},
        },
    ]
    if base_method == "icl":
        specs.append(
            {
                "mode_name": "icl_pad040",
                "clone_method": "icl",
                "changed_parameter": "ref_trailing_silence_seconds",
                "rationale": "Appends 0.40s of silence to in-memory reference audio to force a prosodic phrase reset before Japanese tokens.",
                "settings": {**BASELINE_CLONE_SETTINGS, "ref_trailing_silence_seconds": 0.40},
            }
        )
    return specs


def _read_wav_float32(path):
    """Read mono PCM16 WAV file into float32 numpy array and sample rate."""
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        frames = stream.getnframes()
        if channels != 1 or width != 2 or frames <= 0 or rate <= 0:
            raise ValueError(f"Expected non-empty mono PCM16 WAV: {path}")
        raw = stream.readframes(frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def _read_pcm_for_dsp(wav_path):
    samples, rate = _read_wav_float32(wav_path)
    return samples.astype(np.float64), rate


def _extract_f0_contour(samples, rate):
    x = np.asarray(samples, dtype=np.float64)
    frame = max(1, round(0.04 * rate))
    hop = max(1, round(0.01 * rate))
    padded = np.pad(x, (0, max(0, frame - len(x))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop].copy()
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    activity = rms > max(0.003, float(np.percentile(rms, 95)) * 0.08)
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2 * frame, axis=1)
    ac = np.fft.irfft(fft * np.conj(fft), n=2 * frame, axis=1)[:, :frame]
    energy = np.cumsum(centered ** 2, axis=1)
    f0 = []
    low, high = int(rate / 450), min(frame - 2, int(rate / 60))
    for i in np.where(activity)[0]:
        lags = np.arange(low, high + 1)
        left = energy[i, frame - lags - 1]
        right = energy[i, -1] - energy[i, lags - 1]
        correlations = ac[i, lags] / np.sqrt(np.maximum(left * right, 1e-20))
        peaks = np.where((correlations[1:-1] > correlations[:-2]) & (correlations[1:-1] >= correlations[2:]))[0] + 1
        if len(peaks) == 0:
            continue
        best = float(np.max(correlations[peaks]))
        eligible = peaks[correlations[peaks] >= max(0.65, 0.9 * best)]
        if len(eligible) == 0:
            continue
        index = int(eligible[0])
        lag = lags[index]
        y0, y1, y2 = correlations[index - 1: index + 2]
        correction = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if abs(y0 - 2 * y1 + y2) > 1e-12 else 0.0
        f0.append(float(rate / (lag + np.clip(correction, -0.5, 0.5))))
    return np.asarray(f0, dtype=np.float64)


def _fallback_measure_wav(wav_path):
    import hashlib

    samples, rate = _read_wav_float32(wav_path)
    duration = len(samples) / float(rate)
    clipped = float(np.mean(np.abs(samples) >= (32767.0 / 32768.0)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(samples)))
    sha = hashlib.sha256(Path(wav_path).read_bytes()).hexdigest()
    return {
        "audio_duration": duration,
        "clipped_sample_fraction": round(clipped, 6),
        "rms_amplitude": rms,
        "peak_amplitude": peak,
        "sha256": sha,
    }


def _fallback_dsp_features(samples, rate, f0_trace):
    frame = max(1, round(0.025 * rate))
    hop = max(1, round(0.010 * rate))
    padded = np.pad(samples, (0, max(0, frame - len(samples))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    thr = max(0.004, float(np.percentile(rms, 95)) * 0.1)
    silent = rms < thr
    pauses = []
    start = None
    for idx, is_sil in enumerate(silent):
        if is_sil and start is None:
            start = idx
        elif not is_sil and start is not None:
            dur = (idx - start) * hop / float(rate)
            if start > 0 and dur >= 0.08:
                pauses.append({"start": round(start * hop / float(rate), 3), "duration": round(dur, 3)})
            start = None
    pause_seconds = sum(p["duration"] for p in pauses)
    return {
        "f0_hz_median": float(np.median(f0_trace)) if len(f0_trace) else None,
        "f0_hz_p10": float(np.percentile(f0_trace, 10)) if len(f0_trace) else None,
        "f0_hz_p90": float(np.percentile(f0_trace, 90)) if len(f0_trace) else None,
        "f0_hz_std": float(np.std(f0_trace)) if len(f0_trace) else None,
        "pause_seconds": pause_seconds,
        "low_energy_pauses": pauses,
        "voiced_fraction": float(len(f0_trace)) / max(1.0, float(len(rms))),
    }


def analyze_japanese_prosody(wav_path, *, mora_count=None):
    """Compute F0 contour, semitone range, sentence-final intonation delta, and mora timing metrics."""
    samples, rate = _read_pcm_for_dsp(wav_path)
    f0_trace = _extract_f0_contour(samples, rate)
    try:
        from language.timing import measure_wav
        from voice_profiler import acoustic_features

        measured = measure_wav(wav_path)
        features = acoustic_features(samples, rate)
        dsp = features["dsp_estimates"]
        flatness = features["measured"].get("spectral_flatness_median")
    except ImportError:
        measured = _fallback_measure_wav(wav_path)
        dsp = _fallback_dsp_features(samples, rate, f0_trace)
        flatness = None

    f0_median = dsp.get("f0_hz_median")
    f0_p10 = dsp.get("f0_hz_p10")
    f0_p90 = dsp.get("f0_hz_p90")
    f0_std = dsp.get("f0_hz_std")

    semitone_range = (
        round(12.0 * math.log2(f0_p90 / f0_p10), 3)
        if (f0_p10 and f0_p90 and f0_p10 > 0 and f0_p90 > 0)
        else None
    )

    sentence_final_delta_hz = None
    sentence_final_delta_semitones = None
    frame_step_semitones_median = None
    if len(f0_trace) >= 8:
        split = max(1, int(len(f0_trace) * 0.75))
        body_med = float(np.median(f0_trace[:split]))
        tail_med = float(np.median(f0_trace[split:]))
        sentence_final_delta_hz = round(tail_med - body_med, 2)
        if body_med > 0 and tail_med > 0:
            sentence_final_delta_semitones = round(12.0 * math.log2(tail_med / body_med), 3)
        steps = np.abs(12.0 * np.log2(f0_trace[1:] / f0_trace[:-1]))
        frame_step_semitones_median = round(float(np.median(steps)), 4)

    duration = measured["audio_duration"]
    pause_seconds = float(dsp.get("pause_seconds") or 0.0)
    spoken_seconds = max(0.05, duration - pause_seconds)
    pauses = dsp.get("low_energy_pauses") or []

    mora_ms = round((spoken_seconds / mora_count) * 1000.0, 2) if (mora_count and mora_count > 0) else None
    mora_per_sec = round(mora_count / spoken_seconds, 3) if (mora_count and mora_count > 0) else None

    return {
        "audio_duration_seconds": round(duration, 4),
        "spoken_duration_seconds": round(spoken_seconds, 4),
        "pause_seconds": round(pause_seconds, 4),
        "pause_count": len(pauses),
        "pauses": pauses,
        "mora_count": mora_count,
        "mean_mora_duration_ms": mora_ms,
        "spoken_mora_per_second": mora_per_sec,
        "f0_hz_median": round(f0_median, 2) if f0_median is not None else None,
        "f0_hz_p10": round(f0_p10, 2) if f0_p10 is not None else None,
        "f0_hz_p90": round(f0_p90, 2) if f0_p90 is not None else None,
        "f0_hz_std": round(f0_std, 2) if f0_std is not None else None,
        "f0_range_semitones_p10_p90": semitone_range,
        "sentence_final_f0_delta_hz": sentence_final_delta_hz,
        "sentence_final_f0_delta_semitones": sentence_final_delta_semitones,
        "frame_to_frame_step_semitones_median": frame_step_semitones_median,
        "voiced_fraction": round(float(dsp.get("voiced_fraction") or 0.0), 4),
        "clipped_sample_fraction": measured["clipped_sample_fraction"],
        "rms_amplitude": round(measured["rms_amplitude"], 5),
        "peak_amplitude": round(measured["peak_amplitude"], 5),
        "spectral_flatness_median": flatness,
        "sha256": measured["sha256"],
    }


class QwenProsodyCloneSynthesizer:
    """Selectable ICL vs X-vector-only Voice Clone backend with sampling parameter controls.

    Preserves full compatibility with `tts_feedback.run_feedback(..., backend=..., voices=..., voice_profiles=...)`
    and `voice_cloning.py` while isolating caches by (`clone_method`, `x_vector_only_mode`, `settings`).
    """
    supported_voice_modes = {"approved_clone"}

    def __init__(
        self,
        model_id=DEFAULT_BASE_MODEL,
        *,
        clone_method="xvector",
        seed=0,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
        ref_trailing_silence_seconds=0.0,
    ):
        from importlib.util import find_spec
        from tts_generator import QwenPresetSynthesizer
        from voice_cloning import digest

        if clone_method not in ("icl", "xvector"):
            raise ValueError("clone_method must be 'icl' or 'xvector'.")
        for name, val, lo, hi in [
            ("temperature", temperature, 0.05, 2.0),
            ("top_p", top_p, 0.1, 1.0),
            ("repetition_penalty", repetition_penalty, 0.8, 2.0),
            ("ref_trailing_silence_seconds", ref_trailing_silence_seconds, 0.0, 2.0),
        ]:
            if type(val) not in (int, float) or not math.isfinite(val) or not (lo <= val <= hi):
                raise ValueError(f"Invalid {name}: {val}")
        if type(top_k) is not int or not (1 <= top_k <= 500):
            raise ValueError(f"Invalid top_k: {top_k}")

        base = QwenPresetSynthesizer(model_id, seed=seed, expected_model_type="base")
        self.snapshot = base.snapshot
        self.clone_method = clone_method
        self.x_vector_only_mode = clone_method == "xvector"
        self.ref_trailing_silence_seconds = float(ref_trailing_silence_seconds)
        self.settings = {
            **base.settings,
            "seed": seed,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
        }
        self._model = None
        self._conditioning = {}
        package = Path(find_spec("mlx_audio").origin).parent
        self.identity = {
            **base.identity,
            "adapter_version": 2,
            "voice_mode": "approved_clone",
            "clone_method": self.clone_method,
            "x_vector_only_mode": self.x_vector_only_mode,
            "ref_trailing_silence_seconds": self.ref_trailing_silence_seconds,
            "settings": dict(self.settings),
            "model_quantization": json.loads((self.snapshot / "config.json").read_text()).get("quantization"),
            "conditioning_cache": (
                "x_vector_only speaker embedding conditioning (no ICL ref_code/ref_text)"
                if self.x_vector_only_mode
                else "in-process profile-isolated native ICL cache"
            ),
            "adapter_sha256": digest(__file__),
            "voice_contract_sha256": digest(Path(__file__).parent / "language/voice.py"),
            "native_source_sha256": digest(package / "tts/models/qwen3_tts/qwen3_tts.py"),
        }

    def synthesize(self, *, text, profile, output_path, seed_override=None):
        import mlx.core as mx
        import soundfile as sf
        from qwen_tts_mlx import MLXQwen3TTS
        from voice_cloning import digest, profile_snapshot

        def _snap(prof):
            if hasattr(prof, "technical_snapshot") and callable(prof.technical_snapshot):
                return prof.technical_snapshot()
            return profile_snapshot(prof)

        started = time.perf_counter()
        snap = _snap(profile)
        voice_prof_id = getattr(profile, "voice_profile_id", None) or snap["profile_fingerprint"]
        cond_key = f"{snap['profile_fingerprint']}:{self.clone_method}:{self.ref_trailing_silence_seconds:.3f}"
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Japanese text required.")
        if self._model is None:
            self._model = MLXQwen3TTS.from_pretrained(str(self.snapshot))

        reused = cond_key in self._conditioning
        if not reused:
            wav, rate = sf.read(profile.reference_audio, dtype="float32")
            if digest(profile.reference_audio) != snap["reference_audio_sha256"]:
                raise ValueError("Reference changed during load.")
            if self.ref_trailing_silence_seconds > 0 and not self.x_vector_only_mode:
                pad_frames = round(self.ref_trailing_silence_seconds * rate)
                wav = np.concatenate([wav, np.zeros(pad_frames, dtype=np.float32)])
            prompt_ref_text = None if self.x_vector_only_mode else getattr(profile, "reference_text", None)
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=(wav, rate),
                ref_text=prompt_ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
            self._conditioning[cond_key] = {"prompt": prompt, "icl_cache": {}}

        condition = self._conditioning[cond_key]
        native = self._model.model
        if not hasattr(native, "_icl_cache"):
            raise RuntimeError("Installed mlx-audio lacks the inspected ICL cache contract.")
        native._icl_cache = condition["icl_cache"]
        active_seed = int(self.settings["seed"] if seed_override is None else seed_override)
        mx.random.seed(active_seed)
        options = {
            k: v
            for k, v in self.settings.items()
            if k not in ("seed", "wav_subtype", "language", "instruct")
        }
        prompt = condition["prompt"]
        # CRITICAL: In xvector mode, pass ref_text=None to native.generate so mlx_audio
        # executes _prepare_generation_inputs (speaker embedding only) rather than _prepare_icl_generation_inputs!
        effective_ref_text = None if self.x_vector_only_mode else prompt.ref_text
        try:
            results = list(
                native.generate(
                    text=text,
                    lang_code="Japanese",
                    ref_audio=prompt.ref_audio,
                    ref_text=effective_ref_text,
                    **options,
                )
            )
        finally:
            if self.x_vector_only_mode and native._icl_cache:
                native._icl_cache = {}
                raise RuntimeError("X-vector mode unexpectedly populated _icl_cache; ICL path leaked.")
            condition["icl_cache"] = native._icl_cache
            native._icl_cache = {}

        if not results or any(r.token_count >= self.settings["max_tokens"] for r in results):
            raise RuntimeError("No audio or token limit reached; incomplete speech rejected.")
        rates = {int(r.sample_rate) for r in results}
        if len(rates) != 1 or next(iter(rates)) <= 0:
            raise ValueError("Invalid output sample rate.")
        samples = np.concatenate([np.asarray(r.audio) for r in results])
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Invalid generated waveform.")
        if np.max(np.abs(samples)) > 1.0:
            raise ValueError("Generated amplitude outside PCM range; review required, no silent clipping.")
        if _snap(profile) != snap:
            raise ValueError("Profile/authorization changed during synthesis.")
        rate = next(iter(rates))
        sf.write(output_path, samples, rate, subtype="PCM_16", format="WAV")
        return {
            "sample_rate": rate,
            "sample_frames": len(samples),
            "token_count": sum(r.token_count for r in results),
            "termination": "eos_before_limit",
            "processing_seconds": time.perf_counter() - started,
            "conditioning_reused": reused,
            "clone_method": self.clone_method,
            "x_vector_only_mode": self.x_vector_only_mode,
            "ref_trailing_silence_seconds": self.ref_trailing_silence_seconds,
            "voice_profile_id": voice_prof_id,
            "profile_fingerprint": snap["profile_fingerprint"],
            "reference_audio_sha256": snap["reference_audio_sha256"],
            "reference_text_sha256": None if self.x_vector_only_mode else snap["reference_text_sha256"],
            "model_id": self.identity.get("model"),
            "model_revision": self.identity.get("model_revision"),
            "generation_settings": {**self.settings, "seed": active_seed},
            "peak_memory_bytes": mx.get_peak_memory(),
            "pronunciation_verified": False,
            "voice_similarity_verified": False,
            "prosody_naturalness_verified": False,
            "review_required": True,
        }


def _atomic_write_json(path, data):
    try:
        from language.cache import atomic_json

        atomic_json(path, data)
    except ImportError:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)


def _file_sha256(path):
    try:
        from voice_cloning import digest

        return digest(path)
    except ImportError:
        import hashlib

        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cosine_sim(a, b):
    try:
        from voice_profiler import cosine

        return cosine(a, b)
    except ImportError:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0


class _EmbedderAdapter:
    """Normalizes existing repository embedding backends/callables into a unified interface."""

    def __init__(self, target):
        self.target = target

    def embed(self, samples, rate=24000, *, wav_path=None):
        arr = np.asarray(samples, dtype=np.float32)
        target = self.target
        if hasattr(target, "embed_wav") and wav_path is not None:
            return np.asarray(target.embed_wav(wav_path), dtype=np.float64)
        if hasattr(target, "embed_file") and wav_path is not None:
            return np.asarray(target.embed_file(wav_path), dtype=np.float64)

        fn = None
        for attr in ("embed", "extract_embedding", "embed_waveform", "embed_audio", "embedding"):
            if hasattr(target, attr) and callable(getattr(target, attr)):
                fn = getattr(target, attr)
                break
        if fn is None and callable(target):
            fn = target
        if fn is None:
            raise TypeError(f"Unsupported embedder object: {type(target)!r}")

        target_sr = getattr(target, "sample_rate", None)
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
        except (TypeError, ValueError):
            params = []

        if not params or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            try:
                return np.asarray(fn(arr, int(rate)), dtype=np.float64)
            except TypeError:
                return np.asarray(fn(arr), dtype=np.float64)

        pos_params = [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        kw_names = {p.name for p in params if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}

        if len(pos_params) >= 2:
            return np.asarray(fn(arr, int(rate)), dtype=np.float64)

        for sr_name in ("sample_rate", "rate", "sr", "sampling_rate"):
            if sr_name in kw_names and (len(pos_params) == 1 and pos_params[0].name != sr_name):
                return np.asarray(fn(arr, **{sr_name: int(rate)}), dtype=np.float64)

        if len(pos_params) == 1:
            pname = pos_params[0].name.lower()
            if pname in ("path", "wav_path", "audio_path", "file", "file_path") and wav_path is not None:
                return np.asarray(fn(Path(wav_path)), dtype=np.float64)
            if isinstance(target_sr, int) and target_sr > 0 and target_sr != int(rate) and len(arr) > 1:
                new_len = max(1, int(round(len(arr) * float(target_sr) / float(rate))))
                x_old = np.linspace(0.0, 1.0, len(arr))
                x_new = np.linspace(0.0, 1.0, new_len)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
            return np.asarray(fn(arr), dtype=np.float64)

        return np.asarray(fn(arr), dtype=np.float64)


def _can_instantiate_zero_args(cls):
    try:
        sig = inspect.signature(cls.__init__)
        for name, param in list(sig.parameters.items())[1:]:
            if (
                param.default is inspect.Parameter.empty
                and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ):
                return False
        return True
    except (TypeError, ValueError):
        return True


def resolve_embedder(embedder=None, *, embeddings=False):
    """Connect to the repository's existing embedding implementation in voice_quality / voice_profiler."""
    if embedder is not None:
        return embedder if isinstance(embedder, _EmbedderAdapter) else _EmbedderAdapter(embedder)
    if not embeddings:
        return None

    candidate_modules = []
    for mod_name in ("voice_quality", "cross_lingual_clone", "voice_profiler", "speaker_attribution"):
        try:
            candidate_modules.append(__import__(mod_name))
        except ImportError:
            continue

    factory_names = (
        "default_embedder",
        "load_embedder",
        "create_embedder",
        "build_embedder",
        "get_embedder",
        "pyannote_embedder",
        "_load_embedder",
        "_default_embedder",
        "default_backend",
        "load_backend",
    )
    class_names = (
        "PyannoteBackend",
        "PyannoteDiarizationBackend",
        "PyannoteEmbeddingBackend",
        "PyannoteEmbeddingAdapter",
        "PyannoteAdapter",
        "SpeakerEmbeddingAdapter",
        "EmbeddingBackend",
        "SpeakerEmbedder",
    )
    direct_fn_names = (
        "embed_wav",
        "embed_audio",
        "extract_embedding",
        "compute_embedding",
        "waveform_embedding",
    )

    for mod in candidate_modules:
        for name in factory_names:
            fn = getattr(mod, name, None)
            if inspect.isfunction(fn):
                try:
                    sig = inspect.signature(fn)
                    req = [
                        p
                        for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    ]
                    if not req:
                        inst = fn()
                        if inst is not None:
                            return _EmbedderAdapter(inst)
                except (TypeError, ValueError):
                    continue
        for name in class_names:
            cls = getattr(mod, name, None)
            if inspect.isclass(cls) and _can_instantiate_zero_args(cls):
                return _EmbedderAdapter(cls())
        for name in direct_fn_names:
            fn = getattr(mod, name, None)
            if inspect.isfunction(fn):
                return _EmbedderAdapter(fn)

    for mod in candidate_modules:
        for _, obj in inspect.getmembers(mod):
            if (
                inspect.isclass(obj)
                and obj is not _EmbedderAdapter
                and any(hasattr(obj, m) for m in ("embed", "extract_embedding", "embed_wav", "embedding"))
                and _can_instantiate_zero_args(obj)
            ):
                return _EmbedderAdapter(obj())

    raise RuntimeError(
        "Could not resolve an existing speaker embedding implementation from voice_quality or voice_profiler."
    )


def build_prosody_comparison_bundle(
    *,
    reference_wav,
    speaker_id,
    mode_feedbacks,
    output_dir,
    embedder=None,
    samples_spec=None,
):
    """Build `output/prosody_experiments/` with reference/, preset/, icl/, xvector/, comparison.json, comparison.md, review.html.

    Never creates empty WAVs or fabricated metrics for unexecuted modes.
    """
    samples_spec = samples_spec or CORE_SAMPLES
    dub_to_sample = {s["dub_id"]: s for s in samples_spec}
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ref_path = Path(reference_wav).resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference WAV not found: {ref_path}")
    ref_dir = root / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_dest = ref_dir / "english_reference.wav"
    shutil.copyfile(ref_path, ref_dest)
    ref_prosody = analyze_japanese_prosody(ref_dest, mora_count=None)

    active_embedder = resolve_embedder(embedder, embeddings=embedder is not None)
    vectors = {}
    if active_embedder is not None:
        ref_samples, ref_rate = _read_pcm_for_dsp(ref_dest)
        vectors[str(ref_dest)] = active_embedder.embed(ref_samples.astype(np.float32), ref_rate, wav_path=ref_dest)

    mode_summaries = {}
    sample_comparisons = {s["id"]: {"sample": s, "modes": {}} for s in samples_spec}

    for mode_name, fb_path in mode_feedbacks.items():
        if fb_path is None or not Path(fb_path).is_file():
            continue
        fb_doc = json.loads(Path(fb_path).read_text(encoding="utf-8"))
        entries = {
            r["id"]: r
            for r in fb_doc.get("entries", [])
            if r.get("generation_status") == "complete" and r.get("audio_file")
        }
        if not entries:
            continue
        mode_dir = root / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_wavs = []
        for dub_id, sample_meta in dub_to_sample.items():
            if dub_id not in entries:
                continue
            row = entries[dub_id]
            src_wav = Path(row["audio_file"]).resolve()
            if not src_wav.is_file():
                raise FileNotFoundError(f"Missing generated WAV for {mode_name}/{dub_id}: {src_wav}")
            dst_wav = mode_dir / f"{sample_meta['id']}.wav"
            shutil.copyfile(src_wav, dst_wav)
            if _file_sha256(dst_wav) != _file_sha256(src_wav):
                raise ValueError(f"WAV copy hash mismatch for {dst_wav}")

            mora = row.get("mora_count") or sample_meta.get("mora_count")
            prosody = analyze_japanese_prosody(dst_wav, mora_count=mora)
            ref_cos = None
            if active_embedder is not None:
                s_arr, s_rate = _read_pcm_for_dsp(dst_wav)
                vectors[str(dst_wav)] = active_embedder.embed(s_arr.astype(np.float32), s_rate, wav_path=dst_wav)
                if vectors.get(str(ref_dest)) is not None and vectors.get(str(dst_wav)) is not None:
                    ref_cos = round(_cosine_sim(vectors[str(ref_dest)], vectors[str(dst_wav)]), 4)

            backend_res = row.get("backend_result") or {}
            tts_info = row.get("tts") or fb_doc.get("tts") or {}
            sample_comparisons[sample_meta["id"]]["modes"][mode_name] = {
                "audio_relpath": f"{mode_name}/{sample_meta['id']}.wav",
                "audio_abspath": str(dst_wav),
                "model": tts_info.get("model"),
                "model_revision": tts_info.get("model_revision"),
                "voice": row.get("voice"),
                "clone_method": backend_res.get("clone_method") or tts_info.get("clone_method") or ("preset" if mode_name == "preset" else "icl"),
                "x_vector_only_mode": backend_res.get("x_vector_only_mode", False),
                "profile_fingerprint": backend_res.get("profile_fingerprint"),
                "settings": tts_info.get("settings", {}),
                "processing_seconds": backend_res.get("processing_seconds"),
                "conditioning_reused": backend_res.get("conditioning_reused"),
                "cache_reused": bool(row.get("cache_reused")),
                "prosody_metrics": prosody,
                "reference_embedding_cosine": ref_cos,
                "human_prosody_evaluation": {
                    "pitch_accent_natural": None,
                    "sentence_end_intonation_natural": None,
                    "rhythm_and_mora_timing_natural": None,
                    "foreign_stress_accent_absent": None,
                    "perceptual_timbre_similarity": None,
                },
            }
            mode_wavs.append((sample_meta["id"], str(dst_wav), prosody))

        pairwise_cosines = []
        if active_embedder is not None:
            for (id_a, path_a, _), (id_b, path_b, _) in itertools.combinations(mode_wavs, 2):
                if vectors.get(path_a) is not None and vectors.get(path_b) is not None:
                    pairwise_cosines.append({
                        "pair": [id_a, id_b],
                        "cosine": round(_cosine_sim(vectors[path_a], vectors[path_b]), 4),
                    })
        f0_medians = [p["f0_hz_median"] for _, _, p in mode_wavs if p["f0_hz_median"] is not None]
        mode_summaries[mode_name] = {
            "sample_count": len(mode_wavs),
            "f0_median_range_across_samples_hz": (
                round(max(f0_medians) - min(f0_medians), 2) if len(f0_medians) >= 2 else 0.0
            ),
            "within_mode_embedding_cosines": pairwise_cosines,
        }

    comparison = {
        "kind": "transclip_prosody_experiment_comparison",
        "format_version": 1,
        "speaker_id": speaker_id,
        "reference": {
            "audio_relpath": "reference/english_reference.wav",
            "audio_abspath": str(ref_dest),
            "metrics": ref_prosody,
        },
        "available_modes": list(mode_summaries.keys()),
        "mode_summaries": mode_summaries,
        "samples": sample_comparisons,
        "review_required": True,
        "notes": [
            "ICL mode conditions on both reference codec tokens (ref_code) and reference transcript (ref_text), which can transfer English stress-timed prosody into Japanese.",
            "X-vector mode (x_vector_only_mode=True, ref_text=None) conditions only on the 2048-d ECAPA-TDNN speaker embedding, allowing the model's Japanese language prior to drive pitch accent and mora rhythm.",
            "Acoustic metrics (F0 range, semitone step, embedding cosine) do not prove natural Japanese pitch accent; human listening review via review.html is mandatory.",
        ],
    }

    _atomic_write_json(root / "comparison.json", comparison)
    (root / "comparison.md").write_text(_format_prosody_markdown(comparison), encoding="utf-8")
    (root / "review.html").write_text(_format_prosody_html(comparison), encoding="utf-8")
    return root / "comparison.json", comparison


def _format_prosody_markdown(comparison):
    ref_m = comparison["reference"]["metrics"]
    lines = [
        "# TransClip Cross-lingual Voice Cloning 日本語韻律（Prosody）比較レポート",
        "",
        f"- **対象話者ID**: `{comparison['speaker_id']}`",
        f"- **英語参考音声**: `{comparison['reference']['audio_relpath']}` (長さ {ref_m['audio_duration_seconds']:.2f}s / F0中央値 {ref_m['f0_hz_median']} Hz / SHA-256 `{ref_m['sha256'][:16]}...`)",
        f"- **比較済み方式**: {', '.join(comparison['available_modes'])}",
        "",
        "## 1. 文章・方式別 音響・韻律測定結果",
        "",
        "| Sample | 方式 | 生成方式/設定 | WAV長(s) | 生成時間(s) | 話速(mora/s) | 1モーラ平均(ms) | ポーズ数/秒 | F0中央値(Hz) | F0範囲(半音) | 文末F0変化(半音) | 参考Embedding類似度 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for sample_id, block in comparison["samples"].items():
        s = block["sample"]
        for mode_name, m in block["modes"].items():
            pm = m["prosody_metrics"]
            settings = m.get("settings") or {}
            setting_str = f"{m['clone_method']} (seed={settings.get('seed', 0)}, T={settings.get('temperature', 0.9)}, top_p={settings.get('top_p', 1.0)})"
            proc = f"{m['processing_seconds']:.2f}" if isinstance(m.get("processing_seconds"), (int, float)) else "—"
            cos_s = f"{m['reference_embedding_cosine']:.3f}" if m.get("reference_embedding_cosine") is not None else "未計測"
            lines.append(
                f"| {sample_id} ({s['sentence_type']}) | `{mode_name}` | {setting_str} | "
                f"{pm['audio_duration_seconds']:.3f} | {proc} | {pm['spoken_mora_per_second']} | {pm['mean_mora_duration_ms']} | "
                f"{pm['pause_count']}回 / {pm['pause_seconds']:.2f}s | {pm['f0_hz_median']} | "
                f"{pm['f0_range_semitones_p10_p90']} | {pm['sentence_final_f0_delta_semitones']} | {cos_s} |"
            )
    lines += [
        "",
        "## 2. 方式別の文章間一貫性",
        "",
        "| 方式 | サンプル数 | サンプル間F0中央値差(Hz) | ペアワイズEmbedding類似度 |",
        "|---|---:|---:|---|",
    ]
    for mode_name, ms in comparison["mode_summaries"].items():
        pairs = ", ".join(f"{p['pair'][0]}-{p['pair'][1]}: {p['cosine']:.3f}" for p in ms["within_mode_embedding_cosines"]) or "未計測"
        lines.append(f"| `{mode_name}` | {ms['sample_count']} | {ms['f0_median_range_across_samples_hz']} | {pairs} |")
    lines.append("")
    return "\n".join(lines)


def _format_prosody_html(comparison):
    esc = html.escape
    ref = comparison["reference"]
    ref_m = ref["metrics"]
    cards = []
    for sample_id, block in comparison["samples"].items():
        s = block["sample"]
        rows_html = []
        players_html = []
        for mode_name, m in block["modes"].items():
            pm = m["prosody_metrics"]
            st = m.get("settings") or {}
            cos_s = f"{m['reference_embedding_cosine']:.3f}" if m.get("reference_embedding_cosine") is not None else "未計測"
            proc_s = f"{m['processing_seconds']:.2f}s" if isinstance(m.get("processing_seconds"), (int, float)) else "—"
            rows_html.append(f"""
              <tr>
                <td><strong>{esc(mode_name)}</strong></td>
                <td><code>{esc(str(m.get('model') or '—'))}</code><br><small>mode={esc(str(m['clone_method']))}, xvec_only={esc(str(m['x_vector_only_mode']))}, seed={esc(str(st.get('seed', 0)))}, T={esc(str(st.get('temperature', 0.9)))}, top_p={esc(str(st.get('top_p', 1.0)))}</small></td>
                <td>{pm['audio_duration_seconds']:.3f}s<br><small>(生成 {proc_s})</small></td>
                <td>{pm['spoken_mora_per_second']} mora/s<br><small>({pm['mean_mora_duration_ms']} ms/mora)</small></td>
                <td>{pm['f0_hz_median']} Hz<br><small>範囲 {pm['f0_range_semitones_p10_p90']} ST / 文末Δ {pm['sentence_final_f0_delta_semitones']} ST</small></td>
                <td>{pm['pause_count']}回 ({pm['pause_seconds']:.2f}s)</td>
                <td>{cos_s}</td>
              </tr>
            """)
            players_html.append(f"""
              <div class="player-row">
                <div><strong>[{esc(mode_name)}]</strong> <code>afplay '{esc(m['audio_abspath'])}'</code></div>
                <audio controls preload="metadata" src="{esc(quote(m['audio_relpath']))}"></audio>
              </div>
            """)
        cards.append(f"""
        <div class="card">
          <h2>{esc(sample_id)} ({esc(s['sentence_type'])})</h2>
          <p class="ja-text">{esc(s['text_ja'])}</p>
          <table>
            <thead>
              <tr>
                <th>方式</th>
                <th>モデル / 生成方式 / 設定</th>
                <th>WAV長 (生成時間)</th>
                <th>モーラ速度</th>
                <th>F0中央値 / 抑揚</th>
                <th>ポーズ</th>
                <th>参考声質Cosine</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
          {''.join(players_html)}
        </div>
        """)

    return f"""<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<title>TransClip Prosody Comparison — Preset vs ICL vs X-vector</title>
<style>
  body {{ font: 15px/1.65 system-ui, -apple-system, sans-serif; max-width: 1040px; margin: 28px auto; padding: 0 20px; color: #1f2937; background: #f9fafb; }}
  .card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 18px 20px; margin-bottom: 18px; }}
  .ja-text {{ font-size: 17px; font-weight: 600; background: #f3f4f6; padding: 10px 14px; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13.5px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f3f4f6; }}
  .player-row {{ margin: 10px 0; padding: 8px 12px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; }}
  audio {{ width: 100%; margin-top: 6px; }}
  code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 12.5px; }}
</style>
<h1>TransClip 日本語韻律（Prosody）改善 — 方式別・試聴比較シート</h1>
<div class="card">
  <h2>英語参考音声 (Reference Audio — <code>{esc(comparison['speaker_id'])}</code>)</h2>
  <p>長さ: {ref_m['audio_duration_seconds']:.2f}s | F0中央値: {ref_m['f0_hz_median']} Hz (範囲 {ref_m['f0_range_semitones_p10_p90']} 半音) | SHA-256: <code>{esc(ref_m['sha256'])}</code></p>
  <p>再生コマンド: <code>afplay '{esc(ref['audio_abspath'])}'</code></p>
  <audio controls preload="metadata" src="{esc(quote(ref['audio_relpath']))}"></audio>
</div>
{''.join(cards)}
</html>
"""


def _extract_path_from_result(result, fallback_path):
    if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], (str, Path)):
        return Path(result[0])
    if isinstance(result, (str, Path)):
        return Path(result)
    return Path(fallback_path)


def _invoke_benchmark_script_builder(
    build_fn,
    *,
    output_path,
    speaker,
    voice_mode,
    preset_voice,
    voice=None,
    profile=None,
    profile_fingerprint=None,
    voice_profiles=None,
):
    """Call `cross_lingual_clone.build_phase8_benchmark_script` matching its exact signature."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(build_fn)
    pos_args = []
    kwargs = {}
    for name, param in sig.parameters.items():
        matched = True
        val = None
        if name in ("output_path", "script_path", "path", "output"):
            val = out_p
        elif name in ("speaker", "speaker_id"):
            val = speaker
        elif name in ("voice_mode", "mode"):
            val = voice_mode
        elif name == "preset_voice":
            val = preset_voice
        elif name == "voice" and voice is not None:
            val = voice
        elif name in ("profile", "voice_profile") and profile is not None:
            val = profile
        elif name == "profile_fingerprint" and profile_fingerprint is not None:
            val = profile_fingerprint
        elif name in ("voice_profiles", "profiles") and voice_profiles is not None:
            val = voice_profiles
        else:
            matched = False
        if matched:
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_args.append(val)
            else:
                kwargs[name] = val
    res = build_fn(*pos_args, **kwargs)
    script_path = _extract_path_from_result(res, out_p)
    if isinstance(res, dict) and not script_path.is_file():
        _atomic_write_json(script_path, res)
    return script_path


def _invoke_register_profile(
    register_fn,
    *,
    speaker,
    reference_audio,
    reference_text_path,
    consent_path,
    review_path,
    output_dir,
    preset_voice,
    generation_record_path=None,
):
    """Call `voice_cloning.register_profile` matching its exact signature."""
    sig = inspect.signature(register_fn)
    co_consts = getattr(getattr(register_fn, "__code__", None), "co_consts", ()) or ()
    pos_args = []
    kwargs = {}
    for name, param in sig.parameters.items():
        matched = True
        val = None
        if name in ("speaker", "speaker_id"):
            val = speaker
        elif name in ("reference_audio", "reference_audio_path", "audio_path", "audio"):
            val = Path(reference_audio)
        elif name in ("reference_text_path", "text_path"):
            val = Path(reference_text_path)
        elif name == "reference_text":
            co_names = getattr(getattr(register_fn, "__code__", None), "co_names", ()) or ()
            val = (
                Path(reference_text_path)
                if "read_text" in co_names
                else Path(reference_text_path).read_text(encoding="utf-8").strip()
            )
        elif name in ("consent_path", "consent"):
            val = Path(consent_path)
        elif name in ("review_path", "review"):
            val = Path(review_path)
        elif name in ("output_dir",):
            val = Path(output_dir)
        elif name == "output":
            val = Path(output_dir)
        elif name in ("output_path", "registry_path", "profiles_path", "path"):
            val = Path(output_dir) if "voice_profiles.json" in co_consts else (Path(output_dir) / "voice_profiles.json")
        elif name == "preset_voice":
            val = preset_voice
        elif name in ("generation_record_path", "generation_record") and generation_record_path is not None:
            val = Path(generation_record_path)
        else:
            matched = False
        if matched:
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                pos_args.append(val)
            else:
                kwargs[name] = val
    res = register_fn(*pos_args, **kwargs)
    return _extract_path_from_result(res, Path(output_dir) / "voice_profiles.json")


def run_prosody_experiments(
    *,
    speaker_id=DEFAULT_SPEAKER_ID,
    reference_audio,
    reference_text,
    consent_path,
    review_path,
    output_dir,
    preset_voice="Ryan",
    preset_model=DEFAULT_PRESET_MODEL,
    clone_model=DEFAULT_BASE_MODEL,
    seed=0,
    include_experiment_c=False,
    embeddings=False,
    embedder=None,
    generation_record_path=None,
    expected_sha256=None,
    existing_preset_feedback=None,
    existing_icl_feedback=None,
):
    """Execute Preset, Experiment A (ICL), Experiment B (X-vector-only), and optional Experiment C sweeps."""
    from cross_lingual_clone import (
        CloneBlockedForReviewError,
        audit_english_reference,
        build_phase8_benchmark_script,
    )
    from tts_feedback import run_feedback
    from tts_generator import QwenPresetSynthesizer
    from voice_cloning import load_profiles, profile_snapshot, register_profile

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    work_dir = root / "_runs"
    work_dir.mkdir(parents=True, exist_ok=True)

    ref_audio_path = Path(reference_audio).resolve()
    ref_text_candidate = Path(reference_text)
    if ref_text_candidate.is_file():
        ref_text_path = ref_text_candidate.resolve()
    else:
        ref_text_path = work_dir / "reference.txt"
        ref_text_path.write_text(str(reference_text).strip() + "\n", encoding="utf-8")

    audit_sig = inspect.signature(audit_english_reference)
    audit_kwargs = {
        "speaker": speaker_id,
        "reference_audio": ref_audio_path,
        "reference_text_path": ref_text_path,
        "consent_path": Path(consent_path).resolve(),
        "review_path": Path(review_path).resolve(),
        "output_dir": work_dir / "reference_audit",
    }
    if generation_record_path is not None and "generation_record_path" in audit_sig.parameters:
        audit_kwargs["generation_record_path"] = Path(generation_record_path).resolve()
    if expected_sha256 is not None and "expected_sha256" in audit_sig.parameters:
        audit_kwargs["expected_sha256"] = expected_sha256

    _, audit_report = audit_english_reference(**audit_kwargs)
    if not audit_report.get("ready_for_clone_synthesis"):
        raise CloneBlockedForReviewError(audit_report)

    reg_dir = work_dir / "registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    registry_path = _invoke_register_profile(
        register_profile,
        speaker=speaker_id,
        reference_audio=ref_audio_path,
        reference_text_path=ref_text_path,
        consent_path=Path(consent_path).resolve(),
        review_path=Path(review_path).resolve(),
        output_dir=reg_dir,
        preset_voice=preset_voice,
        generation_record_path=generation_record_path,
    )
    voice_profiles = load_profiles(registry_path)
    if speaker_id not in voice_profiles:
        raise KeyError(f"Registered voice profiles missing speaker {speaker_id!r}")
    speaker_profile = voice_profiles[speaker_id]
    snap = profile_snapshot(speaker_profile)
    clone_voice_id = f"approved_clone:{snap['profile_fingerprint']}"
    preset_voices = {speaker_id: preset_voice}
    clone_voices = {speaker_id: clone_voice_id}

    preset_script_path = _invoke_benchmark_script_builder(
        build_phase8_benchmark_script,
        output_path=work_dir / "scripts" / "preset_script.json",
        speaker=speaker_id,
        voice_mode="preset",
        preset_voice=preset_voice,
        voice=preset_voice,
    )
    clone_script_path = _invoke_benchmark_script_builder(
        build_phase8_benchmark_script,
        output_path=work_dir / "scripts" / "clone_script.json",
        speaker=speaker_id,
        voice_mode="approved_clone",
        preset_voice=preset_voice,
        voice=clone_voice_id,
        profile=speaker_profile,
        profile_fingerprint=snap["profile_fingerprint"],
        voice_profiles=voice_profiles,
    )

    mode_feedbacks = {}

    # 1. Preset (reuse existing feedback if provided, else run/reuse cache)
    if existing_preset_feedback and Path(existing_preset_feedback).is_file():
        mode_feedbacks["preset"] = Path(existing_preset_feedback).resolve()
    else:
        preset_synth = QwenPresetSynthesizer(preset_model, seed=seed)
        preset_run_dir = work_dir / "preset_run"
        preset_res = run_feedback(
            preset_script_path,
            preset_run_dir,
            backend=preset_synth,
            voices=preset_voices,
        )
        mode_feedbacks["preset"] = _extract_path_from_result(preset_res, preset_run_dir / "feedback.json")

    # 2. Experiment A: ICL Clone Baseline (x_vector_only_mode=False)
    if existing_icl_feedback and Path(existing_icl_feedback).is_file():
        mode_feedbacks["icl"] = Path(existing_icl_feedback).resolve()
    else:
        icl_synth = QwenProsodyCloneSynthesizer(
            clone_model,
            clone_method="icl",
            seed=seed,
        )
        icl_run_dir = work_dir / "icl_run"
        icl_res = run_feedback(
            clone_script_path,
            icl_run_dir,
            backend=icl_synth,
            voices=clone_voices,
            voice_profiles=voice_profiles,
        )
        mode_feedbacks["icl"] = _extract_path_from_result(icl_res, icl_run_dir / "feedback.json")

    # 3. Experiment B: X-vector-only Clone (x_vector_only_mode=True, ref_text=None)
    xvec_synth = QwenProsodyCloneSynthesizer(
        clone_model,
        clone_method="xvector",
        seed=seed,
    )
    xvec_run_dir = work_dir / "xvector_run"
    xvec_res = run_feedback(
        clone_script_path,
        xvec_run_dir,
        backend=xvec_synth,
        voices=clone_voices,
        voice_profiles=voice_profiles,
    )
    mode_feedbacks["xvector"] = _extract_path_from_result(xvec_res, xvec_run_dir / "feedback.json")

    # 4. Experiment C: Single-variable sweeps (optional)
    if include_experiment_c:
        for spec in build_experiment_c_specs("xvector") + build_experiment_c_specs("icl")[-1:]:
            st = spec["settings"]
            c_synth = QwenProsodyCloneSynthesizer(
                clone_model,
                clone_method=spec["clone_method"],
                seed=st["seed"],
                temperature=st["temperature"],
                top_k=st["top_k"],
                top_p=st["top_p"],
                repetition_penalty=st["repetition_penalty"],
                ref_trailing_silence_seconds=st["ref_trailing_silence_seconds"],
            )
            c_run_dir = work_dir / f"{spec['mode_name']}_run"
            c_res = run_feedback(
                clone_script_path,
                c_run_dir,
                backend=c_synth,
                voices=clone_voices,
                voice_profiles=voice_profiles,
            )
            mode_feedbacks[spec["mode_name"]] = _extract_path_from_result(c_res, c_run_dir / "feedback.json")

    active_embedder = resolve_embedder(embedder, embeddings=embeddings)
    return build_prosody_comparison_bundle(
        reference_wav=ref_audio_path,
        speaker_id=speaker_id,
        mode_feedbacks=mode_feedbacks,
        output_dir=root,
        embedder=active_embedder,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="TransClip Cross-lingual Voice Cloning Prosody Experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run-all", help="Run Preset, ICL (Exp A), X-vector (Exp B), and optional Exp C sweeps")
    run_p.add_argument("--speaker", default=DEFAULT_SPEAKER_ID)
    run_p.add_argument("--reference-audio", required=True)
    run_p.add_argument("--reference-text", required=True)
    run_p.add_argument("--consent", required=True)
    run_p.add_argument("--review", required=True)
    run_p.add_argument("--generation-record", default=None)
    run_p.add_argument("--preset-voice", default="Ryan")
    run_p.add_argument("--preset-model", default=DEFAULT_PRESET_MODEL)
    run_p.add_argument("--clone-model", default=DEFAULT_BASE_MODEL)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--include-experiment-c", action="store_true")
    run_p.add_argument("--embeddings", action="store_true")
    run_p.add_argument("--existing-preset-feedback", default=None)
    run_p.add_argument("--existing-icl-feedback", default=None)
    run_p.add_argument("--output", default="output/prosody_experiments")

    cmp_p = sub.add_parser("compare", help="Build output/prosody_experiments bundle from existing feedback.json files")
    cmp_p.add_argument("--speaker", default=DEFAULT_SPEAKER_ID)
    cmp_p.add_argument("--reference-audio", required=True)
    cmp_p.add_argument("--preset-feedback", default=None)
    cmp_p.add_argument("--icl-feedback", default=None)
    cmp_p.add_argument("--xvector-feedback", default=None)
    cmp_p.add_argument("--embeddings", action="store_true")
    cmp_p.add_argument("--output", default="output/prosody_experiments")

    args = parser.parse_args(argv)
    if args.command == "run-all":
        comp_path, comp = run_prosody_experiments(
            speaker_id=args.speaker,
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            consent_path=args.consent,
            review_path=args.review,
            generation_record_path=args.generation_record,
            output_dir=args.output,
            preset_voice=args.preset_voice,
            preset_model=args.preset_model,
            clone_model=args.clone_model,
            seed=args.seed,
            include_experiment_c=args.include_experiment_c,
            embeddings=args.embeddings,
            existing_preset_feedback=args.existing_preset_feedback,
            existing_icl_feedback=args.existing_icl_feedback,
        )
        print(json.dumps({"comparison_json": str(comp_path), "available_modes": comp["available_modes"]}, ensure_ascii=False))
        return 0
    if args.command == "compare":
        active_embedder = resolve_embedder(None, embeddings=args.embeddings)
        comp_path, comp = build_prosody_comparison_bundle(
            reference_wav=args.reference_audio,
            speaker_id=args.speaker,
            mode_feedbacks={
                "preset": args.preset_feedback,
                "icl": args.icl_feedback,
                "xvector": args.xvector_feedback,
            },
            output_dir=args.output,
            embedder=active_embedder,
        )
        print(json.dumps({"comparison_json": str(comp_path), "available_modes": comp["available_modes"]}, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
