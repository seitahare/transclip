"""Read-only Phase 2 inputs -> independent speaker timelines -> verified MP4.

No NLP/TTS runtime is loaded. Rendering uses FFmpeg and Python's standard library
(plus numpy for Phase 13 dynamic audio ducking, ambient separation, and loudness verification).
"""
import argparse
from array import array
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from language.cache import atomic_json, fingerprint
from language.timing import measure_wav

try:
    from tts_feedback import UNKNOWN, detect_collisions, number, validate_script, audio_request
except ImportError:
    UNKNOWN = (None, "UNKNOWN", "REVIEW", "MIXED")
    detect_collisions = validate_script = audio_request = None

    def number(value, label, *, positive=False):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {label}.")
        return value

BASE_DIR = Path(__file__).resolve().parent
RATE = 48000
CHANNELS = 2
ENGINE_VERSION = 1
PEAK_TARGET = 10 ** (-2 / 20)
VALID_MIX_MODES = (
    "japanese-only",
    "reference-mix",
    "voiceover",
    "ducking",
    "ambient",
    "hybrid-ambient",
)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read JSON {path}: {error}") from error


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command):
    try:
        return subprocess.run([str(x) for x in command], check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(f"{command[0]} failed (exit {error.returncode}): {detail}") from error
    except OSError as error:
        raise RuntimeError(f"Cannot execute {command[0]}: {error}") from error


def probe(path):
    return json.loads(run_command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))


def ffmpeg(args):
    return run_command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", *args])


def resolve_input(value, parent, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {label}.")
    path = Path(value)
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


# ============================================================================
# Phase 13: Dynamic Audio Ducking (Voice-Over), Ambient Separation & Loudness
# ============================================================================

def merge_speech_intervals(intervals, *, total_duration_seconds=None, pre_duck_seconds=0.06,
                          post_duck_seconds=0.08, min_unspoken_gap_seconds=0.35):
    """Normalize and merge active Japanese TTS speech intervals, bridging micro-gaps."""
    cleaned = []
    for item in intervals or []:
        if isinstance(item, dict):
            a = float(item.get("start", item.get("placement_seconds", item.get("clip_relative_start_seconds", 0.0))))
            b = float(item.get("end", a + float(item.get("duration", item.get("actual_tts_duration_seconds", 0.0)))))
        else:
            a, b = float(item[0]), float(item[1])
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
            continue
        expanded_a = max(0.0, a - max(0.0, float(pre_duck_seconds)))
        expanded_b = b + max(0.0, float(post_duck_seconds))
        if total_duration_seconds is not None and total_duration_seconds > 0:
            expanded_a = min(float(total_duration_seconds), expanded_a)
            expanded_b = min(float(total_duration_seconds), expanded_b)
        if expanded_b > expanded_a:
            cleaned.append((expanded_a, expanded_b))

    if not cleaned:
        return []
    cleaned.sort(key=lambda x: (x[0], x[1]))
    merged = [cleaned[0]]
    bridge = max(0.0, float(min_unspoken_gap_seconds))
    for a, b in cleaned[1:]:
        prev_a, prev_b = merged[-1]
        if a - prev_b <= bridge + 1e-6:
            merged[-1] = (prev_a, max(prev_b, b))
        else:
            merged.append((a, b))
    return [(round(a, 6), round(b, 6)) for a, b in merged]


def build_ducking_envelope(
    total_frames,
    *,
    sample_rate=RATE,
    speech_intervals=(),
    duck_gain=0.12,
    unspoken_gain=1.0,
    fade_out_seconds=0.18,
    fade_in_seconds=0.25,
    pre_duck_seconds=0.06,
    post_duck_seconds=0.08,
    min_unspoken_gap_seconds=0.35,
    curve="cosine",
):
    """Build a sample-accurate 1D gain envelope for voice-over ducking.

    - During active Japanese TTS speech: attenuates original audio to `duck_gain` (e.g. 0.10..0.15).
    - During non-TTS intervals: retains original audio at `unspoken_gain` (default 1.0).
    - Applies smooth raised-cosine (or linear) fade-out before speech starts and fade-in after speech ends.
    """
    if type(total_frames) is not int or total_frames <= 0:
        raise ValueError("total_frames must be a positive integer.")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    for label, val, lo, hi in [
        ("duck_gain", duck_gain, 0.0, 0.5),
        ("unspoken_gain", unspoken_gain, 0.0, 1.5),
        ("fade_out_seconds", fade_out_seconds, 0.0, 3.0),
        ("fade_in_seconds", fade_in_seconds, 0.0, 3.0),
        ("pre_duck_seconds", pre_duck_seconds, 0.0, 1.0),
        ("post_duck_seconds", post_duck_seconds, 0.0, 1.0),
        ("min_unspoken_gap_seconds", min_unspoken_gap_seconds, 0.0, 5.0),
    ]:
        if type(val) not in (int, float) or not math.isfinite(val) or not (lo <= val <= hi):
            raise ValueError(f"Invalid {label}: {val}")
    if curve not in ("cosine", "linear"):
        raise ValueError("curve must be 'cosine' or 'linear'.")

    total_duration = total_frames / float(sample_rate)
    merged = merge_speech_intervals(
        speech_intervals,
        total_duration_seconds=total_duration,
        pre_duck_seconds=pre_duck_seconds,
        post_duck_seconds=post_duck_seconds,
        min_unspoken_gap_seconds=min_unspoken_gap_seconds,
    )

    duck_g = float(duck_gain)
    unspoken_g = float(unspoken_gain)
    env = np.full(total_frames, unspoken_g, dtype=np.float32)
    if not merged or abs(unspoken_g - duck_g) < 1e-7:
        return env, {
            "duck_gain": duck_g,
            "unspoken_gain": unspoken_g,
            "fade_out_seconds": float(fade_out_seconds),
            "fade_in_seconds": float(fade_in_seconds),
            "curve": curve,
            "merged_speech_windows": merged,
            "transition_count": 0,
            "ducked_frame_fraction": 0.0 if not merged else 1.0,
            "unspoken_frame_fraction": 1.0 if not merged else 0.0,
            "min_gain": float(np.min(env)),
            "max_gain": float(np.max(env)),
        }

    fade_out_frames = max(1, int(round(float(fade_out_seconds) * sample_rate))) if fade_out_seconds > 0 else 0
    fade_in_frames = max(1, int(round(float(fade_in_seconds) * sample_rate))) if fade_in_seconds > 0 else 0

    for a_sec, b_sec in merged:
        s_idx = max(0, min(total_frames, int(round(a_sec * sample_rate))))
        e_idx = max(s_idx, min(total_frames, int(round(b_sec * sample_rate))))
        if e_idx > s_idx:
            env[s_idx:e_idx] = np.minimum(env[s_idx:e_idx], duck_g)

        # Smooth fade-out (attack) leading into s_idx
        if fade_out_frames > 0 and s_idx > 0:
            ramp_start = max(0, s_idx - fade_out_frames)
            ramp_len = s_idx - ramp_start
            if ramp_len > 0:
                t = np.linspace(0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32)
                w = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32) if curve == "cosine" else t
                ramp_vals = unspoken_g - (unspoken_g - duck_g) * w
                env[ramp_start:s_idx] = np.minimum(env[ramp_start:s_idx], ramp_vals)

        # Smooth fade-in (release) trailing after e_idx
        if fade_in_frames > 0 and e_idx < total_frames:
            ramp_end = min(total_frames, e_idx + fade_in_frames)
            ramp_len = ramp_end - e_idx
            if ramp_len > 0:
                t = np.linspace(0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32)
                w = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32) if curve == "cosine" else t
                ramp_vals = duck_g + (unspoken_g - duck_g) * w
                env[e_idx:ramp_end] = np.minimum(env[e_idx:ramp_end], ramp_vals)

    ducked_mask = np.abs(env - duck_g) <= 0.02 * max(1.0, abs(unspoken_g - duck_g))
    unspoken_mask = np.abs(env - unspoken_g) <= 0.02 * max(1.0, abs(unspoken_g - duck_g))
    return env, {
        "duck_gain": duck_g,
        "unspoken_gain": unspoken_g,
        "fade_out_seconds": float(fade_out_seconds),
        "fade_in_seconds": float(fade_in_seconds),
        "pre_duck_seconds": float(pre_duck_seconds),
        "post_duck_seconds": float(post_duck_seconds),
        "min_unspoken_gap_seconds": float(min_unspoken_gap_seconds),
        "curve": curve,
        "merged_speech_windows": merged,
        "transition_count": len(merged) * 2,
        "ducked_frame_fraction": round(float(np.mean(ducked_mask)), 4),
        "unspoken_frame_fraction": round(float(np.mean(unspoken_mask)), 4),
        "min_gain": round(float(np.min(env)), 4),
        "max_gain": round(float(np.max(env)), 4),
    }


def _dsp_spectral_separate_ambient(samples, sample_rate=RATE, *, vocal_suppression_db=18.0):
    """Zero-dependency STFT spectral vocal suppression & ambient/room-tone extractor.

    Suppresses human speech fundamental & formant frequencies (120..3800 Hz) where vocal
    harmonicity/energy exceeds the stationary room-tone floor, while preserving stereo side
    ambience, low/high room acoustics, and non-speech atmosphere.
    """
    arr = np.asarray(samples, dtype=np.float32)
    is_1d = arr.ndim == 1
    if is_1d:
        stereo = arr[:, None]
    elif arr.ndim == 2:
        stereo = arr
    else:
        raise ValueError("Audio array must be 1D (mono) or 2D [frames, channels].")

    n_frames, n_ch = stereo.shape
    if n_frames == 0:
        return arr.copy(), np.zeros_like(arr)

    n_fft = max(256, min(2048, 2 ** int( round(math.log2(max(256, sample_rate * 0.04))) )))
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))

    # Vocal formant attenuation profile (smooth bandpass mask in 120..3800 Hz)
    vocal_band = np.zeros_like(freqs, dtype=np.float32)
    in_band = (freqs >= 110.0) & (freqs <= 4000.0)
    vocal_band[in_band] = 1.0
    # Smooth edges around 110..220 Hz and 3000..4000 Hz
    low_edge = (freqs >= 110.0) & (freqs < 220.0)
    vocal_band[low_edge] = (freqs[low_edge] - 110.0) / 110.0
    high_edge = (freqs > 3000.0) & (freqs <= 4000.0)
    vocal_band[high_edge] = (4000.0 - freqs[high_edge]) / 1000.0

    min_vocal_gain = float(10.0 ** (-abs(float(vocal_suppression_db)) / 20.0))

    def _extract_mono_ambient(x_1d):
        if len(x_1d) < n_fft:
            return x_1d * 0.25
        pad_len = n_fft + (hop - (len(x_1d) % hop)) % hop
        padded = np.pad(x_1d, (n_fft // 2, pad_len), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
        windowed = frames * win[None, :]
        spec = np.fft.rfft(windowed, axis=1)
        mag = np.abs(spec)

        # Stationary room-tone floor per frequency bin (20th percentile across time)
        floor_mag = np.percentile(mag, 20, axis=0, keepdims=True).astype(np.float32)
        # Wiener-like ambient mask: keep stationary floor + non-vocal bands, attenuate transient vocal bursts
        ambient_ratio = np.clip(floor_mag / np.maximum(mag, 1e-7), min_vocal_gain, 1.0)
        mask = 1.0 - vocal_band[None, :] * (1.0 - ambient_ratio)
        mask = np.clip(mask, min_vocal_gain, 1.0)

        amb_spec = spec * mask
        amb_frames = np.fft.irfft(amb_spec, n=n_fft, axis=1).astype(np.float32) * win[None, :]

        out_len = (len(amb_frames) - 1) * hop + n_fft
        out = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)
        win_sq = (win * win).astype(np.float32)
        for idx in range(len(amb_frames)):
            pos = idx * hop
            out[pos : pos + n_fft] += amb_frames[idx]
            norm[pos : pos + n_fft] += win_sq
        out /= np.maximum(norm, 1e-6)
        start = n_fft // 2
        return out[start : start + len(x_1d)]

    if n_ch == 2:
        mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
        side = 0.5 * (stereo[:, 0] - stereo[:, 1])
        mid_amb = _extract_mono_ambient(mid)
        # Stereo side channel naturally carries room reverberation/ambience while center voice cancels
        amb_l = np.clip(mid_amb + 0.85 * side, -1.0, 1.0)
        amb_r = np.clip(mid_amb - 0.85 * side, -1.0, 1.0)
        ambient = np.stack([amb_l, amb_r], axis=1).astype(np.float32)
    else:
        channels_amb = [_extract_mono_ambient(stereo[:, c]) for c in range(n_ch)]
        ambient = np.stack(channels_amb, axis=1).astype(np.float32)

    speech = (stereo - ambient).astype(np.float32)
    if is_1d:
        return ambient[:, 0], speech[:, 0]
    return ambient, speech


def separate_ambient_and_speech(
    audio_samples,
    sample_rate=RATE,
    *,
    backend="auto",
    weights_path=None,
    model_config=None,
    allow_fallback=True,
):
    """Separate original audio into background ambient/room-tone stem and speaker voice stem.

    Supports external neural backends ('demucs', 'torchscript', 'convtasnet') when available
    and a built-in DSP spectral ambient separator ('dsp') so separation works in any environment.
    """
    valid_backends = ("auto", "dsp", "demucs", "torchscript", "convtasnet", "none")
    if backend not in valid_backends:
        raise ValueError(f"Unsupported ambient separation backend {backend!r}; expected {valid_backends}")

    arr = np.asarray(audio_samples, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite samples in source audio for ambient separation.")

    if backend == "none":
        if not allow_fallback:
            raise RuntimeError("Ambient separation backend is set to 'none' and fallback is disabled.")
        return {
            "ambient": arr * 0.12,
            "speech": arr * 0.88,
            "backend_requested": backend,
            "backend_used": "ducking_fallback",
            "fallback_used": True,
            "separation_available": False,
        }

    # Try external TorchScript / ConvTasNet separator if explicitly requested and weights exist
    if backend in ("torchscript", "convtasnet"):
        try:
            if not weights_path or not Path(weights_path).is_file():
                raise FileNotFoundError(f"Weights file required for {backend} separation: {weights_path}")
            from separation_backends import ConvTasNetSeparator, TorchScriptSeparator
            mono = arr.mean(axis=1) if arr.ndim == 2 else arr
            sep = (
                ConvTasNetSeparator(weights_path)
                if backend == "convtasnet"
                else TorchScriptSeparator(weights_path, model_config or {})
            )
            stems = sep.separate(mono.astype(np.float32), int(sample_rate), 2)
            # Identify lower-voiced stem as ambient bed
            rms_vals = [float(np.sqrt(np.mean(np.square(s, dtype=np.float64)))) for s in stems]
            amb_idx = int(np.argmin(rms_vals))
            spk_idx = 1 - amb_idx
            amb_mono = np.asarray(stems[amb_idx], dtype=np.float32)
            spk_mono = np.asarray(stems[spk_idx], dtype=np.float32)
            if arr.ndim == 2:
                amb_out = np.repeat(amb_mono[:, None], arr.shape[1], axis=1)
                spk_out = np.repeat(spk_mono[:, None], arr.shape[1], axis=1)
            else:
                amb_out, spk_out = amb_mono, spk_mono
            return {
                "ambient": amb_out,
                "speech": spk_out,
                "backend_requested": backend,
                "backend_used": backend,
                "fallback_used": False,
                "separation_available": True,
            }
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"Ambient separation backend {backend!r} failed: {exc}") from exc

    # Built-in DSP spectral ambient separator (default for 'auto' and 'dsp', or fallback when neural unavailable)
    try:
        ambient, speech = _dsp_spectral_separate_ambient(arr, sample_rate=sample_rate)
        orig_rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))) if arr.size else 0.0
        amb_rms = float(np.sqrt(np.mean(np.square(ambient, dtype=np.float64)))) if ambient.size else 0.0
        spk_rms = float(np.sqrt(np.mean(np.square(speech, dtype=np.float64)))) if speech.size else 0.0
        return {
            "ambient": ambient,
            "speech": speech,
            "backend_requested": backend,
            "backend_used": "dsp_spectral_ambient",
            "fallback_used": backend not in ("auto", "dsp"),
            "separation_available": True,
            "metrics": {
                "original_rms": round(orig_rms, 6),
                "ambient_rms": round(amb_rms, 6),
                "speech_rms": round(spk_rms, 6),
                "ambient_to_original_rms_ratio": round(amb_rms / max(orig_rms, 1e-8), 4),
            },
        }
    except Exception as exc:
        if not allow_fallback:
            raise
        return {
            "ambient": arr * 0.12,
            "speech": arr * 0.88,
            "backend_requested": backend,
            "backend_used": "ducking_fallback",
            "fallback_used": True,
            "separation_available": False,
            "fallback_reason": str(exc),
        }


def apply_soft_limiter(samples, *, ceiling=PEAK_TARGET, knee=0.70):
    """Apply smooth soft-knee limiting above `knee` so peaks never exceed `ceiling` or clip."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32), {"pre_peak": 0.0, "post_peak": 0.0, "limited_samples": 0}
    pre_peak = float(np.max(np.abs(arr)))
    ceil_val = float(ceiling)
    knee_val = min(float(knee), ceil_val * 0.95)
    if pre_peak <= ceil_val:
        return arr.astype(np.float32), {"pre_peak": round(pre_peak, 6), "post_peak": round(pre_peak, 6), "limited_samples": 0}

    abs_x = np.abs(arr)
    over_mask = abs_x > knee_val
    limited_count = int(np.sum(over_mask))
    span = max(1e-6, ceil_val - knee_val)
    out = arr.copy()
    excess = (abs_x[over_mask] - knee_val) / span
    compressed = knee_val + span * np.tanh(excess)
    out[over_mask] = np.sign(arr[over_mask]) * compressed
    post_peak = float(np.max(np.abs(out)))
    if post_peak > ceil_val:
        out *= ceil_val / post_peak
        post_peak = float(np.max(np.abs(out)))
    return out.astype(np.float32), {
        "pre_peak": round(pre_peak, 6),
        "post_peak": round(post_peak, 6),
        "limited_samples": limited_count,
    }


def verify_audio_loudness_and_clipping(
    mixed_samples,
    *,
    japanese_samples=None,
    background_samples=None,
    source_samples=None,
    sample_rate=RATE,
    speech_intervals=(),
):
    """Compute volume, loudness (dBFS / approximate LUFS), dynamic range, and clipping verification metrics."""
    mix_arr = np.asarray(mixed_samples, dtype=np.float64)
    if mix_arr.ndim == 2:
        mix_mono = np.mean(mix_arr, axis=1)
    else:
        mix_mono = mix_arr

    def _dbfs(rms_or_peak):
        return round(20.0 * math.log10(max(float(rms_or_peak), 1e-9)), 2)

    peak_val = float(np.max(np.abs(mix_arr))) if mix_arr.size else 0.0
    rms_val = float(np.sqrt(np.mean(np.square(mix_mono)))) if mix_mono.size else 0.0
    clipped_count = int(np.sum(np.abs(mix_arr) >= 0.9999))
    clipped_fraction = round(clipped_count / max(1, int(mix_arr.size)), 6)

    n_frames = len(mix_mono)
    speech_mask = np.zeros(n_frames, dtype=bool)
    for item in speech_intervals or []:
        if isinstance(item, dict):
            a = float(item.get("start", item.get("placement_seconds", item.get("clip_relative_start_seconds", 0.0))))
            b = float(item.get("end", a + float(item.get("duration", item.get("actual_tts_duration_seconds", 0.0)))))
        else:
            a, b = float(item[0]), float(item[1])
        s_i = max(0, min(n_frames, int(round(a * sample_rate))))
        e_i = max(s_i, min(n_frames, int(round(b * sample_rate))))
        if e_i > s_i:
            speech_mask[s_i:e_i] = True

    unspoken_mask = ~speech_mask
    speech_rms = float(np.sqrt(np.mean(np.square(mix_mono[speech_mask])))) if np.any(speech_mask) else 0.0
    unspoken_rms = float(np.sqrt(np.mean(np.square(mix_mono[unspoken_mask])))) if np.any(unspoken_mask) else 0.0

    bg_speech_rms = None
    src_speech_rms = None
    measured_duck_ratio = None
    if background_samples is not None:
        bg_arr = np.asarray(background_samples, dtype=np.float64)
        bg_mono = np.mean(bg_arr, axis=1) if bg_arr.ndim == 2 else bg_arr
        if len(bg_mono) == n_frames and np.any(speech_mask):
            bg_speech_rms = float(np.sqrt(np.mean(np.square(bg_mono[speech_mask]))))
    if source_samples is not None:
        src_arr = np.asarray(source_samples, dtype=np.float64)
        src_mono = np.mean(src_arr, axis=1) if src_arr.ndim == 2 else src_arr
        if len(src_mono) == n_frames and np.any(speech_mask):
            src_speech_rms = float(np.sqrt(np.mean(np.square(src_mono[speech_mask]))))
            if bg_speech_rms is not None and src_speech_rms > 1e-7:
                measured_duck_ratio = round(bg_speech_rms / src_speech_rms, 4)

    return {
        "peak_amplitude": round(peak_val, 6),
        "peak_dbfs": _dbfs(peak_val),
        "rms_amplitude": round(rms_val, 6),
        "rms_dbfs": _dbfs(rms_val),
        "integrated_lufs_approx": round(_dbfs(rms_val) - 0.691, 2),
        "clipped_sample_count": clipped_count,
        "clipped_sample_fraction": clipped_fraction,
        "speech_interval_rms": round(speech_rms, 6),
        "speech_interval_dbfs": _dbfs(speech_rms) if speech_rms > 0 else None,
        "unspoken_interval_rms": round(unspoken_rms, 6),
        "unspoken_interval_dbfs": _dbfs(unspoken_rms) if unspoken_rms > 0 else None,
        "background_during_speech_rms": round(bg_speech_rms, 6) if bg_speech_rms is not None else None,
        "measured_duck_ratio_during_speech": measured_duck_ratio,
        "clipping_free": clipped_count == 0 and peak_val < 1.0,
    }


def validate_feedback(feedback_path, selection_path, video_id):
    """Check content provenance, not just a WAV filename or declared duration."""
    feedback_path = Path(feedback_path).resolve()
    report = read_json(feedback_path)
    if (report.get("kind") != "tts_feedback" or report.get("format_version") != 1
            or report.get("video_id") != video_id):
        raise ValueError("Feedback format/video_id mismatch.")
    fingerprint(report)  # Reject non-finite values anywhere in provenance/settings.
    for key in ("run_key", "source_script_fingerprint"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(report.get(key, ""))):
            raise ValueError(f"Invalid feedback {key}.")
    selection = read_json(selection_path)
    a, b = (number(selection.get(k), f"selection {k}") for k in ("start", "end"))
    if a >= b or any(report.get("clip", {}).get(k) != selection[k] for k in ("start", "end")):
        raise ValueError("Feedback clip does not match selection.json.")
    if selection.get("video_id", video_id) != video_id:
        raise ValueError("Selection video_id mismatch.")
    source_path = resolve_input(report.get("source_script_path"), feedback_path.parent, "source_script_path")
    script = read_json(source_path)
    source_entries = validate_script(script)
    if (fingerprint(script) != report["source_script_fingerprint"] or script["video_id"] != video_id
            or script["clip"] != report["clip"]):
        raise ValueError("Source script fingerprint/video/clip mismatch.")
    rows = copy.deepcopy(report.get("entries"))
    if not isinstance(rows, list) or [r.get("id") for r in rows] != [r["id"] for r in source_entries]:
        raise ValueError("Feedback entry IDs/order do not match source script.")
    preview = report.get("preview", {})
    if preview.get("relative_start") != 0:
        raise ValueError("Only clip-relative previews starting at zero are supported.")
    limit = number(preview.get("relative_end"), "feedback preview end", positive=True)
    if limit > script["preview"]["relative_end"] or limit > b - a:
        raise ValueError("Feedback preview exceeds source script/selection.")
    if report.get("source_excluded") != script.get("excluded", []):
        raise ValueError("Feedback excluded-source provenance mismatch.")
    for row, source in zip(rows, source_entries):
        uid = row["id"]
        for key in ("source_utterance_ids", "speaker", "relative_start", "relative_end", "start", "end",
                    "text_en", "text_ja", "source_duration_seconds", "estimated_duration_seconds", "provenance"):
            if row.get(key) != source.get(key):
                raise ValueError(f"{uid}: source script {key} mismatch.")
        if type(row.get("review_required")) is not bool or not isinstance(row.get("review_reasons"), list):
            raise ValueError(f"{uid}: invalid review metadata.")
        if (any(not isinstance(x, str) for x in row["review_reasons"])
                or not set(source["review_reasons"]).issubset(row["review_reasons"])
                or (source["review_required"] and not row["review_required"])):
            raise ValueError(f"{uid}: lost source review metadata.")
        status = row.get("generation_status")
        if status not in ("complete", "skipped", "pending", "failed"):
            raise ValueError(f"{uid}: invalid generation_status.")
        if status != "complete":
            if row.get("audio_file") is not None or row.get("actual_tts_duration_seconds") is not None:
                raise ValueError(f"{uid}: incomplete entry claims generated audio.")
            continue
        if row["relative_start"] >= limit or row["relative_end"] > limit:
            raise ValueError(f"{uid}: generated entry is outside feedback preview coverage.")
        if row["speaker"] in UNKNOWN or not re.fullmatch(r"[A-Za-z0-9_-]+", row["speaker"]):
            raise ValueError(f"{uid}: generated audio requires an explicit known speaker.")
        if not isinstance(row.get("voice"), str) or not row["voice"]:
            raise ValueError(f"{uid}: missing voice.")
        tts = row.get("tts")
        if (not isinstance(tts, dict) or not tts.get("model") or not tts.get("backend")
                or not isinstance(tts.get("settings"), dict) or tts != report.get("tts")):
            raise ValueError(f"{uid}: inconsistent TTS identity/settings.")
        request = audio_request(row, tts)
        key = fingerprint(request)
        if row.get("audio_cache_key") != key:
            raise ValueError(f"{uid}: audio cache identity mismatch.")
        path = resolve_input(row.get("audio_file"), feedback_path.parent, f"{uid} audio_file")
        try:
            measured = measure_wav(path)
            receipt = read_json(path.with_name("receipt.json"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{uid}: invalid/missing WAV or receipt {path}: {error}") from error
        if (measured != row.get("wav_measurement") or measured != receipt.get("measurement")
                or receipt.get("cache_key") != key or receipt.get("request") != request):
            raise ValueError(f"{uid}: WAV hash/measurement/receipt mismatch.")
        duration = number(row.get("actual_tts_duration_seconds"), f"{uid} actual duration", positive=True)
        if abs(duration - measured["audio_duration"]) > 1e-9:
            raise ValueError(f"{uid}: actual WAV duration mismatch.")
        if abs(number(row.get("actual_playback_end"), f"{uid} playback end") - row["relative_start"] - duration) > 1e-9:
            raise ValueError(f"{uid}: playback end mismatch.")
        row["audio_file"] = str(path)
    detect_collisions(rows)  # Recompute from measured lengths, never trust cached overlap lists.
    return report, rows, selection, source_path


def overlap_pairs(rows, field):
    pairs = []
    for row in rows:
        for collision in row[field]:
            if row["id"] < collision["entry_id"]:
                pairs.append({"entry_ids": [row["id"], collision["entry_id"]],
                              "speaker": row["speaker"] if field == "actual_same_speaker_conflicts" else None,
                              **{k: v for k, v in collision.items() if k != "entry_id"}})
    return pairs


def attach_speaker_evidence(plan, analysis_path):
    """Cross-check Phase 4 assignments; never relabel already generated WAVs."""
    path = Path(analysis_path).resolve()
    analysis = read_json(path)
    if (analysis.get("kind") != "speaker_analysis" or analysis.get("format_version") != 1
            or analysis.get("video_id") != plan["video_id"] or analysis.get("clip") != plan["clip"]):
        raise ValueError("Speaker analysis video/clip/format mismatch.")
    utterance_path = (path.parent / analysis.get("utterances_file", "utterances.enhanced.json")).resolve()
    if not utterance_path.is_relative_to(path.parent):
        raise ValueError("Speaker utterance path escapes analysis directory.")
    utterances = read_json(utterance_path)
    if utterances.get("video_id") != plan["video_id"] or utterances.get("clip") != plan["clip"]:
        raise ValueError("Speaker utterance video/clip mismatch.")
    indexed = {r["id"]: r for r in utterances["utterances"]}
    if len(indexed) != len(utterances["utterances"]):
        raise ValueError("Duplicate speaker utterance IDs.")
    conflicts = []
    for row in plan["placements"]:
        for uid in row["source_utterance_ids"]:
            if uid not in indexed:
                raise ValueError("Missing source utterance in speaker analysis.")
            proposed = indexed[uid]
            if not (row["relative_start"] <= proposed["relative_start"] < proposed["relative_end"] <= row["relative_end"] + .001):
                raise ValueError("Speaker evidence source interval mismatch.")
            if proposed["speaker"] != row["speaker"]:
                conflicts.append({"entry_id": row["id"], "source_utterance_id": uid,
                                  "tts_speaker": row["speaker"], "analyzed_speaker": proposed["speaker"]})
    plan["speaker_evidence"] = {"analysis_key": analysis.get("analysis_key"), "assignment_conflicts": conflicts,
                                "source_overlap_regions": analysis.get("source_overlap_regions", []),
                                "policy": "Regenerate script/feedback after speaker changes; cached WAVs are not relabeled."}
    plan["inputs"]["speaker_analysis"] = str(path)
    plan["inputs"]["speaker_analysis_fingerprint"] = fingerprint(analysis)
    plan["inputs"]["speaker_utterances"] = str(utterance_path)
    plan["inputs"]["speaker_utterances_fingerprint"] = fingerprint(utterances)
    if conflicts:
        plan["blockers"].append("speaker_assignment_changed_regenerate_script_and_feedback")
    return plan


def build_plan(feedback_path, selection_path, video_path, info_path, video_id, *, seconds=30,
               mode="japanese-only", diagnostic=False, extend_to_fit=False, reference_gain=.12, max_timing_shift=0,
               unspoken_gain=1.0, fade_out_seconds=.18, fade_in_seconds=.25, ambient_backend="auto", ambient_gain=.85):
    number(max_timing_shift, "maximum timing shift")
    if max_timing_shift > 2:raise ValueError("Maximum timing shift is limited to 2 seconds.")
    number(seconds, "seconds", positive=True)
    number(reference_gain, "reference gain")
    if reference_gain > .5 or mode not in VALID_MIX_MODES:
        raise ValueError("Invalid mode or reference gain (allowed 0..0.5).")
    feedback, rows, selection, script_path = validate_feedback(feedback_path, selection_path, video_id)
    metadata = read_json(info_path)
    if metadata.get("id") != video_id:
        raise ValueError("Source video metadata ID mismatch.")
    video_path = Path(video_path).resolve()
    # Bind metadata to its local source asset; do not accept a different folder's video.
    if video_path.parent != Path(info_path).resolve().parent or video_path.stem != "source":
        raise ValueError("Source video must be source.* beside its source.info.json.")
    video_info = probe(video_path)
    if not any(s.get("codec_type") == "video" for s in video_info["streams"]):
        raise ValueError("Source has no video stream.")
    source_duration = float(video_info["format"]["duration"])
    if not math.isfinite(source_duration) or source_duration < selection["end"]:
        raise ValueError("Selection extends beyond source video.")
    if metadata.get("duration") is not None:
        declared = number(metadata["duration"], "source metadata duration", positive=True)
        if abs(source_duration - declared) > 2:
            raise ValueError("Source media duration does not match source.info.json.")
    if mode != "japanese-only" and not any(s.get("codec_type") == "audio" for s in video_info["streams"]):
        raise ValueError("Reference Mix requires a source audio stream.")
    if seconds > feedback["preview"]["relative_end"] or seconds > selection["end"] - selection["start"]:
        raise ValueError("Requested preview exceeds feedback/selection coverage.")
    active = [r for r in rows if r["relative_start"] < seconds]
    complete = [r for r in active if r["generation_status"] == "complete"]
    missing = [r for r in active if r["generation_status"] != "complete"]
    timing_adjustment = None
    if max_timing_shift:
        from timing_adjustment import bounded_placements
        obstacles = [dict(r, id="excluded:"+str(i), speaker=None, generation_status="skipped")
                     for i,r in enumerate(feedback.get("source_excluded",[]))
                     if "relative_start" in r and "relative_end" in r]
        timing_adjustment = bounded_placements([*active,*obstacles],seconds=seconds,max_shift=max_timing_shift)
    # Round only to the nearest output sample; preserve full resampled lengths.
    for row in complete:
        row["placement_sample"] = round((timing_adjustment["starts"][row["id"]] if timing_adjustment else row["relative_start"]) * RATE)
        row["placement_seconds"] = row["placement_sample"] / RATE
        row["end_sample"] = row["placement_sample"] + math.ceil(row["actual_tts_duration_seconds"] * RATE - 1e-7)
        row["absolute_placement_seconds"] = selection["start"] + row["placement_seconds"]
    frames = math.ceil(seconds * RATE - 1e-7)
    overflows = [r["id"] for r in complete if r["end_sample"] > frames]
    blockers = []
    if overflows and extend_to_fit:
        frames = max(r["end_sample"] for r in complete)
        # Extending the boundary may bring additional entries into range.
        if any(seconds <= r["relative_start"] < frames / RATE for r in rows):
            raise ValueError("Extension includes additional entries; regenerate a wider feedback preview first.")
    elif overflows:
        blockers.append("audio_exceeds_preview_use_extend_to_fit")
    duration = frames / RATE
    if duration > selection["end"] - selection["start"] + 1e-9:
        blockers.append("audio_exceeds_selected_source_range")
    detect_collisions(active,use_placements=bool(timing_adjustment))
    same = overlap_pairs(active, "actual_same_speaker_conflicts")
    other = overlap_pairs(active, "actual_other_speaker_overlaps")
    excluded = [r for r in feedback.get("source_excluded", []) if r.get("relative_start", math.inf) < duration]
    review_ids = [r["id"] for r in active if r["review_required"]]
    if not complete:
        blockers.append("no_generated_audio")
    if timing_adjustment and same:
        blockers.append("bounded_timing_unresolved_same_speaker_collision")
    if timing_adjustment and any(p["source_overlap_seconds"] <= 0 for p in other):
        blockers.append("bounded_timing_introduced_other_speaker_overlap")
    if timing_adjustment and any(e["status"]=="placed" for e in timing_adjustment["events"]) and not diagnostic:
        blockers.append("timing_adjustment_listening_review_required")
    if same and any(r.get("voice_mode")=="approved_clone" for r in active):
        blockers.append("clone_same_speaker_collision_requires_resolution")
    if not diagnostic:
        if same: blockers.append("unresolved_same_speaker_collisions")
        if missing or excluded: blockers.append("undubbed_source_intervals")
        if review_ids: blockers.append("unresolved_entry_reviews")
        if any(p["source_overlap_seconds"] <= 0 for p in other): blockers.append("introduced_other_speaker_overlap")
    quality = "diagnostic" if diagnostic else "reviewed"
    if same: quality += "_conflicts"
    if missing or excluded: quality += "_partial"

    if mode == "japanese-only":
        policy = "muted"
    elif mode == "reference-mix":
        policy = "quiet English reference; no speech separation"
    elif mode in ("voiceover", "ducking"):
        policy = f"dynamic voice-over ducking (speech={reference_gain:.2f}, unspoken={unspoken_gain:.2f})"
    else:
        policy = f"separated ambient preservation ({ambient_backend}, ambient_gain={ambient_gain:.2f}, duck_gain={reference_gain:.2f})"

    extra_phase13 = {}
    if mode in ("voiceover", "ducking", "ambient", "hybrid-ambient"):
        extra_phase13 = {
            "unspoken_gain": float(unspoken_gain),
            "fade_out_seconds": float(fade_out_seconds),
            "fade_in_seconds": float(fade_in_seconds),
            "ambient_backend": str(ambient_backend),
            "ambient_gain": float(ambient_gain),
        }

    return {"format_version": 1, "kind": "audio_mix", "engine_version": ENGINE_VERSION,
            "video_id": video_id, "clip": feedback["clip"], "preview_seconds_requested": seconds,
            "duration_seconds": duration, "absolute_video_start": selection["start"],
            "absolute_video_end": selection["start"] + duration,
            "mode": mode, "diagnostic": diagnostic, "quality_label": quality,
            **({"timing_adjustment":timing_adjustment} if timing_adjustment else {}),
            **extra_phase13,
            "reference_gain": reference_gain if mode != "japanese-only" else 0,
            "source_audio_policy": policy,
            "sample_rate": RATE, "channels": CHANNELS, "sample_frames": frames,
            "placements": complete, "undubbed_entries": missing,
            "source_excluded": feedback.get("source_excluded", []), "excluded_intersecting_preview": excluded,
            "same_speaker_conflicts": same, "other_speaker_overlaps": other,
            "introduced_other_speaker_overlaps": [p for p in other if p["source_overlap_seconds"] <= 0],
            "review_entry_ids": review_ids, "boundary_crossing_ids": overflows,
            "extended": duration > seconds, "blockers": blockers,
            "inputs": {"feedback": str(Path(feedback_path).resolve()), "feedback_fingerprint": fingerprint(feedback),
                       "selection": str(Path(selection_path).resolve()), "selection_fingerprint": fingerprint(selection),
                       "script": str(script_path), "script_fingerprint": feedback["source_script_fingerprint"],
                       "source_video": str(video_path), "source_info": str(Path(info_path).resolve()),
                       "source_info_fingerprint": fingerprint(metadata)},
            "source_probe": video_info}


def float_samples(data):
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little": values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError("Non-finite audio samples.")
    return values


def peak(values):
    return max((abs(x) for x in values), default=0.0)


def speaker_audio(rows, frames):
    inputs, filters, labels = [], [], []
    for i, row in enumerate(rows):
        inputs += ["-i", row["audio_file"]]
        filters.append(f"[{i}:a:0]aresample={RATE},aformat=sample_fmts=flt:channel_layouts=stereo,"
                       f"adelay={row['placement_sample']}S:all=1[a{i}]")
        labels.append(f"[a{i}]")
    filters.append("".join(labels) + f"amix=inputs={len(rows)}:normalize=0:duration=longest,apad=whole_len={frames}[mix]")
    values = float_samples(ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[mix]",
                                  "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]))
    if len(values) != frames * CHANNELS:
        raise ValueError("Resampled speech exceeds planned duration or has an unexpected frame count; not truncated.")
    return values


def add_audio(buffers):
    if not buffers or len({len(b) for b in buffers}) != 1:
        raise ValueError("Cannot mix empty/unequal audio buffers.")
    return array("f", (sum(samples) for samples in zip(*buffers)))


def write_pcm(path, samples, gain):
    pcm = array("h", (round(x * gain * 32767) for x in samples))
    if sys.byteorder != "little": pcm.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((CHANNELS, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    return measure_wav(path)


def validate_mp4(path, duration):
    info = probe(path)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ValueError("Rendered MP4 must have one video and one audio stream.")
    # Container video duration is quantized to source frame boundaries; PCM timeline is exact.
    if abs(float(info["format"]["duration"]) - duration) > .05:
        raise ValueError("Rendered MP4 duration mismatch.")
    if abs(float(audio[0].get("duration", 0)) - duration) > .025:
        raise ValueError("Rendered audio duration mismatch.")
    ffmpeg(["-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    decoded = float_samples(ffmpeg(["-i", str(path), "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]))
    decoded_peak = peak(decoded)
    if decoded_peak >= 1:
        raise ValueError("Encoded AAC exceeds full scale; use a lower mix gain target.")
    return {"probe": info, "full_decode_passed": True, "decoded_audio_peak": decoded_peak,
            "decoded_clipped_samples": sum(abs(x) >= 1 for x in decoded)}


def render_plan(plan, output_dir):
    """Publish all outputs atomically; identical reruns verify and reuse files."""
    if plan["blockers"]:
        raise ValueError("Render blocked: " + ", ".join(plan["blockers"]))
    plan = copy.deepcopy(plan)
    for row in plan["placements"]:
        if row.get("voice_profile"):
            audio_request(row, row["tts"])  # Recheck current reference/authorization before rendering or reuse.
    source = Path(plan["inputs"]["source_video"])
    plan["inputs"]["source_video_sha256"] = sha256_file(source)
    plan["ffmpeg_version"] = run_command(["ffmpeg", "-version"]).decode().splitlines()[0]
    plan["implementation_sha256"] = sha256_file(__file__)
    key = fingerprint(plan)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / (plan["mode"].replace("-", "_") + "_" + plan["quality_label"] + "_" + key[:16])
    with (root / ".mixer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another mixer is using this output directory.") from error
        if target.exists():
            saved = read_json(target / "mix_report.json")
            if saved.get("mix_key") != key or not saved.get("artifacts"):
                raise ValueError("Existing mix report mismatch; retained unchanged.")
            for name, digest in saved["artifacts"].items():
                path = (target / name).resolve()
                if not path.is_relative_to(target) or sha256_file(path) != digest:
                    raise ValueError("Existing output changed/corrupt; retained unchanged.")
            return target / "mix_report.json", True
        temp = Path(tempfile.mkdtemp(prefix=".pending-mix-", dir=root))
        try:
            tracks = {}
            for speaker in sorted({r["speaker"] for r in plan["placements"]}):
                tracks[speaker] = speaker_audio([r for r in plan["placements"] if r["speaker"] == speaker], plan["sample_frames"])
            japanese = add_audio(list(tracks.values()))
            final = japanese
            ducking_report = None
            separation_report = None
            if plan["mode"] == "reference-mix":
                data = ffmpeg(["-ss", str(plan["absolute_video_start"]), "-i", source,
                               "-t", str(plan["duration_seconds"]), "-map", "0:a:0", "-vn",
                               "-af", f"aresample={RATE},apad=whole_len={plan['sample_frames']}",
                               "-ar", str(RATE), "-ac", str(CHANNELS), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
                reference = float_samples(data)
                if len(reference) != len(japanese): raise ValueError("Reference track duration mismatch.")
                reference = array("f", (v * plan["reference_gain"] for v in reference))
                final = add_audio([japanese, reference])
            elif plan["mode"] in ("voiceover", "ducking", "ambient", "hybrid-ambient"):
                data = ffmpeg(["-ss", str(plan["absolute_video_start"]), "-i", source,
                               "-t", str(plan["duration_seconds"]), "-map", "0:a:0", "-vn",
                               "-af", f"aresample={RATE},apad=whole_len={plan['sample_frames']}",
                               "-ar", str(RATE), "-ac", str(CHANNELS), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
                raw_ref = float_samples(data)
                if len(raw_ref) != len(japanese):
                    raise ValueError("Reference track duration mismatch.")
                src_stereo = np.frombuffer(raw_ref, dtype=np.float32).reshape(plan["sample_frames"], CHANNELS)
                jp_stereo = np.frombuffer(japanese, dtype=np.float32).reshape(plan["sample_frames"], CHANNELS)
                speech_wins = [
                    (r["placement_seconds"], r["placement_seconds"] + r["actual_tts_duration_seconds"])
                    for r in plan["placements"]
                ]
                env, ducking_report = build_ducking_envelope(
                    plan["sample_frames"],
                    sample_rate=RATE,
                    speech_intervals=speech_wins,
                    duck_gain=float(plan.get("reference_gain", 0.12)),
                    unspoken_gain=float(plan.get("unspoken_gain", 1.0)),
                    fade_out_seconds=float(plan.get("fade_out_seconds", 0.18)),
                    fade_in_seconds=float(plan.get("fade_in_seconds", 0.25)),
                )
                if plan["mode"] in ("ambient", "hybrid-ambient"):
                    sep = separate_ambient_and_speech(
                        src_stereo,
                        sample_rate=RATE,
                        backend=plan.get("ambient_backend", "auto"),
                        allow_fallback=True,
                    )
                    separation_report = {
                        k: v for k, v in sep.items() if k not in ("ambient", "speech")
                    }
                    amb_gain = float(plan.get("ambient_gain", 0.85))
                    residual_voice_gain = float(plan.get("reference_gain", 0.12)) if plan["mode"] == "hybrid-ambient" else float(plan.get("reference_gain", 0.12)) * 0.35
                    # During speech: separated ambient bed + ducked residual voice; during non-speech: smooth transition to original audio
                    speech_weight = (float(plan.get("unspoken_gain", 1.0)) - env) / max(1e-6, abs(float(plan.get("unspoken_gain", 1.0)) - float(plan.get("reference_gain", 0.12))))
                    speech_weight = np.clip(speech_weight, 0.0, 1.0)[:, None]
                    bg_during_speech = sep["ambient"] * amb_gain + sep["speech"] * residual_voice_gain
                    bg_unspoken = src_stereo * float(plan.get("unspoken_gain", 1.0))
                    bg_stereo = speech_weight * bg_during_speech + (1.0 - speech_weight) * bg_unspoken
                else:
                    bg_stereo = src_stereo * env[:, None]

                mixed_stereo = (jp_stereo + bg_stereo).astype(np.float32)
                final = array("f", mixed_stereo.reshape(-1).tolist())

            before_peak = max(peak(v) for v in [*tracks.values(), japanese, final])
            gain = min(1, PEAK_TARGET / before_peak) if before_peak else 1
            measurements = {}
            for speaker, samples in tracks.items():
                name = f"speaker_{speaker}.wav"
                measurements[name] = write_pcm(temp / name, samples, gain)
            prefix = f"preview_{plan['duration_seconds']:g}s"
            jp_name = prefix + "_japanese.wav"
            measurements[jp_name] = write_pcm(temp / jp_name, japanese, gain)
            mix_name = prefix + "_" + plan["mode"].replace("-", "_") + ".wav"
            if mix_name == jp_name: mix_name = prefix + "_mix.wav"
            measurements[mix_name] = write_pcm(temp / mix_name, final, gain)
            video_name = prefix + "_" + plan["mode"].replace("-", "_") + "_" + plan["quality_label"] + ".mp4"
            ffmpeg(["-ss", str(plan["absolute_video_start"]), "-i", source, "-i", temp / mix_name,
                    "-t", str(plan["duration_seconds"]), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-metadata", "title=TransClip " + plan["quality_label"],
                    "-movflags", "+faststart", temp / video_name])
            verification = validate_mp4(temp / video_name, plan["duration_seconds"])
            if float(verification["probe"]["format"]["duration"]) > plan["clip"]["end"] - plan["clip"]["start"] + 1e-6:
                raise ValueError("Encoded video frame boundary exceeds selection; output not published.")
            # Inputs must remain consistent throughout rendering.
            if sha256_file(source) != plan["inputs"]["source_video_sha256"]:
                raise ValueError("Source video changed during rendering.")
            for row in plan["placements"]:
                if sha256_file(row["audio_file"]) != row["wav_measurement"]["sha256"]:
                    raise ValueError("Source WAV changed during rendering.")
            for label in ("feedback", "selection", "script", "source_info", *(["speaker_analysis", "speaker_utterances"] if "speaker_analysis" in plan["inputs"] else [])):
                if fingerprint(read_json(plan["inputs"][label])) != plan["inputs"][label + "_fingerprint"]:
                    raise ValueError(f"Input {label} changed during rendering.")
            result = {**plan, "mix_key": key, "status": "diagnostic_preview" if plan["diagnostic"] else "rendered",
                      "video_file": video_name, "japanese_audio_file": jp_name, "mixed_audio_file": mix_name,
                      **({"ducking_envelope": ducking_report} if ducking_report else {}),
                      **({"ambient_separation": separation_report} if separation_report else {}),
                      "gain": {"linear": gain, "decibels": 20*math.log10(gain), "pre_gain_peak": before_peak,
                               "target_peak": PEAK_TARGET, "policy": "shared static attenuation; no time stretch or limiter"},
                      "wav_measurements": measurements, "verification": verification,
                      "artifacts": {p.name: sha256_file(p) for p in temp.iterdir() if p.is_file()}}
            atomic_json(temp / "mix_report.json", result, overwrite=False)
            os.rename(temp, target)
        finally:
            if temp.exists(): shutil.rmtree(temp)  # Only unpublished files created by this call.
    return target / "mix_report.json", False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Place Phase 2 WAVs on independent timelines and render a dubbed preview.")
    parser.add_argument("video_id")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--speaker-analysis", type=Path, help="Cross-check Phase 4 speaker evidence before mixing")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--mode", choices=VALID_MIX_MODES, default="japanese-only")
    parser.add_argument("--reference-gain", "--duck-gain", dest="reference_gain", type=float, default=.12)
    parser.add_argument("--unspoken-gain", type=float, default=1.0, help="Original audio gain during non-TTS intervals in voiceover/ambient modes")
    parser.add_argument("--fade-out-seconds", type=float, default=.18, help="Ducking fade-out duration before TTS starts")
    parser.add_argument("--fade-in-seconds", type=float, default=.25, help="Ducking fade-in duration after TTS ends")
    parser.add_argument("--ambient-backend", choices=("auto", "dsp", "demucs", "torchscript", "convtasnet", "none"), default="auto")
    parser.add_argument("--ambient-gain", type=float, default=.85)
    parser.add_argument("--diagnostic", action="store_true", help="Explicitly allow reported collisions and undubbed/review entries")
    parser.add_argument("--extend-to-fit", action="store_true", help="Extend only within selection to keep complete speech")
    parser.add_argument("--max-timing-shift",type=float,default=0,help="Opt-in adjacent same-speaker delay, 0..2 seconds; no time stretching")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan; generate no media")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.video_id): raise ValueError("Invalid video_id.")
        inputs = BASE_DIR / "input" / args.video_id
        plan = build_plan(args.feedback, args.selection or inputs / "selection.json",
                          args.source_video or inputs / "source.mp4", args.source_info or inputs / "source.info.json",
                          args.video_id, seconds=args.seconds, mode=args.mode, diagnostic=args.diagnostic,
                          extend_to_fit=args.extend_to_fit, reference_gain=args.reference_gain, max_timing_shift=args.max_timing_shift,
                          unspoken_gain=args.unspoken_gain, fade_out_seconds=args.fade_out_seconds, fade_in_seconds=args.fade_in_seconds,
                          ambient_backend=args.ambient_backend, ambient_gain=args.ambient_gain)
        if args.speaker_analysis:
            attach_speaker_evidence(plan, args.speaker_analysis)
        print(json.dumps({k: plan[k] for k in ("video_id", "quality_label", "duration_seconds", "same_speaker_conflicts",
                          "other_speaker_overlaps", "boundary_crossing_ids", "excluded_intersecting_preview", "blockers")}, ensure_ascii=False, indent=2))
        for row in plan["placements"]:
            print(f"{row['id']} {row['speaker']} / {row['voice']} @ {row['placement_seconds']:.6f}s + {row['actual_tts_duration_seconds']:.6f}s")
        for row in plan["undubbed_entries"]:
            print(f"UNDUBBED {row['id']} {row['speaker'] or 'UNKNOWN'} @ {row['relative_start']}..{row['relative_end']}: {row.get('skip_reason') or row['generation_status']}")
        if args.dry_run: return 0
        path, reused = render_plan(plan, args.output_dir or BASE_DIR / "output" / args.video_id / "mixes")
        print(f"{'Reused verified output' if reused else 'Rendered and verified'}: {path}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
