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
    "continuous-voiceover",
    "long-release-voiceover",
    "ambient",
    "hybrid-ambient",
)
VALID_DUCKING_STYLES = ("dynamic", "continuous", "long-release")


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
# Phase 13 / 13.3: Dynamic Audio Ducking (Voice-Over), Hold Control,
#                  Ambient Separation & ITU-R BS.1770-4 LUFS Normalization
# ============================================================================

DEFAULT_DUCK_HOLD_SECONDS = 3.2
DEFAULT_SAME_SPEAKER_HOLD_SECONDS = 4.0
DEFAULT_TARGET_LUFS = -16.0


def apply_ducking_hold_control(
    intervals,
    *,
    total_duration_seconds=None,
    pre_duck_seconds=0.06,
    post_duck_seconds=0.08,
    min_unspoken_gap_seconds=0.35,
    hold_seconds=DEFAULT_DUCK_HOLD_SECONDS,
    same_speaker_hold_seconds=DEFAULT_SAME_SPEAKER_HOLD_SECONDS,
):
    """Normalize and merge active Japanese TTS speech intervals with Phase 13.3 Ducking Hold Control.

    Bridges inter-utterance gaps and maintains the ducked state when:
      1. The gap to the next TTS start is <= `hold_seconds` (default 3.2s, covering 3-4s gaps), OR
      2. Consecutive utterances belong to the same speaker (`speaker_a == speaker_b`) and the gap is
         <= `same_speaker_hold_seconds` (default 4.0s).
    Returns `{"merged_windows": [...], "raw_windows": [...], "held_gaps": [...]}`.
    """
    cleaned = []
    for idx, item in enumerate(intervals or []):
        spk = None
        uid = None
        if isinstance(item, dict):
            a = float(item.get("start", item.get("placement_seconds", item.get("clip_relative_start_seconds", 0.0))))
            b = float(item.get("end", a + float(item.get("duration", item.get("actual_tts_duration_seconds", 0.0)))))
            spk = item.get("speaker")
            uid = item.get("id", item.get("utterance_id", idx))
        else:
            a, b = float(item[0]), float(item[1])
            if len(item) >= 3:
                spk = item[2]
            if len(item) >= 4:
                uid = item[3]
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
            continue
        expanded_a = max(0.0, a - max(0.0, float(pre_duck_seconds)))
        expanded_b = b + max(0.0, float(post_duck_seconds))
        if total_duration_seconds is not None and total_duration_seconds > 0:
            expanded_a = min(float(total_duration_seconds), expanded_a)
            expanded_b = min(float(total_duration_seconds), expanded_b)
        if expanded_b > expanded_a:
            cleaned.append({
                "start": expanded_a,
                "end": expanded_b,
                "raw_start": a,
                "raw_end": b,
                "speaker": str(spk).strip() if spk not in (None, "") else None,
                "id": uid,
            })

    if not cleaned:
        return {"merged_windows": [], "raw_windows": [], "held_gaps": []}

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    base_bridge = max(0.0, float(min_unspoken_gap_seconds or 0.0))
    general_hold = max(base_bridge, float(hold_seconds)) if hold_seconds is not None else base_bridge
    same_spk_hold = (
        max(general_hold, float(same_speaker_hold_seconds))
        if same_speaker_hold_seconds is not None
        else general_hold
    )

    merged = [[cleaned[0]["start"], cleaned[0]["end"], cleaned[0]["speaker"], cleaned[0]["id"], cleaned[0]["raw_end"]]]
    held_gaps = []

    for cur in cleaned[1:]:
        prev_a, prev_b, prev_spk, prev_id, prev_raw_end = merged[-1]
        raw_gap = max(0.0, cur["raw_start"] - prev_raw_end)
        exp_gap = cur["start"] - prev_b
        is_same_spk = bool(prev_spk and cur["speaker"] and prev_spk == cur["speaker"])

        if hold_seconds is not None:
            eff_bridge = same_spk_hold if is_same_spk else general_hold
        elif is_same_spk and same_speaker_hold_seconds is not None:
            eff_bridge = float(same_speaker_hold_seconds)
        else:
            eff_bridge = base_bridge

        if exp_gap <= eff_bridge + 1e-6 or raw_gap <= eff_bridge + 1e-6:
            if raw_gap > base_bridge + 1e-6:
                reason = "same_speaker_continuous_hold" if is_same_spk else "short_gap_hold_within_threshold"
                held_gaps.append({
                    "prev_id": prev_id,
                    "next_id": cur["id"],
                    "prev_speaker": prev_spk,
                    "next_speaker": cur["speaker"],
                    "same_speaker": is_same_spk,
                    "gap_start_seconds": round(prev_raw_end, 4),
                    "gap_end_seconds": round(cur["raw_start"], 4),
                    "gap_duration_seconds": round(raw_gap, 4),
                    "hold_threshold_seconds": round(eff_bridge, 4),
                    "reason": reason,
                })
            merged[-1][1] = max(prev_b, cur["end"])
            merged[-1][2] = cur["speaker"] or prev_spk
            merged[-1][3] = cur["id"]
            merged[-1][4] = max(prev_raw_end, cur["raw_end"])
        else:
            merged.append([cur["start"], cur["end"], cur["speaker"], cur["id"], cur["raw_end"]])

    return {
        "merged_windows": [(round(m[0], 6), round(m[1], 6)) for m in merged],
        "raw_windows": [(round(c["raw_start"], 6), round(c["raw_end"], 6)) for c in cleaned],
        "held_gaps": held_gaps,
    }


def merge_speech_intervals(
    intervals,
    *,
    total_duration_seconds=None,
    pre_duck_seconds=0.06,
    post_duck_seconds=0.08,
    min_unspoken_gap_seconds=0.35,
    hold_seconds=None,
    same_speaker_hold_seconds=None,
):
    """Normalize and merge active Japanese TTS speech intervals, bridging micro-gaps and optional hold windows."""
    res = apply_ducking_hold_control(
        intervals,
        total_duration_seconds=total_duration_seconds,
        pre_duck_seconds=pre_duck_seconds,
        post_duck_seconds=post_duck_seconds,
        min_unspoken_gap_seconds=min_unspoken_gap_seconds,
        hold_seconds=hold_seconds,
        same_speaker_hold_seconds=same_speaker_hold_seconds,
    )
    return res["merged_windows"]


def build_ducking_envelope(
    total_frames=None,
    *,
    total_samples=None,
    sample_rate=RATE,
    speech_intervals=(),
    duck_gain=0.12,
    unspoken_gain=1.0,
    fade_out_seconds=0.18,
    fade_in_seconds=0.25,
    release_seconds=None,
    pre_duck_seconds=0.06,
    post_duck_seconds=0.08,
    min_unspoken_gap_seconds=0.35,
    hold_seconds=None,
    same_speaker_hold_seconds=None,
    hold_gap_gain=None,
    curve="cosine",
    ducking_style="dynamic",
):
    """Build a sample-accurate 1D gain envelope for voice-over ducking with Phase 13.3 Hold Control.

    Supports three `ducking_style` modes:
    - `'dynamic'` (Phase 13 default): attenuates original audio to `duck_gain` during TTS (and across held
      inter-utterance gaps <= `hold_seconds` or `same_speaker_hold_seconds`) and returns to `unspoken_gain`
      during longer non-TTS intervals with smooth cosine fades.
    - `'continuous'` (Phase 13.1/13.3 stabilized mode): suppresses original audio across the ENTIRE video
      at `duck_gain`, and when `hold_gap_gain` (or `hold_seconds`) is enabled, gently deepens attenuation
      during unmasked inter-utterance hesitation gaps (such as 0:12-0:14) so English hesitations never jump out.
    - `'long-release'` (Phase 13.1 extended release mode): extends release time (`fade_in_seconds`) to
      >= 1.5s (default 1.8s), bridges gaps, and caps unspoken return level so volume transitions remain gradual.
    """
    if total_frames is None and total_samples is not None:
        total_frames = total_samples
    if type(total_frames) is not int or total_frames <= 0:
        raise ValueError("total_frames must be a positive integer.")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if ducking_style not in VALID_DUCKING_STYLES:
        raise ValueError(f"Unsupported ducking_style {ducking_style!r}; expected {VALID_DUCKING_STYLES}")

    effective_fade_in = float(release_seconds) if release_seconds is not None else float(fade_in_seconds)
    effective_unspoken_gain = float(unspoken_gain)
    effective_min_gap = float(min_unspoken_gap_seconds)

    if ducking_style == "continuous":
        if abs(effective_unspoken_gain - 1.0) < 1e-6:
            effective_unspoken_gain = float(duck_gain)
        else:
            effective_unspoken_gain = min(float(effective_unspoken_gain), max(float(duck_gain), 0.22))
    elif ducking_style == "long-release":
        effective_fade_in = max(1.5, effective_fade_in)
        effective_min_gap = max(1.2, effective_min_gap)
        if abs(effective_unspoken_gain - 1.0) < 1e-6:
            effective_unspoken_gain = min(0.32, max(float(duck_gain) * 2.0, 0.24))

    for label, val, lo, hi in [
        ("duck_gain", duck_gain, 0.0, 0.5),
        ("unspoken_gain", effective_unspoken_gain, 0.0, 1.5),
        ("fade_out_seconds", fade_out_seconds, 0.0, 5.0),
        ("fade_in_seconds", effective_fade_in, 0.0, 10.0),
        ("pre_duck_seconds", pre_duck_seconds, 0.0, 2.0),
        ("post_duck_seconds", post_duck_seconds, 0.0, 2.0),
        ("min_unspoken_gap_seconds", effective_min_gap, 0.0, 15.0),
    ]:
        if type(val) not in (int, float) or not math.isfinite(val) or not (lo <= val <= hi):
            raise ValueError(f"Invalid {label}: {val}")
    if curve not in ("cosine", "linear"):
        raise ValueError("curve must be 'cosine' or 'linear'.")

    total_duration = total_frames / float(sample_rate)
    hold_res = apply_ducking_hold_control(
        speech_intervals,
        total_duration_seconds=total_duration,
        pre_duck_seconds=pre_duck_seconds,
        post_duck_seconds=post_duck_seconds,
        min_unspoken_gap_seconds=effective_min_gap,
        hold_seconds=hold_seconds,
        same_speaker_hold_seconds=same_speaker_hold_seconds,
    )
    merged = hold_res["merged_windows"]
    held_gaps = hold_res["held_gaps"]

    duck_g = float(duck_gain)
    unspoken_g = float(effective_unspoken_gain)
    env = np.full(total_frames, unspoken_g, dtype=np.float32)

    fade_out_frames = max(1, int(round(float(fade_out_seconds) * sample_rate))) if fade_out_seconds > 0 else 0
    fade_in_frames = max(1, int(round(float(effective_fade_in) * sample_rate))) if effective_fade_in > 0 else 0

    if merged and abs(unspoken_g - duck_g) >= 1e-7:
        for a_sec, b_sec in merged:
            s_idx = max(0, min(total_frames, int(round(a_sec * sample_rate))))
            e_idx = max(s_idx, min(total_frames, int(round(b_sec * sample_rate))))
            if e_idx > s_idx:
                env[s_idx:e_idx] = np.minimum(env[s_idx:e_idx], duck_g)

            if fade_out_frames > 0 and s_idx > 0:
                ramp_start = max(0, s_idx - fade_out_frames)
                ramp_len = s_idx - ramp_start
                if ramp_len > 0:
                    t = np.linspace(0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32)
                    w = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32) if curve == "cosine" else t
                    ramp_vals = unspoken_g - (unspoken_g - duck_g) * w
                    env[ramp_start:s_idx] = np.minimum(env[ramp_start:s_idx], ramp_vals)

            if fade_in_frames > 0 and e_idx < total_frames:
                ramp_end = min(total_frames, e_idx + fade_in_frames)
                ramp_len = ramp_end - e_idx
                if ramp_len > 0:
                    t = np.linspace(0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32)
                    w = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32) if curve == "cosine" else t
                    ramp_vals = duck_g + (unspoken_g - duck_g) * w
                    env[e_idx:ramp_end] = np.minimum(env[e_idx:ramp_end], ramp_vals)

    # Phase 13.3: Optional deeper hold attenuation during unmasked inter-utterance gaps (e.g. 0:12-0:14 hesitation)
    eff_hold_gap_gain = float(hold_gap_gain) if hold_gap_gain is not None else duck_g
    if held_gaps and eff_hold_gap_gain < duck_g - 1e-5:
        ramp_frames = max(1, int(round(0.12 * sample_rate)))
        for hg in held_gaps:
            gs_idx = max(0, min(total_frames, int(round(float(hg["gap_start_seconds"]) * sample_rate))))
            ge_idx = max(gs_idx, min(total_frames, int(round(float(hg["gap_end_seconds"]) * sample_rate))))
            if ge_idx - gs_idx <= 2 * ramp_frames:
                env[gs_idx:ge_idx] = np.minimum(env[gs_idx:ge_idx], eff_hold_gap_gain)
            else:
                t_r = np.linspace(0.0, 1.0, ramp_frames, endpoint=False, dtype=np.float32)
                w_r = (0.5 * (1.0 - np.cos(np.pi * t_r))).astype(np.float32)
                down_ramp = duck_g - (duck_g - eff_hold_gap_gain) * w_r
                up_ramp = eff_hold_gap_gain + (duck_g - eff_hold_gap_gain) * w_r
                env[gs_idx : gs_idx + ramp_frames] = np.minimum(env[gs_idx : gs_idx + ramp_frames], down_ramp)
                env[gs_idx + ramp_frames : ge_idx - ramp_frames] = np.minimum(
                    env[gs_idx + ramp_frames : ge_idx - ramp_frames],
                    eff_hold_gap_gain,
                )
                env[ge_idx - ramp_frames : ge_idx] = np.minimum(env[ge_idx - ramp_frames : ge_idx], up_ramp)

    ducked_mask = env <= duck_g + 0.02 * max(1.0, abs(unspoken_g - duck_g))
    unspoken_mask = np.abs(env - unspoken_g) <= 0.02 * max(1.0, abs(unspoken_g - duck_g))
    return env, {
        "ducking_style": ducking_style,
        "duck_gain": duck_g,
        "unspoken_gain": unspoken_g,
        "hold_seconds": float(hold_seconds) if hold_seconds is not None else None,
        "same_speaker_hold_seconds": float(same_speaker_hold_seconds) if same_speaker_hold_seconds is not None else None,
        "hold_gap_gain": round(eff_hold_gap_gain, 4),
        "held_gap_count": len(held_gaps),
        "held_gaps": held_gaps,
        "fade_out_seconds": float(fade_out_seconds),
        "fade_in_seconds": float(effective_fade_in),
        "release_seconds": float(effective_fade_in),
        "pre_duck_seconds": float(pre_duck_seconds),
        "post_duck_seconds": float(post_duck_seconds),
        "min_unspoken_gap_seconds": float(effective_min_gap),
        "curve": curve,
        "merged_speech_windows": merged,
        "transition_count": 0 if (not merged or abs(unspoken_g - duck_g) < 1e-7) else len(merged) * 2,
        "ducked_frame_fraction": round(float(np.mean(ducked_mask)), 4) if total_frames > 0 else 0.0,
        "unspoken_frame_fraction": (
            0.0
            if abs(unspoken_g - duck_g) < 1e-7
            else round(float(np.mean(unspoken_mask)), 4)
        ),
        "min_gain": round(float(np.min(env)), 4),
        "max_gain": round(float(np.max(env)), 4),
        "prevents_100pct_return": bool(unspoken_g <= 0.35),
    }


def _apply_bs1770_k_weighting(mono_samples, sample_rate=RATE):
    """Apply exact ITU-R BS.1770-4 two-stage K-weighting filter (Stage 1 high-shelf +4dB above 1.5kHz
    + Stage 2 RLB high-pass at 38Hz) via fast FFT transfer function evaluation.
    """
    x = np.asarray(mono_samples, dtype=np.float64)
    n = len(x)
    if n < 8:
        return x.copy()
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    # Stage 1: Acoustic head/ear high-shelf filter (+4.0 dB above ~1682 Hz, Q=0.707175)
    f0_shelf = 1681.974450955533
    g_linear = 10.0 ** (3.999843853973347 / 20.0)
    s_norm = freqs / f0_shelf
    shelf_mag_sq = (1.0 + (g_linear ** 2) * (s_norm ** 4) + 2.0 * g_linear * (s_norm ** 2)) / (
        1.0 + (s_norm ** 4) + 2.0 * (s_norm ** 2)
    )
    # Stage 2: RLB high-pass weighting filter (f0 = 38.13547 Hz, Q = 0.500327)
    f0_hp = 38.13547087602444
    hp_norm_sq = (freqs / f0_hp) ** 2
    hp_mag_sq = (hp_norm_sq ** 2) / (1.0 + (hp_norm_sq ** 2) + 2.0 * hp_norm_sq)
    hp_mag_sq[0] = 0.0

    k_mag = np.sqrt(shelf_mag_sq * hp_mag_sq)
    spec = np.fft.rfft(x)
    y_k = np.fft.irfft(spec * k_mag, n=n)
    return y_k


def measure_audio_lufs(samples, sample_rate=RATE):
    """Measure ITU-R BS.1770-4 K-weighted loudness (Integrated LUFS, Momentary Max LUFS, Active Speech LUFS),
    RMS dBFS, and True Peak dBFS for mono or stereo audio arrays.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {
            "integrated_lufs": -70.0,
            "momentary_max_lufs": -70.0,
            "active_speech_lufs": -70.0,
            "rms_dbfs": -70.0,
            "peak_dbfs": -70.0,
            "peak_amplitude": 0.0,
            "high_freq_presence_ratio": 0.0,
            "duration_seconds": 0.0,
        }
    mono = np.mean(arr, axis=1) if arr.ndim == 2 else arr
    n = len(mono)
    dur_sec = n / float(max(1, sample_rate))
    peak_amp = float(np.max(np.abs(arr)))
    rms_amp = float(np.sqrt(np.mean(np.square(mono))))
    peak_dbfs = round(20.0 * math.log10(max(peak_amp, 1e-9)), 2)
    rms_dbfs = round(20.0 * math.log10(max(rms_amp, 1e-9)), 2)

    y_k = _apply_bs1770_k_weighting(mono, sample_rate=sample_rate)
    sq_k = np.square(y_k)

    # High-frequency presence energy ratio (1.5 kHz .. 4.5 kHz vs total)
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    tot_e = float(np.sum(spec))
    pres_e = float(np.sum(spec[(freqs >= 1400.0) & (freqs <= 4500.0)]))
    presence_ratio = round(pres_e / max(tot_e, 1e-12), 4)

    # Gated block loudness (400ms BS.1770 blocks for >=0.5s audio; 100ms blocks for short utterances)
    block_sec = 0.400 if dur_sec >= 0.60 else max(0.050, dur_sec * 0.40)
    step_sec = block_sec * 0.25
    b_len = max(1, int(round(block_sec * sample_rate)))
    s_len = max(1, int(round(step_sec * sample_rate)))

    block_powers = []
    for pos in range(0, max(1, n - b_len + 1), s_len):
        blk = sq_k[pos : pos + b_len]
        if blk.size > 0:
            block_powers.append(float(np.mean(blk)))
    if not block_powers:
        block_powers = [float(np.mean(sq_k))]

    bp_arr = np.asarray(block_powers, dtype=np.float64)
    # Absolute gate at -70 LKFS
    abs_gate_p = 10.0 ** ((-70.0 + 0.691) / 10.0)
    above_abs = bp_arr[bp_arr > abs_gate_p]
    if above_abs.size == 0:
        integrated_lufs = -70.0
    else:
        rel_thresh_p = float(np.mean(above_abs)) * (10.0 ** (-10.0 / 10.0))
        above_rel = above_abs[above_abs >= rel_thresh_p]
        gated_mean_p = float(np.mean(above_rel)) if above_rel.size > 0 else float(np.mean(above_abs))
        integrated_lufs = round(-0.691 + 10.0 * math.log10(max(gated_mean_p, 1e-12)), 2)

    # Short-window (150ms) active speech & momentary max LUFS (critical for short utterances like "あらゆることに？")
    short_len = max(1, int(round(min(0.200, max(0.040, dur_sec * 0.35)) * sample_rate)))
    short_step = max(1, short_len // 4)
    short_powers = [
        float(np.mean(sq_k[p : p + short_len]))
        for p in range(0, max(1, n - short_len + 1), short_step)
    ] or [float(np.mean(sq_k))]
    sp_arr = np.asarray(short_powers, dtype=np.float64)
    max_p = float(np.max(sp_arr))
    momentary_max_lufs = round(-0.691 + 10.0 * math.log10(max(max_p, 1e-12)), 2)

    active_sp = sp_arr[sp_arr >= max_p * 0.15]
    active_p = float(np.mean(active_sp)) if active_sp.size > 0 else max_p
    active_speech_lufs = round(-0.691 + 10.0 * math.log10(max(active_p, 1e-12)), 2)

    return {
        "integrated_lufs": integrated_lufs,
        "momentary_max_lufs": momentary_max_lufs,
        "active_speech_lufs": active_speech_lufs,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "peak_amplitude": round(peak_amp, 6),
        "high_freq_presence_ratio": presence_ratio,
        "duration_seconds": round(dur_sec, 4),
    }


def normalize_tts_samples_lufs(
    samples,
    sample_rate=RATE,
    *,
    target_lufs=DEFAULT_TARGET_LUFS,
    max_gain_db=12.0,
    min_gain_db=-12.0,
    ceiling=PEAK_TARGET,
):
    """Normalize a TTS utterance waveform to `target_lufs` (default -16.0 LUFS) using linear K-weighted
    BS.1770-4 gain + soft true-peak limiting so multi-speaker (male/female) loudness matches without
    distorting intra-sentence prosody or clipping.
    """
    arr = np.asarray(samples, dtype=np.float64)
    before = measure_audio_lufs(arr, sample_rate=sample_rate)
    if arr.size == 0 or before["integrated_lufs"] <= -65.0:
        return arr.astype(np.float32), {
            "applied_gain_db": 0.0,
            "linear_gain": 1.0,
            "target_lufs": float(target_lufs),
            "before": before,
            "after": before,
            "limiter": {"pre_peak": 0.0, "post_peak": 0.0, "limited_samples": 0},
        }

    # For short utterances (<1.6s, e.g. "あらゆることに？" at 0:10), weight active/momentary K-weighted loudness
    # and high-frequency presence so short female interrogatives never jump out ahead of sustained speech.
    dur = before["duration_seconds"]
    if dur < 1.6:
        effective_lufs = 0.55 * before["integrated_lufs"] + 0.45 * before["active_speech_lufs"]
    else:
        effective_lufs = 0.80 * before["integrated_lufs"] + 0.20 * before["active_speech_lufs"]

    raw_gain_db = float(target_lufs) - effective_lufs
    gain_db = float(np.clip(raw_gain_db, float(min_gain_db), float(max_gain_db)))
    linear_gain = float(10.0 ** (gain_db / 20.0))

    scaled = arr * linear_gain
    limited, lim_report = apply_soft_limiter(scaled, ceiling=float(ceiling), knee=0.72)
    after = measure_audio_lufs(limited, sample_rate=sample_rate)

    return limited.astype(np.float32), {
        "target_lufs": round(float(target_lufs), 2),
        "effective_source_lufs": round(effective_lufs, 2),
        "applied_gain_db": round(gain_db, 3),
        "linear_gain": round(linear_gain, 5),
        "before": before,
        "after": after,
        "limiter": lim_report,
        "clipping_free": bool(float(np.max(np.abs(limited))) < 1.0),
    }


def normalize_tts_wav_lufs(
    input_wav_path,
    output_wav_path=None,
    *,
    target_lufs=DEFAULT_TARGET_LUFS,
    method="bs1770_linear",
    ceiling=PEAK_TARGET,
    ffmpeg_bin=None,
):
    """Normalize a TTS WAV file on disk to `target_lufs` (-16.0 or -18.0 LUFS), supporting both
    built-in BS.1770-4 linear K-weighted normalization (`'bs1770_linear'`) and FFmpeg `loudnorm` (`'ffmpeg_loudnorm'`).
    """
    in_p = Path(input_wav_path).resolve()
    out_p = Path(output_wav_path).resolve() if output_wav_path else in_p
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(in_p), "rb") as wf:
        sr = int(wf.getframerate() or 24000)
        n_ch = int(wf.getnchannels() or 1)
        n_frames = int(wf.getnframes() or 0)
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    if method == "ffmpeg_loudnorm":
        before = measure_audio_lufs(samples, sample_rate=sr)
        tmp_out = out_p.with_suffix(".loudnorm.tmp.wav")
        tp_db = round(20.0 * math.log10(max(float(ceiling), 0.1)), 2)
        cmd = [
            ffmpeg_bin or "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(in_p),
            "-af",
            f"loudnorm=I={float(target_lufs):.1f}:TP={tp_db:.1f}:LRA=11",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(tmp_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and tmp_out.is_file():
            with wave.open(str(tmp_out), "rb") as wf2:
                raw2 = wf2.readframes(wf2.getnframes())
            norm_samples = np.frombuffer(raw2, dtype="<i2").astype(np.float32) / 32768.0
            norm_samples, lim_rep = apply_soft_limiter(norm_samples, ceiling=float(ceiling), knee=0.72)
            pcm = np.clip(np.round(norm_samples * 32767.0), -32768, 32767).astype("<i2")
            with wave.open(str(out_p), "wb") as wf_out:
                wf_out.setnchannels(1)
                wf_out.setsampwidth(2)
                wf_out.setframerate(sr)
                wf_out.writeframes(pcm.tobytes())
            tmp_out.unlink(missing_ok=True)
            after = measure_audio_lufs(norm_samples, sample_rate=sr)
            return out_p, {
                "method": "ffmpeg_loudnorm",
                "target_lufs": round(float(target_lufs), 2),
                "applied_gain_db": round(after["integrated_lufs"] - before["integrated_lufs"], 3),
                "before": before,
                "after": after,
                "limiter": lim_rep,
                "clipping_free": True,
            }

    norm_samples, rep = normalize_tts_samples_lufs(
        samples,
        sample_rate=sr,
        target_lufs=target_lufs,
        ceiling=ceiling,
    )
    pcm = np.clip(np.round(norm_samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(out_p), "wb") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sr)
        wf_out.writeframes(pcm.tobytes())
    rep["method"] = "bs1770_linear"
    rep["output_wav"] = str(out_p)
    return out_p, rep


def build_lufs_normalization_report(utterance_records, *, target_lufs=DEFAULT_TARGET_LUFS):
    """Build Phase 13.3 verification report for multi-speaker LUFS loudness normalization."""
    records = list(utterance_records or [])
    by_spk_before = {}
    by_spk_after = {}
    all_before = []
    all_after = []

    for r in records:
        spk = str(r.get("speaker") or "SPEAKER_01")
        b_lufs = float(r["before"]["integrated_lufs"])
        a_lufs = float(r["after"]["integrated_lufs"])
        by_spk_before.setdefault(spk, []).append(b_lufs)
        by_spk_after.setdefault(spk, []).append(a_lufs)
        all_before.append(b_lufs)
        all_after.append(a_lufs)

    spk_summary = {}
    for spk in sorted(by_spk_before.keys()):
        b_vals = by_spk_before[spk]
        a_vals = by_spk_after[spk]
        spk_summary[spk] = {
            "utterance_count": len(b_vals),
            "mean_lufs_before": round(float(np.mean(b_vals)), 2),
            "mean_lufs_after": round(float(np.mean(a_vals)), 2),
            "max_lufs_before": round(float(np.max(b_vals)), 2),
            "max_lufs_after": round(float(np.max(a_vals)), 2),
        }

    spk_means_before = [v["mean_lufs_before"] for v in spk_summary.values()]
    spk_means_after = [v["mean_lufs_after"] for v in spk_summary.values()]
    inter_spk_spread_before = round(max(spk_means_before) - min(spk_means_before), 2) if len(spk_means_before) >= 2 else 0.0
    inter_spk_spread_after = round(max(spk_means_after) - min(spk_means_after), 2) if len(spk_means_after) >= 2 else 0.0

    utt_range_before = round(max(all_before) - min(all_before), 2) if len(all_before) >= 2 else 0.0
    utt_range_after = round(max(all_after) - min(all_after), 2) if len(all_after) >= 2 else 0.0

    return {
        "kind": "transclip_phase13_3_lufs_normalization_report",
        "format_version": 1,
        "target_lufs": round(float(target_lufs), 2),
        "utterance_count": len(records),
        "speakers": spk_summary,
        "inter_speaker_lufs_spread_before_lu": inter_spk_spread_before,
        "inter_speaker_lufs_spread_after_lu": inter_spk_spread_after,
        "utterance_lufs_range_before_lu": utt_range_before,
        "utterance_lufs_range_after_lu": utt_range_after,
        "utterances": records,
        "all_clipping_free": all(bool(r.get("clipping_free", True)) for r in records),
        "loudness_disparity_reduced": bool(inter_spk_spread_after <= inter_spk_spread_before + 1e-4 and utt_range_after <= utt_range_before + 1e-4),
    }


def build_ducking_verification_report(
    ducking_envelope_report,
    loudness_verification,
    *,
    source_samples=None,
    background_samples=None,
    sample_rate=RATE,
    focus_window_seconds=(12.0, 14.5),
):
    """Build Phase 13.3 verification report for inter-utterance ducking hold control (including 0:12-0:14 check)."""
    w_start, w_end = float(focus_window_seconds[0]), float(focus_window_seconds[1])
    window_metrics = None
    if source_samples is not None and background_samples is not None:
        src_arr = np.asarray(source_samples, dtype=np.float64)
        bg_arr = np.asarray(background_samples, dtype=np.float64)
        src_mono = np.mean(src_arr, axis=1) if src_arr.ndim == 2 else src_arr
        bg_mono = np.mean(bg_arr, axis=1) if bg_arr.ndim == 2 else bg_arr
        s_i = max(0, min(len(src_mono), int(round(w_start * sample_rate))))
        e_i = max(s_i, min(len(src_mono), int(round(w_end * sample_rate))))
        if e_i > s_i:
            src_rms = float(np.sqrt(np.mean(np.square(src_mono[s_i:e_i]))))
            bg_rms = float(np.sqrt(np.mean(np.square(bg_mono[s_i:e_i]))))
            ratio = round(bg_rms / max(src_rms, 1e-8), 4) if src_rms > 1e-6 else 0.0
            att_db = round(20.0 * math.log10(max(ratio, 1e-6)), 2) if ratio > 0 else -60.0
            window_metrics = {
                "window_seconds": [round(w_start, 2), round(w_end, 2)],
                "source_rms": round(src_rms, 6),
                "ducked_background_rms": round(bg_rms, 6),
                "effective_gain_ratio": ratio,
                "attenuation_db": att_db,
                "english_prominence_suppressed": bool(ratio <= 0.20),
            }

    return {
        "kind": "transclip_phase13_3_ducking_verification_report",
        "format_version": 1,
        "ducking_style": (ducking_envelope_report or {}).get("ducking_style"),
        "duck_gain": (ducking_envelope_report or {}).get("duck_gain"),
        "hold_seconds": (ducking_envelope_report or {}).get("hold_seconds"),
        "same_speaker_hold_seconds": (ducking_envelope_report or {}).get("same_speaker_hold_seconds"),
        "hold_gap_gain": (ducking_envelope_report or {}).get("hold_gap_gain"),
        "held_gap_count": (ducking_envelope_report or {}).get("held_gap_count", 0),
        "held_gaps": (ducking_envelope_report or {}).get("held_gaps", []),
        "merged_speech_windows": (ducking_envelope_report or {}).get("merged_speech_windows", []),
        "prevents_100pct_return": bool((ducking_envelope_report or {}).get("prevents_100pct_return", True)),
        "focus_window_0_12_to_0_14_verification": window_metrics,
        "loudness_verification": loudness_verification,
    }


def apply_boundary_audio_fade(samples, sample_rate=RATE, *, fade_in_seconds=0.0, fade_out_seconds=0.40):
    """Apply smooth raised-cosine fade-in at clip start and fade-out at clip end to prevent abrupt boundary cuts."""
    arr = np.asarray(samples, dtype=np.float32).copy()
    n_frames = arr.shape[0] if arr.ndim >= 1 else 0
    if n_frames == 0:
        return arr
    fi_frames = min(n_frames // 2, max(0, int(round(float(fade_in_seconds or 0.0) * sample_rate))))
    fo_frames = min(n_frames // 2, max(0, int(round(float(fade_out_seconds or 0.0) * sample_rate))))
    if fi_frames > 0:
        t = np.linspace(0.0, 1.0, fi_frames, endpoint=False, dtype=np.float32)
        w = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
        if arr.ndim == 2:
            arr[:fi_frames] *= w[:, None]
        else:
            arr[:fi_frames] *= w
    if fo_frames > 0:
        t = np.linspace(0.0, 1.0, fo_frames, endpoint=False, dtype=np.float32)
        w = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
        if arr.ndim == 2:
            arr[n_frames - fo_frames :] *= w[:, None]
        else:
            arr[n_frames - fo_frames :] *= w
    return arr


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
    bg_unspoken_rms = None
    src_speech_rms = None
    src_unspoken_rms = None
    measured_duck_ratio = None
    measured_duck_ratio_unspoken = None
    max_bg_jump_db = 0.0
    continuous_ducking_verified = None

    if background_samples is not None:
        bg_arr = np.asarray(background_samples, dtype=np.float64)
        bg_mono = np.mean(bg_arr, axis=1) if bg_arr.ndim == 2 else bg_arr
        if len(bg_mono) == n_frames:
            if np.any(speech_mask):
                bg_speech_rms = float(np.sqrt(np.mean(np.square(bg_mono[speech_mask]))))
            if np.any(unspoken_mask):
                bg_unspoken_rms = float(np.sqrt(np.mean(np.square(bg_mono[unspoken_mask]))))

    if source_samples is not None:
        src_arr = np.asarray(source_samples, dtype=np.float64)
        src_mono = np.mean(src_arr, axis=1) if src_arr.ndim == 2 else src_arr
        if len(src_mono) == n_frames:
            if np.any(speech_mask):
                src_speech_rms = float(np.sqrt(np.mean(np.square(src_mono[speech_mask]))))
                if bg_speech_rms is not None and src_speech_rms > 1e-7:
                    measured_duck_ratio = round(bg_speech_rms / src_speech_rms, 4)
            if np.any(unspoken_mask):
                src_unspoken_rms = float(np.sqrt(np.mean(np.square(src_mono[unspoken_mask]))))
                if bg_unspoken_rms is not None and src_unspoken_rms > 1e-7:
                    measured_duck_ratio_unspoken = round(bg_unspoken_rms / src_unspoken_rms, 4)

            if background_samples is not None and len(bg_mono) == n_frames:
                win_len = max(1, int(round(0.25 * sample_rate)))
                win_ratios = []
                for w_i in range(0, n_frames, win_len):
                    s_slice = src_mono[w_i : w_i + win_len]
                    b_slice = bg_mono[w_i : w_i + win_len]
                    s_r = float(np.sqrt(np.mean(np.square(s_slice))))
                    if s_r > 1e-5:
                        b_r = float(np.sqrt(np.mean(np.square(b_slice))))
                        win_ratios.append(max(1e-5, b_r / s_r))
                if len(win_ratios) >= 2:
                    jumps = [
                        abs(20.0 * math.log10(win_ratios[k + 1] / win_ratios[k]))
                        for k in range(len(win_ratios) - 1)
                    ]
                    max_bg_jump_db = round(max(jumps), 3)
                if win_ratios:
                    continuous_ducking_verified = bool(max(win_ratios) <= 0.30)

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
        "background_unspoken_rms": round(bg_unspoken_rms, 6) if bg_unspoken_rms is not None else None,
        "measured_duck_ratio_during_speech": measured_duck_ratio,
        "measured_duck_ratio_unspoken": measured_duck_ratio_unspoken,
        "max_background_gain_jump_db": max_bg_jump_db,
        "continuous_ducking_verified": continuous_ducking_verified,
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
               unspoken_gain=1.0, fade_out_seconds=.18, fade_in_seconds=.25, release_seconds=None,
               ducking_style="dynamic", ambient_backend="auto", ambient_gain=.85):
    number(max_timing_shift, "maximum timing shift")
    if max_timing_shift > 2:raise ValueError("Maximum timing shift is limited to 2 seconds.")
    number(seconds, "seconds", positive=True)
    number(reference_gain, "reference gain")
    if reference_gain > .5 or mode not in VALID_MIX_MODES:
        raise ValueError("Invalid mode or reference gain (allowed 0..0.5).")
    if mode == "continuous-voiceover":
        ducking_style = "continuous"
    elif mode == "long-release-voiceover":
        ducking_style = "long-release"
    if ducking_style not in VALID_DUCKING_STYLES:
        raise ValueError(f"Invalid ducking_style: {ducking_style!r}")
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

    eff_fade_in = float(release_seconds) if release_seconds is not None else float(fade_in_seconds)
    if mode == "japanese-only":
        policy = "muted"
    elif mode == "reference-mix":
        policy = "quiet English reference; no speech separation"
    elif mode in ("voiceover", "ducking", "continuous-voiceover", "long-release-voiceover"):
        policy = f"{ducking_style} voice-over ducking (speech={reference_gain:.2f}, unspoken={unspoken_gain:.2f})"
    else:
        policy = f"separated ambient preservation ({ambient_backend}, ambient_gain={ambient_gain:.2f}, duck_gain={reference_gain:.2f})"

    extra_phase13 = {}
    if mode in ("voiceover", "ducking", "continuous-voiceover", "long-release-voiceover", "ambient", "hybrid-ambient"):
        extra_phase13 = {
            "ducking_style": str(ducking_style),
            "unspoken_gain": float(unspoken_gain),
            "fade_out_seconds": float(fade_out_seconds),
            "fade_in_seconds": eff_fade_in,
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
            elif plan["mode"] in ("voiceover", "ducking", "continuous-voiceover", "long-release-voiceover", "ambient", "hybrid-ambient"):
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
                    ducking_style=plan.get("ducking_style", "dynamic"),
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
                    speech_weight = (float(plan.get("unspoken_gain", 1.0)) - env) / max(1e-6, abs(float(plan.get("unspoken_gain", 1.0)) - float(plan.get("reference_gain", 0.12))))
                    speech_weight = np.clip(speech_weight, 0.0, 1.0)[:, None]
                    bg_during_speech = sep["ambient"] * amb_gain + sep["speech"] * residual_voice_gain
                    bg_unspoken = src_stereo * float(ducking_report.get("unspoken_gain", plan.get("unspoken_gain", 1.0)))
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
            if temp.exists(): shutil.rmtree(temp)
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
    parser.add_argument("--ducking-style", choices=VALID_DUCKING_STYLES, default="dynamic", help="Ducking envelope behavior: dynamic, continuous (15-20% whole video), or long-release (>=1.5s)")
    parser.add_argument("--reference-gain", "--duck-gain", dest="reference_gain", type=float, default=.12)
    parser.add_argument("--unspoken-gain", type=float, default=1.0, help="Original audio gain during non-TTS intervals in voiceover/ambient modes")
    parser.add_argument("--fade-out-seconds", type=float, default=.18, help="Ducking fade-out duration before TTS starts")
    parser.add_argument("--fade-in-seconds", type=float, default=.25, help="Ducking fade-in duration after TTS ends")
    parser.add_argument("--release-seconds", type=float, default=None, help="Optional release time override after TTS ends (e.g. >=1.5s)")
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
                          release_seconds=args.release_seconds, ducking_style=args.ducking_style,
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
