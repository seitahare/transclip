"""Phase 13.2 — Speaker Voice Consistency Registry, Reference Quality Selection,
Intra-Speaker Voice Consistency Evaluation, Time-Stretch Degradation Diagnostics,
and End-to-End Speaker Assignment Verification (`voice_consistency.py`).

Universal design with zero hardcoded video IDs, speaker IDs, or timestamps.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import wave

import numpy as np


class VoiceConsistencyError(RuntimeError):
    """Raised when speaker voice profile locking or consistency invariants are violated."""


class SpeakerProfileMismatchError(VoiceConsistencyError):
    """Raised when an utterance is routed to a different speaker's voice profile or unauthorized preset."""


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_mono_wav_float32(wav_path):
    p = Path(wav_path)
    if not p.is_file() or p.stat().st_size <= 44:
        raise ValueError(f"WAV file missing or empty: {p}")
    with wave.open(str(p), "rb") as wf:
        sr = int(wf.getframerate())
        nch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        nframes = int(wf.getnframes())
        raw = wf.readframes(nframes)
    if sr <= 0 or nframes <= 0:
        raise ValueError(f"Invalid WAV parameters in {p}: sr={sr}, frames={nframes}")
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sw} in {p}")
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr, sr


def compute_acoustic_speaker_embedding(wav_path):
    """Compute a 24-dimensional speaker voice characteristic & spectral envelope embedding,
    plus detailed F0, spectral, harmonic, and distortion metrics from a WAV file.
    """
    samples, sr = _read_mono_wav_float32(wav_path)
    duration = len(samples) / float(sr)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    clipped_count = int(np.sum(np.abs(samples) >= 0.999))

    frame_len = max(256, int(round(sr * 0.032)))
    hop = max(128, int(round(sr * 0.016)))
    win = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sr))

    # 16 log-spaced spectral filterbank bands from 80 Hz to min(7600 Hz, sr/2 - 100)
    max_freq = max(1200.0, min(7600.0, sr * 0.45))
    band_edges = np.geomspace(80.0, max_freq, 17)

    f0_values = []
    centroids = []
    tilts = []
    hnr_values = []
    frame_rms_list = []
    band_energies_list = []

    min_lag = max(1, int(sr / 360.0))
    max_lag = max(min_lag + 2, int(sr / 65.0))
    energy_thr = max(0.008, rms * 0.28)

    for start in range(0, max(1, len(samples) - frame_len + 1), hop):
        frame = samples[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        f_rms = float(np.sqrt(np.mean(frame * frame)))
        frame_rms_list.append(f_rms)
        if f_rms < energy_thr:
            continue

        w_frame = frame * win
        spec = np.abs(np.fft.rfft(w_frame)) + 1e-9
        power = spec * spec
        spec_sum = float(np.sum(spec))
        if spec_sum > 1e-6:
            centroids.append(float(np.sum(freqs * spec) / spec_sum))

        low_mask = (freqs >= 80.0) & (freqs < 1000.0)
        high_mask = (freqs >= 1000.0) & (freqs <= max_freq)
        low_e = float(np.mean(power[low_mask])) if np.any(low_mask) else 1e-9
        high_e = float(np.mean(power[high_mask])) if np.any(high_mask) else 1e-9
        tilts.append(10.0 * math.log10(max(high_e, 1e-12) / max(low_e, 1e-12)))

        # Filterbank log energies
        b_vec = []
        for b_i in range(16):
            m = (freqs >= band_edges[b_i]) & (freqs < band_edges[b_i + 1])
            b_e = float(np.mean(power[m])) if np.any(m) else 1e-12
            b_vec.append(math.log10(max(b_e, 1e-12)))
        band_energies_list.append(b_vec)

        # Autocorrelation F0 and harmonicity (HNR)
        corr = np.correlate(w_frame, w_frame, mode="full")[frame_len - 1 :]
        if corr.size > max_lag and corr[0] > 1e-7:
            sub = corr[min_lag:max_lag]
            best_idx = int(np.argmax(sub)) + min_lag
            r_max = float(corr[best_idx] / corr[0])
            if 0.25 <= r_max < 0.9999:
                f0_values.append(float(sr / best_idx))
                hnr_db = 10.0 * math.log10(max(r_max, 1e-4) / max(1.0 - r_max, 1e-4))
                hnr_values.append(hnr_db)

    if f0_values:
        f0_arr = np.asarray(f0_values, dtype=np.float64)
        f0_median = float(np.median(f0_arr))
        f0_p10 = float(np.percentile(f0_arr, 10))
        f0_p90 = float(np.percentile(f0_arr, 90))
        f0_std = float(np.std(f0_arr))
    else:
        f0_median, f0_p10, f0_p90, f0_std = 125.0, 110.0, 140.0, 8.0

    centroid_med = float(np.median(centroids)) if centroids else 600.0
    tilt_med = float(np.median(tilts)) if tilts else -12.0
    hnr_med = float(np.median(hnr_values)) if hnr_values else 10.0

    if band_energies_list:
        band_mat = np.asarray(band_energies_list, dtype=np.float64)
        band_mean = np.mean(band_mat, axis=0)
        band_mean = band_mean - np.mean(band_mean)
    else:
        band_mean = np.zeros(16, dtype=np.float64)

    # Detect warble / tremolo modulation (4..14 Hz amplitude/pitch flutter typical of time-stretch warble)
    warble_index = 0.0
    if len(frame_rms_list) >= 16:
        env_arr = np.asarray(frame_rms_list, dtype=np.float64)
        env_centered = env_arr - np.mean(env_arr)
        env_spec = np.abs(np.fft.rfft(env_centered * np.hanning(len(env_centered))))
        env_freqs = np.fft.rfftfreq(len(env_centered), d=hop / float(sr))
        flutter_mask = (env_freqs >= 6.0) & (env_freqs <= 16.0)
        syll_mask = (env_freqs >= 1.5) & (env_freqs < 6.0)
        flutter_e = float(np.mean(env_spec[flutter_mask])) if np.any(flutter_mask) else 0.0
        syll_e = float(np.mean(env_spec[syll_mask])) if np.any(syll_mask) else 1e-6
        warble_index = round(float(flutter_e / max(syll_e, 1e-6)), 4)

    # Detect onset distortion (sudden high-frequency burst or F0 instability in first 250ms of active speech)
    active_frames = [i for i, r in enumerate(frame_rms_list) if r >= energy_thr]
    onset_distortion_score = 0.0
    if len(f0_values) >= 4:
        onset_f0_jump = abs(12.0 * math.log2(max(f0_values[0], 50.0) / max(f0_median, 50.0)))
        onset_distortion_score = round(min(1.0, onset_f0_jump / 12.0), 4)

    # Detect click noise (sample-to-sample second-derivative spikes)
    if samples.size >= 4:
        diff2 = np.abs(np.diff(samples, n=2))
        click_thr = max(0.35, peak * 0.65)
        click_count = int(np.sum(diff2 > click_thr))
    else:
        click_count = 0

    voiced_ratio = round(len(band_energies_list) / max(1, len(frame_rms_list)), 4)

    # Construct 24-dim normalized speaker embedding vector
    raw_emb = np.concatenate([
        band_mean / (np.linalg.norm(band_mean) + 1e-8),
        np.asarray([
            f0_median / 250.0,
            f0_p10 / 250.0,
            f0_p90 / 250.0,
            f0_std / 60.0,
            centroid_med / 2000.0,
            (tilt_med + 25.0) / 30.0,
            hnr_med / 25.0,
            voiced_ratio,
        ], dtype=np.float64),
    ])
    norm = float(np.linalg.norm(raw_emb))
    embedding = [round(float(x / (norm + 1e-9)), 6) for x in raw_emb]

    return {
        "duration_seconds": round(duration, 4),
        "sample_rate": sr,
        "peak_amplitude": round(peak, 5),
        "rms_amplitude": round(rms, 5),
        "voiced_frame_ratio": voiced_ratio,
        "f0_median_hz": round(f0_median, 2),
        "f0_p10_hz": round(f0_p10, 2),
        "f0_p90_hz": round(f0_p90, 2),
        "f0_std_hz": round(f0_std, 2),
        "spectral_centroid_hz": round(centroid_med, 2),
        "spectral_tilt_db": round(tilt_med, 2),
        "hnr_db": round(hnr_med, 2),
        "warble_modulation_index": warble_index,
        "onset_distortion_score": onset_distortion_score,
        "clipped_sample_count": clipped_count,
        "click_noise_count": click_count,
        "speaker_embedding": embedding,
    }


def embedding_cosine_similarity(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return round(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)), 4)


def inspect_reference_audio_quality(
    wav_path,
    *,
    overlap_seconds=0.0,
    has_overlap_with_other_speaker=False,
    min_duration_seconds=1.5,
):
    """Inspect a candidate speaker reference WAV for solo-speaker purity, SNR/harmonicity,
    duration, F0 stability, and absence of clipping/noise.
    """
    if has_overlap_with_other_speaker and float(overlap_seconds) <= 0.0:
        overlap_seconds = 0.25
    metrics = compute_acoustic_speaker_embedding(wav_path)
    dur = metrics["duration_seconds"]
    voiced = metrics["voiced_frame_ratio"]
    hnr = metrics["hnr_db"]
    clipped = metrics["clipped_sample_count"]
    f0_std = metrics["f0_std_hz"]

    issues = []
    if dur < min_duration_seconds:
        issues.append(f"duration_too_short:{dur:.2f}s")
    if overlap_seconds > 0.05:
        issues.append(f"contains_other_speaker_overlap:{overlap_seconds:.2f}s")
    if clipped > 0:
        issues.append(f"clipped_samples:{clipped}")
    if voiced < 0.25:
        issues.append(f"low_voiced_ratio:{voiced:.2f}")
    if hnr < 3.0:
        issues.append(f"high_background_noise_or_music:hnr={hnr:.1f}dB")

    # Composite quality score in [0.0, 100.0]
    score = 60.0
    score += min(20.0, max(0.0, (dur - 1.5) * 4.5))
    score += min(12.0, max(-15.0, (hnr - 5.0) * 1.2))
    score += min(8.0, voiced * 10.0)
    if overlap_seconds > 0.0:
        score -= min(50.0, overlap_seconds * 40.0)
    if clipped > 0:
        score -= 25.0
    if f0_std > 45.0:
        score -= min(15.0, (f0_std - 45.0) * 0.4)
    score = round(max(0.0, min(100.0, score)), 2)

    eligible = bool(dur >= 0.5 and overlap_seconds <= 0.05 and clipped == 0 and voiced >= 0.15)
    return {
        "audio_path": str(Path(wav_path).resolve()),
        "audio_sha256": _sha256_file(wav_path),
        "duration_seconds": dur,
        "overlap_seconds": round(float(overlap_seconds), 4),
        "voiced_frame_ratio": voiced,
        "hnr_db": hnr,
        "f0_median_hz": metrics["f0_median_hz"],
        "f0_std_hz": f0_std,
        "spectral_centroid_hz": metrics["spectral_centroid_hz"],
        "clipped_sample_count": clipped,
        "quality_score": score,
        "eligible": eligible,
        "issues": issues,
        "speaker_embedding": metrics["speaker_embedding"],
    }


def select_best_speaker_reference_candidate(candidates):
    """Evaluate multiple candidate reference WAVs for a speaker and select the single best eligible candidate."""
    if not candidates:
        raise VoiceConsistencyError("No reference audio candidates provided for selection.")
    evaluated = []
    for cand in candidates:
        if isinstance(cand, (str, Path)):
            qual = inspect_reference_audio_quality(cand)
        elif isinstance(cand, dict) and "quality_score" in cand:
            qual = dict(cand)
        elif isinstance(cand, dict) and "audio_path" in cand:
            qual = inspect_reference_audio_quality(
                cand["audio_path"],
                overlap_seconds=float(cand.get("overlap_seconds", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported candidate format: {cand!r}")
        evaluated.append(qual)

    evaluated.sort(key=lambda r: (bool(r.get("eligible", True)), float(r.get("quality_score", 0.0)), float(r.get("duration_seconds", 0.0))), reverse=True)
    best = evaluated[0]
    return {
        "selected": best,
        "candidate_count": len(evaluated),
        "candidates": evaluated,
    }


@dataclass(frozen=True)
class LockedSpeakerVoiceProfile:
    """Immutable per-speaker voice profile locked across an entire video/pipeline execution."""

    speaker_id: str
    voice_profile_id: str
    voice_mode: str  # "xvector_clone", "icl_clone", "approved_clone", or "preset"
    reference_audio_path: str | None
    reference_audio_sha256: str | None
    speaker_embedding: list[float] | dict | None
    clone_prompt_cache_key: str | None
    model_id: str
    model_revision: str
    generation_settings: dict
    reference_quality: dict | None
    verification_status: str
    preset_voice: str | None = None
    reference_text: str | None = None

    def to_dict(self):
        return asdict(self)


class SpeakerVoiceRegistry:
    """Per-video registry locking one fixed `LockedSpeakerVoiceProfile` per speaker_id.

    Guarantees:
    1. Every utterance of `speaker_id` uses the exact same `voice_profile_id` and `reference_audio_sha256`.
    2. Never switches reference audio mid-video or substitutes another speaker's reference audio.
    3. Never silently falls back from clone mode to a preset voice.
    """

    def __init__(self, *, video_id="unknown_video"):
        self.video_id = str(video_id)
        self._profiles: dict[str, LockedSpeakerVoiceProfile] = {}
        self._utterance_bindings: list[dict] = []

    @property
    def profiles(self) -> dict[str, LockedSpeakerVoiceProfile]:
        return self._profiles

    def register_speaker_profile(
        self,
        *,
        speaker_id: str,
        voice_mode: str,
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        model_revision: str = "local_snapshot",
        generation_settings: dict | None = None,
        reference_audio_path: str | Path | None = None,
        reference_text: str | None = None,
        reference_quality: dict | None = None,
        verification_status: str = "verified_permitted",
        preset_voice: str | None = None,
        clone_method: str | None = None,
        reference_origin: str | None = None,
        profile_fingerprint: str | None = None,
        **kwargs,
    ) -> LockedSpeakerVoiceProfile:
        spk = str(speaker_id or "").strip()
        if not spk or spk.upper() in ("NONE", "NULL", "UNKNOWN", "MIXED", "REVIEW"):
            raise VoiceConsistencyError(f"Cannot register voice profile for unresolved speaker_id: {speaker_id!r}")
        if voice_mode not in ("xvector_clone", "icl_clone", "approved_clone", "preset"):
            raise VoiceConsistencyError(f"Unsupported voice_mode {voice_mode!r} for speaker {spk!r}")

        ref_path_str = None
        ref_sha = None
        emb = None
        ref_qual = reference_quality

        if voice_mode in ("xvector_clone", "icl_clone", "approved_clone"):
            if not reference_audio_path:
                raise VoiceConsistencyError(
                    f"Speaker {spk!r} is configured for {voice_mode!r}, but no reference_audio_path was provided. "
                    f"Silent fallback to preset or another speaker is prohibited."
                )
            rp = Path(reference_audio_path).resolve()
            if not rp.is_file():
                raise VoiceConsistencyError(f"Reference audio file not found for speaker {spk!r}: {rp}")
            ref_path_str = str(rp)
            ref_sha = _sha256_file(rp)
            if ref_qual is None:
                ref_qual = inspect_reference_audio_quality(rp)
            emb = ref_qual.get("speaker_embedding")

            # Ensure no other speaker in this registry is using the exact same reference audio file/hash!
            for other_spk, other_prof in self._profiles.items():
                if other_spk != spk and other_prof.reference_audio_sha256 and other_prof.reference_audio_sha256 == ref_sha:
                    raise SpeakerProfileMismatchError(
                        f"Reference audio collision detected: speaker {spk!r} and speaker {other_spk!r} "
                        f"were assigned the same reference audio (sha256={ref_sha[:12]})."
                    )
        else:
            if not preset_voice:
                raise VoiceConsistencyError(f"Preset mode for speaker {spk!r} requires an explicit preset_voice.")

        settings = dict(generation_settings or {})
        if clone_method:
            settings.setdefault("clone_method", clone_method)
        if reference_origin:
            settings.setdefault("reference_origin", reference_origin)
        if profile_fingerprint:
            settings.setdefault("profile_fingerprint", profile_fingerprint)

        id_payload = json.dumps(
            {
                "speaker_id": spk,
                "voice_mode": voice_mode,
                "reference_audio_sha256": ref_sha,
                "preset_voice": preset_voice,
                "model_id": str(model_id),
                "model_revision": str(model_revision),
            },
            sort_keys=True,
        )
        profile_id = f"vp_{spk}_{hashlib.sha256(id_payload.encode('utf-8')).hexdigest()[:16]}"
        prompt_cache_key = (
            f"{profile_id}:{voice_mode}:{ref_sha[:16]}"
            if ref_sha
            else f"{profile_id}:preset:{preset_voice}"
        )

        profile = LockedSpeakerVoiceProfile(
            speaker_id=spk,
            voice_profile_id=profile_id,
            voice_mode=voice_mode,
            reference_audio_path=ref_path_str,
            reference_audio_sha256=ref_sha,
            speaker_embedding=emb,
            clone_prompt_cache_key=prompt_cache_key,
            model_id=str(model_id),
            model_revision=str(model_revision),
            generation_settings=settings,
            reference_quality=ref_qual,
            verification_status=str(verification_status),
            preset_voice=preset_voice,
            reference_text=reference_text,
        )

        if spk in self._profiles:
            existing = self._profiles[spk]
            if existing.voice_profile_id != profile.voice_profile_id:
                raise SpeakerProfileMismatchError(
                    f"Speaker {spk!r} already has locked profile {existing.voice_profile_id!r}; "
                    f"cannot overwrite with {profile.voice_profile_id!r} mid-pipeline."
                )
            return existing

        self._profiles[spk] = profile
        return profile

    def get_locked_profile(self, speaker_id: str) -> LockedSpeakerVoiceProfile:
        spk = str(speaker_id or "").strip()
        if spk not in self._profiles:
            raise SpeakerProfileMismatchError(f"Speaker {spk!r} has no locked profile in SpeakerVoiceRegistry.")
        return self._profiles[spk]

    def verify_utterance_routing(
        self,
        utterance: dict,
        *,
        requested_voice: str | None = None,
        requested_profile=None,
    ) -> LockedSpeakerVoiceProfile:
        """Verify that `utterance` is routed to the exact locked profile for `utterance['speaker']`."""
        uid = utterance.get("id", utterance.get("utterance_id", "unknown"))
        spk = str(utterance.get("speaker") or utterance.get("speaker_id") or "").strip()
        locked = self.get_locked_profile(spk)

        if requested_profile is not None:
            prof_spk = getattr(requested_profile, "speaker_id", getattr(requested_profile, "speaker", None))
            if prof_spk and str(prof_spk).strip() != spk:
                raise SpeakerProfileMismatchError(
                    f"Utterance {uid!r} belongs to speaker {spk!r}, but was routed to profile for speaker {prof_spk!r}."
                )
            prof_sha = getattr(requested_profile, "reference_audio_sha256", None)
            if not prof_sha and getattr(requested_profile, "reference_audio", None):
                try:
                    prof_sha = _sha256_file(getattr(requested_profile, "reference_audio"))
                except Exception:
                    prof_sha = None
            if locked.reference_audio_sha256 and prof_sha and prof_sha != locked.reference_audio_sha256:
                raise SpeakerProfileMismatchError(
                    f"Utterance {uid!r} (speaker {spk!r}) reference_audio_sha256 {prof_sha!r} != locked {locked.reference_audio_sha256!r}."
                )

        if locked.voice_mode in ("xvector_clone", "icl_clone", "approved_clone"):
            if requested_voice and not str(requested_voice).startswith("approved_clone:"):
                raise SpeakerProfileMismatchError(
                    f"Utterance {uid!r} (speaker {spk!r}) attempted to use non-clone voice {requested_voice!r} "
                    f"while locked to {locked.voice_mode!r}."
                )
        elif locked.voice_mode == "preset":
            if requested_voice and locked.preset_voice and str(requested_voice) != str(locked.preset_voice):
                raise SpeakerProfileMismatchError(
                    f"Utterance {uid!r} (speaker {spk!r}) attempted to use preset {requested_voice!r} != locked {locked.preset_voice!r}."
                )

        self.verify_and_record_utterance_binding(
            utterance_id=uid,
            speaker_id=spk,
            voice_profile_id=locked.voice_profile_id,
            used_voice_mode=locked.voice_mode,
            used_reference_sha256=locked.reference_audio_sha256,
            used_preset_voice=locked.preset_voice if locked.voice_mode == "preset" else None,
        )
        return locked

    def verify_and_record_utterance_binding(
        self,
        *,
        utterance_id,
        speaker_id: str,
        voice_profile_id: str,
        used_voice_mode: str,
        used_reference_sha256: str | None = None,
        used_preset_voice: str | None = None,
        audio_file: str | None = None,
    ) -> dict:
        locked = self.get_locked_profile(speaker_id)
        if voice_profile_id != locked.voice_profile_id:
            raise SpeakerProfileMismatchError(
                f"Utterance {utterance_id!r} (speaker {speaker_id!r}) attempted to use voice_profile_id "
                f"{voice_profile_id!r}, violating locked profile {locked.voice_profile_id!r}."
            )
        if locked.voice_mode in ("xvector_clone", "icl_clone", "approved_clone"):
            if used_voice_mode == "preset" or used_preset_voice is not None:
                raise VoiceConsistencyError(
                    f"Utterance {utterance_id!r} (speaker {speaker_id!r}) silently fell back to preset "
                    f"{used_preset_voice!r} when locked to {locked.voice_mode!r}."
                )
            if used_reference_sha256 and used_reference_sha256 != locked.reference_audio_sha256:
                raise VoiceConsistencyError(
                    f"Utterance {utterance_id!r} (speaker {speaker_id!r}) used reference_audio_sha256 "
                    f"{used_reference_sha256!r} != locked {locked.reference_audio_sha256!r}."
                )
        record = {
            "utterance_id": utterance_id,
            "speaker_id": locked.speaker_id,
            "voice_profile_id": locked.voice_profile_id,
            "voice_mode": locked.voice_mode,
            "reference_audio_sha256": locked.reference_audio_sha256,
            "preset_voice": locked.preset_voice,
            "audio_file": str(audio_file) if audio_file else None,
            "consistent_with_locked_profile": True,
        }
        self._utterance_bindings.append(record)
        return record

    def to_dict(self):
        return {
            "kind": "speaker_voice_registry",
            "format_version": 1,
            "video_id": self.video_id,
            "speaker_count": len(self._profiles),
            "profiles": {spk: prof.to_dict() for spk, prof in sorted(self._profiles.items())},
            "utterance_bindings": list(self._utterance_bindings),
            "all_utterances_bound_to_locked_profile": True,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)
        return p


def evaluate_speaker_voice_consistency(
    utterance_rows,
    *,
    registry: SpeakerVoiceRegistry | None = None,
    max_allowed_f0_jump_semitones: float = 4.5,
    min_centroid_embedding_cosine: float = 0.82,
    max_warble_index: float = 0.72,
):
    """Evaluate intra-speaker voice consistency AND reference similarity across all generated utterances.

    Separates:
    1. `reference_similarity_by_speaker`: similarity between generated Japanese utterances and English reference audio.
    2. `intra_speaker_consistency_by_speaker` / `speakers`: consistency of voice quality, F0, speaking rate, and spectrum
       across all utterances of the same speaker within the video.
    """
    by_speaker: dict[str, list[dict]] = {}
    for row in utterance_rows or []:
        spk = str(row.get("speaker") or row.get("speaker_id") or "")
        af = row.get("audio_file") or row.get("processed_wav_path") or row.get("raw_tts_wav_path")
        if not spk or not af or not Path(af).is_file():
            continue
        by_speaker.setdefault(spk, []).append(row)

    intra_speaker_summary = {}
    reference_similarity_summary = {}
    all_utterance_metrics = []
    outlier_utterances = []

    for spk, rows in sorted(by_speaker.items()):
        locked_prof = None
        if registry is not None:
            try:
                locked_prof = registry.get_locked_profile(spk)
            except Exception:
                locked_prof = None

        ref_emb = locked_prof.speaker_embedding if locked_prof else None
        ref_f0 = (
            float(locked_prof.reference_quality.get("f0_median_hz"))
            if (locked_prof and isinstance(locked_prof.reference_quality, dict) and locked_prof.reference_quality.get("f0_median_hz"))
            else None
        )

        spk_measurements = []
        for r in rows:
            uid = r.get("id", r.get("utterance_id"))
            af = r.get("audio_file") or r.get("processed_wav_path") or r.get("raw_tts_wav_path")
            m = compute_acoustic_speaker_embedding(af)
            mora = int(r.get("mora_count") or max(1, len(str(r.get("text_ja", "")).strip())))
            dur = max(0.1, float(m["duration_seconds"]))
            mora_per_sec = round(mora / dur, 3)
            m["utterance_id"] = uid
            m["speaker_id"] = spk
            m["voice_profile_id"] = r.get("voice_profile_id") or (locked_prof.voice_profile_id if locked_prof else None)
            m["reference_audio_sha256"] = r.get("reference_audio_sha256") or (locked_prof.reference_audio_sha256 if locked_prof else None)
            m["audio_file"] = str(af)
            m["mora_count"] = mora
            m["speaking_rate_mora_per_sec"] = mora_per_sec
            spk_measurements.append(m)

        # Compute speaker centroid embedding & median F0 across all utterances of `spk`
        emb_mat = np.asarray([m["speaker_embedding"] for m in spk_measurements], dtype=np.float64)
        centroid_vec = np.mean(emb_mat, axis=0)
        c_norm = float(np.linalg.norm(centroid_vec))
        if c_norm > 1e-9:
            centroid_vec = centroid_vec / c_norm

        f0_list = [m["f0_median_hz"] for m in spk_measurements if m["f0_median_hz"] > 0]
        spk_median_f0 = float(np.median(f0_list)) if f0_list else 125.0
        rate_list = [m["speaking_rate_mora_per_sec"] for m in spk_measurements]
        spk_median_rate = float(np.median(rate_list)) if rate_list else 7.0

        pairwise_cosines = []
        for i in range(len(spk_measurements)):
            for j in range(i + 1, len(spk_measurements)):
                pairwise_cosines.append(
                    embedding_cosine_similarity(
                        spk_measurements[i]["speaker_embedding"],
                        spk_measurements[j]["speaker_embedding"],
                    )
                )
        mean_pairwise_cosine = round(float(np.mean(pairwise_cosines)), 4) if pairwise_cosines else 1.0
        min_pairwise_cosine = round(float(np.min(pairwise_cosines)), 4) if pairwise_cosines else 1.0

        adjacent_f0_jumps = []
        for i in range(len(spk_measurements) - 1):
            f_a = max(50.0, spk_measurements[i]["f0_median_hz"])
            f_b = max(50.0, spk_measurements[i + 1]["f0_median_hz"])
            adjacent_f0_jumps.append(round(abs(12.0 * math.log2(f_b / f_a)), 3))
        max_adj_f0_jump = max(adjacent_f0_jumps) if adjacent_f0_jumps else 0.0

        ref_cosines = []
        ref_f0_deltas = []
        spk_outliers = []

        for m in spk_measurements:
            cos_to_centroid = embedding_cosine_similarity(m["speaker_embedding"], centroid_vec)
            semitone_dev = round(abs(12.0 * math.log2(max(50.0, m["f0_median_hz"]) / max(50.0, spk_median_f0))), 3)
            cos_to_ref = embedding_cosine_similarity(m["speaker_embedding"], ref_emb) if isinstance(ref_emb, list) else None
            if cos_to_ref is not None:
                ref_cosines.append(cos_to_ref)
            if ref_f0 is not None and ref_f0 > 0:
                ref_f0_deltas.append(round(abs(m["f0_median_hz"] - ref_f0), 2))

            reasons = []
            if locked_prof is not None:
                if m.get("voice_profile_id") and m["voice_profile_id"] != locked_prof.voice_profile_id:
                    reasons.append(f"voice_profile_mismatch:{m['voice_profile_id']}!={locked_prof.voice_profile_id}")
                if locked_prof.reference_audio_sha256 and m.get("reference_audio_sha256") and m["reference_audio_sha256"] != locked_prof.reference_audio_sha256:
                    reasons.append("reference_audio_sha256_mismatch")
            if len(spk_measurements) >= 2 and cos_to_centroid < min_centroid_embedding_cosine:
                reasons.append(f"low_embedding_similarity_to_speaker_centroid:{cos_to_centroid:.3f}")
            if len(spk_measurements) >= 2 and semitone_dev > max_allowed_f0_jump_semitones:
                reasons.append(f"excessive_f0_deviation_semitones:{semitone_dev:.2f}")
            if m["warble_modulation_index"] > max_warble_index:
                reasons.append(f"warble_or_tremolo_detected:{m['warble_modulation_index']:.3f}")
            if m["clipped_sample_count"] > 0:
                reasons.append(f"clipped_samples:{m['clipped_sample_count']}")

            m["cosine_similarity_to_speaker_centroid"] = cos_to_centroid
            m["cosine_similarity_to_reference"] = cos_to_ref
            m["semitone_deviation_from_speaker_median"] = semitone_dev
            m["has_unnatural_distortion"] = bool(
                m["warble_modulation_index"] > max_warble_index or m["clipped_sample_count"] > 0
            )
            m["is_consistency_outlier"] = len(reasons) > 0
            m["outlier_reasons"] = reasons

            if reasons:
                outlier_rec = {
                    "utterance_id": m["utterance_id"],
                    "speaker_id": spk,
                    "voice_profile_id": m["voice_profile_id"],
                    "reasons": reasons,
                    "f0_median_hz": m["f0_median_hz"],
                    "cosine_similarity_to_speaker_centroid": cos_to_centroid,
                }
                spk_outliers.append(outlier_rec)
                outlier_utterances.append(outlier_rec)

            all_utterance_metrics.append(m)

        profile_ids_used = sorted({str(m["voice_profile_id"]) for m in spk_measurements if m.get("voice_profile_id")})
        ref_shas_used = sorted({str(m["reference_audio_sha256"]) for m in spk_measurements if m.get("reference_audio_sha256")})
        single_prof = len(profile_ids_used) <= 1 and len(ref_shas_used) <= 1
        if locked_prof is not None and profile_ids_used and profile_ids_used != [locked_prof.voice_profile_id]:
            single_prof = False

        intra_speaker_summary[spk] = {
            "speaker_id": spk,
            "utterance_count": len(spk_measurements),
            "locked_voice_profile_id": locked_prof.voice_profile_id if locked_prof else (profile_ids_used[0] if profile_ids_used else None),
            "unique_voice_profile_ids_used": profile_ids_used,
            "single_profile_enforced": single_prof,
            "profile_lock_verified": single_prof,
            "mean_pairwise_embedding_cosine": mean_pairwise_cosine,
            "min_pairwise_embedding_cosine": min_pairwise_cosine,
            "f0_median_across_utterances_hz": round(spk_median_f0, 2),
            "max_adjacent_f0_jump_semitones": round(max_adj_f0_jump, 3),
            "median_speaking_rate_mora_per_sec": round(spk_median_rate, 3),
            "outlier_count": len(spk_outliers),
            "outlier_utterance_ids": [str(o["utterance_id"]) for o in spk_outliers],
            "outlier_utterances": spk_outliers,
            "voice_consistent": len(spk_outliers) == 0 and single_prof,
        }

        reference_similarity_summary[spk] = {
            "speaker_id": spk,
            "reference_audio_path": locked_prof.reference_audio_path if locked_prof else None,
            "reference_audio_sha256": locked_prof.reference_audio_sha256 if locked_prof else None,
            "reference_f0_median_hz": ref_f0,
            "mean_embedding_cosine_to_reference": round(float(np.mean(ref_cosines)), 4) if ref_cosines else None,
            "mean_f0_delta_to_reference_hz": round(float(np.mean(ref_f0_deltas)), 2) if ref_f0_deltas else None,
        }

    return {
        "kind": "voice_consistency_report",
        "format_version": 1,
        "speakers": intra_speaker_summary,
        "intra_speaker_consistency_by_speaker": intra_speaker_summary,
        "reference_similarity_by_speaker": reference_similarity_summary,
        "utterance_metrics": all_utterance_metrics,
        "outlier_utterances": outlier_utterances,
        "all_speakers_consistent": len(outlier_utterances) == 0 and all(
            s["single_profile_enforced"] for s in intra_speaker_summary.values()
        ),
    }


def concatenate_speaker_voice_samples(
    utterance_rows,
    output_dir=None,
    *,
    speaker_id=None,
    output_wav_path=None,
    sample_rate=24000,
    spacer_seconds=0.35,
):
    """Build per-speaker concatenated WAV files (`speaker_00_voice_samples.wav`, `speaker_01_voice_samples.wav`, etc.)
    so humans can directly listen and verify voice consistency across all utterances of each speaker.
    """
    by_speaker: dict[str, list[Path]] = {}
    for row in utterance_rows or []:
        spk = str(row.get("speaker") or row.get("speaker_id") or "").strip()
        if speaker_id and spk != str(speaker_id).strip():
            continue
        af = row.get("audio_file") or row.get("processed_wav_path") or row.get("raw_tts_wav_path")
        if spk and af and Path(af).is_file():
            by_speaker.setdefault(spk, []).append(Path(af))

    spacer = np.zeros(int(round(spacer_seconds * sample_rate)), dtype=np.float32)

    if speaker_id is not None and output_wav_path is not None:
        wav_paths = by_speaker.get(str(speaker_id).strip(), [])
        chunks = []
        for idx, wp in enumerate(wav_paths):
            arr, sr = _read_mono_wav_float32(wp)
            if sr != sample_rate and len(arr) > 1:
                new_len = max(1, int(round(len(arr) * float(sample_rate) / float(sr))))
                x_old = np.linspace(0.0, 1.0, len(arr))
                x_new = np.linspace(0.0, 1.0, new_len)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
            if idx > 0:
                chunks.append(spacer)
            chunks.append(arr)
        if not chunks:
            return {"speaker_id": speaker_id, "output_wav": None, "utterance_count": 0}
        combined = np.concatenate(chunks)
        out_path = Path(output_wav_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(np.round(combined * 32767.0), -32768, 32767).astype("<i2")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return {
            "speaker_id": speaker_id,
            "output_wav": str(out_path),
            "utterance_count": len(wav_paths),
            "duration_seconds": round(len(combined) / float(sample_rate), 4),
        }

    out_dir = Path(output_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for spk, wav_paths in sorted(by_speaker.items()):
        chunks = []
        for idx, wp in enumerate(wav_paths):
            arr, sr = _read_mono_wav_float32(wp)
            if sr != sample_rate and len(arr) > 1:
                new_len = max(1, int(round(len(arr) * float(sample_rate) / float(sr))))
                x_old = np.linspace(0.0, 1.0, len(arr))
                x_new = np.linspace(0.0, 1.0, new_len)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)
            if idx > 0:
                chunks.append(spacer)
            chunks.append(arr)
        if not chunks:
            continue
        combined = np.concatenate(chunks)
        fname = f"{spk.lower()}_voice_samples.wav"
        out_path = out_dir / fname
        pcm = np.clip(np.round(combined * 32767.0), -32768, 32767).astype("<i2")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        written[spk] = str(out_path)
    return written


# ============================================================================
# Requirement 3: Speaker Diarization, Overlap/Backchannel Handling &
#                Contextual Speaker Assignment Verification
# ============================================================================

def classify_conversational_turn_interaction(curr_utt, prev_utt=None, next_utt=None, diarization_turns=None):
    """Classify turn interaction type for `curr_utt`:
    - `short_backchannel`: brief acknowledgment ('Right', 'Yeah', 'Okay', 'Uh-huh')
    - `repeated_word_echo_response`: speaker repeats key word from previous speaker's turn as a question/confirmation (e.g. 'applies to everything.' -> 'Everything.')
    - `simultaneous_overlap`: overlaps in time with another speaker's turn
    - `question_to_answer_transition`: follows or initiates a question (`?`)
    - `sandwiched_short_turn`: short utterance (`<= 1.8s`) by Speaker B between two turns of Speaker A
    - `standard_turn`: regular conversational turn
    """
    c_start = float(curr_utt.get("start", curr_utt.get("source_start", 0.0)))
    c_end = float(curr_utt.get("end", curr_utt.get("source_end", c_start + 1.0)))
    c_dur = max(0.01, c_end - c_start)
    c_spk = curr_utt.get("speaker")
    c_en = str(curr_utt.get("text_en", curr_utt.get("text", curr_utt.get("original_text", ""))) or "").strip()
    c_words = [re.sub(r"[^a-z0-9']+", "", w.lower()) for w in c_en.split() if re.sub(r"[^a-z0-9']+", "", w.lower())]

    interaction_tags = []
    has_multi_speaker_diarization = False
    overlapping_diar_speakers = set()

    if diarization_turns:
        for dt in diarization_turns:
            d_s = float(dt.get("start", 0.0))
            d_e = float(dt.get("end", d_s))
            ov = min(c_end, d_e) - max(c_start, d_s)
            if ov > 0.10 and dt.get("speaker"):
                overlapping_diar_speakers.add(str(dt["speaker"]))
        if len(overlapping_diar_speakers) > 1:
            has_multi_speaker_diarization = True
            interaction_tags.append("simultaneous_overlap")

    if prev_utt is not None:
        p_end = float(prev_utt.get("end", prev_utt.get("source_end", 0.0)))
        p_spk = prev_utt.get("speaker")
        p_en = str(prev_utt.get("text_en", prev_utt.get("text", prev_utt.get("original_text", ""))) or "").strip()
        p_words = [re.sub(r"[^a-z0-9']+", "", w.lower()) for w in p_en.split() if re.sub(r"[^a-z0-9']+", "", w.lower())]

        if p_end > c_start + 0.08 and p_spk and c_spk and p_spk != c_spk:
            if "simultaneous_overlap" not in interaction_tags:
                interaction_tags.append("simultaneous_overlap")

        # Repeated-word echo response (e.g. "applies to everything." -> "Everything.")
        if c_words and p_words and len(c_words) <= 3 and any(w in p_words[-3:] for w in c_words if len(w) >= 4):
            if p_spk != c_spk or c_en.endswith(("?", ".")):
                interaction_tags.append("repeated_word_echo_response")

        if p_en.endswith("?") and p_spk != c_spk:
            interaction_tags.append("question_to_answer_transition")

    if c_en.endswith("?"):
        interaction_tags.append("question_turn")

    if len(c_words) <= 2 and c_dur <= 1.2 and all(w in ("right", "yeah", "yes", "okay", "ok", "sure", "uhhuh", "mhm", "indeed") for w in c_words):
        interaction_tags.append("short_backchannel")

    if prev_utt is not None and next_utt is not None:
        p_spk = prev_utt.get("speaker")
        n_spk = next_utt.get("speaker")
        if p_spk and n_spk and p_spk == n_spk and c_spk and c_spk != p_spk and c_dur <= 2.2:
            interaction_tags.append("sandwiched_short_turn")

    if not interaction_tags:
        interaction_tags.append("standard_turn")

    return {
        "interaction_tags": interaction_tags,
        "has_multi_speaker_diarization": has_multi_speaker_diarization,
        "overlapping_diarization_speakers": sorted(overlapping_diar_speakers),
        "must_not_merge_across_turn": bool(
            has_multi_speaker_diarization
            or "repeated_word_echo_response" in interaction_tags
            or "sandwiched_short_turn" in interaction_tags
            or "question_turn" in interaction_tags
        ),
    }


def evaluate_contextual_speaker_assignment(
    utterance,
    *,
    prev_utterance=None,
    next_utterance=None,
    diarization_turns=None,
    word_speakers=None,
    llm_context_hint=None,
):
    """Evaluate speaker assignment for an utterance (or list of utterances) using acoustic diarization turns,
    word-level speaker alignments, adjacent utterance continuity, and dialogue context.

    CRITICAL INVARIANT (Requirement 3.3):
    - Never overrides or assigns a speaker ID based solely on `llm_context_hint` / contextual guessing!
    - Requires acoustic/diarization evidence + structural continuity (`confidence >= 0.85`) to resolve
      an unassigned (`speaker: null`) segment; otherwise marks `speaker_verification_required = True`.
    """
    if isinstance(utterance, list):
        items = list(utterance)
        results = []
        for idx, item in enumerate(items):
            p_item = items[idx - 1] if idx > 0 else None
            n_item = items[idx + 1] if idx + 1 < len(items) else None
            results.append(
                evaluate_contextual_speaker_assignment(
                    item,
                    prev_utterance=p_item,
                    next_utterance=n_item,
                    diarization_turns=diarization_turns,
                    word_speakers=word_speakers,
                    llm_context_hint=llm_context_hint,
                )
            )
        return results

    uid = utterance.get("id", utterance.get("utterance_id"))
    raw_spk = utterance.get("raw_speaker", utterance.get("speaker"))
    if isinstance(raw_spk, str) and raw_spk.strip().upper() in ("", "NONE", "NULL", "UNKNOWN", "REVIEW", "MIXED"):
        raw_spk = None

    u_s = float(utterance.get("start", utterance.get("source_start", 0.0)))
    u_e = float(utterance.get("end", utterance.get("source_end", u_s + 0.5)))
    u_en = str(utterance.get("text_en", utterance.get("text", utterance.get("original_text", ""))) or "").strip()

    # 1. Acoustic diarization turn overlap & proximity
    diar_scores: dict[str, float] = {}
    nearest_diar_speaker = None
    nearest_diar_dist = math.inf
    for dt in diarization_turns or []:
        d_spk = dt.get("speaker")
        if not d_spk:
            continue
        d_s = float(dt.get("start", 0.0))
        d_e = float(dt.get("end", d_s))
        ov = max(0.0, min(u_e, d_e) - max(u_s, d_s))
        if ov > 0.0:
            diar_scores[d_spk] = diar_scores.get(d_spk, 0.0) + ov / max(0.05, u_e - u_s)
        dist = min(abs(u_s - d_e), abs(d_s - u_e))
        if dist < nearest_diar_dist:
            nearest_diar_dist = dist
            nearest_diar_speaker = d_spk

    best_diar_spk = None
    best_diar_ratio = 0.0
    if diar_scores:
        best_diar_spk, best_diar_ratio = max(diar_scores.items(), key=lambda kv: kv[1])
    elif nearest_diar_speaker and nearest_diar_dist <= 0.35:
        best_diar_spk = nearest_diar_speaker
        best_diar_ratio = max(0.25, 0.70 - nearest_diar_dist)

    # 2. Adjacent utterance zero-gap syntactic continuation
    cont_spk = None
    cont_score = 0.0
    cont_reason = None
    if next_utterance is not None and next_utterance.get("speaker"):
        n_s = float(next_utterance.get("start", next_utterance.get("source_start", 0.0)))
        n_en = str(next_utterance.get("text_en", next_utterance.get("text", "")) or "").strip()
        gap_to_next = abs(n_s - u_e)
        if gap_to_next <= 0.15 and not re.search(r"[.?!]$", u_en) and n_en and n_en[0].islower():
            cont_spk = next_utterance["speaker"]
            cont_score = 0.45
            cont_reason = f"zero_gap_syntactic_continuation_into_next:{next_utterance.get('id')}"

    if prev_utterance is not None and prev_utterance.get("speaker") and cont_spk is None:
        p_e = float(prev_utterance.get("end", prev_utterance.get("source_end", 0.0)))
        p_en = str(prev_utterance.get("text_en", prev_utterance.get("text", "")) or "").strip()
        gap_from_prev = abs(u_s - p_e)
        if gap_from_prev <= 0.15 and not re.search(r"[.?!]$", p_en) and u_en and u_en[0].islower():
            cont_spk = prev_utterance["speaker"]
            cont_score = 0.40
            cont_reason = f"zero_gap_syntactic_continuation_from_prev:{prev_utterance.get('id')}"

    # 3. Contextual / Q&A turn-taking hint (auxiliary only!)
    ctx_spk = None
    ctx_reason = None
    if llm_context_hint and isinstance(llm_context_hint, dict):
        ctx_spk = llm_context_hint.get("suggested_speaker")
        ctx_reason = llm_context_hint.get("reason", "llm_dialogue_context")
    elif prev_utterance is not None and prev_utterance.get("speaker"):
        p_en = str(prev_utterance.get("text_en", prev_utterance.get("text", "")) or "").strip()
        if p_en.endswith("?") and cont_spk:
            ctx_spk = cont_spk
            ctx_reason = "answer_to_preceding_question"

    evidence = []
    if raw_spk is not None:
        confidence = 0.92
        evidence.append(f"diarization_assigned:{raw_spk}")
        if best_diar_spk == raw_spk:
            confidence = min(0.99, round(0.88 + 0.10 * min(1.0, best_diar_ratio), 3))
            evidence.append(f"diarization_turn_overlap:{best_diar_ratio:.2f}")
        elif best_diar_spk and best_diar_spk != raw_spk and best_diar_ratio > 0.80:
            confidence = 0.55
            evidence.append(f"diarization_conflict:{best_diar_spk}")
        if ctx_spk and ctx_spk != raw_spk:
            evidence.append(f"context_hint_differs_not_overridden:{ctx_spk}")
        req_ver = bool(confidence < 0.75)
        return {
            "utterance_id": uid,
            "raw_speaker_id": raw_spk,
            "resolved_speaker_id": raw_spk,
            "final_assigned_speaker": raw_spk,
            "assignment_confidence": confidence,
            "assignment_status": "speaker_verification_required" if req_ver else "verified",
            "evidence": evidence,
            "llm_context_hint_used_as_auxiliary": bool(ctx_spk is not None),
            "llm_only_override_applied": False,
            "llm_only_override_forbidden": True,
            "speaker_verification_required": req_ver,
        }

    # `raw_spk` is None (`speaker: null`): require acoustic/diarization + continuation agreement
    candidate_spk = best_diar_spk or cont_spk
    confidence = 0.0
    if best_diar_spk:
        confidence += min(0.55, 0.35 + 0.30 * min(1.0, best_diar_ratio))
        evidence.append(f"acoustic_diarization_proximity:{best_diar_spk}(score={best_diar_ratio:.2f})")
    if cont_spk:
        if best_diar_spk is None or best_diar_spk == cont_spk:
            confidence += cont_score
            evidence.append(cont_reason)
        else:
            confidence -= 0.25
            evidence.append(f"continuation_conflict:{cont_spk}_vs_{best_diar_spk}")
    if ctx_spk:
        if candidate_spk and ctx_spk == candidate_spk and (best_diar_spk or cont_spk):
            confidence += 0.12
            evidence.append(f"contextual_agreement:{ctx_reason}")
        else:
            evidence.append(f"context_only_insufficient_without_acoustic_proof:{ctx_spk}")
            if not best_diar_spk and not cont_spk:
                candidate_spk = None
                confidence = 0.35

    confidence = round(max(0.0, min(0.98, confidence)), 3)
    if candidate_spk and confidence >= 0.85:
        return {
            "utterance_id": uid,
            "raw_speaker_id": None,
            "resolved_speaker_id": candidate_spk,
            "final_assigned_speaker": candidate_spk,
            "assignment_confidence": confidence,
            "assignment_status": "verified_acoustic_and_continuation",
            "evidence": evidence,
            "llm_context_hint_used_as_auxiliary": bool(ctx_spk is not None),
            "llm_only_override_applied": False,
            "llm_only_override_forbidden": True,
            "speaker_verification_required": False,
        }

    return {
        "utterance_id": uid,
        "raw_speaker_id": None,
        "resolved_speaker_id": None,
        "final_assigned_speaker": None,
        "proposed_speaker_id": candidate_spk,
        "assignment_confidence": confidence,
        "assignment_status": "speaker_verification_required",
        "evidence": evidence,
        "llm_context_hint_used_as_auxiliary": bool(ctx_spk is not None),
        "llm_only_override_applied": False,
        "llm_only_override_forbidden": True,
        "speaker_verification_required": True,
    }


def build_speaker_assignment_report(
    utterance_rows,
    *,
    registry: SpeakerVoiceRegistry | None = None,
    voice_registry: SpeakerVoiceRegistry | None = None,
    feedback_entries=None,
    speaker_verification_targets=None,
    diarization_turns=None,
    source_utterances=None,
):
    """Build Requirement 3.4 `speaker_assignment_report.json` auditing speaker ID lineage across all stages:
    `source audio -> diarization -> word timestamps -> utterances -> translation -> TTS routing -> final audio`.
    """
    active_reg = registry if registry is not None else voice_registry
    src_by_id = {str(u.get("id")): u for u in (source_utterances or []) if u.get("id") is not None}
    fb_by_id = {str(e.get("id", e.get("utterance_id"))): e for e in (feedback_entries or []) if e.get("id", e.get("utterance_id")) is not None}
    entries = []
    mismatch_warnings = []
    verification_required_ids = []

    for idx, row in enumerate(utterance_rows or []):
        uid = row.get("id", row.get("utterance_id", idx + 1))
        fb_row = fb_by_id.get(str(uid), {})
        merged_row = {**fb_row, **row}
        if not merged_row.get("voice_profile_id") and fb_row.get("voice_profile_id"):
            merged_row["voice_profile_id"] = fb_row["voice_profile_id"]
        if not merged_row.get("audio_file") and fb_row.get("audio_file"):
            merged_row["audio_file"] = fb_row["audio_file"]

        src_ids = merged_row.get("source_utterance_ids") or merged_row.get("merged_utterance_ids") or merged_row.get("merged_from_ids") or [uid]
        prev_r = utterance_rows[idx - 1] if idx > 0 else None
        next_r = utterance_rows[idx + 1] if idx + 1 < len(utterance_rows) else None

        interaction = classify_conversational_turn_interaction(
            merged_row, prev_utt=prev_r, next_utt=next_r, diarization_turns=diarization_turns
        )
        ctx_eval = merged_row.get("speaker_assignment_evaluation") or evaluate_contextual_speaker_assignment(
            merged_row,
            prev_utterance=prev_r,
            next_utterance=next_r,
            diarization_turns=diarization_turns,
        )

        # Check source utterance speaker lineage
        raw_src_speakers = []
        for sid in src_ids:
            su = src_by_id.get(str(sid))
            if su is not None:
                raw_src_speakers.append(su.get("speaker"))
        concrete_src_speakers = {s for s in raw_src_speakers if s not in (None, "", "UNKNOWN", "NULL")}

        expected_spk = merged_row.get("source_speaker_id") or merged_row.get("speaker")
        final_tts_spk = merged_row.get("tts_speaker_id") or merged_row.get("speaker")
        voice_prof_id = merged_row.get("voice_profile_id")
        wav_file = merged_row.get("audio_file") or merged_row.get("processed_wav_path")

        locked_prof = None
        if active_reg is not None and final_tts_spk:
            try:
                locked_prof = active_reg.get_locked_profile(final_tts_spk)
                if not voice_prof_id:
                    voice_prof_id = locked_prof.voice_profile_id
            except Exception:
                locked_prof = None

        mismatches = []
        if len(concrete_src_speakers) > 1:
            mismatches.append(f"merged_chunk_contains_multiple_speakers:{sorted(concrete_src_speakers)}")
        if concrete_src_speakers and expected_spk not in concrete_src_speakers:
            mismatches.append(f"utterance_speaker_{expected_spk}_differs_from_source_{sorted(concrete_src_speakers)}")
        if expected_spk != final_tts_spk:
            mismatches.append(f"tts_speaker_{final_tts_spk}_differs_from_utterance_speaker_{expected_spk}")
        if locked_prof is not None:
            if locked_prof.speaker_id != final_tts_spk:
                mismatches.append(f"voice_profile_speaker_{locked_prof.speaker_id}_differs_from_tts_{final_tts_spk}")
            if voice_prof_id and voice_prof_id != locked_prof.voice_profile_id:
                mismatches.append(f"voice_profile_id_{voice_prof_id}_differs_from_locked_{locked_prof.voice_profile_id}")

        has_mismatch = len(mismatches) > 0
        if has_mismatch:
            mismatch_warnings.append({
                "utterance_id": uid,
                "source_utterance_ids": src_ids,
                "mismatches": mismatches,
            })
        if ctx_eval.get("speaker_verification_required"):
            verification_required_ids.append(uid)

        entries.append({
            "utterance_id": uid,
            "source_utterance_ids": src_ids,
            "raw_source_speakers": raw_src_speakers if raw_src_speakers else [merged_row.get("raw_speaker", expected_spk)],
            "source_speaker_id": expected_spk,
            "assignment_confidence": ctx_eval.get("assignment_confidence", 0.95),
            "assignment_evidence": ctx_eval.get("evidence", []),
            "interaction_type": interaction["interaction_tags"],
            "translated_text": merged_row.get("text_ja") or merged_row.get("translated_text") or merged_row.get("translated") or "",
            "final_tts_speaker_id": final_tts_spk,
            "voice_profile_id": voice_prof_id,
            "wav_file": str(wav_file) if wav_file else None,
            "speaker_verification_required": bool(ctx_eval.get("speaker_verification_required", False)),
            "speaker_mismatch_warning": has_mismatch,
            "mismatch_details": mismatches,
        })

    for t_row in speaker_verification_targets or []:
        t_id = t_row.get("id", t_row.get("utterance_id"))
        if t_id not in verification_required_ids:
            verification_required_ids.append(t_id)

    return {
        "kind": "speaker_assignment_report",
        "format_version": 1,
        "stages_tracked": [
            "source_audio",
            "diarization",
            "word_timestamps",
            "utterances",
            "translation",
            "tts_routing",
            "final_audio",
        ],
        "utterance_count": len(entries),
        "mismatch_warning_count": len(mismatch_warnings),
        "mismatch_warnings": mismatch_warnings,
        "speaker_verification_required_ids": verification_required_ids,
        "all_speaker_assignments_consistent": len(mismatch_warnings) == 0,
        "entries": entries,
    }
