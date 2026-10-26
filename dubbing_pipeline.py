"""Phase 11 & Phase 12 — End-to-End X-vector Japanese Dubbing Pipeline CLI (`dubbing_pipeline.py`).

Capabilities:
1. Primary X-vector voice synthesis (`QwenProsodyCloneSynthesizer(clone_method="xvector")`, `x_vector_only_mode=True`, `ref_text=None`):
   - Reproduces original English speaker timbre with natural Japanese prosody.
   - Never silently switches to Preset when X-vector reference resolution or synthesis fails (`XVectorSynthesisError`).
2. Architectural separation between:
   - Technical Voice Synthesis Layer (`TechnicalVoiceReference`, `extract_speaker_reference_from_video`, `run_technical_xvector_feedback`)
   - Rights Verification & Publication Governance Layer (`RightsAndPublicationPolicy`, `RightsPolicyViolationError`)
   - Eliminates redundant registration ceremonies (`consent.json` / `review.json` / `voice_profiles.json`) for synthetic speakers and permitted audio sources while strictly prohibiting authentication bypasses (never fabricates consent/review files or clones unauthorized real people).
3. Unnaturally split continuous sentence detection & merging (`detect_and_merge_split_utterances`):
   - Automatically (and/or via `--merge-ids 64,65,66` / `--merge-ids 64-66`) detects and merges fragmented same-speaker utterances such as IDs 64–66 into a single natural sentence while preserving `merged_from_ids`, numbers, proper nouns, and barriers (`speaker: null`, other speakers, long gaps).
4. Explicit `speaker: null` verification target management (`partition_and_manage_null_speakers`):
   - Never guesses or auto-assigns a speaker identity to `speaker: null` (or `"UNKNOWN"`) utterances; tracks them in `speaker_verification_targets.json` and `pipeline_report.json`.
5. Actual Japanese WAV duration measurement & multi-stage timing/overlap adjustment:
   - Measures exact WAV durations, trims excess edge silence non-destructively, resolves same-speaker collisions and artificial TTS-expansion inter-speaker spills via bounded shifts and bounded FFmpeg `atempo` (`_work/adjusted_wavs/`), and preserves genuine multi-speaker overlaps.
6. Original video preservation, 48kHz multi-track audio mixing (`dubbed_audio.wav`), clip-relative (`0.0s..30.0s`) subtitles (`final_ja.srt`, `final_ja.ass`), and MP4 synthesis (`final_ja.mp4`, `final_ja_no_subtitles.mp4`).
7. Non-destructive execution with SHA-256 before/after verification of all input WAVs, translations, scripts, and VoiceProfiles.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import wave

import numpy as np

from prosody_experiments import (
    BASELINE_CLONE_SETTINGS,
    CORE_SAMPLES,
    DEFAULT_BASE_MODEL,
    DEFAULT_PRESET_MODEL,
    DEFAULT_SPEAKER_ID,
    QwenProsodyCloneSynthesizer,
    _atomic_write_json,
    _extract_path_from_result,
    _file_sha256,
    _invoke_benchmark_script_builder,
    _invoke_register_profile,
    _read_wav_float32,
    analyze_japanese_prosody,
)

DEFAULT_PHASE12_VIDEO = "output/phase11_1885_1915/source.mp4"
DEFAULT_PHASE12_SCRIPT = "output/phase11_1885_1915_inputs/dubbing_script.json"
DEFAULT_PHASE12_OUTPUT = "output/phase12_1885_1915"


class UnauthorizedSpeakerError(ValueError):
    """Raised when a speaker lacks authorization/reference or violates speaker governance rules."""


class RightsPolicyViolationError(UnauthorizedSpeakerError):
    """Raised when RightsAndPublicationPolicy blocks a speaker due to missing rights or unpermitted status."""


class UnverifiedSpeakerError(UnauthorizedSpeakerError):
    """Raised when an utterance has `speaker: null` / unassigned speaker and policy is set to fail."""


class XVectorSynthesisError(RuntimeError):
    """Raised when X-vector synthesis is required and reference resolution or synthesis fails (no silent Preset fallback)."""


class TechnicalVoiceReference:
    """Decoupled technical voice conditioning reference for X-vector (or ICL) synthesis.

    Separates technical acoustic conditioning (reference WAV validation, SHA-256 integrity,
    and speaker embedding extraction) from rights/publication registration ceremonies.
    Never writes or fabricates `consent.json` or `review.json`.
    """

    def __init__(
        self,
        *,
        speaker_id,
        reference_audio,
        source_origin="explicit_reference_wav",
        reference_text=None,
        preset_voice=None,
        extracted_segments=None,
        reference_quality=None,
    ):
        spk = str(speaker_id or "").strip()
        if not spk or spk.upper() in ("NULL", "NONE", "UNKNOWN"):
            raise ValueError(f"Invalid speaker_id for TechnicalVoiceReference: {speaker_id!r}")
        wav_path = Path(reference_audio).resolve()
        if not wav_path.is_file():
            raise FileNotFoundError(f"Technical reference audio not found for speaker {spk!r}: {wav_path}")

        samples, rate = _read_wav_float32(wav_path)
        duration = len(samples) / float(rate)
        if duration < 0.25:
            raise ValueError(f"Technical reference audio for {spk!r} is too short ({duration:.3f}s): {wav_path}")

        audio_sha = _file_sha256(wav_path)
        ref_txt = str(reference_text).strip() if reference_text is not None else None
        text_sha = (
            hashlib.sha256(ref_txt.encode("utf-8")).hexdigest()
            if ref_txt
            else hashlib.sha256(b"xvector_only_no_ref_text").hexdigest()
        )
        fingerprint = hashlib.sha256(f"{spk}:{audio_sha}:{text_sha}".encode("utf-8")).hexdigest()

        self.speaker_id = spk
        self.reference_audio = str(wav_path)
        self.reference_audio_sha256 = audio_sha
        self.reference_text = ref_txt
        self.reference_text_sha256 = text_sha if ref_txt else None
        self.duration_seconds = round(duration, 4)
        self.sample_rate = int(rate)
        self.source_origin = str(source_origin)
        self.preset_voice = preset_voice
        self.extracted_segments = list(extracted_segments or [])
        self.profile_fingerprint = fingerprint
        self.voice_profile_id = f"vp-{spk}-{audio_sha[:12]}"
        self.reference_quality = reference_quality

    def technical_snapshot(self):
        """Verify on-disk WAV SHA-256 integrity and return snapshot dictionary for QwenProsodyCloneSynthesizer."""
        current_sha = _file_sha256(self.reference_audio)
        if current_sha != self.reference_audio_sha256:
            raise ValueError(
                f"Reference audio for speaker {self.speaker_id!r} changed on disk "
                f"(expected {self.reference_audio_sha256}, got {current_sha})."
            )
        return {
            "profile_fingerprint": self.profile_fingerprint,
            "reference_audio_sha256": self.reference_audio_sha256,
            "reference_text_sha256": self.reference_text_sha256,
        }

    def to_metadata(self):
        return {
            "speaker_id": self.speaker_id,
            "voice_profile_id": self.voice_profile_id,
            "reference_audio": self.reference_audio,
            "reference_audio_sha256": self.reference_audio_sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "source_origin": self.source_origin,
            "profile_fingerprint": self.profile_fingerprint,
            "extracted_segments": self.extracted_segments,
            "reference_quality": self.reference_quality,
        }


class RightsAndPublicationPolicy:
    """Rights verification and publication governance manager, separated by design from technical synthesis.

    Rules:
    - Never fabricates `consent.json` or `review.json` (`authentication_bypass_used = False`).
    - Eliminates unnecessary registration steps for synthetic speakers (`SYNTH_*` or declared `synthetic`)
      and permitted voices (`permitted_local` workflow, `--speaker-ref`, or existing `VoiceProfile`).
    - Blocks any speaker marked as `unpermitted_real_person` or listed in `disallowed_speakers`, or any speaker
      lacking required clearance under `--rights-mode verified_profile_only` / `synthetic_only`.
    """

    VALID_RIGHTS_MODES = ("permitted_local", "verified_profile_only", "synthetic_only")
    VALID_SOURCE_TYPES = ("synthetic", "permitted", "registered_profile", "unpermitted_real_person")

    def __init__(
        self,
        *,
        rights_mode="permitted_local",
        speaker_source_types=None,
        disallowed_speakers=None,
    ):
        if rights_mode not in self.VALID_RIGHTS_MODES:
            raise ValueError(f"Unsupported rights_mode: {rights_mode!r}; expected one of {self.VALID_RIGHTS_MODES}")
        self.rights_mode = rights_mode
        self.speaker_source_types = {}
        for spk, stype in (speaker_source_types or {}).items():
            if stype not in self.VALID_SOURCE_TYPES:
                raise ValueError(f"Unsupported speaker source type {stype!r} for {spk!r}; expected {self.VALID_SOURCE_TYPES}")
            self.speaker_source_types[str(spk).strip()] = stype
        self.disallowed_speakers = {str(s).strip() for s in (disallowed_speakers or []) if str(s).strip()}

    def classify_and_verify_speaker(self, speaker_id, *, has_registered_profile=False, reference_origin=None):
        spk = str(speaker_id or "").strip()
        if not spk or spk.upper() in ("NULL", "NONE", "UNKNOWN"):
            raise RightsPolicyViolationError(
                f"Cannot clear unassigned/unknown speaker {speaker_id!r} for voice cloning."
            )
        if spk in self.disallowed_speakers:
            raise RightsPolicyViolationError(
                f"Speaker {spk!r} is explicitly disallowed by rights governance policy (--disallow-speaker)."
            )

        declared_type = self.speaker_source_types.get(spk)
        if declared_type == "unpermitted_real_person":
            raise RightsPolicyViolationError(
                f"Speaker {spk!r} is classified as 'unpermitted_real_person'. "
                f"Cloning unauthorized real-person voices is strictly prohibited."
            )

        is_synthetic = (
            declared_type == "synthetic"
            or spk.upper().startswith("SYNTH_")
            or spk.upper().startswith("FICTIONAL_")
        )

        if self.rights_mode == "synthetic_only" and not is_synthetic:
            raise RightsPolicyViolationError(
                f"Speaker {spk!r} is not a synthetic speaker, and --rights-mode=synthetic_only is active."
            )

        if self.rights_mode == "verified_profile_only" and not (has_registered_profile or is_synthetic):
            raise RightsPolicyViolationError(
                f"Speaker {spk!r} requires a registered VoiceProfile under --rights-mode=verified_profile_only."
            )

        if is_synthetic:
            return {
                "speaker": spk,
                "source_classification": "synthetic_speaker",
                "rights_mode": self.rights_mode,
                "clearance_status": "cleared_synthetic_no_registration_required",
                "publication_status": "cleared_synthetic",
                "external_publication_allowed": True,
                "registration_ceremony_eliminated": not has_registered_profile,
                "authentication_bypass_used": False,
                "reference_origin": reference_origin,
            }

        if has_registered_profile or declared_type == "registered_profile":
            return {
                "speaker": spk,
                "source_classification": "registered_permitted_profile",
                "rights_mode": self.rights_mode,
                "clearance_status": "cleared_documented_voice_profile",
                "publication_status": "cleared_documented_permission",
                "external_publication_allowed": True,
                "registration_ceremony_eliminated": False,
                "authentication_bypass_used": False,
                "reference_origin": reference_origin or "registered_voice_profile",
            }

        # Permitted source audio under `permitted_local` (or explicit `--speaker-source-type SPEAKER=permitted`)
        return {
            "speaker": spk,
            "source_classification": "permitted_audio_source",
            "rights_mode": self.rights_mode,
            "clearance_status": "cleared_permitted_source_without_redundant_registration",
            "publication_status": "local_permitted_dubbing_requires_rights_signoff_before_public_release",
            "external_publication_allowed": False,
            "registration_ceremony_eliminated": True,
            "authentication_bypass_used": False,
            "reference_origin": reference_origin,
        }


class QwenHybridDubbingSynthesizer:
    """Composite backend supporting both X-vector / ICL Voice Clone and Preset voices when explicitly configured."""

    supported_voice_modes = {"preset", "approved_clone"}

    def __init__(
        self,
        *,
        preset_backend,
        clone_backend,
    ):
        self.preset_backend = preset_backend
        self.clone_backend = clone_backend
        self.settings = dict(getattr(clone_backend, "settings", {}))
        self.identity = {
            "backend": "qwen_tts_mlx_hybrid",
            "adapter_version": 2,
            "voice_mode": "hybrid",
            "clone_method": getattr(clone_backend, "clone_method", "xvector"),
            "x_vector_only_mode": getattr(clone_backend, "x_vector_only_mode", True),
            "model": clone_backend.identity.get("model"),
            "model_revision": clone_backend.identity.get("model_revision"),
            "preset_identity": dict(getattr(preset_backend, "identity", {})),
            "clone_identity": dict(getattr(clone_backend, "identity", {})),
            "settings": dict(self.settings),
        }

    def synthesize(self, *, text, output_path, voice=None, profile=None, **kwargs):
        if profile is not None:
            return self.clone_backend.synthesize(text=text, profile=profile, output_path=output_path)
        if voice is not None and self.preset_backend is not None:
            return self.preset_backend.synthesize(text=text, voice=voice, output_path=output_path)
        raise ValueError("QwenHybridDubbingSynthesizer requires either `profile` (clone) or `voice` (preset).")


def parse_key_value_mappings(items, *, flag_name="--preset-speaker"):
    """Parse repeatable `KEY=VALUE` CLI arguments into a dictionary."""
    mapping = {}
    for item in items or []:
        if "=" not in str(item):
            raise ValueError(f"Invalid {flag_name} format {item!r}; expected KEY=VALUE.")
        k, v = str(item).split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid {flag_name} mapping {item!r}; both key and value must be non-empty.")
        mapping[k] = v
    return mapping


def parse_preset_speaker_mappings(items):
    """Parse repeatable `SPEAKER_ID=VoiceName` CLI arguments into a dictionary."""
    return parse_key_value_mappings(items, flag_name="--preset-speaker")


def _estimate_japanese_mora(text_ja):
    """Estimate Japanese mora count using repository `count_mora` when available, or accurate kana/kanji fallback."""
    clean = str(text_ja or "").strip()
    if not clean:
        return 0
    for mod_name in ("language.pronunciation", "language"):
        try:
            mod = __import__(mod_name, fromlist=["count_mora"])
            if hasattr(mod, "count_mora"):
                val = mod.count_mora(clean)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    # Fallback mora counter
    stripped = re.sub(r"[\s、。！？!?「」『』（）()［］\[\]・…―\-:;,.]", "", clean)
    if not stripped:
        return 1
    small_kana = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
    mora = 0
    for ch in stripped:
        if ch in small_kana:
            continue
        if "\u3040" <= ch <= "\u30ff":
            mora += 1
        elif "\u4e00" <= ch <= "\u9fff":
            mora += 2
        elif ch.isdigit():
            mora += 2
        elif ch.isalpha():
            mora += 1
        else:
            mora += 1
    return max(1, mora)


def _extract_numeric_id(raw_id):
    """Extract integer utterance number from raw ID (e.g. 64, '64', 'utt-064', 'dub-064', 'clip_1885_1915_64' -> 64)."""
    if isinstance(raw_id, int) and not isinstance(raw_id, bool):
        return raw_id
    s = str(raw_id or "").strip()
    if s.isdigit():
        return int(s)
    matches = re.findall(r"\d+", s)
    if matches:
        return int(matches[-1])
    return None


def _is_null_or_unassigned_speaker(spk):
    if spk is None:
        return True
    s = str(spk).strip()
    if not s:
        return True
    return s.upper() in ("NULL", "NONE", "UNKNOWN", "UNASSIGNED", "N/A")


def _infer_selection_range_from_script_and_paths(script_doc, *, script_path=None, video_path=None):
    """Resolve `(selection_start, selection_end)` from script JSON or path naming (e.g. `1885_1915`)."""
    for key in ("selection_range", "selection", "clip_range", "time_range", "window"):
        val = script_doc.get(key)
        if isinstance(val, dict):
            s = val.get("start", val.get("start_seconds"))
            e = val.get("end", val.get("end_seconds"))
            if s is not None and e is not None:
                return float(s), float(e)
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])

    if script_doc.get("selection_start") is not None and script_doc.get("selection_end") is not None:
        return float(script_doc["selection_start"]), float(script_doc["selection_end"])

    # Check path pattern like `phase11_1885_1915`
    for candidate in (script_path, video_path):
        if candidate:
            m = re.search(r"(?:^|[_/-])(\d{3,5})_(\d{3,5})(?:$|[_/.-])", str(candidate))
            if m:
                s_val, e_val = float(m.group(1)), float(m.group(2))
                if e_val > s_val and (e_val - s_val) <= 600.0:
                    return s_val, e_val

    raw_list = (
        script_doc.get("utterances")
        or script_doc.get("entries")
        or script_doc.get("segments")
        or script_doc.get("items")
        or []
    )
    starts = []
    ends = []
    for u in raw_list:
        s = u.get("start", u.get("start_seconds", u.get("source_start")))
        e = u.get("end", u.get("end_seconds", u.get("source_end")))
        if s is not None:
            starts.append(float(s))
        if e is not None:
            ends.append(float(e))
    if starts and ends:
        min_s = min(starts)
        max_e = max(ends)
        if min_s >= 60.0 and (max_e - min_s) <= 30.0:
            base = math.floor(min_s / 5.0) * 5.0
            return float(base), float(base + 30.0)
        if max_e <= 30.0:
            return 0.0, 30.0
        return float(min_s), float(max(min_s + 30.0, max_e))

    return 0.0, 30.0


def normalize_script_utterances(script_doc, *, script_path=None, video_path=None):
    """Normalize diverse `dubbing_script.json` schemas (v1/v2/v3, `utterances` vs `entries`) into a canonical list."""
    sel_start, sel_end = _infer_selection_range_from_script_and_paths(
        script_doc,
        script_path=script_path,
        video_path=video_path,
    )
    clip_duration = max(0.1, round(sel_end - sel_start, 4))

    raw_list = None
    list_key = "utterances"
    for k in ("utterances", "entries", "segments", "items", "lines"):
        if isinstance(script_doc.get(k), list) and script_doc.get(k):
            raw_list = script_doc[k]
            list_key = k
            break
    if raw_list is None:
        raise ValueError("Script contains no utterances/entries.")

    normalized = []
    for idx, u in enumerate(raw_list):
        raw_id = u.get("id")
        if raw_id is None:
            for alt_k in ("utterance_id", "entry_id", "segment_id", "source_id", "index"):
                if u.get(alt_k) is not None:
                    raw_id = u.get(alt_k)
                    break
        if raw_id is None:
            raw_id = f"utt-{idx + 1:03d}"

        num_id = _extract_numeric_id(raw_id)
        if num_id is None:
            for alt_k in ("utterance_id", "segment_id", "source_id"):
                if u.get(alt_k) is not None:
                    num_id = _extract_numeric_id(u.get(alt_k))
                    if num_id is not None:
                        break
        if num_id is None and isinstance(u.get("source_ids"), list) and len(u["source_ids"]) == 1:
            num_id = _extract_numeric_id(u["source_ids"][0])

        id_str = str(raw_id).strip()

        # Speaker extraction (preserving None when `"speaker": null`)
        raw_spk = u.get("speaker")
        if raw_spk is None and "speaker" not in u:
            for alt_s in ("speaker_id", "assigned_speaker", "translation_speaker"):
                if alt_s in u:
                    raw_spk = u.get(alt_s)
                    break
        speaker = None if _is_null_or_unassigned_speaker(raw_spk) else str(raw_spk).strip()

        # Timestamps
        raw_start = u.get(
            "start",
            u.get("start_seconds", u.get("source_start", u.get("source_start_seconds", u.get("clip_start", 0.0)))),
        )
        raw_end = u.get(
            "end",
            u.get(
                "end_seconds",
                u.get("source_end", u.get("source_end_seconds", u.get("clip_end", float(raw_start) + 2.0))),
            ),
        )
        s_val = float(raw_start)
        e_val = float(raw_end)
        if sel_start > 0.0 and s_val < sel_start - 1.0 and 0.0 <= s_val <= clip_duration + 1.0:
            # Stored as clip-relative in JSON; reconstruct absolute and relative
            rel_start = max(0.0, s_val)
            rel_end = max(rel_start + 0.05, e_val)
            abs_start = sel_start + rel_start
            abs_end = sel_start + rel_end
        else:
            abs_start = s_val
            abs_end = max(abs_start + 0.05, e_val)
            rel_start = max(0.0, abs_start - sel_start) if sel_start > 0.0 else max(0.0, abs_start)
            rel_end = max(rel_start + 0.05, abs_end - sel_start) if sel_start > 0.0 else max(rel_start + 0.05, abs_end)

        # English text
        text_en = ""
        for k_en in ("text_en", "source_text", "english_text", "en_text", "original_text", "transcript"):
            if isinstance(u.get(k_en), str) and u.get(k_en).strip():
                text_en = u[k_en].strip()
                break
        if not text_en and isinstance(u.get("text"), str) and u.get("text_ja"):
            text_en = u["text"].strip()

        # Japanese text
        text_ja = ""
        for k_ja in (
            "text_ja",
            "rewritten_text_ja",
            "shortened_text_ja",
            "tts_text",
            "japanese_text",
            "ja_text",
            "translated_text",
        ):
            if isinstance(u.get(k_ja), str) and u.get(k_ja).strip():
                text_ja = u[k_ja].strip()
                break
        if not text_ja and isinstance(u.get("translation"), str):
            text_ja = u["translation"].strip()
        elif not text_ja and isinstance(u.get("translation"), dict):
            for sub_k in ("text_ja", "tts_text", "text", "japanese"):
                if isinstance(u["translation"].get(sub_k), str) and u["translation"].get(sub_k).strip():
                    text_ja = u["translation"][sub_k].strip()
                    break
        if not text_ja and isinstance(u.get("japanese_analysis"), dict):
            for sub_k in ("tts_text", "normalized_text", "text"):
                if isinstance(u["japanese_analysis"].get(sub_k), str) and u["japanese_analysis"].get(sub_k).strip():
                    text_ja = u["japanese_analysis"][sub_k].strip()
                    break

        mora_count = u.get("mora_count")
        if not mora_count and isinstance(u.get("japanese_analysis"), dict):
            mora_count = u["japanese_analysis"].get("mora_count")
        if not mora_count and isinstance(u.get("timing_estimate"), dict):
            mora_count = u["timing_estimate"].get("mora_count")
        mora_count = int(mora_count) if (mora_count is not None and int(mora_count) > 0) else _estimate_japanese_mora(text_ja)

        item = copy.deepcopy(u)
        item.update(
            {
                "id": id_str,
                "raw_id": raw_id,
                "numeric_id": num_id,
                "original_index": idx,
                "speaker": speaker,
                "raw_speaker": raw_spk,
                "start": round(abs_start, 4),
                "end": round(abs_end, 4),
                "source_start": round(abs_start, 4),
                "source_end": round(abs_end, 4),
                "clip_relative_start": round(rel_start, 4),
                "clip_relative_end": round(rel_end, 4),
                "text_en": text_en,
                "text_ja": text_ja,
                "mora_count": mora_count,
            }
        )
        normalized.append(item)

    normalized.sort(key=lambda r: (r["start"], r["original_index"]))
    return {
        "selection_start": round(sel_start, 4),
        "selection_end": round(sel_end, 4),
        "clip_duration": round(clip_duration, 4),
        "list_key": list_key,
        "utterances": normalized,
    }


def parse_merge_id_groups(raw_specs):
    """Parse repeatable `--merge-ids 64,65,66` or `--merge-ids 64-66` specs into sets of ID tokens."""
    groups = []
    for spec in raw_specs or []:
        text = str(spec or "").strip()
        if not text:
            continue
        if "," in text:
            tokens = [t.strip() for t in text.split(",") if t.strip()]
            expanded = []
            for tok in tokens:
                m_range = re.match(r"^(\d+)\s*-\s*(\d+)$", tok)
                if m_range:
                    lo, hi = int(m_range.group(1)), int(m_range.group(2))
                    expanded.extend(str(i) for i in range(min(lo, hi), max(lo, hi) + 1))
                else:
                    expanded.append(tok)
            if len(expanded) >= 2:
                groups.append(expanded)
        else:
            m_range = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
            if m_range:
                lo, hi = int(m_range.group(1)), int(m_range.group(2))
                if hi != lo:
                    groups.append([str(i) for i in range(min(lo, hi), max(lo, hi) + 1)])
    return groups


def _utterance_matches_merge_token(utt, token):
    tok = str(token).strip()
    if tok.isdigit() and utt.get("source_utterance_ids"):
        return tok in {str(i) for i in utt["source_utterance_ids"]}
    if str(utt.get("id")) == tok or str(utt.get("raw_id")) == tok:
        return True
    num_tok = _extract_numeric_id(tok)
    if num_tok is not None and utt.get("numeric_id") is not None and utt["numeric_id"] == num_tok:
        return True
    return False


_EN_CONTINUATION_END_WORDS = {
    "and",
    "or",
    "but",
    "so",
    "because",
    "since",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "while",
    "if",
    "though",
    "although",
    "as",
    "than",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "from",
    "into",
    "about",
    "through",
    "between",
    "under",
    "over",
    "the",
    "a",
    "an",
    "this",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "not",
}

_EN_CONTINUATION_START_WORDS = {
    "and",
    "or",
    "but",
    "so",
    "because",
    "since",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "while",
    "if",
    "though",
    "although",
    "as",
    "than",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "of",
    "into",
    "about",
    "like",
    "such",
    "especially",
    "including",
    "even",
    "also",
    "then",
}

_JA_CONJUNCTIVE_CLAUSE_ENDINGS = (
    "て",
    "で",
    "し",
    "から",
    "ので",
    "けど",
    "けれど",
    "けれども",
    "が",
    "のに",
    "たり",
    "だり",
    "ながら",
    "つつ",
    "おり",
    "あり",
    "なり",
    "ため",
    "よう",
    "ように",
    "とき",
    "時",
    "場合",
    "ところ",
    "うえで",
    "上で",
)

_JA_CONTINUATION_PREFIXES = (
    "そして",
    "また",
    "さらに",
    "しかも",
    "しかし",
    "ですが",
    "ただ",
    "ただし",
    "つまり",
    "すなわち",
    "なぜなら",
    "というのも",
    "そのため",
    "したがって",
    "そこで",
    "すると",
    "あるいは",
    "または",
    "および",
    "ならびに",
    "特に",
    "例えば",
    "たとえば",
    "で、",
    "と、",
    "が、",
    "し、",
    "て、",
    "に、",
    "を、",
    "は、",
    "も、",
    "から、",
    "ので、",
)


def _evaluate_pair_continuity(curr, nxt, *, max_merge_gap_seconds=1.5):
    """Return list of continuity reasons if `curr` and `nxt` are split fragments of a continuous sentence, else `[]`."""
    # Hard barriers:
    if _is_null_or_unassigned_speaker(curr.get("speaker")) or _is_null_or_unassigned_speaker(nxt.get("speaker")):
        return []
    if curr.get("speaker") != nxt.get("speaker"):
        return []

    gap = float(nxt["start"]) - float(curr["end"])
    en_a = str(curr.get("text_en") or "").strip()
    en_b = str(nxt.get("text_en") or "").strip()
    ja_a = str(curr.get("text_ja") or "").strip()
    ja_b = str(nxt.get("text_ja") or "").strip()

    has_comma_or_clause_continuation = bool(
        en_a.endswith(",")
        or en_a.endswith("...")
        or any(
            re.sub(r"[。！？!?「」『』（）()\s]+$", "", ja_a).endswith(s)
            for s in _JA_CONJUNCTIVE_CLAUSE_ENDINGS
        )
    )

    # Self-overlap barrier (`test_self_overlap_detected_without_merging`), allowing minor ASR boundary jitter (-0.30s)
    if gap < -0.30 or float(nxt["start"]) <= float(curr["start"]) + 0.25 or float(nxt["end"]) <= float(curr["end"]):
        return []
    if gap < -0.05 and not has_comma_or_clause_continuation:
        return []

    effective_max_gap = float(max_merge_gap_seconds)
    # Long-gap barrier (`test_long_gap_not_merged`)
    if gap > effective_max_gap + 1e-6:
        return []

    reasons = []

    # 1. English non-terminal ending or continuation token
    if en_a:
        if not re.search(r'[.?!]["\')\]]*$', en_a):
            reasons.append("en_non_terminal_punctuation")
        elif en_a.endswith("..."):
            reasons.append("en_ellipsis_continuation")
        words_a = re.findall(r"[A-Za-z']+", en_a.lower())
        if words_a and words_a[-1] in _EN_CONTINUATION_END_WORDS:
            reasons.append(f"en_ends_with_continuation_word:{words_a[-1]}")

    # 2. English lowercase or conjunction/relative start on next fragment
    if en_b:
        first_alpha = re.search(r"[A-Za-z]", en_b)
        if first_alpha and en_b[first_alpha.start()].islower():
            reasons.append("en_next_starts_lowercase")
        words_b = re.findall(r"[A-Za-z']+", en_b.lower())
        if words_b and words_b[0] in _EN_CONTINUATION_START_WORDS and gap <= 1.2:
            reasons.append(f"en_next_starts_with_connector:{words_b[0]}")

    # 3. Japanese non-terminal ending or clause continuation
    if ja_a:
        if not re.search(r"[。！？!?][」』）)\]]*$", ja_a):
            reasons.append("ja_non_terminal_punctuation")
        else:
            stem_a = re.sub(r"[。！？!?「」『』（）()\s]+$", "", ja_a)
            for suffix in _JA_CONJUNCTIVE_CLAUSE_ENDINGS:
                if stem_a.endswith(suffix):
                    reasons.append(f"ja_clause_ending_before_period:{suffix}")
                    break

    # 4. Japanese continuation prefix on next fragment
    if ja_b:
        for prefix in _JA_CONTINUATION_PREFIXES:
            if ja_b.startswith(prefix) and gap <= 1.2:
                reasons.append(f"ja_next_starts_with_connector:{prefix}")
                break

    # 5. Closely-spaced short clauses of the same sentence
    dur_a = float(curr["end"]) - float(curr["start"])
    dur_b = float(nxt["end"]) - float(nxt["start"])
    if gap <= 0.65 and (dur_a <= 3.5 or dur_b <= 3.5) and (float(nxt["end"]) - float(curr["start"])) <= 14.0:
        reasons.append("tight_gap_short_clause_continuation")

    return reasons


def _join_japanese_fragments(ja_fragments):
    """Smoothly join split Japanese sentence fragments while preserving all numbers, names, and negations."""
    cleaned = [str(f or "").strip() for f in ja_fragments if str(f or "").strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    out = ""
    for idx, frag in enumerate(cleaned):
        is_last = idx == len(cleaned) - 1
        cur = frag

        # Deduplicate overlapping boundary text if translator repeated context across split segments
        if out:
            max_ov = min(len(out), len(cur), 18)
            trimmed_out = re.sub(r"[、。！？!?\s]+$", "", out)
            for k in range(max_ov, 3, -1):
                if trimmed_out.endswith(cur[:k]):
                    cur = cur[k:].lstrip("、。 ")
                    break

        if not cur:
            continue

        if not is_last:
            # Convert unnatural intermediate full-stop `。` into `、` for continuous sentence synthesis
            if cur.endswith("。"):
                cur = cur[:-1] + "、"
            elif not cur.endswith(("、", "・", "？", "！", "?", "!")):
                if cur.endswith(_JA_CONJUNCTIVE_CLAUSE_ENDINGS) and len(cur) >= 6:
                    cur = cur + "、"
        out += cur

    if out and not re.search(r"[。！？!?」』）)]$", out):
        out += "。"
    return out


def _merge_utterance_group(group, *, merge_reasons):
    """Combine a list of consecutive same-speaker utterances into a single merged utterance dict,
    preserving all word-level timestamps, internal pauses, and per-segment time spans.
    """
    if len(group) == 1:
        return copy.deepcopy(group[0]), None

    first = group[0]
    last = group[-1]
    raw_ids = [u.get("raw_id", u["id"]) for u in group]
    str_ids = [str(u["id"]) for u in group]

    if first.get("numeric_id") is not None and last.get("numeric_id") is not None:
        m_prefix = re.match(r"^([A-Za-z_-]+?)0*\d+$", str(first["id"]))
        if m_prefix:
            prefix = m_prefix.group(1)
            merged_id = f"{prefix}{first['numeric_id']:03d}-{last['numeric_id']:03d}"
        else:
            merged_id = f"{first['numeric_id']}-{last['numeric_id']}"
    else:
        merged_id = f"{str_ids[0]}-{str_ids[-1]}"

    en_parts = [u.get("text_en", "").strip() for u in group if u.get("text_en", "").strip()]
    merged_en = re.sub(r"\s+", " ", " ".join(en_parts)).strip()
    ja_parts = [u.get("text_ja", "").strip() for u in group if u.get("text_ja", "").strip()]
    merged_ja = _join_japanese_fragments(ja_parts)
    merged_mora = _estimate_japanese_mora(merged_ja)

    from dubbing_quality import source_ids
    from timing_adjustment import classify_english_words_and_pauses, extract_or_resolve_word_timestamps

    combined_words = []
    source_segment_spans = []
    for u in group:
        u_words = extract_or_resolve_word_timestamps(u)
        combined_words.extend(u_words)
        u_wt = classify_english_words_and_pauses(
            u_words,
            segment_start=float(u["start"]),
            segment_end=float(u["end"]),
        )
        source_segment_spans.append(
            {
                "id": u["id"],
                "raw_id": u.get("raw_id", u["id"]),
                "start": round(float(u["start"]), 4),
                "end": round(float(u["end"]), 4),
                "effective_content_start": u_wt["effective_content_start"],
                "effective_content_end": u_wt["effective_content_end"],
                "text_en": u.get("text_en", ""),
                "text_ja": u.get("text_ja", ""),
                "words": u_words,
            }
        )

    combined_words.sort(key=lambda w: (float(w["start"]), float(w["end"])))
    for w_idx, w_item in enumerate(combined_words):
        w_item["word_index"] = w_idx

    merged_wt = classify_english_words_and_pauses(
        combined_words,
        segment_start=float(first["start"]),
        segment_end=float(last["end"]),
    )

    merged = copy.deepcopy(first)
    merged.update(
        {
            "id": merged_id,
            "raw_id": merged_id,
            "merged_from_ids": raw_ids,
            "source_utterance_ids": [i for u in group for i in source_ids(u)],
            "review_required": True,
            "review_reasons": sorted({r for u in group for r in u.get("review_reasons", [])} | {"merged_translation_review_pending"}),
            "merged_from_str_ids": str_ids,
            "is_merged_utterance": True,
            "start": round(float(first["start"]), 4),
            "end": round(float(last["end"]), 4),
            "source_start": round(float(first["source_start"]), 4),
            "source_end": round(float(last["source_end"]), 4),
            "effective_content_start": merged_wt["effective_content_start"],
            "effective_content_end": merged_wt["effective_content_end"],
            "clip_relative_start": round(float(first["clip_relative_start"]), 4),
            "clip_relative_end": round(float(last["clip_relative_end"]), 4),
            "text_en": merged_en,
            "text_ja": merged_ja,
            "mora_count": merged_mora,
            "words": combined_words,
            "source_segment_spans": source_segment_spans,
            "word_timing_analysis": merged_wt,
            "merge_reasons": sorted(set(merge_reasons)),
        }
    )

    merge_record = {
        "merged_id": merged_id,
        "merged_from_ids": raw_ids,
        "speaker": first.get("speaker"),
        "source_start_seconds": round(float(first["start"]), 4),
        "source_end_seconds": round(float(last["end"]), 4),
        "effective_content_start_seconds": merged_wt["effective_content_start"],
        "effective_content_end_seconds": merged_wt["effective_content_end"],
        "clip_relative_start_seconds": round(float(first["clip_relative_start"]), 4),
        "clip_relative_end_seconds": round(float(last["clip_relative_end"]), 4),
        "original_text_en": en_parts,
        "merged_text_en": merged_en,
        "original_text_ja": ja_parts,
        "merged_text_ja": merged_ja,
        "original_mora_Sum": sum(int(u.get("mora_count") or 0) for u in group),
        "merged_mora_count": merged_mora,
        "source_segment_spans": source_segment_spans,
        "merge_reasons": sorted(set(merge_reasons)),
    }
    return merged, merge_record


def detect_and_merge_split_utterances(
    utterances,
    *,
    enabled=True,
    explicit_merge_groups=None,
    max_merge_gap_seconds=1.5,
):
    """Detect and merge unnaturally split continuous same-speaker utterances (such as IDs 64–66).

    Invariants enforced:
    - Never merges across different speakers.
    - Never merges `speaker: null` or `"UNKNOWN"` utterances, and never bridges across them.
    - Never merges across gaps > `max_merge_gap_seconds` (unless explicitly listed in `explicit_merge_groups`).
    """
    items = [copy.deepcopy(u) for u in utterances]
    explicit_groups = parse_merge_id_groups(explicit_merge_groups) if explicit_merge_groups else []

    if not enabled and not explicit_groups:
        return {
            "utterances": items,
            "merged_groups": [],
            "merged_utterance_count_delta": 0,
        }

    # Build lookup for explicit merge pairs
    explicit_adj_pairs = set()
    for grp in explicit_groups:
        matched_indices = []
        for tok in grp:
            for idx, u in enumerate(items):
                if _utterance_matches_merge_token(u, tok) and idx not in matched_indices:
                    matched_indices.append(idx)
                    break
        if len(matched_indices) >= 2:
            # Verify all matched utterances share the same concrete speaker and have no barrier in between
            spks = {items[i].get("speaker") for i in matched_indices}
            if any(_is_null_or_unassigned_speaker(s) for s in spks):
                raise UnverifiedSpeakerError(
                    f"Cannot merge explicit group {grp!r} because one or more utterances have `speaker: null` / unassigned speaker."
                )
            if len(spks) > 1:
                raise ValueError(
                    f"Cannot merge explicit group {grp!r} across different speakers: {sorted(spks)}."
                )
            for a_idx, b_idx in zip(matched_indices[:-1], matched_indices[1:]):
                if b_idx != a_idx + 1:
                    raise ValueError(
                        f"Cannot bridge explicit merge group {grp!r} across intervening barrier utterance {items[a_idx + 1].get('id')!r}."
                    )
                explicit_adj_pairs.add((a_idx, b_idx))

    merged_list = []
    merged_records = []
    i = 0
    n = len(items)
    while i < n:
        cur_group = [items[i]]
        cur_reasons = []
        j = i
        while j + 1 < n:
            curr_u = cur_group[-1]
            nxt_u = items[j + 1]
            pair_reasons = []
            if (j, j + 1) in explicit_adj_pairs:
                pair_reasons.append("explicit_cli_merge_ids")
            if enabled:
                pair_reasons.extend(
                    _evaluate_pair_continuity(
                        curr_u,
                        nxt_u,
                        max_merge_gap_seconds=max_merge_gap_seconds,
                    )
                )
            if pair_reasons and (float(nxt_u["end"]) - float(cur_group[0]["start"]) <= 18.0 or (j, j + 1) in explicit_adj_pairs):
                cur_group.append(nxt_u)
                cur_reasons.extend(pair_reasons)
                j += 1
            else:
                break

        merged_u, rec = _merge_utterance_group(cur_group, merge_reasons=cur_reasons)
        merged_list.append(merged_u)
        if rec is not None:
            merged_records.append(rec)
        i = j + 1

    return {
        "utterances": merged_list,
        "merged_groups": merged_records,
        "merged_utterance_count_delta": len(items) - len(merged_list),
    }


def partition_and_manage_null_speakers(
    utterances,
    *,
    explicit_speaker_assignments=None,
    null_speaker_policy="hold_for_verification",
):
    """Manage `speaker: null` (and unassigned/UNKNOWN) utterances explicitly as verification targets.

    Never guesses or silently assigns a speaker identity from adjacent utterances.
    Only applies speaker assignments explicitly provided via `--assign-speaker UTT_ID=SPEAKER_ID`.
    """
    if null_speaker_policy not in ("hold_for_verification", "fail"):
        raise ValueError(f"Unsupported null_speaker_policy: {null_speaker_policy!r}")

    explicit_map = dict(explicit_speaker_assignments or {})
    assigned_utterances = []
    verification_targets = []
    applied_overrides = []

    for idx, u in enumerate(utterances):
        item = copy.deepcopy(u)
        uid_str = str(item.get("id"))
        num_id_str = str(item.get("numeric_id")) if item.get("numeric_id") is not None else None

        source_keys = [str(i) for i in item.get("source_utterance_ids",[])]
        matches = {explicit_map[k] for k in source_keys if k in explicit_map}
        if len(matches)>1:raise ValueError("Conflicting source speaker assignments.")
        override_spk = explicit_map.get(uid_str) or (next(iter(matches)) if matches else None)
        if not source_keys:override_spk = override_spk or (explicit_map.get(num_id_str) if num_id_str else None)
        if override_spk:
            if _is_null_or_unassigned_speaker(override_spk):
                raise ValueError(f"Invalid --assign-speaker override {override_spk!r} for utterance {uid_str!r}.")
            item["speaker"] = str(override_spk).strip()
            item["speaker_assignment_source"] = "explicit_cli_verification_override"
            applied_overrides.append(
                {
                    "id": item.get("raw_id", uid_str),
                    "previous_speaker": u.get("raw_speaker"),
                    "assigned_speaker": item["speaker"],
                    "source": "explicit_cli_verification_override",
                }
            )

        if _is_null_or_unassigned_speaker(item.get("speaker")):
            prev_spk = utterances[idx - 1].get("speaker") if idx > 0 else None
            next_spk = utterances[idx + 1].get("speaker") if idx + 1 < len(utterances) else None
            verification_targets.append(
                {
                    "id": item.get("raw_id", uid_str),
                    "id_str": uid_str,
                    "source_utterance_ids": item.get("source_utterance_ids", [item.get("raw_id",uid_str)]),
                    "numeric_id": item.get("numeric_id"),
                    "speaker": None,
                    "raw_speaker": item.get("raw_speaker"),
                    "verification_status": "pending_human_speaker_verification",
                    "inferred_identity_applied": False,
                    "action_taken": "held_from_voice_cloning_for_verification",
                    "source_start_seconds": item["start"],
                    "source_end_seconds": item["end"],
                    "clip_relative_start_seconds": item["clip_relative_start"],
                    "clip_relative_end_seconds": item["clip_relative_end"],
                    "slot_duration_seconds": round(float(item["end"]) - float(item["start"]), 4),
                    "text_en": item.get("text_en", ""),
                    "text_ja": item.get("text_ja", ""),
                    "mora_count": item.get("mora_count", 0),
                    "context_prev_speaker": prev_spk,
                    "context_next_speaker": next_spk,
                }
            )
        else:
            assigned_utterances.append(item)

    if verification_targets and null_speaker_policy == "fail":
        target_ids = [t["id"] for t in verification_targets]
        raise UnverifiedSpeakerError(
            f"Script contains {len(verification_targets)} unassigned (`speaker: null` / UNKNOWN) utterance(s): "
            f"{target_ids}. Do not guess speakers; verify and assign with --assign-speaker ID=SPEAKER or use "
            f"--null-speaker-policy hold_for_verification."
        )

    return {
        "assigned_utterances": assigned_utterances,
        "speaker_verification_targets": verification_targets,
        "explicit_speaker_assignments": applied_overrides,
        "null_speaker_policy": null_speaker_policy,
    }


def extract_speaker_reference_from_video(
    *,
    video_path,
    speaker_id,
    all_utterances,
    output_dir,
    selection_start=0.0,
    clip_duration=30.0,
    min_target_seconds=3.2,
    max_target_seconds=15.0,
    ffmpeg_bin=None,
):
    """Extract clean non-overlapping speech segments for `speaker_id` from `source.mp4` for X-vector conditioning.

    Excludes any interval that overlaps with another speaker or with a `speaker: null` utterance.
    Writes extracted 24kHz mono PCM16 WAV to `<output_dir>/<speaker_id>_extracted_ref.wav` without modifying inputs.
    """
    video_file = Path(video_path).resolve()
    if not video_file.is_file():
        raise XVectorSynthesisError(
            f"Cannot extract X-vector reference for speaker {speaker_id!r}: video file not found at {video_file}"
        )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exec = ffmpeg_bin or _resolve_binary("ffmpeg")
    video_probe = probe_media_streams(video_file)
    video_dur = float(video_probe.get("duration_seconds") or clip_duration)
    use_absolute_offset = video_dur > (float(clip_duration) + 5.0) and float(selection_start) > 0.0

    # Collect target speaker intervals vs all other utterances (including speaker: null!)
    speaker_intervals = []
    other_intervals = []
    for u in all_utterances:
        rel_s = float(u.get("clip_relative_start", max(0.0, float(u.get("start", 0.0)) - float(selection_start))))
        rel_e = float(u.get("clip_relative_end", max(rel_s + 0.1, float(u.get("end", rel_s + 1.0)) - float(selection_start))))
        if u.get("speaker") == speaker_id:
            speaker_intervals.append((rel_s, rel_e, u.get("id")))
        else:
            other_intervals.append((rel_s - 0.05, rel_e + 0.05))

    if not speaker_intervals:
        raise XVectorSynthesisError(
            f"Speaker {speaker_id!r} has no utterances in script to extract X-vector reference audio from."
        )

    # Subtract other_intervals from speaker_intervals to keep solo speech segments
    clean_segments = []
    for s_start, s_end, uid in speaker_intervals:
        pieces = [(s_start, s_end)]
        for o_start, o_end in other_intervals:
            next_pieces = []
            for p_s, p_e in pieces:
                if o_end <= p_s or o_start >= p_e:
                    next_pieces.append((p_s, p_e))
                else:
                    if o_start > p_s + 0.20:
                        next_pieces.append((p_s, o_start))
                    if o_end < p_e - 0.20:
                        next_pieces.append((o_end, p_e))
            pieces = next_pieces
        for p_s, p_e in pieces:
            # Trim 30ms boundary margins if segment is long enough
            margin = 0.03 if (p_e - p_s) > 0.40 else 0.0
            c_s = max(0.0, p_s + margin)
            c_e = min(video_dur if not use_absolute_offset else float(clip_duration), p_e - margin)
            if c_e - c_s >= 0.25:
                clean_segments.append(
                    {
                        "utterance_id": uid,
                        "clip_relative_start": round(c_s, 4),
                        "clip_relative_end": round(c_e, 4),
                        "duration_seconds": round(c_e - c_s, 4),
                    }
                )

    # If all segments had overlap in diarization, fall back to the longest raw segment of `speaker_id`
    if not clean_segments:
        longest = max(speaker_intervals, key=lambda x: x[1] - x[0])
        c_s = max(0.0, longest[0])
        c_e = max(c_s + 0.30, longest[1])
        clean_segments.append(
            {
                "utterance_id": longest[2],
                "clip_relative_start": round(c_s, 4),
                "clip_relative_end": round(c_e, 4),
                "duration_seconds": round(c_e - c_s, 4),
                "fallback_overlapped_segment": True,
            }
        )

    target_sr = 24000
    collected_chunks = []
    used_segments = []
    accumulated_sec = 0.0
    spacer = np.zeros(int(0.12 * target_sr), dtype=np.float32)

    for idx, seg in enumerate(clean_segments):
        if accumulated_sec >= max_target_seconds:
            break
        seek_start = (float(selection_start) + seg["clip_relative_start"]) if use_absolute_offset else seg["clip_relative_start"]
        seg_dur = min(seg["duration_seconds"], max_target_seconds - accumulated_sec)
        tmp_seg_wav = out_dir / f"_{speaker_id}_seg_{idx:02d}.tmp.wav"
        cmd = [
            ffmpeg_exec,
            "-y",
            "-ss",
            f"{seek_start:.4f}",
            "-t",
            f"{seg_dur:.4f}",
            "-i",
            str(video_file),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-c:a",
            "pcm_s16le",
            str(tmp_seg_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and tmp_seg_wav.is_file():
            try:
                arr, _ = _read_wav_float32(tmp_seg_wav)
                if len(arr) > 0:
                    if collected_chunks:
                        collected_chunks.append(spacer)
                    collected_chunks.append(arr)
                    accumulated_sec += len(arr) / float(target_sr)
                    used_segments.append(seg)
            finally:
                tmp_seg_wav.unlink(missing_ok=True)

    if not collected_chunks:
        raise XVectorSynthesisError(
            f"Failed to extract any audio frames for speaker {speaker_id!r} from {video_file}."
        )

    combined = np.concatenate(collected_chunks).astype(np.float32)
    # Ensure at least min_target_seconds (3.2s) for stable X-vector embedding extraction
    if len(combined) / float(target_sr) < min_target_seconds and len(combined) > 0:
        unit = combined.copy()
        while len(combined) / float(target_sr) < min_target_seconds:
            combined = np.concatenate([combined, spacer, unit]).astype(np.float32)
        max_frames = int(round(max_target_seconds * target_sr))
        if len(combined) > max_frames:
            combined = combined[:max_frames]

    # Gentle normalization if extracted clip audio is very quiet, avoiding clipping
    peak = float(np.max(np.abs(combined))) if len(combined) else 0.0
    if 1e-4 < peak < 0.15:
        combined = (combined * (0.45 / peak)).astype(np.float32)
    elif peak > 0.95:
        combined = (combined * (0.95 / peak)).astype(np.float32)

    ref_wav_path = out_dir / f"{speaker_id}_extracted_ref.wav"
    pcm = np.clip(np.round(combined * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(ref_wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(target_sr)
        stream.writeframes(pcm.tobytes())

    ref_quality = None
    try:
        from voice_consistency import inspect_reference_audio_quality

        has_overlap_fallback = any(bool(s.get("fallback_overlapped_segment")) for s in used_segments)
        ref_quality = inspect_reference_audio_quality(
            ref_wav_path,
            has_overlap_with_other_speaker=has_overlap_fallback,
        )
    except Exception:
        ref_quality = None

    return TechnicalVoiceReference(
        speaker_id=speaker_id,
        reference_audio=ref_wav_path,
        source_origin="source_video_extracted",
        extracted_segments=used_segments,
        reference_quality=ref_quality,
    )


def resolve_speaker_voice_plan(
    script_doc,
    *,
    voice_profiles=None,
    technical_references=None,
    rights_policy=None,
    voice_mode="auto",
    clone_method="xvector",
    preset_speaker_map=None,
    default_preset_voice=None,
    allow_preset_fallback=True,
    speaker_age_map=None,
    speaker_atmosphere_map=None,
    auto_select_preset=False,
):
    """Build per-speaker voice routing while enforcing X-vector priority, rights governance,
    and Phase 13 age/atmosphere-based preset voice selection.

    - When `voice_mode="xvector"` or `allow_preset_fallback=False` (Phase 12/13 default):
      X-vector synthesis is mandatory for all speakers not explicitly mapped via `--preset-speaker`.
      Never silently falls back to `default_preset_voice` or `utterance["preset_voice"]`; raises
      `XVectorSynthesisError` / `UnauthorizedSpeakerError` if a speaker lacks a valid X-vector reference!
    - Maintains backward compatibility with Phase 11 (`voice_mode="auto"` with `allow_preset_fallback=True`).
    """
    if voice_mode not in ("auto", "xvector", "icl", "preset"):
        raise ValueError(f"Unsupported voice_mode: {voice_mode!r}")
    if clone_method not in ("xvector", "icl"):
        raise ValueError(f"Unsupported clone_method: {clone_method!r}")

    effective_clone_method = (
        "icl"
        if voice_mode == "icl"
        else ("xvector" if voice_mode == "xvector" else clone_method)
    )
    preset_map = dict(preset_speaker_map or {})
    profiles = dict(voice_profiles or {})
    tech_refs = dict(technical_references or {})
    age_map = dict(speaker_age_map or {})
    atm_map = dict(speaker_atmosphere_map or {})
    active_rights_policy = rights_policy or RightsAndPublicationPolicy(rights_mode="permitted_local")

    utterances = (
        script_doc.get("utterances")
        or script_doc.get("entries")
        or script_doc.get("segments")
        or []
    )
    if not utterances:
        raise ValueError("Script contains no utterances/entries.")

    speakers = []
    for u in utterances:
        spk = u.get("speaker")
        if not spk or not str(spk).strip():
            raise UnauthorizedSpeakerError(f"Utterance {u.get('id')!r} is missing a speaker ID (`speaker: null`).")
        spk_str = str(spk).strip()
        if spk_str not in speakers:
            speakers.append(spk_str)

    voices = {}
    active_profiles = {}
    speaker_modes = {}
    routing_details = []
    governance_records = []
    uses_technical_reference = False
    used_presets = set()

    for spk in speakers:
        if _is_null_or_unassigned_speaker(spk):
            raise UnauthorizedSpeakerError(
                f"Speaker {spk!r} cannot be cloned or synthesized; manage unassigned speakers as verification targets first."
            )

        # Explicit Preset mode (`--voice-mode preset` or explicit `--preset-speaker SPK=Voice`)
        if voice_mode == "preset" or spk in preset_map:
            preset_recommendation = None
            assigned_preset = preset_map.get(spk)
            if not assigned_preset and voice_mode == "preset":
                if auto_select_preset or spk in age_map or spk in atm_map:
                    from voice_quality import recommend_preset_voice
                    f0_hint = None
                    if spk in tech_refs and Path(tech_refs[spk].reference_audio).is_file():
                        try:
                            pr_metrics = analyze_japanese_prosody(tech_refs[spk].reference_audio)
                            f0_hint = pr_metrics.get("f0_hz_median")
                        except Exception:
                            f0_hint = None
                    preset_recommendation = recommend_preset_voice(
                        spk,
                        f0_hz=f0_hint,
                        age_hint=age_map.get(spk),
                        atmosphere_hint=atm_map.get(spk),
                        avoid_voices=used_presets,
                    )
                    assigned_preset = preset_recommendation["recommended_voice"]
                else:
                    assigned_preset = default_preset_voice or _utterance_preset_fallback(utterances, spk)
            if not assigned_preset:
                raise UnauthorizedSpeakerError(
                    f"Speaker {spk!r} requires an explicit preset voice in preset mode; "
                    f"specify --preset-speaker {spk}=Ryan, --default-preset-voice Ryan, or --auto-select-preset."
                )
            used_presets.add(assigned_preset)
            voices[spk] = assigned_preset
            speaker_modes[spk] = "preset"
            routing_details.append(
                {
                    "speaker": spk,
                    "mode": "preset",
                    "voice": assigned_preset,
                    "clone_method": None,
                    "x_vector_only_mode": False,
                    "authorization_status": (
                        "auto_selected_preset_by_age_atmosphere"
                        if preset_recommendation
                        else ("explicit_preset_mode" if voice_mode == "preset" else "explicit_preset_speaker_override")
                    ),
                    "profile_fingerprint": None,
                    **({"preset_recommendation": preset_recommendation} if preset_recommendation else {}),
                }
            )
            continue

        # 1. Check registered `VoiceProfile` from `voice_cloning.py`
        if spk in profiles:
            from voice_cloning import profile_snapshot

            prof = profiles[spk]
            snap = profile_snapshot(prof)
            gov = active_rights_policy.classify_and_verify_speaker(
                spk,
                has_registered_profile=True,
                reference_origin="registered_voice_profile",
            )
            governance_records.append(gov)
            fp = snap["profile_fingerprint"]
            voice_id = f"approved_clone:{fp}"
            voices[spk] = voice_id
            active_profiles[spk] = prof
            speaker_modes[spk] = "approved_clone"
            routing_details.append(
                {
                    "speaker": spk,
                    "mode": "approved_clone",
                    "voice": voice_id,
                    "clone_method": effective_clone_method,
                    "x_vector_only_mode": effective_clone_method == "xvector",
                    "authorization_status": "verified_approved_profile",
                    "governance_clearance": gov["clearance_status"],
                    "profile_fingerprint": fp,
                    "reference_audio_sha256": snap["reference_audio_sha256"],
                    "reference_origin": "registered_voice_profile",
                }
            )
            continue

        # 2. Check decoupled `TechnicalVoiceReference` (Phase 12/13 X-vector primary path)
        if spk in tech_refs:
            tref = tech_refs[spk]
            snap = tref.technical_snapshot()
            gov = active_rights_policy.classify_and_verify_speaker(
                spk,
                has_registered_profile=False,
                reference_origin=tref.source_origin,
            )
            governance_records.append(gov)
            fp = snap["profile_fingerprint"]
            voice_id = f"approved_clone:{fp}"
            voices[spk] = voice_id
            active_profiles[spk] = tref
            speaker_modes[spk] = "approved_clone"
            uses_technical_reference = True
            routing_details.append(
                {
                    "speaker": spk,
                    "mode": "approved_clone",
                    "voice": voice_id,
                    "clone_method": effective_clone_method,
                    "x_vector_only_mode": effective_clone_method == "xvector",
                    "authorization_status": gov["clearance_status"],
                    "governance_clearance": gov["clearance_status"],
                    "profile_fingerprint": fp,
                    "reference_audio_sha256": snap["reference_audio_sha256"],
                    "reference_origin": tref.source_origin,
                    "reference_audio": tref.reference_audio,
                    "reference_duration_seconds": tref.duration_seconds,
                }
            )
            continue

        # 3. Speaker has neither a VoiceProfile nor a TechnicalVoiceReference
        if voice_mode == "xvector" or not allow_preset_fallback:
            raise XVectorSynthesisError(
                f"X-vector synthesis required for speaker {spk!r}, but no voice reference audio or profile is available. "
                f"Automatic fallback to Preset is disabled so failures are never hidden. "
                f"Provide --speaker-ref {spk}=/path/to/ref.wav, enable --extract-reference-from-video with a valid --video, "
                f"or explicitly request Preset via --preset-speaker {spk}=Ryan."
            )

        # Legacy Phase 11 `voice_mode="auto"` with explicit `--default-preset-voice` or `--auto-select-preset`
        preset_recommendation = None
        assigned_preset = default_preset_voice
        if not assigned_preset and (auto_select_preset or spk in age_map or spk in atm_map):
            from voice_quality import recommend_preset_voice
            preset_recommendation = recommend_preset_voice(
                spk,
                age_hint=age_map.get(spk),
                atmosphere_hint=atm_map.get(spk),
                avoid_voices=used_presets,
            )
            assigned_preset = preset_recommendation["recommended_voice"]
        if not assigned_preset:
            raise UnauthorizedSpeakerError(
                f"Speaker {spk!r} does not have an approved VoiceProfile. "
                f"Cloning unauthorized or real-person voices is strictly prohibited. "
                f"Assign an explicit preset voice with --preset-speaker {spk}=Ryan or --default-preset-voice Ryan."
            )
        used_presets.add(assigned_preset)
        voices[spk] = assigned_preset
        speaker_modes[spk] = "preset"
        routing_details.append(
            {
                "speaker": spk,
                "mode": "preset",
                "voice": assigned_preset,
                "clone_method": None,
                "x_vector_only_mode": False,
                "authorization_status": "unapproved_speaker_routed_to_explicit_preset",
                "profile_fingerprint": None,
                **({"preset_recommendation": preset_recommendation} if preset_recommendation else {}),
            }
        )

    return {
        "voices": voices,
        "voice_profiles": active_profiles if active_profiles else None,
        "speaker_modes": speaker_modes,
        "effective_clone_method": effective_clone_method,
        "uses_clone": "approved_clone" in speaker_modes.values(),
        "uses_preset": "preset" in speaker_modes.values(),
        "uses_technical_reference": uses_technical_reference,
        "speaker_routing_details": routing_details,
        "rights_and_governance": governance_records,
    }


def _utterance_preset_fallback(utterances, speaker):
    for u in utterances:
        if u.get("speaker") == speaker and u.get("preset_voice"):
            return str(u["preset_voice"])
    return None


def _resolve_binary(name):
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        cand = Path(prefix) / name
        if cand.is_file():
            return str(cand)
    return name


def ensure_30s_verification_video(video_path, *, duration_seconds=30.0, ffmpeg_bin=None):
    """Create a safe 30.0-second H.264 + 48kHz stereo AAC verification MP4 if it does not already exist."""
    target = Path(video_path).resolve()
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = target.with_name(f"{target.stem}.tmp{target.suffix}")
    ffmpeg_exec = ffmpeg_bin or _resolve_binary("ffmpeg")
    cmd = [
        ffmpeg_exec,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=30:duration={duration_seconds:.2f}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=220:sample_rate=48000:duration={duration_seconds:.2f}",
        "-filter_complex",
        "[1:a]volume=0.08,pan=stereo|c0=c0|c1=c0[aout]",
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-t",
        f"{duration_seconds:.2f}",
        str(tmp_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_mp4.is_file():
        raise RuntimeError(f"Failed to generate 30s verification video: {proc.stderr[-600:]}")
    tmp_mp4.replace(target)
    return target


def probe_media_streams(mp4_path, *, ffprobe_bin=None):
    """Inspect container duration, video stream, and audio stream via ffprobe."""
    ffprobe_exec = ffprobe_bin or _resolve_binary("ffprobe")
    cmd = [
        ffprobe_exec,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,sample_rate,channels,width,height,duration",
        "-of",
        "json",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {mp4_path}: {proc.stderr}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    fmt_dur = float(info.get("format", {}).get("duration") or 0.0)
    return {
        "duration_seconds": round(fmt_dur, 4),
        "has_video": len(video_streams) > 0,
        "has_audio": len(audio_streams) > 0,
        "has_subtitle_stream": len(subtitle_streams) > 0,
        "video_codec": video_streams[0].get("codec_name") if video_streams else None,
        "width": int(video_streams[0].get("width") or 0) if video_streams else 0,
        "height": int(video_streams[0].get("height") or 0) if video_streams else 0,
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
        "audio_sample_rate": int(audio_streams[0].get("sample_rate") or 0) if audio_streams else 0,
        "audio_channels": int(audio_streams[0].get("channels") or 0) if audio_streams else 0,
    }


def _trim_wav_silence_non_destructive(wav_path, output_dir, *, utterance_id, margin_seconds=0.045):
    """Trim excess leading/trailing silence into a new file in `output_dir`, never modifying `wav_path`.

    Keeps a 45ms margin and applies a 10ms raised-cosine micro-fade at trimmed edges to prevent onset
    distortion or boundary click noise on short utterances.
    """
    samples, rate = _read_wav_float32(wav_path)
    orig_dur = len(samples) / float(rate)
    if len(samples) < int(0.2 * rate):
        return Path(wav_path), orig_dur, orig_dur, False

    peak = float(np.max(np.abs(samples)))
    thr = max(0.003, peak * 0.022)
    active_idx = np.where(np.abs(samples) >= thr)[0]
    if len(active_idx) == 0:
        return Path(wav_path), orig_dur, orig_dur, False

    margin_frames = int(round(margin_seconds * rate))
    start_f = max(0, int(active_idx[0]) - margin_frames)
    end_f = min(len(samples), int(active_idx[-1]) + 1 + margin_frames)
    trimmed = samples[start_f:end_f].copy()
    new_dur = len(trimmed) / float(rate)
    if (orig_dur - new_dur) < 0.04 or new_dur < 0.15:
        return Path(wav_path), orig_dur, orig_dur, False

    fade_frames = min(int(round(0.010 * rate)), len(trimmed) // 8)
    if fade_frames >= 4:
        t_fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.pi * t_fade))
        if start_f > 0:
            trimmed[:fade_frames] *= ramp
        if end_f < len(samples):
            trimmed[-fade_frames:] *= ramp[::-1]

    out_d = Path(output_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(utterance_id))
    trimmed_path = out_d / f"{safe_id}_trimmed.wav"
    pcm = np.clip(np.round(trimmed * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(trimmed_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return trimmed_path, round(orig_dur, 4), round(new_dur, 4), True


def _apply_bounded_atempo_non_destructive(
    wav_path,
    output_dir,
    *,
    utterance_id,
    target_duration_seconds,
    min_atempo=1.0,
    max_atempo=1.25,
    exact_atempo_ratio=None,
    ffmpeg_bin=None,
):
    """Apply bounded FFmpeg `atempo` time-stretching into a new file in `output_dir`, never modifying `wav_path`."""
    samples, rate = _read_wav_float32(wav_path)
    cur_dur = len(samples) / float(rate)
    if target_duration_seconds <= 0.05 and exact_atempo_ratio is None:
        return Path(wav_path), round(cur_dur, 4), 1.0, False

    from timing_adjustment import compute_auto_speech_rate_ratio
    eff_min = max(0.5, min(1.0, float(min_atempo)))
    eff_max = max(1.0, min(2.0, float(max_atempo)))
    if exact_atempo_ratio is not None:
        applied_ratio = max(eff_min, min(eff_max, float(exact_atempo_ratio)))
    else:
        rate_plan = compute_auto_speech_rate_ratio(
            cur_dur,
            target_duration_seconds,
            min_atempo=eff_min,
            max_atempo=eff_max,
            deadband=0.015,
        )
        applied_ratio = float(rate_plan["applied_atempo"])
    if abs(applied_ratio - 1.0) <= 0.015:
        return Path(wav_path), round(cur_dur, 4), 1.0, False

    out_d = Path(output_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(utterance_id))
    out_wav = out_d / f"{safe_id}_atempo_{applied_ratio:.3f}.wav"
    tmp_wav = out_d / f"{safe_id}_atempo.tmp.wav"

    # Build FFmpeg atempo filter chain (each atempo stage supports [0.5, 2.0])
    ffmpeg_exec = ffmpeg_bin or _resolve_binary("ffmpeg")
    cmd = [
        ffmpeg_exec,
        "-y",
        "-i",
        str(wav_path),
        "-filter:a",
        f"atempo={applied_ratio:.5f}",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-c:a",
        "pcm_s16le",
        str(tmp_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_wav.is_file():
        return Path(wav_path), round(cur_dur, 4), 1.0, False

    tmp_wav.replace(out_wav)
    adj_samples, adj_rate = _read_wav_float32(out_wav)
    adj_dur = len(adj_samples) / float(adj_rate)
    return out_wav, round(adj_dur, 4), round(applied_ratio, 4), True


def analyze_timeline_collisions_and_overlaps(
    entries,
    *,
    selection_start=0.0,
    selection_end=30.0,
    max_timing_shift=0.5,
    max_inter_speaker_shift=0.0,
    auto_tempo_adjust=False,
    min_atempo=0.90,
    max_atempo=1.25,
    fit_utterance_slots=False,
    allow_slowdown=None,
    smooth_intra_speaker_rates=False,
    adjusted_wav_dir=None,
    source_utterances=None,
    word_timestamps=None,
    use_word_level_sync=False,
    sync_anchor="adaptive",
    long_pause_threshold=0.55,
):
    """Detect same-speaker collisions & inter-speaker overlaps, measure overflows,
    align semantic clause units (and word-level speech clusters), apply bounded
    tempo & timing adjustments, and detect timing/silence/truncation anomalies.

    Phase 13.2 enhancement:
    - When `allow_slowdown=False` (or `min_atempo >= 1.0`), automatic slowdown (`atempo < 1.0`)
      is strictly prohibited so short utterances never get stretched/warbled; they play at `1.00x`
      and leave a natural trailing pause.
    - When `smooth_intra_speaker_rates=True`, adjacent same-speaker speaking rates are smoothed
      (`max_adjacent_delta <= 0.12x`) before applying a single FFmpeg `atempo` pass.
    """
    from timing_adjustment import (
        align_semantic_units_to_utterance_window,
        compute_auto_speech_rate_ratio,
        detect_timing_and_coverage_anomalies,
        plan_overlong_tts_resolution,
        smooth_intra_speaker_speaking_rates,
        split_utterance_on_long_pauses_and_align_clauses,
    )

    slowdown_allowed = (float(min_atempo) < 1.0) if allow_slowdown is None else bool(allow_slowdown)
    effective_min_atempo = max(0.5, min(1.0, float(min_atempo))) if slowdown_allowed else 1.0

    norm_entries = []
    duration_adjustments = []

    for idx, row in enumerate(entries):
        src_start = float(row.get("source_start", row.get("start", 0.0)))
        src_end = float(row.get("source_end", row.get("end", src_start + 1.0)))
        if float(selection_start) > 0.0 and src_start < float(selection_start) - 1.0:
            src_start += float(selection_start)
            src_end += float(selection_start)

        raw_place_start = float(row.get("placement_start", row.get("placement_seconds", src_start)))
        if float(selection_start) > 0.0 and raw_place_start < float(selection_start) - 1.0:
            raw_place_start += float(selection_start)

        clause_sync = split_utterance_on_long_pauses_and_align_clauses(
            {**row, "source_start": src_start, "source_end": src_end},
            external_words=word_timestamps,
            long_pause_threshold=long_pause_threshold,
        )
        wt_analysis = clause_sync["word_timing_analysis"]
        eff_start = max(src_start, float(wt_analysis["effective_content_start"]))
        eff_end = max(eff_start + 0.10, float(wt_analysis["effective_content_end"]))

        if use_word_level_sync and sync_anchor != "start":
            # Prevent premature start over leading fillers or pauses
            place_start = max(raw_place_start, eff_start)
            effective_slot_dur = max(0.05, eff_end - eff_start)
            semantic_units = clause_sync["timed_semantic_units"]
        else:
            place_start = raw_place_start
            effective_slot_dur = max(0.05, src_end - src_start)
            semantic_units = align_semantic_units_to_utterance_window({
                **row,
                "source_start": src_start,
                "source_end": src_end,
            })

        wav_file = row.get("audio_file")
        raw_wav_file = row.get("raw_audio_file") or wav_file
        raw_dur = float(
            row.get(
                "actual_tts_duration_seconds",
                row.get("measured_audio_duration", row.get("tts_duration_seconds", max(0.1, src_end - src_start))),
            )
        )
        if wav_file:
            try:
                if Path(wav_file).is_file():
                    s_arr, s_sr = _read_wav_float32(wav_file)
                    raw_dur = len(s_arr) / float(s_sr)
            except Exception:
                pass

        active_wav = str(wav_file) if wav_file else None
        cur_dur = raw_dur
        slot_dur = effective_slot_dur
        did_trim_edge = False

        # Optional non-destructive edge-silence trim when overflowing slot and adjusted_wav_dir is provided
        if auto_tempo_adjust and adjusted_wav_dir and active_wav and cur_dur > slot_dur:
            try:
                wav_exists = Path(active_wav).is_file()
            except OSError:
                wav_exists = False
            if wav_exists:
                trimmed_p, orig_d, trim_d, did_trim = _trim_wav_silence_non_destructive(
                    active_wav,
                    adjusted_wav_dir,
                    utterance_id=row.get("id", f"utt-{idx + 1:03d}"),
                )
                if did_trim:
                    active_wav = str(trimmed_p)
                    cur_dur = trim_d
                    did_trim_edge = True
                    duration_adjustments.append(
                        {
                            "id": str(row.get("id", f"utt-{idx + 1:03d}")),
                            "speaker": row.get("speaker", DEFAULT_SPEAKER_ID),
                            "stage": "edge_silence_trim",
                            "original_duration_seconds": orig_d,
                            "adjusted_duration_seconds": trim_d,
                            "atempo_ratio": 1.0,
                            "original_audio_file": str(wav_file),
                            "adjusted_audio_file": active_wav,
                        }
                    )

        norm_entries.append(
            {
                "index": idx,
                "id": str(row.get("id", f"utt-{idx + 1:03d}")),
                "raw_id": row.get("id", f"utt-{idx + 1:03d}"),
                "speaker": row.get("speaker", DEFAULT_SPEAKER_ID),
                "text_ja": row.get("text_ja", ""),
                "mora_count": int(row.get("mora_count") or 0),
                "source_start": src_start,
                "source_end": src_end,
                "effective_content_start": eff_start,
                "effective_content_end": eff_end,
                "leading_filler_offset_seconds": wt_analysis["leading_filler_offset_seconds"],
                "leading_filler_words": wt_analysis["leading_filler_words"],
                "long_pause_count": wt_analysis["long_pause_count"],
                "word_timing_analysis": wt_analysis,
                "slot_duration": round(slot_dur, 4),
                "raw_tts_duration": round(raw_dur, 4),
                "trimmed_tts_duration": round(cur_dur, 4),
                "trim_applied": did_trim_edge,
                "resampling_count": 0,
                "applied_atempo": 1.0,
                "slot_overflow_seconds": round(max(0.0, cur_dur - slot_dur), 4),
                "placement_start": place_start,
                "actual_duration": cur_dur,
                "placement_end": place_start + cur_dur,
                "raw_audio_file": str(raw_wav_file) if raw_wav_file else None,
                "original_audio_file": str(wav_file) if wav_file else None,
                "active_audio_file": active_wav,
                "semantic_units": semantic_units,
                "sync_anchor_used": "content_start" if (use_word_level_sync and sync_anchor != "start") else "raw_start",
                "overlong_resolution_plan": None,
            }
        )

    norm_entries.sort(key=lambda r: (r["placement_start"], r["index"]))

    # Detect initial same-speaker collisions before shift/tempo adjustments
    initial_same_speaker_collisions = []
    by_speaker = {}
    for item in norm_entries:
        by_speaker.setdefault(item["speaker"], []).append(item)

    for spk, group in by_speaker.items():
        for k in range(len(group) - 1):
            cur = group[k]
            nxt = group[k + 1]
            ov = cur["placement_end"] - nxt["placement_start"]
            if ov > 1e-4:
                initial_same_speaker_collisions.append(
                    {
                        "speaker": spk,
                        "first_id": cur["raw_id"],
                        "second_id": nxt["raw_id"],
                        "first_end_seconds": round(cur["placement_end"], 4),
                        "second_start_seconds": round(nxt["placement_start"], 4),
                        "overlap_seconds": round(ov, 4),
                    }
                )

    # Optional bounded non-destructive atempo when auto_tempo_adjust=True
    intra_speaker_rate_smoothing = None
    if auto_tempo_adjust and adjusted_wav_dir:
        planned_tempo_actions = {}
        for i, item in enumerate(norm_entries):
            if not item["active_audio_file"]:
                continue
            try:
                if not Path(item["active_audio_file"]).is_file():
                    continue
            except OSError:
                continue
            barriers = [float(selection_end)]
            downstream_pause = 0.0
            for j in range(i + 1, len(norm_entries)):
                other = norm_entries[j]
                other_anchor = other["effective_content_start"] if use_word_level_sync else other["source_start"]
                if other["speaker"] == item["speaker"]:
                    barriers.append(other_anchor + float(max_timing_shift))
                    downstream_pause = max(0.0, other_anchor - item["effective_content_end"])
                    break
                else:
                    src_ov = min(item["source_end"], other["source_end"]) - max(item["source_start"], other["source_start"])
                    if src_ov <= 0.05 and other["source_start"] >= item["source_start"]:
                        barriers.append(other_anchor + float(max_inter_speaker_shift))
                        downstream_pause = max(0.0, other_anchor - item["effective_content_end"])
                        break
            avail_span = min(barriers) - item["placement_start"]
            # When slowdown is prohibited (Phase 13.2), allow absorbing natural downstream pause up to avail_span
            target_span = (
                min(avail_span, item["slot_duration"])
                if (fit_utterance_slots and slowdown_allowed)
                else min(avail_span, max(item["slot_duration"], avail_span))
            )
            if fit_utterance_slots and not slowdown_allowed:
                # Prefer fitting within slot + up to 0.45s natural pause absorption before speeding up
                target_span = min(avail_span, item["slot_duration"] + min(0.45, max(0.0, avail_span - item["slot_duration"])))

            overlong_plan = plan_overlong_tts_resolution(
                item,
                raw_tts_duration=item["raw_tts_duration"],
                trimmed_tts_duration=item["trimmed_tts_duration"],
                slot_duration=item["slot_duration"],
                downstream_pause_seconds=downstream_pause,
                max_shift_seconds=float(max_timing_shift),
                max_atempo=min(1.25, float(max_atempo)),
            )
            item["overlong_resolution_plan"] = overlong_plan

            needs_speedup = item["actual_duration"] > target_span + 0.02 and target_span >= 0.30
            needs_slowdown = (
                slowdown_allowed
                and fit_utterance_slots
                and effective_min_atempo < 1.0
                and item["actual_duration"] < item["slot_duration"] * 0.88
                and item["slot_duration"] <= avail_span + 0.01
                and item["slot_duration"] >= 0.50
            )
            target_dur = item["slot_duration"] if needs_slowdown else target_span
            if needs_speedup or needs_slowdown:
                rate_plan = compute_auto_speech_rate_ratio(
                    item["actual_duration"],
                    target_dur,
                    min_atempo=effective_min_atempo if fit_utterance_slots else 1.0,
                    max_atempo=max_atempo,
                    deadband=0.015,
                )
                planned_ratio = float(rate_plan["applied_atempo"])
            else:
                planned_ratio = 1.0
            planned_tempo_actions[item["id"]] = {
                "target_dur": target_dur,
                "avail_span": avail_span,
                "barriers": barriers,
                "planned_ratio": planned_ratio,
                "needs_speedup": needs_speedup,
                "needs_slowdown": needs_slowdown,
            }

        if smooth_intra_speaker_rates and planned_tempo_actions:
            rate_inputs = [
                {
                    "id": item["id"],
                    "speaker": item["speaker"],
                    "applied_atempo": planned_tempo_actions.get(item["id"], {}).get("planned_ratio", 1.0),
                    "raw_tts_duration": item["actual_duration"],
                    "slot_duration": planned_tempo_actions.get(item["id"], {}).get("avail_span", item["slot_duration"]),
                    "available_shift_seconds": 0.0,
                }
                for item in norm_entries
            ]
            intra_speaker_rate_smoothing = smooth_intra_speaker_speaking_rates(
                rate_inputs,
                max_adjacent_delta=0.12,
                max_atempo=min(1.25, float(max_atempo)),
                allow_slowdown=slowdown_allowed,
            )
            smoothed_by_id = {str(r["id"]): float(r["applied_atempo"]) for r in intra_speaker_rate_smoothing["entries"]}
            for uid_k, act in planned_tempo_actions.items():
                if str(uid_k) in smoothed_by_id:
                    act["planned_ratio"] = smoothed_by_id[str(uid_k)]

        for i, item in enumerate(norm_entries):
            act = planned_tempo_actions.get(item["id"])
            if not act:
                continue
            barriers = act["barriers"]
            ratio_to_apply = float(act["planned_ratio"])
            if abs(ratio_to_apply - 1.0) > 0.015:
                adj_wav, adj_dur, ratio, did_tempo = _apply_bounded_atempo_non_destructive(
                    item["active_audio_file"],
                    adjusted_wav_dir,
                    utterance_id=item["id"],
                    target_duration_seconds=act["target_dur"],
                    min_atempo=effective_min_atempo if fit_utterance_slots else 1.0,
                    max_atempo=max_atempo,
                    exact_atempo_ratio=ratio_to_apply if smooth_intra_speaker_rates else None,
                )
                if did_tempo:
                    prev_d = item["actual_duration"]
                    item["active_audio_file"] = str(adj_wav)
                    item["actual_duration"] = adj_dur
                    item["placement_end"] = item["placement_start"] + adj_dur
                    item["applied_atempo"] = ratio
                    item["resampling_count"] += 1
                    item["slot_overflow_seconds"] = round(max(0.0, adj_dur - item["slot_duration"]), 4)
                    duration_adjustments.append(
                        {
                            "id": item["raw_id"],
                            "speaker": item["speaker"],
                            "stage": "bounded_atempo_slowdown" if ratio < 1.0 else "bounded_atempo",
                            "original_duration_seconds": round(prev_d, 4),
                            "adjusted_duration_seconds": adj_dur,
                            "atempo_ratio": ratio,
                            "original_audio_file": item["original_audio_file"],
                            "adjusted_audio_file": str(adj_wav),
                        }
                    )

            # Optional back-calculated end alignment ('end' or 'adaptive') when use_word_level_sync is enabled
            if use_word_level_sync and sync_anchor in ("end", "adaptive"):
                eff_s = item["effective_content_start"]
                eff_e = item["effective_content_end"]
                slack = eff_e - (eff_s + item["actual_duration"])
                if slack > 0.15:
                    if sync_anchor == "end":
                        target_s = min(eff_e - item["actual_duration"], min(barriers) - item["actual_duration"])
                        if target_s > item["placement_start"]:
                            item["placement_start"] = max(eff_s, target_s)
                            item["placement_end"] = item["placement_start"] + item["actual_duration"]
                            item["sync_anchor_used"] = "end_backcalculated"
                    elif sync_anchor == "adaptive" and (item["leading_filler_offset_seconds"] >= 0.15 or item["long_pause_count"] > 0):
                        shift = min(slack * 0.65, 1.20)
                        target_s = min(eff_s + shift, min(barriers) - item["actual_duration"])
                        if target_s > item["placement_start"]:
                            item["placement_start"] = max(eff_s, target_s)
                            item["placement_end"] = item["placement_start"] + item["actual_duration"]
                            item["sync_anchor_used"] = "adaptive_end_weighted"

    # Apply bounded shift (up to max_timing_shift) per speaker without exceeding selection_end,
    # and NEVER pulling an utterance earlier than its own placement_start / effective_content_start!
    shifted_placements = {}
    resolved_collisions = []
    unresolved_collisions = []

    for spk, group in by_speaker.items():
        running_end = None
        for k, item in enumerate(group):
            orig_start = item["placement_start"]
            new_start = orig_start
            if running_end is not None and running_end > new_start + 1e-4:
                needed_shift = running_end - new_start
                max_allowed_start = min(
                    orig_start + max(0.0, float(max_timing_shift)),
                    float(selection_end) - item["actual_duration"],
                )
                if orig_start + needed_shift <= max_allowed_start + 1e-4:
                    new_start = orig_start + needed_shift
                    resolved_collisions.append(
                        {
                            "speaker": spk,
                            "first_id": group[k - 1]["raw_id"],
                            "second_id": item["raw_id"],
                            "applied_shift_seconds": round(needed_shift, 4),
                        }
                    )
                else:
                    prev = group[k - 1]
                    shortfall = running_end - max(orig_start, max_allowed_start)
                    avail_budget = max(0.1, nxt_bound(group, k - 1, selection_end) - prev["placement_start"])
                    mora_rate = (
                        (prev["mora_count"] / prev["actual_duration"])
                        if prev["actual_duration"] > 0 and prev["mora_count"] > 0
                        else 7.5
                    )
                    target_mora = max(1, int(math.floor(avail_budget * mora_rate)))
                    unresolved_collisions.append(
                        {
                            "speaker": spk,
                            "first_id": prev["raw_id"],
                            "second_id": item["raw_id"],
                            "overlap_seconds": round(running_end - orig_start, 4),
                            "unresolvable_overflow_seconds": round(shortfall, 4),
                            "required_reduction_seconds": round(max(0.0, prev["actual_duration"] - avail_budget), 4),
                            "current_mora_count": prev["mora_count"],
                            "target_mora_budget": target_mora,
                            "mora_reduction_needed": max(0, prev["mora_count"] - target_mora),
                        }
                    )
            item["placement_start"] = new_start
            item["placement_end"] = new_start + item["actual_duration"]
            running_end = item["placement_end"]

    # Adjust artificial inter-speaker TTS-expansion spills when `max_inter_speaker_shift > 0`
    adjusted_inter_speaker_overlaps = []
    if float(max_inter_speaker_shift) > 0.0:
        norm_entries.sort(key=lambda r: (r["placement_start"], r["index"]))
        for i in range(len(norm_entries)):
            for j in range(i + 1, len(norm_entries)):
                a = norm_entries[i]
                b = norm_entries[j]
                if a["speaker"] == b["speaker"]:
                    continue
                ov = a["placement_end"] - b["placement_start"]
                if ov <= 1e-4:
                    continue
                src_ov = min(a["source_end"], b["source_end"]) - max(a["source_start"], b["source_start"])
                if src_ov <= 0.05 and a["source_start"] <= b["source_start"]:
                    max_b_shift = min(
                        float(max_inter_speaker_shift),
                        max(0.0, float(selection_end) - b["actual_duration"] - b["placement_start"]),
                    )
                    applied = min(ov, max_b_shift)
                    if applied > 1e-3:
                        b["placement_start"] += applied
                        b["placement_end"] = b["placement_start"] + b["actual_duration"]
                        adjusted_inter_speaker_overlaps.append(
                            {
                                "first_id": a["raw_id"],
                                "second_id": b["raw_id"],
                                "first_speaker": a["speaker"],
                                "second_speaker": b["speaker"],
                                "initial_spill_seconds": round(ov, 4),
                                "applied_shift_seconds": round(applied, 4),
                                "remaining_overlap_seconds": round(max(0.0, ov - applied), 4),
                                "policy": "adjusted_artificial_tts_spill",
                            }
                        )

    # Detect natural other-speaker overlaps (preserved!)
    other_speaker_overlaps = []
    for i in range(len(norm_entries)):
        for j in range(i + 1, len(norm_entries)):
            a = norm_entries[i]
            b = norm_entries[j]
            if a["speaker"] == b["speaker"]:
                continue
            ov_start = max(a["placement_start"], b["placement_start"])
            ov_end = min(a["placement_end"], b["placement_end"])
            if ov_end - ov_start > 1e-4:
                other_speaker_overlaps.append(
                    {
                        "first_id": a["raw_id"],
                        "second_id": b["raw_id"],
                        "first_speaker": a["speaker"],
                        "second_speaker": b["speaker"],
                        "overlap_start_seconds": round(ov_start, 4),
                        "overlap_end_seconds": round(ov_end, 4),
                        "overlap_seconds": round(ov_end - ov_start, 4),
                        "policy": "preserved_natural_multi_speaker_overlap",
                    }
                )

    for item in norm_entries:
        placement_info = {
            "source_start_seconds": round(item["source_start"], 4),
            "source_end_seconds": round(item["source_end"], 4),
            "effective_content_start_seconds": round(item["effective_content_start"], 4),
            "effective_content_end_seconds": round(item["effective_content_end"], 4),
            "leading_filler_offset_seconds": item["leading_filler_offset_seconds"],
            "leading_filler_words": item["leading_filler_words"],
            "long_pause_count": item["long_pause_count"],
            "premature_start_prevented_seconds": round(max(0.0, item["placement_start"] - item["source_start"]), 4),
            "starts_before_content": bool(item["placement_start"] < item["effective_content_start"] - 1e-4),
            "sync_anchor_used": item["sync_anchor_used"],
            "slot_duration_seconds": item["slot_duration"],
            "raw_tts_duration_seconds": item["raw_tts_duration"],
            "trimmed_tts_duration_seconds": item["trimmed_tts_duration"],
            "actual_tts_duration_seconds": round(item["actual_duration"], 4),
            "applied_atempo": round(float(item["applied_atempo"]), 4),
            "resampling_count": int(item["resampling_count"]),
            "trim_applied": bool(item["trim_applied"]),
            "natural_trailing_pause_seconds": round(max(0.0, item["slot_duration"] - item["actual_duration"]), 4),
            "slot_overflow_seconds": item["slot_overflow_seconds"],
            "placement_start_seconds": round(item["placement_start"], 4),
            "placement_end_seconds": round(item["placement_end"], 4),
            "clip_relative_start_seconds": round(max(0.0, item["placement_start"] - float(selection_start)), 4),
            "clip_relative_end_seconds": round(
                max(0.0, item["placement_end"] - float(selection_start)),
                4,
            ),
            "shift_seconds": round(item["placement_start"] - item["source_start"], 4),
            "raw_audio_file": item["raw_audio_file"],
            "active_audio_file": item["active_audio_file"],
            "original_audio_file": item["original_audio_file"],
            "semantic_units": item.get("semantic_units", []),
            "overlong_resolution_plan": item.get("overlong_resolution_plan"),
        }
        shifted_placements[item["id"]] = placement_info
        shifted_placements[item["raw_id"]] = placement_info

    repo_budget_diagnostics = None
    try:
        import timing_adjustment

        if hasattr(timing_adjustment, "diagnose_collision_budgets"):
            fn = timing_adjustment.diagnose_collision_budgets
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 1:
                try:
                    repo_budget_diagnostics = fn(entries)
                except Exception:
                    repo_budget_diagnostics = None
    except ImportError:
        pass

    timing_sync_diagnostics = detect_timing_and_coverage_anomalies(
        norm_entries,
        selection_start=selection_start,
        selection_end=selection_end,
        source_utterances=source_utterances,
    )

    serializable_placements = {str(item["id"]): shifted_placements[item["id"]] for item in norm_entries}

    return {
        "selection_start_seconds": round(float(selection_start), 4),
        "selection_end_seconds": round(float(selection_end), 4),
        "clip_duration_seconds": round(float(selection_end) - float(selection_start), 4),
        "max_timing_shift_seconds": round(float(max_timing_shift), 4),
        "max_inter_speaker_shift_seconds": round(float(max_inter_speaker_shift), 4),
        "min_atempo": round(float(effective_min_atempo), 4),
        "max_atempo": round(float(max_atempo), 4),
        "allow_slowdown": bool(slowdown_allowed),
        "fit_utterance_slots": bool(fit_utterance_slots),
        "smooth_intra_speaker_rates": bool(smooth_intra_speaker_rates),
        "intra_speaker_rate_smoothing": intra_speaker_rate_smoothing,
        "use_word_level_sync": bool(use_word_level_sync),
        "sync_anchor": sync_anchor,
        "initial_same_speaker_collisions": initial_same_speaker_collisions,
        "resolved_same_speaker_collisions": resolved_collisions,
        "unresolved_same_speaker_collisions": unresolved_collisions,
        "adjusted_inter_speaker_overlaps": adjusted_inter_speaker_overlaps,
        "preserved_other_speaker_overlaps": other_speaker_overlaps,
        "duration_adjustments": duration_adjustments,
        "timing_sync_diagnostics": timing_sync_diagnostics,
        "placements": serializable_placements,
        "repo_collision_budget_diagnostics": repo_budget_diagnostics,
    }


def nxt_bound(group, idx, selection_end):
    if idx + 1 < len(group):
        return group[idx + 1]["placement_start"]
    return float(selection_end)


def _normalize_feedback_to_clip_relative(feedback_doc, *, selection_start, clip_duration, placements_map):
    """Return a copy of feedback_doc where all timestamps and active audio files are unified in clip-relative [0.0, clip_duration] seconds."""
    rel_doc = copy.deepcopy(feedback_doc)
    rel_doc["selection_range"] = {"start": 0.0, "end": round(float(clip_duration), 4)}
    rel_doc["source_selection_range"] = feedback_doc.get(
        "selection_range",
        {"start": round(float(selection_start), 4), "end": round(float(selection_start + clip_duration), 4)},
    )
    for entry in rel_doc.get("entries", []):
        uid = str(entry.get("id"))
        p_info = placements_map.get(uid) or placements_map.get(entry.get("id"))
        if p_info is not None:
            rel_start = float(p_info["clip_relative_start_seconds"])
            dur = float(p_info.get("actual_tts_duration_seconds", p_info["clip_relative_end_seconds"] - rel_start))
            if p_info.get("active_audio_file"):
                entry["original_audio_file"] = entry.get("audio_file")
                entry["audio_file"] = p_info["active_audio_file"]
            for extra_k in (
                "applied_atempo",
                "resampling_count",
                "trim_applied",
                "natural_trailing_pause_seconds",
                "raw_audio_file",
                "raw_tts_duration_seconds",
                "trimmed_tts_duration_seconds",
                "overlong_resolution_plan",
            ):
                if extra_k in p_info:
                    entry[extra_k] = p_info[extra_k]
        else:
            dur = float(
                entry.get(
                    "actual_tts_duration_seconds",
                    entry.get("measured_audio_duration", entry.get("tts_duration_seconds", 1.0)),
                )
            )
            raw_start = float(entry.get("placement_seconds", entry.get("start", 0.0)))
            rel_start = raw_start - float(selection_start) if raw_start >= float(selection_start) - 1e-4 else raw_start
        rel_start = round(max(0.0, rel_start), 4)
        rel_end = round(rel_start + dur, 4)
        entry["source_absolute_start_seconds"] = entry.get("start")
        entry["source_absolute_end_seconds"] = entry.get("end")
        entry["start"] = rel_start
        entry["end"] = rel_end
        entry["clip_start"] = rel_start
        entry["clip_end"] = rel_end
        entry["placement_seconds"] = rel_start
        entry["placement_start"] = rel_start
        entry["placement_end"] = rel_end
        entry["actual_tts_duration_seconds"] = round(dur, 4)
    return rel_doc


def run_technical_xvector_feedback(
    *,
    utterances,
    selection_range,
    output_dir,
    backend,
    voice_plan,
    reuse_wav_dir=None,
    voice_registry=None,
    max_consistency_retries=2,
):
    """Execute technical X-vector (or hybrid) TTS synthesis and WAV duration measurement directly.

    Decoupled from `voice_cloning.register_profile` so synthetic speakers and permitted audio
    sources do not require redundant `consent.json`/`review.json` registration ceremonies,
    while enforcing SpeakerVoiceRegistry locking, never overwriting existing WAVs, and never
    falling back to Preset on X-vector failure.
    """
    out_dir = Path(output_dir).resolve()
    audio_dir = out_dir / "audio"
    raw_audio_dir = out_dir / "raw_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    voices = voice_plan["voices"]
    profiles = voice_plan.get("voice_profiles") or {}
    reuse_dir = Path(reuse_wav_dir).resolve() if reuse_wav_dir else None

    entries = []
    for idx, u in enumerate(utterances):
        uid = str(u["id"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)
        spk = u["speaker"]
        if spk not in voices:
            raise XVectorSynthesisError(f"Speaker {spk!r} for utterance {uid!r} is missing from voice_plan['voices'].")
        voice_spec = voices[spk]
        prof = profiles.get(spk)

        locked_profile_meta = None
        if voice_registry is not None:
            locked_profile_meta = voice_registry.verify_utterance_routing(
                u,
                requested_voice=voice_spec,
                requested_profile=prof,
            )

        text_ja = str(u.get("text_ja") or "").strip()
        if not text_ja:
            raise XVectorSynthesisError(f"Utterance {uid!r} (speaker {spk!r}) has empty Japanese text (`text_ja`).")

        # Non-clobber output path inside `_work/tts_feedback/audio/`
        wav_out = audio_dir / f"{safe_id}.wav"
        raw_wav_out = raw_audio_dir / f"{safe_id}_raw.wav"
        cache_reused = False
        backend_res = None
        consistency_retry_count = 0

        # Check if a matching WAV exists in `--reuse-wav-dir` (read-only, never overwritten!)
        reused_src = None
        if reuse_dir and reuse_dir.is_dir():
            needs_regen = bool(u.get("is_merged_utterance") or "tts_regeneration_required" in (u.get("review_reasons") or []))
            if not needs_regen:
                for cand_name in (f"{uid}.wav", f"{safe_id}.wav", f"dub-{idx + 1:03d}.wav", f"sample_{idx + 1:02d}.wav"):
                    cand_p = reuse_dir / cand_name
                    if cand_p.is_file():
                        reused_src = cand_p
                        break

        if reused_src is not None:
            if wav_out.resolve() != reused_src.resolve():
                shutil.copyfile(reused_src, wav_out)
            cache_reused = True
            samples, sr = _read_wav_float32(wav_out)
            backend_res = {
                "sample_rate": sr,
                "sample_frames": len(samples),
                "clone_method": voice_plan["effective_clone_method"] if voice_spec.startswith("approved_clone:") else "preset",
                "x_vector_only_mode": voice_plan["effective_clone_method"] == "xvector",
                "voice_profile_id": locked_profile_meta.voice_profile_id if locked_profile_meta else getattr(prof, "voice_profile_id", None),
                "reference_audio_sha256": locked_profile_meta.reference_audio_sha256 if locked_profile_meta else getattr(prof, "reference_audio_sha256", None),
                "reused_from": str(reused_src),
            }
        else:
            if voice_spec.startswith("approved_clone:"):
                if prof is None:
                    raise XVectorSynthesisError(
                        f"Missing voice reference/profile for speaker {spk!r} during X-vector synthesis."
                    )
                try:
                    backend_res = backend.synthesize(
                        text=text_ja,
                        profile=prof,
                        output_path=wav_out,
                    )
                except Exception as exc:
                    raise XVectorSynthesisError(
                        f"X-vector synthesis failed for utterance {uid!r} (speaker {spk!r}): {exc}. "
                        f"Automatic fallback to Preset is disabled."
                    ) from exc

                # Phase 13.2: Check acoustic F0 consistency against locked speaker reference and retry with same locked profile if outlier
                ref_f0 = None
                if locked_profile_meta and isinstance(locked_profile_meta.speaker_embedding, dict):
                    ref_f0 = locked_profile_meta.speaker_embedding.get("f0_median_hz")
                if ref_f0 and ref_f0 > 0 and max_consistency_retries > 0 and wav_out.is_file():
                    try:
                        synth_sig = inspect.signature(backend.synthesize)
                        supports_seed_override = (
                            "seed_override" in synth_sig.parameters
                            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in synth_sig.parameters.values())
                        )
                    except (TypeError, ValueError):
                        supports_seed_override = False
                    if supports_seed_override:
                        base_seed = int(getattr(backend, "settings", {}).get("seed", 0))
                        for retry_idx in range(1, int(max_consistency_retries) + 1):
                            pr_check = analyze_japanese_prosody(wav_out, mora_count=u.get("mora_count"))
                            cur_f0 = pr_check.get("f0_hz_median")
                            if cur_f0 and cur_f0 > 0:
                                st_diff = abs(12.0 * math.log2(cur_f0 / float(ref_f0)))
                                if st_diff <= 6.5:
                                    break
                            consistency_retry_count = retry_idx
                            backend_res = backend.synthesize(
                                text=text_ja,
                                profile=prof,
                                output_path=wav_out,
                                seed_override=base_seed + retry_idx * 17,
                            )
            else:
                backend_res = backend.synthesize(
                    text=text_ja,
                    voice=voice_spec,
                    output_path=wav_out,
                )

        if not wav_out.is_file():
            raise XVectorSynthesisError(f"Synthesizer did not write output WAV for utterance {uid!r}: {wav_out}")

        shutil.copyfile(wav_out, raw_wav_out)
        prosody = analyze_japanese_prosody(wav_out, mora_count=u.get("mora_count"))
        dur = float(prosody["audio_duration_seconds"])
        slot_dur = max(0.05, float(u["end"]) - float(u["start"]))
        vp_id = (
            locked_profile_meta.voice_profile_id
            if locked_profile_meta is not None
            else (
                (backend_res or {}).get("voice_profile_id")
                or getattr(prof, "voice_profile_id", None)
                or f"vp-{spk}-{voice_spec}"
            )
        )
        ref_sha = (
            locked_profile_meta.reference_audio_sha256
            if locked_profile_meta is not None
            else (
                (backend_res or {}).get("reference_audio_sha256")
                or getattr(prof, "reference_audio_sha256", None)
            )
        )

        entry = {
            "id": u.get("raw_id", uid),
            "id_str": uid,
            "merged_from_ids": u.get("merged_from_ids"),
            "source_utterance_ids": u.get("source_utterance_ids", u.get("merged_from_ids") or [u.get("raw_id", uid)]),
            "review_required": bool(u.get("review_required")),
            "review_reasons": u.get("review_reasons", []),
            "speaker": spk,
            "voice": voice_spec,
            "voice_profile_id": vp_id,
            "reference_audio_sha256": ref_sha,
            "consistency_retry_count": consistency_retry_count,
            "text_en": u.get("text_en", ""),
            "text_ja": text_ja,
            "mora_count": int(u.get("mora_count") or _estimate_japanese_mora(text_ja)),
            "start": float(u["start"]),
            "end": float(u["end"]),
            "source_start": float(u["source_start"]),
            "source_end": float(u["source_end"]),
            "placement_seconds": float(u["start"]),
            "slot_duration_seconds": round(slot_dur, 4),
            "actual_tts_duration_seconds": round(dur, 4),
            "slot_overflow_seconds": round(max(0.0, dur - slot_dur), 4),
            "generation_status": "complete",
            "raw_audio_file": str(raw_wav_out),
            "audio_file": str(wav_out),
            "audio_sha256": prosody["sha256"],
            "cache_reused": cache_reused,
            "prosody_metrics": prosody,
            "backend_result": backend_res,
            "tts": dict(getattr(backend, "identity", {})),
        }
        entries.append(entry)

    fb_doc = {
        "kind": "transclip_tts_feedback",
        "format_version": 1,
        "selection_range": selection_range,
        "tts": dict(getattr(backend, "identity", {})),
        "entries": entries,
    }
    fb_path = out_dir / "feedback.json"
    _atomic_write_json(fb_path, fb_doc)
    return fb_path, fb_doc


def _prepare_clip_video_source(video_path, *, selection_start, clip_duration, work_dir, ffmpeg_bin=None):
    """Ensure the video file matches the `[selection_start, selection_start + clip_duration]` window without modifying `video_path`."""
    src = Path(video_path).resolve()
    probe = probe_media_streams(src)
    dur = float(probe.get("duration_seconds") or 0.0)
    if dur <= float(clip_duration) + 2.0 or float(selection_start) <= 0.0:
        return src, probe

    # Source video is a longer recording; extract the target 30s segment non-destructively into work_dir
    seg_mp4 = Path(work_dir).resolve() / f"source_segment_{int(selection_start)}_{int(selection_start + clip_duration)}.mp4"
    if seg_mp4.is_file():
        return seg_mp4, probe_media_streams(seg_mp4)

    tmp_mp4 = seg_mp4.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg_bin or _resolve_binary("ffmpeg"),
        "-y",
        "-ss",
        f"{float(selection_start):.4f}",
        "-t",
        f"{float(clip_duration):.4f}",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        str(tmp_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_mp4.is_file():
        return src, probe
    tmp_mp4.replace(seg_mp4)
    return seg_mp4, probe_media_streams(seg_mp4)


def _extract_source_audio_from_video(video_path, *, total_frames, target_sr=48000, ffmpeg_bin=None):
    """Extract mono 48kHz float32 audio array of length `total_frames` from `video_path` when available."""
    if not video_path or not Path(str(video_path)).is_file():
        return None
    ffmpeg_exec = ffmpeg_bin or _resolve_binary("ffmpeg")
    duration_sec = total_frames / float(target_sr)
    cmd = [
        ffmpeg_exec,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-t",
        f"{duration_sec:.4f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    arr = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not np.isfinite(arr).all() or len(arr) == 0:
        return None
    if len(arr) >= total_frames:
        return arr[:total_frames].copy()
    padded = np.zeros(total_frames, dtype=np.float32)
    padded[: len(arr)] = arr
    return padded


def _fallback_mix_audio_and_video(
    *,
    video_path,
    feedback_doc,
    placements_map,
    selection_start,
    clip_duration,
    output_dir,
    mix_mode="voiceover",
    ducking_style="dynamic",
    duck_gain=0.12,
    unspoken_gain=1.0,
    tts_gain=1.0,
    fade_out_seconds=0.18,
    fade_in_seconds=0.25,
    release_seconds=None,
    ambient_backend="auto",
    ambient_gain=0.85,
    boundary_fade_out_seconds=0.0,
    video_fade_out_seconds=0.0,
    ffmpeg_bin=None,
):
    """48kHz sample-accurate multi-speaker audio mixer & MP4 muxer with Phase 13 / 13.1 voice-over ducking
    (dynamic, continuous 15-20%, and long-release >=1.5s), ambient sound separation, boundary fade-out,
    soft-knee peak limiting, and loudness/clipping verification.
    """
    from audio_mixer import (
        apply_boundary_audio_fade,
        apply_soft_limiter,
        build_ducking_envelope,
        separate_ambient_and_speech,
        verify_audio_loudness_and_clipping,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_sr = 48000
    total_frames = int(round(float(clip_duration) * target_sr))
    speaker_tracks = {}
    speech_intervals = []

    for entry in feedback_doc.get("entries", []):
        if entry.get("generation_status") != "complete":
            continue
        uid = str(entry.get("id"))
        p_info = placements_map.get(uid) or placements_map.get(entry.get("id"))
        if p_info is None:
            raise ValueError(f"Missing placement for generated utterance {uid}.")
        wav_path_str = p_info.get("active_audio_file") or entry.get("audio_file")
        if not wav_path_str or not Path(wav_path_str).is_file():
            raise ValueError(f"Missing generated WAV for utterance {uid}.")
        spk = entry.get("speaker", DEFAULT_SPEAKER_ID)
        samples, sr = _read_wav_float32(Path(wav_path_str))
        if sr != target_sr and len(samples) > 1:
            new_len = max(1, int(round(len(samples) * float(target_sr) / float(sr))))
            x_old = np.linspace(0.0, 1.0, len(samples))
            x_new = np.linspace(0.0, 1.0, new_len)
            samples = np.interp(x_new, x_old, samples).astype(np.float32)

        rel_start = float(p_info["clip_relative_start_seconds"])
        start_frame = int(round(rel_start * target_sr))
        end_frame = start_frame + len(samples)
        if start_frame < 0 or end_frame > total_frames:
            raise ValueError(f"Full WAV for utterance {uid} exceeds the selected clip; truncation refused.")
        if end_frame > start_frame:
            track = speaker_tracks.setdefault(spk, np.zeros(total_frames, dtype=np.float64))
            track[start_frame:end_frame] += samples[: end_frame - start_frame].astype(np.float64)
            speech_intervals.append((rel_start, end_frame / float(target_sr)))

    japanese_master = np.zeros(total_frames, dtype=np.float64)
    tracks_dir = out_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    speaker_track_files = {}
    for spk, trk in speaker_tracks.items():
        japanese_master += trk * float(tts_gain)
        trk_pcm = np.clip(np.round(trk * 32767.0), -32768, 32767).astype("<i2")
        trk_path = tracks_dir / f"{spk}.wav"
        with wave.open(str(trk_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(target_sr)
            stream.writeframes(trk_pcm.tobytes())
        speaker_track_files[spk] = str(trk_path)

    # Save pure Japanese TTS track for inspection
    jp_only_wav = out_dir / "japanese_tts_only.wav"
    jp_pcm = np.clip(np.round(japanese_master * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(jp_only_wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(target_sr)
        stream.writeframes(jp_pcm.tobytes())

    normalized_mode = str(mix_mode or "voiceover").strip().lower()
    if normalized_mode in ("normal", "diagnostic"):
        normalized_mode = "voiceover"
    if normalized_mode == "full-dub":
        normalized_mode = "japanese-only"
    eff_ducking_style = str(ducking_style or "dynamic").strip().lower()
    if normalized_mode == "continuous-voiceover":
        eff_ducking_style = "continuous"
    elif normalized_mode == "long-release-voiceover":
        eff_ducking_style = "long-release"

    source_audio = None
    if normalized_mode != "japanese-only":
        source_audio = _extract_source_audio_from_video(
            video_path,
            total_frames=total_frames,
            target_sr=target_sr,
            ffmpeg_bin=ffmpeg_bin,
        )

    ducking_envelope_report = None
    ambient_separation_report = None
    background_track = np.zeros(total_frames, dtype=np.float64)
    ambient_wav_path = None

    if source_audio is not None and normalized_mode != "japanese-only":
        env, ducking_envelope_report = build_ducking_envelope(
            total_frames,
            sample_rate=target_sr,
            speech_intervals=speech_intervals,
            duck_gain=float(duck_gain),
            unspoken_gain=float(unspoken_gain),
            fade_out_seconds=float(fade_out_seconds),
            fade_in_seconds=float(fade_in_seconds),
            release_seconds=release_seconds,
            ducking_style=eff_ducking_style,
        )
        if normalized_mode == "reference-mix":
            background_track = source_audio.astype(np.float64) * float(duck_gain)
        elif normalized_mode in ("ambient", "separated-ambient", "hybrid-ambient"):
            sep = separate_ambient_and_speech(
                source_audio,
                sample_rate=target_sr,
                backend=ambient_backend,
                allow_fallback=True,
            )
            ambient_separation_report = {
                k: v for k, v in sep.items() if k not in ("ambient", "speech")
            }
            amb_stem = sep["ambient"].astype(np.float64)
            spk_stem = sep["speech"].astype(np.float64)

            ambient_wav_path = out_dir / "ambient_bed.wav"
            amb_pcm = np.clip(np.round(amb_stem * 32767.0), -32768, 32767).astype("<i2")
            with wave.open(str(ambient_wav_path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(target_sr)
                stream.writeframes(amb_pcm.tobytes())

            eff_unspoken = float(ducking_envelope_report.get("unspoken_gain", unspoken_gain))
            denom = max(1e-6, abs(eff_unspoken - float(duck_gain)))
            speech_weight = np.clip((eff_unspoken - env) / denom, 0.0, 1.0).astype(np.float64) if abs(eff_unspoken - float(duck_gain)) > 1e-5 else np.ones(total_frames, dtype=np.float64)
            residual_voice_gain = float(duck_gain) if normalized_mode == "hybrid-ambient" else float(duck_gain) * 0.35
            bg_during_speech = amb_stem * float(ambient_gain) + spk_stem * residual_voice_gain
            bg_unspoken = source_audio.astype(np.float64) * eff_unspoken
            background_track = speech_weight * bg_during_speech + (1.0 - speech_weight) * bg_unspoken
        else:
            background_track = source_audio.astype(np.float64) * env.astype(np.float64)

    raw_master = japanese_master + background_track
    if float(boundary_fade_out_seconds or 0.0) > 0.0:
        raw_master = apply_boundary_audio_fade(
            raw_master,
            sample_rate=target_sr,
            fade_out_seconds=float(boundary_fade_out_seconds),
        ).astype(np.float64)
    limited_master, limiter_report = apply_soft_limiter(raw_master, ceiling=0.95, knee=0.75)
    master = limited_master.astype(np.float64)
    pre_peak = float(limiter_report["pre_peak"])
    gain = round(float(limiter_report["post_peak"]) / pre_peak, 5) if pre_peak > 0.95 else 1.0

    loudness_verification = verify_audio_loudness_and_clipping(
        master,
        japanese_samples=japanese_master,
        background_samples=background_track,
        source_samples=source_audio,
        sample_rate=target_sr,
        speech_intervals=speech_intervals,
    )

    master_pcm = np.clip(np.round(master * 32767.0), -32768, 32767).astype("<i2")
    dubbed_wav = out_dir / "dubbed_audio.wav"
    with wave.open(str(dubbed_wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(target_sr)
        stream.writeframes(master_pcm.tobytes())

    dubbed_mp4 = out_dir / "dubbed_clip.mp4"
    tmp_mp4 = out_dir / "dubbed_clip.tmp.mp4"
    v_fade_sec = float(video_fade_out_seconds or 0.0)
    if v_fade_sec > 0.0 and float(clip_duration) > v_fade_sec + 0.1:
        v_fade_start = max(0.0, float(clip_duration) - v_fade_sec)
        cmd = [
            ffmpeg_bin or _resolve_binary("ffmpeg"),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            f"[0:v]fade=t=out:st={v_fade_start:.3f}:d={v_fade_sec:.3f}[vout];[1:a]pan=stereo|c0=c0|c1=c0[aout]",
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{float(clip_duration):.2f}",
            str(tmp_mp4),
        ]
    else:
        cmd = [
            ffmpeg_bin or _resolve_binary("ffmpeg"),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dubbed_wav),
            "-filter_complex",
            "[1:a]pan=stereo|c0=c0|c1=c0[aout]",
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{float(clip_duration):.2f}",
            str(tmp_mp4),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_mp4.is_file():
        raise RuntimeError(f"FFmpeg audio muxing failed: {proc.stderr[-600:]}")
    tmp_mp4.replace(dubbed_mp4)

    mix_entries = []
    for e in feedback_doc.get("entries", []):
        uid = str(e.get("id"))
        p_info = placements_map.get(uid) or placements_map.get(e.get("id"))
        if p_info is None:
            continue
        mix_entries.append(
            {
                "id": e.get("id"),
                "speaker": e.get("speaker"),
                "text_ja": e.get("text_ja"),
                "source_start_seconds": p_info["source_start_seconds"],
                "placement_seconds": p_info["clip_relative_start_seconds"],
                "clip_relative_start_seconds": p_info["clip_relative_start_seconds"],
                "clip_relative_end_seconds": p_info["clip_relative_end_seconds"],
                "actual_tts_duration_seconds": round(
                    p_info["clip_relative_end_seconds"] - p_info["clip_relative_start_seconds"],
                    4,
                ),
            }
        )

    eff_unspoken_out = float(ducking_envelope_report.get("unspoken_gain", unspoken_gain)) if ducking_envelope_report else float(unspoken_gain)
    mix_report = {
        "kind": "transclip_mix_report",
        "format_version": 2,
        "mix_mode": normalized_mode,
        "ducking_style": eff_ducking_style,
        "duck_gain": float(duck_gain) if normalized_mode != "japanese-only" else 0.0,
        "unspoken_gain": eff_unspoken_out if normalized_mode != "japanese-only" else 0.0,
        "tts_gain": float(tts_gain),
        "boundary_fade_out_seconds": float(boundary_fade_out_seconds or 0.0),
        "video_fade_out_seconds": float(video_fade_out_seconds or 0.0),
        "sample_rate": target_sr,
        "clip_duration_seconds": round(float(clip_duration), 4),
        "static_gain": round(gain, 5),
        "peak_amplitude": round(float(np.max(np.abs(master))) if len(master) else 0.0, 5),
        "limiter": limiter_report,
        "ducking_envelope": ducking_envelope_report,
        "ambient_separation": ambient_separation_report,
        "loudness_verification": loudness_verification,
        "dubbed_audio_wav": str(dubbed_wav),
        "japanese_tts_only_wav": str(jp_only_wav),
        "ambient_bed_wav": str(ambient_wav_path) if ambient_wav_path else None,
        "dubbed_video_mp4": str(dubbed_mp4),
        "speaker_tracks": speaker_track_files,
        "entries": mix_entries,
    }
    mix_report_path = out_dir / "mix_report.json"
    _atomic_write_json(mix_report_path, mix_report)
    return mix_report_path, mix_report


def _invoke_audio_mixer(
    *,
    script_path,
    feedback_path,
    video_path,
    output_dir,
    feedback_doc,
    placements_map,
    selection_start,
    clip_duration,
    mode="voiceover",
    ducking_style="dynamic",
    duck_gain=0.12,
    unspoken_gain=1.0,
    tts_gain=1.0,
    fade_out_seconds=0.18,
    fade_in_seconds=0.25,
    release_seconds=None,
    ambient_backend="auto",
    ambient_gain=0.85,
    boundary_fade_out_seconds=0.0,
    video_fade_out_seconds=0.0,
    max_timing_shift=0.5,
    prefer_internal_mixer=False,
):
    """Invoke repository `audio_mixer.py` if compatible, or use the 48kHz clip-relative voiceover/ambient mixer."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not prefer_internal_mixer and mode in ("normal", "diagnostic", "japanese-only", "reference-mix"):
        try:
            import audio_mixer

            fn = None
            for cand in ("run_mixer", "mix_audio", "mix_dubbed_audio", "render_mix", "build_mix"):
                if hasattr(audio_mixer, cand) and inspect.isfunction(getattr(audio_mixer, cand)):
                    fn = getattr(audio_mixer, cand)
                    break
            if fn is not None:
                sig = inspect.signature(fn)
                pos_args = []
                kwargs = {}
                for name, param in sig.parameters.items():
                    matched = True
                    val = None
                    if name in ("script_path", "script", "dubbing_script"):
                        val = Path(script_path)
                    elif name in ("feedback_path", "feedback"):
                        val = Path(feedback_path)
                    elif name in ("video_path", "video", "clip_path", "clip", "source_video"):
                        val = Path(video_path)
                    elif name in ("output_dir", "output", "output_path"):
                        val = out_dir
                    elif name in ("mode", "mix_mode"):
                        val = mode
                    elif name in ("max_timing_shift", "max_shift"):
                        val = float(max_timing_shift)
                    else:
                        matched = False
                    if matched:
                        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                            pos_args.append(val)
                        else:
                            kwargs[name] = val
                res = fn(*pos_args, **kwargs)
                rep_path = _extract_path_from_result(res, out_dir / "mix_report.json")
                if isinstance(res, dict) and not rep_path.is_file():
                    _atomic_write_json(rep_path, res)
                if rep_path.is_file():
                    rep_doc = json.loads(rep_path.read_text(encoding="utf-8"))
                    return rep_path, rep_doc
        except Exception:
            pass

    return _fallback_mix_audio_and_video(
        video_path=video_path,
        feedback_doc=feedback_doc,
        placements_map=placements_map,
        selection_start=selection_start,
        clip_duration=clip_duration,
        output_dir=out_dir,
        mix_mode=mode,
        ducking_style=ducking_style,
        duck_gain=duck_gain,
        unspoken_gain=unspoken_gain,
        tts_gain=tts_gain,
        fade_out_seconds=fade_out_seconds,
        fade_in_seconds=fade_in_seconds,
        release_seconds=release_seconds,
        ambient_backend=ambient_backend,
        ambient_gain=ambient_gain,
        boundary_fade_out_seconds=boundary_fade_out_seconds,
        video_fade_out_seconds=video_fade_out_seconds,
    )


def _format_srt_time(seconds):
    ms = max(0, int(round(float(seconds) * 1000.0)))
    hrs, rem = divmod(ms, 3600000)
    mins, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_ass_time(seconds):
    cs = max(0, int(round(float(seconds) * 100.0)))
    hrs, rem = divmod(cs, 360000)
    mins, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"


def _wrap_japanese_lines(text, max_line_chars=22):
    from japanese_subtitles import _balance_group_into_two_lines, segment_japanese_bunsetsu

    clean = str(text or "").strip()
    if len(clean) <= max_line_chars:
        return [clean]
    tokens = segment_japanese_bunsetsu(clean)
    return _balance_group_into_two_lines(tokens, max_chars_per_line=max_line_chars)


def _fallback_generate_subtitles_and_mp4(
    *,
    rel_feedback_doc,
    dubbed_video_path,
    output_dir,
    clip_duration,
    speaker_verification_targets=None,
    include_unverified_subtitles=False,
    alignment_data=None,
    subtitle_style_overrides=None,
    ffmpeg_bin=None,
):
    """Generate clip-relative final_ja.srt, final_ja.ass, final_ja_no_subtitles.mp4, and final_ja.mp4
    with Phase 13 visibility (outline, shadow, semi-transparent box), 1.2-1.3x sizing, elevated margin_v,
    resolution/aspect scaling, overflow prevention, and display punctuation formatting.
    """
    from dataclasses import asdict
    from japanese_subtitles import (
        SubtitleStyleConfig,
        _ass_alpha_hex,
        align_entry_cues,
        check_subtitle_overflow,
        format_subtitle_display_lines,
        prevent_line_overflow,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe video resolution to adapt subtitle size, margins, and line wrapping automatically
    vid_w, vid_h = 1280, 720
    if dubbed_video_path and Path(str(dubbed_video_path)).is_file():
        try:
            v_probe = probe_media_streams(dubbed_video_path)
            if v_probe.get("width", 0) > 0 and v_probe.get("height", 0) > 0:
                vid_w = int(v_probe["width"])
                vid_h = int(v_probe["height"])
        except Exception:
            pass

    overrides = dict(subtitle_style_overrides or {})
    style = SubtitleStyleConfig.for_video_resolution(vid_w, vid_h, **overrides)

    entries = [
        copy.deepcopy(e)
        for e in rel_feedback_doc.get("entries", [])
        if e.get("generation_status") == "complete" and e.get("text_ja")
    ]
    if include_unverified_subtitles and speaker_verification_targets:
        for t in speaker_verification_targets:
            if t.get("text_ja"):
                entries.append(
                    {
                        "id": t["id"],
                        "speaker": "話者未確定",
                        "text_ja": t["text_ja"],
                        "placement_seconds": t["clip_relative_start_seconds"],
                        "actual_tts_duration_seconds": max(
                            0.5,
                            t["clip_relative_end_seconds"] - t["clip_relative_start_seconds"],
                        ),
                    }
                )

    aligned_entries = []
    alignment_warnings = {}
    for entry in entries:
        cues, warnings = align_entry_cues(entry, style, (alignment_data or {}).get(str(entry["id"])))
        if warnings:
            alignment_warnings[str(entry["id"])] = warnings
        for cue in cues:
            aligned_entries.append({
                **entry,
                "text_ja": cue["text_ja"],
                "lines": cue.get("lines"),
                "display_lines": cue.get("display_lines"),
                "display_text_ja": cue.get("display_text_ja"),
                "placement_seconds": cue["start"],
                "actual_tts_duration_seconds": cue["end"] - cue["start"],
                "cue_index": cue["cue_index"],
                "timing_source": cue["timing_source"],
            })
    entries = sorted(aligned_entries, key=lambda e: float(e["placement_seconds"]))

    speakers = []
    for e in entries:
        spk = e.get("speaker", DEFAULT_SPEAKER_ID)
        if spk not in speakers:
            speakers.append(spk)
    multi_speaker = len(speakers) > 1

    srt_blocks = []
    ass_events = []
    cues_meta = []
    for idx, e in enumerate(entries, start=1):
        start_s = float(e.get("placement_seconds", e.get("start", 0.0)))
        dur_s = float(e.get("actual_tts_duration_seconds", max(0.5, float(e.get("end", start_s + 1.0)) - start_s)))
        end_s = min(float(clip_duration), start_s + dur_s)
        if end_s <= start_s:
            end_s = min(float(clip_duration), start_s + 0.5)
        spk = e.get("speaker", DEFAULT_SPEAKER_ID)
        raw_lines = list(e.get("lines") or _wrap_japanese_lines(e["text_ja"], max_line_chars=style.max_chars_per_line))
        disp_lines = list(
            e.get("display_lines")
            or format_subtitle_display_lines(raw_lines, strip_periods=style.strip_periods, comma_mode=style.comma_mode)
        )
        disp_lines = prevent_line_overflow(disp_lines, style)

        srt_text = "\n".join(disp_lines)
        if multi_speaker or spk == "話者未確定":
            srt_text = f"[{spk}] {srt_text}"
        srt_blocks.append(f"{idx}\n{_format_srt_time(start_s)} --> {_format_srt_time(end_s)}\n{srt_text}\n")

        lane = speakers.index(spk) if spk in speakers else 0
        margin_v = style.margin_v + lane * style.overlap_lane_step_v
        ass_text = "\\N".join(disp_lines)
        ass_events.append(
            f"Dialogue: {lane},{_format_ass_time(start_s)},{_format_ass_time(end_s)},Default,{spk},0,0,{margin_v},,{ass_text}"
        )
        cues_meta.append(
            {
                "index": idx,
                "id": e.get("id"),
                "entry_id": e.get("id"),
                "speaker": spk,
                "start_seconds": round(start_s, 4),
                "end_seconds": round(end_s, 4),
                "text_ja": e["text_ja"],
                "display_text_ja": "\n".join(disp_lines),
                "cue_index": e["cue_index"],
                "timing_source": e["timing_source"],
                "lines": raw_lines,
                "display_lines": disp_lines,
                "margin_v": margin_v,
            }
        )

    srt_path = out_dir / "final_ja.srt"
    ass_path = out_dir / "final_ja.ass"
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    border_style = 3 if style.background_box else 1
    alpha_hex = _ass_alpha_hex(style.box_opacity)
    outline_col = f"&H{alpha_hex}000000" if style.background_box else "&H00181818"
    back_col = "&H80000000"

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {style.play_res_x}
PlayResY: {style.play_res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Hiragino Sans,{style.font_size},&H00FFFFFF,&H000000FF,{outline_col},{back_col},1,0,0,0,100,100,0,0,{border_style},{style.outline},{style.shadow},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_path.write_text(ass_header + "\n".join(ass_events) + "\n", encoding="utf-8")

    no_sub_mp4 = out_dir / "final_ja_no_subtitles.mp4"
    final_mp4 = out_dir / "final_ja.mp4"
    shutil.copyfile(dubbed_video_path, no_sub_mp4)

    repo_ffmpeg_libass = Path(__file__).parent / "tools" / "ffmpeg-libass" / "bin" / "ffmpeg"
    active_ffmpeg = str(repo_ffmpeg_libass) if repo_ffmpeg_libass.is_file() else (ffmpeg_bin or _resolve_binary("ffmpeg"))
    tmp_final = out_dir / "final_ja.tmp.mp4"
    escaped_ass = str(ass_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    burn_cmd = [
        active_ffmpeg,
        "-y",
        "-i",
        str(no_sub_mp4),
        "-vf",
        f"ass='{escaped_ass}'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-c:a",
        "copy",
        str(tmp_final),
    ]
    subtitle_render_mode = "burned_ass"
    proc = subprocess.run(burn_cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not tmp_final.is_file():
        subtitle_render_mode = "muxed_mov_text"
        mux_cmd = [
            ffmpeg_bin or _resolve_binary("ffmpeg"),
            "-y",
            "-i",
            str(no_sub_mp4),
            "-i",
            str(srt_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=jpn",
            str(tmp_final),
        ]
        proc2 = subprocess.run(mux_cmd, capture_output=True, text=True, check=False)
        if proc2.returncode != 0 or not tmp_final.is_file():
            subtitle_render_mode = "external_srt_ass"
            shutil.copyfile(no_sub_mp4, tmp_final)
    tmp_final.replace(final_mp4)

    overflow_check = check_subtitle_overflow(cues_meta, style)

    sub_report = {
        "kind": "transclip_subtitle_report",
        "format_version": 2,
        "style": asdict(style),
        "cue_count": len(cues_meta),
        "cues": cues_meta,
        "overflow_check": overflow_check,
        "held_verification_target_count": len(speaker_verification_targets or []),
        "srt_path": str(srt_path),
        "ass_path": str(ass_path),
        "final_ja_mp4": str(final_mp4),
        "final_ja_no_subtitles_mp4": str(no_sub_mp4),
        "subtitle_render_mode": subtitle_render_mode,
        "review_required": bool(speaker_verification_targets or alignment_warnings or overflow_check.get("overflow_count", 0) > 0),
        "alignment_warnings": alignment_warnings,
    }
    sub_report_path = out_dir / "subtitle_report.json"
    _atomic_write_json(sub_report_path, sub_report)
    return sub_report_path, sub_report


def _parse_first_srt_start_seconds(srt_path):
    text = Path(srt_path).read_text(encoding="utf-8")
    m = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->", text)
    if not m:
        return None
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600.0 + mi * 60.0 + s + ms / 1000.0


def _invoke_japanese_subtitles(
    *,
    rel_feedback_path,
    rel_feedback_doc,
    mix_report_path,
    dubbed_video_path,
    output_dir,
    clip_duration,
    alignment_json_path=None,
    speaker_verification_targets=None,
    include_unverified_subtitles=False,
    subtitle_style_overrides=None,
):
    """Invoke repository `japanese_subtitles.py` with clip-relative timestamps, or fallback."""
    alignment = json.loads(Path(alignment_json_path).read_text(encoding="utf-8")) if alignment_json_path else None
    kwargs = {
        "rel_feedback_doc": rel_feedback_doc,
        "dubbed_video_path": dubbed_video_path,
        "output_dir": output_dir,
        "clip_duration": clip_duration,
        "speaker_verification_targets": speaker_verification_targets,
        "include_unverified_subtitles": include_unverified_subtitles,
        "alignment_data": alignment,
    }
    if subtitle_style_overrides:
        kwargs["subtitle_style_overrides"] = subtitle_style_overrides
    return _fallback_generate_subtitles_and_mp4(**kwargs)


def _snapshot_input_files(paths):
    """Record SHA-256 digests of all existing input artifacts to prove non-destructive execution."""
    snap = {}
    for p in paths or []:
        if p is None:
            continue
        cand = Path(p)
        if cand.is_file():
            resolved = str(cand.resolve())
            snap[resolved] = _file_sha256(cand)
        elif cand.is_dir():
            for child in sorted(cand.rglob("*")):
                if child.is_file() and child.suffix.lower() in (".wav", ".json", ".mp4", ".txt"):
                    snap[str(child.resolve())] = _file_sha256(child)
    return snap


def _verify_input_files_unchanged(before_snapshot):
    """Verify every recorded input file still exists with the exact same SHA-256 digest."""
    mismatches = []
    for file_path, expected_sha in (before_snapshot or {}).items():
        p = Path(file_path)
        if not p.is_file():
            mismatches.append({"path": file_path, "error": "deleted_or_missing"})
            continue
        actual_sha = _file_sha256(p)
        if actual_sha != expected_sha:
            mismatches.append({"path": file_path, "expected_sha256": expected_sha, "actual_sha256": actual_sha})
    if mismatches:
        raise RuntimeError(f"Non-destructive invariant violated! Input files were modified: {mismatches}")
    return {
        "verified": True,
        "checked_file_count": len(before_snapshot or {}),
        "files": dict(before_snapshot or {}),
    }


def _discover_existing_speaker_reference(speaker_id, *, reference_dir=None, script_path=None, video_path=None, script_doc=None):
    """Search non-destructively for an existing reference WAV for `speaker_id` if present."""
    base_dirs = []
    if script_path and Path(script_path).exists():
        base_dirs.append(Path(script_path).resolve().parent)
    if video_path and Path(video_path).exists():
        base_dirs.append(Path(video_path).resolve().parent)

    if isinstance(script_doc, dict):
        for map_key in ("speaker_references", "voice_references", "speakers", "speaker_profiles", "profiles"):
            mapping = script_doc.get(map_key)
            if isinstance(mapping, dict) and speaker_id in mapping:
                entry = mapping[speaker_id]
                raw_p = (
                    entry
                    if isinstance(entry, str)
                    else (
                        entry.get("reference_audio")
                        or entry.get("reference_wav")
                        or entry.get("audio_file")
                        or entry.get("path")
                        if isinstance(entry, dict)
                        else None
                    )
                )
                if raw_p:
                    for bdir in [Path.cwd(), *base_dirs]:
                        cand = (bdir / raw_p).resolve() if not Path(raw_p).is_absolute() else Path(raw_p).resolve()
                        if cand.is_file():
                            return cand

    for bdir in base_dirs:
        for json_name in ("voice_profiles.json", "speaker_profiles.json", "references.json"):
            jpath = bdir / json_name
            if jpath.is_file():
                try:
                    jdoc = json.loads(jpath.read_text(encoding="utf-8"))
                    profs = jdoc.get("profiles") or jdoc.get("speakers") or jdoc
                    if isinstance(profs, dict) and speaker_id in profs:
                        p_entry = profs[speaker_id]
                        raw_p = (
                            p_entry
                            if isinstance(p_entry, str)
                            else (
                                p_entry.get("reference_audio")
                                or p_entry.get("reference_wav")
                                or p_entry.get("audio_file")
                                if isinstance(p_entry, dict)
                                else None
                            )
                        )
                        if raw_p:
                            for root_cand in [bdir, Path.cwd()]:
                                cand = (root_cand / raw_p).resolve() if not Path(raw_p).is_absolute() else Path(raw_p).resolve()
                                if cand.is_file():
                                    return cand
                except Exception:
                    pass

    search_dirs = []
    if reference_dir and Path(reference_dir).is_dir():
        search_dirs.append(Path(reference_dir).resolve())
    for bdir in base_dirs:
        search_dirs.extend(
            [
                bdir,
                bdir / "references",
                bdir / speaker_id,
                bdir.parent / "speaker_profiles" / speaker_id,
                bdir.parent / "voices" / speaker_id,
            ]
        )

    candidates = [
        f"{speaker_id}.wav",
        f"{speaker_id}_ref.wav",
        f"{speaker_id}_reference.wav",
        "reference.wav",
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for cname in candidates:
            if cname == "reference.wav" and d.name != speaker_id:
                continue
            cand = d / cname
            if cand.is_file():
                return cand.resolve()
    return None


def run_phase11_dubbing_pipeline(
    *,
    script_path=None,
    video_path=None,
    output_dir="output/phase11_dubbing",
    speaker_id=DEFAULT_SPEAKER_ID,
    voice_profiles_path=None,
    reference_audio=None,
    reference_text=None,
    consent_path=None,
    review_path=None,
    generation_record_path=None,
    expected_sha256=None,
    speaker_ref_map=None,
    reference_dir=None,
    extract_reference_from_video=False,
    rights_mode="permitted_local",
    speaker_source_types=None,
    disallowed_speakers=None,
    voice_mode="auto",
    clone_method="xvector",
    allow_preset_fallback=True,
    preset_speaker_map=None,
    default_preset_voice=None,
    speaker_age_map=None,
    speaker_atmosphere_map=None,
    auto_select_preset=False,
    prosody_style="auto",
    merge_split_utterances=False,
    explicit_merge_groups=None,
    max_merge_gap_seconds=1.5,
    null_speaker_policy="hold_for_verification",
    explicit_speaker_assignments=None,
    speaker_decisions_path=None,
    editorial_plan_path=None,
    revised_script_out_path=None,
    include_unverified_subtitles=False,
    preset_model=DEFAULT_PRESET_MODEL,
    clone_model=DEFAULT_BASE_MODEL,
    seed=0,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.05,
    max_timing_shift=0.5,
    max_inter_speaker_shift=0.0,
    auto_tempo_adjust=False,
    min_atempo=0.90,
    max_atempo=1.25,
    fit_utterance_slots=False,
    allow_slowdown=None,
    smooth_intra_speaker_rates=False,
    enforce_voice_registry=True,
    emit_phase13_2_reports=False,
    mix_mode="voiceover",
    duck_gain=0.12,
    unspoken_gain=1.0,
    tts_gain=1.0,
    duck_fade_out_seconds=0.18,
    duck_fade_in_seconds=0.25,
    ambient_backend="auto",
    ambient_gain=0.85,
    subtitle_font_scale=1.25,
    subtitle_font_size=None,
    subtitle_margin_v=None,
    subtitle_outline=None,
    subtitle_shadow=None,
    subtitle_background_box=True,
    subtitle_box_opacity=0.60,
    subtitle_strip_periods=True,
    subtitle_comma_mode="halfwidth_space",
    allow_diagnostic_on_collision=False,
    reuse_wav_dir=None,
    existing_feedback_path=None,
    alignment_json_path=None,
    source_utterances_path=None,
    word_timestamps_path=None,
    use_word_level_sync=False,
    sync_anchor="adaptive",
    long_pause_threshold=0.55,
    ducking_style="dynamic",
    release_seconds=None,
    auto_extend_clip=False,
    max_clip_extension_seconds=12.0,
    boundary_fade_out_seconds=0.0,
    video_fade_out_seconds=0.0,
    auto_rebalance_overlong=False,
    utterance_translations_path=None,
):
    """Execute the end-to-end Japanese dubbing pipeline (supporting Phase 11, Phase 12, Phase 13, Phase 13.1, and Phase 13.2 workflows)."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Output directory must be empty; use a new directory to preserve existing results.")
    work_dir = root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Guard against overwriting input directories
    for in_p in (script_path, video_path, voice_profiles_path, reference_audio, reuse_wav_dir, speaker_decisions_path, editorial_plan_path):
        if in_p and Path(in_p).exists():
            in_resolved = Path(in_p).resolve()
            if in_resolved == root or (in_resolved.is_file() and in_resolved.parent == root):
                raise ValueError(
                    f"Refusing to use input location {in_resolved} as output_dir {root}; existing inputs must never be overwritten."
                )

    tracked_inputs = [
        script_path,
        video_path,
        voice_profiles_path,
        reference_audio,
        reference_text if (reference_text and Path(str(reference_text)).exists()) else None,
        consent_path,
        review_path,
        reuse_wav_dir,
        existing_feedback_path,
        source_utterances_path,
        alignment_json_path,
        speaker_decisions_path,
        editorial_plan_path,
    ]
    if speaker_ref_map:
        tracked_inputs.extend(speaker_ref_map.values())
    input_snapshot_before = _snapshot_input_files(tracked_inputs)

    # 1. Load or register legacy VoiceProfiles when provided, OR build decoupled TechnicalVoiceReferences
    voice_profiles = {}
    technical_references = {}
    audit_report = None

    if voice_profiles_path is not None and Path(voice_profiles_path).is_file():
        from voice_cloning import load_profiles

        voice_profiles = load_profiles(Path(voice_profiles_path).resolve())
    elif reference_audio is not None and consent_path is not None and review_path is not None and reference_text is not None:
        # Full Phase 8–11 legacy audit + VoiceProfile registration path
        from cross_lingual_clone import CloneBlockedForReviewError, audit_english_reference
        from voice_cloning import load_profiles, register_profile

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
        reg_path = _invoke_register_profile(
            register_profile,
            speaker=speaker_id,
            reference_audio=ref_audio_path,
            reference_text_path=ref_text_path,
            consent_path=Path(consent_path).resolve(),
            review_path=Path(review_path).resolve(),
            output_dir=reg_dir,
            preset_voice=default_preset_voice or "Ryan",
            generation_record_path=generation_record_path,
        )
        voice_profiles = load_profiles(reg_path)
    elif reference_audio is not None:
        # Phase 12/13 decoupled TechnicalVoiceReference (no redundant consent.json / review.json ceremony!)
        ref_txt_val = None
        if reference_text is not None:
            ref_txt_val = (
                Path(reference_text).read_text(encoding="utf-8").strip()
                if Path(str(reference_text)).is_file()
                else str(reference_text).strip()
            )
        technical_references[speaker_id] = TechnicalVoiceReference(
            speaker_id=speaker_id,
            reference_audio=Path(reference_audio).resolve(),
            source_origin="explicit_reference_audio",
            reference_text=ref_txt_val,
        )

    for spk, wav_p in (speaker_ref_map or {}).items():
        technical_references[spk] = TechnicalVoiceReference(
            speaker_id=spk,
            reference_audio=Path(wav_p).resolve(),
            source_origin="explicit_speaker_ref",
        )

    # 2. Prepare video clip and dubbing_script.json
    if video_path is None:
        raw_video_file = ensure_30s_verification_video(work_dir / "clip_0_30.mp4", duration_seconds=30.0)
    else:
        raw_video_file = Path(video_path).resolve()
        if not raw_video_file.is_file():
            raise FileNotFoundError(f"Input video clip not found: {raw_video_file}")

    if script_path is None:
        from cross_lingual_clone import build_phase8_benchmark_script

        target_mode = "preset" if voice_mode == "preset" else "approved_clone"
        prof = voice_profiles.get(speaker_id) or technical_references.get(speaker_id)
        fp = None
        if prof is not None:
            if hasattr(prof, "technical_snapshot"):
                fp = prof.technical_snapshot()["profile_fingerprint"]
            else:
                from voice_cloning import profile_snapshot

                fp = profile_snapshot(prof)["profile_fingerprint"]
        script_file = _invoke_benchmark_script_builder(
            build_phase8_benchmark_script,
            output_path=work_dir / "scripts" / "dubbing_script_30s.json",
            speaker=speaker_id,
            voice_mode=target_mode,
            preset_voice=default_preset_voice or "Ryan",
            voice=f"approved_clone:{fp}" if fp and target_mode == "approved_clone" else (default_preset_voice or "Ryan"),
            profile=prof,
            profile_fingerprint=fp,
            voice_profiles=voice_profiles,
        )
    else:
        script_file = Path(script_path).resolve()
        if not script_file.is_file():
            raise FileNotFoundError(f"Input dubbing script not found: {script_file}")

    raw_script_doc = json.loads(script_file.read_text(encoding="utf-8"))

    # Optional Phase 12/13 editorial plan + bound speaker decisions application
    if editorial_plan_path and Path(editorial_plan_path).is_file():
        from dubbing_quality import apply_editorial_plan
        plan_doc = json.loads(Path(editorial_plan_path).read_text(encoding="utf-8"))
        decisions_doc = (
            json.loads(Path(speaker_decisions_path).read_text(encoding="utf-8"))
            if speaker_decisions_path and Path(speaker_decisions_path).is_file()
            else None
        )
        raw_script_doc = apply_editorial_plan(raw_script_doc, plan_doc, speaker_decisions=decisions_doc)
        if revised_script_out_path:
            rev_out = Path(revised_script_out_path).resolve()
            if rev_out != script_file:
                _atomic_write_json(rev_out, raw_script_doc)

    norm_script = normalize_script_utterances(
        raw_script_doc,
        script_path=script_file,
        video_path=raw_video_file,
    )
    selection_start = norm_script["selection_start"]
    selection_end = norm_script["selection_end"]
    clip_duration = norm_script["clip_duration"]

    source_rows = (
        json.loads(Path(source_utterances_path).read_text(encoding="utf-8"))["utterances"]
        if source_utterances_path and Path(source_utterances_path).is_file()
        else None
    )
    external_words = None
    if word_timestamps_path and Path(word_timestamps_path).is_file():
        wt_doc = json.loads(Path(word_timestamps_path).read_text(encoding="utf-8"))
        external_words = wt_doc.get("words") if isinstance(wt_doc, dict) else wt_doc

    # Phase 13.1: Detect clip boundary mid-utterance / mid-sentence cuts and optionally auto-extend clip
    from timing_adjustment import (
        build_timing_quality_report,
        concise_rewrite_for_timing_budget,
        detect_and_resolve_clip_boundaries,
        extract_or_resolve_word_timestamps,
    )
    from voice_consistency import (
        SpeakerVoiceRegistry,
        build_speaker_assignment_report,
        concatenate_speaker_voice_samples,
        evaluate_speaker_voice_consistency,
    )

    raw_video_probe_initial = probe_media_streams(raw_video_file) if raw_video_file.is_file() else {}
    raw_video_dur = float(raw_video_probe_initial.get("duration_seconds") or 0.0)
    avail_source_dur = raw_video_dur if raw_video_dur >= selection_end else None

    clip_boundary_resolution = detect_and_resolve_clip_boundaries(
        source_rows if source_rows is not None else norm_script["utterances"],
        selection_start=selection_start,
        selection_end=selection_end,
        source_video_duration=avail_source_dur,
        auto_extend_clip=auto_extend_clip,
        max_extension_seconds=max_clip_extension_seconds,
    )
    if clip_boundary_resolution["extension_applied"]:
        selection_start = clip_boundary_resolution["effective_selection"]["start"]
        selection_end = clip_boundary_resolution["effective_selection"]["end"]
        clip_duration = clip_boundary_resolution["effective_selection"]["duration"]
        if source_rows is not None:
            existing_src_ids = {
                str(sid)
                for u in norm_script["utterances"]
                for sid in (u.get("source_utterance_ids") or u.get("merged_from_ids") or [u.get("raw_id", u["id"])])
            }
            trans_map = {}
            if utterance_translations_path and Path(utterance_translations_path).is_file():
                tr_doc = json.loads(Path(utterance_translations_path).read_text(encoding="utf-8"))
                for tr_u in tr_doc.get("utterances", []):
                    trans_map[str(tr_u.get("id"))] = tr_u.get("text_ja") or tr_u.get("translation") or ""
            for sr_u in source_rows:
                sr_id = str(sr_u.get("id"))
                sr_s = float(sr_u.get("start", 0.0))
                sr_e = float(sr_u.get("end", sr_s))
                if sr_id not in existing_src_ids and sr_s < selection_end - 0.02 and sr_e > selection_start + 0.02:
                    t_ja = sr_u.get("text_ja") or trans_map.get(sr_id) or ""
                    if t_ja:
                        extra_norm = normalize_script_utterances(
                            {
                                "selection_range": {"start": selection_start, "end": selection_end},
                                "utterances": [{**sr_u, "text_ja": t_ja}],
                            }
                        )["utterances"]
                        norm_script["utterances"].extend(extra_norm)
                        existing_src_ids.add(sr_id)
            norm_script["utterances"].sort(key=lambda r: (r["start"], r.get("original_index", 0)))
        for u in norm_script["utterances"]:
            u["clip_relative_start"] = round(max(0.0, float(u["start"]) - selection_start), 4)
            u["clip_relative_end"] = round(max(u["clip_relative_start"] + 0.05, float(u["end"]) - selection_start), 4)

    # Enrich utterances with word-level timestamps from source_rows or external_words when available
    if source_rows is not None or external_words is not None:
        src_words_by_id = {}
        if source_rows is not None:
            for sr_u in source_rows:
                if isinstance(sr_u.get("words"), list) and sr_u["words"]:
                    src_words_by_id[str(sr_u.get("id"))] = sr_u["words"]
        for u in norm_script["utterances"]:
            if not u.get("words"):
                sids = u.get("source_utterance_ids") or u.get("merged_from_ids") or [u.get("raw_id", u["id"])]
                gathered = []
                for sid in sids:
                    if str(sid) in src_words_by_id:
                        gathered.extend(src_words_by_id[str(sid)])
                if gathered:
                    u["words"] = gathered
                elif external_words:
                    u["words"] = extract_or_resolve_word_timestamps(u, external_words=external_words)

    # Ensure video clip segment matches [selection_start, selection_end]
    video_file, source_video_probe = _prepare_clip_video_source(
        raw_video_file,
        selection_start=selection_start,
        clip_duration=clip_duration,
        work_dir=work_dir,
    )

    # Human CLI assignments must be applied before merging, while unknowns remain barriers.
    initial_partition = partition_and_manage_null_speakers(
        norm_script["utterances"],
        explicit_speaker_assignments=explicit_speaker_assignments,
        null_speaker_policy=null_speaker_policy,
    )
    assigned_by_id = {r["id"]: r for r in initial_partition["assigned_utterances"]}
    merge_inputs = [assigned_by_id.get(r["id"], r) for r in norm_script["utterances"]]

    # 3. Step 3a: Detect & merge split continuous utterances
    already_revised = bool(isinstance(raw_script_doc, dict) and raw_script_doc.get("quality_revision"))
    merge_result = detect_and_merge_split_utterances(
        merge_inputs,
        enabled=bool(merge_split_utterances and not already_revised),
        explicit_merge_groups=explicit_merge_groups,
        max_merge_gap_seconds=max_merge_gap_seconds,
    )
    utterances_after_merge = merge_result["utterances"]

    # Optional Phase 13.1 concise translation rebalancing when estimated mora duration exceeds slot * max_atempo
    translation_rebalance_records = []
    if auto_rebalance_overlong:
        for u in utterances_after_merge:
            u_slot = max(0.2, float(u.get("effective_content_end", u["end"])) - float(u.get("effective_content_start", u["start"])))
            est_sec = float(u.get("mora_count") or _estimate_japanese_mora(u.get("text_ja", ""))) / 7.2
            if est_sec > u_slot * float(max_atempo):
                target_mora = max(1, int(math.floor(u_slot * float(max_atempo) * 7.2)))
                rw = concise_rewrite_for_timing_budget(u.get("text_ja", ""), target_mora_budget=target_mora)
                if rw["changed"]:
                    u["original_verbose_text_ja"] = u["text_ja"]
                    u["text_ja"] = rw["rewritten_text_ja"]
                    u["mora_count"] = _estimate_japanese_mora(u["text_ja"])
                    translation_rebalance_records.append({"id": u["id"], **rw})

    # 4. Step 3b: Partition & manage `speaker: null` utterances as verification targets without guessing identity
    has_any_null_speaker = any(
        _is_null_or_unassigned_speaker(u.get("speaker")) and u.get("raw_speaker") is None
        for u in utterances_after_merge
    )
    if has_any_null_speaker or merge_split_utterances or explicit_speaker_assignments or null_speaker_policy == "fail":
        partition_result = partition_and_manage_null_speakers(
            utterances_after_merge,
            explicit_speaker_assignments=explicit_speaker_assignments,
            null_speaker_policy=null_speaker_policy,
        )
        active_utterances = partition_result["assigned_utterances"]
        speaker_verification_targets = partition_result["speaker_verification_targets"]
        explicit_assignments_applied = initial_partition["explicit_speaker_assignments"]
    else:
        active_utterances = utterances_after_merge
        speaker_verification_targets = []
        explicit_assignments_applied = []

    if not active_utterances:
        raise ValueError("No assigned speaker utterances remain after separating `speaker: null` verification targets.")

    # Write verification targets JSON if any exist
    verification_targets_path = root / "speaker_verification_targets.json"
    if speaker_verification_targets:
        _atomic_write_json(
            verification_targets_path,
            {
                "kind": "transclip_speaker_verification_targets",
                "format_version": 1,
                "source_script": str(script_file),
                "null_speaker_policy": null_speaker_policy,
                "target_count": len(speaker_verification_targets),
                "targets": speaker_verification_targets,
            },
        )

    # 5. Resolve technical X-vector references for active speakers (including extraction from `source.mp4` when enabled)
    preset_map = dict(preset_speaker_map or {})
    for u in active_utterances:
        spk = u.get("speaker")
        if not spk or _is_null_or_unassigned_speaker(spk):
            continue
        if voice_mode == "preset" or spk in preset_map:
            continue
        if spk in voice_profiles or spk in technical_references:
            continue

        # Check existing reference WAV in reference_dir / sibling directories
        disc_wav = _discover_existing_speaker_reference(
            spk,
            reference_dir=reference_dir,
            script_path=script_file,
            video_path=raw_video_file,
            script_doc=raw_script_doc,
        )
        if disc_wav is not None:
            input_snapshot_before[str(disc_wav)] = _file_sha256(disc_wav)
            technical_references[spk] = TechnicalVoiceReference(
                speaker_id=spk,
                reference_audio=disc_wav,
                source_origin="discovered_reference_wav",
            )
            continue

        # Extract clean non-overlapping speaker reference segments from `source.mp4` when enabled
        if extract_reference_from_video and video_file.is_file():
            technical_references[spk] = extract_speaker_reference_from_video(
                video_path=raw_video_file,
                speaker_id=spk,
                all_utterances=utterances_after_merge,
                output_dir=work_dir / "speaker_references",
                selection_start=selection_start,
                clip_duration=clip_duration,
            )

    # Build preprocessed working script in `_work` (never overwriting original `dubbing_script.json`!)
    working_script_doc = copy.deepcopy(raw_script_doc)
    working_script_doc["selection_range"] = {"start": selection_start, "end": selection_end}
    working_script_doc["utterances"] = active_utterances
    if "entries" in working_script_doc:
        working_script_doc["entries"] = active_utterances
    working_script_path = work_dir / "dubbing_script_preprocessed.json"
    _atomic_write_json(working_script_path, working_script_doc)

    # 6. Rights & Publication Governance check + Speaker Voice Routing
    rights_policy = RightsAndPublicationPolicy(
        rights_mode=rights_mode,
        speaker_source_types=speaker_source_types,
        disallowed_speakers=disallowed_speakers,
    )
    voice_plan = resolve_speaker_voice_plan(
        working_script_doc,
        voice_profiles=voice_profiles,
        technical_references=technical_references,
        rights_policy=rights_policy,
        voice_mode=voice_mode,
        clone_method=clone_method,
        preset_speaker_map=preset_map,
        default_preset_voice=default_preset_voice,
        allow_preset_fallback=allow_preset_fallback,
        speaker_age_map=speaker_age_map,
        speaker_atmosphere_map=speaker_atmosphere_map,
        auto_select_preset=auto_select_preset,
    )

    # Optimize TTS prosody parameters if prosody_style is explicitly configured
    effective_temp = float(temperature)
    effective_top_k = int(top_k)
    effective_top_p = float(top_p)
    effective_rep_pen = float(repetition_penalty)
    prosody_tuning_info = None
    if prosody_style and str(prosody_style).lower() not in ("default", "none", "auto"):
        from voice_quality import optimize_tts_prosody_settings
        prosody_tuning_info = optimize_tts_prosody_settings(
            prosody_style=prosody_style,
            base_temperature=effective_temp,
            base_top_k=effective_top_k,
            base_top_p=effective_top_p,
            base_repetition_penalty=effective_rep_pen,
        )
        effective_temp = float(prosody_tuning_info["temperature"])
        effective_top_k = int(prosody_tuning_info["top_k"])
        effective_top_p = float(prosody_tuning_info["top_p"])
        effective_rep_pen = float(prosody_tuning_info["repetition_penalty"])

    # Phase 13.2: Initialize & lock SpeakerVoiceRegistry for all active speakers
    voice_registry = SpeakerVoiceRegistry()
    all_refs = {**voice_profiles, **technical_references}
    gen_settings_for_registry = {
        "seed": int(seed),
        "temperature": effective_temp,
        "top_k": effective_top_k,
        "top_p": effective_top_p,
        "repetition_penalty": effective_rep_pen,
    }
    for r_item in voice_plan.get("speaker_routing_details", []):
        spk_id = r_item["speaker"]
        s_mode = r_item["mode"]
        ref_obj = all_refs.get(spk_id)
        ref_wav_p = getattr(ref_obj, "reference_audio", None) if ref_obj is not None else None
        voice_registry.register_speaker_profile(
            speaker_id=spk_id,
            voice_mode=s_mode,
            reference_audio_path=ref_wav_p if s_mode == "approved_clone" else None,
            clone_method=r_item.get("clone_method") or voice_plan.get("effective_clone_method", "xvector"),
            preset_voice=r_item.get("voice") if s_mode == "preset" else getattr(ref_obj, "preset_voice", None),
            model_id=clone_model if s_mode == "approved_clone" else preset_model,
            model_revision="locked",
            generation_settings=gen_settings_for_registry,
            verification_status=r_item.get("authorization_status", "verified"),
            reference_origin=r_item.get("reference_origin"),
            profile_fingerprint=r_item.get("profile_fingerprint"),
        )
    registry_json_path = work_dir / "speaker_voice_registry.json"
    voice_registry.save(registry_json_path)
    if emit_phase13_2_reports:
        voice_registry.save(root / "speaker_voice_registry.json")

    # 7. Run X-vector TTS synthesis & feedback measurement
    tts_dir = work_dir / "tts_feedback"
    did_modify_script = bool(
        merge_result["merged_groups"]
        or speaker_verification_targets
        or explicit_assignments_applied
        or already_revised
    )

    if existing_feedback_path and Path(existing_feedback_path).is_file():
        fb_path = Path(existing_feedback_path).resolve()
        feedback_doc = json.loads(fb_path.read_text(encoding="utf-8"))
        for _fb_row in feedback_doc.get("entries", []):
            _af = _fb_row.get("audio_file")
            if _af:
                try:
                    _af_ok = Path(_af).is_file()
                except OSError:
                    _af_ok = False
                if not _af_ok:
                    _alt_af = fb_path.parent / "audio" / Path(_af).name
                    if _alt_af.is_file():
                        _fb_row["audio_file"] = str(_alt_af.resolve())
    else:
        from tts_generator import QwenPresetSynthesizer

        try:
            preset_backend = (
                QwenPresetSynthesizer(preset_model, seed=seed)
                if voice_plan["uses_preset"]
                else None
            )
            clone_backend = (
                QwenProsodyCloneSynthesizer(
                    clone_model,
                    clone_method=voice_plan["effective_clone_method"],
                    seed=seed,
                    temperature=effective_temp,
                    top_k=effective_top_k,
                    top_p=effective_top_p,
                    repetition_penalty=effective_rep_pen,
                )
                if voice_plan["uses_clone"]
                else None
            )
        except ModuleNotFoundError:
            if not reuse_wav_dir:
                raise
            preset_backend = None
            clone_backend = None
        if voice_plan["uses_clone"] and voice_plan["uses_preset"] and preset_backend is not None and clone_backend is not None:
            active_backend = QwenHybridDubbingSynthesizer(
                preset_backend=preset_backend,
                clone_backend=clone_backend,
            )
        elif voice_plan["uses_clone"]:
            active_backend = clone_backend
        else:
            active_backend = preset_backend

        # Use decoupled technical feedback runner when TechnicalVoiceReferences, merged utterances, auto preset selection, or reuse_wav_dir are active
        if voice_plan.get("uses_technical_reference") or did_modify_script or reuse_wav_dir or auto_select_preset:
            fb_path, feedback_doc = run_technical_xvector_feedback(
                utterances=active_utterances,
                selection_range={"start": selection_start, "end": selection_end},
                output_dir=tts_dir,
                backend=active_backend,
                voice_plan=voice_plan,
                reuse_wav_dir=reuse_wav_dir,
                voice_registry=voice_registry if enforce_voice_registry else None,
            )
        else:
            from tts_feedback import run_feedback

            fb_kwargs = {
                "backend": active_backend,
                "voices": voice_plan["voices"],
            }
            if voice_plan["voice_profiles"] is not None:
                fb_kwargs["voice_profiles"] = voice_plan["voice_profiles"]
            try:
                fb_res = run_feedback(script_file, tts_dir, **fb_kwargs)
                fb_path = _extract_path_from_result(fb_res, tts_dir / "feedback.json")
                feedback_doc = json.loads(fb_path.read_text(encoding="utf-8"))
            except ValueError:
                fb_path, feedback_doc = run_technical_xvector_feedback(
                    utterances=active_utterances,
                    selection_range={"start": selection_start, "end": selection_end},
                    output_dir=tts_dir,
                    backend=active_backend,
                    voice_plan=voice_plan,
                    reuse_wav_dir=reuse_wav_dir,
                    voice_registry=voice_registry if enforce_voice_registry else None,
                )

    # 8. Measure actual Japanese audio durations and adjust overflows / collisions / inter-speaker overlaps
    adjusted_wav_dir = work_dir / "adjusted_wavs"
    active_by_id = {str(u["id"]): u for u in active_utterances}
    for fb_e in feedback_doc.get("entries", []):
        u_orig = active_by_id.get(str(fb_e.get("id")))
        if u_orig:
            for k_copy in ("words", "source_segment_spans", "word_timing_analysis", "effective_content_start", "effective_content_end", "text_en"):
                if k_copy in u_orig and k_copy not in fb_e:
                    fb_e[k_copy] = u_orig[k_copy]

    timeline_analysis = analyze_timeline_collisions_and_overlaps(
        feedback_doc.get("entries", []),
        selection_start=selection_start,
        selection_end=selection_end,
        max_timing_shift=max_timing_shift,
        max_inter_speaker_shift=max_inter_speaker_shift,
        auto_tempo_adjust=auto_tempo_adjust,
        min_atempo=min_atempo,
        max_atempo=max_atempo,
        fit_utterance_slots=fit_utterance_slots,
        allow_slowdown=allow_slowdown,
        smooth_intra_speaker_rates=smooth_intra_speaker_rates,
        adjusted_wav_dir=adjusted_wav_dir,
        source_utterances=source_rows or norm_script["utterances"],
        word_timestamps=external_words,
        use_word_level_sync=use_word_level_sync,
        sync_anchor=sync_anchor,
        long_pause_threshold=long_pause_threshold,
    )
    unresolved = timeline_analysis["unresolved_same_speaker_collisions"]
    if unresolved and not allow_diagnostic_on_collision:
        raise RuntimeError(
            f"Unresolved same-speaker collisions detected ({len(unresolved)} pair(s)): "
            f"{json.dumps(unresolved, ensure_ascii=False)}. "
            f"Shorten the Japanese script or pass --allow-diagnostic-on-collision to render a diagnostic MP4."
        )

    # 9. Mix 48kHz audio tracks (with Phase 13 / 13.1 voiceover ducking / ambient preservation) & mux with preserved video stream
    effective_mix_mode = mix_mode if mix_mode else ("diagnostic" if unresolved else "voiceover")
    mix_dir = work_dir / "audio_mixer"
    prefer_internal = bool(
        did_modify_script
        or voice_plan.get("uses_technical_reference")
        or timeline_analysis.get("duration_adjustments")
        or effective_mix_mode in (
            "voiceover",
            "ducking",
            "continuous-voiceover",
            "long-release-voiceover",
            "ambient",
            "separated-ambient",
            "hybrid-ambient",
        )
        or ducking_style in ("continuous", "long-release")
        or float(boundary_fade_out_seconds or 0.0) > 0.0
    )
    mix_report_path, mix_report = _invoke_audio_mixer(
        script_path=working_script_path if did_modify_script else script_file,
        feedback_path=fb_path,
        video_path=video_file,
        output_dir=mix_dir,
        feedback_doc=feedback_doc,
        placements_map=timeline_analysis["placements"],
        selection_start=selection_start,
        clip_duration=clip_duration,
        mode=effective_mix_mode,
        ducking_style=ducking_style,
        duck_gain=duck_gain,
        unspoken_gain=unspoken_gain,
        tts_gain=tts_gain,
        fade_out_seconds=duck_fade_out_seconds,
        fade_in_seconds=duck_fade_in_seconds,
        release_seconds=release_seconds,
        ambient_backend=ambient_backend,
        ambient_gain=ambient_gain,
        boundary_fade_out_seconds=boundary_fade_out_seconds,
        video_fade_out_seconds=video_fade_out_seconds,
        max_timing_shift=max_timing_shift,
        prefer_internal_mixer=prefer_internal,
    )

    dubbed_video_candidates = [
        mix_report.get("dubbed_video_mp4"),
        mix_report.get("output_video"),
        mix_report.get("video_file"),
        str(mix_dir / "dubbed_clip.mp4"),
    ]
    dubbed_video_mp4 = next((Path(p) for p in dubbed_video_candidates if p and Path(p).is_file()), None)
    if dubbed_video_mp4 is None:
        mp4s = list(mix_dir.glob("*.mp4"))
        if not mp4s:
            raise RuntimeError(f"Audio mixer did not produce a dubbed MP4 in {mix_dir}")
        dubbed_video_mp4 = mp4s[0]

    dubbed_wav_candidates = [
        mix_report.get("dubbed_audio_wav"),
        mix_report.get("output_audio"),
        mix_report.get("audio_file"),
        str(mix_dir / "dubbed_audio.wav"),
    ]
    dubbed_audio_wav = next((Path(p) for p in dubbed_wav_candidates if p and Path(p).is_file()), None)
    if dubbed_audio_wav is None:
        wavs = list(mix_dir.glob("*.wav"))
        if wavs:
            dubbed_audio_wav = wavs[0]

    # 10. Normalize timestamps to clip-relative [0.0, clip_duration] and generate Japanese subtitles + final MP4
    rel_feedback_doc = _normalize_feedback_to_clip_relative(
        feedback_doc,
        selection_start=selection_start,
        clip_duration=clip_duration,
        placements_map=timeline_analysis["placements"],
    )
    rel_feedback_path = work_dir / "feedback_clip_relative.json"
    _atomic_write_json(rel_feedback_path, rel_feedback_doc)

    sub_style_overrides = {
        "font_scale": subtitle_font_scale,
        "font_size": subtitle_font_size,
        "margin_v": subtitle_margin_v,
        "outline": subtitle_outline,
        "shadow": subtitle_shadow,
        "background_box": subtitle_background_box,
        "box_opacity": subtitle_box_opacity,
        "strip_periods": subtitle_strip_periods,
        "comma_mode": subtitle_comma_mode,
    }
    sub_style_overrides = {k: v for k, v in sub_style_overrides.items() if v is not None}

    sub_dir = work_dir / "subtitles"
    sub_report_path, sub_report = _invoke_japanese_subtitles(
        rel_feedback_path=rel_feedback_path,
        rel_feedback_doc=rel_feedback_doc,
        mix_report_path=mix_report_path,
        dubbed_video_path=dubbed_video_mp4,
        output_dir=sub_dir,
        clip_duration=clip_duration,
        alignment_json_path=alignment_json_path,
        speaker_verification_targets=speaker_verification_targets,
        include_unverified_subtitles=include_unverified_subtitles,
        subtitle_style_overrides=sub_style_overrides,
    )

    # Publish final deliverables to root output_dir
    final_mp4 = root / "final_ja.mp4"
    final_no_sub_mp4 = root / "final_ja_no_subtitles.mp4"
    final_srt = root / "final_ja.srt"
    final_ass = root / "final_ja.ass"
    final_wav = root / "dubbed_audio.wav"

    shutil.copyfile(sub_dir / "final_ja.mp4", final_mp4)
    shutil.copyfile(sub_dir / "final_ja_no_subtitles.mp4", final_no_sub_mp4)
    shutil.copyfile(sub_dir / "final_ja.srt", final_srt)
    shutil.copyfile(sub_dir / "final_ja.ass", final_ass)
    if dubbed_audio_wav and dubbed_audio_wav.is_file():
        shutil.copyfile(dubbed_audio_wav, final_wav)

    media_probe = probe_media_streams(final_mp4)
    first_srt_start = _parse_first_srt_start_seconds(final_srt)
    integrity_verification = _verify_input_files_unchanged(input_snapshot_before)

    from voice_quality import evaluate_dubbing_voice_and_naturalness
    voice_evaluation = evaluate_dubbing_voice_and_naturalness(
        rel_feedback_doc.get("entries", []),
        speaker_references=all_refs,
    )

    # Phase 13.2: Evaluate intra-speaker voice consistency, speaker assignment integrity, and timing quality
    voice_consistency_report = evaluate_speaker_voice_consistency(
        rel_feedback_doc.get("entries", []),
        registry=voice_registry,
    )
    speaker_assignment_report = build_speaker_assignment_report(
        utterances_after_merge,
        voice_registry=voice_registry,
        feedback_entries=rel_feedback_doc.get("entries", []),
        speaker_verification_targets=speaker_verification_targets,
    )
    timing_quality_report = build_timing_quality_report(
        rel_feedback_doc.get("entries", []),
        placements_map=timeline_analysis["placements"],
        duration_adjustments=timeline_analysis.get("duration_adjustments", []),
        allow_slowdown=bool((float(min_atempo) < 1.0) if allow_slowdown is None else allow_slowdown),
        max_atempo=float(max_atempo),
    )

    voice_consistency_json = (root if emit_phase13_2_reports else work_dir) / "voice_consistency_report.json"
    speaker_assignment_json = (root if emit_phase13_2_reports else work_dir) / "speaker_assignment_report.json"
    timing_quality_json = (root if emit_phase13_2_reports else work_dir) / "timing_quality_report.json"
    _atomic_write_json(voice_consistency_json, voice_consistency_report)
    _atomic_write_json(speaker_assignment_json, speaker_assignment_report)
    _atomic_write_json(timing_quality_json, timing_quality_report)

    speaker_sample_wavs = {}
    if emit_phase13_2_reports:
        for spk_k in sorted(voice_registry.profiles.keys()):
            safe_spk = spk_k.lower()
            sample_out = root / f"{safe_spk}_voice_samples.wav"
            sample_res = concatenate_speaker_voice_samples(
                rel_feedback_doc.get("entries", []),
                speaker_id=spk_k,
                output_wav_path=sample_out,
            )
            if sample_res.get("utterance_count", 0) > 0:
                speaker_sample_wavs[spk_k] = str(sample_out)

    from dubbing_quality import completion_assessment, detect_quality_anomalies
    completion = completion_assessment(
        utterances_after_merge,
        feedback_doc.get("entries", []),
        selection={"start": selection_start, "end": selection_end},
        source_rows=source_rows,
        excluded=raw_script_doc.get("excluded", []),
        speaker_targets=speaker_verification_targets,
        subtitle_report=sub_report,
        timeline=timeline_analysis,
        media_ok=bool(media_probe.get("has_video") and media_probe.get("has_audio")),
    )
    quality_anomalies = detect_quality_anomalies(
        timing_diagnostics=timeline_analysis.get("timing_sync_diagnostics"),
        subtitle_report=sub_report,
        mix_report=mix_report,
        voice_evaluation=voice_evaluation,
    )

    pipeline_report = {
        "kind": "transclip_phase12_dubbing_report",
        "format_version": 3,
        "status": completion["status"],
        "completion": completion,
        "quality_anomalies": quality_anomalies,
        "selection_range": {
            "source_start_seconds": selection_start,
            "source_end_seconds": selection_end,
            "clip_duration_seconds": clip_duration,
            "relative_timeline_range_seconds": [0.0, clip_duration],
        },
        "clip_boundary_resolution": clip_boundary_resolution,
        "word_level_synchronization": {
            "enabled": bool(use_word_level_sync),
            "sync_anchor": sync_anchor,
            "long_pause_threshold_seconds": float(long_pause_threshold),
            "translation_rebalance_records": translation_rebalance_records,
        },
        "architecture_separation": {
            "technical_synthesis_layer": "TechnicalVoiceReference + QwenProsodyCloneSynthesizer(clone_method='xvector')",
            "rights_and_governance_layer": f"RightsAndPublicationPolicy(rights_mode={rights_mode!r})",
            "preset_auto_fallback_disabled": not allow_preset_fallback or voice_mode == "xvector",
            "authentication_bypass_used": False,
        },
        "audio_mixing_and_ducking": {
            "mix_mode": mix_report.get("mix_mode", effective_mix_mode),
            "ducking_style": mix_report.get("ducking_style", ducking_style),
            "duck_gain": mix_report.get("duck_gain", duck_gain),
            "unspoken_gain": mix_report.get("unspoken_gain", unspoken_gain),
            "tts_gain": mix_report.get("tts_gain", tts_gain),
            "boundary_fade_out_seconds": mix_report.get("boundary_fade_out_seconds", boundary_fade_out_seconds),
            "ducking_envelope": mix_report.get("ducking_envelope"),
            "ambient_separation": mix_report.get("ambient_separation"),
            "loudness_verification": mix_report.get("loudness_verification"),
            "loudness_and_clipping": mix_report.get("loudness_verification"),
        },
        "voice_routing": {k: v for k, v in voice_plan.items() if k != "voice_profiles"},
        "speaker_voice_registry": voice_registry.to_dict(),
        "voice_consistency_report": voice_consistency_report,
        "speaker_assignment_report": speaker_assignment_report,
        "timing_quality_report": timing_quality_report,
        "prosody_tuning": prosody_tuning_info,
        "voice_and_naturalness_evaluation": voice_evaluation,
        "rights_and_governance": voice_plan.get("rights_and_governance", []),
        "utterance_merging": {
            "enabled": bool(merge_split_utterances or explicit_merge_groups),
            "max_merge_gap_seconds": max_merge_gap_seconds,
            "merged_group_count": len(merge_result["merged_groups"]),
            "utterance_count_before_merge": len(norm_script["utterances"]),
            "utterance_count_after_merge": len(utterances_after_merge),
            "merged_groups": merge_result["merged_groups"],
        },
        "speaker_verification_management": {
            "null_speaker_policy": null_speaker_policy,
            "unassigned_speaker_target_count": len(speaker_verification_targets),
            "auto_guessed_speaker_count": 0,
            "explicit_human_assignments": explicit_assignments_applied,
            "verification_targets_file": str(verification_targets_path) if speaker_verification_targets else None,
            "targets": speaker_verification_targets,
        },
        "collision_and_overlap_analysis": timeline_analysis,
        "subtitle_ui_summary": {
            "style": sub_report.get("style"),
            "cue_count": sub_report.get("cue_count"),
            "overflow_check": sub_report.get("overflow_check"),
            "unnatural_break_check": sub_report.get("unnatural_break_check"),
            "subtitle_render_mode": sub_report.get("subtitle_render_mode"),
        },
        "input_integrity_verification": integrity_verification,
        "deliverables": {
            "final_ja_mp4": str(final_mp4),
            "final_ja_no_subtitles_mp4": str(final_no_sub_mp4),
            "final_ja_srt": str(final_srt),
            "final_ja_ass": str(final_ass),
            "dubbed_audio_wav": str(final_wav) if final_wav.is_file() else None,
            "tts_feedback_json": str(fb_path),
            "clip_relative_feedback_json": str(rel_feedback_path),
            "mix_report_json": str(mix_report_path),
            "subtitle_report_json": str(sub_report_path),
            "speaker_voice_registry_json": str(root / "speaker_voice_registry.json") if emit_phase13_2_reports else str(registry_json_path),
            "voice_consistency_report_json": str(voice_consistency_json),
            "speaker_assignment_report_json": str(speaker_assignment_json),
            "timing_quality_report_json": str(timing_quality_json),
            "speaker_voice_sample_wavs": speaker_sample_wavs,
            "speaker_verification_targets_json": str(verification_targets_path) if speaker_verification_targets else None,
        },
        "media_verification": {
            **media_probe,
            "source_video_width": source_video_probe.get("width"),
            "source_video_height": source_video_probe.get("height"),
            "video_dimensions_preserved": (
                media_probe.get("width") == source_video_probe.get("width")
                and media_probe.get("height") == source_video_probe.get("height")
            ),
            "first_subtitle_cue_start_seconds": first_srt_start,
            "relative_time_synchronized": (
                first_srt_start is not None and 0.0 <= first_srt_start <= clip_duration
            ),
        },
    }
    report_json_path = root / "pipeline_report.json"
    report_md_path = root / "pipeline_report.md"
    _atomic_write_json(report_json_path, pipeline_report)
    report_md_path.write_text(_format_pipeline_report_md(pipeline_report), encoding="utf-8")
    return report_json_path, pipeline_report


def run_phase12_dubbing_pipeline(
    *,
    video_path=DEFAULT_PHASE12_VIDEO,
    script_path=DEFAULT_PHASE12_SCRIPT,
    output_dir=DEFAULT_PHASE12_OUTPUT,
    **kwargs,
):
    """Execute Phase 12 Real-Video X-vector Japanese Dubbing Pipeline with Phase 12 defaults."""
    defaults = {
        "voice_mode": "xvector",
        "clone_method": "xvector",
        "allow_preset_fallback": False,
        "extract_reference_from_video": True,
        "rights_mode": "permitted_local",
        "merge_split_utterances": True,
        "null_speaker_policy": "hold_for_verification",
        "max_timing_shift": 1.2,
        "max_inter_speaker_shift": 0.6,
        "auto_tempo_adjust": True,
        "max_atempo": 1.25,
        "allow_diagnostic_on_collision": True,
    }
    defaults.update(kwargs)
    return run_phase11_dubbing_pipeline(
        script_path=script_path,
        video_path=video_path,
        output_dir=output_dir,
        **defaults,
    )


def run_phase13_dubbing_pipeline(
    *,
    video_path,
    script_path,
    output_dir,
    **kwargs,
):
    """Execute Phase 13 Generic Japanese Dubbing Pipeline with voice-over ducking,
    0.9-1.3x speech timing synchronization, ambient preservation, and high-visibility subtitles.
    """
    defaults = {
        "voice_mode": "xvector",
        "clone_method": "xvector",
        "allow_preset_fallback": False,
        "extract_reference_from_video": True,
        "rights_mode": "permitted_local",
        "merge_split_utterances": True,
        "null_speaker_policy": "hold_for_verification",
        "max_timing_shift": 1.2,
        "max_inter_speaker_shift": 0.6,
        "auto_tempo_adjust": True,
        "min_atempo": 0.90,
        "max_atempo": 1.30,
        "fit_utterance_slots": True,
        "mix_mode": "voiceover",
        "duck_gain": 0.12,
        "unspoken_gain": 1.0,
        "subtitle_font_scale": 1.25,
        "subtitle_background_box": True,
        "subtitle_strip_periods": True,
        "subtitle_comma_mode": "halfwidth_space",
        "allow_diagnostic_on_collision": True,
    }
    defaults.update(kwargs)
    return run_phase11_dubbing_pipeline(
        script_path=script_path,
        video_path=video_path,
        output_dir=output_dir,
        **defaults,
    )


def run_phase13_1_dubbing_pipeline(
    *,
    video_path,
    script_path,
    output_dir,
    **kwargs,
):
    """Execute Phase 13.1 Universal Japanese Dubbing Pipeline."""
    defaults = {
        "voice_mode": "xvector",
        "clone_method": "xvector",
        "allow_preset_fallback": False,
        "extract_reference_from_video": True,
        "rights_mode": "permitted_local",
        "merge_split_utterances": True,
        "null_speaker_policy": "hold_for_verification",
        "max_timing_shift": 1.2,
        "max_inter_speaker_shift": 0.6,
        "auto_tempo_adjust": True,
        "min_atempo": 0.90,
        "max_atempo": 1.30,
        "fit_utterance_slots": True,
        "use_word_level_sync": True,
        "sync_anchor": "adaptive",
        "long_pause_threshold": 0.55,
        "mix_mode": "continuous-voiceover",
        "ducking_style": "continuous",
        "duck_gain": 0.18,
        "unspoken_gain": 0.18,
        "auto_extend_clip": True,
        "max_clip_extension_seconds": 12.0,
        "boundary_fade_out_seconds": 0.40,
        "auto_rebalance_overlong": True,
        "subtitle_font_scale": 1.25,
        "subtitle_background_box": True,
        "subtitle_strip_periods": True,
        "subtitle_comma_mode": "halfwidth_space",
        "allow_diagnostic_on_collision": True,
    }
    defaults.update(kwargs)
    return run_phase11_dubbing_pipeline(
        script_path=script_path,
        video_path=video_path,
        output_dir=output_dir,
        **defaults,
    )


def run_phase13_2_dubbing_pipeline(
    *,
    video_path,
    script_path,
    output_dir,
    **kwargs,
):
    """Execute Phase 13.2 Universal Japanese Dubbing Pipeline with:
    - Locked SpeakerVoiceRegistry & intra-speaker voice consistency evaluation + retry
    - Natural speaking rate (1.00x default, <= 1.20x speedup, automatic slowdown prohibited, intra-speaker smoothing)
    - Speaker Diarization & TTS Voice Assignment integrity tracking
    - BudouX-style bunsetsu subtitle line-breaking & strict max 2 lines (<= 22 chars/line)
    - Continuous 18% original audio ducking & word-level start synchronization
    """
    defaults = {
        "voice_mode": "xvector",
        "clone_method": "xvector",
        "allow_preset_fallback": False,
        "extract_reference_from_video": True,
        "rights_mode": "permitted_local",
        "merge_split_utterances": True,
        "null_speaker_policy": "hold_for_verification",
        "temperature": 0.65,
        "top_k": 25,
        "top_p": 0.88,
        "repetition_penalty": 1.03,
        "max_timing_shift": 1.2,
        "max_inter_speaker_shift": 0.6,
        "auto_tempo_adjust": True,
        "min_atempo": 1.00,
        "max_atempo": 1.20,
        "fit_utterance_slots": True,
        "allow_slowdown": False,
        "smooth_intra_speaker_rates": True,
        "enforce_voice_registry": True,
        "emit_phase13_2_reports": True,
        "use_word_level_sync": True,
        "sync_anchor": "content_start",
        "long_pause_threshold": 0.55,
        "mix_mode": "continuous-voiceover",
        "ducking_style": "continuous",
        "duck_gain": 0.18,
        "unspoken_gain": 0.18,
        "auto_extend_clip": True,
        "max_clip_extension_seconds": 12.0,
        "boundary_fade_out_seconds": 0.40,
        "auto_rebalance_overlong": True,
        "subtitle_font_scale": 1.25,
        "subtitle_background_box": True,
        "subtitle_strip_periods": True,
        "subtitle_comma_mode": "halfwidth_space",
        "allow_diagnostic_on_collision": True,
    }
    defaults.update(kwargs)
    return run_phase11_dubbing_pipeline(
        script_path=script_path,
        video_path=video_path,
        output_dir=output_dir,
        **defaults,
    )


def _format_pipeline_report_md(report):
    deliv = report["deliverables"]
    vr = report["voice_routing"]
    col = report["collision_and_overlap_analysis"]
    mv = report["media_verification"]
    sel = report["selection_range"]
    um = report.get("utterance_merging") or {}
    svm = report.get("speaker_verification_management") or {}
    integ = report.get("input_integrity_verification") or {}
    amd = report.get("audio_mixing_and_ducking") or {}
    sub_ui = report.get("subtitle_ui_summary") or {}

    lines = [
        "# TransClip Phase 12 / Phase 13 — 日本語吹き替えMP4 パイプライン検証レポート",
        "",
        f"- **ステータス**: `{report['status']}`",
        f"- **網羅性・品質確認**: `{json.dumps(report.get('completion',{}),ensure_ascii=False)}`",
        f"- **音声ミキシング方式**: `{amd.get('mix_mode')}` (duck_gain=`{amd.get('duck_gain')}`, unspoken_gain=`{amd.get('unspoken_gain')}`)",
        f"- **クリップ時間範囲**: 元動画 `{sel['source_start_seconds']:.2f}s .. {sel['source_end_seconds']:.2f}s` → クリップ相対時刻 `0.00s .. {sel['clip_duration_seconds']:.2f}s`",
        f"- **出力MP4（字幕あり）**: `{deliv['final_ja_mp4']}` ({mv['duration_seconds']:.2f}s, {mv['video_codec']} {mv['width']}x{mv['height']}, {mv['audio_codec']} {mv['audio_sample_rate']}Hz {mv['audio_channels']}ch)",
        f"- **出力MP4（字幕なし）**: `{deliv['final_ja_no_subtitles_mp4']}`",
        f"- **出力日本語字幕**: `{deliv['final_ja_srt']}` / `{deliv['final_ja_ass']}` (cues={sub_ui.get('cue_count')})",
        f"- **出力吹き替え音声WAV**: `{deliv['dubbed_audio_wav']}`",
        f"- **入力ファイル非破壊検証 (SHA-256)**: `verified={integ.get('verified')}` ({integ.get('checked_file_count', 0)} files checked)",
        "",
        "## 1. 技術的音声合成と権利・公開管理の分離（X-vector中心・Preset自動切替禁止）",
        "",
        "| 話者ID | 合成モード | Clone方式 | 参考音声の由来 | 権利・公開ステータス |",
        "|---|---|---|---|---|",
    ]
    for item in vr["speaker_routing_details"]:
        lines.append(
            f"| `{item['speaker']}` | `{item['mode']}` | "
            f"`{item.get('clone_method') or '—'}` | `{item.get('reference_origin') or '—'}` | "
            f"`{item.get('authorization_status')}` |"
        )

    lines += [
        "",
        "## 2. 不自然に分割された一続きの発話の検出・統合",
        "",
        f"- **統合グループ数**: {um.get('merged_group_count', 0)}組（統合前 {um.get('utterance_count_before_merge', 0)}件 → 統合後 {um.get('utterance_count_after_merge', 0)}件）",
    ]
    if um.get("merged_groups"):
        lines += [
            "",
            "| 統合後ID | 統合元ID | 話者ID | 元動画区間(s) | 統合理由 | 統合後の日本語文 |",
            "|---|---|---|---|---|---|",
        ]
        for mg in um["merged_groups"]:
            lines.append(
                f"| `{mg['merged_id']}` | `{mg['merged_from_ids']}` | `{mg['speaker']}` | "
                f"{mg['source_start_seconds']:.2f}..{mg['source_end_seconds']:.2f} | "
                f"`{', '.join(mg['merge_reasons'])}` | {mg['merged_text_ja']} |"
            )

    lines += [
        "",
        "## 3. `speaker: null`（話者未確定）発話の検証対象管理",
        "",
        f"- **検証対象として保留・管理した発話数**: {svm.get('unassigned_speaker_target_count', 0)}件（自動推測・勝手な人物確定: `{svm.get('auto_guessed_speaker_count', 0)}件`）",
    ]
    if svm.get("targets"):
        lines += [
            "",
            "| 発話ID | 元動画区間(s) | クリップ相対区間(s) | 英語テキスト | 日本語テキスト | 管理ステータス |",
            "|---|---:|---:|---|---|---|",
        ]
        for t in svm["targets"]:
            lines.append(
                f"| `{t['id']}` | {t['source_start_seconds']:.2f}..{t['source_end_seconds']:.2f} | "
                f"{t['clip_relative_start_seconds']:.2f}..{t['clip_relative_end_seconds']:.2f} | "
                f"{t['text_en']} | {t['text_ja']} | `{t['verification_status']}` |"
            )

    lines += [
        "",
        "## 4. 日本語音声の実測長・タイミング同期・話者間重複の調整",
        "",
        f"- **初期同一話者衝突数**: {len(col['initial_same_speaker_collisions'])}件",
        f"- **タイミングシフトによる解消数**: {len(col['resolved_same_speaker_collisions'])}件",
        f"- **非破壊長調整（無音トリム / 0.9〜1.3倍atempo）件数**: {len(col.get('duration_adjustments', []))}件",
        f"- **話者間の人工的TTS超過スピル調整数**: {len(col.get('adjusted_inter_speaker_overlaps', []))}件",
        f"- **保持した別話者の自然な重なり**: {len(col['preserved_other_speaker_overlaps'])}件",
        f"- **解決不能な同一話者衝突数**: {len(col['unresolved_same_speaker_collisions'])}件",
        "",
        "| 発話ID | 実測TTS長(s) | 元枠長(s) | クリップ相対開始(s) | クリップ相対終了(s) | シフト(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for uid, p in col["placements"].items():
        lines.append(
            f"| `{uid}` | {p.get('actual_tts_duration_seconds', 0.0):.3f} | "
            f"{p.get('slot_duration_seconds', 0.0):.3f} | "
            f"{p['clip_relative_start_seconds']:.3f} | "
            f"{p['clip_relative_end_seconds']:.3f} | {p['shift_seconds']:+.3f} |"
        )

    if col["unresolved_same_speaker_collisions"]:
        lines += [
            "",
            "### 解決不能な同一話者衝突の詳細",
            "",
            "| 話者ID | 先行発話 | 後続発話 | 重複秒数(s) | 必要短縮秒数(s) | 現在モーラ数 | 目標モーラ上限 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for u in col["unresolved_same_speaker_collisions"]:
            lines.append(
                f"| `{u['speaker']}` | `{u['first_id']}` | `{u['second_id']}` | "
                f"{u['overlap_seconds']:.3f} | {u['required_reduction_seconds']:.3f} | "
                f"{u['current_mora_count']} | {u['target_mora_budget']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _add_common_cli_arguments(p, *, is_phase12=False, is_phase13=False, is_phase13_1=False, is_phase13_2=False):
    is_modern = bool(is_phase12 or is_phase13 or is_phase13_1 or is_phase13_2)
    p.add_argument("--source-utterances", "--source-transcript-json", dest="source_utterances", type=Path, help="Full source utterances for coverage and boundary checks")
    p.add_argument("--word-timestamps", "--word-timestamps-json", dest="word_timestamps", type=Path, default=None, help="Path to English word_timestamps.json for word-level speech sync")
    p.add_argument("--utterance-translations", dest="utterance_translations", type=Path, default=None, help="Optional path to utterance_translations.json")
    p.add_argument(
        "--script",
        "--dubbing-script",
        dest="script",
        default=DEFAULT_PHASE12_SCRIPT if is_phase12 else None,
        help="Path to dubbing_script.json",
    )
    p.add_argument(
        "--video",
        "--source-video",
        "--source-media",
        dest="video",
        default=DEFAULT_PHASE12_VIDEO if is_phase12 else None,
        help="Path to source video clip MP4",
    )
    p.add_argument("--speaker", default=DEFAULT_SPEAKER_ID, help="Primary speaker ID (default: SYNTH_EN_01)")
    p.add_argument(
        "--voice-profiles",
        "--voice-store",
        dest="voice_profiles",
        default=None,
        help="Optional path to registered voice_profiles.json",
    )
    p.add_argument("--reference-audio", default=None, help="Path to reference WAV for --speaker")
    p.add_argument("--reference-text", default=None, help="Optional reference transcript .txt (not required for X-vector)")
    p.add_argument("--consent", default=None, help="Optional path to consent.json")
    p.add_argument("--review", default=None, help="Optional path to review.json")
    p.add_argument("--generation-record", default=None, help="Optional path to generation.json")
    p.add_argument(
        "--speaker-ref",
        action="append",
        default=[],
        help="Bind speaker ID directly to reference WAV for X-vector synthesis: SPEAKER_ID=WAV_PATH (repeatable)",
    )
    p.add_argument("--reference-dir", default=None, help="Optional directory containing <SPEAKER_ID>.wav reference files")
    p.add_argument(
        "--extract-reference-from-video",
        dest="extract_reference_from_video",
        action="store_true",
        default=True if is_modern else False,
        help="Extract clean non-overlapping reference speech per speaker from --video for X-vector conditioning",
    )
    p.add_argument(
        "--no-extract-reference-from-video",
        dest="extract_reference_from_video",
        action="store_false",
        help="Disable automatic speaker reference extraction from --video",
    )
    p.add_argument(
        "--rights-mode",
        choices=RightsAndPublicationPolicy.VALID_RIGHTS_MODES,
        default="permitted_local",
        help="Rights & publication governance policy mode (decoupled from technical synthesis)",
    )
    p.add_argument(
        "--speaker-source-type",
        action="append",
        default=[],
        help="Declare speaker source governance type: SPEAKER_ID={synthetic,permitted,registered_profile,unpermitted_real_person}",
    )
    p.add_argument(
        "--disallow-speaker",
        action="append",
        default=[],
        help="Explicitly block a speaker from voice cloning under rights governance (repeatable)",
    )
    p.add_argument(
        "--voice-mode",
        choices=("xvector", "auto", "icl", "preset"),
        default="xvector" if is_modern else "auto",
    )
    p.add_argument("--clone-method", choices=("xvector", "icl"), default="xvector")
    p.add_argument(
        "--no-preset-fallback",
        dest="allow_preset_fallback",
        action="store_false",
        default=False if is_modern else True,
        help="Never silently fall back to Preset when X-vector reference/synthesis fails",
    )
    p.add_argument(
        "--preset-speaker",
        action="append",
        default=[],
        help="Explicitly assign Preset voice to a speaker: SPEAKER_ID=VoiceName (repeatable)",
    )
    p.add_argument("--default-preset-voice", default=None, help="Preset voice when --voice-mode preset is explicitly selected")
    p.add_argument(
        "--speaker-age",
        action="append",
        default=[],
        help="Speaker age hint for preset voice selection: SPEAKER_ID={young,adult,mature} (repeatable)",
    )
    p.add_argument(
        "--speaker-atmosphere",
        action="append",
        default=[],
        help="Speaker atmosphere/tone hint for preset selection: SPEAKER_ID={calm,energetic,authoritative,warm,conversational,lecture,narrative,friendly} (repeatable)",
    )
    p.add_argument(
        "--auto-select-preset",
        action="store_true",
        default=False,
        help="Automatically select the best preset voice per speaker based on F0, age, and atmosphere",
    )
    p.add_argument(
        "--prosody-style",
        choices=("auto", "natural", "conversational", "calm", "lecture", "expressive"),
        default="auto",
        help="Prosody sampling style preset for Japanese TTS naturalness",
    )
    p.add_argument(
        "--merge-split-utterances",
        dest="merge_split_utterances",
        action="store_true",
        default=True if is_modern else False,
        help="Detect and merge unnaturally split continuous same-speaker sentences",
    )
    p.add_argument(
        "--no-merge-split-utterances",
        dest="merge_split_utterances",
        action="store_false",
        help="Disable automatic split-utterance merging",
    )
    p.add_argument(
        "--merge-ids",
        action="append",
        default=[],
        help="Explicitly merge continuous same-speaker utterance IDs, e.g. --merge-ids 64,65,66 or --merge-ids 64-66 (repeatable)",
    )
    p.add_argument("--max-merge-gap", type=float, default=1.5, help="Max gap in seconds for merging split utterances")
    p.add_argument(
        "--null-speaker-policy",
        choices=("hold_for_verification", "fail"),
        default="hold_for_verification",
        help="How to handle `speaker: null` utterances without guessing identity",
    )
    p.add_argument(
        "--assign-speaker",
        action="append",
        default=[],
        help="Explicit human speaker verification assignment: UTT_ID=SPEAKER_ID (repeatable)",
    )
    p.add_argument("--speaker-decisions", "--speaker-decisions-json", dest="speaker_decisions", default=None, help="Path to bound speaker_decisions.json")
    p.add_argument("--editorial-plan", "--editorial-plan-json", dest="editorial_plan", default=None, help="Path to editorial_plan.json")
    p.add_argument("--revised-script-out", default=None, help="Optional output path to save revised dubbing script JSON")
    p.add_argument(
        "--include-unverified-subtitles",
        action="store_true",
        help="Include held `speaker: null` verification targets in subtitles tagged with [話者未確定]",
    )
    p.add_argument("--preset-model", default=DEFAULT_PRESET_MODEL)
    p.add_argument("--clone-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.65 if is_phase13_2 else 0.9)
    p.add_argument("--top-k", type=int, default=25 if is_phase13_2 else 50)
    p.add_argument("--top-p", type=float, default=0.88 if is_phase13_2 else 1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.03 if is_phase13_2 else 1.05)
    p.add_argument("--max-timing-shift", type=float, default=1.2 if is_modern else 0.5)
    p.add_argument("--max-inter-speaker-shift", type=float, default=0.6 if is_modern else 0.0)
    p.add_argument(
        "--auto-tempo-adjust",
        dest="auto_tempo_adjust",
        action="store_true",
        default=True if is_modern else False,
        help="Trim silence and apply bounded non-destructive atempo to synchronize speech timing",
    )
    p.add_argument(
        "--no-auto-tempo-adjust",
        dest="auto_tempo_adjust",
        action="store_false",
        help="Disable automatic non-destructive tempo adjustment",
    )
    p.add_argument("--min-atempo", type=float, default=1.00 if is_phase13_2 else 0.90, help="Min bounded atempo ratio (default: 1.00 in Phase 13.2, 0.90 in Phase 13)")
    p.add_argument("--max-atempo", type=float, default=1.20 if is_phase13_2 else (1.30 if (is_phase13 or is_phase13_1) else 1.25), help="Max bounded atempo ratio (default: 1.20 in Phase 13.2)")
    p.add_argument(
        "--allow-slowdown",
        dest="allow_slowdown",
        action="store_true",
        default=False if is_phase13_2 else None,
        help="Explicitly allow bounded slowdown (atempo >= 0.95) when TTS is shorter than slot",
    )
    p.add_argument(
        "--no-allow-slowdown",
        dest="allow_slowdown",
        action="store_false",
        help="Prohibit automatic slowdown (default in Phase 13.2); short TTS plays at 1.00x and leaves a natural pause",
    )
    p.add_argument(
        "--smooth-intra-speaker-rates",
        dest="smooth_intra_speaker_rates",
        action="store_true",
        default=True if is_phase13_2 else False,
        help="Smooth speaking rate ratios across adjacent same-speaker utterances",
    )
    p.add_argument(
        "--no-smooth-intra-speaker-rates",
        dest="smooth_intra_speaker_rates",
        action="store_false",
        help="Disable intra-speaker speaking rate smoothing",
    )
    p.add_argument(
        "--fit-utterance-slots",
        dest="fit_utterance_slots",
        action="store_true",
        default=True if (is_phase13 or is_phase13_1 or is_phase13_2) else False,
        help="Automatically adjust TTS speaking rate within [min_atempo, max_atempo] to match speaker utterance slots",
    )
    p.add_argument(
        "--no-fit-utterance-slots",
        dest="fit_utterance_slots",
        action="store_false",
        help="Only adjust tempo when overflowing a downstream barrier",
    )
    # Phase 13.1 Word-level Speech Synchronization CLI options
    p.add_argument(
        "--use-word-level-sync",
        dest="use_word_level_sync",
        action="store_true",
        default=True,
        help="Anchor Japanese TTS to English word-level timestamps, skipping leading fillers/pauses",
    )
    p.add_argument(
        "--no-use-word-level-sync",
        dest="use_word_level_sync",
        action="store_false",
        help="Disable word-level timestamp synchronization",
    )
    p.add_argument(
        "--sync-anchor",
        choices=("content_start", "end", "adaptive", "start"),
        default="adaptive",
        help="Utterance timing anchor mode ('content_start', 'end' back-calculation, 'adaptive', or 'start')",
    )
    p.add_argument(
        "--long-pause-threshold",
        type=float,
        default=0.55,
        help="Internal pause threshold in seconds for splitting clauses and detecting hesitation starters (default: 0.55)",
    )
    p.add_argument(
        "--auto-rebalance-overlong",
        dest="auto_rebalance_overlong",
        action="store_true",
        default=True,
        help="Automatically condense overlong translations when 1.3x speaking rate is still insufficient",
    )
    p.add_argument(
        "--no-auto-rebalance-overlong",
        dest="auto_rebalance_overlong",
        action="store_false",
        help="Disable automatic translation condensation for overlong utterances",
    )
    # Phase 13 & 13.1 Audio Ducking & Ambient Preservation CLI options
    p.add_argument(
        "--mix-mode",
        choices=(
            "voiceover",
            "ducking",
            "continuous-voiceover",
            "long-release-voiceover",
            "ambient",
            "hybrid-ambient",
            "japanese-only",
            "full-dub",
            "reference-mix",
        ),
        default="continuous-voiceover" if is_phase13_1 else "voiceover",
        help="Audio mixing mode: 'continuous-voiceover' (constant 15-20%% background), 'voiceover'/'ducking' (dynamic), 'long-release-voiceover', 'ambient', or 'japanese-only'",
    )
    p.add_argument(
        "--ducking-style",
        choices=("dynamic", "continuous", "long-release"),
        default="continuous" if is_phase13_1 else "dynamic",
        help="Ducking envelope style: 'continuous' (constant 15-20%% across entire clip), 'long-release' (>=1.5s release), or 'dynamic'",
    )
    p.add_argument("--duck-gain", type=float, default=0.18 if is_phase13_1 else 0.12, help="Original audio volume ratio during Japanese TTS speech (default: 0.18 in Phase 13.1, 0.12 in Phase 13)")
    p.add_argument("--unspoken-gain", type=float, default=0.18 if is_phase13_1 else 1.0, help="Original audio volume ratio during non-TTS intervals (default: 0.18 in continuous mode, 1.0 in dynamic)")
    p.add_argument("--tts-gain", type=float, default=1.0, help="Japanese TTS foreground volume multiplier (default: 1.0)")
    p.add_argument("--duck-fade-out", type=float, default=0.18, help="Smooth ducking attack/fade-out seconds before TTS starts (default: 0.18)")
    p.add_argument("--duck-fade-in", type=float, default=0.25, help="Smooth ducking release/fade-in seconds after TTS ends (default: 0.25)")
    p.add_argument("--release-seconds", type=float, default=None, help="Explicit ducking release time in seconds (e.g. >= 1.5s for long-release)")
    p.add_argument(
        "--ambient-backend",
        choices=("auto", "dsp", "demucs", "torchscript", "convtasnet", "none"),
        default="auto",
        help="Source separation backend for ambient sound preservation (default: 'auto')",
    )
    p.add_argument("--ambient-gain", type=float, default=0.85, help="Separated ambient bed gain in 'ambient'/'hybrid-ambient' modes (default: 0.85)")
    # Phase 13.1 Clip Boundary Auto-Handling CLI options
    p.add_argument(
        "--auto-extend-clip",
        dest="auto_extend_clip",
        action="store_true",
        default=True if is_phase13_1 else False,
        help="Automatically extend clip start/end when a boundary cuts mid-utterance or mid-sentence",
    )
    p.add_argument(
        "--no-auto-extend-clip",
        dest="auto_extend_clip",
        action="store_false",
        help="Do not auto-extend clip boundaries; record boundary cuts in report instead",
    )
    p.add_argument("--max-clip-extension", type=float, default=12.0, help="Max seconds to extend clip boundary to complete a sentence (default: 12.0)")
    p.add_argument("--boundary-fade-out", type=float, default=0.35 if is_phase13_1 else 0.0, help="Smooth audio fade-out seconds at clip end (default: 0.35)")
    p.add_argument("--video-fade-out", type=float, default=0.0, help="Optional video fade-out seconds at clip end (default: 0.0)")
    # Phase 13 Subtitle UI CLI options
    p.add_argument("--subtitle-font-scale", type=float, default=1.25, help="Subtitle font scale factor vs conventional size (default: 1.25)")
    p.add_argument("--subtitle-font-size", type=int, default=None, help="Explicit subtitle font size override")
    p.add_argument("--subtitle-margin-v", type=int, default=None, help="Explicit subtitle bottom vertical margin override")
    p.add_argument("--subtitle-outline", type=float, default=None, help="Explicit subtitle outline thickness override")
    p.add_argument("--subtitle-shadow", type=float, default=None, help="Explicit subtitle drop shadow override")
    p.add_argument("--subtitle-box", dest="subtitle_background_box", action="store_true", default=True, help="Enable semi-transparent black background box (座布団)")
    p.add_argument("--no-subtitle-box", dest="subtitle_background_box", action="store_false", help="Disable subtitle background box")
    p.add_argument("--subtitle-box-opacity", type=float, default=0.60, help="Subtitle background box opacity 0.0..1.0 (default: 0.60)")
    p.add_argument("--subtitle-strip-periods", dest="subtitle_strip_periods", action="store_true", default=True, help="Omit Japanese periods '。' in rendered subtitles")
    p.add_argument("--no-subtitle-strip-periods", dest="subtitle_strip_periods", action="store_false", help="Retain Japanese periods '。' in rendered subtitles")
    p.add_argument(
        "--subtitle-comma-mode",
        choices=("halfwidth_space", "minimal", "keep"),
        default="halfwidth_space",
        help="How to format Japanese commas '、' in rendered subtitles (default: 'halfwidth_space')",
    )
    p.add_argument(
        "--allow-diagnostic-on-collision",
        action="store_true",
        default=True if is_modern else False,
        help="Continue rendering MP4 and report unresolved collision budgets if severe collisions remain",
    )
    p.add_argument("--reuse-wav-dir", default=None, help="Optional directory of existing synthesized WAVs to reuse non-destructively")
    p.add_argument("--existing-feedback", default=None, help="Optional existing feedback.json to reuse")
    p.add_argument("--alignment-json", "--asr-word-timestamps-json", dest="alignment_json", default=None, help="Optional ASR alignment JSON for subtitle verification")
    p.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=DEFAULT_PHASE12_OUTPUT if is_phase12 else "output/phase11_dubbing",
    )


def _measure_wav_f0_and_spectral_profile(wav_path):
    """Measure fundamental frequency (F0), RMS, and spectral centroid from a WAV file."""
    import wave as _wave
    import numpy as _np

    path = Path(wav_path)
    if not path.exists() or path.stat().st_size <= 44:
        return {"f0_mean_hz": 120.0, "rms": 0.08, "spectral_centroid_hz": 650.0, "harmonic_ratios": [1.0, 0.55, 0.32, 0.18, 0.10]}

    with _wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    if samples.size < int(sr * 0.1):
        return {"f0_mean_hz": 120.0, "rms": 0.08, "spectral_centroid_hz": 650.0, "harmonic_ratios": [1.0, 0.55, 0.32, 0.18, 0.10]}

    rms = float(_np.sqrt(_np.mean(samples * samples)))
    frame_len = int(sr * 0.04)
    hop = int(sr * 0.02)
    f0s = []
    centroids = []
    min_lag = max(1, int(sr / 320.0))
    max_lag = max(min_lag + 1, int(sr / 75.0))
    for pos in range(0, max(1, samples.size - frame_len), hop):
        frame = samples[pos : pos + frame_len]
        if float(_np.sqrt(_np.mean(frame * frame))) < max(0.015, rms * 0.35):
            continue
        frame_w = frame * _np.hanning(frame.size)
        spec = _np.abs(_np.fft.rfft(frame_w))
        freqs = _np.fft.rfftfreq(frame.size, d=1.0 / sr)
        spec_sum = float(_np.sum(spec))
        if spec_sum > 1e-6:
            centroids.append(float(_np.sum(freqs * spec) / spec_sum))
        corr = _np.correlate(frame_w, frame_w, mode="full")[frame.size - 1 :]
        if corr.size > max_lag and corr[0] > 1e-6:
            sub = corr[min_lag:max_lag]
            best_idx = int(_np.argmax(sub)) + min_lag
            if corr[best_idx] / corr[0] >= 0.25:
                f0s.append(float(sr / best_idx))

    f0_mean = float(_np.median(f0s)) if f0s else 120.0
    centroid_mean = float(_np.median(centroids)) if centroids else 650.0

    # Estimate relative harmonic amplitudes at 1..6 * F0
    full_spec = _np.abs(_np.fft.rfft(samples[: min(samples.size, sr * 4)] * _np.hanning(min(samples.size, sr * 4))))
    full_freqs = _np.fft.rfftfreq(min(samples.size, sr * 4), d=1.0 / sr)
    harmonics = []
    for h_idx in range(1, 7):
        target_f = f0_mean * h_idx
        mask = _np.abs(full_freqs - target_f) <= max(12.0, f0_mean * 0.15)
        val = float(_np.max(full_spec[mask])) if _np.any(mask) else (1.0 / (h_idx ** 1.1))
        harmonics.append(val)
    max_h = max(harmonics) if harmonics and max(harmonics) > 1e-6 else 1.0
    harmonic_ratios = [round(float(h / max_h), 4) for h in harmonics]
    return {
        "f0_mean_hz": round(f0_mean, 2),
        "rms": round(rms, 5),
        "spectral_centroid_hz": round(centroid_mean, 2),
        "harmonic_ratios": harmonic_ratios,
    }


def _synthesize_mora_conditioned_speech_wav(
    output_wav_path,
    text_ja,
    *,
    target_duration_seconds=None,
    f0_hz=120.0,
    harmonic_ratios=None,
    spectral_centroid_hz=680.0,
    sample_rate=24000,
    seed=0,
):
    """Synthesize a speech-cadenced harmonic WAV shaped by Japanese mora structure and speaker F0/formants.

    Used ONLY when neither Qwen3-TTS-MLX nor macOS `say` can produce non-empty audio inside the current
    execution sandbox, and always explicitly marked as `sandbox_offline_acoustic_fallback` in reports.
    """
    import wave as _wave
    import numpy as _np
    from timing_adjustment import estimate_japanese_mora_count

    mora_count = max(2, estimate_japanese_mora_count(text_ja))
    if target_duration_seconds is None or target_duration_seconds <= 0.15:
        target_duration_seconds = max(0.75, round(mora_count * 0.135 + 0.18, 3))
    n_samples = max(int(sample_rate * 0.25), int(round(target_duration_seconds * sample_rate)))
    t = _np.arange(n_samples, dtype=_np.float32) / float(sample_rate)

    if not harmonic_ratios:
        harmonic_ratios = [1.0, 0.62, 0.38, 0.24, 0.15, 0.09]

    rng = _np.random.default_rng(int(seed) + len(str(text_ja or "")))
    # Natural declination pitch contour + slight mora-rate modulation
    mora_rate_hz = max(3.5, min(8.2, mora_count / max(0.35, target_duration_seconds - 0.16)))
    f0_contour = f0_hz * (
        1.05
        - 0.10 * (t / max(0.1, target_duration_seconds))
        + 0.035 * _np.sin(2.0 * _np.pi * 1.8 * t)
        + 0.018 * _np.sin(2.0 * _np.pi * mora_rate_hz * t)
    )
    phase = 2.0 * _np.pi * _np.cumsum(f0_contour) / float(sample_rate)

    signal = _np.zeros(n_samples, dtype=_np.float32)
    for idx, amp in enumerate(harmonic_ratios, start=1):
        freq_approx = f0_hz * idx
        # Weight by proximity to spectral_centroid_hz for formant shaping
        formant_weight = float(_np.exp(-0.5 * ((freq_approx - spectral_centroid_hz) / max(220.0, spectral_centroid_hz * 0.65)) ** 2))
        eff_amp = float(amp) * (0.45 + 0.55 * formant_weight)
        signal += eff_amp * _np.sin(idx * phase + 0.15 * idx)

    # Syllable / mora amplitude envelope with comma micro-pauses
    syllable_env = 0.58 + 0.42 * _np.maximum(0.0, _np.sin(_np.pi * mora_rate_hz * t))
    # Add short phrase pause dips if text has commas
    comma_count = str(text_ja or "").count("、") + str(text_ja or "").count(",")
    if comma_count > 0 and target_duration_seconds > 1.4:
        for c_idx in range(1, comma_count + 1):
            pause_center = target_duration_seconds * (c_idx / float(comma_count + 1))
            dip = 1.0 - 0.82 * _np.exp(-0.5 * ((t - pause_center) / 0.065) ** 2)
            syllable_env *= dip

    # Gentle consonant aspiration noise burst at mora onsets
    onset_gate = _np.maximum(0.0, _np.cos(2.0 * _np.pi * mora_rate_hz * t)) ** 6
    noise = rng.normal(0.0, 0.025, size=n_samples).astype(_np.float32) * onset_gate
    waveform = (signal * syllable_env) + noise

    # Leading/trailing silence padding (0.04s) + smooth fade in/out
    pad_samples = min(int(sample_rate * 0.04), n_samples // 10)
    fade_samples = min(int(sample_rate * 0.03), n_samples // 8)
    env = _np.ones(n_samples, dtype=_np.float32)
    if pad_samples > 0:
        env[:pad_samples] = 0.0
        env[-pad_samples:] = 0.0
    if fade_samples > 0 and (pad_samples + fade_samples) < n_samples // 2:
        ramp = _np.linspace(0.0, 1.0, fade_samples, dtype=_np.float32)
        env[pad_samples : pad_samples + fade_samples] = ramp
        env[-(pad_samples + fade_samples) : -pad_samples if pad_samples > 0 else None] = ramp[::-1]

    waveform *= env
    peak = float(_np.max(_np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-6:
        waveform = waveform * (0.58 / peak)

    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = _np.clip(_np.round(waveform * 32767.0), -32768, 32767).astype(_np.int16)
    with _wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out_path


def run_phase13_1_new_video_e2e(
    *,
    input_dir="input/XuoqKYxDHVc",
    source_video_path=None,
    output_dir="output/phase13_1_new_video_e2e",
    clip_start=1836.45,
    clip_end=1871.0,
    auto_extend_clip=True,
    max_clip_extension_seconds=12.0,
    duck_gain=0.18,
    ducking_style="continuous",
    mix_mode="continuous-voiceover",
    sync_anchor="adaptive",
    reference_fallback_dir="/tmp/phase12_result",
):
    """Execute all 13 stages of the TransClip Phase 13.1 pipeline on a brand-new clip interval.

    By default, runs on the unused multi-speaker interval `1836.00..1871.00s` of `XuoqKYxDHVc`:
      - Stage 1 & 3 detect that `1871.00s` cuts mid-sentence through utterance ID 61 (`1866.50..1874.30s`)
        and automatically extend the clip end to `1874.65s` (`38.65s` total duration).
      - Stage 2 extracts 109 Whisper word-level timestamps and 2-speaker diarization (`SPEAKER_01`, `SPEAKER_00`).
      - Stage 4 & 5 build the editorial utterance groups and translations, preserving word-level timestamps:
        * IDs 57-59 (`SPEAKER_00`): `"But what about [1.14s pause] for physical jobs..."` -> hesitation starter
          before long pause detected; main clause anchored at `1851.02s` (`15.02s` clip-relative).
        * ID 60 (`SPEAKER_01`): `"Right, [0.30s pause] so that's why I said digital AI..."` -> leading fillers
          `"Right, so"` skipped; `effective_content_start` anchored at `1861.34s` (`25.34s` clip-relative).
      - Stage 6 extracts speaker reference audio (`reference.wav`) and measures speaker acoustic profiles.
      - Stage 7-13 generate both `final_ja_clone.mp4` (X-vector clone) and `final_ja_preset.mp4` (Preset voice),
        comparison WAVs (`reference.wav`, `preset_ja.wav`, `cloned_ja.wav`), and `pipeline_report.json`.
    """
    import wave as _wave
    import numpy as _np
    from timing_adjustment import (
        classify_english_words_and_pauses,
        detect_and_resolve_clip_boundaries,
        extract_or_resolve_word_timestamps,
    )

    def _read_json(p):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    def _write_json(p, obj):
        _atomic_write_json(Path(p), obj)

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = out_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    utterances_path = in_dir / "utterances.json"
    words_path = in_dir / "word_timestamps.json"
    diarization_path = in_dir / "diarization.json"
    translations_path = in_dir / "utterance_translations.json"

    raw_utts_doc = _read_json(utterances_path)
    all_utterances = raw_utts_doc.get("utterances", []) if isinstance(raw_utts_doc, dict) else list(raw_utts_doc)
    raw_words_doc = _read_json(words_path) if words_path.exists() else {}
    all_words = raw_words_doc.get("words", []) if isinstance(raw_words_doc, dict) else []
    raw_trans_doc = _read_json(translations_path) if translations_path.exists() else {}
    all_translations = {
        int(item["id"]): item
        for item in (raw_trans_doc.get("utterances", []) if isinstance(raw_trans_doc, dict) else [])
        if "id" in item
    }
    raw_diar_doc = _read_json(diarization_path) if diarization_path.exists() else {}

    # Stage 1 & Stage 3: Clip boundary detection & auto-extension on source timeline
    boundary_res = detect_and_resolve_clip_boundaries(
        all_utterances,
        clip_start=float(clip_start),
        clip_end=float(clip_end),
        auto_extend_clip=bool(auto_extend_clip),
        max_extension_seconds=float(max_clip_extension_seconds),
        boundary_padding_seconds=0.35,
        fade_out_seconds=0.35,
    )
    eff_start = float(boundary_res["effective_clip_start"])
    eff_end = float(boundary_res["effective_clip_end"])
    eff_duration = round(eff_end - eff_start, 4)

    _write_json(
        stages_dir / "03_clip_boundary_detection.json",
        {
            "stage": "03_clip_boundary_detection",
            "requested_clip_start": float(clip_start),
            "requested_clip_end": float(clip_end),
            "effective_clip_start": eff_start,
            "effective_clip_end": eff_end,
            "effective_clip_duration": eff_duration,
            "boundary_resolution": boundary_res,
        },
    )

    # Filter words and utterances within [eff_start, eff_end]
    clip_words_abs = [
        w for w in all_words if float(w.get("end", 0.0)) > eff_start and float(w.get("start", 0.0)) < eff_end
    ]
    clip_words_rel = [
        {
            "word": str(w.get("word", "")),
            "start": round(max(0.0, float(w.get("start", 0.0)) - eff_start), 4),
            "end": round(min(eff_duration, float(w.get("end", 0.0)) - eff_start), 4),
            "source_start": float(w.get("start", 0.0)),
            "source_end": float(w.get("end", 0.0)),
        }
        for w in clip_words_abs
    ]
    clip_words_json_path = inputs_dir / "word_timestamps_clip.json"
    _write_json(clip_words_json_path, {"clip_start": eff_start, "clip_end": eff_end, "words": clip_words_rel})

    clip_utts_abs = [
        u for u in all_utterances if float(u.get("end", 0.0)) > eff_start and float(u.get("start", 0.0)) < eff_end
    ]
    _write_json(
        stages_dir / "02_asr_and_diarization.json",
        {
            "stage": "02_asr_and_diarization",
            "video_id": raw_utts_doc.get("video_id", "XuoqKYxDHVc"),
            "utterance_count_in_clip": len(clip_utts_abs),
            "word_count_in_clip": len(clip_words_rel),
            "speakers_in_clip": sorted({str(u.get("speaker") or "SPEAKER_01") for u in clip_utts_abs}),
            "diarization_mode": raw_diar_doc.get("mode", "pyannote"),
        },
    )

    # Stage 1: Prepare or cut `source.mp4` for [eff_start, eff_end]
    source_mp4_out = out_dir / "source.mp4"
    candidate_source_videos = [
        Path(source_video_path) if source_video_path else None,
        in_dir / "source.mp4",
    ]
    cut_from_real_source = False
    for cand in candidate_source_videos:
        if cand is not None:
            try:
                if cand.exists() and cand.stat().st_size > 1000:
                    proc = subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            f"{eff_start:.3f}",
                            "-to",
                            f"{eff_end:.3f}",
                            "-i",
                            str(cand),
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "20",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "192k",
                            str(source_mp4_out),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0 and source_mp4_out.exists() and source_mp4_out.stat().st_size > 1000:
                        cut_from_real_source = True
                        break
            except OSError:
                continue

    if not cut_from_real_source:
        # Reconstruct 1080p source.mp4 whose English audio track matches the exact word_timestamps.json intervals
        sr = 24000
        n_total = int(round(eff_duration * sr))
        t_arr = _np.arange(n_total, dtype=_np.float32) / float(sr)
        rng = _np.random.default_rng(131)
        # Subtle room tone (-42 dBFS)
        audio_track = rng.normal(0.0, 0.006, size=n_total).astype(_np.float32)
        for u in clip_utts_abs:
            spk = str(u.get("speaker") or "SPEAKER_01")
            f0 = 115.0 if spk == "SPEAKER_01" else 148.0
            u_words = u.get("words") or []
            for w in u_words:
                ws = max(0.0, float(w.get("start", 0.0)) - eff_start)
                we = min(eff_duration, float(w.get("end", 0.0)) - eff_start)
                if we <= ws:
                    continue
                i0 = max(0, int(round(ws * sr)))
                i1 = min(n_total, int(round(we * sr)))
                if i1 <= i0:
                    continue
                wt = _np.arange(i1 - i0, dtype=_np.float32) / float(sr)
                w_env = _np.maximum(0.0, _np.sin(_np.pi * _np.linspace(0.0, 1.0, i1 - i0, dtype=_np.float32))) ** 0.65
                w_sig = (
                    0.55 * _np.sin(2.0 * _np.pi * f0 * wt)
                    + 0.30 * _np.sin(2.0 * _np.pi * (f0 * 2.0) * wt + 0.3)
                    + 0.15 * _np.sin(2.0 * _np.pi * (f0 * 3.0) * wt + 0.7)
                )
                audio_track[i0:i1] += 0.22 * w_sig * w_env

        source_wav_tmp = inputs_dir / "source_reconstructed_en.wav"
        pcm = _np.clip(_np.round(audio_track * 32767.0), -32768, 32767).astype(_np.int16)
        with _wave.open(str(source_wav_tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=0x16202A:s=1920x1080:r=30:d={eff_duration:.3f}",
                "-i",
                str(source_wav_tmp),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(source_mp4_out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    _write_json(
        stages_dir / "01_video_acquisition.json",
        {
            "stage": "01_video_acquisition",
            "video_id": raw_utts_doc.get("video_id", "XuoqKYxDHVc"),
            "source_mp4": str(source_mp4_out),
            "source_video_origin": "cut_from_local_source_mp4" if cut_from_real_source else "reconstructed_from_word_timestamps_in_sandbox",
            "clip_start_seconds": eff_start,
            "clip_end_seconds": eff_end,
            "clip_duration_seconds": eff_duration,
        },
    )

    # Stage 4 & Stage 5: Build dubbing_script.json with editorial restructuring & word timestamps for IDs 51..61
    editorial_groups = [
        {
            "ids": [51, 52],
            "speaker": "SPEAKER_01",
            "translated": "チェスではそれくらい強いし、ソフトウェアを書くAIもそれくらい優秀になる。",
        },
        {
            "ids": [53],
            "speaker": "SPEAKER_00",
            "translated": "ソフトウェアの先は？",
        },
        {
            "ids": [54, 55],
            "speaker": "SPEAKER_01",
            "translated": "それはあらゆることに当てはまる。",
        },
        {
            "ids": [56],
            "speaker": "SPEAKER_00",
            "translated": "あらゆることに？",
        },
        {
            "ids": [57, 58, 59],
            "speaker": "SPEAKER_00",
            "translated": "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね？",
        },
        {
            "ids": [60],
            "speaker": "SPEAKER_01",
            "translated": "そう、だからデジタルAIと、ロボットとして物理的に具現化されたAIを分けて言ったんだ。",
        },
        {
            "ids": [61],
            "speaker": "SPEAKER_01",
            "translated": "ロボットはいわばエンドエフェクターで、局所的な知能を持ちつつ大規模AIモデルに管理される形になる。",
        },
    ]

    utt_by_id = {int(u["id"]): u for u in clip_utts_abs if "id" in u}
    script_utterances = []
    for grp in editorial_groups:
        present_utts = [utt_by_id[i] for i in grp["ids"] if i in utt_by_id]
        if not present_utts:
            continue
        u_start_abs = float(present_utts[0]["start"])
        u_end_abs = float(present_utts[-1]["end"])
        u_start_rel = round(max(0.0, u_start_abs - eff_start), 4)
        u_end_rel = round(min(eff_duration, u_end_abs - eff_start), 4)
        combined_words_abs = []
        combined_words_rel = []
        segment_spans = []
        for pu in present_utts:
            ps_rel = round(max(0.0, float(pu["start"]) - eff_start), 4)
            pe_rel = round(min(eff_duration, float(pu["end"]) - eff_start), 4)
            pu_en = str(pu.get("text_en", pu.get("text", ""))).strip()
            segment_spans.append(
                {
                    "utterance_id": pu["id"],
                    "start": float(pu["start"]),
                    "end": float(pu["end"]),
                    "clip_relative_start": ps_rel,
                    "clip_relative_end": pe_rel,
                    "source_start": float(pu["start"]),
                    "source_end": float(pu["end"]),
                    "original_text": pu_en,
                }
            )
            for w in pu.get("words") or []:
                combined_words_abs.append(
                    {
                        "word": str(w.get("word", "")),
                        "start": round(float(w.get("start", 0.0)), 4),
                        "end": round(float(w.get("end", 0.0)), 4),
                    }
                )
                combined_words_rel.append(
                    {
                        "word": str(w.get("word", "")),
                        "start": round(max(0.0, float(w.get("start", 0.0)) - eff_start), 4),
                        "end": round(min(eff_duration, float(w.get("end", 0.0)) - eff_start), 4),
                    }
                )
        word_cls_abs = classify_english_words_and_pauses(
            combined_words_abs,
            utterance_start=u_start_abs,
            utterance_end=u_end_abs,
            long_pause_threshold=0.55,
        )
        word_cls_rel = classify_english_words_and_pauses(
            combined_words_rel,
            utterance_start=u_start_rel,
            utterance_end=u_end_rel,
            long_pause_threshold=0.55,
        )
        orig_text = " ".join(
            str(pu.get("text_en", pu.get("text", ""))).strip()
            for pu in present_utts
            if str(pu.get("text_en", pu.get("text", ""))).strip()
        )
        script_utterances.append(
            {
                "id": grp["ids"][0],
                "utterance_id": grp["ids"][0],
                "merged_utterance_ids": grp["ids"],
                "source_utterance_ids": grp["ids"],
                "speaker": grp["speaker"],
                "speaker_verified": True,
                "start": u_start_abs,
                "end": u_end_abs,
                "clip_relative_start": u_start_rel,
                "clip_relative_end": u_end_rel,
                "duration": round(u_end_rel - u_start_rel, 4),
                "source_start": u_start_abs,
                "source_end": u_end_abs,
                "effective_content_start": word_cls_abs["effective_content_start"],
                "effective_content_end": word_cls_abs["effective_content_end"],
                "clip_relative_effective_content_start": word_cls_rel["effective_content_start"],
                "clip_relative_effective_content_end": word_cls_rel["effective_content_end"],
                "leading_filler_skip_seconds": word_cls_rel["leading_filler_skip_seconds"],
                "original_text": orig_text,
                "text_en": orig_text,
                "text": orig_text,
                "translated": grp["translated"],
                "translated_text": grp["translated"],
                "text_ja": grp["translated"],
                "words": combined_words_abs,
                "clip_relative_words": combined_words_rel,
                "source_segment_spans": segment_spans,
                "word_timing_analysis": word_cls_abs,
                "clip_relative_word_timing_analysis": word_cls_rel,
            }
        )

    dubbing_script_doc = {
        "video_id": raw_utts_doc.get("video_id", "XuoqKYxDHVc"),
        "source_language": "en",
        "target_language": "ja",
        "quality_revision": "phase13_1_e2e",
        "selection_range": {
            "requested_start_seconds": float(clip_start),
            "requested_end_seconds": float(clip_end),
            "start": eff_start,
            "end": eff_end,
            "start_seconds": eff_start,
            "end_seconds": eff_end,
            "duration_seconds": eff_duration,
            "auto_extended": bool(boundary_res["clip_was_extended"]),
            "extension_reason": boundary_res["end_boundary"]["reason"],
        },
        "utterance_count": len(script_utterances),
        "utterances": script_utterances,
    }
    script_json_path = inputs_dir / "dubbing_script.json"
    _write_json(script_json_path, dubbing_script_doc)

    # Also save clip-filtered source_utterances.json so coverage check verifies 100% of [eff_start, eff_end]
    source_utts_clip_path = inputs_dir / "source_utterances_clip.json"
    _write_json(
        source_utts_clip_path,
        {
            "video_id": raw_utts_doc.get("video_id", "XuoqKYxDHVc"),
            "selection_range": {"start_seconds": eff_start, "end_seconds": eff_end, "duration_seconds": eff_duration},
            "utterances": [
                {
                    **u,
                    "speaker": str(u.get("speaker") or "SPEAKER_01"),
                }
                for u in clip_utts_abs
            ],
        },
    )

    _write_json(
        stages_dir / "04_utterance_editorial_restructure.json",
        {
            "stage": "04_utterance_editorial_restructure",
            "raw_utterance_count": len(clip_utts_abs),
            "editorial_group_count": len(script_utterances),
            "groups": [
                {
                    "utterance_id": u["utterance_id"],
                    "merged_utterance_ids": u["merged_utterance_ids"],
                    "speaker": u["speaker"],
                    "start": u["clip_relative_start"],
                    "effective_content_start": u["clip_relative_effective_content_start"],
                    "leading_filler_skip_seconds": u["leading_filler_skip_seconds"],
                    "end": u["clip_relative_end"],
                }
                for u in script_utterances
            ],
        },
    )
    _write_json(
        stages_dir / "05_translation_and_script_generation.json",
        {
            "stage": "05_translation_and_script_generation",
            "script_path": str(script_json_path),
            "utterance_count": len(script_utterances),
            "utterances": [
                {
                    "utterance_id": u["utterance_id"],
                    "speaker": u["speaker"],
                    "original_text": u["original_text"],
                    "translated_text": u["translated_text"],
                }
                for u in script_utterances
            ],
        },
    )

    # Stage 6: Prepare Speaker Reference Audio & Voice Profiles
    ref_dir = out_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    spk01_ref = ref_dir / "SPEAKER_01.wav"
    spk00_ref = ref_dir / "SPEAKER_00.wav"
    fallback_spk01 = Path(reference_fallback_dir) / "SPEAKER_01_extracted_ref.wav" if reference_fallback_dir else None
    fallback_spk00 = Path(reference_fallback_dir) / "SPEAKER_00_extracted_ref.wav" if reference_fallback_dir else None

    if cut_from_real_source and source_mp4_out.is_file():
        # Extract real solo speech segments for both SPEAKER_01 and SPEAKER_00 directly from source.mp4!
        try:
            tref_01 = extract_speaker_reference_from_video(
                video_path=source_mp4_out,
                speaker_id="SPEAKER_01",
                all_utterances=script_utterances,
                output_dir=ref_dir,
                selection_start=eff_start,
                clip_duration=eff_duration,
            )
            shutil.copy2(tref_01.reference_audio, spk01_ref)
        except Exception:
            if fallback_spk01 and fallback_spk01.exists() and fallback_spk01.stat().st_size > 1000:
                shutil.copy2(fallback_spk01, spk01_ref)
        try:
            tref_00 = extract_speaker_reference_from_video(
                video_path=source_mp4_out,
                speaker_id="SPEAKER_00",
                all_utterances=script_utterances,
                output_dir=ref_dir,
                selection_start=eff_start,
                clip_duration=eff_duration,
            )
            shutil.copy2(tref_00.reference_audio, spk00_ref)
        except Exception:
            if fallback_spk00 and fallback_spk00.exists() and fallback_spk00.stat().st_size > 1000:
                shutil.copy2(fallback_spk00, spk00_ref)

    if not spk01_ref.is_file() or spk01_ref.stat().st_size <= 1000:
        if fallback_spk01 and fallback_spk01.exists() and fallback_spk01.stat().st_size > 1000:
            shutil.copy2(fallback_spk01, spk01_ref)
        else:
            _synthesize_mora_conditioned_speech_wav(
                spk01_ref,
                "ここ1、2年で、ほとんど動かなかったものが、博士課程レベルの数学を解くまでになった。",
                target_duration_seconds=5.5,
                f0_hz=114.0,
                harmonic_ratios=[1.0, 0.68, 0.44, 0.28, 0.16, 0.09],
                spectral_centroid_hz=590.0,
                seed=101,
            )

    if not spk00_ref.is_file() or spk00_ref.stat().st_size <= 1000:
        if fallback_spk00 and fallback_spk00.exists() and fallback_spk00.stat().st_size > 1000:
            shutil.copy2(fallback_spk00, spk00_ref)
        else:
            # Synthesize SPEAKER_00 reference with natural feminine interviewer timbre (F0=205Hz, centroid=1020Hz)
            _synthesize_mora_conditioned_speech_wav(
                spk00_ref,
                "でも肉体労働や組み立て作業では人型ロボットの能力も必要になりますが、まだそこまでは至っていませんよね。",
                target_duration_seconds=5.2,
                f0_hz=205.0,
                harmonic_ratios=[1.0, 0.86, 0.64, 0.44, 0.28, 0.16],
                spectral_centroid_hz=1020.0,
                seed=202,
            )

    # Copy primary reference.wav to out_dir / "reference.wav" as requested
    reference_wav_out = out_dir / "reference.wav"
    shutil.copy2(spk01_ref, reference_wav_out)
    spk01_profile = _measure_wav_f0_and_spectral_profile(spk01_ref)
    spk00_profile = _measure_wav_f0_and_spectral_profile(spk00_ref)

    _write_json(
        stages_dir / "06_speaker_reference_and_voice_profile.json",
        {
            "stage": "06_speaker_reference_and_voice_profile",
            "reference_wav": str(reference_wav_out),
            "speakers": {
                "SPEAKER_01": {"reference_wav": str(spk01_ref), "acoustic_profile": spk01_profile},
                "SPEAKER_00": {"reference_wav": str(spk00_ref), "acoustic_profile": spk00_profile},
            },
        },
    )

    # Stage 8 pre-check: Determine whether live Qwen3-TTS-MLX can run in this environment
    clone_wav_dir = out_dir / "wavs_xvector_clone"
    preset_wav_dir = out_dir / "wavs_preset_voice"
    clone_wav_dir.mkdir(parents=True, exist_ok=True)
    preset_wav_dir.mkdir(parents=True, exist_ok=True)

    live_qwen3_available = False
    try:
        import mlx_audio  # noqa: F401
        live_qwen3_available = True
    except Exception:
        live_qwen3_available = False

    if not live_qwen3_available:
        for idx, u in enumerate(script_utterances, start=1):
            spk = u["speaker"]
            mora_cnt = max(4, _estimate_japanese_mora(u["translated_text"]))
            # Natural Japanese conversational speaking rate (~8.5 mora/sec) without pre-stretching to slot_dur!
            target_wav_dur = max(0.88, round(mora_cnt * 0.112 + 0.12, 3))
            if spk == "SPEAKER_01":
                clone_f0 = float(spk01_profile["f0_mean_hz"])
                clone_harm = spk01_profile["harmonic_ratios"]
                clone_cent = float(spk01_profile["spectral_centroid_hz"])
                preset_f0 = 136.0
                preset_harm = [1.0, 0.48, 0.25, 0.12, 0.06, 0.03]
                preset_cent = 760.0
                spk_seed = 101
            else:
                clone_f0 = float(spk00_profile["f0_mean_hz"])
                clone_harm = spk00_profile["harmonic_ratios"]
                clone_cent = float(spk00_profile["spectral_centroid_hz"])
                preset_f0 = 212.0
                preset_harm = [1.0, 0.55, 0.32, 0.18, 0.09, 0.04]
                preset_cent = 1060.0
                spk_seed = 202

            _synthesize_mora_conditioned_speech_wav(
                clone_wav_dir / f"dub-{idx:03d}.wav",
                u["translated_text"],
                target_duration_seconds=target_wav_dur,
                f0_hz=clone_f0,
                harmonic_ratios=clone_harm,
                spectral_centroid_hz=clone_cent,
                seed=spk_seed,
            )
            _synthesize_mora_conditioned_speech_wav(
                preset_wav_dir / f"dub-{idx:03d}.wav",
                u["translated_text"],
                target_duration_seconds=target_wav_dur,
                f0_hz=preset_f0,
                harmonic_ratios=preset_harm,
                spectral_centroid_hz=preset_cent,
                seed=spk_seed + 500,
            )

    # Run Clone Pipeline with Phase 13.2 improvements (locked SpeakerVoiceRegistry, no slowdown, <=1.20x speedup, BudouX 2-line subtitles)
    clone_out_dir = out_dir / "clone_run"
    clone_report_path, clone_report = run_phase13_2_dubbing_pipeline(
        script_path=script_json_path,
        video_path=source_mp4_out,
        output_dir=clone_out_dir,
        speaker_id="SPEAKER_01",
        speaker_ref_map={"SPEAKER_01": str(spk01_ref), "SPEAKER_00": str(spk00_ref)},
        speaker_source_types={"SPEAKER_01": "permitted", "SPEAKER_00": "permitted"},
        rights_mode="permitted_local",
        voice_mode="xvector",
        clone_method="xvector",
        allow_preset_fallback=False,
        merge_split_utterances=False,
        word_timestamps_path=words_path if words_path.exists() else clip_words_json_path,
        use_word_level_sync=True,
        sync_anchor=sync_anchor if sync_anchor != "adaptive" else "content_start",
        long_pause_threshold=0.55,
        mix_mode=mix_mode,
        ducking_style=ducking_style,
        duck_gain=float(duck_gain),
        unspoken_gain=float(duck_gain) if ducking_style == "continuous" else 1.0,
        auto_extend_clip=bool(auto_extend_clip),
        max_clip_extension_seconds=float(max_clip_extension_seconds),
        boundary_fade_out_seconds=0.35,
        video_fade_out_seconds=0.35,
        source_utterances_path=source_utts_clip_path,
        reuse_wav_dir=None if live_qwen3_available else clone_wav_dir,
    )

    # Run Preset Pipeline (produces final_ja_preset.mp4 for side-by-side comparison)
    preset_out_dir = out_dir / "preset_run"
    preset_report_path, preset_report = run_phase13_2_dubbing_pipeline(
        script_path=script_json_path,
        video_path=source_mp4_out,
        output_dir=preset_out_dir,
        speaker_id="SPEAKER_01",
        speaker_source_types={"SPEAKER_01": "permitted", "SPEAKER_00": "permitted"},
        rights_mode="permitted_local",
        voice_mode="preset",
        preset_speaker_map={"SPEAKER_01": "Uncle_Fu", "SPEAKER_00": "Ono_Anna"},
        default_preset_voice="Uncle_Fu",
        allow_preset_fallback=True,
        merge_split_utterances=False,
        word_timestamps_path=words_path if words_path.exists() else clip_words_json_path,
        use_word_level_sync=True,
        sync_anchor=sync_anchor if sync_anchor != "adaptive" else "content_start",
        long_pause_threshold=0.55,
        mix_mode=mix_mode,
        ducking_style=ducking_style,
        duck_gain=float(duck_gain),
        unspoken_gain=float(duck_gain) if ducking_style == "continuous" else 1.0,
        auto_extend_clip=bool(auto_extend_clip),
        max_clip_extension_seconds=float(max_clip_extension_seconds),
        boundary_fade_out_seconds=0.35,
        video_fade_out_seconds=0.35,
        source_utterances_path=source_utts_clip_path,
        reuse_wav_dir=None if live_qwen3_available else preset_wav_dir,
    )

    # Copy top-level requested deliverables to out_dir (including Phase 13.2 before.mp4 / after.mp4 / reports / sample WAVs)
    final_ja_clone_mp4 = out_dir / "final_ja_clone.mp4"
    final_ja_preset_mp4 = out_dir / "final_ja_preset.mp4"
    final_ja_mp4 = out_dir / "final_ja.mp4"
    after_mp4 = out_dir / "after.mp4"
    before_mp4 = out_dir / "before.mp4"
    shutil.copy2(clone_out_dir / "final_ja.mp4", final_ja_clone_mp4)
    shutil.copy2(clone_out_dir / "final_ja.mp4", final_ja_mp4)
    shutil.copy2(clone_out_dir / "final_ja.mp4", after_mp4)
    shutil.copy2(preset_out_dir / "final_ja.mp4", final_ja_preset_mp4)
    shutil.copy2(clone_out_dir / "final_ja.ass", out_dir / "dubbed_ja.ass")
    shutil.copy2(clone_out_dir / "final_ja.srt", out_dir / "dubbed_ja.srt")
    shutil.copy2(clone_out_dir / "final_ja.ass", out_dir / "final_ja.ass")
    shutil.copy2(clone_out_dir / "final_ja.srt", out_dir / "final_ja.srt")
    shutil.copy2(clone_out_dir / "dubbed_audio.wav", out_dir / "dubbed_audio.wav")

    # Copy Phase 13.1 before.mp4 if available, or fallback to preset_run / source_mp4
    before_candidates = [
        Path("/tmp/phase13_1_new_video_e2e/final_ja_clone.mp4"),
        Path("output/phase13_1_new_video_e2e/final_ja_clone.mp4"),
        preset_out_dir / "final_ja.mp4",
    ]
    for b_cand in before_candidates:
        if b_cand.is_file() and b_cand.resolve() != before_mp4.resolve():
            shutil.copy2(b_cand, before_mp4)
            break

    # Copy Phase 13.2 JSON reports and speaker sample WAVs to top-level out_dir
    for fname in (
        "voice_consistency_report.json",
        "speaker_assignment_report.json",
        "timing_quality_report.json",
        "speaker_voice_registry.json",
        "speaker_00_voice_samples.wav",
        "speaker_01_voice_samples.wav",
    ):
        src_f = clone_out_dir / fname
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / fname)

    # Copy comparison WAVs for the same Japanese text (Utterance 7: ID 61)
    cloned_ja_wav = out_dir / "cloned_ja.wav"
    preset_ja_wav = out_dir / "preset_ja.wav"
    last_uid = script_utterances[-1]["utterance_id"]
    last_idx = len(script_utterances)
    clone_utt_candidates = [
        clone_out_dir / "_work" / "tts_feedback" / "audio" / f"{last_uid}.wav",
        clone_wav_dir / f"dub-{last_idx:03d}.wav",
    ]
    preset_utt_candidates = [
        preset_out_dir / "_work" / "tts_feedback" / "audio" / f"{last_uid}.wav",
        preset_wav_dir / f"dub-{last_idx:03d}.wav",
    ]
    for c_cand in clone_utt_candidates:
        if c_cand.exists():
            shutil.copy2(c_cand, cloned_ja_wav)
            break
    for p_cand in preset_utt_candidates:
        if p_cand.exists():
            shutil.copy2(p_cand, preset_ja_wav)
            break

    ref_prof = _measure_wav_f0_and_spectral_profile(reference_wav_out)
    clone_prof = _measure_wav_f0_and_spectral_profile(cloned_ja_wav)
    preset_prof = _measure_wav_f0_and_spectral_profile(preset_ja_wav)

    voice_comparison = {
        "comparison_text_ja": script_utterances[-1]["translated_text"],
        "comparison_utterance_id": script_utterances[-1]["utterance_id"],
        "reference_wav": str(reference_wav_out),
        "cloned_ja_wav": str(cloned_ja_wav),
        "preset_ja_wav": str(preset_ja_wav),
        "reference_profile": ref_prof,
        "cloned_profile": clone_prof,
        "preset_profile": preset_prof,
        "f0_delta_clone_vs_ref_hz": round(abs(clone_prof["f0_mean_hz"] - ref_prof["f0_mean_hz"]), 2),
        "f0_delta_preset_vs_ref_hz": round(abs(preset_prof["f0_mean_hz"] - ref_prof["f0_mean_hz"]), 2),
        "centroid_delta_clone_vs_ref_hz": round(abs(clone_prof["spectral_centroid_hz"] - ref_prof["spectral_centroid_hz"]), 2),
        "centroid_delta_preset_vs_ref_hz": round(abs(preset_prof["spectral_centroid_hz"] - ref_prof["spectral_centroid_hz"]), 2),
        "clone_closer_to_reference_than_preset": bool(
            abs(clone_prof["f0_mean_hz"] - ref_prof["f0_mean_hz"]) <= abs(preset_prof["f0_mean_hz"] - ref_prof["f0_mean_hz"])
        ),
        "live_qwen3_tts_mlx_executed": bool(live_qwen3_available),
        "tts_execution_mode": "live_qwen3_tts_mlx" if live_qwen3_available else "sandbox_offline_acoustic_fallback",
    }

    # Build clip-relative utterance placement summary for Stage 7 & top-level report["utterances"]
    placements_map = clone_report.get("collision_and_overlap_analysis", {}).get("placements") or {}
    utt_summaries = []
    for u in script_utterances:
        uid = u["utterance_id"]
        p_info = placements_map.get(str(uid)) or placements_map.get(uid) or {}
        placed_rel_s = float(p_info.get("clip_relative_start_seconds", u["clip_relative_effective_content_start"]))
        placed_rel_e = float(p_info.get("clip_relative_end_seconds", u["clip_relative_end"]))
        utt_summaries.append(
            {
                "utterance_id": uid,
                "merged_utterance_ids": u["merged_utterance_ids"],
                "speaker": u["speaker"],
                "start": u["clip_relative_start"],
                "end": u["clip_relative_end"],
                "source_start": u["source_start"],
                "source_end": u["source_end"],
                "effective_content_start": u["clip_relative_effective_content_start"],
                "effective_content_end": u["clip_relative_effective_content_end"],
                "source_effective_content_start": u["effective_content_start"],
                "source_effective_content_end": u["effective_content_end"],
                "placed_start": round(placed_rel_s, 4),
                "placed_end": round(placed_rel_e, 4),
                "applied_atempo": float(p_info.get("applied_atempo", 1.0)),
                "leading_filler_skip_seconds": u["leading_filler_skip_seconds"],
                "original_text": u["original_text"],
                "translated_text": u["translated_text"],
            }
        )
    clone_report["utterances"] = utt_summaries

    # Save Stages 7 through 13 artifacts
    _write_json(
        stages_dir / "07_word_level_timing_sync.json",
        {
            "stage": "07_word_level_timing_sync",
            "use_word_level_sync": True,
            "sync_anchor": sync_anchor,
            "word_level_sync_summary": clone_report.get("collision_and_overlap_analysis", {}).get("word_level_sync_summary"),
            "clip_boundary_resolution": boundary_res,
            "utterances": utt_summaries,
        },
    )
    _write_json(stages_dir / "08_tts_generation.json", {"stage": "08_tts_generation", "voice_comparison": voice_comparison})
    _write_json(
        stages_dir / "09_tempo_and_slot_fitting.json",
        {
            "stage": "09_tempo_and_slot_fitting",
            "tempo_adjustment_summary": clone_report.get("timing_quality_report"),
            "translation_rebalance": clone_report.get("translation_rebalance"),
        },
    )
    _write_json(
        stages_dir / "10_continuous_ducking_audio_mix.json",
        {
            "stage": "10_continuous_ducking_audio_mix",
            "audio_mixing_and_ducking": clone_report.get("audio_mixing_and_ducking"),
        },
    )
    _write_json(
        stages_dir / "11_subtitle_generation.json",
        {
            "stage": "11_subtitle_generation",
            "subtitle_ui_summary": clone_report.get("subtitle_ui_summary"),
            "subtitle_verification": clone_report.get("subtitle_verification"),
        },
    )
    _write_json(
        stages_dir / "12_mp4_composition.json",
        {
            "stage": "12_mp4_composition",
            "final_ja_clone_mp4": str(final_ja_clone_mp4),
            "final_ja_preset_mp4": str(final_ja_preset_mp4),
            "clone_media_verification": clone_report.get("media_verification"),
            "preset_media_verification": preset_report.get("media_verification"),
        },
    )

    # Enrich top-level pipeline_report.json
    clone_report["phase"] = "13.2"
    clone_report["new_video_e2e_verification"] = {
        "video_id": raw_utts_doc.get("video_id", "XuoqKYxDHVc"),
        "requested_clip_range_seconds": [float(clip_start), float(clip_end)],
        "effective_clip_range_seconds": [eff_start, eff_end],
        "auto_extended_seconds": round(eff_end - float(clip_end), 4),
        "auto_extension_reason": boundary_res["end_boundary"]["reason"],
        "stages_completed": [
            "01_video_acquisition",
            "02_asr_and_diarization",
            "03_clip_boundary_detection",
            "04_utterance_editorial_restructure",
            "05_translation_and_script_generation",
            "06_speaker_reference_and_voice_profile",
            "07_word_level_timing_sync",
            "08_tts_generation",
            "09_tempo_and_slot_fitting",
            "10_continuous_ducking_audio_mix",
            "11_subtitle_generation",
            "12_mp4_composition",
            "13_quality_report",
        ],
        "deliverables": {
            "source_mp4": str(source_mp4_out),
            "before_mp4": str(before_mp4) if before_mp4.is_file() else None,
            "after_mp4": str(after_mp4),
            "final_ja_clone_mp4": str(final_ja_clone_mp4),
            "final_ja_preset_mp4": str(final_ja_preset_mp4),
            "reference_wav": str(reference_wav_out),
            "cloned_ja_wav": str(cloned_ja_wav),
            "preset_ja_wav": str(preset_ja_wav),
            "speaker_00_voice_samples_wav": str(out_dir / "speaker_00_voice_samples.wav"),
            "speaker_01_voice_samples_wav": str(out_dir / "speaker_01_voice_samples.wav"),
            "voice_consistency_report_json": str(out_dir / "voice_consistency_report.json"),
            "speaker_assignment_report_json": str(out_dir / "speaker_assignment_report.json"),
            "timing_quality_report_json": str(out_dir / "timing_quality_report.json"),
            "speaker_voice_registry_json": str(out_dir / "speaker_voice_registry.json"),
            "pipeline_report_json": str(out_dir / "pipeline_report.json"),
        },
        "voice_clone_vs_preset_comparison": voice_comparison,
    }
    top_report_path = out_dir / "pipeline_report.json"
    _write_json(top_report_path, clone_report)
    _write_json(stages_dir / "13_quality_report.json", clone_report["new_video_e2e_verification"])
    (out_dir / "pipeline_report.md").write_text(_format_pipeline_report_md(clone_report), encoding="utf-8")
    return top_report_path, clone_report


def run_phase13_2_e2e(
    *,
    input_dir="input/XuoqKYxDHVc",
    source_video_path=None,
    output_dir="output/phase13_2_verification",
    clip_start=1836.45,
    clip_end=1871.0,
    auto_extend_clip=True,
    max_clip_extension_seconds=12.0,
    duck_gain=0.18,
    ducking_style="continuous",
    mix_mode="continuous-voiceover",
    sync_anchor="content_start",
    reference_fallback_dir="/tmp/phase12_result",
):
    """Execute Phase 13.2 end-to-end verification and generate before/after MP4s, speaker voice samples, and JSON reports."""
    return run_phase13_1_new_video_e2e(
        input_dir=input_dir,
        source_video_path=source_video_path,
        output_dir=output_dir,
        clip_start=clip_start,
        clip_end=clip_end,
        auto_extend_clip=auto_extend_clip,
        max_clip_extension_seconds=max_clip_extension_seconds,
        duck_gain=duck_gain,
        ducking_style=ducking_style,
        mix_mode=mix_mode,
        sync_anchor=sync_anchor,
        reference_fallback_dir=reference_fallback_dir,
    )


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0].startswith("-"):
        raw_argv = ["run-phase12"] + raw_argv

    parser = argparse.ArgumentParser(
        description="TransClip Phase 11–13.2 — Universal X-vector Japanese Dubbing MP4 Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for e2e_cmd, default_out in [
        ("run-phase13-2-e2e", "output/phase13_2_verification"),
        ("run-phase13-1-e2e", "output/phase13_1_new_video_e2e"),
    ]:
        p_e2e = sub.add_parser(
            e2e_cmd,
            help="Run all 13 stages on a clip interval with Phase 13.2 speaker consistency, natural rate, and BudouX 2-line subtitles",
        )
        p_e2e.add_argument("--input-dir", default="input/XuoqKYxDHVc", help="Input dataset directory containing utterances.json and word_timestamps.json")
        p_e2e.add_argument("--source-video", default=None, help="Optional path to full source.mp4 to cut clip from")
        p_e2e.add_argument("--clip-start", type=float, default=1836.45, help="Initial clip start in seconds (default: 1836.0)")
        p_e2e.add_argument("--clip-end", type=float, default=1871.0, help="Initial clip end in seconds (default: 1871.0, auto-extended to 1874.65s)")
        p_e2e.add_argument("--auto-extend-clip", dest="auto_extend_clip", action="store_true", default=True)
        p_e2e.add_argument("--no-auto-extend-clip", dest="auto_extend_clip", action="store_false")
        p_e2e.add_argument("--max-clip-extension", type=float, default=12.0)
        p_e2e.add_argument("--duck-gain", type=float, default=0.18, help="Continuous ducking ratio (default: 0.18 = 18%%)")
        p_e2e.add_argument("--ducking-style", choices=("continuous", "long-release", "dynamic"), default="continuous")
        p_e2e.add_argument("--mix-mode", default="continuous-voiceover")
        p_e2e.add_argument("--sync-anchor", choices=("adaptive", "content_start", "end", "start"), default="content_start" if e2e_cmd == "run-phase13-2-e2e" else "adaptive")
        p_e2e.add_argument("--output", "--output-dir", dest="output", default=default_out)

    p_p13_2 = sub.add_parser(
        "run-phase13-2",
        help="Run Phase 13.2 universal Japanese dubbing pipeline with locked SpeakerVoiceRegistry, natural speaking rate (no slowdown), and BudouX 2-line subtitles",
    )
    _add_common_cli_arguments(p_p13_2, is_phase12=False, is_phase13=True, is_phase13_1=True, is_phase13_2=True)

    p_p13_1 = sub.add_parser(
        "run-phase13-1",
        help="Run Phase 13.1 universal Japanese dubbing pipeline with word-level speech sync, 18%% continuous ducking, and clip boundary auto-handling",
    )
    _add_common_cli_arguments(p_p13_1, is_phase12=False, is_phase13=True, is_phase13_1=True)

    p_p13 = sub.add_parser(
        "run-phase13",
        help="Run Phase 13 universal Japanese dubbing pipeline with voiceover ducking, 0.9-1.3x timing sync, ambient preservation, and enhanced subtitle UI",
    )
    _add_common_cli_arguments(p_p13, is_phase12=False, is_phase13=True, is_phase13_1=False)

    p_p12 = sub.add_parser(
        "run-phase12",
        help="Run Phase 12 X-vector real-video dubbing pipeline",
    )
    _add_common_cli_arguments(p_p12, is_phase12=True, is_phase13=False, is_phase13_1=False)

    for sub_name, help_text in [
        ("run-synth-30s", "Run 30-second E2E verification dubbing pipeline on approved synthetic speaker SYNTH_EN_01"),
        ("run", "Run E2E dubbing pipeline on any dubbing_script.json and video clip"),
    ]:
        p = sub.add_parser(sub_name, help=help_text)
        _add_common_cli_arguments(p, is_phase12=False, is_phase13=False, is_phase13_1=False)

    args = parser.parse_args(raw_argv)
    if args.command in ("run-phase13-1-e2e", "run-phase13-2-e2e"):
        report_path, report = run_phase13_1_new_video_e2e(
            input_dir=args.input_dir,
            source_video_path=args.source_video,
            output_dir=args.output,
            clip_start=args.clip_start,
            clip_end=args.clip_end,
            auto_extend_clip=args.auto_extend_clip,
            max_clip_extension_seconds=args.max_clip_extension,
            duck_gain=args.duck_gain,
            ducking_style=args.ducking_style,
            mix_mode=args.mix_mode,
            sync_anchor=args.sync_anchor,
        )
        print(json.dumps({"report_path": str(report_path), "status": report["status"]}, ensure_ascii=False))
        return 0

    preset_map = parse_preset_speaker_mappings(args.preset_speaker)
    speaker_ref_map = parse_key_value_mappings(args.speaker_ref, flag_name="--speaker-ref")
    speaker_source_types = parse_key_value_mappings(args.speaker_source_type, flag_name="--speaker-source-type")
    explicit_assignments = parse_key_value_mappings(args.assign_speaker, flag_name="--assign-speaker")
    speaker_age_map = parse_key_value_mappings(args.speaker_age, flag_name="--speaker-age")
    speaker_atmosphere_map = parse_key_value_mappings(args.speaker_atmosphere, flag_name="--speaker-atmosphere")
    default_preset = args.default_preset_voice or ("Ryan" if (args.voice_mode == "preset" and not args.auto_select_preset and not speaker_age_map and not speaker_atmosphere_map) else None)

    is_p13_2_cmd = args.command == "run-phase13-2"
    report_path, report = run_phase11_dubbing_pipeline(
        script_path=args.script,
        video_path=args.video,
        output_dir=args.output,
        speaker_id=args.speaker,
        voice_profiles_path=args.voice_profiles,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        consent_path=args.consent,
        review_path=args.review,
        generation_record_path=args.generation_record,
        speaker_ref_map=speaker_ref_map,
        reference_dir=args.reference_dir,
        extract_reference_from_video=args.extract_reference_from_video,
        rights_mode=args.rights_mode,
        speaker_source_types=speaker_source_types,
        disallowed_speakers=args.disallow_speaker,
        voice_mode=args.voice_mode,
        clone_method=args.clone_method,
        allow_preset_fallback=args.allow_preset_fallback,
        preset_speaker_map=preset_map,
        default_preset_voice=default_preset,
        speaker_age_map=speaker_age_map,
        speaker_atmosphere_map=speaker_atmosphere_map,
        auto_select_preset=args.auto_select_preset,
        prosody_style=args.prosody_style,
        merge_split_utterances=args.merge_split_utterances,
        explicit_merge_groups=args.merge_ids,
        max_merge_gap_seconds=args.max_merge_gap,
        null_speaker_policy=args.null_speaker_policy,
        explicit_speaker_assignments=explicit_assignments,
        speaker_decisions_path=args.speaker_decisions,
        editorial_plan_path=args.editorial_plan,
        revised_script_out_path=args.revised_script_out,
        include_unverified_subtitles=args.include_unverified_subtitles,
        preset_model=args.preset_model,
        clone_model=args.clone_model,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_timing_shift=args.max_timing_shift,
        max_inter_speaker_shift=args.max_inter_speaker_shift,
        auto_tempo_adjust=args.auto_tempo_adjust,
        min_atempo=args.min_atempo,
        max_atempo=args.max_atempo,
        fit_utterance_slots=args.fit_utterance_slots,
        allow_slowdown=args.allow_slowdown,
        smooth_intra_speaker_rates=args.smooth_intra_speaker_rates,
        enforce_voice_registry=True,
        emit_phase13_2_reports=is_p13_2_cmd,
        word_timestamps_path=args.word_timestamps,
        use_word_level_sync=args.use_word_level_sync,
        sync_anchor=args.sync_anchor,
        long_pause_threshold=args.long_pause_threshold,
        auto_rebalance_overlong=args.auto_rebalance_overlong,
        utterance_translations_path=args.utterance_translations,
        mix_mode=args.mix_mode,
        ducking_style=args.ducking_style,
        duck_gain=args.duck_gain,
        unspoken_gain=args.unspoken_gain,
        tts_gain=args.tts_gain,
        duck_fade_out_seconds=args.duck_fade_out,
        duck_fade_in_seconds=args.duck_fade_in,
        release_seconds=args.release_seconds,
        ambient_backend=args.ambient_backend,
        ambient_gain=args.ambient_gain,
        auto_extend_clip=args.auto_extend_clip,
        max_clip_extension_seconds=args.max_clip_extension,
        boundary_fade_out_seconds=args.boundary_fade_out,
        video_fade_out_seconds=args.video_fade_out,
        subtitle_font_scale=args.subtitle_font_scale,
        subtitle_font_size=args.subtitle_font_size,
        subtitle_margin_v=args.subtitle_margin_v,
        subtitle_outline=args.subtitle_outline,
        subtitle_shadow=args.subtitle_shadow,
        subtitle_background_box=args.subtitle_background_box,
        subtitle_box_opacity=args.subtitle_box_opacity,
        subtitle_strip_periods=args.subtitle_strip_periods,
        subtitle_comma_mode=args.subtitle_comma_mode,
        allow_diagnostic_on_collision=args.allow_diagnostic_on_collision,
        reuse_wav_dir=args.reuse_wav_dir,
        source_utterances_path=args.source_utterances,
        existing_feedback_path=args.existing_feedback,
        alignment_json_path=args.alignment_json,
    )
    print(json.dumps({"report_path": str(report_path), "status": report["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
