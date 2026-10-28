"""Phase 14 — Universal Multi-Video Japanese Dubbing Support (`phase14_universal.py`).

Implements:
1. Universal project loader (`load_phase14_video_project`) reading `selection.json`,
   `utterances.json`, `utterance_translations.json`, `word_timestamps.json`,
   `dubbing_script.json`, and `output/<video_id>/clip_<start>_<end>.mp4` without
   any hardcoded dependency on `XuoqKYxDHVc`.
2. Safe handling of split utterances (`"I'm not telling"` + `"you."`) and unverified
   speakers (`UNKNOWN` / `null`):
   - Merges split utterances ONLY when both fragments belong to the same confirmed speaker.
   - Never changes `UNKNOWN` / `null` speakers into confirmed speakers.
   - Records all `UNKNOWN` / `null` intervals and blocked splits as `review_required`.
   - When using a provisional voice for verification, binds a distinct provisional
     speaker ID (`UNVERIFIED_PROVISIONAL`) and preset voice (`Serena`) clearly separated
     from confirmed speakers.
3. Semantic-preserving Japanese script adjustment (`adjust_overlong_script_entries`):
   - Generates shortened candidates for overlong entries while recording `semantic_review_required: True`
     because full human semantic fidelity cannot be mechanically guaranteed.
   - Validates surface preservation of numbers, proper nouns, negations, questions, and uncertainty.
   - Applies post-TTS `<= 1.20x` rate adjustment and reports any remaining overflows without truncation.
4. Strict separation between real `Qwen3-TTS-MLX` synthesis (default: fails with error if
   `Qwen3-TTS-MLX` cannot be loaded) and explicit test-only synthetic audio (`test_synthetic_tts=True`
   / `--test-synthetic-tts`).
5. Exact timeline reconciliation between source video selection range (e.g. `300..480s`) and
   pre-cut clip MP4 timeline (`0..180s`), ensuring audio mixing, subtitle timestamps, and
   clip end auto-extension never drift.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave
from typing import Any

import numpy as np

PHASE14_CONFIRMED_PRESET_POOL = (
    "Uncle_Fu",
    "Ono_Anna",
    "Ryan",
    "Aiden",
    "Vivian",
    "Dylan",
    "Eric",
    "Sohee",
)
PHASE14_PROVISIONAL_PRESET_VOICE = "Serena"
PHASE14_PROVISIONAL_SPEAKER_ID = "UNVERIFIED_PROVISIONAL"

_EN_INCOMPLETE_END_WORDS = {
    "telling", "tell", "ask", "asking", "give", "giving", "show", "showing",
    "talk", "talking", "say", "saying", "look", "looking", "wait", "waiting",
    "with", "for", "to", "from", "about", "into", "of", "in", "on", "at", "by",
    "and", "or", "but", "because", "since", "if", "when", "while", "that",
    "which", "who", "whom", "whose", "where", "how", "what", "whether",
    "the", "a", "an", "my", "your", "his", "her", "our", "their", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "must", "not",
}

_JA_SAFE_FILLER_PREFIXES = (
    r"^(?:えっと、|ええと、|あのう、|あの、|まあ、|その、|なんか、|実はですね、|つまりですね、|要するにですね、)+",
)

_JA_SAFE_CONCISE_REWRITES = (
    ("ということになります", "です"),
    ("ということになるのです", "になります"),
    ("ということなのです", "なのです"),
    ("ということです", "です"),
    ("というふうに思います", "と思います"),
    ("というふうに考えています", "と考えています"),
    ("というふうに言えます", "と言えます"),
    ("というような", "のような"),
    ("というふうな", "のような"),
    ("することができます", "できます"),
    ("することが可能になります", "可能になります"),
    ("することが可能です", "可能です"),
    ("することができるようになります", "できるようになります"),
    ("しなければなりません", "必要があります"),
    ("しなければならないのです", "必要があります"),
    ("ではないかと思います", "と思います"),
    ("ではないでしょうか", "でしょう"),
    ("についてお話ししたいと思います", "を話します"),
    ("について説明したいと思います", "を説明します"),
    ("について考えてみたいと思います", "を考えます"),
    ("させていただきたいと思います", "します"),
    ("させていただきます", "します"),
    ("ていく必要があります", "る必要があります"),
    ("ていくことができます", "ることができます"),
    ("非常に重要なポイントとして", "重要な点として"),
    ("非常に重要な", "重要な"),
    ("非常に大きな", "大きな"),
    ("非常に多くの", "多くの"),
    ("基本的に言えば、", "基本的に、"),
    ("具体的に言うと、", "具体的には、"),
    ("率直に言って、", "率直に、"),
    ("実際のところ、", "実際、"),
    ("言い換えれば、", "つまり、"),
    ("別の言い方をすれば", "つまり"),
    ("なぜなのかというと、", "なぜなら、"),
    ("どういうことかというと、", "つまり、"),
    ("私が思うには、", ""),
    ("私の考えでは、", ""),
    ("ご存知のように、", ""),
    ("言うまでもなく、", ""),
    ("であると言えます", "と言えます"),
    ("であると考えています", "と考えています"),
    ("であると思います", "と思います"),
    ("なのです。", "です。"),
    ("わけです。", "です。"),
)

_JA_NEGATION_PATTERNS = (
    "ない", "ません", "ず", "ぬ", "未", "非", "不", "無",
    "じゃない", "ではない", "ありません", "なかった", "ませんでした",
)
_JA_UNCERTAINTY_PATTERNS = (
    "かも", "かもしれ", "おそらく", "たぶん", "多分", "だろう", "でしょう",
    "思われ", "推測", "可能性", "みたい", "らしい",
)


class QwenModelUnavailableError(RuntimeError):
    """Raised when Qwen3-TTS-MLX is required (production default) and cannot be loaded."""


def is_unverified_speaker_label(speaker: Any) -> bool:
    if speaker is None:
        return True
    s = str(speaker).strip()
    if not s:
        return True
    return s.upper() in {"UNKNOWN", "NULL", "NONE", "UNASSIGNED", "N/A", "SPEAKER_UNKNOWN"}


def extract_semantic_tokens(text_en: str, text_ja: str) -> dict[str, Any]:
    en = str(text_en or "").strip()
    ja = str(text_ja or "").strip()
    numbers = sorted(set(re.findall(r"\d+(?:\.\d+)?%?", ja)))
    katakana_nouns = sorted({m for m in re.findall(r"[ァ-ヴー]{3,}", ja)})
    ascii_terms = sorted({m for m in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", ja)})
    has_negation = any(pat in ja for pat in _JA_NEGATION_PATTERNS)
    has_question = (
        ("？" in ja)
        or ("?" in ja)
        or ("?" in en)
        or bool(re.search(r"(?:ですか|ますか|でしょうか|のか|かな)[。！？!?]?$", ja))
    )
    has_uncertainty = any(pat in ja for pat in _JA_UNCERTAINTY_PATTERNS)
    return {
        "numbers": numbers,
        "katakana_nouns": katakana_nouns,
        "ascii_terms": ascii_terms,
        "has_negation": has_negation,
        "has_question": has_question,
        "has_uncertainty": has_uncertainty,
    }


def validate_shortened_candidate(original_ja: str, candidate_ja: str, *, text_en: str = "") -> dict[str, Any]:
    """Check surface-level preservation of numbers, proper nouns, negations, questions, and uncertainty.

    Note: Passing this surface check does not mechanically guarantee full semantic equivalence;
    any modified entry is still flagged `semantic_review_required: True` in the report.
    """
    orig = str(original_ja or "").strip()
    cand = str(candidate_ja or "").strip()
    if not cand:
        return {"valid": False, "reasons": ["empty_candidate"], "mechanically_guaranteed": False}
    orig_sem = extract_semantic_tokens(text_en, orig)
    cand_sem = extract_semantic_tokens(text_en, cand)
    reasons = []
    for num in orig_sem["numbers"]:
        if num not in cand:
            reasons.append(f"missing_number:{num}")
    for term in orig_sem["ascii_terms"]:
        if term not in cand:
            reasons.append(f"missing_ascii_proper_noun:{term}")
    for kata in orig_sem["katakana_nouns"]:
        if kata not in cand:
            reasons.append(f"missing_katakana_term:{kata}")
    if orig_sem["has_negation"] != cand_sem["has_negation"]:
        reasons.append("negation_polarity_changed")
    if orig_sem["has_question"] and not cand_sem["has_question"]:
        reasons.append("question_modality_lost")
    if orig_sem["has_uncertainty"] and not cand_sem["has_uncertainty"]:
        reasons.append("uncertainty_modality_lost")
    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "mechanically_guaranteed": False,
        "human_semantic_review_required": bool(orig != cand),
        "original_semantics": orig_sem,
        "candidate_semantics": cand_sem,
    }


def propose_semantic_preserving_shortening(
    text_ja: str,
    *,
    text_en: str = "",
    target_mora: int | None = None,
    estimate_mora_fn=None,
) -> dict[str, Any]:
    from dubbing_pipeline import _estimate_japanese_mora

    mora_fn = estimate_mora_fn or _estimate_japanese_mora
    original = str(text_ja or "").strip()
    orig_mora = mora_fn(original)
    if not original or (target_mora is not None and orig_mora <= target_mora):
        return {
            "original_text_ja": original,
            "shortened_text_ja": original,
            "applied": False,
            "original_mora": orig_mora,
            "shortened_mora": orig_mora,
            "mora_saved": 0,
            "applied_rules": [],
            "semantic_review_required": False,
            "semantic_validation": validate_shortened_candidate(original, original, text_en=text_en),
        }

    current = original
    applied_rules: list[str] = []

    for pat in _JA_SAFE_FILLER_PREFIXES:
        nxt = re.sub(pat, "", current).strip()
        if nxt and nxt != current:
            val = validate_shortened_candidate(original, nxt, text_en=text_en)
            if val["valid"]:
                current = nxt
                applied_rules.append("strip_leading_filler")

    for src_phrase, dst_phrase in _JA_SAFE_CONCISE_REWRITES:
        if target_mora is not None and mora_fn(current) <= target_mora:
            break
        if src_phrase in current:
            nxt = current.replace(src_phrase, dst_phrase).strip()
            nxt = re.sub(r"、+", "、", nxt).strip("、 ")
            if nxt and nxt != current:
                val = validate_shortened_candidate(original, nxt, text_en=text_en)
                if val["valid"]:
                    current = nxt
                    applied_rules.append(f"replace:{src_phrase}->{dst_phrase or '<omit>'}")

    final_val = validate_shortened_candidate(original, current, text_en=text_en)
    if not final_val["valid"]:
        current = original
        applied_rules = []
        final_val = validate_shortened_candidate(original, original, text_en=text_en)

    new_mora = mora_fn(current)
    applied = bool(current != original and new_mora < orig_mora)
    return {
        "original_text_ja": original,
        "shortened_text_ja": current,
        "applied": applied,
        "original_mora": orig_mora,
        "shortened_mora": new_mora,
        "mora_saved": max(0, orig_mora - new_mora),
        "applied_rules": applied_rules,
        "semantic_review_required": applied,
        "semantic_validation": final_val,
    }


def detect_unnatural_english_split_pair(
    curr: dict[str, Any],
    nxt: dict[str, Any],
    *,
    max_gap_seconds: float = 1.25,
) -> dict[str, Any]:
    en_a = str(curr.get("text_en") or curr.get("text") or "").strip()
    en_b = str(nxt.get("text_en") or nxt.get("text") or "").strip()
    start_a = float(curr.get("start", curr.get("source_start", 0.0)))
    end_a = float(curr.get("end", curr.get("source_end", start_a)))
    start_b = float(nxt.get("start", nxt.get("source_start", end_a)))
    gap = round(start_b - end_a, 4)

    spk_a_raw = curr.get("speaker") if curr.get("speaker") is not None else curr.get("raw_speaker")
    spk_b_raw = nxt.get("speaker") if nxt.get("speaker") is not None else nxt.get("raw_speaker")
    spk_a_unverified = is_unverified_speaker_label(spk_a_raw)
    spk_b_unverified = is_unverified_speaker_label(spk_b_raw)

    reasons: list[str] = []
    if -0.35 <= gap <= max_gap_seconds:
        words_a = re.findall(r"[A-Za-z']+", en_a)
        words_b = re.findall(r"[A-Za-z']+", en_b)
        if en_a and not re.search(r'[.?!]["\')\]]*$', en_a):
            reasons.append("first_lacks_terminal_punctuation")
        if words_a and words_a[-1].lower() in _EN_INCOMPLETE_END_WORDS:
            reasons.append(f"first_ends_with_incomplete_word:{words_a[-1].lower()}")
        if en_b and re.match(r"^[a-z]", en_b):
            reasons.append("second_starts_lowercase")
        if 1 <= len(words_b) <= 2 and gap <= 0.65 and (reasons or len(words_a) <= 4):
            reasons.append(f"short_tail_completion:{len(words_b)}_words")
        if 1 <= len(words_a) <= 3 and not re.search(r'[.?!]["\')\]]*$', en_a) and gap <= 0.65:
            reasons.append(f"short_head_fragment:{len(words_a)}_words")

    is_split_candidate = len(reasons) > 0
    same_confirmed_speaker = (
        not spk_a_unverified
        and not spk_b_unverified
        and str(spk_a_raw).strip() == str(spk_b_raw).strip()
    )

    if not is_split_candidate:
        decision = "not_a_split"
    elif same_confirmed_speaker:
        decision = "merge_confirmed_same_speaker"
    elif spk_a_unverified or spk_b_unverified:
        decision = "hold_unverified_speaker_review_required"
    else:
        decision = "hold_different_speakers_review_required"

    return {
        "first_id": curr.get("id"),
        "second_id": nxt.get("id"),
        "first_speaker": None if spk_a_unverified else str(spk_a_raw).strip(),
        "second_speaker": None if spk_b_unverified else str(spk_b_raw).strip(),
        "first_raw_speaker": spk_a_raw,
        "second_raw_speaker": spk_b_raw,
        "gap_seconds": gap,
        "first_text_en": en_a,
        "second_text_en": en_b,
        "is_split_candidate": is_split_candidate,
        "same_confirmed_speaker": same_confirmed_speaker,
        "decision": decision,
        "reasons": reasons,
        "review_required": bool(is_split_candidate and not same_confirmed_speaker),
    }


def resolve_phase14_splits_and_unverified_speakers(
    entries: list[dict[str, Any]],
    *,
    max_merge_gap_seconds: float = 1.25,
) -> dict[str, Any]:
    from dubbing_pipeline import _estimate_japanese_mora, _join_japanese_fragments

    items = [copy.deepcopy(e) for e in entries]
    merged_entries: list[dict[str, Any]] = []
    merged_groups: list[dict[str, Any]] = []
    blocked_splits: list[dict[str, Any]] = []

    i = 0
    n = len(items)
    while i < n:
        curr = items[i]
        if i + 1 < n:
            pair_eval = detect_unnatural_english_split_pair(
                curr, items[i + 1], max_gap_seconds=max_merge_gap_seconds
            )
            if pair_eval["decision"] == "merge_confirmed_same_speaker":
                nxt = items[i + 1]
                en_joined = (
                    str(curr.get("text_en") or "").strip().rstrip(",")
                    + " "
                    + str(nxt.get("text_en") or "").strip()
                ).strip()
                ja_joined = _join_japanese_fragments(
                    [str(curr.get("text_ja") or ""), str(nxt.get("text_ja") or "")]
                )
                src_ids = list(curr.get("source_utterance_ids") or [curr.get("id")]) + list(
                    nxt.get("source_utterance_ids") or [nxt.get("id")]
                )
                combined = copy.deepcopy(curr)
                combined["end"] = float(nxt.get("end", nxt.get("source_end", curr["end"])))
                combined["source_end"] = combined["end"]
                combined["clip_relative_end"] = float(
                    nxt.get("clip_relative_end", combined.get("clip_relative_end", 0.0))
                )
                combined["text_en"] = en_joined
                combined["text_ja"] = ja_joined
                combined["mora_count"] = _estimate_japanese_mora(ja_joined)
                combined["merged_from_ids"] = src_ids
                combined["source_utterance_ids"] = src_ids
                combined["split_merge_reasons"] = pair_eval["reasons"]
                if curr.get("words") or nxt.get("words"):
                    combined["words"] = list(curr.get("words") or []) + list(nxt.get("words") or [])
                merged_entries.append(combined)
                merged_groups.append(
                    {
                        "merged_id": combined["id"],
                        "merged_from_ids": src_ids,
                        "speaker": combined.get("speaker"),
                        "text_en": en_joined,
                        "text_ja": ja_joined,
                        "reasons": pair_eval["reasons"],
                    }
                )
                i += 2
                continue
            elif pair_eval["is_split_candidate"]:
                curr["review_required"] = True
                curr["split_review_note"] = pair_eval["decision"]
                items[i + 1]["review_required"] = True
                items[i + 1]["split_review_note"] = pair_eval["decision"]
                blocked_splits.append(pair_eval)
        merged_entries.append(curr)
        i += 1

    unverified_targets: list[dict[str, Any]] = []
    for item in merged_entries:
        raw_s = item.get("speaker") if item.get("speaker") is not None else item.get("raw_speaker")
        if is_unverified_speaker_label(raw_s):
            item["raw_speaker"] = raw_s if raw_s is not None else "UNKNOWN"
            item["speaker"] = None
            item["review_required"] = True
            unverified_targets.append(
                {
                    "id": item.get("id"),
                    "source_utterance_ids": item.get("source_utterance_ids") or [item.get("id")],
                    "raw_speaker": item["raw_speaker"],
                    "speaker": None,
                    "review_required": True,
                    "split_review_note": item.get("split_review_note"),
                    "start_seconds": item.get("start"),
                    "end_seconds": item.get("end"),
                    "clip_relative_start_seconds": item.get("clip_relative_start"),
                    "clip_relative_end_seconds": item.get("clip_relative_end"),
                    "text_en": item.get("text_en", ""),
                    "text_ja": item.get("text_ja", ""),
                }
            )

    return {
        "entries": merged_entries,
        "merged_groups": merged_groups,
        "blocked_unverified_splits": blocked_splits,
        "unverified_speaker_targets": unverified_targets,
    }


def adjust_overlong_script_entries(
    entries: list[dict[str, Any]],
    *,
    max_atempo: float = 1.20,
    assumed_mora_per_second: float = 7.8,
) -> dict[str, Any]:
    from dubbing_pipeline import _estimate_japanese_mora

    adjusted_entries: list[dict[str, Any]] = []
    adjustments_log: list[dict[str, Any]] = []
    unverified_semantic_shortening_items: list[dict[str, Any]] = []
    overlong_before = 0
    shortened_count = 0

    for item in entries:
        row = copy.deepcopy(item)
        slot_dur = max(0.15, float(row.get("end", 0.0)) - float(row.get("start", 0.0)))
        orig_ja = str(row.get("text_ja") or "").strip()
        text_en = str(row.get("text_en") or "").strip()
        orig_mora = _estimate_japanese_mora(orig_ja)
        est_dur = float(
            row.get("estimated_duration")
            or (row.get("timing_estimate") or {}).get("estimated_seconds")
            or round(orig_mora / assumed_mora_per_second, 3)
        )
        is_overlong = bool(
            row.get("overlong")
            or row.get("status") == "overlong"
            or (row.get("timing_estimate") or {}).get("overlong")
            or est_dur > slot_dur * 1.02
        )
        if is_overlong:
            overlong_before += 1
            target_mora = max(4, int(math.floor(slot_dur * assumed_mora_per_second * min(1.18, max_atempo))))
            proposal = propose_semantic_preserving_shortening(
                orig_ja,
                text_en=text_en,
                target_mora=target_mora,
                estimate_mora_fn=_estimate_japanese_mora,
            )
            if proposal["applied"]:
                shortened_count += 1
                row["original_text_ja_before_shortening"] = orig_ja
                row["text_ja"] = proposal["shortened_text_ja"]
                row["mora_count"] = proposal["shortened_mora"]
                row["shortening_applied"] = True
                row["semantic_review_required"] = True
                row["review_required"] = True
                unverified_semantic_shortening_items.append(
                    {
                        "id": row.get("id"),
                        "speaker": row.get("speaker"),
                        "text_en": text_en,
                        "original_text_ja": orig_ja,
                        "shortened_text_ja": proposal["shortened_text_ja"],
                        "applied_rules": proposal["applied_rules"],
                        "reason": "automated_shortening_semantic_equivalence_cannot_be_mechanically_guaranteed",
                    }
                )
            else:
                row["shortening_applied"] = False
                row["semantic_review_required"] = False
            new_est_dur = round(row["mora_count"] / assumed_mora_per_second, 3)
            fits_with_atempo = bool(new_est_dur <= slot_dur * max_atempo + 0.15)
            adjustments_log.append(
                {
                    "id": row.get("id"),
                    "speaker": row.get("speaker"),
                    "slot_duration_seconds": round(slot_dur, 4),
                    "original_text_ja": orig_ja,
                    "shortened_text_ja": row["text_ja"],
                    "shortening_applied": bool(proposal["applied"]),
                    "semantic_review_required": bool(proposal["applied"]),
                    "original_mora": orig_mora,
                    "shortened_mora": row["mora_count"],
                    "mora_saved": proposal["mora_saved"],
                    "target_mora_budget": target_mora,
                    "estimated_duration_before_seconds": round(est_dur, 3),
                    "estimated_duration_after_seconds": new_est_dur,
                    "fits_within_1_20x_estimate": fits_with_atempo,
                    "applied_rules": proposal["applied_rules"],
                    "semantic_validation": proposal["semantic_validation"],
                }
            )
        else:
            row["shortening_applied"] = False
            row["semantic_review_required"] = False
        adjusted_entries.append(row)

    return {
        "entries": adjusted_entries,
        "overlong_count_before": overlong_before,
        "shortened_entry_count": shortened_count,
        "unverified_semantic_shortening_items": unverified_semantic_shortening_items,
        "adjustments": adjustments_log,
    }


def reconcile_utterance_and_word_timestamps_with_selection(
    entries: list[dict[str, Any]],
    words: list[dict[str, Any]],
    *,
    selection_start: float,
    selection_end: float,
) -> dict[str, Any]:
    """Ensure every entry and word timestamp has consistent source-absolute (`300..480s`)
    and clip-relative (`0..180s`) coordinates regardless of whether the input JSON
    stored clip-relative (`0..180s`) or source-absolute (`300..480s`) numbers.
    """
    sel_s = float(selection_start)
    sel_e = float(selection_end)
    clip_dur = max(0.1, round(sel_e - sel_s, 4))

    reconciled_words: list[dict[str, Any]] = []
    for w in words or []:
        w_copy = copy.deepcopy(w)
        ws = float(w.get("start", w.get("source_start", 0.0)))
        we = float(w.get("end", w.get("source_end", ws)))
        if sel_s > 0.0 and ws < sel_s - 1.0 and 0.0 <= ws <= clip_dur + 5.0:
            rel_ws = max(0.0, ws)
            rel_we = max(rel_ws, we)
            abs_ws = round(sel_s + rel_ws, 4)
            abs_we = round(sel_s + rel_we, 4)
        else:
            abs_ws = round(ws, 4)
            abs_we = round(max(ws, we), 4)
            rel_ws = round(max(0.0, abs_ws - sel_s), 4)
            rel_we = round(max(rel_ws, abs_we - sel_s), 4)
        if abs_we > sel_s and abs_ws < sel_e + 15.0:
            w_copy["start"] = abs_ws
            w_copy["end"] = abs_we
            w_copy["source_start"] = abs_ws
            w_copy["source_end"] = abs_we
            w_copy["clip_relative_start"] = rel_ws
            w_copy["clip_relative_end"] = rel_we
            reconciled_words.append(w_copy)

    reconciled_entries: list[dict[str, Any]] = []
    for item in entries:
        row = copy.deepcopy(item)
        # Prefer explicit relative_start/relative_end when selection_start > 0 and start looks relative
        raw_s = float(row.get("start", row.get("source_start", 0.0)))
        raw_e = float(row.get("end", row.get("source_end", raw_s + 1.0)))
        if sel_s > 0.0 and raw_s < sel_s - 1.0 and 0.0 <= raw_s <= clip_dur + 15.0:
            rel_s = round(max(0.0, raw_s), 4)
            rel_e = round(max(rel_s + 0.05, raw_e), 4)
            abs_s = round(sel_s + rel_s, 4)
            abs_e = round(sel_s + rel_e, 4)
        else:
            abs_s = round(raw_s, 4)
            abs_e = round(max(raw_s + 0.05, raw_e), 4)
            rel_s = round(max(0.0, abs_s - sel_s), 4)
            rel_e = round(max(rel_s + 0.05, abs_e - sel_s), 4)

        row["start"] = abs_s
        row["end"] = abs_e
        row["source_start"] = abs_s
        row["source_end"] = abs_e
        row["clip_relative_start"] = rel_s
        row["clip_relative_end"] = rel_e

        # Also reconcile embedded words on the entry
        if isinstance(row.get("words"), list) and row["words"]:
            entry_words = []
            for uw in row["words"]:
                uws = float(uw.get("start", uw.get("source_start", 0.0)))
                uwe = float(uw.get("end", uw.get("source_end", uws)))
                if sel_s > 0.0 and uws < sel_s - 1.0 and 0.0 <= uws <= clip_dur + 15.0:
                    abs_uws = round(sel_s + uws, 4)
                    abs_uwe = round(sel_s + max(uws, uwe), 4)
                else:
                    abs_uws = round(uws, 4)
                    abs_uwe = round(max(uws, uwe), 4)
                entry_words.append(
                    {
                        **uw,
                        "start": abs_uws,
                        "end": abs_uwe,
                        "source_start": abs_uws,
                        "source_end": abs_uwe,
                        "clip_relative_start": round(max(0.0, abs_uws - sel_s), 4),
                        "clip_relative_end": round(max(0.0, abs_uwe - sel_s), 4),
                    }
                )
            row["words"] = entry_words
        elif reconciled_words:
            u_words = [
                w
                for w in reconciled_words
                if float(w["end"]) > abs_s + 0.01 and float(w["start"]) < abs_e - 0.01
            ]
            if u_words:
                row["words"] = u_words

        reconciled_entries.append(row)

    return {
        "entries": reconciled_entries,
        "words": reconciled_words,
        "selection_start": sel_s,
        "selection_end": sel_e,
        "clip_duration": clip_dur,
    }


def load_phase14_video_project(
    *,
    video_id: str | None = None,
    input_dir: str | Path | None = None,
    script_path: str | Path | None = None,
    video_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    from dubbing_pipeline import normalize_script_utterances

    root_base = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
    if input_dir is not None:
        in_dir = Path(input_dir).resolve()
        vid = video_id or in_dir.name
    elif video_id is not None:
        vid = str(video_id).strip()
        in_dir = (root_base / "input" / vid).resolve()
    elif script_path is not None:
        in_dir = Path(script_path).resolve().parent
        vid = in_dir.name
    else:
        raise ValueError("Specify --video-id, --input-dir, or --script for Phase 14.")

    def _load_opt_json(p: Path | None) -> dict[str, Any]:
        if p and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    selection_file = in_dir / "selection.json"
    utterances_file = in_dir / "utterances.json"
    translations_file = in_dir / "utterance_translations.json"
    words_file = in_dir / "word_timestamps.json"
    script_file = Path(script_path).resolve() if script_path else (in_dir / "dubbing_script.json")
    source_info_file = in_dir / "source.info.json"

    selection_doc = _load_opt_json(selection_file)
    utterances_doc = _load_opt_json(utterances_file)
    translations_doc = _load_opt_json(translations_file)
    words_doc = _load_opt_json(words_file)
    script_doc = _load_opt_json(script_file)
    source_info_doc = _load_opt_json(source_info_file)

    sel_start = None
    sel_end = None
    if isinstance(selection_doc.get("start"), (int, float)) and isinstance(selection_doc.get("end"), (int, float)):
        sel_start = float(selection_doc["start"])
        sel_end = float(selection_doc["end"])
    elif isinstance(script_doc.get("clip"), dict):
        c = script_doc["clip"]
        if c.get("start") is not None and c.get("end") is not None:
            sel_start = float(c["start"])
            sel_end = float(c["end"])
    elif isinstance(translations_doc.get("clip"), dict):
        c = translations_doc["clip"]
        if c.get("start") is not None and c.get("end") is not None:
            sel_start = float(c["start"])
            sel_end = float(c["end"])

    if not script_doc and translations_doc.get("utterances"):
        t_utts = translations_doc["utterances"]
        if sel_start is None or sel_end is None:
            sel_start = float(min((u.get("start", 0.0) for u in t_utts), default=0.0))
            sel_end = float(max((u.get("end", 30.0) for u in t_utts), default=30.0))
        script_doc = {
            "video_id": vid,
            "clip": {"start": sel_start, "end": sel_end, "duration": round(sel_end - sel_start, 4)},
            "entries": t_utts,
        }

    if not script_doc:
        raise FileNotFoundError(
            f"Neither dubbing_script.json nor utterance_translations.json found in {in_dir}"
        )

    if sel_start is not None and sel_end is not None:
        script_doc.setdefault("selection_range", {"start": sel_start, "end": sel_end})

    norm = normalize_script_utterances(script_doc, script_path=script_file, video_path=video_path)
    sel_start = float(sel_start if sel_start is not None else norm["selection_start"])
    sel_end = float(sel_end if sel_end is not None else norm["selection_end"])
    clip_dur = round(max(0.1, sel_end - sel_start), 4)

    raw_utts_by_id: dict[str, dict[str, Any]] = {}
    for u in utterances_doc.get("utterances", []) if isinstance(utterances_doc, dict) else []:
        if u.get("id") is not None:
            raw_utts_by_id[str(u["id"])] = u
    trans_by_id: dict[str, dict[str, Any]] = {}
    for u in translations_doc.get("utterances", []) if isinstance(translations_doc, dict) else []:
        if u.get("id") is not None:
            trans_by_id[str(u["id"])] = u

    all_words = words_doc.get("words", []) if isinstance(words_doc, dict) else []

    enriched_entries: list[dict[str, Any]] = []
    for item in norm["utterances"]:
        row = copy.deepcopy(item)
        src_ids = row.get("utterance_ids") or row.get("source_utterance_ids") or [row.get("raw_id", row["id"])]
        row["source_utterance_ids"] = list(src_ids)
        if not row.get("text_en"):
            en_parts = []
            for sid in src_ids:
                u_ref = trans_by_id.get(str(sid)) or raw_utts_by_id.get(str(sid))
                if u_ref and (u_ref.get("text") or u_ref.get("text_en")):
                    en_parts.append(str(u_ref.get("text") or u_ref.get("text_en")).strip())
            if en_parts:
                row["text_en"] = " ".join(en_parts)
        enriched_entries.append(row)

    reconciled = reconcile_utterance_and_word_timestamps_with_selection(
        enriched_entries,
        all_words,
        selection_start=sel_start,
        selection_end=sel_end,
    )

    s_int = int(round(sel_start))
    e_int = int(round(sel_end))
    video_candidates = [
        Path(video_path).resolve() if video_path else None,
        root_base / "output" / vid / f"clip_{s_int}_{e_int}.mp4",
        in_dir.parent.parent / "output" / vid / f"clip_{s_int}_{e_int}.mp4",
        in_dir / f"clip_{s_int}_{e_int}.mp4",
        in_dir / "source.mp4",
        root_base / "input" / vid / "source.mp4",
    ]
    resolved_video = None
    for cand in video_candidates:
        if cand is not None and cand.is_file() and cand.stat().st_size > 500:
            resolved_video = cand.resolve()
            break

    source_video_duration = None
    if isinstance(source_info_doc.get("duration"), (int, float)):
        source_video_duration = float(source_info_doc["duration"])

    return {
        "video_id": vid,
        "input_dir": in_dir,
        "selection_start": sel_start,
        "selection_end": sel_end,
        "clip_duration": clip_dur,
        "resolved_video_path": resolved_video,
        "source_video_duration": source_video_duration,
        "selection_file": selection_file if selection_file.is_file() else None,
        "utterances_file": utterances_file if utterances_file.is_file() else None,
        "translations_file": translations_file if translations_file.is_file() else None,
        "words_file": words_file if words_file.is_file() else None,
        "script_file": script_file if script_file.is_file() else None,
        "raw_utterance_count": len(utterances_doc.get("utterances", [])),
        "translation_utterance_count": len(translations_doc.get("utterances", [])),
        "script_entry_count": len(reconciled["entries"]),
        "entries": reconciled["entries"],
        "words": reconciled["words"],
        "raw_script_doc": script_doc,
    }


def _reconstruct_clip_video_for_test(
    output_mp4: Path,
    *,
    duration_seconds: float,
    entries: list[dict[str, Any]],
    selection_start: float,
) -> Path:
    sr = 24000
    n_total = max(sr, int(round(duration_seconds * sr)))
    rng = np.random.default_rng(140)
    audio_track = rng.normal(0.0, 0.005, size=n_total).astype(np.float32)
    for idx, u in enumerate(entries):
        rel_s = max(0.0, float(u.get("start", selection_start)) - selection_start)
        rel_e = min(duration_seconds, float(u.get("end", selection_start + 1.0)) - selection_start)
        if rel_e <= rel_s:
            continue
        i0 = max(0, int(round(rel_s * sr)))
        i1 = min(n_total, int(round(rel_e * sr)))
        if i1 <= i0:
            continue
        f0 = 120.0 + (idx % 3) * 32.0
        wt = np.arange(i1 - i0, dtype=np.float32) / float(sr)
        env = np.sin(np.pi * np.linspace(0.0, 1.0, i1 - i0, dtype=np.float32)) ** 0.5
        sig = 0.20 * (np.sin(2.0 * np.pi * f0 * wt) + 0.4 * np.sin(4.0 * np.pi * f0 * wt)) * env
        audio_track[i0:i1] += sig

    wav_tmp = output_mp4.with_suffix(".tmp_src.wav")
    pcm = np.clip(np.round(audio_track * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(wav_tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=0x16202A:s=1280x720:r=30:d={duration_seconds:.3f}",
            "-i", str(wav_tmp),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(output_mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if wav_tmp.exists():
        wav_tmp.unlink()
    return output_mp4


def verify_qwen3_tts_mlx_available(*, voice_mode: str = "preset", preset_model: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Verify that the real Qwen3-TTS-MLX backend and cached model weights are loadable."""
    try:
        from tts_generator import DEFAULT_MODEL, QwenPresetSynthesizer

        model_id = preset_model or DEFAULT_MODEL
        synth = QwenPresetSynthesizer(model_id=model_id, seed=seed)
        return {
            "available": True,
            "backend_identity": synth.identity,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "backend_identity": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_phase14_universal_dubbing(
    *,
    video_id: str | None = None,
    input_dir: str | Path | None = None,
    script_path: str | Path | None = None,
    video_path: str | Path | None = None,
    output_dir: str | Path = "output/phase14_new_video_01",
    voice_mode: str = "preset",
    preset_speaker_map: dict[str, str] | None = None,
    provisional_unknown_voice: bool = False,
    test_synthetic_tts: bool = False,
    auto_extend_clip: bool = True,
    max_clip_extension_seconds: float = 12.0,
    target_lufs: float = -16.0,
    duck_hold_seconds: float = 3.2,
    same_speaker_hold_seconds: float = 4.0,
    duck_gain: float = 0.18,
    ducking_style: str = "continuous",
    mix_mode: str = "continuous-voiceover",
    base_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Execute Phase 14 Universal Multi-Video Japanese Dubbing Pipeline.

    Strict TTS Policy:
    - By default (`test_synthetic_tts=False`), `Qwen3-TTS-MLX` MUST be available and is used
      to generate real Japanese speech. If `Qwen3-TTS-MLX` is unavailable, raises
      `QwenModelUnavailableError` immediately instead of silently switching to synthetic audio!
    - Synthetic acoustic fallback audio is generated ONLY when `test_synthetic_tts=True`
      (or `--test-synthetic-tts` on the CLI) is explicitly passed for automated tests.
    """
    from dubbing_pipeline import (
        _atomic_write_json,
        _estimate_japanese_mora,
        _snapshot_input_files,
        _synthesize_mora_conditioned_speech_wav,
        _verify_input_files_unchanged,
        probe_media_streams,
        run_phase13_3_dubbing_pipeline,
    )

    out_root = Path(output_dir).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(
            f"Output directory {out_root} is not empty; specify a new --output path to preserve existing results."
        )

    project = load_phase14_video_project(
        video_id=video_id,
        input_dir=input_dir,
        script_path=script_path,
        video_path=video_path,
        base_dir=base_dir,
    )

    reuse_wav_dir = kwargs.pop("reuse_wav_dir", None)
    qwen_check = verify_qwen3_tts_mlx_available(voice_mode=voice_mode)

    if not qwen_check["available"] and reuse_wav_dir is None and not test_synthetic_tts:
        raise QwenModelUnavailableError(
            f"Qwen3-TTS-MLX is not available in the current Python environment ({qwen_check['error']}). "
            f"Automatic fallback to synthetic audio is disabled. "
            f"Run with '~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python' on Mac, "
            f"or pass --test-synthetic-tts explicitly for offline unit testing."
        )

    active_video_path = project["resolved_video_path"]
    if (active_video_path is None or not active_video_path.is_file()) and not test_synthetic_tts:
        raise FileNotFoundError(
            f"Clip video MP4 not found for video_id={project['video_id']!r} "
            f"(expected output/{project['video_id']}/clip_{int(project['selection_start'])}_{int(project['selection_end'])}.mp4). "
            f"Provide --video or pass --test-synthetic-tts in unit tests."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    work_dir = out_root / "_phase14_prep"
    work_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = [
        project["selection_file"],
        project["utterances_file"],
        project["translations_file"],
        project["words_file"],
        project["script_file"],
        project["resolved_video_path"],
    ]
    snapshot_before = _snapshot_input_files([p for p in tracked_files if p is not None])

    # 1. Resolve unnatural splits & UNKNOWN speakers safely
    split_and_speaker_res = resolve_phase14_splits_and_unverified_speakers(
        project["entries"],
        max_merge_gap_seconds=1.25,
    )

    # 2. Shorten overlong Japanese script entries with surface validation + human review tracking
    shortening_res = adjust_overlong_script_entries(
        split_and_speaker_res["entries"],
        max_atempo=1.20,
    )
    processed_entries = shortening_res["entries"]

    # 3. Assign distinct preset voices to confirmed speakers; NEVER turn UNKNOWN into a confirmed speaker
    confirmed_speakers = sorted(
        {
            str(e["speaker"]).strip()
            for e in processed_entries
            if e.get("speaker") and not is_unverified_speaker_label(e.get("speaker"))
        }
    )
    resolved_preset_map: dict[str, str] = dict(preset_speaker_map or {})
    used_voices = set(resolved_preset_map.values())
    pool_idx = 0
    for spk in confirmed_speakers:
        if spk not in resolved_preset_map:
            while (
                pool_idx < len(PHASE14_CONFIRMED_PRESET_POOL)
                and PHASE14_CONFIRMED_PRESET_POOL[pool_idx] in used_voices
            ):
                pool_idx += 1
            chosen = PHASE14_CONFIRMED_PRESET_POOL[pool_idx % len(PHASE14_CONFIRMED_PRESET_POOL)]
            resolved_preset_map[spk] = chosen
            used_voices.add(chosen)
            pool_idx += 1

    active_pipeline_entries: list[dict[str, Any]] = []
    provisional_assigned_ids: list[Any] = []
    held_silent_ids: list[Any] = []

    for e in processed_entries:
        row = copy.deepcopy(e)
        if is_unverified_speaker_label(row.get("speaker")):
            if provisional_unknown_voice and voice_mode == "preset":
                row["speaker"] = PHASE14_PROVISIONAL_SPEAKER_ID
                row["provisional_unknown_voice"] = True
                row["review_required"] = True
                if not str(row.get("text_ja") or "").startswith("[話者未確定"):
                    row["subtitle_text_ja"] = f"[話者未確定(仮音声)] {row.get('text_ja', '')}".strip()
                resolved_preset_map[PHASE14_PROVISIONAL_SPEAKER_ID] = PHASE14_PROVISIONAL_PRESET_VOICE
                active_pipeline_entries.append(row)
                provisional_assigned_ids.append(row.get("id"))
            else:
                # Keep speaker as None so partition_and_manage_null_speakers holds it out of synthesis
                row["speaker"] = None
                row["review_required"] = True
                held_silent_ids.append(row.get("id"))
                active_pipeline_entries.append(row)
        else:
            active_pipeline_entries.append(row)

    prepared_script_doc = {
        "video_id": project["video_id"],
        "selection_range": {
            "start": project["selection_start"],
            "end": project["selection_end"],
            "duration": project["clip_duration"],
        },
        "clip": {
            "start": project["selection_start"],
            "end": project["selection_end"],
            "duration": project["clip_duration"],
        },
        "utterances": active_pipeline_entries,
        "entries": active_pipeline_entries,
    }
    prepared_script_path = work_dir / "dubbing_script_phase14.json"
    _atomic_write_json(prepared_script_path, prepared_script_doc)

    # Save reconciled words file in work_dir so word timestamps match [selection_start, selection_end]
    prepared_words_path = None
    if project["words"]:
        prepared_words_path = work_dir / "word_timestamps_phase14.json"
        _atomic_write_json(
            prepared_words_path,
            {
                "video_id": project["video_id"],
                "clip": {"start": project["selection_start"], "end": project["selection_end"]},
                "words": project["words"],
            },
        )

    video_origin = "existing_clip_mp4"
    if active_video_path is None or not active_video_path.is_file():
        active_video_path = _reconstruct_clip_video_for_test(
            work_dir / f"clip_{int(project['selection_start'])}_{int(project['selection_end'])}.mp4",
            duration_seconds=project["clip_duration"],
            entries=active_pipeline_entries,
            selection_start=project["selection_start"],
        )
        video_origin = "test_mode_reconstructed_clip_mp4"

    input_clip_probe = probe_media_streams(active_video_path)
    input_clip_duration = float(input_clip_probe.get("duration_seconds") or project["clip_duration"])
    is_precut_clip_0_to_duration = bool(
        project["selection_start"] > 0.0
        and abs(input_clip_duration - project["clip_duration"]) <= 2.5
    )

    tts_execution_mode = "live_qwen3_tts_mlx"
    if test_synthetic_tts and reuse_wav_dir is None:
        tts_execution_mode = "explicit_test_synthetic_tts"
        fallback_wav_dir = work_dir / "test_synthetic_wavs"
        fallback_wav_dir.mkdir(parents=True, exist_ok=True)
        synth_entries = [
            u for u in active_pipeline_entries if u.get("speaker") and not is_unverified_speaker_label(u.get("speaker"))
        ]
        for idx, u in enumerate(synth_entries, start=1):
            spk = str(u["speaker"])
            mora_cnt = max(4, _estimate_japanese_mora(u.get("text_ja", "")))
            slot_d = max(0.4, float(u["end"]) - float(u["start"]))
            target_d = max(0.65, min(slot_d * 1.12, round(mora_cnt * 0.115 + 0.10, 3)))
            f0_map = {
                "SPEAKER_00": 132.0,
                "SPEAKER_01": 185.0,
                "SPEAKER_02": 112.0,
                PHASE14_PROVISIONAL_SPEAKER_ID: 215.0,
            }
            f0_val = f0_map.get(spk, 145.0)
            safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(u["id"]))
            for fname in (f"{u['id']}.wav", f"{safe_uid}.wav", f"dub-{idx:03d}.wav"):
                _synthesize_mora_conditioned_speech_wav(
                    fallback_wav_dir / fname,
                    u.get("text_ja", ""),
                    target_duration_seconds=target_d,
                    f0_hz=f0_val,
                    seed=1400 + idx,
                )
        reuse_wav_dir = fallback_wav_dir
    elif reuse_wav_dir is not None:
        tts_execution_mode = "reused_wav_dir"

    inner_out_dir = work_dir / "pipeline_run"
    primary_speaker = confirmed_speakers[0] if confirmed_speakers else PHASE14_PROVISIONAL_SPEAKER_ID
    default_voice = resolved_preset_map.get(primary_speaker, "Uncle_Fu")

    speaker_source_types = {spk: "permitted" for spk in resolved_preset_map}
    _, inner_report = run_phase13_3_dubbing_pipeline(
        script_path=prepared_script_path,
        video_path=active_video_path,
        output_dir=inner_out_dir,
        speaker_id=primary_speaker,
        voice_mode=voice_mode,
        preset_speaker_map=(resolved_preset_map if voice_mode == "preset" else None),
        default_preset_voice=default_voice,
        allow_preset_fallback=(voice_mode not in ("xvector", "icl")),
        extract_reference_from_video=(voice_mode != "preset"),
        speaker_source_types=speaker_source_types,
        merge_split_utterances=False,
        null_speaker_policy="hold_for_verification",
        include_unverified_subtitles=True,
        word_timestamps_path=prepared_words_path,
        use_word_level_sync=bool(prepared_words_path),
        auto_extend_clip=auto_extend_clip,
        max_clip_extension_seconds=max_clip_extension_seconds,
        target_lufs=target_lufs,
        duck_hold_seconds=duck_hold_seconds,
        same_speaker_hold_seconds=same_speaker_hold_seconds,
        duck_gain=duck_gain,
        unspoken_gain=duck_gain if ducking_style == "continuous" else 1.0,
        ducking_style=ducking_style,
        mix_mode=mix_mode,
        source_video_duration=project["source_video_duration"],
        reuse_wav_dir=reuse_wav_dir,
        prevent_same_speaker_overlap=True,
        **kwargs,
    )

    for fname in (
        "final_ja.mp4",
        "final_ja_no_subtitles.mp4",
        "dubbed_audio.wav",
        "final_ja.ass",
        "final_ja.srt",
        "voice_consistency_report.json",
        "speaker_assignment_report.json",
        "timing_quality_report.json",
        "subtitle_sync_report.json",
        "ducking_verification_report.json",
        "lufs_normalization_report.json",
        "speaker_voice_registry.json",
    ):
        src_p = inner_out_dir / fname
        if src_p.is_file():
            shutil.copy2(src_p, out_root / fname)

    col_analysis = inner_report.get("collision_and_overlap_analysis", {})
    placements = col_analysis.get("placements", {})
    post_tts_overflows = []
    for adj in shortening_res["adjustments"]:
        uid = str(adj["id"])
        p_info = placements.get(uid) or {}
        actual_dur = float(p_info.get("actual_tts_duration_seconds", 0.0))
        slot_dur = float(adj["slot_duration_seconds"])
        applied_atempo = float(p_info.get("applied_atempo", 1.0))
        adj["post_tts_actual_duration_seconds"] = round(actual_dur, 4)
        adj["post_tts_applied_atempo"] = round(applied_atempo, 4)
        adj["post_tts_slot_overflow_seconds"] = round(max(0.0, actual_dur - slot_dur), 4)
        if actual_dur > slot_dur + 0.25:
            post_tts_overflows.append(
                {
                    "id": adj["id"],
                    "speaker": adj["speaker"],
                    "slot_duration_seconds": slot_dur,
                    "actual_tts_duration_seconds": round(actual_dur, 4),
                    "applied_atempo": round(applied_atempo, 4),
                    "overflow_seconds": round(actual_dur - slot_dur, 4),
                    "text_ja": adj["shortened_text_ja"],
                }
            )

    # Verify timeline alignment between source selection [300..480] and pre-cut clip [0..180]
    eff_sel = inner_report.get("selection_range", {})
    eff_clip_dur = float(eff_sel.get("clip_duration_seconds", project["clip_duration"]))
    final_mp4_probe = probe_media_streams(out_root / "final_ja.mp4")
    final_mp4_dur = float(final_mp4_probe.get("duration_seconds") or 0.0)
    placement_bounds_valid = True
    for p_uid, p_val in placements.items():
        c_rel_s = float(p_val.get("clip_relative_start_seconds", 0.0))
        c_rel_e = float(p_val.get("clip_relative_end_seconds", 0.0))
        src_abs_s = float(p_val.get("placement_start_seconds", 0.0))
        expected_rel_s = src_abs_s - project["selection_start"]
        if c_rel_s < -1e-3 or c_rel_e > eff_clip_dur + 0.05 or abs(c_rel_s - expected_rel_s) > 0.05:
            placement_bounds_valid = False
            break

    timeline_consistency_report = {
        "source_selection_start_seconds": project["selection_start"],
        "source_selection_end_seconds": project["selection_end"],
        "effective_source_selection_end_seconds": float(eff_sel.get("source_end_seconds", project["selection_end"])),
        "input_clip_mp4_duration_seconds": round(input_clip_duration, 4),
        "input_clip_is_precut_0_to_duration": is_precut_clip_0_to_duration,
        "effective_clip_duration_seconds": round(eff_clip_dur, 4),
        "final_mp4_duration_seconds": round(final_mp4_dur, 4),
        "all_placements_mapped_to_clip_relative_0_to_duration": placement_bounds_valid,
        "duration_matches_effective_clip": abs(final_mp4_dur - eff_clip_dur) <= 0.25,
    }

    script_adjustment_report = {
        "phase": "14",
        "video_id": project["video_id"],
        "selection_range_seconds": [project["selection_start"], project["selection_end"]],
        "raw_utterance_count": project["raw_utterance_count"],
        "translation_utterance_count": project["translation_utterance_count"],
        "initial_script_entry_count": project["script_entry_count"],
        "post_merge_entry_count": len(processed_entries),
        "overlong_count_before": shortening_res["overlong_count_before"],
        "shortened_entry_count": shortening_res["shortened_entry_count"],
        "semantic_equivalence_mechanically_guaranteed": False,
        "unverified_semantic_shortening_count": len(shortening_res["unverified_semantic_shortening_items"]),
        "unverified_semantic_shortening_items": shortening_res["unverified_semantic_shortening_items"],
        "unresolved_same_speaker_collisions": col_analysis.get("unresolved_same_speaker_collisions", []),
        "unresolved_post_tts_slot_overflows": post_tts_overflows,
        "adjustments": shortening_res["adjustments"],
    }
    _atomic_write_json(out_root / "script_adjustment_report.json", script_adjustment_report)

    speaker_verification_doc = {
        "phase": "14",
        "video_id": project["video_id"],
        "unknown_converted_to_confirmed_speaker": False,
        "unverified_speaker_policy": (
            "provisional_distinct_preset_voice_marked_for_review"
            if (provisional_unknown_voice and voice_mode == "preset")
            else "held_silent_for_human_verification_with_tagged_subtitles"
        ),
        "provisional_speaker_id": PHASE14_PROVISIONAL_SPEAKER_ID if provisional_assigned_ids else None,
        "provisional_preset_voice": PHASE14_PROVISIONAL_PRESET_VOICE if provisional_assigned_ids else None,
        "confirmed_speaker_preset_map": {
            k: v for k, v in resolved_preset_map.items() if k != PHASE14_PROVISIONAL_SPEAKER_ID
        },
        "merged_confirmed_same_speaker_splits": split_and_speaker_res["merged_groups"],
        "blocked_unverified_or_cross_speaker_splits": split_and_speaker_res["blocked_unverified_splits"],
        "unverified_speaker_targets": split_and_speaker_res["unverified_speaker_targets"],
        "provisional_voiced_utterance_ids": provisional_assigned_ids,
        "held_silent_utterance_ids": held_silent_ids,
    }
    _atomic_write_json(out_root / "speaker_verification_targets.json", speaker_verification_doc)

    unverified_items_summary = {
        "unverified_speaker_targets": split_and_speaker_res["unverified_speaker_targets"],
        "blocked_unverified_or_cross_speaker_splits": split_and_speaker_res["blocked_unverified_splits"],
        "unverified_semantic_shortening_items": shortening_res["unverified_semantic_shortening_items"],
        "unresolved_post_tts_slot_overflows": post_tts_overflows,
    }

    integrity_check = _verify_input_files_unchanged(snapshot_before)

    inner_report["phase"] = "14"
    inner_report["video_id"] = project["video_id"]
    inner_report["tts_execution_mode"] = tts_execution_mode
    inner_report["is_real_qwen_model_generation"] = (tts_execution_mode == "live_qwen3_tts_mlx")
    inner_report["is_test_synthetic_generation"] = (tts_execution_mode == "explicit_test_synthetic_tts")
    inner_report["video_source_origin"] = video_origin
    inner_report["timeline_consistency_verification"] = timeline_consistency_report
    inner_report["unverified_items_requiring_human_review"] = unverified_items_summary
    inner_report["phase14_universal_summary"] = {
        "video_id": project["video_id"],
        "selection_start_seconds": project["selection_start"],
        "selection_end_seconds": project["selection_end"],
        "clip_duration_seconds": project["clip_duration"],
        "effective_clip_duration_seconds": round(eff_clip_dur, 4),
        "voice_mode": voice_mode,
        "tts_execution_mode": tts_execution_mode,
        "is_real_qwen_model_generation": (tts_execution_mode == "live_qwen3_tts_mlx"),
        "is_test_synthetic_generation": (tts_execution_mode == "explicit_test_synthetic_tts"),
        "unknown_converted_to_confirmed_speaker": False,
        "confirmed_speaker_preset_map": speaker_verification_doc["confirmed_speaker_preset_map"],
        "provisional_unknown_voice_used": bool(provisional_assigned_ids),
        "provisional_preset_voice": PHASE14_PROVISIONAL_PRESET_VOICE if provisional_assigned_ids else None,
        "merged_split_count": len(split_and_speaker_res["merged_groups"]),
        "blocked_unverified_split_count": len(split_and_speaker_res["blocked_unverified_splits"]),
        "unverified_speaker_target_count": len(split_and_speaker_res["unverified_speaker_targets"]),
        "overlong_count_before": shortening_res["overlong_count_before"],
        "shortened_entry_count": shortening_res["shortened_entry_count"],
        "unverified_semantic_shortening_count": len(shortening_res["unverified_semantic_shortening_items"]),
        "unresolved_post_tts_overflow_count": len(post_tts_overflows),
        "timeline_consistency_passed": bool(
            timeline_consistency_report["all_placements_mapped_to_clip_relative_0_to_duration"]
            and timeline_consistency_report["duration_matches_effective_clip"]
        ),
        "input_files_unchanged": bool(integrity_check.get("verified", True)),
    }

    report_path = out_root / "pipeline_report.json"
    _atomic_write_json(report_path, inner_report)
    if (inner_out_dir / "pipeline_report.md").is_file():
        shutil.copy2(inner_out_dir / "pipeline_report.md", out_root / "pipeline_report.md")

    return report_path, inner_report
