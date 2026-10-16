"""Local, traceable dubbing script MVP. No network, TTS or audio truncation."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from language import Analyzer
from language.cache import atomic_json, fingerprint
from language.english import EnglishAnalyzer
from language.japanese import JapaneseAnalyzer
from language.rulebook import Rulebook
from language.timing import TimingConfig, TimingEngine

FORMAT_VERSION = 1
ENGINE_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def clip_range(data, label):
    if not isinstance(data, dict) or not isinstance(data.get("clip"), dict):
        raise ValueError(f"{label}: missing clip.")
    clip = data["clip"]
    start, end = (finite_number(clip.get(k), label + ".clip." + k) for k in ("start", "end"))
    if not 0 <= start < end:
        raise ValueError(f"{label}: invalid clip range.")
    return {"start": start, "end": end}


def validate_inputs(source, translations, selection=None):
    clip = clip_range(source, "utterances")
    other = clip_range(translations, "translations")
    if any(abs(clip[k] - other[k]) > 0.01 for k in clip):
        raise ValueError("Utterances and translations use different clip ranges.")
    if not isinstance(source.get("video_id"), str) or not source["video_id"].strip():
        raise ValueError("Missing source video_id.")
    if translations.get("format_version", 1) != 1:
        raise ValueError("Unsupported translation format version.")
    if source.get("video_id") != translations.get("video_id"):
        raise ValueError("Mismatched video IDs.")
    if source.get("format_version", 3) != 3:
        raise ValueError("Unsupported utterance format; expected v3.")
    if selection is not None:
        selected = clip_range({"clip": selection}, "selection")
        if any(abs(clip[k] - selected[k]) > 0.01 for k in clip):
            raise ValueError("selection.json is stale or belongs to a different clip.")
    mappings = []
    for label, data in [("utterances", source), ("translations", translations)]:
        items = data.get("utterances")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{label}: utterances must be a non-empty list.")
        mapping = {}
        for item in items:
            if not isinstance(item, dict) or type(item.get("id")) is not int:
                raise ValueError(f"{label}: each utterance needs an integer ID.")
            uid = item["id"]
            if uid in mapping:
                raise ValueError(f"{label}: duplicate ID {uid}.")
            if not isinstance(item.get("text_en"), str) or not item["text_en"].strip():
                raise ValueError(f"{label}: empty English at ID {uid}.")
            if "speaker" not in item:
                raise ValueError(f"{label}: speaker must be explicit (null is allowed).")
            if "review_required" in item and type(item["review_required"]) is not bool:
                raise ValueError(f"{label}: review_required must be boolean.")
            speaker = item.get("speaker")
            if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
                raise ValueError(f"{label}: invalid speaker at ID {uid}.")
            times = {k: finite_number(item.get(k), f"{label}.{uid}.{k}")
                     for k in ("start", "end", "relative_start", "relative_end")}
            if not 0 <= times["relative_start"] < times["relative_end"] <= clip["end"] - clip["start"] + 0.01:
                raise ValueError(f"{label}: invalid relative time at ID {uid}.")
            for k in ("start", "end"):
                if abs(times[k] - clip["start"] - times["relative_" + k]) > 0.01:
                    raise ValueError(f"{label}: absolute/relative time mismatch at ID {uid}.")
            if "review_reasons" in item and (not isinstance(item["review_reasons"], list) or
                    any(not isinstance(reason, str) for reason in item["review_reasons"])):
                raise ValueError(f"{label}: invalid review reasons.")
            if "words" in item and (not isinstance(item["words"], list) or
                                     any(not isinstance(w, dict) for w in item["words"])):
                raise ValueError(f"{label}: invalid words.")
            if label == "translations" and (not isinstance(item.get("text_ja"), str) or not item["text_ja"].strip()):
                raise ValueError(f"Missing Japanese translation for ID {uid}.")
            mapping[uid] = item
        mappings.append(mapping)
    sources, translated = mappings
    if sources.keys() != translated.keys():
        raise ValueError("Source/translation IDs differ; partial translations cannot be mixed.")
    for uid, item in sources.items():
        other = translated[uid]
        for k in ("speaker", "text_en"):
            if item.get(k) != other.get(k):
                raise ValueError(f"Stale translation {uid}: {k} mismatch.")
        for k in ("start", "end", "relative_start", "relative_end"):
            if abs(item[k] - other[k]) > 0.01:
                raise ValueError(f"Stale translation {uid}: {k} mismatch.")
    return clip, sorted(sources.values(), key=lambda x: (x["relative_start"], x["relative_end"], x["id"])), translated


def group_utterances(items, english: Analyzer, max_gap=0.5, max_duration=14.0):
    """Merge only adjacent, non-overlapping known-speaker fragments/short turns.

    Another speaker or UNKNOWN is a hard barrier, even during overlap. This
    avoids inventing an attribution; unresolved cross-turn grammar is reviewed.
    """
    groups = []
    for item in items:
        if groups:
            group = groups[-1]
            last = group[-1]
            gap = item["relative_start"] - last["relative_end"]
            merge = (item["speaker"] not in (None, "UNKNOWN", "REVIEW") and
                     item["speaker"] == last["speaker"] and 0 <= gap <= max_gap and
                     item["relative_end"] - group[0]["relative_start"] <= max_duration and
                     (english.analyze(last["text_en"])["possibly_fragmented"] or
                      last["relative_end"] - last["relative_start"] <= 4.5))
            if merge:
                group.append(item)
                continue
        groups.append([item])
    return groups


def source_review(group, translated):
    reasons = set()
    for item in group:
        for evidence in (item, translated[item["id"]]):
            reasons.update(evidence.get("review_reasons", []))
            if evidence.get("review_required"):
                reasons.add("source_review_required")
        if item["speaker"] in (None, "UNKNOWN", "REVIEW"):
            reasons.add("unknown_speaker")
        for word in item.get("words", []):
            if word.get("original_status", word.get("status", "single")) != "single" or word.get("review_required"):
                reasons.add("word_assignment_needs_review")
            if word.get("overlapping_speech"):
                reasons.add("overlapping_speech")
    return reasons


def optimize(source, translations, *, english: Analyzer, japanese: Analyzer,
             rulebook=None, timing=None, seconds=30.0, selection=None,
             candidates=None, provider=None):
    seconds = finite_number(seconds, "seconds")
    if seconds <= 0:
        raise ValueError("seconds must be positive.")
    rulebook, timing = rulebook or Rulebook(), timing or TimingEngine()
    clip, all_items, translated = validate_inputs(source, translations, selection)
    limit = min(seconds, clip["end"] - clip["start"])
    included, excluded = [], []
    for item in all_items:
        if item["relative_start"] >= limit:
            reason = "outside_preview"
        elif item["relative_end"] > limit:
            reason = "crosses_preview_boundary"
        else:
            included.append(item)
            continue
        excluded.append({"source_utterance_id": item["id"], "reason": reason,
                         "relative_start": item["relative_start"], "relative_end": item["relative_end"]})
    if not included:
        raise ValueError("No complete utterances within preview; increase --seconds.")
    input_hash = fingerprint({"source": source, "translations": translations})
    proposals = {}
    if candidates is not None:
        if not isinstance(candidates, dict) or candidates.get("format_version") != 1 or candidates.get("input_fingerprint") != input_hash:
            raise ValueError("Candidate file is stale or has an unsupported format.")
        if not isinstance(candidates.get("entries"), list):
            raise ValueError("Candidate entries must be a list.")
        for candidate in candidates["entries"]:
            if (not isinstance(candidate, dict) or not isinstance(candidate.get("source_utterance_ids"), list)
                    or not candidate["source_utterance_ids"] or
                    any(type(uid) is not int for uid in candidate["source_utterance_ids"])):
                raise ValueError("Invalid candidate source IDs.")
            ids = tuple(candidate["source_utterance_ids"])
            if ids in proposals or len(ids) != len(set(ids)):
                raise ValueError("Duplicate candidate IDs.")
            if not isinstance(candidate.get("text_ja"), str) or not candidate["text_ja"].strip():
                raise ValueError("Candidate text must be non-empty.")
            proposals[ids] = candidate
    entries = []
    groups = group_utterances(included, english)
    if set(proposals) - {tuple(x["id"] for x in g) for g in groups}:
        raise ValueError("Candidates must match complete selected groups; never mix speakers or excluded IDs.")
    for index, group in enumerate(groups):
        ids = [u["id"] for u in group]
        start, end = group[0]["relative_start"], max(u["relative_end"] for u in group)
        available = end - start
        en = " ".join(u["text_en"] for u in group)
        original = "".join(translated[u["id"]]["text_ja"].strip() for u in group)
        en_analysis = english.analyze(en)
        original_analysis = japanese.analyze(original)
        baseline = timing.estimate(original_analysis, available)
        reasons = source_review(group, translated)
        if en_analysis["possibly_fragmented"]:
            reasons.add("english_fragment")
        if not all(en_analysis["capabilities"].values()):
            reasons.add("english_analysis_limited")
        original_findings = rulebook.validate(en, original, en_analysis)
        proposed, applied = rulebook.rewrite(original)
        method = "local_rules" if applied else "unchanged"
        proposal_reason = ", ".join(applied)
        manual = proposals.get(tuple(ids))
        if manual:
            proposed, method = manual["text_ja"].strip(), "candidate_file"
            proposal_reason = manual.get("reason", "User-supplied candidate; semantic review pending.")
        elif provider is not None and (baseline["timing_status"] == "overlong" or len(group) > 1):
            positions = [all_items.index(u) for u in group]
            context = all_items[max(0, min(positions) - 2): min(len(all_items), max(positions) + 3)]
            request = {"engine_version": ENGINE_VERSION, "input_fingerprint": input_hash,
                       "rules": rulebook.data, "analyzers": [english.identity, japanese.identity],
                       "timing": asdict(timing.config), "source_utterance_ids": ids,
                       "speaker": group[0]["speaker"], "text_en": en, "original_text_ja": original,
                       "available_seconds": available,
                       "context_reference_only": [{"id": u["id"], "speaker": u["speaker"],
                                                   "text_en": u["text_en"]} for u in context if u["id"] not in ids]}
            proposed, method = provider.propose(request), "provider_candidate"
            proposal_reason = "Context-aware candidate; semantic review pending."
            if not isinstance(proposed, str) or not proposed.strip():
                raise ValueError("Invalid provider candidate.")
            proposed = proposed.strip()
        analysis = japanese.analyze(proposed)
        estimated = timing.estimate(analysis, available)
        findings = rulebook.validate(en, proposed, en_analysis, original)
        rejected = []
        if proposed != original:
            # Existing uncertainty in the baseline remains reviewable. New
            # violations or numeric/polarity loss never enter the selected text.
            rejected = sorted(set(findings) - set(original_findings))
            if estimated["estimated_duration_seconds"] > baseline["estimated_duration_seconds"]:
                rejected.append("candidate_not_shorter")
            if analysis["warnings"]:
                rejected.append("candidate_reading_unverified")
        proposal = {"method": method, "text_ja": proposed, "reason": proposal_reason,
                    "rejected_reasons": rejected, "selected": not rejected}
        if rejected:
            proposed, analysis, estimated, findings = original, original_analysis, baseline, original_findings
            reasons.add("candidate_rejected")
        if proposed != original:
            reasons.add("rewrite_semantics_need_review")
        reasons.update(findings)
        reasons.update(analysis["warnings"])
        if estimated["timing_status"] == "overlong":
            reasons.add("duration_exceeds_source")
        entry = {"id": f"dub-{index + 1:03d}", "source_utterance_ids": ids,
                 "speaker": group[0]["speaker"], "speaker_confidence": "estimated" if group[0]["speaker"] not in (None, "UNKNOWN", "REVIEW") else "unresolved",
                 "start": round(clip["start"] + start, 3), "end": round(clip["start"] + end, 3),
                 "relative_start": start, "relative_end": end,
                 "text_en": en, "original_text_ja": original, "text_ja": proposed,
                 "english_analysis": en_analysis, "japanese_analysis": analysis,
                 "original_japanese_analysis": original_analysis,
                 "original_timing": baseline, **estimated,
                 "estimated_playback_end": round(start + estimated["estimated_duration_seconds"], 3),
                 "proposal": proposal, "validation": {"semantic_equivalence": "not_proven",
                 "original_findings": original_findings, "findings": findings},
                 "review_reasons": sorted(reasons),
                 "provenance": [{"source": u, "translation": translated[u["id"]]} for u in group],
                 "overlapping_entry_ids": [], "same_speaker_conflict_ids": []}
        entries.append(entry)
    # Independent timelines: detect both source overlap and estimated self-collision,
    # without ever changing another entry's end or deleting a speaker's turn.
    for i, first in enumerate(entries):
        for second in entries[i + 1:]:
            overlap = min(first["relative_end"], second["relative_end"]) > max(first["relative_start"], second["relative_start"])
            if overlap:
                first["overlapping_entry_ids"].append(second["id"])
                second["overlapping_entry_ids"].append(first["id"])
                first["review_reasons"].append("source_overlap")
                second["review_reasons"].append("source_overlap")
            if (first["speaker"] not in (None, "UNKNOWN", "REVIEW") and first["speaker"] == second["speaker"] and
                    first["estimated_playback_end"] > second["relative_start"]):
                first["same_speaker_conflict_ids"].append(second["id"])
                second["same_speaker_conflict_ids"].append(first["id"])
                first["review_reasons"].append("same_speaker_timing_conflict")
                second["review_reasons"].append("same_speaker_timing_conflict")
    timelines = defaultdict(list)
    for entry in entries:
        entry["review_reasons"] = sorted(set(entry["review_reasons"]))
        entry["review_required"] = bool(entry["review_reasons"])
        # Unknown entries are not assumed to share an identity with each other.
        key = entry["speaker"] if entry["speaker_confidence"] == "estimated" else "UNRESOLVED:" + entry["id"]
        entry["timeline_id"] = key
        timelines[key].append(entry["id"])
    recovered = [uid for entry in entries for uid in entry["source_utterance_ids"]]
    if Counter(recovered) != Counter(u["id"] for u in included):
        raise RuntimeError("Source coverage invariant violated.")
    original_total = sum(e["original_timing"]["estimated_duration_seconds"] for e in entries)
    optimized_total = sum(e["estimated_duration_seconds"] for e in entries)
    config = {"engine_version": ENGINE_VERSION, "english": english.identity, "japanese": japanese.identity,
              "rules": rulebook.data, "timing": asdict(timing.config), "seconds": seconds,
              "candidates": candidates, "provider": provider.identity if provider else None}
    return {"format_version": FORMAT_VERSION, "engine_version": ENGINE_VERSION,
            "video_id": source.get("video_id"), "clip": clip,
            "preview": {"relative_start": 0, "relative_end": limit, "boundary_policy": "exclude_crossing_utterance"},
            "input_fingerprint": input_hash, "configuration": config,
            "cache_key": fingerprint({"input": input_hash, "config": config, "entries": entries}),
            "entries": entries, "speaker_timelines": dict(timelines), "excluded": excluded,
            "summary": {"included_source_count": len(recovered), "entry_count": len(entries),
                        "changed_entry_count": sum(e["text_ja"] != e["original_text_ja"] for e in entries),
                        "review_count": sum(e["review_required"] for e in entries),
                        "overlong_count": sum(e["timing_status"] == "overlong" for e in entries),
                        "original_estimated_speech_seconds": round(original_total, 3),
                        "optimized_estimated_speech_seconds": round(optimized_total, 3),
                        "estimated_speech_reduction_seconds": round(original_total - optimized_total, 3)},
            "limitations": ["Timing is estimated, not synthesized WAV duration.",
                            "Summed speech seconds are not wall-clock runtime when turns overlap.",
                            "Lexical checks cannot prove semantic equivalence or detect every hallucination.",
                            "UNKNOWN and upstream estimates remain unresolved; no audio is truncated."]}


def comparison_markdown(result):
    lines = ["# 吹き替え台本比較", "", f"動画: {result['video_id']} / 冒頭 {result['preview']['relative_end']} 秒", "",
             "時間は読み・モーラ・ポーズによる推定値です。意味の同一性と音声品質は未保証です。", "",
             "| ID / 話者 / 元ID | 英語 | 既存訳 | 最適化案 | 元区間秒 | 既存推定秒 | 最適化推定秒 | 差（最適化−元区間） | 要確認 |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for e in result["entries"]:
        values = [f"{e['id']} / {e['speaker'] or 'UNKNOWN'} / {e['source_utterance_ids']}",
                  e["text_en"], e["original_text_ja"], e["text_ja"], e["source_duration_seconds"],
                  e["original_timing"]["estimated_duration_seconds"], e["estimated_duration_seconds"],
                  e["duration_delta_seconds"], ", ".join(e["review_reasons"]) or "なし"]
        lines.append("| " + " | ".join(escape(v) for v in values) + " |")
    lines += ["", "## 30秒境界をまたぐため対象外", ""]
    lines += [f"- 元ID {e['source_utterance_id']}: {e['relative_start']}〜{e['relative_end']}秒（切断なし）"
              for e in result["excluded"] if e["reason"] == "crosses_preview_boundary"]
    lines += ["", "## 集計", "", "```json", json.dumps(result["summary"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_result(path, result):
    path = Path(path)
    if path.exists():
        if read_json(path) == result:
            return False
        raise FileExistsError(f"Refusing to overwrite existing output: {path}. Choose a new --output.")
    atomic_json(path, result, overwrite=False)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", nargs="?")
    parser.add_argument("--input-dir", type=Path, help="Directory containing the two utterance JSON files")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--candidates", type=Path, help="Local candidate file bound to input_fingerprint")
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--english-model", default="en_core_web_sm")
    parser.add_argument("--mora-per-second", type=float, default=7.0)
    args = parser.parse_args(argv)
    try:
        if args.input_dir is None and not args.video_id:
            raise ValueError("Provide video_id or --input-dir.")
        if args.video_id and (Path(args.video_id).name != args.video_id or args.video_id in {".", ".."}):
            raise ValueError("video_id must be a single directory name.")
        directory = args.input_dir or BASE_DIR / "input" / args.video_id
        source, translations = read_json(directory / "utterances.json"), read_json(directory / "utterance_translations.json")
        if args.video_id and source.get("video_id") != args.video_id:
            raise ValueError("video_id does not match input data.")
        rules = Rulebook(read_json(args.rules) if args.rules else None)
        selection = read_json(directory / "selection.json") if (directory / "selection.json").exists() else None
        result = optimize(source, translations, english=EnglishAnalyzer(args.english_model),
                          japanese=JapaneseAnalyzer(rules.data["pronunciations"]), rulebook=rules,
                          timing=TimingEngine(TimingConfig(mora_per_second=args.mora_per_second)),
                          seconds=args.seconds, selection=selection,
                          candidates=read_json(args.candidates) if args.candidates else None)
        output = args.output or directory / "dubbing_script.json"
        # Validate all destinations before writing either file.
        if args.report and args.report.resolve() == output.resolve():
            raise ValueError("Report and JSON destinations must differ.")
        report = comparison_markdown(result)
        if args.report and args.report.exists() and args.report.read_text() != report:
            raise FileExistsError("Refusing to overwrite an existing comparison report.")
        created = save_result(output, result)
        if args.report and not args.report.exists():
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(report)
        print(report)
        print(f"{'Saved' if created else 'Reused identical output'}: {output}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
