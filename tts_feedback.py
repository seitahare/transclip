"""TTS feedback orchestration, independent of MLX, NLP models and network APIs."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from language.cache import atomic_json, fingerprint
from language.timing import measure_wav, feedback_profiles
from language.voice import VoiceProfile, synthesize_profile

FORMAT_VERSION = 1
UNKNOWN = (None, "UNKNOWN", "REVIEW", "MIXED")


def number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {label}.")
    return value


def validate_script(script):
    if not isinstance(script, dict) or script.get("format_version") != 1:
        raise ValueError("Expected dubbing_script format_version 1.")
    if not isinstance(script.get("video_id"), str) or not script["video_id"]:
        raise ValueError("Missing video ID.")
    clip = script.get("clip", {})
    start, end = number(clip.get("start"), "clip start"), number(clip.get("end"), "clip end")
    if start >= end:
        raise ValueError("Invalid clip range.")
    entries = script.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty dubbing script.")
    ids, source_ids = set(), set()
    for e in entries:
        if not isinstance(e, dict) or not isinstance(e.get("id"), str) or not e["id"] or e["id"] in ids:
            raise ValueError("Duplicate or invalid entry ID.")
        ids.add(e["id"])
        if "speaker" not in e or (e["speaker"] is not None and not isinstance(e["speaker"], str)):
            raise ValueError("Missing or invalid speaker.")
        for key in ("text_en", "text_ja"):
            if not isinstance(e.get(key), str) or not e[key].strip():
                raise ValueError(f"Empty {key} for {e['id']}.")
        refs = e.get("source_utterance_ids")
        if (not isinstance(refs, list) or not refs or any(type(x) is not int for x in refs)
                or len(refs) != len(set(refs)) or source_ids.intersection(refs)):
            raise ValueError("Duplicate or invalid source IDs.")
        source_ids.update(refs)
        a = number(e.get("relative_start"), "relative start")
        b = number(e.get("relative_end"), "relative end")
        if not a < b <= end - start + .01:
            raise ValueError("Invalid utterance interval.")
        for key, relative in (("start", a), ("end", b)):
            if abs(number(e.get(key), key) - start - relative) > .01:
                raise ValueError("Absolute/relative time mismatch.")
        if abs(number(e.get("source_duration_seconds"), "source duration", positive=True) - (b-a)) > .01:
            raise ValueError("Source duration mismatch.")
        number(e.get("estimated_duration_seconds"), "estimated duration", positive=True)
        number(e.get("pause_seconds"), "pause seconds")
        analysis = e.get("japanese_analysis", {})
        if type(analysis.get("mora_count")) is not int or analysis["mora_count"] < 0:
            raise ValueError("Missing mora count.")
        if not isinstance(analysis.get("pauses"), list) or not isinstance(analysis.get("reading"), str):
            raise ValueError("Missing reading/pause features.")
        if not isinstance(e.get("review_reasons"), list) or any(not isinstance(x, str) for x in e["review_reasons"]):
            raise ValueError("Invalid review reasons.")
        if type(e.get("review_required")) is not bool:
            raise ValueError("Invalid review_required.")
    preview = script.get("preview", {})
    if preview.get("relative_start") != 0:
        raise ValueError("Only previews beginning at clip start are supported.")
    limit = number(preview.get("relative_end"), "preview end", positive=True)
    if limit > end - start + .01:
        raise ValueError("Preview exceeds clip.")
    if script.get("cache_key") is not None:
        expected = fingerprint({"input": script.get("input_fingerprint"),
                                "config": script.get("configuration"), "entries": entries})
        if script["cache_key"] != expected:
            raise ValueError("Script content/analysis hash mismatch; regenerate via dubbing_script.py.")
    # JSON serialization also rejects NaN hidden in settings/provenance.
    fingerprint(script)
    return entries


def plan_script(script, voices, *, seconds=30, entry_ids=None):
    entries = validate_script(script)
    number(seconds, "seconds", positive=True)
    limit = min(seconds, script["preview"]["relative_end"])
    if entry_ids and (len(entry_ids) != len(set(entry_ids)) or set(entry_ids) - {e["id"] for e in entries}):
        raise ValueError("Unknown or duplicate --entry-id.")
    if not isinstance(voices, dict):
        raise ValueError("Voice map must be a JSON object.")
    for speaker, voice in voices.items():
        if speaker in UNKNOWN or not re.fullmatch(r"[A-Za-z0-9_-]+", speaker) or not isinstance(voice, str) or not voice.strip():
            raise ValueError("Voice map must explicitly name safe, known speaker IDs and presets.")
    rows = []
    for entry in entries:
        e = copy.deepcopy(entry)
        reason = None
        if entry_ids and e["id"] not in entry_ids:
            reason = "not_requested"
        elif e["relative_start"] >= limit:
            reason = "outside_preview"
        elif e["relative_end"] > limit:
            reason = "crosses_preview_boundary"
        elif e["speaker"] in UNKNOWN:
            reason = "unknown_speaker"
        elif e["speaker"] not in voices:
            reason = "no_preset_mapping"
        e.update({"generation_status": "skipped" if reason else "pending",
                  "skip_reason": reason, "voice": voices.get(e["speaker"]),
                  "actual_tts_duration_seconds": None, "audio_file": None,
                  "prediction_error_seconds": None, "actual_minus_source_seconds": None,
                  "actual_playback_end": None, "actual_timing_status": "unmeasured",
                  "actual_same_speaker_conflicts": [], "actual_other_speaker_overlaps": [],
                  "mora_count": e["japanese_analysis"]["mora_count"],
                  "reading": e["japanese_analysis"]["reading"],
                  "punctuation_pauses": e["japanese_analysis"]["pauses"],
                  "analyzer_identity": script.get("configuration", {}).get("japanese"),
                  "timing_config": script.get("configuration", {}).get("timing"),
                  "tts": None, "audio_cache_key": None})
        if reason:
            e["review_reasons"] = sorted(set(e["review_reasons"] + [reason]))
            e["review_required"] = True
        rows.append(e)
    return rows


def audio_request(row, identity, *, validate_live=True):
    """Shared receipt contract for feedback and mixer; preserve legacy preset keys."""
    snapshot=row.get('voice_profile')
    if snapshot is None:
        if row.get('voice_mode','preset')!='preset':raise ValueError('Clone requires Voice Profile provenance.')
        return {'format_version':1,'tts':identity,'voice':row['voice'],'text_ja':row['text_ja'],'mode':'preset'}
    from voice_cloning import validate_snapshot
    validate_snapshot(snapshot,row['speaker'],live=validate_live)
    if row.get('voice_mode')!='approved_clone' or row.get('voice')!='approved_clone:'+snapshot['profile_fingerprint']:
        raise ValueError('Voice mode/label and profile mismatch.')
    return {'format_version':2,'tts':identity,'voice':row['voice'],'speaker_id':row['speaker'],
            'text_ja':row['text_ja'],'mode':'approved_clone','voice_profile':snapshot}


def cached_audio(output_dir, backend, row):
    """Publish audio + receipt as one directory rename; never overwrite a WAV."""
    request = audio_request(row, backend.identity)
    key = fingerprint(request)
    parent = output_dir / "audio" / row["speaker"]
    target = parent / key
    if target.exists():
        receipt = json.loads((target / "receipt.json").read_text())
        if receipt.get("request") != request or receipt.get("cache_key") != key:
            raise ValueError("TTS cache receipt mismatch.")
        measured = measure_wav(target / "speech.wav")
        if measured != receipt.get("measurement"):
            raise ValueError("TTS cache WAV changed or is corrupt; original file retained.")
        return target / "speech.wav", measured, receipt["backend_result"], key, True
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        profile=VoiceProfile(row['speaker'],voice=row['voice'])
        if row.get('voice_profile'):
            from voice_cloning import validate_snapshot
            profile=validate_snapshot(row['voice_profile'],row['speaker'])
        generated = synthesize_profile(backend,text=row['text_ja'],profile=profile,output_path=str(temporary/'speech.wav'))
        if audio_request(row,backend.identity)!=request:raise ValueError('Profile or permission changed during generation.')
        measured = measure_wav(temporary / "speech.wav")
        atomic_json(temporary / "receipt.json", {"cache_key": key, "request": request,
                    "measurement": measured, "backend_result": generated}, overwrite=False)
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this call's private, unpublished files.
    return target / "speech.wav", measured, generated, key, False


def detect_collisions(rows):
    """Use unmodified starts and full measured durations on independent tracks."""
    active = [e for e in rows if e["generation_status"] == "complete"]
    for e in rows:
        e["actual_same_speaker_conflicts"] = []
        e["actual_other_speaker_overlaps"] = []
        e["review_reasons"] = [r for r in e["review_reasons"] if r != "actual_same_speaker_collision"]
    for i, a in enumerate(active):
        for b in active[i+1:]:
            overlap = min(a["actual_playback_end"], b["actual_playback_end"]) - max(a["relative_start"], b["relative_start"])
            if overlap <= 1e-9:
                continue
            same = a["speaker"] == b["speaker"] and a["speaker"] not in UNKNOWN
            field = "actual_same_speaker_conflicts" if same else "actual_other_speaker_overlaps"
            source_overlap = max(0, min(a["relative_end"], b["relative_end"]) - max(a["relative_start"], b["relative_start"]))
            for first, second in ((a,b),(b,a)):
                first[field].append({"entry_id": second["id"], "overlap_seconds": round(overlap, 6),
                                     "source_overlap_seconds": round(source_overlap, 6)})
                if same:
                    first["review_reasons"].append("actual_same_speaker_collision")
                    first["review_required"] = True
    for e in rows:
        e["review_reasons"] = sorted(set(e["review_reasons"]))


def compare_baseline(script, rows, manifest_path):
    """Read-only comparison to the old text/WAV, never reuse it as new audio."""
    path = Path(manifest_path)
    old = json.loads(path.read_text())
    if old.get("video_id") != script["video_id"]:
        raise ValueError("Baseline video mismatch.")
    if old.get("clip") is not None and old["clip"] != script["clip"]:
        raise ValueError("Baseline clip mismatch.")
    comparisons = []
    for row in rows:
        matching = [j for j in old.get("jobs", []) if j.get("utterance_ids") == row["source_utterance_ids"]
                    and j.get("speaker") == row["speaker"] and j.get("text_ja") == row.get("original_text_ja")
                    and abs(j.get("start", -1) - row["relative_start"]) < .01
                    and abs(j.get("end", -1) - row["relative_end"]) < .01]
        if len(matching) > 1:
            raise ValueError("Ambiguous baseline match.")
        if not matching:
            continue
        job = matching[0]
        wav = (path.parent / job["audio_file"]).resolve()
        if not wav.is_relative_to(path.parent.resolve()):
            raise ValueError("Baseline audio path escapes manifest directory.")
        measurement = measure_wav(wav)
        actual = row["actual_tts_duration_seconds"]
        comparisons.append({"entry_id": row["id"], "source_utterance_ids": row["source_utterance_ids"],
                            "original_text_ja": job["text_ja"], "optimized_text_ja": row["text_ja"],
                            "baseline_audio_file": str(wav), "baseline_model": old.get("model"),
                            "baseline_voice": job.get("voice"), "baseline_measurement": measurement,
                            "optimized_actual_seconds": actual,
                            "reduction_seconds": measurement["audio_duration"] - actual if actual is not None else None,
                            "same_model_and_voice": old.get("model") == (row.get("tts") or {}).get("model") and job.get("voice") == row["voice"],
                            "settings_comparable": False,  # Legacy manifest omitted generation parameters/seed.
                            "provenance": "legacy_exact_text_ids_speaker_relative_times" if "clip" not in old else "clip_and_content_match"})
    return comparisons


def feedback_markdown(report):
    lines = ["# TTS実測フィードバック", "", f"動画: {report['video_id']} / 状態: {report['status']}", "",
             "誤差＝実測−推定、元との差＝実測−英語区間。音声は切断・倍速化していません。", "",
             "| 発話 / 話者 | 日本語 | 元区間秒 | 推定秒 | 実測秒 | 誤差秒 | 元との差秒 | 状態 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    def cell(value):
        if value is None: return "—"
        if isinstance(value, float): return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    for r in report["entries"]:
        values = [f"{r['id']} / {r['speaker'] or 'UNKNOWN'}", r['text_ja'], r['source_duration_seconds'],
                  r['estimated_duration_seconds'], r['actual_tts_duration_seconds'], r['prediction_error_seconds'],
                  r['actual_minus_source_seconds'], r['generation_status'] + (": " + r['skip_reason'] if r['skip_reason'] else '')]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    for c in report.get("baseline_comparisons", []):
        lines += ["", f"- {c['entry_id']}: 旧WAV {cell(c['baseline_measurement']['audio_duration'])}秒 → 新WAV {cell(c['optimized_actual_seconds'])}秒。短縮 {cell(c['reduction_seconds'])}秒。旧生成設定が不明なため厳密な条件統制比較ではありません。"]
    lines += ["", "レビュー情報・実測衝突・校正標本はfeedback.jsonに保存しています。校正値は自動適用しません。", ""]
    return "\n".join(lines)


def run_feedback(script_path, output_dir, backend, voices, *, seconds=30,
                 max_jobs=None, entry_ids=None, baseline_manifest=None, voice_profiles=None):
    script_path, output_dir = Path(script_path), Path(output_dir)
    original_bytes = script_path.read_bytes()
    script = json.loads(original_bytes)
    rows = plan_script(script, voices, seconds=seconds, entry_ids=entry_ids)
    snapshots={}
    if voice_profiles is not None:
        from voice_cloning import profile_snapshot
        if baseline_manifest:raise ValueError('Legacy preset baseline cannot be mislabeled as clone; compare separate reports.')
        for speaker,profile in voice_profiles.items():
            if profile.speaker!=speaker:raise ValueError('Profile map speaker mismatch.')
            snapshots[speaker]=profile_snapshot(profile)
            if voices.get(speaker)!='approved_clone:'+snapshots[speaker]['profile_fingerprint']:raise ValueError('Profile voice label mismatch.')
        for row in rows:
            if row['speaker'] in snapshots:
                row.update(voice_mode='approved_clone',voice_profile=snapshots[row['speaker']])
                row['review_required']=True
                row['review_reasons']=sorted(set(row['review_reasons']+['clone_pronunciation_and_voice_similarity_unverified']))
            elif row['generation_status']=='pending':raise ValueError('Missing clone profile for planned speaker.')
    elif getattr(backend,'supported_voice_modes',{'preset'})=={'approved_clone'}:
        raise ValueError('Clone backend requires explicit voice_profiles.')
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs <= 0):
        raise ValueError("max_jobs must be positive.")
    identity = copy.deepcopy(backend.identity)
    request = {"format_version": 1, "script": script, "tts": identity, "voices": voices,
               "seconds": seconds, "entry_ids": sorted(entry_ids) if entry_ids else None,
               "baseline": fingerprint(json.loads(Path(baseline_manifest).read_text())) if baseline_manifest else None}
    if snapshots:request['voice_profiles']=snapshots
    run_key = fingerprint(request)
    run_dir = output_dir / "runs" / run_key
    report = {"format_version": FORMAT_VERSION, "kind": "tts_feedback", "run_key": run_key,
              "video_id": script["video_id"], "clip": script["clip"],
              "preview": {**script["preview"], "relative_end": min(seconds, script["preview"]["relative_end"])},
              "speaker_timelines": script.get("speaker_timelines", {}),
              "source_script_fingerprint": fingerprint(script), "source_script_path": str(script_path.resolve()),
              "tts": identity, "status": "running", "entries": rows,
              "source_excluded": script.get("excluded", []), "baseline_comparisons": [],
              "calibration_profiles": [], "calibration_auto_applied": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".feedback.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another feedback run is using this output directory.") from error
        run_dir.mkdir(parents=True, exist_ok=True)

        def save():
            detect_collisions(rows)
            report["calibration_profiles"] = feedback_profiles(rows)
            if baseline_manifest:
                report["baseline_comparisons"] = compare_baseline(script, rows, baseline_manifest)
            report["summary"] = {status: sum(r["generation_status"] == status for r in rows)
                                 for status in ("complete", "pending", "skipped", "failed")}
            report["summary"]["overlong"] = sum(r["actual_timing_status"] == "overlong" for r in rows)
            atomic_json(run_dir / "feedback.json", report)
            (run_dir / "comparison.md").write_text(feedback_markdown(report), encoding="utf-8")

        save()  # Baseline and input errors surface before loading TTS.
        attempted = 0
        for row in rows:
            if row["generation_status"] == "skipped":
                continue
            if max_jobs is not None and attempted >= max_jobs:
                break
            row["tts"] = identity
            try:
                print(f"TTS {row['id']} / {row['speaker']} / {row['voice']}", flush=True)
                wav, measured, details, key, reused = cached_audio(output_dir, backend, row)
                if not reused:
                    attempted += 1
                actual = measured["audio_duration"]
                tolerance = (row.get("timing_config") or {}).get("tolerance_seconds", .25)
                row.update({"generation_status": "complete", "audio_file": str(wav.resolve()),
                            "audio_cache_key": key, "cache_reused": reused, "wav_measurement": measured,
                            "backend_result": details, "actual_tts_duration_seconds": actual,
                            "prediction_error_seconds": actual - row["estimated_duration_seconds"],
                            "actual_minus_source_seconds": actual - row["source_duration_seconds"],
                            "actual_playback_end": row["relative_start"] + actual,
                            "actual_timing_status": "overlong" if actual - row["source_duration_seconds"] > tolerance else "measured_fit",
                            "audio_quality_warnings": ["clipping_detected"] if measured["clipped_sample_fraction"] > .001 else []})
                row["review_reasons"] += row["audio_quality_warnings"]
                if row["actual_timing_status"] == "overlong":
                    row["review_reasons"].append("actual_duration_exceeds_source")
                row["review_required"] = bool(row["review_reasons"])
                print(f"  {'cached' if reused else 'generated'}: {actual:.3f}s; estimate error {row['prediction_error_seconds']:+.3f}s", flush=True)
                save()
            except Exception as error:
                row["generation_status"] = "failed"
                row["error"] = {"type": type(error).__name__, "message": str(error)}
                row["review_required"] = True
                row["review_reasons"].append("tts_generation_or_cache_failed")
                report["status"] = "failed"
                save()
                raise
        report["status"] = "partial" if any(r["generation_status"] == "pending" for r in rows) else "complete_with_skips" if any(r["generation_status"] == "skipped" for r in rows) else "complete"
        save()
        if script_path.read_bytes() != original_bytes:
            raise RuntimeError("Source script changed during generation; it was not modified by this runner.")
        print(f"Feedback: {run_dir / 'feedback.json'}", flush=True)
    return run_dir / "feedback.json"
