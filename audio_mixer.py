"""Read-only Phase 2 inputs -> independent speaker timelines -> verified MP4.

No NLP/TTS runtime is loaded. Rendering uses FFmpeg and Python's standard library.
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

from language.cache import atomic_json, fingerprint
from language.timing import measure_wav
from tts_feedback import UNKNOWN, detect_collisions, number, validate_script, audio_request

BASE_DIR = Path(__file__).resolve().parent
RATE = 48000
CHANNELS = 2
ENGINE_VERSION = 1
PEAK_TARGET = 10 ** (-2 / 20)


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
               mode="japanese-only", diagnostic=False, extend_to_fit=False, reference_gain=.12):
    number(seconds, "seconds", positive=True)
    number(reference_gain, "reference gain")
    if reference_gain > .5 or mode not in ("japanese-only", "reference-mix"):
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
    if mode == "reference-mix" and not any(s.get("codec_type") == "audio" for s in video_info["streams"]):
        raise ValueError("Reference Mix requires a source audio stream.")
    if seconds > feedback["preview"]["relative_end"] or seconds > selection["end"] - selection["start"]:
        raise ValueError("Requested preview exceeds feedback/selection coverage.")
    active = [r for r in rows if r["relative_start"] < seconds]
    complete = [r for r in active if r["generation_status"] == "complete"]
    missing = [r for r in active if r["generation_status"] != "complete"]
    # Round only to the nearest output sample; preserve full resampled lengths.
    for row in complete:
        row["placement_sample"] = round(row["relative_start"] * RATE)
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
    detect_collisions(active)
    same = overlap_pairs(active, "actual_same_speaker_conflicts")
    other = overlap_pairs(active, "actual_other_speaker_overlaps")
    excluded = [r for r in feedback.get("source_excluded", []) if r.get("relative_start", math.inf) < duration]
    review_ids = [r["id"] for r in active if r["review_required"]]
    if not complete:
        blockers.append("no_generated_audio")
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
    return {"format_version": 1, "kind": "audio_mix", "engine_version": ENGINE_VERSION,
            "video_id": video_id, "clip": feedback["clip"], "preview_seconds_requested": seconds,
            "duration_seconds": duration, "absolute_video_start": selection["start"],
            "absolute_video_end": selection["start"] + duration,
            "mode": mode, "diagnostic": diagnostic, "quality_label": quality,
            "reference_gain": reference_gain if mode == "reference-mix" else 0,
            "source_audio_policy": "quiet English reference; no speech separation" if mode == "reference-mix" else "muted",
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
            if plan["mode"] == "reference-mix":
                data = ffmpeg(["-ss", str(plan["absolute_video_start"]), "-i", source,
                               "-t", str(plan["duration_seconds"]), "-map", "0:a:0", "-vn",
                               "-af", f"aresample={RATE},apad=whole_len={plan['sample_frames']}",
                               "-ar", str(RATE), "-ac", str(CHANNELS), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
                reference = float_samples(data)
                if len(reference) != len(japanese): raise ValueError("Reference track duration mismatch.")
                reference = array("f", (v * plan["reference_gain"] for v in reference))
                final = add_audio([japanese, reference])
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
    parser.add_argument("--mode", choices=("japanese-only", "reference-mix"), default="japanese-only")
    parser.add_argument("--reference-gain", type=float, default=.12)
    parser.add_argument("--diagnostic", action="store_true", help="Explicitly allow reported collisions and undubbed/review entries")
    parser.add_argument("--extend-to-fit", action="store_true", help="Extend only within selection to keep complete speech")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan; generate no media")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.video_id): raise ValueError("Invalid video_id.")
        inputs = BASE_DIR / "input" / args.video_id
        plan = build_plan(args.feedback, args.selection or inputs / "selection.json",
                          args.source_video or inputs / "source.mp4", args.source_info or inputs / "source.info.json",
                          args.video_id, seconds=args.seconds, mode=args.mode, diagnostic=args.diagnostic,
                          extend_to_fit=args.extend_to_fit, reference_gain=args.reference_gain)
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
