"""Phase 10: Japanese Subtitle Engine synchronized to post-TTS speech placement.

Generates SRT, multi-speaker ASS, and FFmpeg subtitle-burned MP4s strictly from
the final synthesized TTS script (`text_ja` and measured WAV placement), never
from pre-TTS raw translations.
"""
import argparse
from dataclasses import asdict, dataclass, field
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata

from audio_mixer import ffmpeg, probe, read_json, sha256_file, validate_feedback, validate_mp4
from language.cache import atomic_json, fingerprint
from language.pronunciation import count_mora
from tts_feedback import UNKNOWN, detect_collisions, number

CLAUSE_BREAK_CHARS = set("。！？!?、，,…")
PARTICLE_SURFACES = {"は", "が", "を", "に", "で", "と", "から", "まで", "より", "へ", "や", "て", "ば", "たら", "なら", "ので", "のに", "けれど", "けど", "し"}
DEFAULT_SPEAKER_COLORS = [
    "&H00FFFFFF",  # White
    "&H00A8F0FF",  # Warm light yellow/amber (ASS BGR)
    "&H00FFD8A8",  # Soft sky blue (ASS BGR)
    "&H00C8FFC8",  # Soft mint green (ASS BGR)
]


@dataclass(frozen=True)
class SubtitleStyleConfig:
    font_name: str = "Noto Sans CJK JP"
    font_size: int = 42
    max_chars_per_line: int = 22
    max_lines: int = 2
    outline: float = 2.5
    shadow: float = 1.0
    alignment: int = 2
    margin_l: int = 40
    margin_r: int = 40
    margin_v: int = 36
    overlap_lane_step_v: int = 92
    play_res_x: int = 1920
    play_res_y: int = 1080
    speaker_colors: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.font_name, str) or not self.font_name.strip():
            raise ValueError("Invalid font_name.")
        if type(self.font_size) is not int or not 12 <= self.font_size <= 120:
            raise ValueError("Invalid font_size.")
        if type(self.max_chars_per_line) is not int or not 8 <= self.max_chars_per_line <= 60:
            raise ValueError("Invalid max_chars_per_line.")
        if type(self.max_lines) is not int or not 1 <= self.max_lines <= 4:
            raise ValueError("Invalid max_lines.")
        if type(self.alignment) is not int or not 1 <= self.alignment <= 9:
            raise ValueError("Invalid ASS alignment (1..9).")
        for name, val in [("outline", self.outline), ("shadow", self.shadow)]:
            if type(val) not in (int, float) or not math.isfinite(val) or val < 0 or val > 12:
                raise ValueError(f"Invalid {name}.")
        for name, val in [("margin_l", self.margin_l), ("margin_r", self.margin_r), ("margin_v", self.margin_v), ("overlap_lane_step_v", self.overlap_lane_step_v)]:
            if type(val) is not int or val < 0 or val > 600:
                raise ValueError(f"Invalid {name}.")


def format_srt_timestamp(seconds):
    ms = max(0, round(float(seconds) * 1000))
    hours, rem = divmod(ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_timestamp(seconds):
    cs = max(0, round(float(seconds) * 100))
    hours, rem = divmod(cs, 360_000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _normalize_for_comparison(text):
    norm = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s、。！？!?,.…「」『』（）()・]", "", norm)


def split_japanese_clauses(text, japanese_analysis=None):
    """Split Japanese text into natural bunsetsu/clause units without losing any characters."""
    text = text.strip()
    if not text:
        raise ValueError("Cannot split empty subtitle text.")

    tokens = (japanese_analysis or {}).get("tokens")
    if isinstance(tokens, list) and tokens and all(isinstance(t, dict) and "surface" in t for t in tokens):
        reconstructed = "".join(t["surface"] for t in tokens)
        if _normalize_for_comparison(reconstructed) == _normalize_for_comparison(text):
            units = []
            current = ""
            for tok in tokens:
                surf = tok["surface"]
                pos = tok.get("pos") or []
                current += surf
                is_punct = any(ch in CLAUSE_BREAK_CHARS for ch in surf)
                is_particle = (pos and pos[0] == "助詞") or surf in PARTICLE_SURFACES
                if is_punct or (is_particle and len(current) >= 4):
                    units.append(current)
                    current = ""
            if current:
                if units and len(current) <= 2:
                    units[-1] += current
                else:
                    units.append(current)
            return units

    # Regex fallback keeping numbers/English words intact and breaking after punctuation or common particles.
    pattern = re.compile(
        r"[0-9A-Za-z.,%]+(?:兆|億|万|千|百|十|回|秒|分|人|個|年|月|日)?|"
        r"[^。！？!?、，,…\s]+[。！？!?、，,…]*",
        re.UNICODE,
    )
    raw_chunks = [m.group(0) for m in pattern.finditer(text) if m.group(0)]
    if not raw_chunks:
        return [text]
    units = []
    for chunk in raw_chunks:
        # Further split long non-punctuated chunks at particle boundaries while keeping >=4 chars per unit.
        sub_parts = re.split(r"(?<=[はがをにでとへやも])(?![0-9A-Za-z。！？!?、，,…])", chunk)
        buf = ""
        for part in sub_parts:
            if not part:
                continue
            if len(buf) + len(part) <= 7:
                buf += part
            else:
                if buf:
                    units.append(buf)
                buf = part
        if buf:
            units.append(buf)
    return units or [text]


def wrap_clause_units(units, *, max_chars_per_line=22, max_lines=2):
    """Group clause units into cue blocks of at most max_lines lines and max_chars_per_line per line."""
    max_cue_chars = max_chars_per_line * max_lines
    cue_groups = []
    current_group = []
    current_len = 0

    for unit in units:
        u_len = len(unit)
        if current_group and current_len + u_len > max_cue_chars:
            cue_groups.append(current_group)
            current_group = [unit]
            current_len = u_len
        else:
            current_group.append(unit)
            current_len += u_len
    if current_group:
        cue_groups.append(current_group)

    formatted_cues = []
    for group in cue_groups:
        joined = "".join(group).strip()
        if len(joined) <= max_chars_per_line or max_lines == 1:
            lines = [joined]
        elif len(group) >= 2:
            # Choose clause split point that best balances the two lines.
            best_idx = 1
            best_score = math.inf
            for idx in range(1, len(group)):
                left = "".join(group[:idx]).strip()
                right = "".join(group[idx:]).strip()
                overflow = max(0, len(left) - max_chars_per_line) + max(0, len(right) - max_chars_per_line)
                score = overflow * 100 + abs(len(left) - len(right))
                if score < best_score:
                    best_score = score
                    best_idx = idx
            lines = ["".join(group[:best_idx]).strip(), "".join(group[best_idx:]).strip()]
        else:
            mid = min(max_chars_per_line, max(1, len(joined) // 2))
            lines = [joined[:mid].strip(), joined[mid:].strip()]
        lines = [line for line in lines if line]
        formatted_cues.append({
            "text": "".join(lines),
            "lines": lines[:max_lines] if len(lines) <= max_lines else lines,
            "weight": max(1, count_mora(joined) or len(joined)),
        })
    return formatted_cues


def align_entry_cues(entry, style: SubtitleStyleConfig, alignment_entry=None):
    """Create time-aligned subtitle cues for one completed TTS utterance."""
    start = float(entry.get("placement_seconds", entry["relative_start"]))
    duration = float(entry["actual_tts_duration_seconds"])
    end = start + duration
    text_ja = entry["text_ja"].strip()

    units = split_japanese_clauses(text_ja, entry.get("japanese_analysis"))
    cue_specs = wrap_clause_units(
        units,
        max_chars_per_line=style.max_chars_per_line,
        max_lines=style.max_lines,
    )

    warnings = []
    timing_source = "tts_measured_proportional"
    if alignment_entry is not None:
        recognized_text = alignment_entry.get("recognized_text", "")
        if recognized_text:
            if _normalize_for_comparison(recognized_text) != _normalize_for_comparison(text_ja):
                warnings.append("asr_transcript_differs_from_tts_script")
        segments = alignment_entry.get("segments")
        if isinstance(segments, list) and len(segments) == len(cue_specs):
            valid_segs = True
            for seg in segments:
                s_a, s_b = seg.get("relative_start"), seg.get("relative_end")
                if type(s_a) not in (int, float) or type(s_b) not in (int, float) or not (start - 0.05 <= s_a < s_b <= end + 0.05):
                    valid_segs = False
                    break
            if valid_segs:
                timing_source = "forced_or_whisper_alignment"
                cues = []
                for idx, (spec, seg) in enumerate(zip(cue_specs, segments)):
                    cues.append({
                        "entry_id": entry["id"],
                        "cue_index": idx,
                        "speaker": entry["speaker"],
                        "start": round(max(start, float(seg["relative_start"])), 4),
                        "end": round(min(end, float(seg["relative_end"])), 4),
                        "text_ja": spec["text"],
                        "lines": spec["lines"],
                        "timing_source": timing_source,
                        "alignment_warnings": list(warnings),
                    })
                return cues, warnings
        elif segments is not None:
            warnings.append("alignment_segment_count_mismatch_used_proportional")

    total_weight = sum(c["weight"] for c in cue_specs)
    cursor = start
    cues = []
    for idx, spec in enumerate(cue_specs):
        if idx == len(cue_specs) - 1:
            cue_end = end
        else:
            share = duration * (spec["weight"] / total_weight)
            cue_end = min(end, cursor + share)
        cues.append({
            "entry_id": entry["id"],
            "cue_index": idx,
            "speaker": entry["speaker"],
            "start": round(cursor, 4),
            "end": round(cue_end, 4),
            "text_ja": spec["text"],
            "lines": spec["lines"],
            "timing_source": timing_source,
            "alignment_warnings": list(warnings),
        })
        cursor = cue_end
    return cues, warnings


def build_subtitle_plan(
    feedback_or_mix_path,
    selection_path,
    video_id,
    *,
    seconds=30.0,
    style=None,
    alignment_data=None,
    diagnostic=False,
):
    """Build a post-TTS synchronized Japanese subtitle plan from feedback.json or mix_report.json."""
    style = style or SubtitleStyleConfig()
    number(seconds, "seconds", positive=True)
    doc = read_json(feedback_or_mix_path)
    if doc.get("kind") == "audio_mix" and doc.get("format_version") == 1:
        feedback_path = Path(doc["inputs"]["feedback"]).resolve()
        _, rows, selection, script_path = validate_feedback(feedback_path, selection_path, video_id)
        placed_by_id = {r["id"]: r for r in doc.get("placements", [])}
        for r in rows:
            if r["id"] in placed_by_id:
                r["placement_seconds"] = placed_by_id[r["id"]]["placement_seconds"]
        duration_limit = float(doc.get("duration_seconds", seconds))
    else:
        _, rows, selection, script_path = validate_feedback(feedback_or_mix_path, selection_path, video_id)
        duration_limit = float(seconds)

    active = [r for r in rows if r["relative_start"] < seconds]
    complete = [r for r in active if r["generation_status"] == "complete"]
    missing = [r for r in active if r["generation_status"] != "complete"]

    for r in complete:
        if "placement_seconds" not in r or r["placement_seconds"] is None:
            r["placement_seconds"] = r["relative_start"]

    detect_collisions(active, use_placements=True)
    same_conflicts = [
        {"entry_ids": [r["id"], c["entry_id"]], "speaker": r["speaker"], "overlap_seconds": c["overlap_seconds"]}
        for r in active
        for c in r["actual_same_speaker_conflicts"]
        if r["id"] < c["entry_id"]
    ]
    other_overlaps = [
        {"entry_ids": [r["id"], c["entry_id"]], "overlap_seconds": c["overlap_seconds"], "source_overlap_seconds": c["source_overlap_seconds"]}
        for r in active
        for c in r["actual_other_speaker_overlaps"]
        if r["id"] < c["entry_id"]
    ]

    blockers = []
    if not complete:
        blockers.append("no_generated_tts_audio")
    if same_conflicts and not diagnostic:
        blockers.append("unresolved_same_speaker_collisions")
    if missing and not diagnostic:
        blockers.append("undubbed_entries_present")

    all_cues = []
    alignment_warnings = {}
    for entry in sorted(complete, key=lambda x: (x["placement_seconds"], x["id"])):
        align_item = (alignment_data or {}).get(entry["id"])
        entry_cues, warns = align_entry_cues(entry, style, align_item)
        if warns:
            alignment_warnings[entry["id"]] = warns
        all_cues.extend(entry_cues)

    all_cues.sort(key=lambda c: (c["start"], c["end"], c["speaker"], c["entry_id"], c["cue_index"]))

    # Assign simultaneous overlap lanes so different speakers overlapping in time do not collide visually.
    lane_ends = []
    for cue in all_cues:
        assigned_lane = None
        for lane_idx, last_end in enumerate(lane_ends):
            if cue["start"] >= last_end - 1e-6:
                assigned_lane = lane_idx
                lane_ends[lane_idx] = cue["end"]
                break
        if assigned_lane is None:
            assigned_lane = len(lane_ends)
            lane_ends.append(cue["end"])
        cue["lane"] = assigned_lane
        cue["simultaneous_overlap"] = False

    for i, a in enumerate(all_cues):
        for b in all_cues[i + 1:]:
            if a["speaker"] != b["speaker"] and min(a["end"], b["end"]) - max(a["start"], b["start"]) > 1e-6:
                a["simultaneous_overlap"] = True
                b["simultaneous_overlap"] = True

    speakers = sorted({r["speaker"] for r in complete if r["speaker"] not in UNKNOWN})
    speaker_styles = {}
    for idx, spk in enumerate(speakers):
        color = style.speaker_colors.get(spk) or DEFAULT_SPEAKER_COLORS[idx % len(DEFAULT_SPEAKER_COLORS)]
        speaker_styles[spk] = {
            "style_name": f"Speaker_{re.sub(r'[^A-Za-z0-9_]', '_', spk)}",
            "primary_color": color,
        }

    return {
        "kind": "japanese_subtitle_plan",
        "format_version": 1,
        "video_id": video_id,
        "clip": selection,
        "duration_seconds": duration_limit,
        "diagnostic": diagnostic,
        "script_authority": "post_tts_dubbing_script_text_ja",
        "source_script_path": str(script_path),
        "style": asdict(style),
        "speaker_styles": speaker_styles,
        "cues": all_cues,
        "undubbed_entries": [
            {"id": r["id"], "speaker": r["speaker"], "relative_start": r["relative_start"], "relative_end": r["relative_end"], "reason": r.get("skip_reason") or r["generation_status"]}
            for r in missing
        ],
        "same_speaker_conflicts": same_conflicts,
        "other_speaker_overlaps": other_overlaps,
        "alignment_warnings": alignment_warnings,
        "review_required": bool(same_conflicts or missing or alignment_warnings),
        "blockers": blockers,
    }


def render_srt(plan):
    """Render standard UTF-8 SRT subtitles from a Japanese subtitle plan."""
    blocks = []
    for idx, cue in enumerate(plan["cues"], 1):
        start_ts = format_srt_timestamp(cue["start"])
        end_ts = format_srt_timestamp(cue["end"])
        lines = list(cue["lines"])
        if cue.get("simultaneous_overlap") and lines:
            lines[0] = f"[{cue['speaker']}] {lines[0]}"
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n" + "\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_ass(plan):
    """Render UTF-8 ASS subtitles with per-speaker styles and multi-lane simultaneous overlap positioning."""
    st = plan["style"]
    header = [
        "[Script Info]",
        f"Title: TransClip Japanese Subtitles ({plan['video_id']})",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {st['play_res_x']}",
        f"PlayResY: {st['play_res_y']}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for spk, info in sorted(plan["speaker_styles"].items()):
        header.append(
            f"Style: {info['style_name']},{st['font_name']},{st['font_size']},{info['primary_color']},&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},{st['alignment']},{st['margin_l']},{st['margin_r']},{st['margin_v']},1"
        )
    if not plan["speaker_styles"]:
        header.append(
            f"Style: Default,{st['font_name']},{st['font_size']},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"0,0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},{st['alignment']},{st['margin_l']},{st['margin_r']},{st['margin_v']},1"
        )

    events = [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in plan["cues"]:
        spk = cue["speaker"]
        style_name = plan["speaker_styles"].get(spk, {}).get("style_name", "Default")
        start_ts = format_ass_timestamp(cue["start"])
        end_ts = format_ass_timestamp(cue["end"])
        margin_v = st["margin_v"] + int(cue.get("lane", 0)) * st["overlap_lane_step_v"]
        escaped_lines = [
            line.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            for line in cue["lines"]
        ]
        text_body = r"\N".join(escaped_lines)
        events.append(
            f"Dialogue: {cue.get('lane', 0)},{start_ts},{end_ts},{style_name},{spk},0,0,{margin_v},,{text_body}"
        )
    return "\n".join(header + events) + "\n"


def _escape_ffmpeg_filter_path(path):
    text = str(Path(path).resolve())
    for ch in ("\\", ":", "'", "[", "]", ",", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


def export_subtitles(plan, output_dir, *, dubbed_video_path=None):
    """Atomically publish SRT, ASS, subtitle report, and optional burned/unburned MP4s."""
    if plan["blockers"]:
        raise ValueError("Subtitle export blocked: " + ", ".join(plan["blockers"]))
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    req_fingerprint = fingerprint({
        "plan": plan,
        "dubbed_video_sha256": sha256_file(dubbed_video_path) if dubbed_video_path else None,
    })
    target = root / f"subtitles_{req_fingerprint[:16]}"
    with (root / ".subtitles.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            saved = read_json(target / "subtitle_report.json")
            for name, expected_sha in saved.get("artifacts", {}).items():
                p = target / name
                if not p.is_file() or sha256_file(p) != expected_sha:
                    raise ValueError("Existing subtitle artifact is missing or modified.")
            return target / "subtitle_report.json", True

        tmp = Path(tempfile.mkdtemp(prefix=".pending-subs-", dir=root))
        try:
            srt_path = tmp / "final_ja.srt"
            ass_path = tmp / "final_ja.ass"
            srt_path.write_text(render_srt(plan), encoding="utf-8")
            ass_path.write_text(render_ass(plan), encoding="utf-8")

            video_verification = None
            if dubbed_video_path is not None:
                src_video = Path(dubbed_video_path).resolve()
                no_sub_mp4 = tmp / "final_ja_no_subtitles.mp4"
                burned_mp4 = tmp / "final_ja.mp4"
                shutil.copyfile(src_video, no_sub_mp4)
                escaped_ass = _escape_ffmpeg_filter_path(ass_path)
                ffmpeg([
                    "-i", no_sub_mp4,
                    "-vf", f"ass=filename={escaped_ass}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    burned_mp4,
                ])
                video_verification = {
                    "no_subtitles_mp4": validate_mp4(no_sub_mp4, plan["duration_seconds"]),
                    "burned_subtitles_mp4": validate_mp4(burned_mp4, plan["duration_seconds"]),
                }

            artifacts = {p.name: sha256_file(p) for p in tmp.iterdir() if p.is_file()}
            report = {
                **plan,
                "subtitle_key": req_fingerprint,
                "srt_file": "final_ja.srt",
                "ass_file": "final_ja.ass",
                "no_subtitles_video_file": "final_ja_no_subtitles.mp4" if dubbed_video_path is not None else None,
                "subtitled_video_file": "final_ja.mp4" if dubbed_video_path is not None else None,
                "video_verification": video_verification,
                "artifacts": artifacts,
            }
            atomic_json(tmp / "subtitle_report.json", report, overwrite=False)
            tmp.rename(target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    return target / "subtitle_report.json", False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--feedback", type=Path, required=True, help="Path to feedback.json or mix_report.json")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dubbed-video", type=Path, help="Optional dubbed MP4 to produce final_ja_no_subtitles.mp4 and burned final_ja.mp4")
    parser.add_argument("--alignment-json", type=Path, help="Optional Whisper/Forced Alignment JSON keyed by entry_id")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--font-name", default="Noto Sans CJK JP")
    parser.add_argument("--font-size", type=int, default=42)
    parser.add_argument("--max-chars-per-line", type=int, default=22)
    parser.add_argument("--max-lines", type=int, default=2)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        style = SubtitleStyleConfig(
            font_name=args.font_name,
            font_size=args.font_size,
            max_chars_per_line=args.max_chars_per_line,
            max_lines=args.max_lines,
        )
        alignment_data = read_json(args.alignment_json) if args.alignment_json else None
        plan = build_subtitle_plan(
            args.feedback,
            args.selection,
            args.video_id,
            seconds=args.seconds,
            style=style,
            alignment_data=alignment_data,
            diagnostic=args.diagnostic,
        )
        if args.dry_run:
            print(json.dumps({
                "video_id": plan["video_id"],
                "cue_count": len(plan["cues"]),
                "same_speaker_conflicts": plan["same_speaker_conflicts"],
                "other_speaker_overlaps": plan["other_speaker_overlaps"],
                "blockers": plan["blockers"],
            }, ensure_ascii=False, indent=2))
            return 0
        report_path, reused = export_subtitles(plan, args.output_dir, dubbed_video_path=args.dubbed_video)
        print(f"{'Reused' if reused else 'Exported'}: {report_path}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
