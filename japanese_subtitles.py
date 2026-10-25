"""Phase 10–13: Japanese Subtitle Engine synchronized to post-TTS speech placement.

Generates SRT, multi-speaker ASS, and FFmpeg subtitle-burned MP4s strictly from
the final synthesized TTS script (`text_ja` and measured WAV placement), never
from pre-TTS raw translations.

Phase 13 UI & Readability Enhancements:
- High-visibility styling: enhanced outline, drop shadow, and semi-transparent black background box (座布団 / BorderStyle=3) readable on bright backgrounds.
- Size & placement: 1.2–1.3x enlarged font size, elevated bottom margin (`margin_v`), automatic scaling for arbitrary video resolutions and aspect ratios (16:9, 9:16, 1:1, 4:3), and line-overflow prevention/detection.
- Display punctuation formatting: omits Japanese periods `「。」` and converts commas `「、」` to half-width spaces (or minimal) in rendered subtitle display lines without mutating the source translation script.
"""
import argparse
from dataclasses import asdict, dataclass, field
import fcntl
from difflib import SequenceMatcher
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

try:
    from tts_feedback import UNKNOWN, detect_collisions, number
except ImportError:
    UNKNOWN = (None, "UNKNOWN", "REVIEW", "MIXED")
    detect_collisions = None

    def number(value, label, *, positive=False):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {label}.")
        return value

CLAUSE_BREAK_CHARS = set("。！？!?、，,…")
PARTICLE_SURFACES = {"は", "が", "を", "に", "で", "と", "から", "まで", "より", "へ", "や", "て", "ば", "たら", "なら", "ので", "のに", "けれど", "けど", "し"}
DEFAULT_SPEAKER_COLORS = [
    "&H00FFFFFF",  # White
    "&H00A8F0FF",  # Warm light yellow/amber (ASS BGR)
    "&H00FFD8A8",  # Soft sky blue (ASS BGR)
    "&H00C8FFC8",  # Soft mint green (ASS BGR)
]
CONVENTIONAL_BASE_FONT_SIZE_1080P = 42
CONVENTIONAL_BASE_MARGIN_V_1080P = 36


def _ass_alpha_hex(opacity):
    """Convert opacity in [0.0, 1.0] (where 1.0 is opaque) to ASS 2-digit hex transparency (00=opaque, FF=transparent)."""
    op = max(0.0, min(1.0, float(opacity)))
    alpha = int(round((1.0 - op) * 255.0))
    return f"{alpha:02X}"


def format_subtitle_display_text(text, *, strip_periods=True, comma_mode="halfwidth_space"):
    """Format Japanese subtitle text for on-screen display without modifying the underlying script.

    - Omits Japanese full stops `「。」` when `strip_periods=True`.
    - Replaces Japanese commas `「、」` / `「，」` with a half-width space `' '` when `comma_mode='halfwidth_space'`,
      or removes redundant commas when `comma_mode='minimal'`.
    """
    if comma_mode not in ("halfwidth_space", "minimal", "keep"):
        raise ValueError(f"Unsupported comma_mode: {comma_mode!r}")
    out = str(text or "").strip()
    if not out:
        return ""
    if strip_periods:
        out = out.replace("。", "")
    if comma_mode == "halfwidth_space":
        out = re.sub(r"[、，]+\s*", " ", out)
    elif comma_mode == "minimal":
        # Remove commas after short clauses (<5 chars) or before line end; convert remaining to half-width space
        out = re.sub(r"[、，]+$", "", out)
        out = re.sub(r"(?<=^.{1,4})[、，]+", "", out)
        out = re.sub(r"[、，]+\s*", " ", out)
    out = re.sub(r" {2,}", " ", out).strip()
    return out


def format_subtitle_display_lines(lines, *, strip_periods=True, comma_mode="halfwidth_space"):
    """Format a list of wrapped subtitle lines for display while preserving non-empty line structure."""
    formatted = []
    for line in lines or []:
        disp = format_subtitle_display_text(line, strip_periods=strip_periods, comma_mode=comma_mode)
        if disp:
            formatted.append(disp)
    if not formatted and lines:
        fallback = " ".join(str(x).strip() for x in lines if str(x).strip()).strip()
        if fallback:
            formatted = [fallback]
    return formatted


@dataclass(frozen=True)
class SubtitleStyleConfig:
    font_name: str = "Noto Sans CJK JP"
    font_size: int = 52  # 1.24x of conventional 42pt on 1080p
    font_scale: float = 1.24
    max_chars_per_line: int = 22
    max_lines: int = 2
    outline: float = 3.5  # Enhanced outline (up from 2.5)
    shadow: float = 2.5   # Enhanced drop shadow (up from 1.0)
    background_box: bool = True  # Semi-transparent black background box (座布団)
    box_opacity: float = 0.60    # 60% opacity black background box
    strip_periods: bool = True   # Omit 「。」 in display subtitles
    comma_mode: str = "halfwidth_space"  # Replace 「、」 with half-width space in display subtitles
    alignment: int = 2
    margin_l: int = 60
    margin_r: int = 60
    margin_v: int = 64  # Elevated from bottom edge (up from 36)
    overlap_lane_step_v: int = 112
    play_res_x: int = 1920
    play_res_y: int = 1080
    speaker_colors: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.font_name, str) or not self.font_name.strip():
            raise ValueError("Invalid font_name.")
        if type(self.font_size) is not int or not 12 <= self.font_size <= 160:
            raise ValueError("Invalid font_size.")
        if type(self.font_scale) not in (int, float) or not math.isfinite(self.font_scale) or not 0.5 <= self.font_scale <= 3.0:
            raise ValueError("Invalid font_scale.")
        if type(self.max_chars_per_line) is not int or not 6 <= self.max_chars_per_line <= 60:
            raise ValueError("Invalid max_chars_per_line.")
        if type(self.max_lines) is not int or not 1 <= self.max_lines <= 4:
            raise ValueError("Invalid max_lines.")
        if type(self.alignment) is not int or not 1 <= self.alignment <= 9:
            raise ValueError("Invalid ASS alignment (1..9).")
        for name, val in [("outline", self.outline), ("shadow", self.shadow)]:
            if type(val) not in (int, float) or not math.isfinite(val) or val < 0 or val > 16:
                raise ValueError(f"Invalid {name}.")
        if type(self.background_box) is not bool:
            raise ValueError("Invalid background_box.")
        if type(self.box_opacity) not in (int, float) or not math.isfinite(self.box_opacity) or not 0.0 <= self.box_opacity <= 1.0:
            raise ValueError("Invalid box_opacity.")
        if type(self.strip_periods) is not bool:
            raise ValueError("Invalid strip_periods.")
        if self.comma_mode not in ("halfwidth_space", "minimal", "keep"):
            raise ValueError("Invalid comma_mode.")
        for name, val in [("margin_l", self.margin_l), ("margin_r", self.margin_r), ("margin_v", self.margin_v), ("overlap_lane_step_v", self.overlap_lane_step_v)]:
            if type(val) is not int or val < 0 or val > 800:
                raise ValueError(f"Invalid {name}.")

    @classmethod
    def for_video_resolution(
        cls,
        width: int = 1920,
        height: int = 1080,
        *,
        font_scale: float = 1.25,
        background_box: bool = True,
        box_opacity: float = 0.60,
        strip_periods: bool = True,
        comma_mode: str = "halfwidth_space",
        **overrides,
    ):
        """Construct a resolution- and aspect-ratio-aware SubtitleStyleConfig.

        Automatically scales font_size (1.2–1.3x base), bottom margin_v (slightly elevated),
        horizontal margins, outline/shadow thickness, and max_chars_per_line so subtitles
        never overflow horizontally on 16:9, 9:16 vertical, 1:1 square, or 4:3 videos.
        """
        w = max(160, int(width or 1920))
        h = max(120, int(height or 1080))
        scale = float(font_scale)
        aspect = w / float(h)

        # Conventional base font size is ~3.88% of height on landscape (42pt on 1080p, 28pt on 720p)
        if aspect < 0.85:
            # Vertical / portrait video (e.g. 9:16 1080x1920): constrain font size by width
            base_font = w * 0.042
            margin_v = max(48, int(round(h * 0.085)))
            margin_lr = max(24, int(round(w * 0.055)))
        elif aspect < 1.15:
            # Square video (e.g. 1:1 1080x1080)
            base_font = h * 0.038
            margin_v = max(36, int(round(h * 0.068)))
            margin_lr = max(28, int(round(w * 0.05)))
        else:
            # Landscape video (16:9, 4:3, etc.)
            base_font = h * (CONVENTIONAL_BASE_FONT_SIZE_1080P / 1080.0)
            margin_v = max(20, int(round(h * 0.060)))
            margin_lr = max(20, int(round(w * 0.035)))

        computed_font_size = max(14, min(140, int(round(base_font * scale))))
        usable_width = max(100, w - 2 * margin_lr)
        # Ensure max_chars_per_line fits comfortably within usable_width (with 1.05x CJK safety factor)
        fit_chars = max(8, min(26, int(math.floor(usable_width / max(1.0, computed_font_size * 1.04)))))
        outline_px = round(max(1.5, min(8.0, computed_font_size * 0.068)), 2)
        shadow_px = round(max(1.0, min(6.0, computed_font_size * 0.048)), 2)
        lane_step = max(28, int(round(computed_font_size * 2.15)))

        params = {
            "font_name": "Noto Sans CJK JP",
            "font_size": computed_font_size,
            "font_scale": round(scale, 3),
            "max_chars_per_line": fit_chars,
            "max_lines": 2,
            "outline": outline_px,
            "shadow": shadow_px,
            "background_box": bool(background_box),
            "box_opacity": float(box_opacity),
            "strip_periods": bool(strip_periods),
            "comma_mode": str(comma_mode),
            "alignment": 2,
            "margin_l": margin_lr,
            "margin_r": margin_lr,
            "margin_v": margin_v,
            "overlap_lane_step_v": lane_step,
            "play_res_x": w,
            "play_res_y": h,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**params)


def estimate_line_pixel_width(line, style: SubtitleStyleConfig):
    """Estimate rendered pixel width of a subtitle line for overflow verification."""
    text = str(line or "")
    em_units = 0.0
    for ch in text:
        if ch.isascii():
            em_units += 0.55
        else:
            em_units += 1.0
    return round(em_units * float(style.font_size) + 2.0 * float(style.outline), 2)


def prevent_line_overflow(lines, style: SubtitleStyleConfig):
    """Ensure no subtitle line exceeds `style.max_chars_per_line` or available horizontal pixel width."""
    usable_width = max(80.0, float(style.play_res_x - style.margin_l - style.margin_r))
    max_chars_by_px = max(6, int(math.floor((usable_width - 2.0 * style.outline) / max(1.0, float(style.font_size)))))
    effective_max_chars = min(int(style.max_chars_per_line), max_chars_by_px)

    safe_lines = []
    for line in lines or []:
        clean = str(line).strip()
        if not clean:
            continue
        while len(clean) > effective_max_chars:
            # Find natural break point near effective_max_chars or midpoint
            target = min(effective_max_chars, max(4, len(clean) // 2))
            split_at = target
            for offset in range(0, min(6, target - 2)):
                for cand in (target - offset, min(effective_max_chars, target + offset)):
                    if 2 <= cand < len(clean):
                        prev_c = clean[cand - 1]
                        next_c = clean[cand]
                        if prev_c.isascii() and prev_c.isalnum() and next_c.isascii() and next_c.isalnum():
                            continue
                        if prev_c in " 、。！？はがをにでとへもやの":
                            split_at = cand
                            break
                if split_at != target:
                    break
            safe_lines.append(clean[:split_at].strip())
            clean = clean[split_at:].strip()
        if clean:
            safe_lines.append(clean)

    # If wrapping exceeded max_lines, rebalance across max_lines if possible
    if len(safe_lines) > style.max_lines and style.max_lines >= 2:
        combined = " ".join(safe_lines) if any(" " in s for s in safe_lines) else "".join(safe_lines)
        if len(combined) <= effective_max_chars * style.max_lines:
            rebalanced = []
            rem = combined
            for i in range(style.max_lines):
                slots_left = style.max_lines - i
                if slots_left == 1 or len(rem) <= effective_max_chars:
                    rebalanced.append(rem.strip())
                    rem = ""
                    break
                cut = min(effective_max_chars, max(1, math.ceil(len(rem) / slots_left)))
                rebalanced.append(rem[:cut].strip())
                rem = rem[cut:].strip()
            if not rem:
                safe_lines = [s for s in rebalanced if s]
    return safe_lines[: style.max_lines] if len(safe_lines) <= style.max_lines else safe_lines


def check_subtitle_overflow(cues, style: SubtitleStyleConfig):
    """Audit all subtitle cues for character count, pixel width, and vertical margin bounds."""
    usable_width = max(80.0, float(style.play_res_x - style.margin_l - style.margin_r))
    overflow_cues = []
    max_px = 0.0
    max_chars = 0

    for cue in cues or []:
        disp_lines = cue.get("display_lines") or cue.get("lines") or []
        for l_idx, line in enumerate(disp_lines):
            px = estimate_line_pixel_width(line, style)
            ch_len = len(line)
            max_px = max(max_px, px)
            max_chars = max(max_chars, ch_len)
            if ch_len > style.max_chars_per_line or px > usable_width + 1.0:
                overflow_cues.append({
                    "entry_id": cue.get("entry_id", cue.get("id")),
                    "cue_index": cue.get("cue_index", 0),
                    "line_index": l_idx,
                    "line_text": line,
                    "char_count": ch_len,
                    "max_chars_per_line": style.max_chars_per_line,
                    "estimated_pixel_width": px,
                    "available_pixel_width": usable_width,
                })

    return {
        "overflow_count": len(overflow_cues),
        "overflow_cues": overflow_cues,
        "max_line_char_count": max_chars,
        "max_line_pixel_width": round(max_px, 2),
        "available_pixel_width": round(usable_width, 2),
        "all_lines_within_bounds": len(overflow_cues) == 0,
    }


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
        if current_group and (current_len + u_len > max_cue_chars or re.search(r"[。！？!?]$", current_group[-1])):
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


def _attach_display_formatting(cue, style: SubtitleStyleConfig):
    """Attach punctuation-formatted, overflow-protected `display_lines` and `display_text_ja` while keeping `text_ja` intact."""
    raw_lines = prevent_line_overflow(cue.get("lines") or [cue["text_ja"]], style)
    disp_lines = format_subtitle_display_lines(
        raw_lines,
        strip_periods=style.strip_periods,
        comma_mode=style.comma_mode,
    )
    disp_lines = prevent_line_overflow(disp_lines, style)
    cue["lines"] = raw_lines
    cue["display_lines"] = disp_lines
    cue["display_text_ja"] = "\n".join(disp_lines)
    return cue


def _word_aligned_cues(entry, style, alignment):
    """Map script clauses to monotonic WAV-bound ASR words, never replace script text."""
    if alignment.get("time_basis") != "wav_relative" or not entry.get("audio_file"):
        raise ValueError("WAV-relative alignment and source audio are required.")
    if alignment.get("audio_sha256") != sha256_file(entry["audio_file"]):
        raise ValueError("Alignment belongs to a different WAV.")
    text = entry["text_ja"].strip()
    # Punctuation defines sentence/clause units; layout wrapping may split long clauses further.
    phrases = re.findall(r"[^。！？!?、，]+[。！？!?、，]*", text)
    specs = []
    for phrase in phrases:
        specs.extend(wrap_clause_units(split_japanese_clauses(phrase), max_chars_per_line=style.max_chars_per_line, max_lines=style.max_lines))
    script = "".join(_normalize_for_comparison(c["text"]) for c in specs)
    recognized = ""; times = []; previous = 0.0
    duration = float(entry["actual_tts_duration_seconds"])
    for w in alignment["words"]:
        a, b = w.get("start"), w.get("end")
        if type(a) not in (int,float) or type(b) not in (int,float) or not (math.isfinite(a) and math.isfinite(b) and previous <= a < b <= duration + .05):
            raise ValueError("Invalid/non-monotonic word timing.")
        word = _normalize_for_comparison(w.get("word", ""))
        recognized += word; times.extend([(a,b)] * len(word)); previous = b
    mapping = {}
    for block in SequenceMatcher(None, script, recognized, autojunk=False).get_matching_blocks():
        mapping.update({block.a+i:times[block.b+i] for i in range(block.size)})
    if not script or len(mapping)/len(script) < .85:
        raise ValueError("Insufficient script-to-audio alignment coverage.")
    warnings = [] if script == recognized else ["asr_transcript_differs_from_tts_script"]
    warnings.append("asr_word_timing_requires_listening_review")
    cues=[]; offset=0; start=float(entry.get("placement_seconds",entry.get("relative_start",0)));last=start
    for index,spec in enumerate(specs):
        size=len(_normalize_for_comparison(spec["text"]))
        matched=[mapping[i] for i in range(offset,offset+size) if i in mapping];offset+=size
        if not matched or len(matched)/max(size,1)<.7:raise ValueError("Clause is not sufficiently aligned.")
        a=start+matched[0][0];b=min(start+duration,start+matched[-1][1])
        if a<last-1e-6 or b<=a:raise ValueError("Clause boundaries overlap or are empty.")
        cue_obj = {"entry_id":entry["id"],"cue_index":index,"speaker":entry["speaker"],"start":round(a,4),"end":round(b,4),
                   "text_ja":spec["text"],"lines":spec["lines"],"timing_source":"wav_bound_whisper_words", "alignment_warnings":warnings.copy()}
        _attach_display_formatting(cue_obj, style)
        cues.append(cue_obj)
        last=b
    return cues,warnings


def align_entry_cues(entry, style: SubtitleStyleConfig, alignment_entry=None):
    """Create time-aligned subtitle cues for one completed TTS utterance."""
    start = float(entry.get("placement_seconds", entry.get("relative_start", 0)))
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
    if alignment_entry is not None and "words" in alignment_entry:
        try:
            return _word_aligned_cues(entry, style, alignment_entry)
        except (ValueError, KeyError, OSError) as error:
            warnings.append("word_alignment_rejected:" + str(error))
            alignment_entry = None
    timing_source = "tts_measured_proportional"
    if alignment_entry is not None:
        recognized_text = alignment_entry.get("recognized_text", "")
        if recognized_text:
            if _normalize_for_comparison(recognized_text) != _normalize_for_comparison(text_ja):
                warnings.append("asr_transcript_differs_from_tts_script")
        segments = alignment_entry.get("segments")
        if isinstance(segments, list) and len(segments) == len(cue_specs):
            valid_segs = True
            previous_end = start - .05
            for seg in segments:
                s_a, s_b = seg.get("relative_start"), seg.get("relative_end")
                if type(s_a) not in (int, float) or type(s_b) not in (int, float) or not (previous_end <= s_a < s_b <= end + 0.05):
                    valid_segs = False
                    break
                previous_end = s_b
            if valid_segs:
                timing_source = "forced_or_whisper_alignment"
                cues = []
                for idx, (spec, seg) in enumerate(zip(cue_specs, segments)):
                    cue_obj = {
                        "entry_id": entry["id"],
                        "cue_index": idx,
                        "speaker": entry["speaker"],
                        "start": round(max(start, float(seg["relative_start"])), 4),
                        "end": round(min(end, float(seg["relative_end"])), 4),
                        "text_ja": spec["text"],
                        "lines": spec["lines"],
                        "timing_source": timing_source,
                        "alignment_warnings": list(warnings),
                    }
                    _attach_display_formatting(cue_obj, style)
                    cues.append(cue_obj)
                return cues, warnings
        elif segments is not None:
            warnings.append("alignment_segment_count_mismatch_used_proportional")

    warnings.append("proportional_timing_not_audio_alignment")
    total_weight = sum(c["weight"] for c in cue_specs)
    cursor = start
    cues = []
    for idx, spec in enumerate(cue_specs):
        if idx == len(cue_specs) - 1:
            cue_end = end
        else:
            share = duration * (spec["weight"] / total_weight)
            cue_end = min(end, cursor + share)
        cue_obj = {
            "entry_id": entry["id"],
            "cue_index": idx,
            "speaker": entry["speaker"],
            "start": round(cursor, 4),
            "end": round(cue_end, 4),
            "text_ja": spec["text"],
            "lines": spec["lines"],
            "timing_source": timing_source,
            "alignment_warnings": list(warnings),
        }
        _attach_display_formatting(cue_obj, style)
        cues.append(cue_obj)
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

    overflow_check = check_subtitle_overflow(all_cues, style)

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
        "overflow_check": overflow_check,
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
    """Render standard UTF-8 SRT subtitles from a Japanese subtitle plan using formatted display lines."""
    st = plan.get("style") or {}
    strip_p = bool(st.get("strip_periods", True))
    comma_m = str(st.get("comma_mode", "halfwidth_space"))
    blocks = []
    for idx, cue in enumerate(plan["cues"], 1):
        start_ts = format_srt_timestamp(cue["start"])
        end_ts = format_srt_timestamp(cue["end"])
        raw_lines = cue.get("display_lines")
        if not raw_lines:
            raw_lines = format_subtitle_display_lines(cue["lines"], strip_periods=strip_p, comma_mode=comma_m)
        lines = list(raw_lines)
        if cue.get("simultaneous_overlap") and lines:
            lines[0] = f"[{cue['speaker']}] {lines[0]}"
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n" + "\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_ass(plan):
    """Render UTF-8 ASS subtitles with per-speaker styles, semi-transparent background box (座布団),
    enhanced outline/shadow, and multi-lane simultaneous overlap positioning.
    """
    st = plan["style"]
    bg_box = bool(st.get("background_box", True))
    box_opacity = float(st.get("box_opacity", 0.60))
    border_style = 3 if bg_box else 1
    alpha_hex = _ass_alpha_hex(box_opacity)
    # In ASS BorderStyle=3, OutlineColour is the box fill color and BackColour is the box drop shadow.
    outline_col = f"&H{alpha_hex}000000" if bg_box else "&H00000000"
    back_col = "&H80000000"
    strip_p = bool(st.get("strip_periods", True))
    comma_m = str(st.get("comma_mode", "halfwidth_space"))

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
            f"Style: {info['style_name']},{st['font_name']},{st['font_size']},{info['primary_color']},&H000000FF,{outline_col},{back_col},"
            f"1,0,0,0,100,100,0,0,{border_style},{st['outline']},{st['shadow']},{st['alignment']},{st['margin_l']},{st['margin_r']},{st['margin_v']},1"
        )
    if not plan["speaker_styles"]:
        header.append(
            f"Style: Default,{st['font_name']},{st['font_size']},&H00FFFFFF,&H000000FF,{outline_col},{back_col},"
            f"1,0,0,0,100,100,0,0,{border_style},{st['outline']},{st['shadow']},{st['alignment']},{st['margin_l']},{st['margin_r']},{st['margin_v']},1"
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
        disp_lines = cue.get("display_lines")
        if not disp_lines:
            disp_lines = format_subtitle_display_lines(cue["lines"], strip_periods=strip_p, comma_mode=comma_m)
        escaped_lines = [
            line.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            for line in disp_lines
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
    parser.add_argument("--font-size", type=int, default=52)
    parser.add_argument("--font-scale", type=float, default=1.25)
    parser.add_argument("--max-chars-per-line", type=int, default=22)
    parser.add_argument("--max-lines", type=int, default=2)
    parser.add_argument("--outline", type=float, default=3.5)
    parser.add_argument("--shadow", type=float, default=2.5)
    parser.add_argument("--margin-v", type=int, default=64)
    parser.add_argument("--background-box", dest="background_box", action="store_true", default=True)
    parser.add_argument("--no-background-box", dest="background_box", action="store_false")
    parser.add_argument("--box-opacity", type=float, default=0.60)
    parser.add_argument("--strip-periods", dest="strip_periods", action="store_true", default=True)
    parser.add_argument("--keep-periods", dest="strip_periods", action="store_false")
    parser.add_argument("--comma-mode", choices=("halfwidth_space", "minimal", "keep"), default="halfwidth_space")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        style = SubtitleStyleConfig(
            font_name=args.font_name,
            font_size=args.font_size,
            font_scale=args.font_scale,
            max_chars_per_line=args.max_chars_per_line,
            max_lines=args.max_lines,
            outline=args.outline,
            shadow=args.shadow,
            margin_v=args.margin_v,
            background_box=args.background_box,
            box_opacity=args.box_opacity,
            strip_periods=args.strip_periods,
            comma_mode=args.comma_mode,
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
                "overflow_check": plan["overflow_check"],
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
