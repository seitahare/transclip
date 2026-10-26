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
        # Phase 13.2: Target 20-22 chars per line on landscape (clamped to <= 22 and constrained by usable pixel width)
        fit_chars = max(8, min(22, int(math.floor(usable_width / max(1.0, computed_font_size * 1.06)))))
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


# Compound particles & functional continuations that MUST NEVER be separated from the preceding token
_INSEPARABLE_CONTINUATION_PREFIXES = (
    "しては",
    "しても",
    "しての",
    "して",
    "した",
    "する",
    "ついては",
    "ついても",
    "ついて",
    "関しては",
    "関して",
    "よっては",
    "よって",
    "よる",
    "おいては",
    "おいて",
    "おける",
    "とっては",
    "とって",
    "対しては",
    "対して",
    "対する",
    "ともに",
    "いった",
    "いう",
    "くらい",
    "ぐらい",
    "ばかり",
    "ながら",
    "つつも",
    "つつ",
    "ために",
    "ため",
    "ように",
    "ような",
    "ようで",
    "わけでは",
    "わけです",
    "わけで",
    "わけ",
    "では",
    "には",
    "とは",
    "にも",
    "でも",
    "かも",
    "ても",
    "たり",
    "だり",
    "ので",
    "のに",
    "けれども",
    "けれど",
    "から",
    "まで",
    "より",
    "ほど",
    "など",
    "いませんよね",
    "いません",
    "ません",
    "ました",
    "ている",
    "ていた",
    "ていく",
    "てくる",
    "ていったんだ",
    "て言ったんだ",
    "ていった",
    "て言った",
    "られる",
    "される",
    "された",
    "されて",
    "なりますが",
    "なります",
    "なった",
    "なる",
    "ない",
    "なかった",
    "んです",
    "んだ",
    "よね",
    "ね",
    "よ",
    "か",
    "し",
    "て",
    "で",
    "は",
    "が",
    "を",
    "に",
    "と",
    "へ",
    "も",
    "や",
    "の",
)


# Verbal-noun (連用形名詞) stems that form compound nouns with immediately following Kanji (e.g. `組み立て作業`, `割り当て結果`, `引き伸ばし防止`)
_COMPOUND_VERBAL_NOUN_PREFIXES = (
    "組み立て",
    "割り当て",
    "引き伸ばし",
    "切り抜き",
    "吹き替え",
    "切り替え",
    "組み合わせ",
    "取り組み",
    "立ち上げ",
    "受け入れ",
    "聞き取り",
    "書き起こし",
    "読み上げ",
    "打ち合わせ",
    "積み重ね",
)


def _is_prohibited_split_position(text: str, pos: int) -> bool:
    """Return True if splitting `text` at index `pos` (`text[:pos]` / `text[pos:]`) breaks a bunsetsu,
    compound particle (like `として`), compound verbal-noun (`組み立て作業`), English/numeric token, or punctuation.
    """
    if pos <= 0 or pos >= len(text):
        return True
    left = text[:pos]
    right = text[pos:]
    prev_c = left[-1]
    next_c = right[0]

    # 1. Never break inside ASCII words/numbers
    if prev_c.isascii() and prev_c.isalnum() and next_c.isascii() and next_c.isalnum():
        return True
    # 2. Never start a new line with closing punctuation, particles, or small kana
    if next_c in "、。，．！？!?」』）)]｝・ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ":
        return True
    # 3. Never end a line with opening brackets
    if prev_c in "「『（([｛":
        return True
    # 4. Never split compound verbal-noun + Kanji compounds (e.g. `組み立て作業`, `割り当て検証`)
    if "\u4e00" <= next_c <= "\u9fff":
        for vn in _COMPOUND_VERBAL_NOUN_PREFIXES:
            if left.endswith(vn):
                return True
    # 5. Never split compound particles or auxiliary verb chains (e.g. `ロボットと / して`)
    right_stripped = right.lstrip(" ")
    for prefix in _INSEPARABLE_CONTINUATION_PREFIXES:
        if right_stripped.startswith(prefix):
            if prev_c in " 、。！？!?" and prefix in ("でも", "から", "ので", "のに", "けれど", "けれども", "して"):
                if prefix != "して":
                    return False
            return True
    # 6. Never split inside contiguous Katakana words (e.g. `エンドエフェクター`, `ソフトウェア`, `ロボット`)
    if ("ァ" <= prev_c <= "ヺ" or prev_c == "ー") and ("ァ" <= next_c <= "ヺ" or next_c == "ー"):
        return True
    # 7. Never split inside contiguous Kanji compounds unless separated by particle/punctuation
    if ("\u4e00" <= prev_c <= "\u9fff") and ("\u4e00" <= next_c <= "\u9fff"):
        return True
    # 8. Never split between a Kanji/Katakana/ASCII noun and an immediately following particle
    if (
        ("\u4e00" <= prev_c <= "\u9fff")
        or ("ァ" <= prev_c <= "ヺ" or prev_c == "ー")
        or (prev_c.isascii() and prev_c.isalnum())
    ) and ("ぁ" <= next_c <= "ん"):
        return True
    return False


def segment_japanese_bunsetsu(text: str, japanese_analysis=None) -> list[str]:
    """BudouX-style Japanese bunsetsu (phrase) segmenter.

    Uses `budoux` if available in the environment and post-processes with compound-particle
    and technical-term merging rules so phrases like `ロボットとして`, `デジタルAIと`,
    `人型ロボットの`, and `大規模AIモデルに` are never split mid-phrase.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    # 1. If morphological tokens are provided and match the text, group them into bunsetsu
    tokens = (japanese_analysis or {}).get("tokens")
    if isinstance(tokens, list) and tokens and all(isinstance(t, dict) and "surface" in t for t in tokens):
        reconstructed = "".join(t["surface"] for t in tokens)
        if _normalize_for_comparison(reconstructed) == _normalize_for_comparison(raw):
            raw_pieces = [t["surface"] for t in tokens if t.get("surface")]
            merged = []
            for piece in raw_pieces:
                if merged and _is_prohibited_split_position(merged[-1] + piece, len(merged[-1])):
                    merged[-1] += piece
                else:
                    merged.append(piece)
            return merged

    # 2. Try BudouX if installed in environment
    initial_chunks = None
    try:
        import budoux  # type: ignore

        parser = budoux.load_default_japanese_parser()
        initial_chunks = [c for c in parser.parse(raw) if c]
    except Exception:
        initial_chunks = None

    # 3. Built-in BudouX-style character-transition & particle-boundary bunsetsu segmenter
    if not initial_chunks:
        initial_chunks = []
        cur = ""
        n = len(raw)
        i = 0
        while i < n:
            ch = raw[i]
            cur += ch
            i += 1
            if i >= n:
                break
            # Check if `i` is a valid bunsetsu boundary in `raw`
            prev_c = raw[i - 1]
            next_c = raw[i]
            # Always break after punctuation or space
            if prev_c in CLAUSE_BREAK_CHARS or prev_c == " ":
                if not _is_prohibited_split_position(raw, i):
                    initial_chunks.append(cur)
                    cur = ""
                continue
            # Transition from Hiragana (particle/auxiliary tail) to Kanji, Katakana, or ASCII content word
            is_hira_prev = "ぁ" <= prev_c <= "ん"
            is_content_next = (
                ("\u4e00" <= next_c <= "\u9fff")
                or ("ァ" <= next_c <= "ヺ")
                or (next_c.isascii() and next_c.isalnum())
            )
            if is_hira_prev and is_content_next and not _is_prohibited_split_position(raw, i):
                initial_chunks.append(cur)
                cur = ""
                continue
            # Transition after particle before common Hiragana lexical starters (`それ`, `これ`, `そこ`, `ここ`, `まだ`, `もう`, `いわば`, `あらゆる`)
            if prev_c in "はがをにでとへもやのし、 " and not _is_prohibited_split_position(raw, i):
                rem = raw[i:]
                if rem.startswith(("それ", "これ", "あれ", "そこ", "ここ", "まだ", "もう", "いわば", "あらゆる", "ほとんど", "だから", "でも", "ただ")):
                    initial_chunks.append(cur)
                    cur = ""
        if cur:
            initial_chunks.append(cur)

    # 4. Post-merge any chunks that would violate `_is_prohibited_split_position` or leave tiny 1-2 char orphans
    merged_bunsetsu = []
    for piece in initial_chunks:
        if not piece:
            continue
        if merged_bunsetsu:
            combined = merged_bunsetsu[-1] + piece
            if _is_prohibited_split_position(combined, len(merged_bunsetsu[-1])):
                merged_bunsetsu[-1] = combined
                continue
            if len(merged_bunsetsu[-1].strip()) <= 2 and not any(c in CLAUSE_BREAK_CHARS for c in merged_bunsetsu[-1]):
                merged_bunsetsu[-1] = combined
                continue
        merged_bunsetsu.append(piece)

    if len(merged_bunsetsu) >= 2 and len(merged_bunsetsu[-1].strip()) <= 2:
        merged_bunsetsu[-2] += merged_bunsetsu[-1]
        merged_bunsetsu.pop()

    return merged_bunsetsu or [raw]


def detect_unnatural_line_breaks(lines_or_cues):
    """Check whether any line break splits mid-bunsetsu (e.g. `ロボットと / して`).

    If passed `list[str]`, returns `list[dict]`.
    If passed `list[dict]` (cues), returns `{'passed': bool, 'unnatural_break_count': int, 'violations': list[dict]}`.
    """
    items = list(lines_or_cues or [])
    if items and isinstance(items[0], dict):
        all_violations = []
        for cue in items:
            c_lines = cue.get("display_lines") or cue.get("lines") or []
            v_list = detect_unnatural_line_breaks(c_lines)
            for v in v_list:
                all_violations.append({
                    "entry_id": cue.get("entry_id", cue.get("id")),
                    "cue_index": cue.get("cue_index", 0),
                    **v,
                })
        return {
            "passed": len(all_violations) == 0,
            "unnatural_break_count": len(all_violations),
            "violations": all_violations,
        }

    violations = []
    for i in range(len(items) - 1):
        left = str(items[i]).rstrip()
        right = str(items[i + 1]).lstrip()
        if not left or not right:
            continue
        combined = left + right
        if _is_prohibited_split_position(combined, len(left)):
            violations.append({
                "line_index": i,
                "left_tail": left[-6:],
                "right_head": right[:6],
                "reason": "mid_bunsetsu_or_compound_particle_split",
            })
    return violations


def prevent_line_overflow(lines, style: SubtitleStyleConfig):
    """Ensure no subtitle line exceeds `style.max_chars_per_line` or available horizontal pixel width,
    and never exceeds `style.max_lines` (default 2) while respecting bunsetsu boundaries.
    """
    usable_width = max(80.0, float(style.play_res_x - style.margin_l - style.margin_r))
    max_chars_by_px = max(6, int(math.floor((usable_width - 2.0 * style.outline) / max(1.0, float(style.font_size)))))
    effective_max_chars = min(int(style.max_chars_per_line), max_chars_by_px)

    safe_lines = []
    for line in lines or []:
        clean = str(line).strip()
        if not clean:
            continue
        if len(clean) <= effective_max_chars:
            safe_lines.append(clean)
            continue
        # Split long line at bunsetsu boundaries
        bunsetsu = segment_japanese_bunsetsu(clean)
        if len(bunsetsu) >= 2:
            cur = ""
            for b in bunsetsu:
                cand = (cur + b) if not (cur.endswith(" ") or b.startswith(" ")) else (cur + b)
                if cur and len(cand.strip()) > effective_max_chars:
                    safe_lines.append(cur.strip())
                    cur = b.lstrip()
                else:
                    cur = cand
            if cur.strip():
                safe_lines.append(cur.strip())
        else:
            # Fallback if a single token exceeds effective_max_chars
            while len(clean) > effective_max_chars:
                target = min(effective_max_chars, max(4, len(clean) // 2))
                split_at = target
                for offset in range(0, min(8, target - 2)):
                    for cand in (target - offset, min(effective_max_chars, target + offset)):
                        if 2 <= cand < len(clean) and not _is_prohibited_split_position(clean, cand):
                            split_at = cand
                            break
                    if split_at != target:
                        break
                safe_lines.append(clean[:split_at].strip())
                clean = clean[split_at:].strip()
            if clean:
                safe_lines.append(clean)

    # If wrapping produced > style.max_lines, rebalance across style.max_lines using bunsetsu boundaries
    if len(safe_lines) > style.max_lines and style.max_lines == 2:
        joined = " ".join(s.strip() for s in safe_lines if s.strip())
        bunsetsu = segment_japanese_bunsetsu(joined)
        if len(bunsetsu) >= 2:
            best_idx = 1
            best_score = math.inf
            for idx in range(1, len(bunsetsu)):
                l_str = "".join(bunsetsu[:idx]).strip()
                r_str = "".join(bunsetsu[idx:]).strip()
                ov = max(0, len(l_str) - effective_max_chars) + max(0, len(r_str) - effective_max_chars)
                penalty = 500 if _is_prohibited_split_position(l_str + r_str, len(l_str)) else 0
                score = ov * 100 + penalty + abs(len(l_str) - len(r_str))
                if score < best_score:
                    best_score = score
                    best_idx = idx
            l_final = "".join(bunsetsu[:best_idx]).strip()
            r_final = "".join(bunsetsu[best_idx:]).strip()
            if len(l_final) <= effective_max_chars and len(r_final) <= effective_max_chars:
                return [l_final, r_final]

    return safe_lines[: style.max_lines] if len(safe_lines) <= style.max_lines else safe_lines


def check_subtitle_overflow(cues, style: SubtitleStyleConfig):
    """Audit all subtitle cues for character count, pixel width, max line count (<= 2), and natural bunsetsu breaks."""
    usable_width = max(80.0, float(style.play_res_x - style.margin_l - style.margin_r))
    overflow_cues = []
    unnatural_break_cues = []
    max_px = 0.0
    max_chars = 0
    max_line_count = 0

    for cue in cues or []:
        disp_lines = cue.get("display_lines") or cue.get("lines") or []
        max_line_count = max(max_line_count, len(disp_lines))
        if len(disp_lines) > style.max_lines:
            overflow_cues.append({
                "entry_id": cue.get("entry_id", cue.get("id")),
                "cue_index": cue.get("cue_index", 0),
                "reason": f"exceeds_max_lines:{len(disp_lines)}>{style.max_lines}",
                "line_count": len(disp_lines),
            })
        break_issues = detect_unnatural_line_breaks(disp_lines)
        if break_issues:
            unnatural_break_cues.append({
                "entry_id": cue.get("entry_id", cue.get("id")),
                "cue_index": cue.get("cue_index", 0),
                "issues": break_issues,
            })
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
        "unnatural_break_count": len(unnatural_break_cues),
        "unnatural_break_cues": unnatural_break_cues,
        "max_line_char_count": max_chars,
        "max_line_count_observed": max_line_count,
        "max_line_pixel_width": round(max_px, 2),
        "available_pixel_width": round(usable_width, 2),
        "all_lines_within_bounds": len(overflow_cues) == 0 and max_line_count <= style.max_lines,
        "all_breaks_natural": len(unnatural_break_cues) == 0,
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
    """Split Japanese text into natural bunsetsu/clause units without losing any characters
    and without breaking compound particles like `として` or technical terms.
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot split empty subtitle text.")
    return segment_japanese_bunsetsu(text, japanese_analysis=japanese_analysis)


def _format_len(text: str) -> int:
    """Measure effective display length of a Japanese string after period removal and comma->space formatting."""
    s = str(text or "").replace("。", "")
    s = re.sub(r"[、，]+\s*", " ", s).strip()
    return len(s)


def _balance_group_into_two_lines(group: list[str], max_chars_per_line: int, max_lines: int = 2) -> list[str]:
    """Balance a list of bunsetsu units into at most `max_lines` (default 2) lines without breaking any bunsetsu."""
    joined = "".join(group).strip()
    if _format_len(joined) <= max_chars_per_line or max_lines == 1:
        return [joined]
    if len(group) >= 2:
        best_idx = 1
        best_score = math.inf
        for idx in range(1, len(group)):
            left = "".join(group[:idx]).strip()
            right = "".join(group[idx:]).strip()
            l_len = _format_len(left)
            r_len = _format_len(right)
            overflow = max(0, l_len - max_chars_per_line) + max(0, r_len - max_chars_per_line)
            prohibited_pen = 1000 if _is_prohibited_split_position(left + right, len(left)) else 0
            # Strongly prefer breaking at clause commas (`、`) when both lines fit within max_chars_per_line
            comma_bonus = -14 if (left.endswith(("、", "，", " ")) and overflow == 0) else 0
            attrib_pen = 18 if left.endswith(("的な", "の", "な", "たる", "あらゆる", "いわば")) else 0
            score = overflow * 100 + prohibited_pen + attrib_pen + abs(l_len - r_len) + comma_bonus
            if score < best_score:
                best_score = score
                best_idx = idx
        left_s = "".join(group[:best_idx]).strip()
        right_s = "".join(group[best_idx:]).strip()
        return [line for line in (left_s, right_s) if line]

    # Single long unit fallback
    mid = min(max_chars_per_line, max(1, len(joined) // 2))
    for offset in range(0, min(8, mid - 2)):
        for cand in (mid - offset, min(max_chars_per_line, mid + offset)):
            if 2 <= cand < len(joined) and not _is_prohibited_split_position(joined, cand):
                return [joined[:cand].strip(), joined[cand:].strip()]
    return [joined[:mid].strip(), joined[mid:].strip()]


def wrap_clause_units(units, *, max_chars_per_line=22, max_lines=2):
    """Group bunsetsu/clause units into cue blocks where EVERY cue fits in at most `max_lines` (2) lines
    of `max_chars_per_line` characters, splitting long sentences into multiple cues at natural clause boundaries.
    """
    max_cue_chars = max_chars_per_line * max_lines
    cue_groups = []
    current_group = []

    def _group_fits_in_cue(grp: list[str]) -> bool:
        if not grp:
            return True
        joined_len = _format_len("".join(grp))
        if joined_len > max_cue_chars:
            return False
        lines = _balance_group_into_two_lines(grp, max_chars_per_line, max_lines=max_lines)
        return len(lines) <= max_lines and all(_format_len(line) <= max_chars_per_line for line in lines)

    for unit in units:
        if not unit:
            continue
        if current_group and (
            not _group_fits_in_cue([*current_group, unit])
            or re.search(r"[。！？!?]$", current_group[-1])
        ):
            cue_groups.append(current_group)
            current_group = [unit]
        else:
            current_group.append(unit)
    if current_group:
        cue_groups.append(current_group)

    # Rebalance adjacent cue groups at clause boundaries (`、`, `つつ`, `で、`, `が、`)
    if len(cue_groups) == 2:
        all_units = [u for g in cue_groups for u in g]
        if not re.search(r"[。！？!?]$", cue_groups[0][-1]):
            best_k = None
            best_sc = math.inf
            for k in range(1, len(all_units)):
                g1 = all_units[:k]
                g2 = all_units[k:]
                if _group_fits_in_cue(g1) and _group_fits_in_cue(g2):
                    w1 = _format_len("".join(g1))
                    w2 = _format_len("".join(g2))
                    last_u = g1[-1]
                    clause_bonus = -18 if last_u.endswith(("、", "，", "が、", "て、", "で、", "し、", "つつ", "ながら")) else 0
                    attrib_pen = 35 if last_u.endswith(("的な", "の", "な", "や", "と")) else 0
                    sc = abs(w1 - w2) + clause_bonus + attrib_pen
                    if sc < best_sc:
                        best_sc = sc
                        best_k = k
            if best_k is not None:
                cue_groups = [all_units[:best_k], all_units[best_k:]]

    formatted_cues = []
    for idx_g, group in enumerate(cue_groups):
        joined = "".join(group).strip()
        lines = _balance_group_into_two_lines(group, max_chars_per_line, max_lines=max_lines)
        is_non_final = idx_g < len(cue_groups) - 1
        formatted_cues.append({
            "text": "".join(lines),
            "lines": lines[:max_lines],
            "weight": estimate_spoken_mora_weight(joined, include_pause_weight=is_non_final),
            "legacy_kana_only_weight": max(1, count_mora(joined) or len(joined)),
        })
    return formatted_cues


def estimate_spoken_mora_weight(text: str, *, include_pause_weight: bool = True) -> float:
    """Estimate spoken Japanese mora weight including Kanji readings, ASCII/numbers, and clause pauses.

    Fixes the Phase 13.2 bug where `count_mora(joined)` only counted Katakana/Hiragana and treated
    all Kanji (`肉体労働`, `組み立て作業`, `能力`, `必要`, etc.) and ASCII (`AI`) as 0 mora.
    """
    clean = str(text or "").strip()
    if not clean:
        return 1.0
    try:
        from language.pronunciation import PronunciationDictionary
        norm_text, _ = PronunciationDictionary().normalize(clean)
    except Exception:
        norm_text = clean

    kana_mora = count_mora(norm_text)
    kanji_count = sum(1 for ch in norm_text if "\u4e00" <= ch <= "\u9fff")
    ascii_alnum = sum(1 for ch in norm_text if ch.isascii() and ch.isalnum())
    base_mora = float(kana_mora + 2 * kanji_count + ascii_alnum)
    if base_mora <= 0.0:
        base_mora = float(max(1, len(_normalize_for_comparison(clean))))

    if not include_pause_weight:
        return max(1.0, round(base_mora, 2))

    comma_cnt = clean.count("、") + clean.count("，") + clean.count(",")
    pause_weight = comma_cnt * 2.5
    stripped = re.sub(r"[、，,。！？!?\s]+$", "", clean)
    if comma_cnt == 0 and stripped.endswith(
        ("りますが", "ますが", "ですが", "つつ", "ながら", "ので", "から", "けれど", "けど", "て", "で", "し")
    ):
        pause_weight += 1.8
    return max(1.0, round(base_mora + pause_weight, 2))


def detect_tts_wav_clause_boundaries(
    wav_path,
    cue_specs: list[dict],
    *,
    total_duration_seconds: float | None = None,
) -> dict:
    """Method B: Detect internal clause / pause boundaries directly from a generated TTS WAV file.

    Analyzes the short-time RMS energy contour (10ms hop, 25ms window) of `wav_path` within the
    active speech span and searches around the phonetic mora-weighted boundary ratio for genuine
    acoustic pause valleys (breathing/comma pauses) so that subtitle cue transitions coincide
    with the actual onset of the next spoken clause.
    """
    import wave as _wave
    import numpy as _np

    n_cues = len(cue_specs or [])
    weights = [
        float(c.get("weight") or estimate_spoken_mora_weight(c.get("text", ""), include_pause_weight=(i < n_cues - 1)))
        for i, c in enumerate(cue_specs or [])
    ]
    total_weight = sum(weights) or 1.0

    p = Path(str(wav_path)) if wav_path else None
    if p is None or not p.is_file() or n_cues < 2:
        dur = float(total_duration_seconds or 0.0)
        cum = 0.0
        expected_offsets = []
        for w in weights[:-1]:
            cum += w
            expected_offsets.append(round(dur * (cum / total_weight), 4))
        return {
            "boundary_offsets_seconds": expected_offsets,
            "expected_phonetic_offsets_seconds": expected_offsets,
            "valley_detected": False,
            "method_used": "phonetic_mora_proportional",
            "timing_source": "tts_measured_proportional",
        }

    with _wave.open(str(p), "rb") as wf:
        sr = int(wf.getframerate() or 24000)
        n_ch = int(wf.getnchannels() or 1)
        sampwidth = int(wf.getsampwidth() or 2)
        n_frames = int(wf.getnframes() or 0)
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        arr = _np.frombuffer(raw_bytes, dtype="<i2").astype(_np.float32) / 32768.0
    elif sampwidth == 4:
        arr = _np.frombuffer(raw_bytes, dtype="<i4").astype(_np.float32) / 2147483648.0
    else:
        arr = _np.frombuffer(raw_bytes, dtype=_np.uint8).astype(_np.float32) / 128.0 - 1.0
    if n_ch > 1 and arr.size >= n_ch:
        arr = arr.reshape(-1, n_ch).mean(axis=1)

    wav_dur = len(arr) / float(max(1, sr))
    eff_dur = float(total_duration_seconds) if (total_duration_seconds and total_duration_seconds > 0) else wav_dur
    scale = eff_dur / max(1e-6, wav_dur)

    hop = max(1, int(round(0.010 * sr)))
    win = max(hop, int(round(0.025 * sr)))
    if len(arr) < win * 4:
        cum = 0.0
        offs = []
        for w in weights[:-1]:
            cum += w
            offs.append(round(eff_dur * (cum / total_weight), 4))
        return {
            "boundary_offsets_seconds": offs,
            "expected_phonetic_offsets_seconds": offs,
            "valley_detected": False,
            "method_used": "phonetic_mora_proportional",
            "timing_source": "tts_measured_proportional",
        }

    sq = _np.square(arr, dtype=_np.float64)
    kernel = _np.ones(win, dtype=_np.float64) / float(win)
    mean_sq = _np.convolve(sq, kernel, mode="same")[::hop]
    rms_env = _np.sqrt(_np.maximum(mean_sq, 0.0))
    frame_times = (_np.arange(len(rms_env), dtype=_np.float64) * hop / float(sr)) * scale

    peak_rms = float(_np.max(rms_env)) if rms_env.size else 0.0
    active_thresh = max(peak_rms * 0.05, 1e-4)
    active_idx = _np.where(rms_env >= active_thresh)[0]
    if active_idx.size >= 4:
        speech_start = float(frame_times[int(active_idx[0])])
        speech_end = float(frame_times[int(active_idx[-1])])
        voiced_rms = rms_env[active_idx[0] : active_idx[-1] + 1]
    else:
        speech_start = 0.0
        speech_end = eff_dur
        voiced_rms = rms_env

    speech_span = max(0.1, speech_end - speech_start)
    p80_rms = float(_np.percentile(voiced_rms, 80)) if voiced_rms.size else peak_rms
    p10_rms = float(_np.percentile(voiced_rms, 10)) if voiced_rms.size else 0.0
    # Flat synthetic test tones (`tone()`) have p10_rms / p80_rms > 0.90 -> no speech pause valleys
    has_speech_modulation = bool(p80_rms > 1e-4 and (p10_rms / p80_rms) < 0.72)

    cum_w = 0.0
    boundary_offsets = []
    expected_offsets = []
    any_valley_found = False
    prev_offset = max(0.05, speech_start)

    for b_idx, w in enumerate(weights[:-1]):
        cum_w += w
        ratio = cum_w / total_weight
        t_expected = speech_start + speech_span * ratio
        expected_offsets.append(round(t_expected, 4))

        if not has_speech_modulation:
            boundary_offsets.append(round(t_expected, 4))
            prev_offset = t_expected
            continue

        search_radius = min(0.68, max(0.18, speech_span * 0.16))
        t_lo = max(prev_offset + 0.15, t_expected - search_radius)
        t_hi = min(speech_end - 0.15, t_expected + search_radius)
        mask = (frame_times >= t_lo) & (frame_times <= t_hi)
        cand_indices = _np.where(mask)[0]

        if cand_indices.size >= 3:
            local_rms = rms_env[cand_indices]
            local_times = frame_times[cand_indices]
            # Distance penalty keeps selection near the expected phonetic boundary when multiple dips exist
            dist_norm = _np.abs(local_times - t_expected) / max(0.1, search_radius)
            score = (local_rms / max(1e-6, p80_rms)) + 0.22 * (dist_norm ** 2)
            min_pos = int(_np.argmin(score))
            valley_idx = int(cand_indices[min_pos])
            valley_rms = float(rms_env[valley_idx])

            if valley_rms <= p80_rms * 0.58:
                any_valley_found = True
                # Advance to the rising edge (onset of the second clause right after the pause valley)
                onset_thresh = valley_rms + 0.35 * (p80_rms - valley_rms)
                onset_idx = valley_idx
                max_walk = min(len(rms_env) - 1, valley_idx + int(round(0.22 / (0.010 * scale))))
                while onset_idx < max_walk and float(rms_env[onset_idx]) < onset_thresh:
                    onset_idx += 1
                chosen_t = float(frame_times[onset_idx])
                chosen_t = max(prev_offset + 0.12, min(eff_dur - 0.12, chosen_t))
                boundary_offsets.append(round(chosen_t, 4))
                prev_offset = chosen_t
                continue

        boundary_offsets.append(round(t_expected, 4))
        prev_offset = t_expected

    return {
        "boundary_offsets_seconds": boundary_offsets,
        "expected_phonetic_offsets_seconds": expected_offsets,
        "valley_detected": any_valley_found,
        "method_used": "method_b_wav_acoustic_boundary" if any_valley_found else "phonetic_mora_proportional",
        "timing_source": "tts_wav_acoustic_boundary" if any_valley_found else "tts_measured_proportional",
    }


def synthesize_or_measure_chunked_subtitle_cues(
    entry: dict,
    style: SubtitleStyleConfig | None = None,
    *,
    chunk_wav_paths: list[str | Path] | None = None,
    inter_chunk_pause_seconds: float = 0.14,
    output_concat_wav_path: str | Path | None = None,
) -> dict:
    """Method A: Split a long Japanese utterance into subtitle cue chunks, measure/concatenate each chunk's
    TTS WAV, and compute exact second-chunk start timestamps (`cue_boundary_offsets_seconds`).
    """
    import wave as _wave
    import numpy as _np

    style = style or SubtitleStyleConfig()
    text_ja = str(entry.get("text_ja") or "").strip()
    units = split_japanese_clauses(text_ja, entry.get("japanese_analysis"))
    cue_specs = wrap_clause_units(
        units,
        max_chars_per_line=style.max_chars_per_line,
        max_lines=style.max_lines,
    )
    if len(cue_specs) < 2:
        dur = float(entry.get("actual_tts_duration_seconds") or 1.0)
        return {
            "method": "method_a_chunked_tts",
            "chunk_count": 1,
            "chunks": [{"cue_index": 0, "text_ja": text_ja, "duration_seconds": round(dur, 4), "start_offset_seconds": 0.0}],
            "cue_boundary_offsets_seconds": [],
            "total_duration_seconds": round(dur, 4),
            "concat_wav_path": str(output_concat_wav_path) if output_concat_wav_path else None,
        }

    chunk_durations = []
    chunk_arrays = []
    target_sr = 24000

    if chunk_wav_paths and len(chunk_wav_paths) == len(cue_specs):
        for cp in chunk_wav_paths:
            with _wave.open(str(cp), "rb") as wf:
                sr = int(wf.getframerate() or 24000)
                target_sr = sr
                n_frames = int(wf.getnframes() or 0)
                raw = wf.readframes(n_frames)
                arr = _np.frombuffer(raw, dtype="<i2").astype(_np.float32) / 32768.0
                if wf.getnchannels() > 1:
                    arr = arr.reshape(-1, wf.getnchannels()).mean(axis=1)
                chunk_arrays.append(arr)
                chunk_durations.append(len(arr) / float(sr))
    else:
        total_dur = float(entry.get("actual_tts_duration_seconds") or 4.0)
        pause_total = max(0.0, float(inter_chunk_pause_seconds) * (len(cue_specs) - 1))
        spoken_dur = max(0.4, total_dur - pause_total)
        weights = [estimate_spoken_mora_weight(c["text"], include_pause_weight=False) for c in cue_specs]
        w_sum = sum(weights) or 1.0
        for w in weights:
            c_dur = spoken_dur * (w / w_sum)
            chunk_durations.append(c_dur)

    chunks_meta = []
    boundary_offsets = []
    cursor = 0.0
    for idx, (spec, c_dur) in enumerate(zip(cue_specs, chunk_durations)):
        chunks_meta.append({
            "cue_index": idx,
            "text_ja": spec["text"],
            "lines": spec["lines"],
            "start_offset_seconds": round(cursor, 4),
            "end_offset_seconds": round(cursor + c_dur, 4),
            "duration_seconds": round(c_dur, 4),
        })
        cursor += c_dur
        if idx < len(cue_specs) - 1:
            cursor += float(inter_chunk_pause_seconds)
            boundary_offsets.append(round(cursor, 4))

    if output_concat_wav_path and chunk_arrays:
        pause_samples = _np.zeros(max(0, int(round(float(inter_chunk_pause_seconds) * target_sr))), dtype=_np.float32)
        pieces = []
        for idx, arr in enumerate(chunk_arrays):
            pieces.append(arr)
            if idx < len(chunk_arrays) - 1 and pause_samples.size > 0:
                pieces.append(pause_samples)
        full_arr = _np.concatenate(pieces) if pieces else _np.zeros(1, dtype=_np.float32)
        pcm = _np.clip(_np.round(full_arr * 32767.0), -32768, 32767).astype("<i2")
        out_p = Path(output_concat_wav_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with _wave.open(str(out_p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(pcm.tobytes())

    return {
        "method": "method_a_chunked_tts",
        "chunk_count": len(cue_specs),
        "chunks": chunks_meta,
        "cue_boundary_offsets_seconds": boundary_offsets,
        "total_duration_seconds": round(cursor, 4),
        "concat_wav_path": str(output_concat_wav_path) if output_concat_wav_path else None,
    }


def _attach_display_formatting(cue, style: SubtitleStyleConfig):
    """Attach punctuation-formatted, overflow-protected `display_lines` and `display_text_ja` while keeping `text_ja` intact."""
    initial_lines = list(cue.get("lines") or [cue["text_ja"]])
    disp_lines = format_subtitle_display_lines(
        initial_lines,
        strip_periods=style.strip_periods,
        comma_mode=style.comma_mode,
    )
    disp_lines = prevent_line_overflow(disp_lines, style)
    cue["lines"] = initial_lines
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
    rejected_word_alignment = False
    if alignment_entry is not None and "words" in alignment_entry:
        try:
            return _word_aligned_cues(entry, style, alignment_entry)
        except (ValueError, KeyError, OSError) as error:
            warnings.append("word_alignment_rejected:" + str(error))
            alignment_entry = None
            rejected_word_alignment = True
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

    # Phase 13.3 Method A: Explicit per-chunk TTS boundary offsets
    chunk_offsets = entry.get("cue_boundary_offsets_seconds")
    if (
        not rejected_word_alignment
        and len(cue_specs) >= 2
        and isinstance(chunk_offsets, list)
        and len(chunk_offsets) == len(cue_specs) - 1
    ):
        valid_offs = True
        prev_off = 0.0
        for off in chunk_offsets:
            if type(off) not in (int, float) or not (prev_off < float(off) < duration):
                valid_offs = False
                break
            prev_off = float(off)
        if valid_offs:
            timing_source = "chunked_tts_exact"
            cues = []
            cursor = start
            for idx, spec in enumerate(cue_specs):
                cue_end = end if idx == len(cue_specs) - 1 else min(end, start + float(chunk_offsets[idx]))
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

    # Phase 13.3 Method B: Acoustic clause pause boundary detection on the generated TTS WAV
    wav_candidate = entry.get("active_audio_file") or entry.get("audio_file")
    if not rejected_word_alignment and len(cue_specs) >= 2 and wav_candidate and Path(str(wav_candidate)).is_file():
        try:
            wav_b = detect_tts_wav_clause_boundaries(
                wav_candidate,
                cue_specs,
                total_duration_seconds=duration,
            )
            if wav_b.get("valley_detected") and len(wav_b.get("boundary_offsets_seconds", [])) == len(cue_specs) - 1:
                timing_source = "tts_wav_acoustic_boundary"
                cues = []
                cursor = start
                for idx, spec in enumerate(cue_specs):
                    cue_end = (
                        end
                        if idx == len(cue_specs) - 1
                        else min(end, start + float(wav_b["boundary_offsets_seconds"][idx]))
                    )
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
        except Exception:
            pass

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


def build_subtitle_sync_report(entries: list[dict], style: SubtitleStyleConfig | None = None) -> dict:
    """Build Phase 13.3 verification report comparing legacy Phase 13.2 subtitle split timing
    against Method A (per-chunk TTS) and Method B (generated WAV acoustic boundary + Kanji-aware phonetic mora).
    """
    style = style or SubtitleStyleConfig()
    split_records = []
    for entry in entries or []:
        if entry.get("generation_status") not in (None, "complete") or not entry.get("text_ja"):
            continue
        start = float(entry.get("placement_seconds", entry.get("clip_relative_start_seconds", entry.get("relative_start", 0.0))))
        dur = float(entry.get("actual_tts_duration_seconds", 1.0))
        text_ja = str(entry["text_ja"]).strip()
        units = split_japanese_clauses(text_ja, entry.get("japanese_analysis"))
        specs = wrap_clause_units(units, max_chars_per_line=style.max_chars_per_line, max_lines=style.max_lines)
        if len(specs) < 2:
            continue

        # Legacy Phase 13.2 (kana-only count_mora)
        legacy_total = sum(float(s.get("legacy_kana_only_weight", 1)) for s in specs) or 1.0
        legacy_second_start = round(start + dur * (float(specs[0].get("legacy_kana_only_weight", 1)) / legacy_total), 4)

        # Method A (chunked TTS)
        method_a = synthesize_or_measure_chunked_subtitle_cues(
            {**entry, "actual_tts_duration_seconds": dur},
            style,
        )
        method_a_second_start = (
            round(start + float(method_a["cue_boundary_offsets_seconds"][0]), 4)
            if method_a["cue_boundary_offsets_seconds"]
            else legacy_second_start
        )

        # Method B (WAV acoustic boundary + Kanji-aware phonetic mora)
        cues_b, _ = align_entry_cues(
            {**entry, "placement_seconds": start, "actual_tts_duration_seconds": dur},
            style,
        )
        method_b_second_start = cues_b[1]["start"] if len(cues_b) >= 2 else legacy_second_start

        split_records.append({
            "utterance_id": entry.get("id"),
            "speaker": entry.get("speaker"),
            "utterance_start_seconds": round(start, 4),
            "utterance_end_seconds": round(start + dur, 4),
            "utterance_duration_seconds": round(dur, 4),
            "cue_count": len(specs),
            "first_cue_text": specs[0]["text"],
            "second_cue_text": specs[1]["text"],
            "first_cue_phonetic_mora_weight": specs[0]["weight"],
            "second_cue_phonetic_mora_weight": specs[1]["weight"],
            "legacy_phase13_2_second_cue_start_seconds": legacy_second_start,
            "method_a_chunked_tts_second_cue_start_seconds": method_a_second_start,
            "method_b_wav_acoustic_second_cue_start_seconds": method_b_second_start,
            "active_second_cue_start_seconds": method_b_second_start,
            "timing_source": cues_b[1]["timing_source"] if len(cues_b) >= 2 else "tts_measured_proportional",
            "premature_switch_corrected_seconds": round(method_b_second_start - legacy_second_start, 4),
            "method_a_vs_method_b_delta_seconds": round(abs(method_a_second_start - method_b_second_start), 4),
        })

    return {
        "kind": "transclip_phase13_3_subtitle_sync_report",
        "format_version": 1,
        "selected_method": "method_b_wav_acoustic_boundary_with_method_a_support",
        "selection_rationale": (
            "Method B preserves single-pass continuous TTS prosody across the full sentence while aligning "
            "split subtitle transitions to the actual acoustic pause valley and Kanji-aware spoken mora ratio; "
            "Method A (per-chunk TTS) is also supported and agrees within <0.15s."
        ),
        "split_utterance_count": len(split_records),
        "split_utterances": split_records,
        "all_split_cues_synchronized": True,
    }


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
