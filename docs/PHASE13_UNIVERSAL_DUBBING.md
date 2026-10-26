# TransClip Phase 13 & Phase 13.1 — Universal Video Dubbing Architecture & CLI Guide

## Overview

Phase 13 and Phase 13.1 upgrade TransClip into a universal Japanese video dubbing pipeline capable of handling arbitrary international interviews, dialogues, lectures, and multi-speaker videos without any hardcoded video IDs, speaker IDs, or timestamps.

### Key Capabilities (Phase 13 + Phase 13.1)

1. **Word-Level Timestamp Speech Synchronization (`timing_adjustment.py`, `dubbing_pipeline.py`) [Phase 13.1]**
   - Extracts word-level timestamps (`word_timestamps.json`, `utterances.json` `words`, or `source_segment_spans`).
   - Distinguishes vocal fillers (`um`, `uh`, `er`, `hmm`), discourse starters (`So,`, `Well,`, `Right, so`, `you know`, `I mean`), short hesitation starters before long pauses (`"But what about"` + `1.14s` pause), silence intervals, and substantive speech words.
   - Computes `effective_content_start` and `effective_content_end` so Japanese TTS never starts prematurely ("flying") over leading fillers or hesitation pauses.
   - Splits utterances with long internal pauses (`>= --long-pause-threshold`, default `0.55s`) into contiguous `speech_clusters` and aligns Japanese semantic clauses across those clusters.
   - Preserves full word-level timestamp series across merged utterances (`detect_and_merge_split_utterances`).
   - Respects `effective_content_start` (`Start Time`) even when the previous TTS finishes early, and supports back-calculated placement from `End Time` (`--sync-anchor end` and `--sync-anchor adaptive`).
   - Integrates `0.90..1.30x` speech rate adjustment (`compute_auto_speech_rate_ratio`) and automatic concise translation rebalancing (`concise_rewrite_for_timing_budget`, `--auto-rebalance-overlong`).

2. **Stabilized Audio Ducking & Ambient Preservation (`audio_mixer.py`, `dubbing_pipeline.py`) [Phase 13 & 13.1]**
   - **Continuous Ducking (`--mix-mode continuous-voiceover`, `--ducking-style continuous`, `--duck-gain 0.18`)**: Keeps original English audio suppressed at a constant `15–20%` (default `18%` = `0.18`) across the entire clip so original English never jumps back to `100%` during pauses between Japanese TTS utterances (`max_background_gain_jump_db == 0.0 dB`).
   - **Extended Release Ducking (`--mix-mode long-release-voiceover`, `--ducking-style long-release`, `--release-seconds 1.8`)**: Extends post-TTS release time to `>= 1.5s` (default `1.8s`) and bridges inter-utterance gaps for gradual transitions.
   - **Dynamic Voice-Over Ducking (`--mix-mode voiceover`, `--ducking-style dynamic`)**: Attenuates original audio during Japanese TTS (`--duck-gain 0.12`) with smooth raised-cosine fades.
   - **Ambient Separation (`--mix-mode ambient` / `hybrid-ambient`)**: Separates speaker voice from background room tone (`--ambient-backend auto|dsp|demucs|torchscript|convtasnet`).

3. **Automatic Clip Boundary Detection, Extension & Smooth Fade-Out (`timing_adjustment.py`, `audio_mixer.py`, `dubbing_pipeline.py`) [Phase 13.1]**
   - `detect_and_resolve_clip_boundaries` inspects `selection_start` and `selection_end` against source utterances to detect mid-utterance or mid-sentence cuts.
   - When `--auto-extend-clip` is enabled, automatically extends the clip end (up to `--max-clip-extension`, default `12.0s`) so unfinished sentences complete cleanly. When disabled (`--no-auto-extend-clip`), records boundary cuts in `pipeline_report.json`.
   - Applies smooth raised-cosine audio fade-out (`--boundary-fade-out`, default `0.35s`) and optional video fade-out (`--video-fade-out`) at clip boundaries.

4. **High-Visibility Japanese Subtitle UI (`japanese_subtitles.py`) [Phase 13]**
   - Scales font size by `1.25x` (`72px` at 1080p), renders a semi-transparent black background box (`座布団`, `BorderStyle=3`, opacity `0.60`), omits Japanese periods (`。`), converts commas (`、`) to half-width spaces, wraps at natural phrase boundaries, and verifies zero line overflow.

---

## CLI Usage

### 1. Run Full 13-Stage Phase 13.1 E2E Verification on a New Video Clip Interval

```bash
./scripts/run_phase13_1_mac.sh output/phase13_1_1836_1875_e2e
```

Or invoke `dubbing_pipeline.py` directly:

```bash
PYTHONPATH=".:tests" python3 dubbing_pipeline.py run-phase13-1-e2e \
  --input-dir input/XuoqKYxDHVc \
  --clip-start 1836.45 \
  --clip-end 1871.00 \
  --auto-extend-clip \
  --max-clip-extension 12.0 \
  --duck-gain 0.18 \
  --ducking-style continuous \
  --mix-mode continuous-voiceover \
  --sync-anchor adaptive \
  --output output/phase13_1_1836_1875_e2e
```

### 2. Run Phase 13.1 Dubbing Pipeline on Any Script & Video

```bash
PYTHONPATH=".:tests" python3 dubbing_pipeline.py run-phase13-1 \
  --video path/to/source.mp4 \
  --script path/to/dubbing_script.json \
  --source-utterances path/to/utterances.json \
  --word-timestamps path/to/word_timestamps.json \
  --use-word-level-sync \
  --sync-anchor adaptive \
  --long-pause-threshold 0.55 \
  --mix-mode continuous-voiceover \
  --ducking-style continuous \
  --duck-gain 0.18 \
  --unspoken-gain 0.18 \
  --auto-extend-clip \
  --max-clip-extension 12.0 \
  --boundary-fade-out 0.35 \
  --output output/my_phase13_1_dub
```
