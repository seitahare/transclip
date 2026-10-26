#!/usr/bin/env bash
# TransClip Phase 13.2 — Safe Local Mac Integration & Live Qwen3-TTS-MLX Execution Script
# Safely backs up existing files, applies Phase 13.2 modules, runs the full test suite,
# and executes the 13-stage Phase 13.2 E2E pipeline on the real source video.

set -euo pipefail

REPO_DIR="${1:-$HOME/Projects/transclip}"
CLIP_START="${CLIP_START:-1836.45}"
CLIP_END="${CLIP_END:-1871.0}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/output/phase13_2_verification}"

echo "============================================================"
echo " TransClip Phase 13.2 — Local Mac Execution & Verification"
echo "============================================================"
echo "Target Repository : $REPO_DIR"
echo "Clip Interval     : ${CLIP_START}s .. ${CLIP_END}s (auto-extends mid-sentence cut to 1874.65s)"
echo "Output Directory  : $OUT_DIR"

if [ ! -d "$REPO_DIR" ]; then
  echo "[ERROR] Repository directory not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

# 1. Create non-destructive backup of modified files before running
BACKUP_DIR="$REPO_DIR/.backup_before_phase13_2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
for f in voice_consistency.py dubbing_pipeline.py timing_adjustment.py japanese_subtitles.py prosody_experiments.py tts_generator.py; do
  if [ -f "$f" ]; then
    cp -p "$f" "$BACKUP_DIR/$f"
  fi
done
echo "[1/4] Backed up existing files to: $BACKUP_DIR"

# 2. Resolve Python environment (.venv preferred)
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[ERROR] python3 not found." >&2
  exit 1
fi
export PATH="/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "[2/4] Using Python: $PYTHON_BIN"

# 3. Run Phase 13.2 regression test suite (12 required test cases) + existing tests
echo "[3/4] Running Phase 13.2 regression tests..."
PYTHONPATH=".:tests" "$PYTHON_BIN" -m unittest \
  tests/test_phase13_2_improvements.py \
  tests/test_phase13_1_improvements.py \
  tests/test_phase13_improvements.py \
  tests/test_japanese_subtitles.py -v

# 4. Locate real source video if available
SOURCE_VIDEO_ARGS=()
for cand in \
  "$REPO_DIR/input/XuoqKYxDHVc/source.mp4" \
  "$REPO_DIR/output/XuoqKYxDHVc/source.mp4" \
  "$REPO_DIR/output/phase13_1_new_video_e2e/source.mp4"; do
  if [ -f "$cand" ]; then
    SOURCE_VIDEO_ARGS=(--source-video "$cand")
    echo "      Found real source video: $cand"
    break
  fi
done

rm -rf "$OUT_DIR"
echo "[4/4] Running Phase 13.2 E2E Pipeline (SpeakerVoiceRegistry + Natural 1.00x..1.20x Rate + BudouX 2-Line Subtitles)..."
"$PYTHON_BIN" dubbing_pipeline.py run-phase13-2-e2e \
  --input-dir "$REPO_DIR/input/XuoqKYxDHVc" \
  "${SOURCE_VIDEO_ARGS[@]}" \
  --clip-start "$CLIP_START" \
  --clip-end "$CLIP_END" \
  --auto-extend-clip \
  --duck-gain 0.18 \
  --ducking-style continuous \
  --mix-mode continuous-voiceover \
  --sync-anchor content_start \
  --output "$OUT_DIR"

echo ""
echo "============================================================"
echo " Phase 13.2 Complete! Generated Deliverables:"
echo "  - After MP4 (Phase 13.2)      : $OUT_DIR/after.mp4 ($OUT_DIR/final_ja_clone.mp4)"
echo "  - Before MP4 (Phase 13.1)     : $OUT_DIR/before.mp4"
echo "  - SPEAKER_00 Voice Samples    : $OUT_DIR/speaker_00_voice_samples.wav"
echo "  - SPEAKER_01 Voice Samples    : $OUT_DIR/speaker_01_voice_samples.wav"
echo "  - Voice Consistency Report    : $OUT_DIR/voice_consistency_report.json"
echo "  - Speaker Assignment Report   : $OUT_DIR/speaker_assignment_report.json"
echo "  - Timing Quality Report       : $OUT_DIR/timing_quality_report.json"
echo "  - Speaker Voice Registry      : $OUT_DIR/speaker_voice_registry.json"
echo "  - Pipeline Report             : $OUT_DIR/pipeline_report.json"
echo "============================================================"
