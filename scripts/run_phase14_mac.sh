#!/usr/bin/env bash
set -euo pipefail

# TransClip Phase 14 — Safe Mac Execution Script
# Uses ~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python and ffmpeg-full.
# Never deletes or overwrites existing output directories or input files.

VIDEO_ID="${1:-wJasRLuu_LA}"
OUTPUT_DIR="${2:-output/phase14_new_video_01}"
VOICE_MODE="${3:-preset}"

export PATH="/opt/homebrew/opt/ffmpeg-full/bin:/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

QWEN_VENV_PYTHON="${HOME}/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python"
if [ -x "${QWEN_VENV_PYTHON}" ]; then
  PYTHON_BIN="${QWEN_VENV_PYTHON}"
else
  echo "ERROR: Required real-model Python not found at ${QWEN_VENV_PYTHON}" >&2
  echo "Phase 14 production mode refuses to fall back to synthetic audio." >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Using ffmpeg: $(command -v ffmpeg)"

if [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" ]; then
  echo "ERROR: Output directory '${OUTPUT_DIR}' already exists and is not empty." >&2
  echo "Refusing to overwrite or delete existing files. Specify a new output directory." >&2
  exit 1
fi

echo "=== [1/4] Verifying CLI Registration (--help) ==="
"${PYTHON_BIN}" dubbing_pipeline.py --help | grep -q "run-phase14"
"${PYTHON_BIN}" dubbing_pipeline.py run-phase14 --help >/dev/null
echo "  OK: run-phase14 is registered in dubbing_pipeline.py"

echo "=== [2/4] Running Phase 14 Dedicated Tests & Phase 13.3 Non-Regression Suite ==="
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_phase14_*.py" -v
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py"

echo "=== [3/4] Running Phase 14 Real-Model Dubbing Pipeline for video_id=${VIDEO_ID} ==="
"${PYTHON_BIN}" dubbing_pipeline.py run-phase14 \
  --video-id "${VIDEO_ID}" \
  --voice-mode "${VOICE_MODE}" \
  --output "${OUTPUT_DIR}"

echo "=== [4/4] Verifying Generated Deliverables & Real-Model Execution ==="
for f in \
  "final_ja.mp4" \
  "final_ja_no_subtitles.mp4" \
  "dubbed_audio.wav" \
  "final_ja.ass" \
  "final_ja.srt" \
  "pipeline_report.json" \
  "script_adjustment_report.json" \
  "speaker_verification_targets.json" \
  "subtitle_sync_report.json" \
  "ducking_verification_report.json" \
  "lufs_normalization_report.json"; do
  if [ ! -s "${OUTPUT_DIR}/${f}" ]; then
    echo "ERROR: Expected deliverable missing or empty: ${OUTPUT_DIR}/${f}" >&2
    exit 1
  fi
  echo "  OK: ${OUTPUT_DIR}/${f}"
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}/pipeline_report.json" << 'PYVERIFY'
import json
import sys
from pathlib import Path

rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = rep.get("tts_execution_mode")
is_real = rep.get("is_real_qwen_model_generation")
print(f"  tts_execution_mode: {mode} (is_real_qwen_model_generation={is_real})")
if not is_real or mode != "live_qwen3_tts_mlx":
    raise SystemExit(f"ERROR: Expected live_qwen3_tts_mlx generation, got {mode!r}")
PYVERIFY

echo "Phase 14 completed! Output saved to: ${OUTPUT_DIR}/final_ja.mp4"
