#!/usr/bin/env bash
set -euo pipefail

# TransClip Phase 13.1 — Host Mac End-to-End Execution Script
# Runs all 13 stages on the new unused interval (1836.45s..1871.00s -> auto-extended to 1874.65s)
# using local Qwen3-TTS-MLX (.venv) and input/XuoqKYxDHVc/source.mp4 when present.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  :
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif [[ -x "/usr/local/bin/python3.13" ]]; then
  PYTHON_BIN="/usr/local/bin/python3.13"
else
  PYTHON_BIN="$(command -v python3)"
fi

OUTPUT_DIR="${1:-output/phase13_1_1836_1875_e2e}"
rm -rf "$OUTPUT_DIR"

echo "[Phase 13.1] Using Python: $PYTHON_BIN"
echo "[Phase 13.1] Running 13-stage E2E pipeline -> $OUTPUT_DIR"

PYTHONPATH=".:tests" "$PYTHON_BIN" dubbing_pipeline.py run-phase13-1-e2e \
  --input-dir "input/XuoqKYxDHVc" \
  --clip-start 1836.45 \
  --clip-end 1871.00 \
  --auto-extend-clip \
  --max-clip-extension 12.0 \
  --duck-gain 0.18 \
  --ducking-style continuous \
  --mix-mode continuous-voiceover \
  --sync-anchor adaptive \
  --output "$OUTPUT_DIR"

echo "[Phase 13.1] Completed! Deliverables written to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
