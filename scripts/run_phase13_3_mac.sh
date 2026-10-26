#!/usr/bin/env bash
# TransClip Phase 13.3 — Mac Execution & Verification Script
# Usage:
#   bash scripts/run_phase13_3_mac.sh [OUTPUT_DIR] [OPTIONAL_FULL_SOURCE_MP4]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${1:-output/phase13_3_verification}"
SOURCE_VIDEO="${2:-}"

if [ -x "/usr/local/bin/python3" ] && /usr/local/bin/python3 -c "import numpy" >/dev/null 2>&1; then
  PYTHON_BIN="/usr/local/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Error: python3 with numpy is required." >&2
  exit 1
fi

export PATH="/usr/local/bin:/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin:/opt/homebrew/bin:$PATH"

echo "=== [1/2] Running Phase 13.3 Regression Test Suite ==="
"$PYTHON_BIN" -m unittest tests/test_phase13_3_improvements.py -v

echo "=== [2/2] Running Phase 13.3 E2E Verification Pipeline ==="
rm -rf "$OUTPUT_DIR"
if [ -n "$SOURCE_VIDEO" ] && [ -f "$SOURCE_VIDEO" ]; then
  "$PYTHON_BIN" dubbing_pipeline.py run-phase13-3-e2e \
    --input-dir input/XuoqKYxDHVc \
    --source-video "$SOURCE_VIDEO" \
    --clip-start 1836.45 \
    --clip-end 1871.0 \
    --auto-extend-clip \
    --duck-gain 0.18 \
    --ducking-style continuous \
    --mix-mode continuous-voiceover \
    --sync-anchor content_start \
    --output "$OUTPUT_DIR"
else
  "$PYTHON_BIN" dubbing_pipeline.py run-phase13-3-e2e \
    --input-dir input/XuoqKYxDHVc \
    --clip-start 1836.45 \
    --clip-end 1871.0 \
    --auto-extend-clip \
    --duck-gain 0.18 \
    --ducking-style continuous \
    --mix-mode continuous-voiceover \
    --sync-anchor content_start \
    --output "$OUTPUT_DIR"
fi

echo "=== Phase 13.3 Deliverables Generated in: $OUTPUT_DIR ==="
ls -lh "$OUTPUT_DIR"
