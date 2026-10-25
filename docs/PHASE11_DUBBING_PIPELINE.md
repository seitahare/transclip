# Phase 11 — 30秒の日本語吹き替えMP4 E2Eパイプライン仕様・実行手順書 (`dubbing_pipeline.py`)

## 1. 概要と設計原則

Phase 11（`dubbing_pipeline.py`）は、Phase 1〜10および日本語韻律改善（`prosody_experiments.py` の X-vector Voice Cloning）を単一のCLIパイプラインとして統合し、**30秒の日本語吹き替え＋日本語字幕付きMP4（`final_ja.mp4`）** を完成させるエンドツーエンド・モジュールです。

1. **承認済みVoice Profileに対するX-vector方式の既定採用**:
   - 承認済みVoice Profile（例: 合成話者 `SYNTH_EN_01`）が存在する話者には、既定で `QwenProsodyCloneSynthesizer(clone_method="xvector")`（`x_vector_only_mode=True, ref_text=None`）を適用し、元話者の声色（ECAPA-TDNN 2048次元 Speaker Embedding）を維持しながら自然な日本語モーラ拍リズム・高低アクセントで発声します。
2. **CLIからの方式選択（`xvector` / `icl` / `preset`）**:
   - `--voice-mode {auto,xvector,icl,preset}` および `--clone-method {xvector,icl}` により、X-vector方式・ICL方式・Preset方式をCLIから自由に切り替えられます。
3. **未承認・実在人物の無断クローン防止（厳格な権限ゲート）**:
   - `voice_profiles.json` に登録・承認されていない話者（実動画内の未承認話者など）は **絶対にクローンしません**。
   - 未承認話者が台本に含まれる場合、`--preset-speaker SPEAKER_ID=Ryan` または `--default-preset-voice Ryan` で明示的にPreset音声を指定しない限り、`UnauthorizedSpeakerError` で実行を拒否します。
   - 承認済みClone話者（`SYNTH_EN_01`）と未承認Preset話者（`SPEAKER_00` 等）が同一クリップ内に混在する場合は、`QwenHybridDubbingSynthesizer` によって話者ごとにCloneバックエンドとPresetバックエンドを安全に振り分けます。
4. **同一話者衝突の検出と別話者の自然な重なりの保持**:
   - 同一話者の発話衝突（`same_speaker_collisions`）を検出し、`--max-timing-shift`（既定 `0.5s`）以内の小幅遅延で解消可能なものは自動調整します。解消不能な衝突は `pipeline_report.json` / `pipeline_report.md` に必要短縮秒数と目標モーラ上限（`target_mora_budget`）を明記します。
   - 異なる話者同士の重なり（`preserved_other_speaker_overlaps`）は自然な会話としてそのまま保持し、48kHz話者別トラックとASS字幕の別レーン（`MarginV`）で合成します。
5. **クリップ相対時刻（`0.00s .. 30.00s`）への統一**:
   - 元動画の選択範囲（例: `1880.0s .. 1910.0s` または `0.0s .. 30.0s`）にかかわらず、30秒動画クリップに対する吹き替え音声配置時刻と日本語字幕（`.srt` / `.ass`）の開始・終了時刻を **クリップ相対時刻（`0.00s .. 30.00s`）** に完全統一します。

---

## 2. Macローカルでの実行コマンド

### 2.1. パッチの展開と全自動テスト（既存273件＋Phase 11新規4件＝277件）の実行
```bash
cd ~/Projects/transclip
unzip -o /tmp/transclip-phase11-pipeline.zip -d .
.venv-language/bin/python -m unittest discover -s tests -v
```

### 2.2. Step 1：承認済み合成話者 `SYNTH_EN_01` を用いた安全な30秒検証素材でのE2E実行（X-vector方式）
既に `prosody_experiments.py` で生成済みの `xvector_run/feedback.json` がある場合は `--existing-feedback` でキャッシュ再利用でき、指定しない場合も自動でQwen3-TTS Base（`xvector` モード）により生成して30秒の `final_ja.mp4` を完成させます。

```bash
# A. Qwen3-TTS (X-vector方式) で実音声生成から30秒MP4・字幕合成まで一括実行する場合
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run-synth-30s \
  --speaker SYNTH_EN_01 \
  --reference-audio output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.wav \
  --reference-text output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.txt \
  --consent ~/Downloads/consent.json \
  --review ~/Downloads/review.json \
  --voice-mode xvector \
  --output output/phase11_synth_30s_xvector

# B. 既に生成済みの xvector_run/feedback.json を再利用して即座に30秒MP4を合成する場合
.venv-language/bin/python dubbing_pipeline.py run-synth-30s \
  --speaker SYNTH_EN_01 \
  --reference-audio output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.wav \
  --reference-text output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.txt \
  --consent ~/Downloads/consent.json \
  --review ~/Downloads/review.json \
  --voice-mode xvector \
  --existing-feedback output/prosody_experiments/_runs/xvector_run/feedback.json \
  --output output/phase11_synth_30s_xvector
```

### 2.3. Step 2：方式切り替え（ICL方式 / Preset方式）での30秒MP4生成
```bash
# ICL方式 (--voice-mode icl) を選択する場合
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run-synth-30s \
  --speaker SYNTH_EN_01 \
  --reference-audio output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.wav \
  --reference-text output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.txt \
  --consent ~/Downloads/consent.json \
  --review ~/Downloads/review.json \
  --voice-mode icl \
  --output output/phase11_synth_30s_icl

# Preset方式 (--voice-mode preset) を選択する場合
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run-synth-30s \
  --speaker SYNTH_EN_01 \
  --voice-mode preset \
  --default-preset-voice Ryan \
  --output output/phase11_synth_30s_preset
```

### 2.4. Step 3：実動画30秒クリップ（未承認話者にPreset方式を明示適用）でのE2E実行
```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run \
  --script output/dubbing_script.json \
  --video output/clip_1880_1910.mp4 \
  --voice-mode preset \
  --default-preset-voice Ryan \
  --max-timing-shift 0.5 \
  --allow-diagnostic-on-collision \
  --output output/phase11_real_clip_30s
```

---

## 3. 出力される成果物パス一覧

`<output_dir>/`（例: `output/phase11_synth_30s_xvector/`）直下に以下のファイルが出力されます。

- `final_ja.mp4`: 日本語吹き替え音声＋日本語字幕焼き込み済みの30秒MP4動画
- `final_ja_no_subtitles.mp4`: 日本語吹き替え音声のみ（字幕なし）の30秒MP4動画
- `final_ja.srt`: クリップ相対時刻（`00:00:xx,xxx`）に統一されたSRT日本語字幕
- `final_ja.ass`: 複数話者同時発話時のレーン分離（`MarginV`）に対応したASS日本語字幕
- `dubbed_audio.wav`: 48kHz PCM16の日本語吹き替えミックス音声WAV
- `pipeline_report.json` / `pipeline_report.md`: 話者別ルーティング・権限検証、同一話者衝突診断、別話者重複保持、FFprobeストリーム検証、相対時刻同期の検証レポート
