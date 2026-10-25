# TransClip Phase 12 — 実動画X-vector日本語吹き替えパイプライン (`dubbing_pipeline.py`)

## 1. 概要と目的

Phase 12では、実際の英語動画クリップ（`output/phase11_1885_1915/source.mp4`）と翻訳台本（`output/phase11_1885_1915_inputs/dubbing_script.json`）を入力とし、**元の話者の声質を再現できるX-vector方式（`x_vector_only_mode=True`, `ref_text=None`）を中心とした日本語吹き替え動画（`final_ja.mp4`）** を生成します。

---

## 2. Phase 12 の設計原則と実装仕様

### 2.1. X-vector音声合成の中心化とPreset自動フォールバック禁止（要件1）
- `run-phase12` サブコマンド（および `--voice-mode xvector` / `--no-preset-fallback`）では、すべての確定話者に対して **X-vector方式**（`QwenProsodyCloneSynthesizer(clone_method="xvector")`）による音声合成を行います。
- 参考音声の解決失敗やX-vector推論エラーが発生した場合に、**Preset音声（`Ryan` 等）へ自動的に切り替えて失敗を隠蔽することは一切ありません**（`XVectorSynthesisError` を送出して明示的に停止します）。

### 2.2. 技術的音声合成層と権利・公開管理層の分離（要件2）
- **技術的音声合成層 (`TechnicalVoiceReference` / `extract_speaker_reference_from_video` / `run_technical_xvector_feedback`)**:
  - X-vectorの音響条件付けに必要な参考WAVの検証（モノラルPCM16、長さ、SHA-256、話者別非重複区間のクリーン抽出）のみを担当します。
  - X-vector方式では `ref_text=None` で音響埋め込みのみを抽出するため、合成話者（`SYNTH_*`）や許可済み動画・参考WAVに対して不要な `consent.json` / `review.json` / `reference.txt` の手動登録手続きを要求しません。
- **権利確認・公開管理層 (`RightsAndPublicationPolicy`)**:
  - 技術層とは独立して各話者のソース種別（`synthetic_speaker`, `registered_permitted_profile`, `permitted_audio_source`）と公開可否ステータスを管理し、`pipeline_report.json["rights_and_governance"]` に記録します。
  - `consent.json` や `review.json` を偽造するような認証回避は一切行わず（`authentication_bypass_used: false`）、無許可の実在人物（`--speaker-source-type SPEAKER=unpermitted_real_person` や `--disallow-speaker`）のクローンを厳格に拒否します（`RightsPolicyViolationError`）。

### 2.3. 不自然に分割された一続きの文章の検出・統合（発話ID 64〜66 等）（要件3）
- `detect_and_merge_split_utterances()` により、同一話者の連続する分割発話（発話ID `64`, `65`, `66` 等）を自動検出（または `--merge-ids 64,65,66` / `--merge-ids 64-66` で明示指定）し、1つの自然な日本語文（ID `"64-66"`, `merged_from_ids: [64, 65, 66]`）に統合してからX-vector合成を行います。
- 分割断片の中間にある不自然な文末句点（`〜し。`、`〜ため。` 等）を接続読点（`、`）へ整形し、数値・固有名詞・否定表現を保持したままモーラ数を再計算します。
- 異なる話者、`speaker: null` / `"UNKNOWN"` 発話（バリアとして機能）、および長大な無音間隔（`> --max-merge-gap`）をまたぐ統合は行いません。

### 2.4. `speaker: null` 発話の検証対象管理（要件4）
- `partition_and_manage_null_speakers()` により、台本内で `"speaker": null`（または `"UNKNOWN"` / 空文字）となっている発話について、前後の話者から勝手に人物を推測・確定することを禁止しています（`auto_guessed_speaker_count: 0`, `inferred_identity_applied: false`）。
- 既定の `--null-speaker-policy hold_for_verification` では、`speaker: null` の発話をVoice Clone合成から保留し、`<output_dir>/speaker_verification_targets.json` および検証レポート（`pipeline_report.json` / `pipeline_report.md`）に検証対象として一覧出力します。
- 人間が確認した後に `--assign-speaker 67=SPEAKER_00` のように明示指定した場合のみ、人間による確定オーバーライドとして記録・合成します。

### 2.5. 日本語音声の実測長・時間超過・話者間重複の調整（要件5）
- 生成された各日本語WAVの実測長（`raw_tts_duration_seconds` / `actual_tts_duration_seconds`）をサンプル単位で計測します。
- 元の英語発話枠（`slot_duration_seconds`）を超過する場合や後続発話と衝突する場合、**元の生成WAVを上書きせず** `<output_dir>/_work/adjusted_wavs/` 配下に以下の非破壊調整を行います：
  1. 語頭・語尾の余剰デジタル無音のトリミング（25msの安全マージン保持）
  2. 同一話者衝突に対する開始時刻シフト（`--max-timing-shift`、既定 `1.2秒`）
  3. 日本語TTSの伸長によって生じた別話者ターンへの人工的スピル（元英語では非重複）に対するシフト調整（`--max-inter-speaker-shift`、既定 `0.6秒`）
  4. シフト後も残る時間超過に対する上限付きFFmpeg `atempo` 調整（`--max-atempo`、既定 `1.25倍`）
- 元英語の時点で複数話者が同時に発話していた自然な重なりはシフトせず独立トラックで保持し、ASS字幕では別レーン（`MarginV` 分離）に配置します。

### 2.6. 既存ファイルの非破壊保護（要件7）
- 入力動画（`output/phase11_1885_1915/source.mp4`）、入力台本（`output/phase11_1885_1915_inputs/dubbing_script.json`）、既存のWAV、翻訳ファイル、`VoiceProfile` のSHA-256ハッシュを実行前後に照合し、1バイトも変更されていないことを `pipeline_report.json["input_integrity_verification"]` に記録します。

---

## 3. Macでの実行可能なCLIコマンド

### 3.1. パッケージの展開と全テスト実行

```bash
cd ~/Projects/transclip
unzip -o /tmp/transclip-phase12-complete.zip -d .

# 全テスト実行（既存284件 + dubbing_quality 16件 = 300件）
PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH" .venv-language/bin/python -m unittest discover -s tests -v
```

### 3.2. Phase 12 標準実行（`output/phase11_1885_1915/source.mp4` ＋ `dubbing_script.json`）

`run-phase12` サブコマンドは、`output/phase11_1885_1915/source.mp4` から各話者の非重複英語音声区間を自動抽出してX-vector条件付けを行い、発話ID 64〜66等の分割文統合、`speaker: null` の検証対象分離、実測長調整、映像保持＋日本語音声・字幕合成までを一括実行します。

```bash
PYTHONPATH="$HOME/Projects/transclip/.venv-language/lib/python3.11/site-packages:$PYTHONPATH" \
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run-phase12 \
  --video output/phase11_1885_1915/source.mp4 \
  --script output/phase11_1885_1915_inputs/dubbing_script.json \
  --source-utterances input/XuoqKYxDHVc/utterances.json \
  --output output/phase12_1885_1915
```

### 3.3. 話者確認記録と再構成済み台本（ID 65・69の確定、64〜66統合、ID 68の4文分割）を適用して実行する場合

`output/phase12_1885_1915_quality/` には、音声SHA-256ハッシュに紐付けた発話ID 65・69の話者確認記録（`speaker_decisions.json`）、64〜66の統合・ID 68の4文常体分割を定義した編集計画（`editorial_plan.json`）、およびそれらを非破壊適用した新しい台本（`dubbing_script.revised.json`）が保存されています。

```bash
PYTHONPATH="$HOME/Projects/transclip/.venv-language/lib/python3.11/site-packages:$PYTHONPATH" \
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python dubbing_pipeline.py run-phase12 \
  --video output/phase11_1885_1915/source.mp4 \
  --script output/phase12_1885_1915_quality/dubbing_script.revised.json \
  --source-utterances input/XuoqKYxDHVc/utterances.json \
  --speaker-ref SPEAKER_01=output/prosody_experiments/reference/english_reference.wav \
  --output output/phase12_1885_1915_revised_run
```

- 特定話者に外部の参考WAVを指定したい場合は `--speaker-ref SPEAKER_01=path/to/ref.wav`（複数指定可）を追加できます。
- 生成されたWAVに対するWhisper等の単語時刻アライメントJSONを渡す場合は `--alignment-json path/to/alignment.json` を指定します。
- `--source-utterances input/XuoqKYxDHVc/utterances.json` を指定すると、30秒境界（`1915.0s`）をまたぐ発話ID `70` も含めた完全な発話網羅性・境界跨ぎ判定が `pipeline_report.json["completion"]` に記録されます。

---

## 4. 生成物一覧

`<OUTPUT_DIR>`（既定: `output/phase12_1885_1915`）に以下のファイルが生成されます：

| 生成物 | パス | 内容 |
| :--- | :--- | :--- |
| **日本語吹き替え動画（字幕付き）** | `output/phase12_1885_1915/final_ja.mp4` | 元動画の映像を保持し、X-vector日本語音声（48kHz AAC）と日本語字幕を合成したMP4 |
| **日本語吹き替え動画（字幕なし）** | `output/phase12_1885_1915/final_ja_no_subtitles.mp4` | 字幕焼き込み前の日本語吹き替えMP4 |
| **日本語字幕ファイル** | `output/phase12_1885_1915/final_ja.srt`<br>`output/phase12_1885_1915/final_ja.ass` | クリップ相対時刻（`0.0s .. 30.0s`）同期済み字幕ファイル |
| **吹き替えマスター音声WAV** | `output/phase12_1885_1915/dubbed_audio.wav` | 48kHz PCM16 ミックス済み日本語音声WAV（発話別WAVは `_work/tts_feedback/audio/*.wav`） |
| **検証レポート** | `output/phase12_1885_1915/pipeline_report.json`<br>`output/phase12_1885_1915/pipeline_report.md` | 技術合成と権利管理の分離結果、ID 64〜66等の統合結果、`speaker: null` 検証対象、実測長・重複調整、非破壊SHA-256検証レポート |
| **話者未確定（`speaker: null`）一覧** | `output/phase12_1885_1915/speaker_verification_targets.json` | 勝手な人物確定を行わず検証対象として管理された `speaker: null` 発話の詳細 |
