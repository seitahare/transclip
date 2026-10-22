# Phase 8: Cross-lingual Voice Cloning (英語参考音声 → 日本語音声生成)

## 1. 目的と設計原則
英語の架空参考音声（Phase 7で `Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` により生成された 12.48秒 / seed 24681 / SHA-256 `4a19ef1c00d3ee87d3afbb707f59166f830fe6a99c196a2df7d7fecdd7f73696`）から、`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` を用いて日本語音声3発話を生成し、Preset Voiceと比較評価する。

- **機械的検査と人間確認の分離**: WAV正常性・モノラルPCM16・長さ・クリッピング・語頭/語尾50ms振幅比・F0・SHA-256ハッシュ照合は機械的に自動検査する一方、発声内容と文字起こしの一致・単一話者性・明瞭さ・語頭語尾の聴感確認は人間による試聴確認が揃うまで自動承認しない。
- **日本語参考レビューの流用禁止**: 日本語参考音声（12.32秒 / seed 24680）の `consent.json` / `review.json` や生成レコードを英語参考音声の承認に代用することはハッシュ・言語検証により拒否される。
- **既存キャッシュの保護**: `voice_cloning.py`、`synthetic_reference.py`、`language/voice.py`、`tts_generator.py` のハッシュを変更せず独立モジュール `cross_lingual_clone.py` として実装しているため、既存のPhase 7生成キャッシュ・Presetキャッシュはそのまま再利用できる。

## 2. Step 1: 英語参考音声の検査と試聴確認シート出力

Macのローカル環境（`~/Projects/transclip`）で以下を実行すると、SHA-256（`4a19ef1c00d3ee87d3afbb707f59166f830fe6a99c196a2df7d7fecdd7f73696`）に一致する英語参考WAVを探索・機械検査し、試聴確認用HTML（`english_reference_review.html`）を出力する。

```bash
cd ~/Projects/transclip
.venv-language/bin/python cross_lingual_clone.py audit-reference \
  --speaker SYNTH_EN \
  --search-root output \
  --search-root /Users/doiseita/Documents/Codex/2026-10-16/files-mentioned-by-the-user-transclip/outputs/phase7_integration \
  --output output/phase8_cross_lingual/reference_audit
```

出力された `english_reference_review.html` をブラウザで開いて英語参考WAVを試聴し、問題がなければ画面上から `consent.json` と `review.json` をダウンロードする。未確認の状態では `run-benchmark` は終了コード `2`（`blocked_pending_human_review`）で安全に停止する。

## 3. Step 2: Qwen3-TTS Base による日本語3文の実生成と比較評価

試聴確認済みの `consent.json` と `review.json` を指定して、既存のQwen環境から以下を実行する。

```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python cross_lingual_clone.py run-benchmark \
  --speaker SYNTH_EN \
  --reference-audio /absolute/path/to/english/reference.wav \
  --reference-text /absolute/path/to/english/reference.txt \
  --generation-record /absolute/path/to/english/generation.json \
  --consent /absolute/path/to/reviewed/consent.json \
  --review /absolute/path/to/reviewed/review.json \
  --preset-voice Ryan \
  --output output/phase8_cross_lingual
```

Speaker Embedding類似度（pyannoteオフライン推論）も含めて評価する場合は、生成後に `.venv-diarize` で再実行するとキャッシュされたWAVを再利用してEmbedding類似度が追加計測される。

```bash
.venv-diarize/bin/python cross_lingual_clone.py run-benchmark \
  --speaker SYNTH_EN \
  --reference-audio /absolute/path/to/english/reference.wav \
  --reference-text /absolute/path/to/english/reference.txt \
  --generation-record /absolute/path/to/english/generation.json \
  --consent /absolute/path/to/reviewed/consent.json \
  --review /absolute/path/to/reviewed/review.json \
  --preset-voice Ryan \
  --embeddings \
  --output output/phase8_cross_lingual
```
