# Phase 10: Japanese Subtitle Engine (`japanese_subtitles.py`)

## 概要
翻訳直後のテキストではなく、最終的にTTSへ渡した日本語台本（`feedback.json` または `mix_report.json` の `text_ja`）と実測配置時刻（`placement_seconds` + `actual_tts_duration_seconds`）を基準として日本語字幕を生成する独立CLI。

## 主な機能
1. **最終TTS台本・実測音声との同期**: `audio_mixer.validate_feedback` により台本フィンガープリントとWAV実測値・receiptを照合し、配置時刻に同期した字幕を生成する。
2. **Whisper / Forced Alignment 照合（任意）**: `--alignment-json` が指定された場合、認識文字列と `text_ja` の食い違い（例: 同音異義語や数値の読み誤り候補）を `alignment_warnings` に記録し、台本文面を勝手に書き換えずレビュー対象として保持する。
3. **自然な文節分割と最大2行改行**: Sudachi形態素情報（`japanese_analysis.tokens`）または句読点・助詞境界を用いて、数値（`144兆回` 等）や英単語を分断せずに最大2行へ整形する。
4. **同時発話する複数話者の独立表示**: 異なる話者の同時発話（`other_speaker_overlaps`）では、ASSの垂直レーン（`MarginV`）と話者別スタイルを分離して重なり表示を防ぎ、SRTでは `[SPEAKER]` プレフィックスを付与する。同一話者の未解決衝突がある場合は通常モードで停止する。
5. **SRT / ASS / 字幕焼き込みMP4 / 字幕なしMP4出力**: `final_ja.srt`、`final_ja.ass`、および `--dubbed-video` 指定時には `final_ja_no_subtitles.mp4` と FFmpeg `ass` フィルタで焼き込んだ `final_ja.mp4` をアトミックに出力し、`validate_mp4` で検証する。
