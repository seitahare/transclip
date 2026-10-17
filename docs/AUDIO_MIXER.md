# Phase 3: Audio Mixer / 日本語吹き替えプレビュー

## 実装範囲

`audio_mixer.py` は Phase 2 の `feedback.json` と既存WAV、元動画、`selection.json` を読み、話者別WAV・日本語合成WAV・日本語吹き替えMP4・`mix_report.json` を作る独立CLI。翻訳、TTS、ネットワーク、モデルのロードは行わない。Python標準ライブラリと既存FFmpeg/ffprobeのみを使用する。通常の `.venv` で実行可能。

30秒MVPでは未解決の衝突と未吹き替えが残るため、完成品質の動画とは区別する。`main.py` への自動接続はまだ行っていない。

## 実行例

```bash
cd ~/Projects/transclip
# 実際に使用する Phase 2 レポートのパス
FEEDBACK=output/XuoqKYxDHVc/tts_feedback/runs/6902544cfae329b326a66dd7274e0582276636fc3672e79d66009330dc57bb7f/feedback.json

# 検証・配置予定・衝突・UNKNOWN・境界・生成阻止理由を表示。ファイルを生成しない。
.venv/bin/python audio_mixer.py XuoqKYxDHVc --feedback "$FEEDBACK" --seconds 30 --dry-run

# 日本語のみ。未解決項目がある現在のデータは --diagnostic が必要。
.venv/bin/python audio_mixer.py XuoqKYxDHVc --feedback "$FEEDBACK" --seconds 30 \
  --diagnostic --mode japanese-only --output-dir output/XuoqKYxDHVc/mixes

# 元英語を小さく残した比較用ミックス。音声分離ではない。
.venv/bin/python audio_mixer.py XuoqKYxDHVc --feedback "$FEEDBACK" --seconds 30 \
  --diagnostic --mode reference-mix --reference-gain 0.12 \
  --output-dir output/XuoqKYxDHVc/mixes
```

- `--selection`, `--source-video`, `--source-info`: 既定は `input/VIDEO_ID/selection.json`, `source.mp4`, `source.info.json`。元動画はメタデータと同じディレクトリの `source.*` を要求する。切り抜き済み動画に元動画の絶対開始時刻を重ねて適用しない。
- `--mode japanese-only`: 元動画の音声を使わず、日本語のみ。未吹き替え区間に別の日本語が重なっていなければ無音となる。
- `--mode reference-mix`: 英語音声を既定0.12倍（約−18.4 dB）で加算。許容値0〜0.5。背景音の分離や英語の除去は実装しない。
- `--diagnostic`: 衝突、未生成、レビュー未解決を明示的に許容。出力ファイル名とJSONに `diagnostic`、該当時は `conflicts`・`partial` を付ける。字幕や警告を映像に焼き込む処理はない。
- 通常モード: 未解決の同一話者衝突・未吹き替え・レビュー項目があれば停止。入力を手編集してレビューを消す方法は推奨しない。レビュー承認の専用ワークフローは今後の課題。
- `--extend-to-fit`: プレビューを越えるWAVがある場合のみ、WAV全長が入るまで延長。選択範囲を越える場合は診断モードでも停止。延長によって新たな台本項目が範囲内に入る場合は、より広い範囲のフィードバックを用意して再計画する。エンコード後のフレーム境界でも選択終端超過を検証する。
- dry-run は問題のない入力を読めれば終了0。生成を阻止する条件は `blockers` に表示する。本生成では `blockers` があると終了1。

## 再生

実行終了時に表示される `mix_report.json` の隣に、レポートの `video_file` と `japanese_audio_file` で指定したファイルが保存される。macOSでは以下で再生できる。

```bash
open '/実際の出力ディレクトリ/preview_30s_japanese_only_diagnostic_conflicts_partial.mp4'
afplay '/実際の出力ディレクトリ/preview_30s_japanese.wav'
```

## 時刻と話者別トラック

`absolute_time = selection.start + relative_start`。

動画は元動画の `selection.start` から切り出す。日本語音声はクリップ上の `relative_start` へ置く。たとえば実データの `dub-002` はクリップ1.90秒（元動画1655.50秒）から16.56秒間で、18.46秒まで再生する。元英語の終了13.16秒でも、別話者の開始でも切らない。

FFmpegで各WAVを48kHz/stereo/floatへ変換し、`adelay` のサンプル指定と `amix normalize=0` で話者別の独立トラックを作る。開始位置の丸めは最寄りの48kHzサンプル（最大約10.4μs）。サンプル数から決めた全長を保持し、後方の無音を足す。予期しない長さはエラーにし、発話を切るフィルタや速度変更は使わない。

話者別floatトラックを加算して日本語音声を作り、Reference Mixでは元動画音声も加算する。全トラック・日本語音声・最終音声の最大ピークから共通の固定ゲインを求め、必要な場合だけ−2dBFS以下へ下げる。個々の発話のタイミングや速度は変えない。話者別PCM16 WAVにも同じゲインを適用し、最終ミックスの調査に使えるようにする。リミッタや音圧正規化は行わない。

MP4は映像H.264・音声AAC、元の映像サイズを維持。ffprobeによるストリーム数・時間確認、全映像／音声デコード、AACデコード後の浮動小数点ピーク検査を通してから公開する。PCMの時間はフレーム数/48000で正確に保存し、MP4の映像時間には動画フレーム境界の丸めがある（最大50msまで検証）。AAC時間差は25ms以内を検査し、デコード後のフルスケール超過は失敗とする。

## 衝突・未吹き替え

WAV実測に基づく区間交差を再計算する。同一話者と異なる話者は別リストに記録し、同一話者の各ペアは1回だけ報告する。話者をまたいだ自然な同時発話を直列化・削除しない。診断モードでは同一話者の自己重複をそのまま聞ける。

UNKNOWN/null/REVIEWには音声を補完しない。未生成項目は元のID・原文・日本語・時刻・レビュー理由を含めて `undubbed_entries` に保存する。既知話者の pending/failed/skipped も未吹き替えとして残す。上流の `source_excluded` とプレビューに交差する除外区間も報告し、完全な吹き替えと誤認させない。

## 入力検証・非破壊・再実行

`feedback.json` v1、動画ID、選択区間、発話ID、原文、日本語、話者、相対／絶対時刻を検証する。元台本を `source_script_path` から読み、内容フィンガープリントと各発話を照合する。台本・receiptが失われた単体WAVのみの入力は受け付けない。

各complete項目について、実ファイルを最後まで読み、WAVのサンプル数・時間・SHA-256・ピーク等を再測定する。Phase 2の `wav_measurement`、隣接する `receipt.json`、TTSモデル・声・設定から再計算した音声キャッシュキーを照合する。破損、無音、欠落、異なる音声、設定の不一致は発話IDと原因を表示して停止。生成済みUNKNOWNを受け入れない。

元動画のメタデータIDとメディア長を照合し、元動画全体のSHA-256も保存する。これはローカル素材の取り違え・変更を検出するための来歴であり、動画の内容からYouTube IDを独立に証明する仕組みではない。

元のJSON/WAV/動画は変更しない。出力は `--output-dir` 以下の `MODE_QUALITY_HASH/` にまとめる。

```text
mixes/japanese_only_diagnostic_conflicts_partial_HASH/
  preview_30s_japanese_only_diagnostic_conflicts_partial.mp4
  preview_30s_japanese.wav
  preview_30s_japanese_only.wav
  speaker_SPEAKER_00.wav
  speaker_SPEAKER_01.wav
  mix_report.json
```

キーは入力フィンガープリント、WAVハッシュ、元動画ハッシュ、配置、モード、ゲイン設定、実装・FFmpeg版を含む。同じ入力の再実行は出力ハッシュを照合し、エンコードせず再利用。既存出力が破損していれば上書きせず停止する。別出力先で調査可能。

出力先を排他ロックし、成功した一式を一時ディレクトリからアトミックに公開する。FFmpeg失敗時には未公開の作業ファイルのみ片づけ、元データと既存完成出力を残す。再実行で作り直せる。

`mix_report.json` は `kind: audio_mix`, `format_version: 1`。配置、未生成、上流除外、衝突、別話者重複、境界、レビュー、静的ゲイン、WAV測定値、MP4検証、成果物ハッシュ、入力来歴を含む。Phase 2の入力スキーマは変更しない。

## テストと拡張

```bash
.venv-language/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s tests -p 'test_audio_mixer.py' -v
```

小さな正弦波WAVと64×48のローカル動画で、相対時刻、無音、同時発話、衝突、UNKNOWN、WAV欠落／破損、動画・設定不一致、終端延長、音量、非破壊、再利用、FFmpeg失敗を検証する。外部APIやTTSモデルは不要。FFmpeg/ffprobeがなければ新規テストを明示的にスキップする。

`validate_feedback` → `build_plan` → `render_plan` を分離し、将来メインCLIから利用できる。自動配置案は元の開始時刻・変更理由を保持する別計画として扱い、今回の無変更配置を上書きしない設計とする。短縮候補の再評価、UNKNOWNの人手確認、背景音分離、Voice Cloningは別フェーズ。

実データでは既知6件、UNKNOWN5件、同一話者衝突5組、WAVの30秒超過なし。上流で除外された発話15（27.28〜40.52秒）も冒頭30秒に交差する。11.26秒の元区間に対する16.56秒のWAVは5.30秒の超過が残り、口の動きや発話終端への一致は保証しない。機械的なデコード・音量検証と、発音・意味・自然さの聴感評価は区別する。
