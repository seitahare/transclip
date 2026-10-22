# 旧TRANS統合：参考生成・実Clone・声質比較・小幅な配置調整

## 実装と検証の区別

2026-10-22（日本時間）時点。主リポジトリは `~/Projects/transclip`。旧TRANSの指定8モジュールとQwen workerを調査し、再利用の判断を [TRANS_COMPARISON.md](TRANS_COMPARISON.md) に記録した。既存のBaseネイティブAPI、許可検証、receipt、キャッシュ、任意数のプロフィールを維持し、重複する大きなサービス・GUI・速度変更は移植していない。

新しい架空日本語参考（12.32秒）をVoiceDesignで生成し、ユーザーが再利用許可と発声内容・単一話者・明瞭さ・境界を確認した後で登録した。Base実モデルで日本語3発話を生成し、既存Preset Ryanの同文3発話と比較した。参考音声の承認は生成後のCloneの品質承認ではない。その後ユーザーがCloneの「日本語として正しい」ことを確認し、WAVハッシュ付きの別レビューに保存した。全文・数値の正確な一致、声質の近さ、一貫性は未確認。

|発話|Preset WAV / 生成時間|Clone WAV / 生成時間|参考とのEmbedding cosine Preset / Clone|
|---|---:|---:|---:|
|1|6.080秒 / 8.384秒|4.240秒 / 8.261秒|0.286 / 0.804|
|2|6.160秒 / 3.024秒|4.857秒 / 4.637秒|0.298 / 0.846|
|3|4.640秒 / 2.246秒|3.680秒 / 4.337秒|0.429 / 0.846|

最初の生成時間は初期ロード込み。Presetは以前の同文・seed・設定による実測を再利用し、同時条件の速度ベンチマークとはしない。全WAVは24kHz mono PCM16、クリッピング0。Clone 2/3発話目は同一プロフィールの条件付けを再利用。WAV長はサンプル数/サンプルレートで測定する。

参考F0中央値は110.1Hz、Cloneは121.9〜141.7Hz（DSP推定）。文章間のcosineはClone 0.814〜0.868、Preset 0.715〜0.737。3発話から成功閾値を作らず、声色・イントネーション・発音・知覚上の一貫性を試聴対象として残す。特に発話2の「144兆回」はWhisperで「144調解」と認識され、誤発音か認識誤りかを自動断定しない。

新しい英語架空参考も12.48秒で生成したが、再利用許可（①OK）のみ確認済み。発声内容・品質（②）が未確認のため英語参考→日本語Cloneは未検証。旧 reference_v0.3.wav とPhase 4の実在人物参考は使用していない。

## 参考音声の生成から登録

以下はメインリポジトリ内で実行する例。`SYNTH_A` は任意の架空話者IDへ変更する。テキスト・声質指示ファイルを事前に用意し、実在人物の模倣を指示しない。既存モデルをローカル限定で利用し、未取得のモデルは自動取得しない。

```sh
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python voice_cloning.py design-reference \
  --speaker SYNTH_A --text-file /absolute/path/text_ja.txt \
  --instruction-file /absolute/path/fictional_voice.txt \
  --language Japanese --seed 24680 --fictional \
  --output output/synthetic_references
```

出力 `HASH/` は `reference.wav`, `reference.txt`, `generation.json`, `review.html`, `pending_review/`。モデル名/revision・指示・入力text・seed・設定・モデル/コードのハッシュ・WAV SHA-256・実測時間・波形検査を保存する。生成時点では常に未承認。`--fictional` は生成方針の明示であり、許可の代わりではない。同一入力ではハッシュ照合後に再利用し、破損や不一致は上書きしない。

試聴し、再利用権限、発声とtextの一致、単一話者、混入、明瞭さ、語頭/語尾を人が確認する。確認した内容だけをテンプレートから**別ファイル**へ記録する。`generation.json` と未承認テンプレートを承認済みに書き換えない。許可・レビューの必須項目は [VOICE_CLONING.md](VOICE_CLONING.md) を参照。

```sh
.venv-language/bin/python voice_cloning.py register \
  --speaker SYNTH_A \
  --reference-audio /absolute/path/HASH/reference.wav \
  --reference-text /absolute/path/HASH/reference.txt \
  --generation-record /absolute/path/HASH/generation.json \
  --consent /absolute/path/reviewed/consent.json \
  --review /absolute/path/reviewed/review.json \
  --output output/voice_registry
```

`--generation-record` は合成生成の出典を結び付けるオプション。これだけでは登録不可。手元の許可済み録音を登録する既存フローも維持する。複数話者はそれぞれ登録し、`generate --profiles` を複数指定する。話者数・動画ID・参考パスはコードに固定していない。

英語参考は英文と英語の声質指示を用意して `--language English` を使う。日本語参考の承認を英語参考に流用しない。承認済みなら同じBase生成CLIで日本語台本を与える構成で、実測の品質評価は別途必要。

## 日本語生成と比較

既存の `dubbing_script.json` を指定する。選択範囲、話者、許可、品質証拠、参照SHAを検証してから生成する。

```sh
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python voice_cloning.py generate VIDEO_ID \
  --script /absolute/path/dubbing_script.json --mode approved_clone \
  --profiles /absolute/path/voice_profiles.json --seed 12345 \
  --seconds 30 --max-jobs 3 --output output/VIDEO_ID/clone
```

同じコマンドで途中再開。Presetは `--mode preset --voice-map voices.json` で別出力へ。同じ台本のFeedback同士は既存 `compare` を使用する。独立した参考/Preset/Clone WAVの音響比較は次のv1マニフェストを使う。パスはマニフェスト基準か絶対パス。

```json
{
  "kind": "voice_quality_request",
  "format_version": 1,
  "references": {"SYNTH_A": "reference.wav"},
  "samples": [{
    "id": "sample-01", "speaker": "SYNTH_A", "text_ja": "新しい技術を確かめます。",
    "preset_audio": "preset.wav", "clone_audio": "clone.wav",
    "preset_generation_wall_seconds": 8.0, "clone_generation_wall_seconds": 8.5
  }]
}
```

```sh
.venv-diarize/bin/python voice_cloning.py quality \
  --manifest /absolute/path/comparison_request.json \
  --output output/voice_quality --embeddings
```

既存pyannoteモデル/環境のみを使用する。`--embeddings` を省略すれば `.venv-language` でDSP比較可能。モデル推定値と波形の測定値を分け、1話者でも他話者比較の空集合をエラーにしない。`quality.json`, 音声コピー, `review.html` を保存。数値で自動承認せず、試聴結果は別記録にする。

## 時間調整とAudio Mixer

旧TRANSの「前発話の実測終端＋間隔」を利用する考え方を限定導入した。`--max-timing-shift` の既定は0（従来互換）、上限2秒。隣接する同一話者のグループにだけ遅延を提案し、元の同時発話・別話者・UNKNOWN・未吹き替え区間・プレビュー終端を越える提案を採用しない。元時刻と配置時刻を別フィールドに残す。

```sh
.venv-language/bin/python audio_mixer.py VIDEO_ID \
  --feedback /absolute/path/feedback.json --seconds 30 \
  --max-timing-shift 0.5 --diagnostic --dry-run
```

計画を確認後、`--dry-run` を外して診断プレビューを生成する。実測波形の切断・倍速化・原文の削除・話者補完は行わない。未解決の同一話者衝突や新しい別話者重複があれば、この調整オプションでは診断モードでも描画を拒否する。調整を採用した通常出力はタイミング試聴が必要な状態として停止する。今回の実装は診断評価までで、人手レビューの取込による最終出力解禁は次工程。

実Clone WAVの合成30秒テストでは0.24秒の衝突を0.28秒の遅延で解消し、完全なWAVを使ってMP4を出力・再利用した。人工的な開始時刻を使うテストで、元英語との同期精度を証明しない。Feedbackへ3 WAVを再利用し、入力JSON非破壊・モデル再ロード無しを確認した。推定誤差は−0.416 / −1.227 / −0.690秒で、校正用3標本を保存し自動適用しない。

開発動画の既存30秒Feedbackを再計画したところ、複数発話の結果では同一話者の衝突5組が残った。大きな時間超過は未解決。意味保持付き短縮候補→TTS再生成→実測、必要に応じた同一話者結合/ポーズ調整は後工程とする。旧速度変更・大規模repairは採用していない。

## テストと保存場所

開始時232件成功。合成参考8件、配置9件、声質比較6件を追加し、合計255件成功（18.105秒）。許可検証・参照変更検出・任意数話者・キャッシュ・途中失敗・異常WAV・UNKNOWN・同一/別話者重複・Mixer・入力非破壊を既存/追加テストで確認。これらのモック検証と実モデル生成を区別する。

```sh
.venv-language/bin/python -m unittest discover -s tests -v
```

今回のローカル実測成果物は `/Users/doiseita/Documents/Codex/2026-10-16/files-mentioned-by-the-user-transclip/outputs/phase7_integration/` に保存した。`index.html` が入口、`REPORT.md` が詳細、`comparison/` が試聴用WAV/HTMLと特徴量、`clone_ja/benchmark.json` が実生成記録、`mixes/` が診断MP4、`existing_timing_validation.json` が既存動画の未解決衝突、`all_tests.txt` がテストログ。生成記録・許可・検証出力はGit管理対象のコードと分離する。

新規大型モデル取得・有料API・既存環境の依存変更・commit/pushは行っていない。次は日本語Cloneの試聴確認、英語参考の承認後の英日実生成、大きな時間超過の台本短縮を優先する。
