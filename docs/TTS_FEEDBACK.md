# Phase 2: TTS Feedback Engine

## 接続と環境

`tts_generator.py --script` が `dubbing_script.json` v1を受け取り、既存Qwen3-TTSのCustomVoiceモデルでプリセット音声を生成する。従来の引数（`--script`なし）の動作は残している。

新規の重い依存はない。生成には既存のQwen専用Python、言語解析・全テストには `.venv-language` を使う。既存の仮想環境へインストールや設定変更をしない。台本の読み・モーラはJSONから受け取り、TTSプロセスにspaCyやSudachiを読み込まない。

`QwenPresetSynthesizer` は `language.voice.SpeechSynthesizer` の差し替え可能なアダプタ。モデルのロードは未生成のWAVが必要になるまで遅延する。参照音声やVoice Cloning、速度変更は受け付けない。

Phase 2の入口は常に `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` と `snapshot_download(local_files_only=True)` を使う。キャッシュにないモデルは失敗し、自動ダウンロードしない。READMEや.gitattributesのような実行に不要なモデル付属ファイルは要求しない。必要な設定・重み・speech tokenizerの存在を確認する。

## 実行

モデルをロードせず対象・話者・除外理由を確認:

```bash
cd ~/Projects/transclip
.venv-language/bin/python tts_generator.py XuoqKYxDHVc \
  --script input/XuoqKYxDHVc/dubbing_script.json --seconds 30 --dry-run
```

既存Qwen環境から実行:

```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python tts_generator.py XuoqKYxDHVc \
  --script input/XuoqKYxDHVc/dubbing_script.json --seconds 30 \
  --feedback-dir output/XuoqKYxDHVc/tts_feedback \
  --baseline-manifest output/XuoqKYxDHVc/tts/manifest.json
```

- `--entry-id dub-002`: 指定した発話だけ生成する。複数回指定可。
- `--max-jobs 1`: 1回の実行で新規に生成するWAV数の上限。キャッシュヒットは消費しない。同じコマンドの再実行で進められる。
- `--seed 0`: 再現用乱数seed。キャッシュキーにも含める。
- `--voice-map voices.json`: 例 `{"SPEAKER_00":"Ryan","SPEAKER_01":"Aiden"}`。未知話者へ推定で声を割り当てることはできない。
- `--model`: 既存ローカルキャッシュ内のCustomVoiceモデルID。モデル・声・設定が変われば別キャッシュになる。

まず問題の1項目を生成し、次に同じfeedback-dirで全項目を実行すると、その1項目のWAVは再利用される。`--script` の内容を直接手編集すると、既存の内容ハッシュが不一致になり拒否する。変更は候補JSON→`dubbing_script.py`→新しい台本ファイルの順に行う。

## 出力と互換性

```text
output/VIDEO_ID/tts_feedback/
  audio/SPEAKER_ID/CONTENT_HASH/
    speech.wav
    receipt.json
  runs/RUN_HASH/
    feedback.json
    comparison.md
```

- 入力台本と旧 `tts/manifest.json`、旧WAVは変更しない。Phase 1のJSONスキーマや従来TTSの入口も変更しない。
- 各entryの既存フィールド・出典・レビュー理由をコピーし、新しい測定フィールドを追加する。別形式の結果として `kind: tts_feedback`、`format_version: 1` を明示する。
- WAVは話者別ディレクトリへ保存。日時や台本の行番号だけに依存せず、モデルrevision・必要ファイルの識別情報・バックエンド版・生成設定・声・正確な日本語文からキーを作る。元時刻だけが変わった場合は音声を再利用して配置と衝突を再計算する。
- 音声と検証済みreceiptを一つのディレクトリとして公開する。1件成功ごとにフィードバックを保存し、途中失敗後はreceiptから成功分を復元する。失敗した一時ファイルは再利用しない。
- 出力ディレクトリは排他ロックする。同一出力への並行生成はエラーにする。既存キャッシュのWAVが壊れている・内容が変わった場合はエラーを記録し、無断で上書きしない。調査後に新しいfeedback-dirを選べる。
- 校正やレポート用のrunは入力台本・モデル・声設定・対象範囲に応じて分かれる。run内の進捗レポートは再実行時に更新する。

## 測定値

生成WAVはPCM16。`measure_wav` は実データを最後まで読み、ヘッダのフレーム数と一致することを検査する。

`actual_seconds = sample_frames / sample_rate`

ステレオの場合もチャンネル数で時間を二重に数えない。破損・途中切れ・空・完全無音は失敗扱い。波形ハッシュ・チャンネル数・sample width・ピーク・RMS・飽和サンプル率も保存する。浮動小数点WAVは現時点の計測器では対象外。

各entryに次を保存する:

- `id`, `source_utterance_ids`, `speaker`, `text_en`, `text_ja`
- `source_duration_seconds`: 元英語の区間長
- `estimated_duration_seconds`: Phase 1の推定値（上書きしない）
- `actual_tts_duration_seconds`: サンプルから計測したWAV長
- `prediction_error_seconds`: **実測−推定**
- `actual_minus_source_seconds`: **実測−元英語区間**
- `actual_playback_end`: 元の開始位置＋WAV全長
- `tts`, `voice`: モデルrevision・生成設定・ライブラリ版・声
- `review_required`, `review_reasons`, `generation_status`
- 読み、モーラ、句読点・ポーズ、元の時間推定設定

token上限に達したQwen生成は、完全な発話として保存しない。時間に合わせる目的のトリミング、atempo、次の話者の開始時刻による切断は行わない。

## 未確定話者・同時発話・衝突

UNKNOWN/null/REVIEWは保持し、音声を生成せず `skipped / unknown_speaker` として記録する。台本や原文から削除した意味ではない。明示的なプリセット割当のない既知話者もスキップ理由を記録する。

既知話者はそれぞれ元の開始位置に置く前提で診断する。同一話者の実測区間の重なりを `actual_same_speaker_conflicts` として警告する。異なる話者の重なりは `actual_other_speaker_overlaps` に保存し、元区間にも重なりがあったかを `source_overlap_seconds` で区別する。音声の長さや開始位置を自動変更しない。ミックス・最終動画出力はこの段階では行わない。

## 既存音声との比較

`--baseline-manifest` は、動画ID、発話IDの並び、話者、完全一致する旧日本語、相対開始/終了時刻を照合する。旧WAVもサンプルから再計測する。元のJSONに絶対clipがあればそれも照合する。

旧manifestにはseedや詳細な生成設定がないため、旧/新のWAV比較は厳密に条件統制した実験とはしない。旧モデル名とプリセットの一致、照合根拠を記録する。旧WAVを新しい日本語の音声として転用しない。

## 校正と候補再評価

`language.timing.feedback_profiles()` は、話者・プリセット・モデルrevision・バックエンド/生成設定・解析器・時間設定ごとに標本を分ける。同じキャッシュ音声は重複計上しない。読みが不確かなもの、2秒未満、音声の品質警告があるもの、未知話者を校正から除外する。

標本数、利用したID、音声キャッシュキー、モーラ毎秒の診断値を保存する。最低3標本を既定の明示利用条件にするが、3標本で十分な精度を保証するわけではない。**自動適用はしない。** 別話者や別モデルへ無条件に流用しない。

```python
from language.timing import estimate_with_profile
# scopeは対象話者・TTS設定・解析器・時間設定を表す完全な辞書。
result = estimate_with_profile(analysis, available_seconds, profile,
                               scope=current_scope, config=current_timing_config)
```

同じ標本への当てはまりと未知データへの精度を区別する。作用域が不一致、または標本不足ならエラー。

時間超過を改善する場合は既存の候補JSONを編集して、`dubbing_script.py --candidates ... --output NEW_SCRIPT.json` で再評価し、`tts_generator.py --script NEW_SCRIPT.json` に渡す。意味の検査は従来の最適化器に任せ、変更されていない文章のWAVは再利用する。フィードバックエンジンは日本語を勝手に短縮せず、新しい有料LLMにも接続しない。

## 検証

```bash
.venv-language/bin/python -m unittest discover -s tests -v
```

Phase 1の54件に加え、フレーム/サンプルレート、推定誤差、同一/別話者の重なり、UNKNOWN、キャッシュ、途中失敗・進捗保存失敗からの再開、入力非破壊、破損/無音WAV、旧WAV照合、旧CLI互換、校正の分離と明示利用、モデルのオフライン参照をテストする。合成した短いPCMと偽バックエンドで行い、テスト自体は実モデルやネットワークを使わない。

実データの先行検証では、元11.26秒の発話2〜5について最適化後WAV16.56秒を生成し、旧24.16秒から7.60秒の短縮を確認した。推定16.811秒との差は−0.251秒。ただし元区間からは5.30秒超過している。音声の自然さ・意味・発音の聴感確認はWAVを再生して行う。

macOSで再生するには、feedback.jsonの `audio_file` を使う:

```bash
afplay '/absolute/path/to/speech.wav'
```


## Phase 7の追加

許可済み参考音声によるQwen Baseアダプタ、登録CLI、Clone用キャッシュとMixer検証を追加した。本文のPhase 4時点の未実装記述については、[VOICE_CLONING.md](VOICE_CLONING.md)を参照。実際のクローン生成・発音・声質の検証は権限確認済み音声の準備後に行う。
