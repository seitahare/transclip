# Language Engine / Dubbing Script MVP

## 実装範囲

2026-10-17時点。既存の取得・字幕・話者識別・翻訳・Qwen3-TTSのスクリプトは変更せず、既存JSONを読む独立CLIを追加した。

- `language/english.py`: spaCyのトークン化、文境界、品詞、依存構文、固有名詞、数字、否定、推測表現、断片の可能性を返す。`Analyzer` Protocolを満たす別実装に交換できる。
- `language/japanese.py`: SudachiPy core辞書による元テキストの形態素・品詞・読み、および発音正規化後の読み・モーラ・ポーズ。未知の読み、助数詞・日付は要確認。
- `language/pronunciation.py`: 数字・小数・割合・大文字略語の読み、固有名詞の辞書、モーラ計算。全文を解析するトークナイザは自作していない。小書き仮名は直前の仮名に結合し、促音・撥音・長音は各1モーラ。
- `language/rules.json` / `rulebook.py`: 編集可能な翻訳方針、名前の表記候補、発音辞書、ローカル短縮規則、数値・割合・名前・否定・推測の警告。数値の単位も比較する。漢数字と英語の数詞、否定の作用域、完全な意味同一性は未解決。
- `language/timing.py`: モーラ毎秒と句読点のポーズから時間・元区間との差を推定。PCM WAVヘッダからの実測と、モデル・声ごとの校正プロファイル生成関数も提供する。
- `dubbing_script.py`: 入力整合性、発話グルーピング、短縮候補の検査、独立タイムライン、衝突検出、出典、比較Markdown、JSON保存。
- `language/cache.py`: 将来の候補生成プロバイダ用Protocol、内容ハッシュのキャッシュ、再開・呼び出し上限。ネットワーク実装は含めない。API接続には明示的な許可と別途アダプタ実装が必要。
- `language/voice.py`: 将来のTTS用インターフェース。プリセット声と承諾記録を必要とする参照音声を区別する。音声の生成・クローン化は実装しない。

既存コードを確認した結果、`main.py`の接続範囲は動画取得〜区間選択〜切り抜きまで。話者識別、単語時刻、対応付け、発話再構成、翻訳、TTSは独立スクリプトとして存在する。最終ミックスと吹き替えMP4の統合は未実装。

## 環境

新依存は `requirements-language.txt` に分離した。spaCyと小型英語モデルを英語の品詞・構文・NERに、SudachiPyとcore辞書を日本語の形態素・読みに使う。pyopenjtalk、Torch、MLXは追加しない。モデルを実行時に自動ダウンロードしない。

```bash
cd ~/Projects/transclip
python3 -m venv .venv-language
.venv-language/bin/python -m pip install -r requirements-language.txt
.venv-language/bin/python -m unittest discover -s tests -v
```

元の `.venv`、`.venv-diarize`、Qwen環境は変更不要。`requirements.txt` も変更しない。英語モデルを利用できない場合は `--english-model blank` を明示できるが、品詞・構文・NERが欠けることを出力とレビュー理由に記録する。

根拠: [spaCy linguistic features](https://spacy.io/usage/linguistic-features)、[SudachiPy tutorial](https://github.com/WorksApplications/SudachiPy/blob/develop/docs/tutorial.md)。

## CLI

既存データを使い、ローカル短縮規則のみで生成する。API呼び出しはない。

```bash
.venv-language/bin/python dubbing_script.py XuoqKYxDHVc --seconds 30 \
  --output output/XuoqKYxDHVc/dubbing_script.local.json \
  --report output/XuoqKYxDHVc/dubbing_script.local.md
```

`--output` を省略すると `input/VIDEO_ID/dubbing_script.json`。同一結果は再利用する。異なる結果で既存ファイルを上書きしないので、設定変更時は新しい出力名を指定する。`--report` は任意で、比較表は標準出力にも表示する。

データがない環境でのフィクスチャ実行:

```bash
.venv-language/bin/python dubbing_script.py \
  --input-dir tests/fixtures/language_mvp \
  --output output/fixture/dubbing_script.json \
  --report output/fixture/comparison.md
```

今回の実データ向け候補を再現する場合（候補ファイルはローカルのGit管理外データ）:

```bash
.venv-language/bin/python dubbing_script.py XuoqKYxDHVc --seconds 30 \
  --candidates input/XuoqKYxDHVc/dubbing_candidates.json \
  --output input/XuoqKYxDHVc/dubbing_script.json \
  --report output/XuoqKYxDHVc/dubbing_comparison.md
```

追加オプション: `--rules PATH`、`--mora-per-second 7.0`。話速値は推定パラメータであり、WAVを倍速化するものではない。

## 短縮の扱い

既定では、既存訳を結合してルールブックの限定的な表現置換を試す。自由な再翻訳や一般的な意味の推敲をローカル規則だけで達成したとは扱わない。自然な言い換え候補は別JSONで入力できる。

```json
{
  "format_version": 1,
  "input_fingerprint": "台本JSONのinput_fingerprintをコピー",
  "entries": [
    {
      "source_utterance_ids": [2, 3, 4, 5],
      "text_ja": "比較したい短縮案",
      "reason": "変更理由と保持した内容"
    }
  ]
}
```

候補は完全なグループと一致しなければ拒否する。元JSONが変わるとハッシュ不一致で拒否する。新たな保護情報違反、既存訳からの数値変更、読み不明、推定時間の増加があれば元の訳へ戻し、拒否理由と提案文を保存する。**検査を通っても意味の同一性を証明したことにはならず、変更された訳はレビュー対象**。今回の比較案はCodexが原文を読んで作成したローカル候補であり、自動API生成や人間の承認済み訳ではない。

`CandidateProvider.propose(request)` には対象発話・同一話者の原文・前後の会話（参照専用）・ルール全文・時間条件を渡す。`CachedProvider` のキーにはプロバイダのモデル／プロンプトID、入力、文脈、辞書、解析器、時間設定を含む。成功ごとに保存し、失敗時は自動再試行しない。既定の `max_calls=0` はキャッシュ利用のみ。テストは偽プロバイダで実行する。将来APIを接続する際も、料金を説明し、承認された呼び出し上限を設定すること。

## グルーピングと時刻

- 入力は `utterances.json` v3 と既存 `utterance_translations.json`。`video_id`、clip、ID集合、話者、原文、絶対／相対時刻を照合する。部分翻訳や古い選択範囲との混用はエラー。
- 既存ファイルに形式バージョンがない場合は現行形式として扱う。新出力は `format_version: 1` と `engine_version` を持つ。
- 既知の同一話者の隣接発話に限り、間隔0〜0.5秒、合計14秒以内、直前が断片または4.5秒以下の短い発話なら結合する。元の各境界は `provenance` に保存。
- 異なる話者、UNKNOWN、重なる同一話者の発話は自動で結合しない。UNKNOWNは一つの人物を意味しないため、発話ごとに独立した未確定タイムラインに入れる。
- `start` / `end` は元動画時刻、`relative_start` / `relative_end` はclip基準。出力で元の終了時刻を書き換えない。
- `estimated_playback_end` は推定の終了位置。別話者の開始時刻で切断しない。同一話者の推定衝突は `same_speaker_conflict_ids` とレビュー理由に記録。
- 30秒境界をまたぐ発話は全体を対象外とし、`excluded` にIDと理由を残す。これは動画から削除した意味ではない。
- 単語の元status、候補話者、assignment_method、レビュー理由などは元オブジェクトごと `provenance` に保存する。

## 出力契約

`entries[]` に話者、出典ID、英語、既存日本語、最適化日本語、解析結果、推定時間差、採用／拒否された候補、レビュー理由、元JSONオブジェクトを保持する。`speaker_timelines` は話者別のentry ID一覧で、UNKNOWNごとに異なるキーを使う。入力は変更しない。

`original_timing` と現在の推定時間は区別する。`actual_tts_duration_seconds` は新しい音声を実測していない限り `null`。過去の別テキストのWAV長を転用しない。`summary` の合計秒数は発話量の比較であり、同時発話がある動画の経過時間ではない。

`review_required` は上流のレビュー情報、断片、意味検査の未確定、読み不明、時間超過、衝突、変更案の意味確認を集約する。上流のUNKNOWNや推定話者を確定に格上げしない。

## 実測と校正

```python
from language.timing import wav_duration, calibrate
seconds = wav_duration("existing.wav")  # PCM WAV。新規音声生成はない
profile = calibrate([
    {"model": "model-id", "voice": "Ryan", "mora_count": 140,
     "pause_seconds": 1.0, "actual_seconds": seconds}
], model="model-id", voice="Ryan")
```

必ずそのWAVに使用した正確なテキストのモーラ数・ポーズを与える。モデルや声が異なる標本は混ぜない。これは校正用関数で、既存manifestの自動移行やTTSへの自動接続ではない。浮動小数点WAVなど標準 `wave` 非対応形式は別の読取アダプタが必要。

## 検証結果と限界

実動画ID `XuoqKYxDHVc`、clip 1653.6〜1942.99秒の冒頭30秒。発話1〜14を11項目に整理し、UNKNOWN 5発話を保持。発話15は27.28〜40.52秒で境界をまたぐため明示的に対象外。

Codexが作成した3件の候補を使った比較では、合計推定発話量43.976秒→35.757秒（8.219秒減）。元11.26秒の発話2〜5は、既存訳の推定22.480秒→候補16.811秒。ただし元区間に収まらず、レビューを継続する。全11項目に上流の不確定性等のレビュー理由があり、9項目が推定時間超過。

既存WAV3件を読み取り、0.32秒・24.16秒・0.96秒を確認。24.16秒の既存訳に対する既定推定22.48秒は1.68秒短い。長い1標本からのRyan用診断話速は約6.48モーラ/秒だが、標本が少ないため新推定には自動適用しない。新しい最適化音声は生成しておらず、実測の短縮率は未検証。

次の段階は、承認された少数のプリセットTTSで最適化文の実測・聴取評価を行い、話者／モデルごとの複数標本で校正すること。UNKNOWNの確認、候補生成APIアダプタ、意味検査の強化、独立音声のミックスは別の作業になる。

## Phase 2との接続

`tts_generator.py --script` と `tts_feedback.py` が実装され、実際のWAVと推定値の比較、再開可能なキャッシュ、話者別の校正標本を保存できるようになりました。本書の上記MVP実測結果はPhase 1時点の記録です。現行TTSの使い方と新規WAVの結果は [TTS_FEEDBACK.md](TTS_FEEDBACK.md) を参照してください。`language.timing` に `measure_wav`、`feedback_profiles`、`estimate_with_profile` を追加し、従来の `TimingEngine` の既定挙動は維持しています。
