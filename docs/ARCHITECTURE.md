# TransClip — アーキテクチャとデータ契約

この文書はTransClipの**既存実装と計画中の構成を区別**し、Codexと開発者が変更の影響を判断するための設計資料です。

- 開発上の強制ルール: [`../AGENTS.md`](../AGENTS.md)
- 起動・使用方法: [`../README.md`](../README.md)
- 参照状況: GitHubの `seitahare/transclip` で確認したコード、および開発者がMacで実行した個別モジュールの結果に基づく。ローカル未pushの可能性があるので、作業時に必ず再確認すること。

## 1. 目標とスコープ

**長期目標**: YouTube URLを入力 → 英語動画・字幕取得 → チャプター／AIトピック／時刻から選択 → 話者・単語・発話の整列 → 意味とテンポを保つ日本語台本 → 話者別の音声生成 → 同時発話を保つミックス → 吹き替えMP4。

**現状**: `main.py fetch` で自動接続されているのは取得・英語字幕準備・選択・クリップ作成まで。翻訳・話者識別・TTSなどは独立した試作モジュールとして動作確認されている。最終MP4の自動一括生成は未完成。

**次のMVP**: 日英言語解析と吹き替え台本最適化を、既存データを破壊せず30秒の区間で検証する。音声の長さを予測し、TTS実測と比較できることを優先する。

## 2. データフローと実装ステータス

```text
YouTube URL
    │
    ▼
main.py / yt-dlp ──────► input/<video_id>/source.mp4
    │                            source.info.json
    ▼
subtitles.py + caption_cleaner.py
    ├─ manual English captions
    ├─ automatic English captions
    └─ faster-whisper fallback
    ▼
transcript.json
    │
    ├─► ai_topics.py ──► topics.json (OpenAI API)
    │
    ▼
selector.py ──► selection.json (元動画基準)
    │
    ▼
clipper.py ──► output/<video_id>/clip_<start>_<end>.mp4
    │
    ├─► translator.py ──► translation.json [独立コマンド]
    │
    └─► 話者・吹き替え用のローカル試作群 [別環境・独立コマンド]
             │
             ├─ diarizer.py ─────────► diarization.json
             ├─ word_timestamps.py ─► word_timestamps.json
             ├─ speaker_alignment.py ─► speaker_segments.json
             ├─ word_speaker_alignment.py ─► word_speakers.json
             ├─ utterance_builder.py ─► utterances.json
             ├─ utterance_translator.py ─► utterance_translations.json
             └─ tts_generator.py ───► output/<video_id>/tts/*.wav
                                         tts/manifest.json

[実装済みMVP: 詳細は LANGUAGE_ENGINE.md]
  language/* + dubbing_script.py
       └─► dubbing_script.json + timing diagnostics
                  └─► tts_generator.py --script / tts_feedback.py
                         └─► 話者別WAV / feedback.json / 校正用標本
                                └─► audio_mixer.py ──► 話者別トラック / mix_report.json / 診断用MP4
[計画中]
  approved_voice_profile/* ──────► speaker-aware TTS (権利確認後)
```

※矢印は概念上の依存関係です。全ステージが単一CLIで連続実行されるわけではありません。入力動画の拡張子は現実には `mp4` 以外の場合もあり、`clipper.py` の現行実装は `source.mp4` を前提とします。

## 3. データ契約（現行）

### 3.1. 動画と選択範囲

| ファイル | 必須情報 | 時刻基準 |
| --- | --- | --- |
| `source.info.json` | 動画メタデータ（`duration`, `chapters` 等） | 元動画 |
| `selection.json` | `start`, `end` (秒) | 元動画 |
| `transcript.json` | `source`, `language: "en"`, `segments` | 元動画 |

`selection.json` の例:

```json
{"start": 1653.6, "end": 1942.99}
```

`transcript.json` の各要素はおおむね次の形です。

```json
{"start": 1653.6, "end": 1654.87, "text": ">> Yeah."}
```

この例は説明用です。YouTubeの字幕区間は厳密な発話区間や単語境界とは限らず、`>>` が混ざった字幕もあります。

### 3.2. 字幕翻訳

`translator.py` の `translation.json` は選択範囲に入る字幕をもとに、`clip: {start,end}`、`segments[]`（`id`, `start`, `end`, `relative_start`, `relative_end`, `text_en`, `text_ja`）を持ちます。字幕境界が日本語の文の境界と一致する保証はありません。

### 3.3. 話者・単語・発話（ローカル追加モジュール）

| ファイル | 主な内容 | 注意 |
| --- | --- | --- |
| `diarization.json` | `speakers`, `turns`, `exclusive_turns`, `clip` | `turns` の重なりを保持する。`exclusive_turns` は照合補助。 |
| `word_timestamps.json` | `words[]` (`word`, 絶対/相対時刻) | Whisperによる再認識であり、字幕原文と異なる可能性。 |
| `speaker_segments.json` | 字幕単位の話者候補・レビュー状態 | 時間による暫定対応。 |
| `word_speakers.json` | 単語単位 `speaker`, `status`, `speaker_candidates` | `single` もモデル推定であり確定ではない。 |
| `utterances.json` | `utterances[]`, `speaker`, `text_en`, `words`, `review_required`, `end_reason` | 文分割・話者補完を行う暫定発話データ。 |
| `utterance_translations.json` | `utterances[]`, `text_en`, `text_ja`, `speaker`, 絶対/相対時刻, レビュー情報 | 生成結果はTTS向けに再調整が必要。 |
| `tts/manifest.json` | 音声ジョブ、参照発話ID、声、相対時刻、`audio_duration`, WAVファイル名 | プリセット声による試作。日本語が元英語より長くなることがある。 |

時刻の原則:

```text
clip_start = selection.start
absolute_time = clip_start + relative_time
clip_duration = selection.end - selection.start
```

JSON同士を統合するときは `clip.start` と `clip.end` を照合し、30秒プレビューと全尺の解析結果を混用しない。`selection.json` が更新された場合、派生データの有効性を再判定する。

### 3.4. 既知の観察事実（参考、一般的な保証ではない）

開発者の試作では、選択範囲 1653.6〜1942.99 秒（289.39秒）について次が得られています。

- 英語字幕の翻訳: 42セグメント。
- 話者識別: 2話者、51区間。
- Whisper単語タイムスタンプ: 718語。
- 単語・話者の暫定照合: `single` 635、`overlap` 32、`ambiguous` 1、`uncertain` 50。
- 発話再構成v3: 82発話、45語が話者未確定、52発話がレビュー対象。
- 発話単位翻訳: 82発話分の結果をローカル保存済み。
- Qwen3-TTSの試作では、元の約11秒の発話に相当する日本語WAVが約24秒になった。自然な同時発話と**同一話者が自分の声と重なる不自然な状態**は別々に扱う必要がある。

これらは特定クリップの実測例であり、アルゴリズムの正解率を示すものではありません。ローカルのデータは通常Gitには含まれません。

## 4. 絶対に守る設計上の不変条件

1. **発話の重なりを保存**: `diarization.turns` と音声タイムラインは重なりを許す。後から始まった別話者の音声を理由に、先行音声を切断しない。
2. **同一話者の二重読み上げを防ぐ**: 発話間の時間衝突を見つけたら、自然な再台本化・配置見直しを行い、無条件に多重再生しない。
3. **レビュー情報を残す**: `UNKNOWN`, `uncertain`, `overlap`, `REVIEW` を入力の出自を示さず上書きしない。補完した場合は推定方法と根拠を保存。
4. **トレーサビリティ**: 吹き替え台本と音声ジョブから元動画の時刻・原文・入力発話IDを追跡できること。
5. **意味と構造を守る**: 翻訳では否定、数値、名前、推測や発言者の意図を保つ。極端な短縮が必要なら「収まらない」と報告する。
6. **データを破壊しない**: 原本、中間JSON、生成WAVを書き換える必要があれば別出力・バージョン管理・移行方法を用意する。
7. **費用をコントロール**: モデル/API呼び出しには明確な入口を設け、`--dry-run` / 30秒 / 少件数の試験を優先する。
8. **承諾のある音声のみ声質再現**: 声質のクロスリンガル再現は検討対象だが、実在人物の無断複製や権利不明の公開を前提としない。

## 5. `language/` の責務（MVP実装済み）

```text
language/
  __init__.py
  english.py        # 英語のトークン・品詞・句構造等
  japanese.py       # 形態素・読み・正規化
  pronunciation.py  # 読み・モーラ・辞書の扱い
  timing.py         # 推定発話長とTTS実測値の比較
  rules.json        # バージョン付きの翻訳・時間制約

dubbing_script.py   # 会話文脈・話者・時間に応じた台本作成
```

- 言語解析は既存ライブラリ（例: spaCy、SudachiPy、pyopenjtalk）を検討し、まず小さな抽象APIで包む。LLMの内部トークナイザを自作する必要はない。
- ルールは **意味保持を優先するソフト制約**。発話時間の上限を超えたからといって音声を切断したり、情報を勝手に捨てたりしない。
- モーラ数や文字数は**時間推定の特徴量**であり真の所要時間ではない。TTSで生成したWAVの実測長、ポーズ、話速、話者／モデル別差異を反映する。
- 構文解析器は説明可能な補助情報を提供するもので、英日で単語・文法を一対一対応させる必要はない。

## 6. 台本データの設計とMVP

中間データと混同しないよう、別の `dubbing_script.json` を設計します。以下は当初の最小設計例です。実装済みのv1は出典・比較用旧訳・解析結果・タイムライン等を追加しています。現行の契約と制約は [LANGUAGE_ENGINE.md](LANGUAGE_ENGINE.md) を参照してください。

```json
{
  "format_version": 1,
  "clip": {"start": 1653.6, "end": 1942.99},
  "entries": [
    {
      "id": "dub-001",
      "source_utterance_ids": [2, 3],
      "speaker": "SPEAKER_00",
      "speaker_confidence": "estimated",
      "review_required": true,
      "relative_start": 1.9,
      "relative_end": 9.36,
      "text_en": "Original speech assembled from source IDs",
      "text_ja": "意味を保った自然な吹き替え台本",
      "estimated_duration_seconds": null,
      "actual_tts_duration_seconds": null,
      "timing_status": "unmeasured"
    }
  ]
}
```

この例の文章や時間はスキーマ説明用です。推定長とWAV実測長を分け、`review_required` と原文の出典を保持することが重要です。既存の話者・翻訳JSONとの照合を実装済みです。TTSとの自動連携は今後の課題です。

## 7. 処理と環境の境界

| 種類 | 実行環境・依存 | 備考 |
| --- | --- | --- |
| 字幕取得・編集・OpenAI翻訳 | `./.venv`、`requirements.txt` | API処理は都度課金。 |
| 話者識別 | `./.venv-diarize`、`pyannote.audio`、`HF_TOKEN` | モデルの利用条件承認が必要。 |
| 単語推定 | `./.venv`、`faster-whisper` | 一部の短い語を見落とす場合がある。 |
| Qwen3-TTS-MLX | 専用Python 3.12環境（Mac） | モデルキャッシュ、WAVの実測長、利用可能な音声APIを確認。 |
| 映像・音声加工 | `ffmpeg` / `ffprobe` | 絶対秒数／相対秒数に注意。 |

独立プロセスで中間JSON/WAVを受け渡す設計を優先します。重い機械学習ライブラリをすべてひとつのvenvに統合しないこと。

## 8. 検証方針・受け入れ条件

**単体テスト**

- 選択範囲の端、空テキスト、字幕欠落、無効なJSON、キャッシュ無効化。
- 複数話者の重複区間、話者未確定の扱い、発話IDと原文の追跡。
- 日本語の数字、略語、句読点、モーラ（小書き文字、促音、長音など）に対する正規化と計測。
- 時間見積もりが実測値と一致しない場合の警告、および意味を損なう強制短縮の防止。

**30秒MVPのチェック**

1. 選択クリップ基準で正しいタイムスタンプが返る。
2. 元の発話が重複・消失しない。未確定話者や重複箇所は適切にレビューへ残る。
3. 英語の意味、固有名詞、数値、否定の保持を人間が比較できる。
4. 台本と元の `utterance_translations.json` を比較し、変更理由が追える。
5. 少数の日本語WAVを生成し、音声長が実測できる。
6. 異なる話者の自然な重なりを保持できる一方、同一話者の不自然な自己重複は検出できる。
7. 元動画・ローカルJSON・既存WAVを破壊しない。

通常のテスト入口:

```bash
python -m unittest discover -s tests -v
```

外部APIや大容量モデルを必須にしないユニットテストを先に追加し、実モデルの評価は明示的な別コマンドにすること。

## 9. Codex向け変更手順

1. `git status --short` で未コミットの変更を確認する。
2. ローカルの実ファイルと [`../AGENTS.md`](../AGENTS.md) を読む。GitHubとローカルの差分を確認する。
3. 変更対象の入力・出力JSONを小さなフィクスチャで再現する。
4. 30秒の最小単位で新コードを追加し、依存関係とキャッシュキーを明示する。
5. テストと手動の比較用コマンドを用意する。API・TTS実行は利用者が選べるようにする。
6. 変更ファイル、検証結果、残る未確定点、必要な移行手順を日本語で報告する。

**作らないもの（今回のMVP）**: トークナイザのフルスクラッチ実装、元動画の大規模一括再ダウンロード、全尺の自動吹き替え、無承諾での声質複製、既存のデータスキーマの破壊的な書き換え。

## 10. Phase 2: TTS Feedback Engine（実装済み）

`tts_generator.py --script` を独立したQwen環境で実行し、`tts_feedback.py` が入力検証・WAV/receiptキャッシュ・再開・実測比較・衝突・校正を管理します。元の台本、旧manifest、旧WAVを保持した別出力です。UNKNOWNは生成せず記録し、別話者の重なりを直列化しません。校正は話者・モデル・音声設定ごとに分離し、自動適用しません。詳細は [TTS_FEEDBACK.md](TTS_FEEDBACK.md) を参照してください。

## 11. Phase 3: Audio Mixer（実装済み）

独立CLI `audio_mixer.py` がPhase 2のWAVを再測定・来歴照合し、`relative_start` を48kHzサンプルへ変換して話者別タイムラインに配置します。元動画は `selection.start` から切り出します。FFmpegによるリサンプル・遅延・話者別ミックス、共通の固定ゲイン、H.264/AACへの結合を行います。WAVの切断・倍速化・話者の直列化は行いません。

通常モードは同一話者衝突・未吹き替え・未解決レビューで停止。`--diagnostic` でのみ未解決状態を明示した部分プレビューを生成します。UNKNOWNと上流除外をレポートに残します。日本語のみ／英語を小音量で残す比較モードがあり、音声分離とは区別します。

プレビュー終端を越える音声は停止し、明示的な延長でも選択区間を越えません。入力を変更せず、検証済みの成果物一式を別ディレクトリへアトミックに保存します。全デコード・ストリーム・時間・ピーク・再実行のハッシュ確認を実装しました。詳細は [AUDIO_MIXER.md](AUDIO_MIXER.md)。メインCLIへの一括接続と完成品質への調整は今後の課題です。

## 12. Phase 4: 話者解析・声質プロフィール（実装済み）

`voice_profiler.py` / `speaker_refinement.py` が既存pyannoteのsegmentation・embedding・PLDAをローカルキャッシュから読み、全選択範囲を話者数自動で解析します。新旧の非重複区間の埋め込みを照合してIDを維持し、窓ごとの類似度・境界とASRの矛盾を保存します。原本JSONや同時発話を削除せず、未知発話の自動確定を避けます。

`speaker_profiles.json` は波形測定、DSP推定、ASR推定、モデル埋め込みを分離し、品質評価付き参考候補を持ちます。`speaker_review.py` のHTMLで音声を聞き比べ、人間の判断と文字起こし修正を別JSONへ保存・検証取込できます。

`language/voice.py` でPreset / Voice Design / Approved Cloneの契約を区別。現行QwenはPresetのみで、解析を添付できるものの声質再現は行いません。`audio_mixer.py --speaker-analysis` は話者変更に対して既存WAVの付け替えを禁止し、元の区間にない別話者の重なりも分類します。詳細は [SPEAKER_PROFILES.md](SPEAKER_PROFILES.md)。実モデルの精度向上は人手正解ラベルなしには主張しません。


## Phase 5: Speaker Attribution Refinement

`speaker_attribution.py` をPhase 4と台本生成の間に任意で追加する。既存の参考窓・埋め込み・音響特徴・レビュー保存を再利用する。対象と文脈音声に重なる参考を除き、複数窓の埋め込み、音質、F0、segmentationの発話人数を比較する。長さ別の参考比較はモデルラベルによる自己整合性確認で、人間の正解データによる校正ではない。

出力 `speaker_attribution.json` は発話ID・元時刻・旧ラベル・新ラベル・自動判定・人間確認・根拠・入力ハッシュを保持する。互換形式の発話・翻訳・単語JSONを別出力へ保存する。異なる話者を結合せず、mixed/UNKNOWNはnullの話者として残す。元の話者ごとの重複区間も独立に保存し、推定上の重なりと音響上の混在仮説を区別する。音声分離や混在部分の話者別文字起こしは未実装。

Speaker ReviewはPhase 4/5両方の出力に対応する。人間確認JSONはマニフェスト・音声ハッシュ・レビューキーを照合して取り込み、自動判定とは別に保存する。`human_evaluation.jsonl` に将来の評価用ラベルを保存し、ラベルのない項目の正解率を計算しない。`speaker_timeline.json` は既存Mixerの話者照合に、`tts_regeneration.json` は古いWAVの再生成要否確認に利用する。詳細は [SPEAKER_ATTRIBUTION.md](SPEAKER_ATTRIBUTION.md)。
