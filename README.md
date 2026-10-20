# TransClip

英語の動画から必要な区間を選び、会話の意味・話者・タイミングを可能な限り保ちながら、日本語吹き替えを生成するための **Python CLI 開発プロジェクト**です。

> **開発中 / experimental** — 現在の `main.py` は取得・字幕準備・クリップ選択・切り抜きを実行します。翻訳や話者識別、TTS、最終MP4合成は、現時点では主に個別スクリプトによる試作です。URLを渡すだけで完成版の日本語吹き替えが自動生成される状態ではありません。

開発者向けのルールは [`AGENTS.md`](AGENTS.md)、詳しいデータフローと設計方針は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) を参照してください。

## 現在できること

**GitHub上の既存コードで確認できる処理**

- `yt-dlp` で YouTube 動画とメタデータを取得し、既存ファイルがあれば再利用する。
- 手動の英語字幕を優先し、自動英語字幕を試行する。使用可能な字幕がなければローカルの `faster-whisper` に切り替える。
- ローリング字幕の重複を減らし、時刻付き `transcript.json` に正規化する。
- チャプター、OpenAI API によるAIトピック、手動時刻指定から切り抜く区間を選択する。
- `ffmpeg` で選択範囲をMP4に切り抜く。
- `translator.py` で選択範囲の字幕を日本語に翻訳する（独立コマンド）。

**開発者のMacで動作確認済みの追加試作**

- `diarizer.py`: `pyannote.audio` による話者識別。
- `word_timestamps.py`: `faster-whisper` による英単語単位の時刻推定。
- `speaker_alignment.py` / `word_speaker_alignment.py`: 話者と字幕・単語の対応付け。
- `utterance_builder.py`: 発話単位への再構成（暫定）。
- `utterance_translator.py`: 発話単位の日本語翻訳。
- `tts_generator.py`: Qwen3-TTS-MLX による日本語音声の試作。

これらの追加ファイルはローカルには存在していても **GitHubにまだpushされていない場合があります**。利用前に実際のファイル構成を確認してください。

## 必要な環境

- macOS / Apple Silicon を主な開発・検証環境としています。
- Python 3（ライブラリの対応バージョンを確認すること）。
- `yt-dlp`、`ffmpeg`、`ffprobe` が実行可能であること。
- AIトピック生成・翻訳を行う場合のみ OpenAI API キーと利用枠。
- 話者識別を実行する場合は、別のPython環境と Hugging Face の認証・対象モデルへのアクセス承認が必要です。
- Qwen3-TTS-MLX は別のPython環境に分離します。モデルによっては追加のストレージを消費します。

macOSでの外部ツール導入例:

```bash
brew install yt-dlp ffmpeg
```

メイン環境のセットアップ例:

```bash
cd ~/Projects/transclip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

依存関係の完全な再現性やすべてのPythonバージョンでの動作は保証されていません。`requirements.txt` はメイン環境向けで、話者識別とTTSの重い依存関係は含みません。

## 基本的な使い方

動画のダウンロード、英語字幕の準備、区間選択、MP4切り抜き:

```bash
python main.py fetch 'https://www.youtube.com/watch?v=VIDEO_ID'
```

選択画面では次の3つが使用できます。

1. 動画のチャプターから選択
2. AIが生成したトピックから選択（OpenAI APIを使用・有料）
3. 開始・終了時刻を手入力

選択結果は `input/VIDEO_ID/selection.json`、切り抜いた動画は `output/VIDEO_ID/` に保存されます。

選択範囲の日本語字幕を翻訳する独立コマンド:

```bash
python translator.py VIDEO_ID --preview   # API呼び出しなし
python translator.py VIDEO_ID             # OpenAI APIを使用
```

### 話者識別から音声生成までの試作（該当ファイルがある場合）

下記はローカルで実装・検証した個別スクリプトの呼び出し例です。**`main.py` から自動実行されるわけではありません**。

```bash
# 話者識別: 専用の .venv-diarize 環境、HF_TOKENが必要
source .venv-diarize/bin/activate
python diarizer.py VIDEO_ID --max-seconds 30
python diarizer.py VIDEO_ID

# メインの .venv に戻す
source .venv/bin/activate
python word_timestamps.py VIDEO_ID
python speaker_alignment.py VIDEO_ID
python word_speaker_alignment.py VIDEO_ID
python utterance_builder.py VIDEO_ID
python utterance_translator.py VIDEO_ID --preview --limit 8
python utterance_translator.py VIDEO_ID --limit 8   # 有料API
python utterance_translator.py VIDEO_ID             # 有料API
python tts_generator.py VIDEO_ID --seconds 30 --dry-run
```

TTSの本実行はQwen3-TTS-MLX専用のPython実行環境から行います（パスは各マシンで異なります）。

```bash
/path/to/Qwen3-TTS-MLX/.venv/bin/python tts_generator.py VIDEO_ID --seconds 30 --max-jobs 3
```

TTSの試作では既定のプリセット声を使います。**許可済み参考音声によるVoice Cloneの登録・生成アダプタはPhase 7で追加しました。実モデルによる声質・発音の検証は、権限確認済み音声の準備後に行います**。

## ディレクトリとデータ

```text
transclip/
├── AGENTS.md                       # Codex向け開発ルール
├── README.md                       # このファイル
├── docs/ARCHITECTURE.md            # 設計・データ形式
├── main.py                         # 取得～切り抜きの入口
├── selector.py / subtitles.py      # 区間選択・字幕処理
├── ai_topics.py / clipper.py       # トピック生成・切り抜き
├── translator.py                   # 選択字幕の翻訳
├── ...                             # ローカルの追加試作モジュール
├── tests/
├── input/VIDEO_ID/                 # 元動画・中間JSON（通常Git管理外）
└── output/VIDEO_ID/                # MP4・WAV等（通常Git管理外）
```

主な中間ファイル:

| ファイル | 内容 |
| --- | --- |
| `source.info.json` | 動画メタデータ |
| `transcript.json` | 英語字幕またはWhisper文字起こし |
| `selection.json` | 元動画基準の選択開始・終了秒数 |
| `topics.json` | AI生成トピックのキャッシュ |
| `translation.json` | 選択字幕の日本語訳 |
| `diarization.json` | 話者区間と重複発話情報 |
| `word_timestamps.json` | Whisper推定の単語と時刻 |
| `word_speakers.json` | 単語と話者の暫定対応 |
| `utterances.json` | 話者別の暫定発話 |
| `utterance_translations.json` | 発話単位の日本語訳 |
| `output/VIDEO_ID/tts/manifest.json` | 生成WAVとその実測長（試作） |

**時刻の意味:** `start`/`end` は通常「元動画基準の秒数」、`relative_start`/`relative_end` は「選択クリップ基準の秒数」です。区間選択を変えたら、対応する中間ファイルのキャッシュと時刻の整合性を再検証してください。

## テスト

```bash
python -m unittest discover -s tests -v
```

既存の字幕・切り抜き4テストに加え、言語エンジンと台本生成のテストを追加しています。解析器を使うテストは `requirements-language.txt` の環境で実行してください。依存未導入なら該当テストは明示的にスキップされます。テスト成功と音声・字幕の品質保証は別物です。

## 日英言語エンジン・台本最適化MVP

`dubbing_script.py` を追加しました。既存の発話・翻訳JSONを読み、同一話者のグルーピング、spaCy / SudachiPy解析、読み・モーラによる時間推定、翻訳ルールの検査、比較表を出力します。UNKNOWNと同時発話を保持し、長すぎる音声は切断せずレビューに残します。

新依存は独立した `.venv-language` と `requirements-language.txt` で扱います。通常環境やQwen環境の変更は不要です。セットアップ、30秒MVP、フィクスチャ、ローカル候補入力、実測との区別は [docs/LANGUAGE_ENGINE.md](docs/LANGUAGE_ENGINE.md) を参照してください。

```bash
.venv-language/bin/python dubbing_script.py VIDEO_ID --seconds 30
.venv-language/bin/python -m unittest discover -s tests -v
```

既定処理はAPIを呼ばず、限定的なローカル短縮規則を適用します。自由な言い換えは別途候補入力で比較します。候補の意味と実音声の長さはレビューが必要です。異なる既存出力は上書きしません。

## TTS Feedback Engine（Phase 2）

`tts_generator.py --script` で最適化台本からプリセットWAVを生成し、サンプル数による実測、推定誤差、時間超過、同一話者の衝突、声別の校正データを保存できます。既存Qwen環境とモデルキャッシュを使い、UNKNOWNは未確定のまま記録します。既存台本・旧WAVは変更しません。

```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python tts_generator.py VIDEO_ID \
  --script input/VIDEO_ID/dubbing_script.json --seconds 30
```

[docs/TTS_FEEDBACK.md](docs/TTS_FEEDBACK.md) にオフライン実行、再開、旧WAV比較、候補再評価、校正、再生方法を記載しています。

## Audio Mixer（Phase 3）

`audio_mixer.py` が Phase 2 の既存WAVを相対開始時刻へ配置し、話者別トラック、日本語音声、H.264/AAC MP4、配置レポートを作ります。日本語のみ／小音量の英語を残す比較モードを選べます。同一話者の衝突や未生成・未レビュー項目があれば通常モードは停止し、`--diagnostic` 指定時だけ問題を残した部分プレビューを生成します。

```bash
.venv/bin/python audio_mixer.py VIDEO_ID --feedback /path/to/feedback.json --dry-run
.venv/bin/python audio_mixer.py VIDEO_ID --feedback /path/to/feedback.json --diagnostic
```

詳細・実データの実行例・再生方法は [docs/AUDIO_MIXER.md](docs/AUDIO_MIXER.md) を参照してください。新しいTTSやAPIは実行しません。

## 話者解析と声質プロフィール（Phase 4）

`voice_profiler.py` が既存pyannoteモデルをオフラインで再利用し、全選択範囲の話者埋め込み・F0等の特徴・参考WAV・話者確認HTMLを生成します。UNKNOWNの発話は人間確認まで保持し、実名やClone許可を推測しません。Preset / Voice Design / 許可済みCloneの契約を分離し、Mixerでは話者変更後の古いWAVの流用を止めます。

```bash
.venv-diarize/bin/python voice_profiler.py VIDEO_ID
```

[docs/SPEAKER_PROFILES.md](docs/SPEAKER_PROFILES.md) に入力、特徴の測定／推定の区別、確認結果の保存、TTS/Mixerへの接続と制約を記載しています。実際の精度向上は人手の正解確認後に評価します。

## 音響による話者再判定（Phase 5）

`speaker_attribution.py` がPhase 4の参考音声・埋め込みを再利用し、対象音声、前後の音響窓、短時間の発話人数、長さ別の参考比較から話者を再評価します。元のUNKNOWNを条件付きで再判定し、混在・境界・ASR時刻ずれの可能性と根拠を保存します。Speaker Reviewで自動判定と人の判断を分けて確認できます。

```bash
.venv-diarize/bin/python speaker_attribution.py VIDEO_ID \
  --analysis-dir output/VIDEO_ID/speaker_analysis/CONTENT_KEY
```

元JSON・WAVは変更しません。人の確認ラベルを優先し、ラベルがない区間の正解率は算出しません。実装・未校正スコア・互換出力・下流への接続は [docs/SPEAKER_ATTRIBUTION.md](docs/SPEAKER_ATTRIBUTION.md) を参照してください。

## 次の開発課題

- 言語エンジンの数詞・助数詞・意味保持検査を拡充する。
- 台本候補の生成プロバイダを接続し、短縮案の聴取・意味確認を進める。
- TTSフィードバックの校正標本を増やし、短縮候補を実測・聴取で反復評価する。
- `UNKNOWN`・同時発話を確認し、正しい話者へ割り当てるためのレビュー手段を作る。
- 適切な権利・許可がある場合に限り、参照音声による声質再現を検証する。
- 診断用MP4の聴取結果を使って衝突と未吹き替えを解消し、Audio MixerをメインCLIへ接続する。

実装状態や具体的なデータ契約は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) を参照してください。

## 権利・安全・秘密情報

- 動画・字幕・音声は、取得・加工・公開が許可されている素材で検証してください。
- 実在人物の声を本人に似せる場合は本人の承諾や必要な権利を確認し、公開時はAI吹き替えであることを適切に明示してください。
- APIキーとHugging Faceトークンは環境変数で管理し、ソースコード、ログ、Gitに含めないでください。
- 現行の `.gitignore` は `input/`、`output/`、`.venv/` 等を除外します。追加の仮想環境 `.venv-diarize/` も **Git管理前に除外対象へ追加**してください。
- OpenAI APIを使う機能は有料で、モデルダウンロードもネットワーク・ディスク容量を消費します。

## Phase 6: Speech Separation

選択済みクリップの重複候補だけを分離し、話者候補・単語時刻・元音声との比較を別ファイルへ保存します。既存の話者プロフィールとローカルASRを再利用します。操作方法・制約は [docs/SPEECH_SEPARATION.md](docs/SPEECH_SEPARATION.md) を参照してください。

## 許可済み参考音声によるVoice Clone（Phase 7）

`voice_cloning.py` で参考WAV・正確な文字起こし・利用許可・品質確認を登録し、Baseモデルと既存Presetを切り替えられます。Feedback/Mixerへの接続、キャッシュ、途中再開、比較表・試聴HTMLを用意しました。実音声生成は権限確認済み素材の準備待ちで、今回はモックテストとローカルモデル事前確認までです。[登録形式・実行手順](docs/VOICE_CLONING.md) を参照してください。
