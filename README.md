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

TTSの試作では既定のプリセット声を使います。**元の話者の声質を再現するVoice Cloneは、今後の設計・検証対象であり、現行の完成済み機能ではありません**。

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

## 次の開発課題

- 言語エンジンの数詞・助数詞・意味保持検査を拡充する。
- 台本候補の生成プロバイダを接続し、短縮案の聴取・意味確認を進める。
- Qwen3-TTSのWAV実測長を利用し、発話時間を反復的に改善する。
- `UNKNOWN`・同時発話を確認し、正しい話者へ割り当てるためのレビュー手段を作る。
- 適切な権利・許可がある場合に限り、参照音声による声質再現を検証する。
- 日本語WAVを配置・ミックスし、最終MP4を生成する処理をCLIに接続する。

実装状態や具体的なデータ契約は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) を参照してください。

## 権利・安全・秘密情報

- 動画・字幕・音声は、取得・加工・公開が許可されている素材で検証してください。
- 実在人物の声を本人に似せる場合は本人の承諾や必要な権利を確認し、公開時はAI吹き替えであることを適切に明示してください。
- APIキーとHugging Faceトークンは環境変数で管理し、ソースコード、ログ、Gitに含めないでください。
- 現行の `.gitignore` は `input/`、`output/`、`.venv/` 等を除外します。追加の仮想環境 `.venv-diarize/` も **Git管理前に除外対象へ追加**してください。
- OpenAI APIを使う機能は有料で、モデルダウンロードもネットワーク・ディスク容量を消費します。
