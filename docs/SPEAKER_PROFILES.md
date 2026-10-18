# Phase 4: 話者埋め込み・声質プロフィール・Speaker Review

## 実装と実行環境

独立CLI `voice_profiler.py` が `speaker_refinement.py` を呼び、選択クリップ全体を再解析する。既存 `.venv-diarize` のpyannote.audio、Torch、NumPyとFFmpegだけを使用する。新しいモデル・ライブラリの導入は不要。

既存の `pyannote/speaker-diarization-community-1` にはsegmentation、speaker embedding、PLDAが含まれている。ローカルsnapshotと各重みのSHA-256、バージョンを保存する。`HF_HUB_OFFLINE=1`・`local_files_only=True` で読み、pyannoteのメトリクス送信も無効にする。キャッシュがなければ失敗し、ダウンロードしない。従来の `diarizer.py` のCLIと出力は変更しない。

```bash
cd ~/Projects/transclip
.venv-diarize/bin/python voice_profiler.py XuoqKYxDHVc \
  --output-dir output/XuoqKYxDHVc/speaker_analysis
```

`--input-dir` も指定可能。必須入力は `selection.json`, `source.mp4`, `source.info.json`, `diarization.json`, `word_timestamps.json`, `word_speakers.json`, `utterances.json`, `utterance_translations.json`。動画ID、選択時刻、単語、発話ID、翻訳の来歴を照合する。単語認識・翻訳を再実行するCLIではない。

元データを変更せず `output-dir/CONTENT_KEY/` に成果物をまとめる。同じ入力・モデル・実装なら成果物ハッシュを検証して再利用する。設定や入力が変われば別キーになり、既存ファイルを上書きしない。排他ロックと一時ディレクトリへの保存後のアトミック公開を行う。失敗した一時出力は成功扱いしない。モデル推論途中の細粒度再開は未実装。

## 話者識別の強化

話者数の固定や2人への制限は行わない。pyannoteを全選択範囲に対して一度に動かし、regular diarizationの同時発話を保持する。exclusive diarizationで重なりを消さない。短いターンを後処理で削除しない。

1. 以前の話者区間から、他話者が重ならない複数の窓を取り、話者埋め込みの基準を作る。
2. 新しい推定区間からも複数の埋め込みを計算する。
3. 新旧の基準間で相互に最良となる十分なコサイン類似度・マージンがある場合だけ、以前の話者IDを引き継ぐ。
4. 対応が曖昧、新規、競合するクラスタには `SPEAKER_NEW_*` を与え、全クリップで一貫させる。無条件の統合や実名の特定はしない。
5. 窓ごとの類似度・他話者との差・同一話者の独立した比較窓数を保存する。類似度は確率や精度ではない。

閾値は暫定的なレビュー用の規則で、検証データから学習した校正値ではない。グローバルID対応は類似度0.65以上、他候補との差0.10以上と相互最良を要求する。新旧で同じ音声区間・モデルなら類似度1.0になる場合があるが、精度向上の証拠ではない。

pyannoteの最終フレームが選択終端をわずかに越える場合がある。0.20秒以内のモデル区間超過だけを派生メタデータ上で選択終端へ制限し、元時刻・変更理由を記録する。原本JSONと音声は変更しない。大きな超過はエラー。

## 単語対応・境界・矛盾

既存 `word_speaker_alignment.assign_speaker` を再利用し、時間カバレッジに埋め込みの根拠を追加する。独立した音響根拠は各単語そのものとは限らず、重なりのない周辺窓であることを記録する。

- 時間上の候補と音響候補が矛盾した場合はUNKNOWNにし、確認対象へ残す。
- 旧結果がoverlap/ambiguousだった語、非常に短い語、話者境界近傍は自動確定しない。
- 旧UNKNOWN語の補完には、高い時間カバレッジ、周辺埋め込みの一致・十分なマージン、境界からの距離を要求し、補完後もレビュー対象とする。
- 発話単位のUNKNOWNは人間の確認まで保持する。候補の単語票、時間カバレッジ、利用可能な場合の発話埋め込みを別フィールドに保存する。
- 0.8秒未満または同時発話を含む発話から、単独話者の埋め込みを無理に作らない。
- 各話者区間の境界について、最寄りのASR境界との距離と前後80msのRMSを保存する。実際の話者交代の正解を保証するものではない。
- ASRに語があるのに対応する話者区間がない、または同時発話で一意に対応できない場合を確認対象として記録する。

以前の `word_speakers.json` のUNKNOWNと、`utterance_builder.py` の文脈補完後のUNKNOWNは母集団が異なる。比較レポートでは両方を区別する。

## 声質特徴と参考音声

`speaker_profiles.json` v1は、動画ID、選択範囲、解析キー、各speaker_id、解析区間、品質、使用モデル、参考音声を持つ。

各解析窓の値を次の区分に分ける。

|区分|内容|
|---|---|
|`measured`|サンプルレート・サンプル数・長さ、ピーク、RMS、クリッピング率、スペクトル重心・平坦度、ゼロ交差率|
|`dsp_estimates`|正規化自己相関によるF0中央値・10/90百分位・標準偏差・周期性、低エネルギーポーズ、リズム指標、ダイナミックレンジの代理値|
|`asr_estimates`|既存Whisperの語数、語/秒、単語間隔・リズム。人手確認済み文字起こしではない|
|`model_estimates`|pyannoteの話者埋め込み、次元、標本数、モデル情報|

F0は40ms窓・10ms刻み、60〜450Hzの信号処理推定。オクターブ誤りや無声区間を考慮し、確かなF0を得られなければnullとする。エネルギーの低下は言語的なポーズそのものではなく、ダイナミックレンジは測定済みSNRではない。性別、年齢、実名などは推定しない。

参考候補は2.5〜8秒、長い区間では4秒間隔で複数探索する。新旧どちらかで検出された他話者の重なりを除外し、境界に0.1秒の余裕を置く。窓間を無断でつなげない。音量、クリッピング、周期的な音声の割合、雑音に似たスペクトルの代理値、話者類似度・マージンで評価する。

自身を含む窓で信頼度を水増ししないよう、同じ話者の**時間的に重ならない比較窓**との類似度を品質判定に使う。十分な比較窓がなければ選択を見送る。話者ごとに上位3本まで、互いに重ならない候補をWAV出力する。明瞭さ・背景雑音はヒューリスティック評価であり、聴取確認は別途必要。

参考音声のASRテキスト、含まれる単語ID、境界に部分語がある可能性も保存する。人間の修正は `corrected_text` と確認者として別出力へ保存し、元のASRテキストを残す。参考音声の選択や文字起こし修正は、Voice Cloneの利用許可を意味しない。

## Speaker Review

解析先の `speaker_review.html` をブラウザで開く。参考音声と各発話を再生し、候補・時間カバレッジ・音響類似度・判断不能の理由・単語単位の根拠を確認できる。重点5件を先頭に表示する。

- Dario
- famously, what
- You're building humanoid robots.
- Yes, he
- like,

対象そのものと前後0.65秒を含む音声を分ける。文脈音声には別話者が入る可能性があることを明示する。HTMLはネットワークや外部ライブラリを使わない。

確認者を入力し、話者またはUNKNOWNを選び、必要なら判断メモや参考音声の文字起こし修正を記入する。「確認結果を別JSONへ保存」でファイルをダウンロードする。未確認項目は変更しない。確認者の代わりにAIが人間の承認を記録することはしない。

```bash
.venv/bin/python speaker_review.py \
  --analysis-dir output/XuoqKYxDHVc/speaker_analysis/CONTENT_KEY \
  --decisions ~/Downloads/speaker_review_decisions.json \
  --input-dir input/XuoqKYxDHVc \
  --output-dir output/XuoqKYxDHVc/speaker_review/review_001
```

レビュー対象・音声ハッシュ・解析キー・原本の内容を照合し、古い判断や別音声への流用を拒否する。出力は `review_decisions.json`, `utterances.json`, `utterance_translations.json`, `selection.json`, `speaker_profiles.json`, `speaker_timeline.json`。翻訳テキストは変更せず、確認された話者だけ同期する。話者確認後も意味・時刻などの未解決レビュー理由は残す。既存出力と異なる判断は別出力先を要求する。

## TTSとの接続

`language.voice.VoiceProfile` に解析来歴と次の方式を追加した。

- `preset`: Ryan/Aidenなど明示したプリセット。解析情報は渡すが、既存Qwenプリセットの声質を変えたことにはしない。
- `voice_design`: 明示的な合成声の説明を渡す契約。参照音声を受け付けず、対応バックエンドがなければ停止する。今回生成アダプタは実装しない。
- `approved_clone`: 参考音声と利用許可の記録を要求。旧 `approved_reference` は互換名として残す。対応生成アダプタは今回未実装。

`synthesize_profile` は方式を勝手にフォールバックさせない。Cloneでは、明示的な `authorized: true`, `scope: voice_cloning`, speaker_id、対象音声のSHA-256、権限の根拠と確認者を持つ記録を検証する。実際の許可の有効性をモデルが推測して埋めることはしない。

既存QwenへPreset解析を添付する入口:

```bash
# dry-runではQwenモデルをロードしない
.venv-language/bin/python tts_generator.py XuoqKYxDHVc \
  --script input/XuoqKYxDHVc/dubbing_script.json \
  --speaker-profiles output/XuoqKYxDHVc/speaker_analysis/CONTENT_KEY/speaker_profiles.json \
  --seconds 30 --dry-run
```

本生成は従来どおり専用Qwen環境で実行する。今回はTTS・Voice Design・Cloneを実行しない。解析情報の内容ハッシュがTTS生成識別情報に入り、異なるプロフィールを混用しない。プロフィール引数なしの従来CLI・キャッシュ仕様は維持する。

## Audio Mixerとの接続

```bash
.venv/bin/python audio_mixer.py XuoqKYxDHVc \
  --feedback /path/to/feedback.json \
  --speaker-analysis output/XuoqKYxDHVc/speaker_analysis/CONTENT_KEY/diarization.enhanced.json \
  --seconds 30 --diagnostic --dry-run
```

人間確認後は `speaker_timeline.json` も指定できる。生成済みWAVと話者確認結果が異なる場合は、診断モードでも生成を止め、台本・TTSフィードバックの再作成を要求する。既存WAVのspeakerだけを勝手に付け替えない。

Mixerは元の発話区間に重なりがないのに日本語WAVが別話者と重なるペアを `introduced_other_speaker_overlaps` に分類し、通常モードで停止する。同一話者衝突の停止、自然な別話者同時発話の保持、UNKNOWN未補完、切断・倍速化禁止は維持する。自動配置最適化は今回の対象外。

## 成果物

```text
speaker_analysis/CONTENT_KEY/
  selected_source.wav
  diarization.enhanced.json
  word_speakers.enhanced.json
  utterances.enhanced.json
  speaker_profiles.json
  comparison.json
  speaker_review.json
  speaker_review.html
  review_decisions.template.json
  references/*.wav
  review_audio/*.wav
  manifest.json
```

`diarization.enhanced.json` は元の `turns` に加え、ID対応、窓ごとの埋め込み、相互類似度、境界監査、新旧検出の保守的な重なり除外区間を保持する。`comparison.json` はUNKNOWNや同時発話の件数・秒数を比較する。**人間による正解確認がないため、精度改善・誤り率は計算しない。**

## テストと限界

```bash
.venv-language/bin/python -m unittest discover -s tests -v
```

小さな合成波形・動画・偽Embedding backendでF0、変動、ポーズ、参考選択、同時発話除外、3人以上のID対応、曖昧一致、UNKNOWN、矛盾、非破壊、再実行、人手レビュー取込、TTS方式・Clone権限、Mixer話者変更停止を検証する。実モデルはテストの必須条件ではない。

今回の同じpyannoteモデルでの再解析は以前の区間推定と一致した。改善したのは音響根拠の保持・検証・レビューと安全な接続であり、実測の識別精度向上を証明していない。次は5つの問題箇所の人手聴取、必要ならASR時刻の再確認・注釈付き評価、誤り原因に応じたモデル・閾値の比較を行う。
