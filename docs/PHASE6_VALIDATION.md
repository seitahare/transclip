# Phase 6 実装・実データ検証（2026-10-19）

実装・自動テスト・冒頭30秒の実音声推論まで実施しました。分離候補3区間から6本のWAVを生成しましたが、人間の正解確認を経た「きれいに分離できた区間」は0件です。話者識別精度の向上は未確認です。

## 試聴と成果物

- [最終Speaker Review](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/run-033d62175c1e-595j157m/review.html)：元音声と全Stemを比較試聴し、判断を別JSONへ保存できます。
- [区間・原発話別の比較](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/run-033d62175c1e-595j157m/comparison.md)
- [分離結果と根拠](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/run-033d62175c1e-595j157m/speaker_separation.json)
- [話者別文字起こしと単語時刻](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/run-033d62175c1e-595j157m/speaker_transcriptions.json)
- [初回推論の計測記録](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/run-033d62175c1e-n7h6ln59/speaker_separation.json)

ローカルでレビューHTMLを開き、各音声の再生ボタンを押してください。個別WAVは下表から開くか、Macで `afplay '/絶対パス/対象.wav'` でも再生できます。別PCへ移す場合は、runフォルダーだけでなく隣接するcache内の参照音声も必要です。

## 入力と実行条件

- 動画：XuoqKYxDHVc。選択済み範囲：1653.60–1942.99秒、今回の処理はその0–30秒。
- 既存のPhase 4プロフィール・pyannote埋め込み、Phase 5フレーム活動票、元ASR、YouTube自動字幕を利用。
- Conv-TasNet Libri2Mix：同時2人、モデル入力8 kHz、比較用出力16 kHz。3人以上の実音声は今回未検証。
- 承認済みの新規ダウンロードは重み20,049,107バイト（約19.1 MiB）のみ。
- SHA-256：`ddf97d8341f616c6d8d1a4b20dd51042760392c6543e9eaac276c9f96eba0f0e`。
- Apple Silicon / macOS arm64、既存 `.venv-diarize` のCPUで動作。PyTorch 2.14.1 / torchaudio 2.11。ASRは既存 `.venv` のキャッシュ済みfaster-whisper small.enを別プロセスで再利用。
- パッケージ追加・環境再構築・有料API・Voice Cloningは実行していません。

## 実測結果

|項目|結果|
|---|---:|
|処理窓|18|
|元音声を保持した窓|15|
|分離波形を生成した窓|3|
|分離WAV|6|
|最終実行の処理失敗|0|
|人間が分離成功を確認した窓|0（未確認）|
|初回処理本体|41.321秒|
|分離モデル呼び出し合計|1.016秒|
|再実行キャッシュ|18 / 18窓|
|最終レビュー再出力の処理本体|0.103秒|
|初回親プロセスの最大RSS|0.678 GiB|
|初回ASRワーカーの最大RSS|1.504 GiB|

時刻は処理本体の計測で、Python起動・事前モデルファイルハッシュ計算などは含みません。RSSは各プロセスのhigh-water値で、2つを足して同時最大使用量とは扱えません。キャッシュ再出力のASRメモリ欄は保存済みジョブの初回値です。再出力時にASRワーカーを新規起動して測定した値ではありません。

Conv-TasNetの生出力には大きな振幅・反転極性がありました。全Stem共通の符号付き係数で元波形に合わせ、係数と補正前後の残差を保存しています。合成したStemと元音声の補正後RMS残差比は0.048 / 0.082 / 0.078。これは音声の再構成診断であり、各Stemが単独話者になった証明ではありません。

## 重点5か所

「実際の同時発話」は聴取確認がないため未確定です。下記の分離必要性はモデルに基づく候補判断です。全5か所で人間の確認が必要です。

|元発話|今回の処理|ASR比較|話者・残る問題|
|---|---|---|---|
|Dario|15.134–15.522秒を重複候補として2本生成。文脈付きWAVは14.784–15.872秒。|Stem 1は “I'm Dario”、Stem 2は対象核心の認識語なし。追加の “I'm” は根拠不足として警告。|核心0.388秒で埋め込み対象外。両方UNKNOWN。元単語時刻15.42–15.68秒と重複境界がずれており、境界・短い相づち・ASR時刻を要確認。|
|famously, what|16.535–18.087秒を重複候補として2本生成。|Stem 1は “famously what”。Stem 2は “Hahaha, that's all.” で元ASR/字幕の支持がない語句を警告。|最良cosineは各0.307 / 0.257で採用しきい値未満。両方UNKNOWN。笑い声・混入・ASR幻覚の可能性を未確定で保存。|
|You're building humanoid robots.|23.572–25.394秒を重複候補として2本生成。|Stem 1は元の文に対応。Stem 2は “Yes.” と “welcome.”。特に追加語句・時刻を要確認。|最良cosineは0.415 / 0.165で採用しきい値未満。両方UNKNOWN。文が認識できたことだけでは分離成功とはしない。|
|Yes, he|この発話そのものは強い重複根拠がなく、25.394秒以降の元音声を保持。|区間ASRは “he also said …”。“Yes” は前の分離候補側にも出ており境界不整合の可能性。|周囲の声にはSPEAKER_01候補があるが、短いUNKNOWNそのものの根拠には使わず文字起こしの話者はUNKNOWN。|
|like,|強い重複根拠がなく元音声を保持。|区間ASRに “like” が含まれる。|元の0.06秒の断片から本人を確定できずUNKNOWN。周囲の話者ラベルを機械的に継承しない。|

元の30秒に重なる15発話のUNKNOWNは5件で、元ファイルは変更していません。新しいASRは異なる区切りの15行でUNKNOWNが10行、分離6Stemの自動話者IDはすべてUNKNOWNです。分割単位が違うため、この数の増減を識別精度として比較してはいけません。既知話者に強い根拠がある単独発話だけ候補を記録しています。実名推定はしていません。

## 比較用WAV

|分離核心（クリップ相対秒）|元音声|Stem 1|Stem 2|
|---|---|---|---|
|15.134–15.522|[原音](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/b7c52d01068068af3a43a8575f1990ccd95f438b0027654bd1dbc6ab5469c9dc/attempt-yw0wlle1/original.wav)|[Stem 1](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/b7c52d01068068af3a43a8575f1990ccd95f438b0027654bd1dbc6ab5469c9dc/attempt-yw0wlle1/stem-1.wav)|[Stem 2](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/b7c52d01068068af3a43a8575f1990ccd95f438b0027654bd1dbc6ab5469c9dc/attempt-yw0wlle1/stem-2.wav)|
|16.535–18.087|[原音](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/652a81be6668937797736049e69174e0fe8c56666b55e40f99924bb7ae7f399b/attempt-52ii_mq8/original.wav)|[Stem 1](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/652a81be6668937797736049e69174e0fe8c56666b55e40f99924bb7ae7f399b/attempt-52ii_mq8/stem-1.wav)|[Stem 2](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/652a81be6668937797736049e69174e0fe8c56666b55e40f99924bb7ae7f399b/attempt-52ii_mq8/stem-2.wav)|
|23.572–25.394|[原音](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/3a5c6afa0e340ed4155213ea0e4d49dfea8915413c2ec11bc77799b6a43a3c20/attempt-2qn4y2mi/original.wav)|[Stem 1](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/3a5c6afa0e340ed4155213ea0e4d49dfea8915413c2ec11bc77799b6a43a3c20/attempt-2qn4y2mi/stem-1.wav)|[Stem 2](/Users/doiseita/Projects/transclip/output/XuoqKYxDHVc/speech_separation/phase6/cache/3a5c6afa0e340ed4155213ea0e4d49dfea8915413c2ec11bc77799b6a43a3c20/attempt-2qn4y2mi/stem-2.wav)|

## 変更ファイルと検証

追加：`speech_separation.py`、`separation_backends.py`、`speaker_transcription.py`、`separation_review.py`、`tests/test_speech_separation.py`、`docs/SPEECH_SEPARATION.md`、本レポート。

変更：`speaker_review.py`（新レビューへの入口6行）、`README.md`、`docs/ARCHITECTURE.md`。翻訳・TTS・Mixerの本体を大規模変更していません。

- 開始時：既存153テスト成功。
- 追加：合成音声・モック40テスト成功。1人/2人/3人、途中登場、同時3人、容量超過、短い相づち、Stem順序、欠落/漏れ/不正WAV、幻覚警告、時刻、UNKNOWN、人間の判断、キャッシュ、破損再生成、途中失敗から再開、1時間分割、元入力非破壊、翻訳スキーマ、TTS声変更、Mixerの古いID拒否を確認。
- 全193テスト成功。合成フィクスチャの正解を確認した結果であり、実音声の識別精度測定ではありません。
- 実音声：上記3区間の実モデル推論とWAV出力、全18区間のキャッシュ再利用を確認。
- レビューHTML内の39音声参照がすべて存在し、JavaScript構文検査成功。ブラウザーでの表示・操作の実視認は未実施。
- 原入力JSONと元動画のハッシュ一致を確認。既存WAVは読み取りのみ。Git commit / pushなし。

## 未解決と次の段階

まずレビューで重点5か所と全Stemを聴取し、話者、混入、ASRの語句/時刻を確認してください。判断済みJSONがまだないため、実データの下流向け確定出力は作成していません。未確認出力から翻訳API、TTS、MP4生成は行っていません。

実装済みの互換アダプタは、人間が全音声・文字起こし・発言の網羅を確認した場合のみ、新しいv3発話とMixer用話者情報を生成します。冒頭30秒は元発話15の途中で終わるため、そのままの下流出力は境界検査で拒否します。該当発話の終わりまで処理範囲を延ばして再確認する必要があります。ASR修正の注記がある場合も、単語時刻を勝手に作らず出力を保留します。

優先する追加改善は、(1) 正解付きの人間レビュー、(2) 分離音声に適した埋め込み・短発話校正、(3) 単語境界の再整列とASR修正UI、(4) 必要な動画での3人以上対応モデル比較です。追加モデルは未導入で、MPS性能・別動画・多人数の実音声性能は未検証です。
