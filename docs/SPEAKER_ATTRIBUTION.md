# Phase 5: 音響情報による話者再判定

`speaker_attribution.py` はPhase 4の成果物と元のJSONを照合し、元動画を変更せず発話単位の話者を再評価する。既存モデル・環境を使う独立CLIで、従来の話者識別、翻訳、TTSは自動実行しない。

```bash
cd ~/Projects/transclip
.venv-diarize/bin/python speaker_attribution.py XuoqKYxDHVc \
  --analysis-dir output/XuoqKYxDHVc/speaker_analysis/1cc6eb8e5f0d7ccb
```

別の動画では、その動画のPhase 4出力先を指定する。`--input-dir`、`--output-dir`も指定できる。モデルは既存の `pyannote/speaker-diarization-community-1`。Phase 4と同じモデル・重み・環境情報であることを確認してから、保存済み埋め込みを再利用する。ネットワーク、モデルダウンロード、テレメトリは無効。新規ライブラリや音声分離モデルは不要。

## 再判定の流れ

1. Phase 4のマニフェスト、各成果物、元動画、元JSONの内容とハッシュを検証する。
2. 既存segmentationモデルを全選択範囲へ適用する。重なり合う推論窓の「1人以上」「2人以上」の発話判定を時間ごとに集計する。区間表が重なるだけで同時発話を確定しない。
3. 対象そのもの、少し広い窓、長い発話内の局所窓から埋め込み・F0・スペクトル・音量等を取得する。0.8秒未満の対象には単独の埋め込みを作らず、左右と中央の文脈窓を別々に調べる。
4. 対象と**すべての文脈窓**に重なる参考窓を除外する。参考窓同士も重なりを除き、複数の別区間との比較に限定する。既存プロフィールの全体平均をそのまま使うと対象音声自身が含まれる場合があるため、対象ごとに参照集合を組み直す。
5. 1・2・4秒の参考検証窓を作り、その親区間を除いた同話者の参考集合・他話者集合と比較する。長さによるスコアの違いを記録する。参考ラベルもモデル推定であり、これは人の正解データによる校正ではない。
6. 音質、発話長、混在票、候補間の差、独立参考数、複数窓の一致を満たす場合に限って話者を自動割当てする。前後の質問・応答は補助的な仮説として保存し、会話だけでは確定しない。
7. 人の確認ラベルがあれば、それを最終結果として優先する。自動判定は別欄に残して評価する。

ASRの単語区間はポーズを含む場合があるため、音声活動が区間の半分以上という条件ではなく、発話人数モデルの活動票と窓長から得る音声証拠量が0.40秒以上であることを要求する。この量もモデル推定で、正確な発音時間ではない。

F0不足だけで通常発話を不合格にはしない。無声子音やポーズを含むため、エネルギー・スペクトル・埋め込みをあわせて確認する。短い発話の文脈補完では、対象のF0と左右のF0の整合性も条件とする。

既存の話者ラベルは、新しい根拠が不足しただけでは削除しない。その場合は `previous_assignment_unverified` として未検証の継承を明示する。局所窓間の明確な不一致や混在根拠があれば、既知ラベルでも確認対象として保留する。

## スコアとしきい値

`SETTINGS` と発話ごとの `duration_reference_checks` に保存する。通常の固定基準はコサイン類似度0.70、次点との差0.18、独立参考2本以上。3本以上の有効な長さ別検証窓がある場合は、同話者比較の10百分位と他話者比較の95百分位の中点・他話者比較の95百分位+0.20・下限0.50の最大値を類似度基準に使う。次点との差・複数窓の一致は別途要求する。

短い発話の文脈による割当ては、類似度0.75・差0.25・左右中央の一致を要求する。0.12秒未満は自動割当てしない。音響の混在、音質、F0の不一致があれば保留する。これらは暫定規則で、スコアは正解確率ではない。

参考音声のラベルは同じpyannote由来であり、誤りが相関する。長さ別の自己整合性を調べても、人による精度検証の代わりにはならない。評価用の人間ラベルをしきい値選択に流用しない。

## 同時発話・境界・相づち・ASR時刻

`automatic.overlap` に次の仮説を保存する。

- A: 短時間の音響モデルが複数音声を支持する。`mixed` として、単一話者へ押し込まない。
- B: 区間表は重なるが、音響窓では単一話者が安定している。境界の見かけ上の重なりの可能性。
- C: 短い発話に混在の根拠がある。相づちの可能性も記録する。
- D: 単語時刻で音声活動が乏しい、音響候補と区間表が異なる等。ASR時刻ずれを含む仮説。
- E: 音響が不足、混在が曖昧、候補の差が小さい等。UNKNOWN。

これはA〜Eを常に正しく分類できるという意味ではない。segmentationは従来のpyannoteと同じモデルであり、Aも人が確認した事実ではない。元の全話者区間と対象に重なる話者ごとの区間を独立に保存し、音声や時刻を直列化・切断・変更しない。

音声分離や話者別の重複文字起こしは実装していない。混在する区間の文字を2人へ複製して、独立した発話が認識できたように扱わない。`mixed` はspeaker=nullで下流生成を保留し、元の音声と各話者区間を確認できるようにする。

## Speaker Reviewと人間ラベル

出力先の `speaker_review.html` を開く。対象そのものと前後音声、参考音声、新しい自動判定、窓ごとの類似度、混在票、特徴・しきい値・前後の会話を表示する。「複数話者の混在」も選択できる。確認者を入力して判断JSONを保存する。未確認の判断を人間ラベルとして作らない。

保存済みPhase 4/5レビューJSONを使って評価する例:

```bash
.venv-diarize/bin/python speaker_attribution.py XuoqKYxDHVc \
  --analysis-dir output/XuoqKYxDHVc/speaker_analysis/1cc6eb8e5f0d7ccb \
  --review-dir /path/to/review-bundle \
  --decisions ~/Downloads/speaker_review_decisions.json
```

レビューのキー、元データ、対象音声ハッシュ、確認者、話者IDを検証する。人間がUNKNOWN・混在と判断した場合も自動判定より優先する。人間ラベルがない場合、`evaluation.json` の正解率はnullである。ラベルがある場合だけ、自動割当て数、一致件数、保留を含むカバレッジ、以前のラベルとの比較を保存する。少数・偏ったレビュー集合の一致率を全体精度として解釈しない。

`human_evaluation.jsonl` に、人間ラベル・確認者・対象WAVハッシュ・時刻・自動判定・参考除外区間を保存する。将来の評価用であり、今回しきい値の学習に使っていない。人間ラベルが0件なら空ファイルとなる。

生成済みPhase 5の判断を再推論せず取り込むときは、既存 `speaker_review.py` を使える。未確認項目の自動判定は保持し、確認した項目だけ変更する。

```bash
.venv/bin/python speaker_review.py \
  --analysis-dir /path/to/phase5-output \
  --decisions ~/Downloads/speaker_review_decisions.json \
  --input-dir input/XuoqKYxDHVc \
  --output-dir output/XuoqKYxDHVc/speaker_review/review_002
```

## 下流処理

元のJSONは変更しない。出力先へ互換形式の `utterances.json`、`utterance_translations.json`、`word_speakers.json`、`selection.json` を保存する。ID・原文・訳文・元時刻を保ち、発話を勝手に結合しない。単語へ引き継ぐラベルも「発話音声による判断」であることを明記し、単語自体の独立した音響識別と混同しない。

```bash
.venv-language/bin/python dubbing_script.py XuoqKYxDHVc \
  --input-dir /path/to/phase5-output --seconds 30 --output /new/path/dubbing_script.json

.venv/bin/python audio_mixer.py XuoqKYxDHVc \
  --feedback /path/to/existing/feedback.json \
  --speaker-analysis /path/to/phase5-output/speaker_timeline.json \
  --seconds 30 --diagnostic --dry-run
```

`tts_regeneration.json` はラベル変更により台本・TTSフィードバックの作り直しが必要な発話、UNKNOWN/混在で生成を保留する発話を記録する。既存のWAVと新しい話者が異なる場合は、既存Mixerの照合で診断モードでも停止する。WAVの話者情報だけを変更して再利用しない。同一話者衝突・元にない他話者重なりの検出は既存Mixerを再利用する。

## 保存とテスト

出力は `output/<video_id>/speaker_attribution/<CONTENT_KEY>/`。入力・モデル・設定・人間判断・関連実装のハッシュでキーを作る。排他ロック、一時保存後の公開、既存結果のハッシュ検証・再利用に対応する。失敗した途中出力を成功扱いしない。モデル推論途中の細粒度再開は未実装。

主な追加成果物: `speaker_attribution.json`、`frame_activity.json`、`comparison.json`、`evaluation.json`、`human_evaluation.jsonl`、`speaker_timeline.json`、`tts_regeneration.json`、Speaker Review一式。旧Phase 4の成果物も変更しない。

```bash
.venv-language/bin/python -m unittest discover -s tests -v
```

合成音・モックで同一/異なる/3人以上の話者、短い相づち、境界、不確実性、混在、参考区間の独立性、長さ別の整合性、UNKNOWN、人間確認の優先、互換性、再利用・失敗後再実行、原本非破壊、台本/Mixer接続を検証する。
