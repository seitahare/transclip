# Phase 7 — Cross-lingual Voice Preservation

## 実装範囲と現在の検証状態

`voice_cloning.py` にローカルのApproved Voice Cloneバックエンド、登録・権限/品質検証、任意数の話者プロフィール、生成CLI、Preset比較HTMLを実装した。既存のCustomVoiceプリセットは維持する。

**2026-10-22更新: 新しい架空日本語参考音声に対するユーザーの再利用許可・試聴確認後、実モデルで3発話を生成した。** 日本語参考→日本語生成、Preset比較、Feedbackと診断MP4まで確認済み。生成後の日本語の正しさはユーザー確認済み。全文一致・数値の発音、声質の近さ、複数文章での聴感一貫性は別途確認待ち。Phase 4のインタビュー参考および旧 reference_v0.3.wav は未承認のままで使用していない。英語の新規架空参考も生成したが、再利用許可のみ確認済みで、発声内容・品質確認待ちのため英語参考→日本語Cloneは未実施。[統合結果](PHASE7_INTEGRATION.md) を参照。

## モデルと実行環境

既存 `~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv` を利用する。追加インストール、環境変更、モデルダウンロード、有料APIは不要だった。

|項目|確認結果|
|---|---|
|Base既定値|`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`|
|ローカルrevision|`e7dd0585652209fa0d7783659aad4e8a324de11c`|
|量子化|configの `bits: 8`, `group_size: 64`, `mode: affine`|
|ローカルsnapshot|14ファイル、合計3,104,158,788バイト（実行時メモリ量ではない）|
|確認したライブラリ|qwen3-tts-mlx 0.2.0 / mlx-audio 0.5.1 / mlx 0.32.2|
|実モデル推論|承認済み架空日本語参考から3発話生成成功。日本語の正しさはユーザー確認済み。全文一致・声質は確認待ち|

[Qwen公式](https://github.com/QwenLM/Qwen3-TTS)のBaseの参照音声・文字起こしによる条件付けと、[MLX版モデルカード](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit)、[使用中のMLXラッパー](https://github.com/howon-kim/Qwen3-TTS-MLX)を確認した。[0.6B Base 8bit](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit)も代替候補だがローカルに未導入で、性能/メモリの実測比較は行っていない。1.7Bがメモリ不足になる場合に、別途承認を得て検討する。途中ダウンロードのBase bf16は使用しない。

`QwenBaseSynthesizer` は既存 `QwenPresetSynthesizer` のオフラインsnapshot/設定検証を再利用し、モデル種別 `base` を要求する。`snapshot_download(local_files_only=True)` で存在確認し、推論ロードは最初の未生成WAVが必要になるまで遅延する。Presetへの暗黙フォールバックはしない。

実装は `create_voice_clone_prompt(ref_audio=(samples, rate), ref_text=...)` で参考を準備し、ネイティブ `model.generate(text=..., lang_code='Japanese', ref_audio=..., ref_text=...)` を呼ぶ。ファサードが捨てるtoken数を保持して上限到達を検出するため、この呼び分けを行う。空/非有限/不正な波形、振幅範囲超過、異常サンプルレート、token上限到達は公開前に拒否する。PCM16 WAVで保存し、Feedbackがフレーム数・長さ・音量・ハッシュを再測定する。

モデル自身がデコード時に参考音声部分を取り除く処理と、発話を元動画の尺に合わせて切る処理は別である。本実装で生成した日本語を途中切断・倍速化する処理は追加しない。

## 用意するファイル

1話者ごとに次の4点を用意する。ファイル名・話者ID・動画ID・パスは任意で、特定人物に固定していない。

|ファイル|条件|
|---|---|
|`reference.wav`|非圧縮PCM WAV、モノラル、2.5～30秒。PCM16・24 kHzを推奨。他話者の混入がなく、文頭/語尾を切らず、明瞭な単独発話を使用。|
|`reference.txt`|そのWAVで実際に話された正確な英語。UTF-8。推測で補完せず、聞いて一致を確認する。空白・改行を含めた文字列全体をハッシュ照合する。|
|`consent.json`|本人同意等の根拠、確認者、対象speaker_id、WAVハッシュ、scope、明示許可。動画を所有しているだけでは作成できない。|
|`review.json`|文字起こしとWAVの組合せ、および話者/混在/明瞭さ/切断の人間の確認記録。|

自分で録音した場合も、自分の音声である根拠と確認者を記録する。音声と文字起こしが不一致なら修正して再確認し、検証済みとして勝手に補完しない。分離音声より明瞭な元の単独発話を優先する。分離音声を使用する場合は、`source_kind: "separated"` と `separation_artifacts_reviewed: true` が追加で必要。

## 登録手順

まず未承認テンプレートを作る。これは許可・品質確認を与える操作ではない。

```sh
cd ~/Projects/transclip
.venv-language/bin/python voice_cloning.py template \
  --speaker SPEAKER_A \
  --reference-audio /absolute/path/reference.wav \
  --reference-text /absolute/path/reference.txt \
  --output /absolute/path/voice-review
```

`consent.template.json` と `review.template.json` に対象ハッシュが入る。原本テンプレートとは別名の `consent.json`, `review.json` として、人間が実際に確認した内容を保存する。

許可記録の形式（`authorized: false` のままでは登録不可）:

```json
{
  "authorized": false,
  "scope": "voice_cloning",
  "speaker_id": "SPEAKER_A",
  "reference_audio_sha256": "テンプレートで計算されたWAVのSHA-256",
  "evidence": "",
  "verified_by": ""
}
```

品質確認はテンプレート内の以下を確認後に記入する:

- `transcript.status: "verified"`, `transcript.verified_by`。音声・文字起こしハッシュは確認対象と一致していること。
- `quality.verified_by`, `speaker_identity_verified`, `single_speaker_verified`, `intelligibility_verified`, `boundaries_verified` を実際の確認に基づいて記入。
- `known_overlap_seconds: 0`, `unresolved_issues: []`。不明・未解決なら登録しない。

登録:

```sh
.venv-language/bin/python voice_cloning.py register \
  --speaker SPEAKER_A \
  --reference-audio /absolute/path/reference.wav \
  --reference-text /absolute/path/reference.txt \
  --consent /absolute/path/consent.json \
  --review /absolute/path/review.json \
  --output output/voice_registry
```

`output/voice_registry/SPEAKER_A-CONTENT_HASH/voice_profiles.json` が出力される。WAV・文字起こし・許可・レビューを別フォルダーへコピーし、元ファイルは変更しない。既存の登録先を別内容で上書きしない。登録後に生成で参照するのは、この登録フォルダー内の記録である。**許可撤回時は登録済みのconsent.jsonを更新/無効化し、そのプロフィールを使用しないこと。登録元の記録更新を自動監視する仕組みではない。**

複数話者は話者ごとに登録する。生成時に `--profiles` を複数指定できる。重複speaker_idを拒否し、UNKNOWN / REVIEW / MIXEDにはプロフィールを割り当てない。

### Phase 4の参考候補

```sh
.venv-language/bin/python voice_cloning.py candidates \
  --phase4-profiles output/VIDEO_ID/speaker_analysis/KEY/speaker_profiles.json \
  --speaker SPEAKER_A
```

既存の候補評価順、品質、特徴、重複除外、ハッシュを利用する。どの候補も、許可・正確な文字起こしの確認なしでは `eligible_for_clone: false` のまま。

登録時に `--phase4-profiles ... --reference-id ref-001` を加えると、指定音声のハッシュとPhase 4の適格性を照合し、F0等のDSP推定・埋め込みモデル情報・候補根拠を `voice_characteristics` へ引き継ぐ。新たに人物名や同意を推定しない。Phase 4なしの自己録音も登録でき、その場合は任意の声質分析データがなくても許可・品質確認とWAV測定を必須にする。

## Voice Profileとキャッシュ

登録形式は `kind: synthesis_voice_profiles`, `format_version: 1`, `profiles[]`。`VoiceProfile.speaker` がspeaker_id、`mode` がvoice_modeに対応する。参考音声パス・正確なtext・WAV SHA-256・許可記録パス・文字起こし確認・品質確認・任意の声質特徴を持つ。

生成時はプロフィール全体と実際の許可記録・WAV測定を `voice_profile` snapshotとして各Feedback entryに保存する。モデル/revision/settingsは各entryの `tts` と同じキャッシュ要求に保存する。人物ごとの設定と実行モデルをこの組合せで追跡する。

- Presetは従来のv1キャッシュ要求を維持。Cloneはspeaker_id・mode・完全なprofile snapshotを含むv2要求。
- WAV、文字起こし、話者、プロフィール、許可記録、モデル/revision、生成設定、日本語、関連生成コード/ライブラリが変われば旧音声を使わない。
- キャッシュがあっても、現在の登録済み音声・文字起こし確認・許可を再検証する。撤回・ハッシュ不一致は拒否。
- 1件ずつWAVとreceiptをアトミックに公開し、既存Feedbackのロック/途中再開を利用。成功済みWAVは再実行時に再生成しない。
- モデルと参照promptをプロセス内で再利用する。ネイティブの参照codec/textキャッシュはprofile SHA単位で分離する。使用ライブラリの内部キーが音声長/和に依存するため、別話者間で同じ内部キーになっても混ざらないようにしている。
- 音声埋め込みのディスク保存/pickle化は行わない。再起動後、未生成発話を処理する場合は参考の再エンコードが必要。speaker encoder部分の再計算も残る。キャッシュ利用が聴感上の一貫性を保証するわけではない。

## 生成CLI（権限確認済み音声の準備後）

まずdry-run。モデルを推論ロードせず、許可・対象発話・除外理由を確認する。

```sh
.venv-language/bin/python voice_cloning.py generate VIDEO_ID \
  --script /absolute/path/dubbing_script.json \
  --mode approved_clone \
  --profiles output/voice_registry/SPEAKER_A-HASH/voice_profiles.json \
  --output output/VIDEO_ID/approved_clone --seconds 30 --dry-run
```

実生成は既存Qwen環境:

```sh
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python voice_cloning.py generate VIDEO_ID \
  --script /absolute/path/dubbing_script.json \
  --mode approved_clone \
  --profiles output/voice_registry/SPEAKER_A-HASH/voice_profiles.json \
  --output output/VIDEO_ID/approved_clone \
  --entry-id dub-001 --entry-id dub-002 --entry-id dub-003 --max-jobs 3 --seconds 30
```

`--model` は別のローカルBaseモデル、`--seed` は生成seed。話者2人・3人以上でも `--profiles` を追加するだけで、数を固定しない。選択台本と動画IDを照合し、対象範囲外・UNKNOWN・プロフィール未登録の発話は補完生成しない。`--max-jobs` は新規生成数の上限で、途中再開は同じコマンドを使う。

Presetベースライン用 `voices.json` は `{"SPEAKER_A":"Ryan"}` のように明示する。同じ台本・発話ID・secondsで、別の出力先を使う:

```sh
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python voice_cloning.py generate VIDEO_ID \
  --script /absolute/path/dubbing_script.json --mode preset \
  --voice-map /absolute/path/voices.json --output output/VIDEO_ID/preset_baseline \
  --entry-id dub-001 --entry-id dub-002 --entry-id dub-003 --max-jobs 3 --seconds 30
```

従来の `tts_generator.py --script` も利用可能。架空Voice Design参考の生成は `design-reference` を追加した。生成だけでCloneを許可しない。

## Feedback・Mixer・比較

各entryにID、speaker、日本語、元時間・推定時間・実測時間、誤差、profile snapshot、TTS設定、cache key、WAV品質、review_requiredを記録する。Cloneは発音/声質未確認のレビュー理由を保持する。音声の長さはsample_frames / sample_rateで計測する。

Mixerはv1 Preset / v2 Cloneのreceiptを照合し、話者とプロフィールの不一致を拒否する。許可は計画作成時と描画前に確認する。Cloneを含む同一話者衝突は**診断モードでも停止**。異なる話者の元からある重なりは保持し、新しく生じた重なりは区別して通常モードで停止する。音声切断・倍速化はしない。既存のPreset診断モードの仕様は維持する。

比較表と試聴HTML:

```sh
.venv-language/bin/python voice_cloning.py compare \
  --preset-feedback /absolute/path/preset/runs/KEY/feedback.json \
  --clone-feedback /absolute/path/clone/runs/KEY/feedback.json \
  --output output/VIDEO_ID/voice_comparison
```

同じ発話ID/話者/テキスト/時刻を照合し、`comparison.json`, `comparison.md`, `review.html` と参照・Preset・CloneのWAVコピーを別フォルダーに出す。HTMLの再生ボタン、またはMacの `afplay '/absolute/path/speech.wav'` で聞き比べる。生成時間差・音量・破損検査・同一プロフィール使用を記録するが、発音/声質の成功や聴感一貫性を自動で宣言しない。日本語参考→日本語生成のEmbedding比較は `quality` に追加した。英語参考→日本語生成の比較は未実施。

## テスト・残る検証

```sh
.venv-language/bin/python -m unittest discover -s tests -v
```

初期Phase 7は開始時193件、追加後232件成功（新規39件）。旧TRANS統合時はこの232件を再実行し、追加23件を含め合計255件成功。新規39件は、参照WAV/品質、text/audioの一致確認、許可/ハッシュ、任意数話者、UNKNOWN、ネイティブ生成呼び出し/失敗、条件付け再利用/分離、キャッシュ/再開/変更、Feedback、Mixer、比較、非破壊をモック・短いテスト波形で検証する。**テストの許可記録は架空のfixtureで、実在人物の許可記録ではない。**

新しい架空の1話者から3発話の日本語WAVを実生成し、時間・波形品質・条件付け再利用を確認した。次に発音・欠落・声質・文章間の一貫性を聴取評価し、英語参考の承認後に言語をまたぐ生成を検証する。モック成功・実生成成功・聴感品質の合格は区別する。原音声・既存JSON・仮想環境は変更せず、commit/pushは行わない。
