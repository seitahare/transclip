# TransClip Phase 13 — 汎用動画吹き替え品質改善（Universal Video Dubbing Quality Improvements）

Phase 13 では、Phase 12 で構築した日本語吹き替えパイプラインを基盤とし、**任意の海外動画・インタビュー・対談・講義などを自然な日本語吹き替えで視聴できる汎用パイプライン**へと拡張しました。特定の動画・人物・話者ID・タイムスタンプに依存した実装は一切含まず、入力された動画と解析結果（`dubbing_script.json` 等）から汎用的に処理します。

## 実装された主要機能

### 1. オリジナル音声のダッキング（ボイスオーバー方式）[`audio_mixer.py` / `dubbing_pipeline.py`]
- **日本語TTSメイン＋背景オリジナル音声（10〜15%）**:
  - 日本語TTS発話が存在する区間では、オリジナル音声を既定 `12%`（`--duck-gain 0.12`、0.0〜1.0で任意調整可能）に減衰して重ねます。
- **滑らかな音量遷移とマイクロギャップ連結**:
  - 発話開始前に `fade_out_seconds`（既定 `0.18s`）、終了後に `fade_in_seconds`（既定 `0.25s`）のコサイン（または線形）フェードを適用します。
  - 近接する発話間の短い隙間（`min_unspoken_gap_seconds=0.35s` 未満）はダッキング状態を維持してブリッジし、不自然な音量のバタつき（ポンピング）を防止します。
- **未発話区間の原音保持**:
  - 日本語TTSが存在しない区間ではオリジナル音声を `100%`（`--unspoken-gain 1.0`）保持し、元話者の息遣い・間・会場の拍手や空気感をそのまま残します。
- **クリッピング防止と完全吹き替え方式の併存**:
  - ソフトニー・ピークリミッタ（`apply_soft_limiter`, 上限 `-1.0 dBFS`）とラウドネス検証（`verify_audio_loudness_and_clipping`）を常時適用します。
  - 従来の完全吹き替え方式（`--mix-mode japanese-only` / `full-dub`）も引き続き利用可能です。

### 2. 映像とTTSの発話タイミング同期 [`timing_adjustment.py` / `dubbing_pipeline.py`]
- **文節・意味単位とタイムスタンプの対応付け**:
  - `align_semantic_units_to_utterance_window()` により、各発話の `sentence_units` または日本語文節（clauses）をモーラ重みと句読点ポーズに基づいて元発話の `[start, end]` 時間枠へ汎用的にマッピングします。
- **0.9〜1.3倍の双方向・話速自動調整**:
  - `compute_auto_speech_rate_ratio()` および `_apply_bounded_atempo_non_destructive()` により、日本語TTSが元発話枠より長い場合は最大 `1.30倍`（`--max-atempo 1.30`）まで加速し、逆に日本語TTSが早く終わる場合は最小 `0.90倍`（`--min-atempo 0.90`）まで緩やかに減速して口の動き（リップシンク）と発話枠に合わせます。
  - 過度な時間圧縮・伸長（0.9倍未満・1.3倍超）や音声の途中切断を厳格に防止します。
- **早期終了・未吹き替え発話・長尺無音の自動検出**:
  - `detect_timing_and_coverage_anomalies()` により、日本語TTSが早く終わるケース（`early_tts_finishes`）、元話者が発話中なのに吹き替えが存在しない区間（`undubbed_active_speech_intervals`）、長い無音（`long_silence_gaps`）、タイミングずれ（`lip_sync_drift_records`）を自動検出してレポートに記録します。

### 3. 環境音の保持と音源分離ミキシング [`audio_mixer.py` / `dubbing_pipeline.py`]
- **話者音声と背景環境音・ルームトーンの分離**:
  - `separate_ambient_and_speech()` により、外部ステム分離モデル（Demucs / TorchScript / ConvTasNet）または内蔵の **STFTスペクトル＆Mid/Side空間マスク環境音抽出エンジン（`dsp_spectral_ambient`）** を用いて、話者の声（300〜3400Hzセンター帯域）を抑えつつルームトーンや会場環境音を保持します。
- **複数ミキシング方式のサポートと自動フォールバック**:
  - `voiceover`（元音声を10〜15%残すダッキング方式）
  - `ambient`（分離した環境音ステムを背景に残す方式）
  - `hybrid-ambient`（分離環境音＋軽微なダッキング原音をブレンドする方式）
  - `japanese-only`（完全吹き替え方式）
  - 音源分離バックエンドが利用できない環境では、自動的にダッキング方式（`ducking_fallback`）へフォールバックします。

### 4. 声質とTTSの改善 [`voice_quality.py` / `dubbing_pipeline.py`]
- **X-vectorボイスクローニングとプリセット音声カタログ選択**:
  - 参照音声からのX-vector抽出に加え、6種類のプリセット音声カタログ（`Ryan`, `Aiden`, `Ethan`, `Claire`, `Mia`, `Sophia`）から話者の年齢層（`young` / `adult` / `mature`）、雰囲気（`calm` / `energetic` / `authoritative` / `warm` 等）、基本周波数（F0）に合う声を自動推薦・選択する `recommend_preset_voice()` を実装しました。
- **日本語プロソディ最適化と声質・自然さ検証**:
  - `optimize_tts_prosody_settings()` による話し方スタイル別（`natural`, `calm`, `conversational`, `expressive`, `lecture`）の生成パラメータ最適化。
  - `evaluate_dubbing_voice_and_naturalness()` により、F0距離（半音単位）・スペクトル重心類似度・モーラ速度（4.0〜11.5 mora/s）・ポーズ比率を自動計測します（人手による聴覚確認とは区別して記録）。
- **権利・公開ガバナンスの分離**:
  - `RightsAndPublicationPolicy` により、実験用合成話者（`SYNTH_*`）と実在人物の利用権限確認を厳格に分離して管理します。

### 5. 字幕UIの改善 [`japanese_subtitles.py` / `dubbing_pipeline.py`]
- **視認性強化（アウトライン・シャドウ・半透明黒座布団）**:
  - 文字アウトライン（`outline=3.5`）、ドロップシャドウ（`shadow=2.5`）、および半透明の黒背景ボックス（座布団：`BorderStyle=3`, 既定不透明度60% `&H66000000`）により、明るい背景映像でも確実に読める字幕を描画します。
- **サイズ拡大（1.2〜1.3倍）と位置引き上げ・解像度自動適応**:
  - 従来の字幕サイズ（1080p基準42pt）から **約1.25倍（52pt）** に拡大し、下端マージンを **36px → 64px** へ引き上げました。
  - `SubtitleStyleConfig.for_video_resolution(width, height)` により、横型（16:9）、縦型ショート（9:16）、正方形（1:1）など任意の解像度・アスペクト比へ自動スケーリングし、文字のはみ出し（overflow）を防止・自動検出します。
- **表示用句読点整形（元台本非破壊）**:
  - 翻訳台本（`dubbing_script.json` / `text_ja`）は一切変更せず、字幕表示用テキスト（`display_lines` → SRT/ASS）では **句点「。」を省略**し、**読点「、」を半角スペースに置換**（または `minimal` で最小限化）します。

---

## CLI の使い方

### 1. Phase 13 推奨設定（ボイスオーバー12%ダッキング＋0.9〜1.3x同期＋改善字幕UI）で実行
```bash
python dubbing_pipeline.py run-phase13 \
  --video path/to/source_video.mp4 \
  --script path/to/dubbing_script.json \
  --output output/my_video_phase13
```

### 2. 環境音分離モード（`ambient` / `hybrid-ambient`）や完全吹き替えモード（`japanese-only`）の指定
```bash
# 音源分離による環境音保持モード
python dubbing_pipeline.py run-phase13 \
  --video path/to/source_video.mp4 \
  --script path/to/dubbing_script.json \
  --mix-mode ambient \
  --ambient-backend auto \
  --ambient-gain 0.85 \
  --output output/my_video_ambient

# 完全吹き替えモード（元音声を消去）
python dubbing_pipeline.py run-phase13 \
  --video path/to/source_video.mp4 \
  --script path/to/dubbing_script.json \
  --mix-mode japanese-only \
  --output output/my_video_fulldub
```

### 3. 話者の年齢・雰囲気に合わせたプリセット音声の自動選択
```bash
python dubbing_pipeline.py run-phase13 \
  --video path/to/lecture.mp4 \
  --script path/to/dubbing_script.json \
  --voice-mode preset \
  --auto-select-preset \
  --speaker-age SPEAKER_00=mature \
  --speaker-atmosphere SPEAKER_00=authoritative \
  --speaker-age SPEAKER_01=young \
  --speaker-atmosphere SPEAKER_01=energetic \
  --prosody-style lecture \
  --output output/lecture_preset_dub
```

### 4. Phase 12 の話者確認記録・改訂プランとの組み合わせ
```bash
python dubbing_pipeline.py run-phase13 \
  --video output/phase11_1885_1915/source.mp4 \
  --script output/phase11_1885_1915_inputs/dubbing_script.json \
  --speaker-decisions-json output/phase12_1885_1915_quality/speaker_decisions.json \
  --editorial-plan-json output/phase12_1885_1915_quality/editorial_plan.json \
  --revised-script-out output/phase13_1885_1915/dubbing_script.revised.json \
  --mix-mode voiceover \
  --duck-gain 0.12 \
  --min-atempo 0.90 \
  --max-atempo 1.30 \
  --subtitle-font-scale 1.25 \
  --subtitle-box-opacity 0.60 \
  --subtitle-comma-mode halfwidth_space \
  --output output/phase13_1885_1915
```
