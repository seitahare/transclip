# Phase 8+ — Cross-lingual Voice Cloning 日本語韻律（Prosody）改善仕様書

## 1. 課題と原因の特定（ICL方式 vs X-vector-only方式）

### 1.1. 現行 `voice_cloning.py`（ICL方式）の内部挙動
現行の `voice_cloning.py`（`QwenBaseSynthesizer`）では、[Qwen3-TTS-MLX](https://www.google.com/search?q=https://github.com/howon-kim/Qwen3-TTS-MLX)（`qwen_tts_mlx` v0.2.0）と [mlx-audio](https://github.com/Blaizzy/mlx-audio)（`Blaizzy/mlx-audio` v0.5.1）を以下の経路で呼び出している。

1. `self._model.create_voice_clone_prompt(ref_audio=(wav, rate), ref_text=profile.reference_text, x_vector_only_mode=False)`
   - 英語参考音声（12.48秒、話者ID `SYNTH_EN_01`）を24kHzにリサンプルした波形 `prompt.ref_audio` と英語テキスト `prompt.ref_text` を保持する。
2. `native.generate(text=text, lang_code='Japanese', ref_audio=prompt.ref_audio, ref_text=prompt.ref_text, **options)`
   - `mlx_audio/tts/models/qwen3_tts/qwen3_tts.py` の `Model.generate` は、`ref_audio` と非空の `ref_text` が同時に渡されると **ICL（In-Context Learning）経路 (`_prepare_icl_generation_inputs`)** を選択する。
   - ICL経路では：
     - ECAPA-TDNN `speaker_encoder` で2048次元の **Speaker Embedding（x-vector）** を抽出して `<|spk|>` トークン位置に注入するだけでなく、
     - `speech_tokenizer.encode(ref_audio)` によって **英語参考音声12.48秒分（12.5Hz × 12.48s ≈ 156フレーム、16コードブック）の離散音響トークン (`ref_code`)** と、**英語参考テキストのトークン列 (`ref_ids`)** を自己回帰Transformerのコンテキスト先頭にそのまま連結（プリペンド）する。

### 1.2. なぜ「単語の発音は正しいのに外国人が片言で話しているようなイントネーション」になるのか
- **日本語の単語発音が比較的正しい理由**: 生成ヘッダに `lang_code="Japanese"` が設定されており、対象テキストが日本語トークンとして入力されるため、音素・拍（カナ単位）の選択自体は日本語としてデコードされる。
- **イントネーションとリズムが外国語的になる根本原因**:
  1. **156フレームの英語音響トークン (`ref_code`) によるIn-Context韻律転移**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) の12Hzトークナイザの第1コードブック（意味・韻律トークン）とサブトーカー（残差音響コードブック）は、直前の156フレームにわたる **英語の強勢拍リズム（stress-timed rhythm）、英語特有の強いピッチ上下動、英語の句末イントネーション** に強く条件付けされた状態から自己回帰生成を継続する。
  2. **英語 `ref_text` と日本語 `text` のクロスリンガル境界**: 英語テキストの直後に日本語テキストを連結して音声継続（speech continuation）を行うため、日本語本来の **モーラ拍リズム（等時拍）** や **高低ピッチアクセント（下り目核と助詞の滑らかな接続）** ではなく、英語の強弱ストレスを日本語の文節頭に当てはめたような「片言・外国人風」の抑揚が生じる。

---

## 2. X-vector-only モード（`x_vector_only_mode=True`）の実装状況と注意すべき罠

### 2.1. `qwen_tts_mlx` と `mlx_audio` における対応状況
- `qwen_tts_mlx` の `MLXQwen3TTS.create_voice_clone_prompt(..., x_vector_only_mode=True)` は、`prompt.ref_text = None` かつ `prompt.x_vector_only_mode = True` を設定した `VoiceClonePromptItem` を返す。
- `mlx_audio` の `native.generate(text=..., lang_code="Japanese", ref_audio=prompt.ref_audio, ref_text=None, **options)` は、`ref_audio` が存在し `ref_text is None` のとき、ICL経路（`_prepare_icl_generation_inputs`）を通らず、**通常生成経路（`_prepare_generation_inputs`）** を通る。
- `_prepare_generation_inputs` の内部では：
  - `self.extract_speaker_embedding(ref_audio)` により **2048次元のECAPA-TDNN Speaker Embedding（声色ベクトル）のみ** が抽出され、`<|spk|>` トークン位置に埋め込まれる。
  - `speech_tokenizer.encode(ref_audio)` による英語の音響トークン (`ref_code`) も、英語テキスト (`ref_text`) も一切コンテキストに入らない。
  - そのため、**声色（Timbre）は英語参考音声のSpeaker Embeddingから再現しつつ、ピッチアクセント・モーラ拍リズム・文末イントネーションは `lang_code="Japanese"` の日本語ネイティブ事前分布に従わせる** ことができる。

### 2.2. 実装上の重要な注意点（`prosody_experiments.py` で対処済み）
- 単に `create_voice_clone_prompt(..., x_vector_only_mode=True)` を呼ぶだけでは不十分であり、`native.generate(...)` へ渡す `ref_text` が確実に `None` であること、および実行後に `native._icl_cache` が空であること（ICL経路への意図しないフォールバックが起きていないこと）を検証する必要がある。
- `prosody_experiments.py` の `QwenProsodyCloneSynthesizer` では：
  1. `clone_method="xvector"` 時に `x_vector_only_mode=True` かつ `ref_text=None` を強制する。
  2. `native._icl_cache` が万一汚染された場合は即座に `RuntimeError` で検出する。
  3. キャッシュ識別子（`identity`）に `clone_method`（`"icl"` / `"xvector"`）、`x_vector_only_mode`、`settings`（`temperature`, `top_k`, `top_p`, `repetition_penalty`, `seed`）、`ref_trailing_silence_seconds` を含め、既存の `voice_cloning.py`（自己ハッシュを持つため無変更で保護）や既存ICLキャッシュと完全に分離する。
- また、現行 `main` ブランチの既存モジュール群（`cross_lingual_clone.audit_english_reference`、`cross_lingual_clone.build_phase8_benchmark_script`、`voice_cloning.register_profile`、`voice_cloning.load_profiles`、`tts_feedback.run_feedback(..., backend=..., voices=...)`、`voice_quality` / `voice_profiler.PyannoteBackend`）と直接接続する。

---

## 3. Mac環境での実験実行コマンド（Experiment A / B / C）

### 3.1. パッチ展開と単体・結合テストの実行
```bash
cd ~/Projects/transclip
unzip -o /tmp/transclip-prosody-experiments.zip -d .
.venv-language/bin/python -m unittest discover -s tests -v
```

### 3.2. Preset・実験A（ICL）・実験B（X-vector-only）の一括生成と比較バンドル出力
既存のPreset / ICL生成済み `feedback.json` がある場合は `--existing-preset-feedback` / `--existing-icl-feedback` で指定でき、指定しない場合も同一ディレクトリ内のキャッシュを自動再利用します。

```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python prosody_experiments.py run-all \
  --speaker SYNTH_EN_01 \
  --reference-audio output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.wav \
  --reference-text output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.txt \
  --consent ~/Downloads/consent.json \
  --review ~/Downloads/review.json \
  --preset-voice Ryan \
  --output output/prosody_experiments
```

### 3.3. 実験C（1変数ずつの生成条件スイープ：`temperature=0.7`, `top_p=0.9`, `top_k=25`, `repetition_penalty=1.02`, `seed=42`, `icl_pad040`）も同時に実行する場合
```bash
~/Documents/techclipsjp/Qwen3-TTS-MLX/.venv/bin/python prosody_experiments.py run-all \
  --speaker SYNTH_EN_01 \
  --reference-audio output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.wav \
  --reference-text output/phase8_cross_lingual/reference_audit/audit-SYNTH_EN_01-4a19ef1c00d3/reference.txt \
  --consent ~/Downloads/consent.json \
  --review ~/Downloads/review.json \
  --preset-voice Ryan \
  --include-experiment-c \
  --output output/prosody_experiments
```

### 3.4. Speaker Embedding類似度（pyannote）を付与して `comparison.json` / `comparison.md` / `review.html` を更新する場合
```bash
.venv-diarize/bin/python prosody_experiments.py compare \
  --speaker SYNTH_EN_01 \
  --reference-audio output/prosody_experiments/reference/english_reference.wav \
  --preset-feedback output/prosody_experiments/_runs/preset_run/feedback.json \
  --icl-feedback output/prosody_experiments/_runs/icl_run/feedback.json \
  --xvector-feedback output/prosody_experiments/_runs/xvector_run/feedback.json \
  --embeddings \
  --output output/prosody_experiments

# 試聴比較HTMLをブラウザで開く
open output/prosody_experiments/review.html
```
