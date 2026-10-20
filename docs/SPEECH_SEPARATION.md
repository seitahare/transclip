# Speech Separation / Speaker-Aligned Transcription (Phase 6)

## Scope and invariants

Independent local CLI for selected clips with existing diarization and word ASR. It consumes Phase 4 speaker profiles and Phase 5 frame activity, so it does not replace those stages or download models automatically. No paid API, translation, TTS, cloning, commit or publication occurs in this stage.

Global speaker IDs are an arbitrary set; simultaneous count is calculated per time interval from non-exclusive turns. The two counts are separate. Speakers appearing later are retained. A backend's trained simultaneous capacity is enforced, including surrounding context. Three simultaneous voices are **not** successfully separated by a two-source checkpoint.

All model labels and ASR are reviewable hypotheses. The engine does not claim improved identification or clean separation without listening ground truth. Original JSON, media and WAV files remain untouched. Different-speaker overlap remains representable; no speech is accelerated or cut at the next speaker's onset. Short unknown targets are not resolved solely by a surrounding dominant voice.

## Modules and flow

- `speech_separation.py`: bounded WAV reader, overlap routing, quality measurements, speaker registry, job cache and CLI.
- `separation_backends.py`: `SpeechSeparator` protocol, local Conv-TasNet, configurable local TorchScript adapter, persistent ASR worker controller.
- `speaker_transcription.py`: local faster-whisper worker, word timestamps, source/caption comparison.
- `separation_review.py`: arbitrary-N-stem listening review, separate human decision validation, strict downstream adapter.
- `speaker_review.build_separation_review`: existing Speaker Review entry point for the new schema.

1. Validate selected source WAV (16 kHz, mono PCM16) and relative processing range.
2. Partition by non-exclusive turn boundaries; retain tiny backchannels and split long regions into bounded windows (default maximum 8 seconds including 0.35-second context on each side).
3. Route temporal overlap with frame overlap vote >= 0.35 and frame coverage >= 0.5 to separation. Frame votes come from the same pyannote segmentation model as diarization: they are **not independent evidence or calibrated probabilities**. With no frame activity, retain original audio and report temporal overlap as unresolved.
4. Retain single-speaker audio; do not blindly separate every window. Unknown/music/noise and timing discrepancies remain hypotheses A–F, not definitive classes.
5. Separate supported candidate windows. Check channel count, finite samples, exact duration, missing/duplicate stems and mixture reconstruction. All stems receive one recorded signed least-squares gain, bounded to avoid clipping. This handles the scale/sign ambiguity of the checkpoint while preserving relative stem gains; no per-stem loudness normalization. Raw and scale-aligned residuals are saved. Low residual alone does **not** establish clean separation.
6. Compute features and embeddings on the core, not neighboring context. Reuse Phase 4/5 independent reference selection, excluding target/context plus guard time. Rank all available speaker profiles, regardless of stem index. Candidate match uses cosine >= 0.72 and margin >= 0.12 (uncalibrated). Short cores under 0.8 seconds remain UNKNOWN. Duplicate speaker claims in a simultaneous window become UNKNOWN rather than forcing a permutation.
7. A distant new cluster needs >=1.5-second eligible audio in two non-overlapping windows with cosine >=0.85, and low similarity to existing reference speakers. New IDs are provisional model clusters, never real identities; earlier ambiguous windows remain UNKNOWN. Upstream diarization remains responsible for initial global clustering.
8. ASR original context and stems in the existing main environment. The worker is reused, CPU int8, local-files-only, beam 3, word timestamps, no VAD filtering, no previous-text conditioning. ASR model and package versions/settings are recorded.
9. Keep full word boundaries; midpoint assigns a word to one processing core. Boundary-crossing words remain flagged. Absolute time = selection start + context start + ASR time. Word alignment is still an ASR estimate, not a forced-alignment guarantee. Differences between independently recognized windows can still create boundary omissions or inconsistencies; raw ASR is retained for review.
10. Compare stem words with existing original ASR, new original-window ASR and supplied YouTube captions. Unsupported words, high no-speech probability and repetitive text are warnings. Empty ASR is explicit. No automatic confirmation is allowed.

## Models investigated

| Model | Trained simultaneous voices | Audio / size | Decision |
|---|---:|---|---|
| [Official Conv-TasNet Libri2Mix](https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.html) | 2 | 8 kHz mono, about 19.1 MiB weights | Download approved by user; actual CPU inference tested on Apple Silicon. |
| [SepFormer WSJ02Mix](https://huggingface.co/speechbrain/sepformer-wsj02mix) | 2 | 8 kHz, roughly 113 MB inference weights | Apache-2.0 model card; not installed or tested. Requires additional SpeechBrain dependency. |
| [SepFormer WSJ03Mix](https://huggingface.co/speechbrain/sepformer-wsj03mix) | 3 | 8 kHz, roughly 113 MB inference weights | Not downloaded. Repository also contains ~206 MB optimizer state, unnecessary for inference. |
| [X-SepFormer research](https://arxiv.org/abs/2303.05023) | Target-speaker extraction | Requires reference speaker conditioning | Future candidate; not implemented or tested. |

Conv-TasNet code is [BSD-2-Clause](https://github.com/pytorch/audio/blob/main/LICENSE). The bundle does not separately state a checkpoint-specific grant here; do not equate code licensing with independently verified weight/data rights. The [official pipeline definition](https://github.com/pytorch/audio/blob/main/src/torchaudio/pipelines/_source_separation_pipeline.py) fixes this checkpoint at two sources, 8 kHz. Changing `num_sources` does not retrain it. No claim of verified MPS support; this implementation uses CPU.

`SeparationSpec` records model name/revision, supported simultaneous counts, sample rate, output channels, devices, required memory, license and maximum window. Unknown required memory is `null`, not a fabricated estimate. Measured process high-water RSS and per-job separation/total times are recorded. ASR runs in another process with its own high-water RSS; these maxima should not be added and described as a simultaneous peak.

The optional TorchScript adapter accepts an explicitly supplied local model with `[1,1,T] -> [1,N,T]`. Configuration uses the fields of `SeparationSpec`; sample rate and capacity are checked. This is an extension interface, not evidence of real three-speaker performance. Only trusted local model artifacts should be supplied.

## CLI

From the repository, using the existing separate environments:

```sh
.venv-diarize/bin/python speech_separation.py VIDEO_ID \
  --input-dir input/VIDEO_ID \
  --audio output/VIDEO_ID/speaker_analysis/ANALYSIS/selected_source.wav \
  --profile-dir output/VIDEO_ID/speaker_analysis/ANALYSIS \
  --activity output/VIDEO_ID/speaker_attribution/ATTRIBUTION/frame_activity.json \
  --captions selected_captions.json \
  --output output/VIDEO_ID/speech_separation \
  --start 0 --end 30 \
  --backend convtasnet --weights /absolute/path/conv_tasnet_base_libri2mix.pt \
  --asr-python .venv/bin/python --asr-model /absolute/path/cached-whisper-snapshot
```

`--start/--end` are relative to the selected clip, not the full video. `--dry-run` reports identity/routing without inference or output mutation. `--backend none` records unsupported overlap. `--no-embedding` leaves matches unresolved. Without ASR options, empty/unavailable ASR is explicit. `--backend torchscript --model-config config.json --weights local.pt` selects an alternative local backend. Resumption is automatic by rerunning the same command; no separate resume switch is required.

Input directory: `selection.json`, `diarization.json` (non-exclusive turns), `utterances.json` v3, `word_timestamps.json`. Profiles require matching Phase 4 `speaker_profiles.json` and `diarization.enhanced.json`. These upstream outputs must correspond to the same selected source. No full-video download/extraction or diarization run is performed implicitly.

Optional captions are a JSON list with `text`, `relative_start`, `relative_end`. To reuse `transcript.json`, select full-video segments intersecting the selection, subtract selection start, and bound the resulting relative endpoints to the clip. The caption text itself remains unchanged. Caption timing can be coarse and is not ground truth.

## Outputs and cache

Each invocation creates a fresh `run-...` directory; old runs/audio are retained:

- `speaker_separation.json`: video/clip/range, source WAV hash, arbitrary detected IDs/count, model/provenance, windows, count hypotheses, original/refined time fields (refined null until verified), source utterance IDs, original text, raw ASR, stem features/embeddings/candidates, quality, audio paths/hashes, times/RSS, failures.
- `speaker_transcriptions.json`: speaker-aligned utterances with full words and both timelines, transcription/separation source, source IDs, support comparisons and review reasons. Overlapping rows are valid. Original utterance IDs and new transcript IDs are distinct.
- `review.html`, `separation_review.json`, `review_decisions.template.json`: local listening comparisons and downloadable human decisions.
- `comparison.md`: per-window and per-original-utterance comparison.

Immutable WAVs and job results live under sibling `cache/<content-hash>/attempt-*`. A small pointer publishes a complete attempt after its artifact hashes are saved. Incomplete/failed attempts never become valid cache entries. Damaged audio is retained and regenerated into another attempt. Identity includes source content, selected range, original inputs, profiles/activity/captions, model/settings/ASR versions and implementation hashes. Cache hits verify all artifacts. Changing these inputs invalidates reuse. Models are lazy and reused within a run; cached reruns avoid loading/invoking them.

Long sources are read by WAV seek per bounded window. JSON provenance and metadata still scale with clip length, and completed outputs accumulate on disk. Concurrent invocations may duplicate computation but use private immutable attempts; there is no cross-process model server or strict OS memory cap.

## Human review and downstream handoff

Open `review.html` locally. Each window has original audio and any number of stems, texts, candidates, similarities, features and overlap reasons. Save choices: speaker ID, MIXED, UNKNOWN, ASR_ERROR, SEPARATION_FAILED, OTHER. Decisions have a review fingerprint, audio hash and reviewer identity and are saved separately. Notes hold correction candidates; they are not silently applied to word timestamps.

```sh
.venv-diarize/bin/python separation_review.py /absolute/path/run-directory \
  --decisions /absolute/path/separation_review_decisions.json \
  --output output/VIDEO_ID/reviewed_separation
```

The adapter requires all stems/text reviewed and explicit confirmation that original speech is accounted for. Failed/unsupported windows, unresolved error choices, correction notes, missing source coverage and a processing boundary cutting an original utterance block export. A human can explicitly preserve UNKNOWN (exported as null). An empty-ASR stem can only be accepted through explicit listening/text and coverage confirmation. Edits to words/timing require corrected evidence and another review; this MVP does not provide a word editor.

Successful export creates a new v3 `utterances.json`, `selection.json`, separate human decisions, and `speaker_analysis.json` compatible with Audio Mixer's `--speaker-analysis`. New transcript rows receive IDs above the original maximum; outside-range original utterances are retained. Same-speaker time collisions are reported and different-speaker overlaps remain untouched.

Point the existing utterance translator at this new input directory only when translation execution is authorized; then rebuild the dubbing script and feedback. No stale translations are synthesized here. New source IDs prevent old feedback matching the reviewed analysis. TTS already keys storage by speaker and voice/text/model settings; tests verify changed speaker/voice creates separate WAVs. Pass the exported `speaker_analysis.json` to Audio Mixer to enforce assignment consistency. Do not mix old scripts/feedback with new reviewed transcripts. Speaker profiles and presets remain separate from clone consent.

## Verification and limitations

Baseline 153 tests passed before changes. Phase 6 adds 40 synthetic/mock tests; all 193 passed. Coverage includes 1/2/3 speakers, late new speakers, 3 concurrent sources, backend capacity, short interjections, permutation, UNKNOWN, missing/leaking/invalid audio, word timing, hallucination warnings, human priority, immutable cache/retry, one-hour chunking, v3 translation compatibility, stale TTS and mixer evidence.

Actual first-30-second validation uses the selected source of `XuoqKYxDHVc` (1653.60–1942.99), but neither video ID nor focus text/times are constants in the new engine. See `PHASE6_VALIDATION.md` for measured results and playback paths. Synthetic routing/identity tests have known fixture truth; the real clip has no human ground truth. Waveform creation, reconstruction residual and matching ASR text are not clean-separation or speaker-accuracy scores.

Known limitations: short targets lack reliable embeddings; reference quality and domain mismatch limit matches; caption/ASR timing can disagree; unseen overlap without upstream evidence may be missed; three-or-more concurrent voices need a compatible trained backend; generated transcripts can omit or hallucinate words; UI is statically checked but browser visual operation was not revalidated in this environment. No Japanese audio or MP4 is regenerated from unreviewed Phase 6 output.
