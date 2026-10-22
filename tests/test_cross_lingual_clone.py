from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

import audio_mixer
from cross_lingual_clone import (
    CloneBlockedForReviewError,
    PHASE8_BENCHMARK_SENTENCES,
    audit_english_reference,
    build_phase8_benchmark_script,
    discover_reference_by_sha256,
    inspect_waveform_boundaries,
    main,
    run_phase8_pipeline,
)
from language.cache import atomic_json
from synthetic_reference import generate_reference
from voice_cloning import digest, text_digest


def write_shaped_wav(path, *, seconds=12.48, rate=24000, freq=135.0, fade_edges=True):
    frames = round(seconds * rate)
    t = np.arange(frames, dtype=np.float64) / rate
    env = np.ones(frames, dtype=np.float64) * 0.25
    if fade_edges:
        edge = min(frames // 6, round(0.12 * rate))
        env[:edge] = np.linspace(0.0, 0.25, edge)
        env[-edge:] = np.linspace(0.25, 0.0, edge)
    signal = env * np.sin(2.0 * math.pi * freq * t)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())


class EnglishDesignMock:
    def __init__(self, *, language="English", seed=24681, duration=12.48):
        self.identity = {
            "backend": "mock-design",
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            "settings": {"seed": seed, "language": language},
        }
        self.duration = duration
        self.calls = 0

    def synthesize(self, *, text, profile, output_path):
        self.calls += 1
        write_shaped_wav(output_path, seconds=self.duration, rate=24000, freq=125.0)
        return {"sample_rate": 24000, "sample_frames": round(self.duration * 24000), "termination": "eos_before_limit"}


class PresetSynthMock:
    supported_voice_modes = {"preset"}

    def __init__(self, *, seed=0):
        self.identity = {
            "backend": "mock-preset",
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
            "model_revision": "rev-preset-1",
            "settings": {"language": "Japanese", "seed": seed},
        }
        self.calls = []

    def synthesize(self, *, text, profile, output_path):
        self.calls.append((text, profile.voice))
        write_shaped_wav(output_path, seconds=3.2, rate=24000, freq=150.0)
        return {"sample_rate": 24000, "sample_frames": round(3.2 * 24000), "processing_seconds": 1.25, "termination": "eos_before_limit"}


class CloneSynthMock:
    supported_voice_modes = {"approved_clone"}

    def __init__(self, *, seed=0):
        self.identity = {
            "backend": "mock-base",
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            "model_revision": "e7dd0585652209fa0d7783659aad4e8a324de11c",
            "settings": {"language": "Japanese", "seed": seed},
        }
        self.calls = []

    def synthesize(self, *, text, profile, output_path):
        reused = len(self.calls) > 0
        self.calls.append((text, profile.speaker, profile.reference_audio_sha256))
        write_shaped_wav(output_path, seconds=2.8, rate=24000, freq=128.0)
        return {
            "sample_rate": 24000,
            "sample_frames": round(2.8 * 24000),
            "processing_seconds": 1.4 if not reused else 0.75,
            "conditioning_reused": reused,
            "termination": "eos_before_limit",
            "review_required": True,
        }


class MockEmbedder:
    identity = {"model": "mock-pyannote-embedding"}

    def embed(self, x):
        mean_abs = float(np.mean(np.abs(x)))
        return [1.0, mean_abs, 0.2]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class CrossLingualCloneTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.speaker = "SYNTH_EN"
        self.en_text = (
            "Hello, this is a fictional English reference voice generated for cross-lingual "
            "dubbing evaluation. Every word and boundary must be verified by a human listener."
        )
        self.en_instruction = "A calm, clear fictional English narrator voice; no real person imitation."
        self.gen_path = generate_reference(
            EnglishDesignMock(language="English", seed=24681, duration=12.48),
            speaker=self.speaker,
            text=self.en_text,
            instruction=self.en_instruction,
            output=self.root / "synthetic_refs",
            fictional=True,
        )
        self.ref_dir = self.gen_path.parent
        self.ref_wav = self.ref_dir / "reference.wav"
        self.ref_txt = self.ref_dir / "reference.txt"
        self.ref_sha = digest(self.ref_wav)

    def make_verified_records(self, *, audio_sha=None, text_sha=None, authorized=True, transcript_verified=True, quality_verified=True):
        a_sha = audio_sha or self.ref_sha
        t_sha = text_sha or text_digest(self.en_text)
        consent_path = self.root / "consent.json"
        review_path = self.root / "review.json"
        atomic_json(
            consent_path,
            {
                "authorized": authorized,
                "scope": "voice_cloning",
                "speaker_id": self.speaker,
                "reference_audio_sha256": a_sha,
                "evidence": "Fictional English VoiceDesign reference (seed 24681) reuse authorized for testing.",
                "verified_by": "test_reviewer",
            },
        )
        atomic_json(
            review_path,
            {
                "transcript": {
                    "status": "verified" if transcript_verified else "pending",
                    "verified_by": "test_reviewer" if transcript_verified else "",
                    "reference_audio_sha256": a_sha,
                    "reference_text_sha256": t_sha,
                },
                "quality": {
                    "reference_audio_sha256": a_sha,
                    "verified_by": "test_reviewer" if quality_verified else "",
                    "speaker_identity_verified": quality_verified,
                    "single_speaker_verified": quality_verified,
                    "intelligibility_verified": quality_verified,
                    "boundaries_verified": quality_verified,
                    "known_overlap_seconds": 0 if quality_verified else None,
                    "unresolved_issues": [] if quality_verified else ["not_reviewed"],
                    "source_kind": "original",
                },
            },
        )
        return consent_path, review_path

    def test_audit_separates_mechanical_and_human_checks_and_never_auto_approves(self):
        consent_path, review_path = self.make_verified_records(
            authorized=True, transcript_verified=False, quality_verified=False
        )
        report_path, report = audit_english_reference(
            speaker=self.speaker,
            reference_audio=self.ref_wav,
            reference_text_path=self.ref_txt,
            output_dir=self.root / "audit",
            generation_record_path=self.gen_path,
            consent_path=consent_path,
            review_path=review_path,
            expected_sha256=self.ref_sha,
            expected_duration=12.48,
            expected_seed=24681,
        )
        self.assertTrue(report["mechanical_verification"]["passed"])
        self.assertAlmostEqual(report["mechanical_verification"]["wav_measurement"]["audio_duration"], 12.48, places=2)
        self.assertTrue(report["human_verification"]["checks"]["reuse_permission_authorized"])
        self.assertFalse(report["human_verification"]["checks"]["transcript_matches_speech"])
        self.assertFalse(report["ready_for_clone_synthesis"])
        self.assertEqual(report["status"], "blocked_pending_human_review")
        self.assertIn("transcript_matches_speech", report["pending_human_items"])
        self.assertTrue((report_path.parent / "english_reference_review.html").is_file())

    def test_japanese_reference_or_review_cannot_substitute_for_english(self):
        ja_gen = generate_reference(
            EnglishDesignMock(language="Japanese", seed=24680, duration=12.32),
            speaker=self.speaker,
            text="これは日本語の架空参考音声です。",
            instruction="Calm Japanese voice.",
            output=self.root / "ja_refs",
            fictional=True,
        )
        ja_wav = ja_gen.parent / "reference.wav"
        ja_txt = ja_gen.parent / "reference.txt"
        ja_consent, ja_review = self.make_verified_records(
            audio_sha=digest(ja_wav), text_sha=text_digest(ja_txt.read_text(encoding="utf-8"))
        )
        _, rep_lang = audit_english_reference(
            speaker=self.speaker,
            reference_audio=ja_wav,
            reference_text_path=ja_txt,
            output_dir=self.root / "audit_ja",
            generation_record_path=ja_gen,
            consent_path=ja_consent,
            review_path=ja_review,
            expected_sha256=digest(ja_wav),
            expected_duration=12.32,
            expected_seed=24680,
            expected_language="English",
        )
        self.assertFalse(rep_lang["mechanical_verification"]["passed"])
        self.assertFalse(rep_lang["ready_for_clone_synthesis"])
        self.assertIn("generation_language_mismatch:Japanese", rep_lang["mechanical_verification"]["issues"])

        _, rep_sub = audit_english_reference(
            speaker=self.speaker,
            reference_audio=self.ref_wav,
            reference_text_path=self.ref_txt,
            output_dir=self.root / "audit_sub",
            generation_record_path=self.gen_path,
            consent_path=ja_consent,
            review_path=ja_review,
            expected_sha256=self.ref_sha,
            expected_duration=12.48,
            expected_seed=24681,
        )
        self.assertFalse(rep_sub["human_verification"]["checks"]["not_substituted_from_other_reference"])
        self.assertFalse(rep_sub["ready_for_clone_synthesis"])

    def test_boundary_truncation_diagnostics_detects_abrupt_edges(self):
        clean_bounds = inspect_waveform_boundaries(self.ref_wav)
        self.assertEqual(clean_bounds["mechanical_boundary_warnings"], [])
        abrupt_wav = self.root / "abrupt.wav"
        write_shaped_wav(abrupt_wav, seconds=3.0, rate=24000, fade_edges=False)
        abrupt_bounds = inspect_waveform_boundaries(abrupt_wav)
        self.assertIn("leading_edge_high_energy_possible_onset_cut", abrupt_bounds["mechanical_boundary_warnings"])
        self.assertIn("trailing_edge_high_energy_possible_tail_cut", abrupt_bounds["mechanical_boundary_warnings"])

    def test_discover_reference_by_sha256(self):
        found = discover_reference_by_sha256([self.root / "nonexistent", self.root / "synthetic_refs"], self.ref_sha)
        self.assertEqual(found, self.ref_wav.resolve())
        self.assertIsNone(discover_reference_by_sha256([self.root], "0" * 64))

    def test_phase8_benchmark_script_matches_specification(self):
        script = build_phase8_benchmark_script(self.speaker)
        self.assertEqual(len(script["entries"]), 3)
        self.assertEqual(
            [e["text_ja"] for e in script["entries"]],
            [s["text_ja"] for s in PHASE8_BENCHMARK_SENTENCES],
        )
        self.assertIn("ヒャクヨンジュウヨンチョウカイ", script["entries"][1]["japanese_analysis"]["reading"])

    def test_unverified_reference_blocks_pipeline_and_cli(self):
        consent_path, review_path = self.make_verified_records(
            authorized=True, transcript_verified=False, quality_verified=False
        )
        preset_mock = PresetSynthMock()
        clone_mock = CloneSynthMock()
        with self.assertRaises(CloneBlockedForReviewError):
            run_phase8_pipeline(
                speaker=self.speaker,
                reference_audio=self.ref_wav,
                reference_text_path=self.ref_txt,
                output_dir=self.root / "blocked_run",
                generation_record_path=self.gen_path,
                consent_path=consent_path,
                review_path=review_path,
                preset_backend=preset_mock,
                clone_backend=clone_mock,
                expected_sha256=self.ref_sha,
                expected_duration=12.48,
                expected_seed=24681,
            )
        self.assertEqual(len(preset_mock.calls), 0)
        self.assertEqual(len(clone_mock.calls), 0)

        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            code = main([
                "run-benchmark",
                "--speaker", self.speaker,
                "--reference-audio", str(self.ref_wav),
                "--reference-text", str(self.ref_txt),
                "--generation-record", str(self.gen_path),
                "--consent", str(consent_path),
                "--review", str(review_path),
                "--expected-sha256", self.ref_sha,
                "--output", str(self.root / "cli_blocked"),
            ])
        self.assertEqual(code, 2)
        self.assertIn("blocked_pending_human_review", err.getvalue())

    def test_end_to_end_phase8_pipeline_cache_reuse_and_mixer_compatibility(self):
        consent_path, review_path = self.make_verified_records(
            authorized=True, transcript_verified=True, quality_verified=True
        )
        preset_mock = PresetSynthMock()
        clone_mock = CloneSynthMock()
        out_dir = self.root / "phase8_out"
        with redirect_stdout(io.StringIO()):
            result = run_phase8_pipeline(
                speaker=self.speaker,
                reference_audio=self.ref_wav,
                reference_text_path=self.ref_txt,
                output_dir=out_dir,
                generation_record_path=self.gen_path,
                consent_path=consent_path,
                review_path=review_path,
                preset_backend=preset_mock,
                clone_backend=clone_mock,
                embedder=MockEmbedder(),
                expected_sha256=self.ref_sha,
                expected_duration=12.48,
                expected_seed=24681,
            )
        self.assertEqual(len(preset_mock.calls), 3)
        self.assertEqual(len(clone_mock.calls), 3)
        self.assertTrue(Path(result["evaluation_json"]).is_file())
        self.assertTrue(Path(result["evaluation_markdown"]).is_file())
        self.assertTrue(Path(result["evaluation_html"]).is_file())

        summary = result["summary"]
        self.assertEqual(len(summary["utterances"]), 3)
        self.assertFalse(summary["perceptual_quality_approved"])
        self.assertFalse(summary["utterances"][0]["clone"]["conditioning_reused"])
        self.assertTrue(summary["utterances"][1]["clone"]["conditioning_reused"])
        self.assertTrue(summary["utterances"][2]["clone"]["conditioning_reused"])
        self.assertIsNotNone(summary["utterances"][0]["clone"]["pitch"]["f0_hz_median"])
        self.assertIsNotNone(summary["utterances"][0]["clone"]["reference_embedding_cosine"])

        preset_mock_2 = PresetSynthMock()
        clone_mock_2 = CloneSynthMock()
        with redirect_stdout(io.StringIO()):
            second = run_phase8_pipeline(
                speaker=self.speaker,
                reference_audio=self.ref_wav,
                reference_text_path=self.ref_txt,
                output_dir=out_dir,
                generation_record_path=self.gen_path,
                consent_path=consent_path,
                review_path=review_path,
                preset_backend=preset_mock_2,
                clone_backend=clone_mock_2,
                embedder=MockEmbedder(),
                expected_sha256=self.ref_sha,
                expected_duration=12.48,
                expected_seed=24681,
            )
        self.assertEqual(len(preset_mock_2.calls), 0)
        self.assertEqual(len(clone_mock_2.calls), 0)
        self.assertTrue(all(u["preset"]["cache_reused"] for u in second["summary"]["utterances"]))
        self.assertTrue(all(u["clone"]["cache_reused"] for u in second["summary"]["utterances"]))

        selection_path = self.root / "selection.json"
        atomic_json(selection_path, {"start": 0.0, "end": 30.0})
        info_path = self.root / "source.info.json"
        atomic_json(info_path, {"id": "phase8_cross_lingual", "duration": 30.0})
        video_path = self.root / "source.mp4"
        video_path.write_bytes(b"fixture")
        with patch("audio_mixer.probe", return_value={"format": {"duration": "30.0"}, "streams": [{"codec_type": "video"}]}):
            plan = audio_mixer.build_plan(
                result["clone_feedback"],
                selection_path,
                video_path,
                info_path,
                "phase8_cross_lingual",
                seconds=30.0,
                diagnostic=True,
            )
        self.assertEqual(plan["same_speaker_conflicts"], [])
        self.assertEqual(plan["blockers"], [])
        self.assertEqual(len(plan["placements"]), 3)


if __name__ == "__main__":
    unittest.main()
