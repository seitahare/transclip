"""Phase 8: Cross-lingual Voice Cloning (English reference -> Japanese speech) audit, execution, and evaluation CLI.

Separates mechanical waveform/hash/provenance checks from human listening verification,
blocks unverified reference audio from synthesis, reuses verified Preset caches when
conditions match, and generates portable cross-lingual listening comparison reports.
"""
import argparse
import copy
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote
import wave

import numpy as np

from language.cache import atomic_json, fingerprint
from language.pronunciation import PronunciationDictionary, count_mora, PAUSES
from language.timing import TimingConfig, TimingEngine, measure_wav
from synthetic_reference import validate_generation_record
from tts_feedback import run_feedback, validate_script
from voice_cloning import (
    DEFAULT_BASE,
    UNKNOWN,
    compare_feedback,
    digest,
    load_profiles,
    prepare_review_template,
    profile_snapshot,
    register_profile,
    text_digest,
)
from voice_profiler import acoustic_features
from voice_quality import compare_voices

PHASE7_ENGLISH_REFERENCE = {
    "duration_seconds": 12.48,
    "seed": 24681,
    "sha256": "4a19ef1c00d3ee87d3afbb707f59166f830fe6a99c196a2df7d7fecdd7f73696",
    "language": "English",
}

PHASE8_BENCHMARK_SENTENCES = [
    {
        "id": "dub-001",
        "source_utterance_ids": [1],
        "relative_start": 0.0,
        "relative_end": 5.5,
        "text_en": "New technology is gradually changing the way we live.",
        "text_ja": "新しい技術が、私たちの暮らしを少しずつ変えています。",
    },
    {
        "id": "dub-002",
        "source_utterance_ids": [2],
        "relative_start": 6.5,
        "relative_end": 12.5,
        "text_en": "This computer performs 144 trillion calculations per second.",
        "text_ja": "このコンピューターは、1秒間に144兆回の計算を行います。",
    },
    {
        "id": "dub-003",
        "source_utterance_ids": [3],
        "relative_start": 13.5,
        "relative_end": 18.5,
        "text_en": "So what should we verify in the next experiment?",
        "text_ja": "それでは、次の実験で何を確かめればよいのでしょうか。",
    },
]

BENCHMARK_READINGS = {
    "新しい": "アタラシイ",
    "技術": "ギジュツ",
    "私たち": "ワタシタチ",
    "暮らし": "クラシ",
    "少しずつ": "スコシズツ",
    "変えて": "カエテ",
    "秒間": "ビョウカン",
    "兆回": "チョウカイ",
    "計算": "ケイサン",
    "行います": "オコナイマス",
    "次の": "ツギノ",
    "実験": "ジッケン",
    "何を": "ナニヲ",
    "確かめれば": "タシカメレバ",
}


class CloneBlockedForReviewError(RuntimeError):
    """Raised when human listening verification is incomplete for a reference voice."""

    def __init__(self, message, audit_report_path=None, review_html_path=None):
        super().__init__(message)
        self.audit_report_path = audit_report_path
        self.review_html_path = review_html_path


def _read_pcm16_mono_normalized(path):
    """Read mono PCM16 WAV into float64 [-1, 1] array and sample rate."""
    measured = measure_wav(path)
    if measured["channels"] != 1 or measured["sample_width_bytes"] != 2:
        raise ValueError("Reference must be mono 16-bit PCM WAV for boundary inspection.")
    with wave.open(str(path), "rb") as stream:
        raw = stream.readframes(measured["sample_frames"])
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return samples, measured["sample_rate"], measured


def inspect_waveform_boundaries(path, *, edge_window_seconds=0.05):
    """Measure leading/trailing edge energy as a mechanical proxy for word boundary cutoffs.

    Mechanical indicators never replace human listening verification (`boundaries_verified`).
    """
    samples, rate, measured = _read_pcm16_mono_normalized(path)
    edge_frames = max(1, min(len(samples) // 4, round(edge_window_seconds * rate)))
    leading = samples[:edge_frames]
    trailing = samples[-edge_frames:]
    leading_rms = float(np.sqrt(np.mean(leading * leading)))
    trailing_rms = float(np.sqrt(np.mean(trailing * trailing)))
    leading_peak = float(np.max(np.abs(leading)))
    trailing_peak = float(np.max(np.abs(trailing)))
    overall_rms = max(measured["rms_amplitude"], 1e-8)
    leading_ratio = leading_rms / overall_rms
    trailing_ratio = trailing_rms / overall_rms
    warnings = []
    if leading_ratio > 0.65 and leading_peak > 0.15:
        warnings.append("leading_edge_high_energy_possible_onset_cut")
    if trailing_ratio > 0.65 and trailing_peak > 0.15:
        warnings.append("trailing_edge_high_energy_possible_tail_cut")
    return {
        "edge_window_seconds": edge_frames / rate,
        "leading_rms": round(leading_rms, 6),
        "trailing_rms": round(trailing_rms, 6),
        "leading_peak": round(leading_peak, 6),
        "trailing_peak": round(trailing_peak, 6),
        "leading_to_overall_rms_ratio": round(leading_ratio, 4),
        "trailing_to_overall_rms_ratio": round(trailing_ratio, 4),
        "mechanical_boundary_warnings": warnings,
        "human_listening_still_required": True,
    }


def discover_reference_by_sha256(search_roots, target_sha256=PHASE7_ENGLISH_REFERENCE["sha256"]):
    """Locate a reference WAV matching target_sha256 across candidate directories."""
    target_sha256 = target_sha256.lower()
    for root in search_roots:
        base = Path(root).expanduser()
        if not base.exists():
            continue
        for wav_path in sorted(base.rglob("*.wav")):
            if not wav_path.is_file():
                continue
            try:
                if digest(wav_path).lower() == target_sha256:
                    return wav_path.resolve()
            except OSError:
                continue
    return None


def _render_english_reference_review_html(report, audio_filename):
    esc = html.escape
    mech = report["mechanical_verification"]
    human = report["human_verification"]
    meas = mech.get("wav_measurement") or {}
    bounds = mech.get("boundary_diagnostics") or {}
    dsp = (mech.get("acoustic_features") or {}).get("dsp_estimates") or {}
    gen = mech.get("generation_provenance") or {}
    pending_list = "".join(f"<li><code>{esc(item)}</code></li>" for item in report["pending_human_items"]) or "<li>なし（すべて確認済み）</li>"
    mech_issues = "".join(f"<li><code>{esc(item)}</code></li>" for item in mech["issues"]) or "<li>なし（機械検査すべて合格）</li>"

    template_consent = {
        "authorized": True if human["checks"]["reuse_permission_authorized"] else False,
        "scope": "voice_cloning",
        "speaker_id": report["speaker_id"],
        "reference_audio_sha256": report["reference_audio_sha256"],
        "evidence": (human.get("consent_record") or {}).get("evidence", ""),
        "verified_by": (human.get("consent_record") or {}).get("verified_by", ""),
    }
    template_review = {
        "transcript": {
            "status": "verified" if human["checks"]["transcript_matches_speech"] else "pending",
            "verified_by": ((human.get("review_record") or {}).get("transcript") or {}).get("verified_by", ""),
            "reference_audio_sha256": report["reference_audio_sha256"],
            "reference_text_sha256": report["reference_text_sha256"],
        },
        "quality": {
            "reference_audio_sha256": report["reference_audio_sha256"],
            "verified_by": ((human.get("review_record") or {}).get("quality") or {}).get("verified_by", ""),
            "speaker_identity_verified": bool(human["checks"]["speaker_identity_verified"]),
            "single_speaker_verified": bool(human["checks"]["single_speaker_no_other_voice"]),
            "intelligibility_verified": bool(human["checks"]["intelligibility_and_noise_ok"]),
            "boundaries_verified": bool(human["checks"]["word_boundaries_not_cut"]),
            "known_overlap_seconds": 0 if human["checks"]["single_speaker_no_other_voice"] else None,
            "unresolved_issues": [] if human["all_verified"] else ["pending_human_listening_review"],
            "source_kind": "original",
        },
        "measured": meas,
    }

    peak_s = f"{meas['peak_amplitude']:.4f}" if isinstance(meas.get("peak_amplitude"), (int, float)) else "—"
    rms_s = f"{meas['rms_amplitude']:.4f}" if isinstance(meas.get("rms_amplitude"), (int, float)) else "—"
    clip_s = f"{meas['clipped_sample_fraction']:.6f}" if isinstance(meas.get("clipped_sample_fraction"), (int, float)) else "—"

    return f"""<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<title>Phase 8 English Reference Audio Audit &amp; Human Review</title>
<style>
  body {{ font: 15px/1.65 system-ui, -apple-system, sans-serif; max-width: 960px; margin: 28px auto; padding: 0 20px; color: #1f2937; background: #f9fafb; }}
  h1, h2, h3 {{ color: #111827; }}
  .card {{ background: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; padding: 18px 20px; margin-bottom: 18px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }}
  .ok {{ background: #dcfce7; color: #166534; }}
  .warn {{ background: #fef3c7; color: #92400e; }}
  .block {{ background: #fee2e2; color: #991b1b; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
  th {{ background: #f3f4f6; width: 34%; }}
  audio {{ width: 100%; margin: 10px 0; }}
  code, pre {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 13px; }}
  pre {{ padding: 12px; overflow-x: auto; }}
  label {{ display: block; margin: 8px 0; }}
  input[type="text"] {{ width: 100%; max-width: 480px; padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 4px; }}
  button {{ background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 9px 16px; font-size: 14px; cursor: pointer; margin-right: 10px; margin-top: 10px; }}
  button:hover {{ background: #1d4ed8; }}
</style>
<h1>Phase 8 英語参考音声 検査・試聴確認シート</h1>
<div class="card">
  <p>
    <strong>ステータス:</strong>
    <span class="badge {'ok' if report['ready_for_clone_synthesis'] else 'warn'}">{esc(report['status'])}</span>
    &nbsp;|&nbsp; <strong>話者ID:</strong> <code>{esc(report['speaker_id'])}</code>
    &nbsp;|&nbsp; <strong>言語:</strong> <code>{esc(str(gen.get('language') or report.get('expected_language') or 'English'))}</code>
  </p>
  <p>機械的に確認可能な項目と、人間の試聴が必要な項目を分離しています。人間の試聴確認が完了するまで、この英語参考音声を用いたVoice Clone生成は保留されます（日本語参考音声のレビューで代用することはできません）。</p>
  <h3>1. 英語参考WAVの試聴</h3>
  <audio controls preload="metadata" src="{esc(quote(audio_filename))}"></audio>
  <p>macOSターミナルでの再生コマンド: <code>afplay '{esc(report['reference_audio_path'])}'</code></p>
  <h3>2. 照合対象の英語テキスト (reference.txt)</h3>
  <pre>{esc(report['reference_text'])}</pre>
</div>

<div class="card">
  <h2>機械的検査結果 (Mechanical Verification)</h2>
  <p>判定: <span class="badge {'ok' if mech['passed'] else 'block'}">{'PASSED' if mech['passed'] else 'FAILED'}</span></p>
  <table>
    <tr><th>WAV SHA-256</th><td><code>{esc(report['reference_audio_sha256'])}</code></td></tr>
    <tr><th>期待SHA-256との一致</th><td><code>{esc(str(mech['checks']['sha256_matches_expected']))}</code></td></tr>
    <tr><th>テキスト SHA-256</th><td><code>{esc(report['reference_text_sha256'])}</code></td></tr>
    <tr><th>長さ / サンプルレート / チャンネル</th><td>{meas.get('audio_duration', '—')} 秒 ({meas.get('sample_frames', '—')} frames @ {meas.get('sample_rate', '—')} Hz, {meas.get('channels', '—')} ch)</td></tr>
    <tr><th>ピーク振幅 / RMS振幅 / クリッピング率</th><td>peak={peak_s} / rms={rms_s} / clipped={clip_s}</td></tr>
    <tr><th>語頭・語尾50ms RMS比 (切断指標)</th><td>語頭比={bounds.get('leading_to_overall_rms_ratio', '—')} / 語尾比={bounds.get('trailing_to_overall_rms_ratio', '—')} (警告: {esc(', '.join(bounds.get('mechanical_boundary_warnings', [])) or 'なし')})</td></tr>
    <tr><th>F0中央値 / 変化幅 (P10–P90, std)</th><td>中央値={dsp.get('f0_hz_median', '—')} Hz / P10={dsp.get('f0_hz_p10', '—')} Hz / P90={dsp.get('f0_hz_p90', '—')} Hz / std={dsp.get('f0_hz_std', '—')} Hz</td></tr>
    <tr><th>合成生成レコード (seed / 言語)</th><td>seed={esc(str(gen.get('seed', '—')))} / language={esc(str(gen.get('language', '—')))} / record_valid={esc(str(gen.get('record_valid', '—')))}</td></tr>
  </table>
  <p><strong>機械的検査の指摘事項:</strong></p>
  <ul>{mech_issues}</ul>
</div>

<div class="card">
  <h2>人間による試聴確認項目 (Human Verification Required)</h2>
  <p>未確認項目:</p>
  <ul>{pending_list}</ul>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0;">
  <h3>試聴後の確認レコード作成（ローカル保存用）</h3>
  <p>上記のWAVを実際に試聴し、問題がないことを確認した項目にチェックを入れて <code>consent.json</code> と <code>review.json</code> をダウンロードしてください。</p>
  <label>確認者名 (verified_by): <input id="verified_by" type="text" value="{esc(template_consent['verified_by'])}" placeholder="例: doiseita"></label>
  <label>利用許可の根拠 (consent evidence): <input id="evidence" type="text" value="{esc(template_consent['evidence'])}" placeholder="例: Fictional VoiceDesign English reference (seed 24681) authorized by owner"></label>
  <label><input id="chk_consent" type="checkbox" {'checked' if human['checks']['reuse_permission_authorized'] else ''}> 再利用許可の対象とSHA-256ハッシュが一致し、Voice Cloningへの使用を承認する (<code>authorized: true</code>)</label>
  <label><input id="chk_transcript" type="checkbox" {'checked' if human['checks']['transcript_matches_speech'] else ''}> 英語の発声内容と上記の文字起こしテキストが完全に一致している (<code>transcript.status: "verified"</code>)</label>
  <label><input id="chk_single" type="checkbox" {'checked' if human['checks']['single_speaker_no_other_voice'] else ''}> 他人の声や重なりが一切混ざっていない (<code>single_speaker_verified: true, known_overlap_seconds: 0</code>)</label>
  <label><input id="chk_identity" type="checkbox" {'checked' if human['checks']['speaker_identity_verified'] else ''}> 対象話者（架空英語話者）の声として一貫している (<code>speaker_identity_verified: true</code>)</label>
  <label><input id="chk_boundaries" type="checkbox" {'checked' if human['checks']['word_boundaries_not_cut'] else ''}> 語頭・語尾が途中で切断されていない (<code>boundaries_verified: true</code>)</label>
  <label><input id="chk_intelligibility" type="checkbox" {'checked' if human['checks']['intelligibility_and_noise_ok'] else ''}> 明瞭であり、不自然なノイズや音割れがない (<code>intelligibility_verified: true</code>)</label>
  <button type="button" onclick="downloadConsent()">consent.json をダウンロード</button>
  <button type="button" onclick="downloadReview()">review.json をダウンロード</button>
</div>
<script>
const baseConsent = {json.dumps(template_consent, ensure_ascii=False)};
const baseReview = {json.dumps(template_review, ensure_ascii=False)};
function saveJson(filename, obj) {{
  const blob = new Blob([JSON.stringify(obj, null, 2) + "\\n"], {{type: "application/json"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}}
function downloadConsent() {{
  const verifier = document.getElementById("verified_by").value.trim();
  const evidence = document.getElementById("evidence").value.trim();
  const c = Object.assign({{}}, baseConsent, {{
    authorized: document.getElementById("chk_consent").checked,
    verified_by: verifier,
    evidence: evidence
  }});
  saveJson("consent.json", c);
}}
function downloadReview() {{
  const verifier = document.getElementById("verified_by").value.trim();
  const single = document.getElementById("chk_single").checked;
  const idOk = document.getElementById("chk_identity").checked;
  const intell = document.getElementById("chk_intelligibility").checked;
  const bound = document.getElementById("chk_boundaries").checked;
  const txOk = document.getElementById("chk_transcript").checked;
  const r = JSON.parse(JSON.stringify(baseReview));
  r.transcript.status = txOk ? "verified" : "pending";
  r.transcript.verified_by = verifier;
  r.quality.verified_by = verifier;
  r.quality.single_speaker_verified = single;
  r.quality.speaker_identity_verified = idOk;
  r.quality.intelligibility_verified = intell;
  r.quality.boundaries_verified = bound;
  r.quality.known_overlap_seconds = single ? 0 : null;
  r.quality.unresolved_issues = (txOk && single && idOk && intell && bound) ? [] : ["pending_human_listening_review"];
  saveJson("review.json", r);
}}
</script>
</html>
"""


def audit_english_reference(
    *,
    speaker,
    reference_audio,
    reference_text_path,
    output_dir,
    generation_record_path=None,
    consent_path=None,
    review_path=None,
    expected_sha256=None,
    expected_duration=None,
    expected_seed=None,
    expected_language="English",
):
    """Run mechanical verification and inspect human review records without auto-approving."""
    if speaker in UNKNOWN or not isinstance(speaker, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", speaker):
        raise ValueError("Explicit safe speaker ID is required for reference audit.")
    audio_path = Path(reference_audio).resolve()
    text_path = Path(reference_text_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    if not text_path.is_file():
        raise FileNotFoundError(f"Reference transcript not found: {text_path}")

    text = text_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Reference transcript is empty.")
    audio_sha256 = digest(audio_path)
    text_sha256 = text_digest(text)

    mech_checks = {
        "wav_valid": False,
        "mono_pcm16": False,
        "duration_in_allowed_range": False,
        "duration_matches_expected": True,
        "sha256_matches_expected": True,
        "clipping_within_limit": False,
        "rms_level_sufficient": False,
        "generation_record_valid": True,
        "language_is_english": True,
        "seed_matches_expected": True,
    }
    mech_issues = []
    measured = None
    boundaries = None
    features = None

    try:
        samples, rate, measured = _read_pcm16_mono_normalized(audio_path)
        mech_checks["wav_valid"] = True
        mech_checks["mono_pcm16"] = measured["channels"] == 1 and measured["sample_width_bytes"] == 2
        mech_checks["duration_in_allowed_range"] = 2.5 <= measured["audio_duration"] <= 30.0
        if not mech_checks["duration_in_allowed_range"]:
            mech_issues.append("duration_outside_2.5_to_30_seconds")
        mech_checks["clipping_within_limit"] = measured["clipped_sample_fraction"] <= 0.001
        if not mech_checks["clipping_within_limit"]:
            mech_issues.append("clipping_detected")
        mech_checks["rms_level_sufficient"] = measured["rms_amplitude"] >= 0.005
        if not mech_checks["rms_level_sufficient"]:
            mech_issues.append("insufficient_rms_volume")
        boundaries = inspect_waveform_boundaries(audio_path)
        if rate >= 4000:
            features = acoustic_features(samples, rate)
            features.pop("asr_estimates", None)
    except Exception as exc:
        mech_issues.append(f"wav_inspection_failed:{exc}")

    if expected_sha256 is not None:
        mech_checks["sha256_matches_expected"] = audio_sha256.lower() == expected_sha256.lower()
        if not mech_checks["sha256_matches_expected"]:
            mech_issues.append("reference_audio_sha256_mismatch")

    if expected_duration is not None and measured is not None:
        mech_checks["duration_matches_expected"] = abs(measured["audio_duration"] - float(expected_duration)) <= 0.02
        if not mech_checks["duration_matches_expected"]:
            mech_issues.append("reference_duration_mismatch")

    gen_info = {"supplied": generation_record_path is not None, "record_valid": None, "language": None, "seed": None}
    if generation_record_path is not None:
        try:
            gen_record = validate_generation_record(
                generation_record_path, audio=audio_path, text=text, speaker=speaker
            )
            settings = ((gen_record.get("request") or {}).get("tts") or {}).get("settings") or {}
            gen_lang = settings.get("language")
            gen_seed = settings.get("seed")
            gen_info.update({"record_valid": True, "language": gen_lang, "seed": gen_seed, "cache_key": gen_record.get("cache_key")})
            if expected_language and gen_lang != expected_language:
                mech_checks["language_is_english"] = False
                mech_issues.append(f"generation_language_mismatch:{gen_lang}")
            if expected_seed is not None and gen_seed != expected_seed:
                mech_checks["seed_matches_expected"] = False
                mech_issues.append(f"generation_seed_mismatch:{gen_seed}")
        except Exception as exc:
            gen_info["record_valid"] = False
            mech_checks["generation_record_valid"] = False
            mech_issues.append(f"generation_record_invalid:{exc}")

    human_checks = {
        "reuse_permission_authorized": False,
        "consent_bound_to_exact_audio_and_speaker": False,
        "transcript_matches_speech": False,
        "single_speaker_no_other_voice": False,
        "speaker_identity_verified": False,
        "word_boundaries_not_cut": False,
        "intelligibility_and_noise_ok": False,
        "not_substituted_from_other_reference": True,
    }
    consent_record = None
    review_record = None

    if consent_path is not None and Path(consent_path).is_file():
        consent_record = json.loads(Path(consent_path).read_text(encoding="utf-8"))
        c_sha = consent_record.get("reference_audio_sha256")
        if c_sha and c_sha != audio_sha256:
            human_checks["not_substituted_from_other_reference"] = False
            mech_issues.append("consent_belongs_to_different_audio_sha256")
        if (
            consent_record.get("authorized") is True
            and consent_record.get("scope") == "voice_cloning"
            and consent_record.get("speaker_id") == speaker
            and c_sha == audio_sha256
            and isinstance(consent_record.get("evidence"), str)
            and consent_record["evidence"].strip()
            and isinstance(consent_record.get("verified_by"), str)
            and consent_record["verified_by"].strip()
        ):
            human_checks["reuse_permission_authorized"] = True
            human_checks["consent_bound_to_exact_audio_and_speaker"] = True

    if review_path is not None and Path(review_path).is_file():
        review_record = json.loads(Path(review_path).read_text(encoding="utf-8"))
        tx = review_record.get("transcript") or {}
        qual = review_record.get("quality") or {}
        for field_sha in (tx.get("reference_audio_sha256"), qual.get("reference_audio_sha256")):
            if field_sha and field_sha != audio_sha256:
                human_checks["not_substituted_from_other_reference"] = False
                mech_issues.append("review_belongs_to_different_audio_sha256")
        if tx.get("reference_text_sha256") and tx.get("reference_text_sha256") != text_sha256:
            human_checks["not_substituted_from_other_reference"] = False
            mech_issues.append("review_belongs_to_different_text_sha256")
        if (
            tx.get("status") == "verified"
            and isinstance(tx.get("verified_by"), str)
            and tx["verified_by"].strip()
            and tx.get("reference_audio_sha256") == audio_sha256
            and tx.get("reference_text_sha256") == text_sha256
        ):
            human_checks["transcript_matches_speech"] = True
        q_verifier = isinstance(qual.get("verified_by"), str) and bool(qual["verified_by"].strip())
        q_hash_ok = qual.get("reference_audio_sha256") == audio_sha256
        if q_verifier and q_hash_ok:
            if qual.get("single_speaker_verified") is True and qual.get("known_overlap_seconds") == 0 and type(qual.get("known_overlap_seconds")) in (int, float):
                human_checks["single_speaker_no_other_voice"] = True
            if qual.get("speaker_identity_verified") is True:
                human_checks["speaker_identity_verified"] = True
            if qual.get("boundaries_verified") is True:
                human_checks["word_boundaries_not_cut"] = True
            if qual.get("intelligibility_verified") is True and qual.get("unresolved_issues") == []:
                human_checks["intelligibility_and_noise_ok"] = True

    mechanical_passed = all(mech_checks.values()) and not mech_issues and human_checks["not_substituted_from_other_reference"]
    pending_human_items = [k for k, v in human_checks.items() if not v]
    all_human_verified = len(pending_human_items) == 0
    ready = mechanical_passed and all_human_verified

    if not mechanical_passed:
        status = "blocked_mechanical_or_provenance_error"
    elif not all_human_verified:
        status = "blocked_pending_human_review"
    else:
        status = "verified_ready_for_clone"

    report = {
        "kind": "phase8_english_reference_audit",
        "format_version": 1,
        "status": status,
        "ready_for_clone_synthesis": ready,
        "speaker_id": speaker,
        "expected_language": expected_language,
        "reference_audio_path": str(audio_path),
        "reference_text_path": str(text_path),
        "reference_audio_sha256": audio_sha256,
        "reference_text_sha256": text_sha256,
        "reference_text": text,
        "mechanical_verification": {
            "passed": mechanical_passed,
            "checks": mech_checks,
            "issues": mech_issues,
            "wav_measurement": measured,
            "boundary_diagnostics": boundaries,
            "acoustic_features": features,
            "generation_provenance": gen_info,
        },
        "human_verification": {
            "all_verified": all_human_verified,
            "checks": human_checks,
            "consent_supplied": consent_record is not None,
            "review_supplied": review_record is not None,
            "consent_record": consent_record,
            "review_record": review_record,
        },
        "pending_human_items": pending_human_items,
        "policy": "Mechanical waveform checks never grant human transcript/quality approval; Japanese reference reviews cannot substitute for English reference reviews.",
    }

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"audit-{speaker}-{audio_sha256[:12]}"
    tmp = Path(tempfile.mkdtemp(prefix=".audit-", dir=root))
    try:
        copied_wav = "reference.wav"
        shutil.copyfile(audio_path, tmp / copied_wav)
        if digest(tmp / copied_wav) != audio_sha256:
            raise ValueError("Reference audio changed while writing audit bundle.")
        (tmp / "reference.txt").write_text(text, encoding="utf-8")
        prepare_review_template(tmp / copied_wav, tmp / "reference.txt", speaker, tmp / "templates")
        html_content = _render_english_reference_review_html(report, copied_wav)
        (tmp / "english_reference_review.html").write_text(html_content, encoding="utf-8")
        atomic_json(tmp / "audit_report.json", report, overwrite=False)
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return target / "audit_report.json", report


def build_phase8_benchmark_script(speaker, *, video_id="phase8_cross_lingual", sentences=None, timing_config=None):
    """Build a valid v1 dubbing_script.json for the 3 Phase 8 Japanese benchmark sentences."""
    if speaker in UNKNOWN or not isinstance(speaker, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", speaker):
        raise ValueError("Valid known speaker ID required for benchmark script.")
    rows = copy.deepcopy(sentences or PHASE8_BENCHMARK_SENTENCES)
    dictionary = PronunciationDictionary(BENCHMARK_READINGS)
    timing = TimingEngine(timing_config or TimingConfig())
    entries = []
    for item in rows:
        text_ja = item["text_ja"].strip()
        normalized, changes = dictionary.normalize(text_ja)
        reading = "".join(c for c in normalized if ("ァ" <= c <= "ヺ" or c == "ー"))
        pauses = [
            {"text": m.group(), "kind": PAUSES[m.group()[0]], "start": m.start()}
            for m in re.finditer("[" + re.escape("".join(PAUSES)) + "]+", normalized)
        ]
        ja_analysis = {
            "reading": reading,
            "mora_count": count_mora(reading),
            "pauses": pauses,
            "normalizations": changes,
            "warnings": [],
        }
        start = float(item["relative_start"])
        end = float(item["relative_end"])
        available = round(end - start, 3)
        est = timing.estimate(ja_analysis, available)
        entries.append({
            "id": item["id"],
            "source_utterance_ids": list(item["source_utterance_ids"]),
            "speaker": speaker,
            "speaker_confidence": "verified_reference",
            "start": round(start, 3),
            "end": round(end, 3),
            "relative_start": start,
            "relative_end": end,
            "text_en": item["text_en"],
            "original_text_ja": text_ja,
            "text_ja": text_ja,
            "japanese_analysis": ja_analysis,
            "original_timing": est,
            **est,
            "estimated_playback_end": round(start + est["estimated_duration_seconds"], 3),
            "review_required": False,
            "review_reasons": [],
            "provenance": [{"benchmark_sentence": item["id"]}],
            "overlapping_entry_ids": [],
            "same_speaker_conflict_ids": [],
            "timeline_id": speaker,
        })
    config = {
        "engine_version": "phase8-benchmark-1",
        "japanese": {"engine": "pronunciation_dictionary", "pronunciations": BENCHMARK_READINGS},
        "timing": asdict(timing.config),
        "seconds": 30.0,
    }
    input_fp = fingerprint({"video_id": video_id, "speaker": speaker, "sentences": rows})
    script = {
        "format_version": 1,
        "engine_version": "phase8-benchmark-1",
        "video_id": video_id,
        "clip": {"start": 0.0, "end": 30.0},
        "preview": {"relative_start": 0, "relative_end": 30.0, "boundary_policy": "exclude_crossing_utterance"},
        "input_fingerprint": input_fp,
        "configuration": config,
        "cache_key": fingerprint({"input": input_fp, "config": config, "entries": entries}),
        "entries": entries,
        "speaker_timelines": {speaker: [e["id"] for e in entries]},
        "excluded": [],
    }
    validate_script(script)
    return script


def evaluate_cross_lingual_quality(
    *,
    preset_feedback_path,
    clone_feedback_path,
    output_dir,
    embedder=None,
    asr_transcriptions=None,
):
    """Compare Preset vs English-reference Voice Clone across all Phase 8 metrics."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = compare_feedback(preset_feedback_path, clone_feedback_path, output_dir / "feedback_comparison")

    preset_doc = json.loads(Path(preset_feedback_path).read_text(encoding="utf-8"))
    clone_doc = json.loads(Path(clone_feedback_path).read_text(encoding="utf-8"))
    preset_rows = {r["id"]: r for r in preset_doc["entries"] if r["generation_status"] == "complete"}
    clone_rows = {r["id"]: r for r in clone_doc["entries"] if r["generation_status"] == "complete"}

    references = {}
    speaker_languages = {}
    for row in clone_rows.values():
        spk = row["speaker"]
        vp = row["voice_profile"]["profile"]
        references[spk] = vp["reference_audio"]
        gen_meta = ((vp.get("voice_characteristics") or {}).get("synthetic_generation") or {})
        lang = (((gen_meta.get("request") or {}).get("tts") or {}).get("settings") or {}).get("language", "English")
        speaker_languages[spk] = lang

    samples = []
    for uid, c_row in clone_rows.items():
        p_row = preset_rows[uid]
        samples.append({
            "id": uid,
            "speaker": c_row["speaker"],
            "text_ja": c_row["text_ja"],
            "preset_audio": p_row["audio_file"],
            "clone_audio": c_row["audio_file"],
            "preset_generation_wall_seconds": (p_row.get("backend_result") or {}).get("processing_seconds"),
            "clone_generation_wall_seconds": (c_row.get("backend_result") or {}).get("processing_seconds"),
        })

    manifest_path = output_dir / "quality_request.json"
    atomic_json(
        manifest_path,
        {
            "kind": "voice_quality_request",
            "format_version": 1,
            "references": references,
            "samples": samples,
        },
    )
    quality_json_path = compare_voices(manifest_path, output_dir / "acoustic_quality", embedder=embedder)
    quality_doc = json.loads(quality_json_path.read_text(encoding="utf-8"))

    q_rows = {(r["id"], r["mode"]): r for r in quality_doc["rows"]}
    ref_features = quality_doc["references"]

    detailed_utterances = []
    for uid in sorted(clone_rows.keys()):
        p_fb = preset_rows[uid]
        c_fb = clone_rows[uid]
        p_q = q_rows[(uid, "preset")]
        c_q = q_rows[(uid, "clone")]
        spk = c_fb["speaker"]
        ref_info = ref_features[spk]
        ref_dsp = ref_info["features"]["dsp_estimates"]
        p_dsp = p_q["features"]["dsp_estimates"]
        c_dsp = c_q["features"]["dsp_estimates"]
        p_meas = p_q["measurement"]
        c_meas = c_q["measurement"]

        def pitch_block(dsp, ref_median):
            med = dsp.get("f0_hz_median")
            p10 = dsp.get("f0_hz_p10")
            p90 = dsp.get("f0_hz_p90")
            return {
                "f0_hz_median": round(med, 2) if med is not None else None,
                "f0_hz_std": round(dsp["f0_hz_std"], 2) if dsp.get("f0_hz_std") is not None else None,
                "f0_hz_p10": round(p10, 2) if p10 is not None else None,
                "f0_hz_p90": round(p90, 2) if p90 is not None else None,
                "f0_hz_span_p90_p10": round(p90 - p10, 2) if (p10 is not None and p90 is not None) else None,
                "f0_delta_vs_reference_hz": round(med - ref_median, 2) if (med is not None and ref_median is not None) else None,
                "voiced_fraction": round(dsp.get("voiced_fraction", 0.0), 4),
            }

        ref_f0 = ref_dsp.get("f0_hz_median")
        asr_info = (asr_transcriptions or {}).get(uid)
        detailed_utterances.append({
            "id": uid,
            "speaker": spk,
            "reference_language": speaker_languages.get(spk, "English"),
            "target_language": "Japanese",
            "text_ja": c_fb["text_ja"],
            "source_duration_seconds": c_fb["source_duration_seconds"],
            "estimated_duration_seconds": c_fb["estimated_duration_seconds"],
            "reference_f0_hz_median": round(ref_f0, 2) if ref_f0 is not None else None,
            "preset": {
                "audio_file": p_fb["audio_file"],
                "audio_duration_seconds": p_meas["audio_duration"],
                "generation_wall_seconds": p_q.get("generation_wall_seconds"),
                "cache_reused": bool(p_fb.get("cache_reused")),
                "pitch": pitch_block(p_dsp, ref_f0),
                "reference_embedding_cosine": p_q.get("reference_cosine"),
                "noise_and_clipping": {
                    "clipped_sample_fraction": p_meas["clipped_sample_fraction"],
                    "peak_amplitude": round(p_meas["peak_amplitude"], 5),
                    "rms_amplitude": round(p_meas["rms_amplitude"], 5),
                    "spectral_flatness_median": p_q["features"]["measured"].get("spectral_flatness_median"),
                    "clipping_detected": p_meas["clipped_sample_fraction"] > 0.001,
                },
            },
            "clone": {
                "audio_file": c_fb["audio_file"],
                "audio_duration_seconds": c_meas["audio_duration"],
                "generation_wall_seconds": c_q.get("generation_wall_seconds"),
                "conditioning_reused": (c_fb.get("backend_result") or {}).get("conditioning_reused"),
                "cache_reused": bool(c_fb.get("cache_reused")),
                "pitch": pitch_block(c_dsp, ref_f0),
                "reference_embedding_cosine": c_q.get("reference_cosine"),
                "noise_and_clipping": {
                    "clipped_sample_fraction": c_meas["clipped_sample_fraction"],
                    "peak_amplitude": round(c_meas["peak_amplitude"], 5),
                    "rms_amplitude": round(c_meas["rms_amplitude"], 5),
                    "spectral_flatness_median": c_q["features"]["measured"].get("spectral_flatness_median"),
                    "clipping_detected": c_meas["clipped_sample_fraction"] > 0.001,
                },
            },
            "comparison_deltas": {
                "clone_minus_preset_duration_seconds": round(c_meas["audio_duration"] - p_meas["audio_duration"], 4),
                "clone_minus_preset_embedding_cosine": (
                    round(c_q["reference_cosine"] - p_q["reference_cosine"], 4)
                    if (c_q.get("reference_cosine") is not None and p_q.get("reference_cosine") is not None)
                    else None
                ),
            },
            "asr_check": asr_info,
            "human_evaluation_pending": {
                "japanese_pronunciation_verified": False,
                "intonation_naturalness_verified": False,
                "text_omission_free_verified": False,
                "perceptual_voice_similarity_verified": False,
                "noise_and_artifact_free_verified": False,
            },
        })

    summary = {
        "kind": "phase8_cross_lingual_evaluation",
        "format_version": 1,
        "video_id": clone_doc["video_id"],
        "cross_lingual_pair": "English reference -> Japanese synthesis",
        "preset_model": preset_doc["tts"],
        "clone_model": clone_doc["tts"],
        "utterances": detailed_utterances,
        "within_speaker_consistency": quality_doc["consistency"],
        "feedback_comparison_dir": str(comp_dir),
        "acoustic_quality_json": str(quality_json_path),
        "review_required": True,
        "perceptual_quality_approved": False,
        "cross_lingual_caveats": [
            "English reference speech and Japanese generated speech differ in phoneme inventory and prosody, which can shift Speaker Embedding cosine similarity.",
            "Higher embedding cosine or closer F0 median does not guarantee perceptual voice identity, natural Japanese intonation, or complete pronunciation.",
            "All generated utterances require human listening verification before approval.",
        ],
    }

    report_json = output_dir / "phase8_evaluation.json"
    atomic_json(report_json, summary)
    report_md = output_dir / "phase8_evaluation.md"
    report_md.write_text(_render_cross_lingual_markdown(summary), encoding="utf-8")
    review_html = output_dir / "phase8_review.html"
    review_html.write_text(_render_cross_lingual_html(summary, quality_json_path.parent), encoding="utf-8")
    return report_json, summary


def _render_cross_lingual_markdown(summary):
    lines = [
        "# Phase 8: Cross-lingual Voice Cloning (英語参考 → 日本語生成) 比較評価レポート",
        "",
        "- **比較条件**: Preset Voice vs 英語参考音声からの Approved Voice Clone",
        "- **注意**: 英語参考音声と日本語生成音声の比較では、言語差（音素・韻律の違い）により Speaker Embedding のコサイン類似度が変動します。数値のみで声質再現の成功は断定せず、必ず試聴確認を行ってください。",
        "",
        "## 1. 発話別 測定・比較表",
        "",
        "| ID | 日本語テキスト | 方式 | WAV長(秒) | 生成時間(秒) | 参考F0(Hz) | 生成F0中央値(Hz) | F0変化(std / P10-P90) | 参考とのEmbedding類似度 | クリップ率 | 試聴確認 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in summary["utterances"]:
        for mode_key, label in (("preset", "Preset"), ("clone", "Clone (EN->JA)")):
            m = row[mode_key]
            p = m["pitch"]
            nc = m["noise_and_clipping"]
            wall = f"{m['generation_wall_seconds']:.3f}" if isinstance(m.get("generation_wall_seconds"), (int, float)) else "—"
            f0_med = f"{p['f0_hz_median']:.1f}" if p["f0_hz_median"] is not None else "—"
            ref_f0 = f"{row['reference_f0_hz_median']:.1f}" if row["reference_f0_hz_median"] is not None else "—"
            f0_var = (
                f"std={p['f0_hz_std']:.1f} ({p['f0_hz_p10']:.1f}–{p['f0_hz_p90']:.1f})"
                if p["f0_hz_std"] is not None and p["f0_hz_p10"] is not None and p["f0_hz_p90"] is not None
                else "—"
            )
            cos_val = f"{m['reference_embedding_cosine']:.3f}" if m.get("reference_embedding_cosine") is not None else "未計測"
            lines.append(
                f"| {row['id']} | {row['text_ja']} | {label} | {m['audio_duration_seconds']:.3f} | {wall} | {ref_f0} | {f0_med} | {f0_var} | {cos_val} | {nc['clipped_sample_fraction']:.5f} | 未確認 (要試聴) |"
            )
    lines += [
        "",
        "## 2. 同一話者としての一貫性 (発話間 Embedding Cosine)",
        "",
        "| 話者 | 方式 | 発話ペア | Cosine類似度 | 聴感上の一貫性確認 |",
        "|---|---|---|---:|---|",
    ]
    for group in summary["within_speaker_consistency"]:
        for pair in group.get("pairs", []):
            score = f"{pair['cosine']:.3f}" if pair.get("cosine") is not None else "未計測"
            lines.append(f"| {group['speaker']} | {group['mode']} | {pair['ids'][0]} vs {pair['ids'][1]} | {score} | 未確認 |")
    lines += [
        "",
        "## 3. 人間による試聴確認が必須の項目",
        "",
        "- 日本語発音の正確さ（数値・専門用語の読み誤りがないか）",
        "- 日本語としてのイントネーション・アクセントの自然さ",
        "- テキストの欠落（語頭・語尾の欠けやスキップがないか）",
        "- 英語参考音声との知覚上の声質類似性と、3文を通じた声質の一貫性",
        "",
    ]
    return "\n".join(lines)


def _render_cross_lingual_html(summary, quality_bundle_dir):
    esc = html.escape
    cards = []
    for row in summary["utterances"]:
        p = row["preset"]
        c = row["clone"]
        pp, cp = p["pitch"], c["pitch"]
        p_cos = f"{p['reference_embedding_cosine']:.3f}" if p.get("reference_embedding_cosine") is not None else "未計測"
        c_cos = f"{c['reference_embedding_cosine']:.3f}" if c.get("reference_embedding_cosine") is not None else "未計測"
        cards.append(f"""
        <div class="card">
          <h2>{esc(row['id'])} — {esc(row['speaker'])} ({esc(row['reference_language'])} 参考 → {esc(row['target_language'])} 生成)</h2>
          <p class="sentence"><strong>対象文:</strong> {esc(row['text_ja'])}</p>
          <table>
            <thead>
              <tr><th>指標</th><th>Preset Voice</th><th>Cross-lingual Voice Clone</th><th>差分 (Clone − Preset)</th></tr>
            </thead>
            <tbody>
              <tr>
                <th>WAV長さ (秒)</th>
                <td>{p['audio_duration_seconds']:.3f}s</td>
                <td>{c['audio_duration_seconds']:.3f}s</td>
                <td>{row['comparison_deltas']['clone_minus_preset_duration_seconds']:+.3f}s</td>
              </tr>
              <tr>
                <th>生成時間 (秒)</th>
                <td>{p['generation_wall_seconds'] if p['generation_wall_seconds'] is not None else '—'} (cache={p['cache_reused']})</td>
                <td>{c['generation_wall_seconds'] if c['generation_wall_seconds'] is not None else '—'} (cond_reused={c['conditioning_reused']})</td>
                <td>—</td>
              </tr>
              <tr>
                <th>F0中央値 (参考: {row['reference_f0_hz_median']} Hz)</th>
                <td>{pp['f0_hz_median']} Hz (Δ {pp['f0_delta_vs_reference_hz']} Hz)</td>
                <td>{cp['f0_hz_median']} Hz (Δ {cp['f0_delta_vs_reference_hz']} Hz)</td>
                <td>—</td>
              </tr>
              <tr>
                <th>声の高さの変化 (std / P10–P90)</th>
                <td>std={pp['f0_hz_std']} Hz ({pp['f0_hz_p10']}–{pp['f0_hz_p90']} Hz)</td>
                <td>std={cp['f0_hz_std']} Hz ({cp['f0_hz_p10']}–{cp['f0_hz_p90']} Hz)</td>
                <td>—</td>
              </tr>
              <tr>
                <th>参考とのSpeaker Embedding類似度</th>
                <td>{p_cos}</td>
                <td>{c_cos}</td>
                <td>{row['comparison_deltas']['clone_minus_preset_embedding_cosine'] if row['comparison_deltas']['clone_minus_preset_embedding_cosine'] is not None else '—'}</td>
              </tr>
              <tr>
                <th>クリッピング率 / RMS振幅</th>
                <td>clip={p['noise_and_clipping']['clipped_sample_fraction']:.5f} / rms={p['noise_and_clipping']['rms_amplitude']:.4f}</td>
                <td>clip={c['noise_and_clipping']['clipped_sample_fraction']:.5f} / rms={c['noise_and_clipping']['rms_amplitude']:.4f}</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <p><strong>Preset WAV:</strong> <code>afplay '{esc(p['audio_file'])}'</code></p>
          <audio controls preload="none" src="{esc(p['audio_file'])}"></audio>
          <p><strong>Clone WAV:</strong> <code>afplay '{esc(c['audio_file'])}'</code></p>
          <audio controls preload="none" src="{esc(c['audio_file'])}"></audio>
        </div>
        """)
    return f"""<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<title>Phase 8 Cross-lingual Voice Cloning Comparison</title>
<style>
  body {{ font: 15px/1.65 system-ui, -apple-system, sans-serif; max-width: 980px; margin: 28px auto; padding: 0 20px; color: #1f2937; background: #f9fafb; }}
  .card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 18px 20px; margin-bottom: 18px; }}
  .sentence {{ font-size: 17px; background: #f3f4f6; padding: 10px 14px; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
  th {{ background: #f3f4f6; }}
  audio {{ width: 100%; margin: 6px 0 12px; }}
  code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 13px; }}
</style>
<h1>Phase 8: Cross-lingual Voice Cloning (英語参考 → 日本語生成) 試聴・比較レポート</h1>
<div class="card">
  <p><strong>重要注意事項:</strong> 英語参考音声と日本語生成音声では言語・音素体系が異なるため、Speaker Embedding類似度やF0分布が同言語比較時と変動する場合があります。数値だけで声質再現の成功を断定せず、日本語発音・イントネーション・テキスト欠落・知覚上の声質と3文間の一貫性を必ず試聴して確認してください。</p>
  <p>ポータブル音声コピー付き試聴画面: <code>{esc(str(quality_bundle_dir / 'review.html'))}</code></p>
</div>
{''.join(cards)}
</html>
"""


def run_phase8_pipeline(
    *,
    speaker,
    reference_audio,
    reference_text_path,
    output_dir,
    preset_voice="Ryan",
    generation_record_path=None,
    consent_path=None,
    review_path=None,
    preset_backend=None,
    clone_backend=None,
    embedder=None,
    preset_feedback_dir=None,
    clone_feedback_dir=None,
    seed=0,
    expected_sha256=None,
    expected_duration=None,
    expected_seed=None,
):
    """Execute Phase 8 end-to-end: audit English reference -> block if unverified -> run Preset & Clone -> compare."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_path, audit_report = audit_english_reference(
        speaker=speaker,
        reference_audio=reference_audio,
        reference_text_path=reference_text_path,
        output_dir=root / "reference_audit",
        generation_record_path=generation_record_path,
        consent_path=consent_path,
        review_path=review_path,
        expected_sha256=expected_sha256,
        expected_duration=expected_duration,
        expected_seed=expected_seed,
        expected_language="English",
    )
    if not audit_report["ready_for_clone_synthesis"]:
        raise CloneBlockedForReviewError(
            f"English reference audio cannot be cloned until review completes ({audit_report['status']}): "
            f"pending={audit_report['pending_human_items']}. Open {audit_path.parent / 'english_reference_review.html'}.",
            audit_report_path=audit_path,
            review_html_path=audit_path.parent / "english_reference_review.html",
        )

    characteristics = {"reference_language": "English", "target_language": "Japanese"}
    if generation_record_path is not None:
        characteristics["synthetic_generation"] = validate_generation_record(
            generation_record_path,
            audio=reference_audio,
            text=Path(reference_text_path).read_text(encoding="utf-8"),
            speaker=speaker,
        )

    profile_json = register_profile(
        speaker=speaker,
        audio=reference_audio,
        text_path=reference_text_path,
        consent_path=consent_path,
        review_path=review_path,
        output=root / "voice_registry",
        characteristics=characteristics,
    )
    profiles = load_profiles(profile_json)

    script = build_phase8_benchmark_script(speaker)
    script_path = root / "phase8_dubbing_script.json"
    if not script_path.exists() or json.loads(script_path.read_text(encoding="utf-8")) != script:
        atomic_json(script_path, script)

    if preset_backend is None:
        from tts_generator import DEFAULT_MODEL, QwenPresetSynthesizer
        preset_backend = QwenPresetSynthesizer(DEFAULT_MODEL, seed=seed)
    if clone_backend is None:
        from voice_cloning import QwenBaseSynthesizer
        clone_backend = QwenBaseSynthesizer(DEFAULT_BASE, seed=seed)

    preset_dir = Path(preset_feedback_dir).resolve() if preset_feedback_dir else (root / "preset_feedback")
    clone_dir = Path(clone_feedback_dir).resolve() if clone_feedback_dir else (root / "clone_feedback")

    preset_fb = run_feedback(
        script_path,
        preset_dir,
        preset_backend,
        {speaker: preset_voice},
        seconds=30.0,
    )
    clone_voices = {speaker: "approved_clone:" + profile_snapshot(profiles[speaker])["profile_fingerprint"]}
    clone_fb = run_feedback(
        script_path,
        clone_dir,
        clone_backend,
        clone_voices,
        seconds=30.0,
        voice_profiles=profiles,
    )

    eval_json, eval_summary = evaluate_cross_lingual_quality(
        preset_feedback_path=preset_fb,
        clone_feedback_path=clone_fb,
        output_dir=root / "evaluation",
        embedder=embedder,
    )
    return {
        "status": "completed_pending_listening_review",
        "audit_report": str(audit_path),
        "registered_profile": str(profile_json),
        "script_path": str(script_path),
        "preset_feedback": str(preset_fb),
        "clone_feedback": str(clone_fb),
        "evaluation_json": str(eval_json),
        "evaluation_markdown": str(eval_json.with_suffix(".md")),
        "evaluation_html": str(eval_json.parent / "phase8_review.html"),
        "summary": eval_summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit-reference", help="Inspect English reference WAV and generate human review HTML")
    audit_p.add_argument("--speaker", default="SYNTH_EN")
    audit_p.add_argument("--reference-audio", type=Path)
    audit_p.add_argument("--reference-text", type=Path)
    audit_p.add_argument("--generation-record", type=Path)
    audit_p.add_argument("--consent", type=Path)
    audit_p.add_argument("--review", type=Path)
    audit_p.add_argument("--search-root", type=Path, action="append", help="Search directory for Phase 7 English WAV by SHA-256")
    audit_p.add_argument("--expected-sha256", default=PHASE7_ENGLISH_REFERENCE["sha256"])
    audit_p.add_argument("--expected-duration", type=float, default=PHASE7_ENGLISH_REFERENCE["duration_seconds"])
    audit_p.add_argument("--expected-seed", type=int, default=PHASE7_ENGLISH_REFERENCE["seed"])
    audit_p.add_argument("--output", type=Path, default=Path("output/phase8_cross_lingual/reference_audit"))

    run_p = sub.add_parser("run-benchmark", help="Run Phase 8 English->Japanese Voice Clone benchmark after human review")
    run_p.add_argument("--speaker", default="SYNTH_EN")
    run_p.add_argument("--reference-audio", type=Path, required=True)
    run_p.add_argument("--reference-text", type=Path, required=True)
    run_p.add_argument("--generation-record", type=Path)
    run_p.add_argument("--consent", type=Path, required=True)
    run_p.add_argument("--review", type=Path, required=True)
    run_p.add_argument("--preset-voice", default="Ryan")
    run_p.add_argument("--preset-feedback-dir", type=Path)
    run_p.add_argument("--clone-feedback-dir", type=Path)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--embeddings", action="store_true", help="Compute pyannote speaker embeddings offline")
    run_p.add_argument("--expected-sha256", default=PHASE7_ENGLISH_REFERENCE["sha256"])
    run_p.add_argument("--expected-duration", type=float, default=PHASE7_ENGLISH_REFERENCE["duration_seconds"])
    run_p.add_argument("--expected-seed", type=int, default=PHASE7_ENGLISH_REFERENCE["seed"])
    run_p.add_argument("--output", type=Path, default=Path("output/phase8_cross_lingual"))

    args = parser.parse_args(argv)
    try:
        if args.command == "audit-reference":
            audio = args.reference_audio
            if audio is None and args.search_root:
                audio = discover_reference_by_sha256(args.search_root, args.expected_sha256)
            if audio is None:
                raise FileNotFoundError("Provide --reference-audio or a --search-root containing the target WAV.")
            text_path = args.reference_text or (audio.parent / "reference.txt")
            gen_path = args.generation_record
            if gen_path is None and (audio.parent / "generation.json").is_file():
                gen_path = audio.parent / "generation.json"
            report_path, report = audit_english_reference(
                speaker=args.speaker,
                reference_audio=audio,
                reference_text_path=text_path,
                output_dir=args.output,
                generation_record_path=gen_path,
                consent_path=args.consent,
                review_path=args.review,
                expected_sha256=args.expected_sha256,
                expected_duration=args.expected_duration,
                expected_seed=args.expected_seed,
            )
            print(json.dumps({
                "status": report["status"],
                "ready_for_clone_synthesis": report["ready_for_clone_synthesis"],
                "mechanical_passed": report["mechanical_verification"]["passed"],
                "pending_human_items": report["pending_human_items"],
                "audit_report": str(report_path),
                "review_html": str(report_path.parent / "english_reference_review.html"),
            }, ensure_ascii=False, indent=2))
            return 0 if report["ready_for_clone_synthesis"] else 2

        if args.command == "run-benchmark":
            embedder = None
            if args.embeddings:
                from speaker_refinement import OfflinePyannote
                embedder = OfflinePyannote()
            result = run_phase8_pipeline(
                speaker=args.speaker,
                reference_audio=args.reference_audio,
                reference_text_path=args.reference_text,
                output_dir=args.output,
                preset_voice=args.preset_voice,
                generation_record_path=args.generation_record,
                consent_path=args.consent,
                review_path=args.review,
                embedder=embedder,
                preset_feedback_dir=args.preset_feedback_dir,
                clone_feedback_dir=args.clone_feedback_dir,
                seed=args.seed,
                expected_sha256=args.expected_sha256,
                expected_duration=args.expected_duration,
                expected_seed=args.expected_seed,
            )
            print(json.dumps({k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False, indent=2))
            return 0
    except CloneBlockedForReviewError as blocked:
        print(json.dumps({
            "status": "blocked_pending_human_review",
            "message": str(blocked),
            "audit_report": str(blocked.audit_report_path),
            "review_html": str(blocked.review_html_path),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
