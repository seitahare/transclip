"""Reference/preset/clone listening comparisons, preset voice catalog by age/atmosphere,
and automated voice similarity & Japanese naturalness verification (scores never fabricate human approval).

Reuses TRANS voice_qa's reference and within-speaker comparisons with existing
TransClip acoustic/embedding adapters. Handles one or many speakers without
inventing a small-sample acceptance threshold.
"""
import html
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from language.cache import atomic_json, fingerprint
from language.timing import measure_wav
import voice_profiler
from voice_profiler import acoustic_features, cosine


# ============================================================================
# Phase 13: Preset Voice Catalog & Speaker-Adaptive Voice/Prosody Selection
# ============================================================================

PRESET_VOICE_CATALOG = {
    "Ryan": {
        "voice": "Ryan",
        "age_group": "adult",
        "gender": "masculine",
        "pitch_register": "medium_low",
        "f0_hz_range": (95.0, 160.0),
        "f0_center_hz": 125.0,
        "atmospheres": ("calm", "conversational", "lecture", "professional", "neutral"),
        "description": "落ち着いた成人男性トーン（対談・インタビュー・解説向け）",
    },
    "Aiden": {
        "voice": "Aiden",
        "age_group": "young",
        "gender": "masculine",
        "pitch_register": "medium",
        "f0_hz_range": (125.0, 195.0),
        "f0_center_hz": 152.0,
        "atmospheres": ("energetic", "friendly", "conversational", "casual", "warm"),
        "description": "明るく親しみやすい若年〜成人男性トーン（対話・カジュアル向け）",
    },
    "Ethan": {
        "voice": "Ethan",
        "age_group": "mature",
        "gender": "masculine",
        "pitch_register": "low",
        "f0_hz_range": (75.0, 130.0),
        "f0_center_hz": 102.0,
        "atmospheres": ("authoritative", "deep", "lecture", "narrative", "calm"),
        "description": "深みのある落ち着いた熟年男性トーン（講義・ナレーション向け）",
    },
    "Claire": {
        "voice": "Claire",
        "age_group": "adult",
        "gender": "feminine",
        "pitch_register": "medium_high",
        "f0_hz_range": (160.0, 235.0),
        "f0_center_hz": 195.0,
        "atmospheres": ("calm", "warm", "conversational", "professional", "neutral"),
        "description": "自然で落ち着いた成人女性トーン（インタビュー・対談向け）",
    },
    "Mia": {
        "voice": "Mia",
        "age_group": "young",
        "gender": "feminine",
        "pitch_register": "high",
        "f0_hz_range": (195.0, 280.0),
        "f0_center_hz": 232.0,
        "atmospheres": ("bright", "energetic", "friendly", "casual", "expressive"),
        "description": "明瞭で活気のある若年女性トーン（プレゼン・対話向け）",
    },
    "Sophia": {
        "voice": "Sophia",
        "age_group": "mature",
        "gender": "feminine",
        "pitch_register": "medium",
        "f0_hz_range": (145.0, 210.0),
        "f0_center_hz": 175.0,
        "atmospheres": ("authoritative", "lecture", "narrative", "calm", "professional"),
        "description": "知的で安定した熟年女性トーン（講義・ドキュメンタリー向け）",
    },
}


def recommend_preset_voice(
    speaker_id,
    *,
    f0_hz=None,
    age_hint=None,
    atmosphere_hint=None,
    gender_hint=None,
    avoid_voices=(),
):
    """Select the best-matching preset voice for a speaker based on F0 pitch, age, and atmosphere/tone."""
    age_norm = str(age_hint or "").strip().lower() or None
    atm_norm = str(atmosphere_hint or "").strip().lower() or None
    gen_norm = str(gender_hint or "").strip().lower() or None
    avoid_set = {str(v) for v in (avoid_voices or ())}

    ranked = []
    for name, spec in PRESET_VOICE_CATALOG.items():
        score = 0.0
        reasons = []

        # 1. Age group match
        if age_norm:
            if spec["age_group"] == age_norm:
                score += 3.0
                reasons.append(f"age_match:{age_norm}")
            elif (age_norm, spec["age_group"]) in (("young", "adult"), ("adult", "young"), ("adult", "mature"), ("mature", "adult")):
                score += 1.0

        # 2. Atmosphere / speaking tone match
        if atm_norm:
            if atm_norm in spec["atmospheres"]:
                score += 3.5
                reasons.append(f"atmosphere_match:{atm_norm}")

        # 3. Gender presentation match
        if gen_norm:
            if spec["gender"] == gen_norm:
                score += 3.0
                reasons.append(f"gender_match:{gen_norm}")
            else:
                score -= 2.5

        # 4. Acoustic F0 fundamental frequency proximity
        if f0_hz is not None and isinstance(f0_hz, (int, float)) and math.isfinite(f0_hz) and f0_hz > 40.0:
            lo, hi = spec["f0_hz_range"]
            if lo <= f0_hz <= hi:
                score += 3.0
                reasons.append(f"f0_in_range:{f0_hz:.1f}Hz")
            semitone_dist = abs(12.0 * math.log2(float(f0_hz) / spec["f0_center_hz"]))
            score += max(0.0, 2.5 - 0.35 * semitone_dist)

        # 5. Multi-speaker diversity bonus so different speakers get distinct voices
        if name not in avoid_set:
            score += 0.75
        else:
            score -= 1.25

        ranked.append({
            "voice": name,
            "score": round(score, 4),
            "age_group": spec["age_group"],
            "gender": spec["gender"],
            "atmospheres": list(spec["atmospheres"]),
            "f0_center_hz": spec["f0_center_hz"],
            "reasons": reasons,
            "description": spec["description"],
        })

    # Deterministic tie-breaking by score descending, then catalog order
    ranked.sort(key=lambda r: (-r["score"], list(PRESET_VOICE_CATALOG.keys()).index(r["voice"])))
    best = ranked[0]
    return {
        "speaker_id": speaker_id,
        "recommended_voice": best["voice"],
        "match_score": best["score"],
        "match_reasons": best["reasons"],
        "voice_metadata": PRESET_VOICE_CATALOG[best["voice"]],
        "candidates": ranked,
    }


def optimize_tts_prosody_settings(
    *,
    prosody_style="auto",
    atmosphere_hint=None,
    speaker_f0_std_hz=None,
    base_temperature=0.9,
    base_top_k=50,
    base_top_p=1.0,
    base_repetition_penalty=1.05,
):
    """Select Japanese TTS sampling parameters tailored to the speaker's speaking tone and prosody style."""
    style = str(prosody_style or "auto").strip().lower()
    atm = str(atmosphere_hint or "").strip().lower()
    if style == "auto":
        if atm in ("calm", "lecture", "narrative", "authoritative", "professional"):
            style = "lecture" if atm in ("lecture", "authoritative") else "calm"
        elif atm in ("energetic", "bright", "expressive"):
            style = "expressive"
        elif speaker_f0_std_hz is not None and isinstance(speaker_f0_std_hz, (int, float)) and speaker_f0_std_hz > 32.0:
            style = "expressive"
        else:
            style = "natural"

    presets = {
        "natural": {
            "temperature": 0.78,
            "top_k": 35,
            "top_p": 0.92,
            "repetition_penalty": 1.03,
            "rationale": "Balanced Japanese pitch accent stability and natural conversational cadence.",
        },
        "conversational": {
            "temperature": 0.82,
            "top_k": 40,
            "top_p": 0.94,
            "repetition_penalty": 1.03,
            "rationale": "Warm conversational inflection suitable for interviews and dialogues.",
        },
        "calm": {
            "temperature": 0.72,
            "top_k": 28,
            "top_p": 0.90,
            "repetition_penalty": 1.02,
            "rationale": "Low-variance pitch contour and steady mora timing for calm speakers.",
        },
        "lecture": {
            "temperature": 0.70,
            "top_k": 25,
            "top_p": 0.88,
            "repetition_penalty": 1.02,
            "rationale": "Clear articulation and measured tempo for lectures and technical explanations.",
        },
        "expressive": {
            "temperature": 0.88,
            "top_k": 45,
            "top_p": 0.96,
            "repetition_penalty": 1.04,
            "rationale": "Wider dynamic F0 range for energetic or expressive speakers.",
        },
    }
    chosen = presets.get(style, {
        "temperature": float(base_temperature),
        "top_k": int(base_top_k),
        "top_p": float(base_top_p),
        "repetition_penalty": float(base_repetition_penalty),
        "rationale": "Explicit custom TTS sampling parameters.",
    })
    return {
        "prosody_style": style,
        **chosen,
    }


def evaluate_dubbing_voice_and_naturalness(
    entries,
    *,
    speaker_references=None,
    embedder=None,
):
    """Evaluate voice similarity against speaker references and Japanese speech naturalness metrics.

    Never fabricates human perceptual verification (`human_perceptual_verified` is always False).
    """
    from prosody_experiments import _read_pcm_for_dsp, analyze_japanese_prosody, _cosine_sim

    ref_features_by_speaker = {}
    ref_vectors_by_speaker = {}
    for spk, ref_obj in (speaker_references or {}).items():
        ref_path = getattr(ref_obj, "reference_audio", None) or (ref_obj.get("reference_audio") if isinstance(ref_obj, dict) else str(ref_obj))
        if ref_path and Path(ref_path).is_file():
            try:
                samples, sr = _read_pcm_for_dsp(ref_path)
                feats = acoustic_features(samples, sr)
                ref_features_by_speaker[spk] = feats
                if embedder is not None:
                    vec = embedder.embed(samples.astype(np.float32), sr) if hasattr(embedder, "embed") else None
                    if vec is not None:
                        ref_vectors_by_speaker[spk] = vec
            except Exception:
                pass

    utterance_evaluations = []
    naturalness_warnings = []

    for e in entries or []:
        if e.get("generation_status") != "complete":
            continue
        uid = e.get("id")
        spk = e.get("speaker")
        wav_path = e.get("active_audio_file") or e.get("audio_file")
        if not wav_path or not Path(wav_path).is_file():
            continue

        mora = e.get("mora_count")
        prosody = e.get("prosody_metrics") or analyze_japanese_prosody(wav_path, mora_count=mora)
        mora_rate = prosody.get("spoken_mora_per_second")
        step_semi = prosody.get("frame_to_frame_step_semitones_median")
        clipped = float(prosody.get("clipped_sample_fraction") or 0.0)
        f0_med = prosody.get("f0_hz_median")

        issues = []
        if clipped > 0.001:
            issues.append("audio_clipping_detected")
        if mora_rate is not None and (mora_rate < 3.8 or mora_rate > 12.5):
            issues.append(f"abnormal_speaking_rate:{mora_rate:.2f}_mora_per_s")
        if step_semi is not None and step_semi > 2.2:
            issues.append(f"pitch_instability:{step_semi:.2f}_semitones")

        ref_f0_med = None
        f0_semitone_diff = None
        ref_cosine = None
        if spk in ref_features_by_speaker:
            ref_dsp = ref_features_by_speaker[spk].get("dsp_estimates") or {}
            ref_f0_med = ref_dsp.get("f0_hz_median")
            if f0_med and ref_f0_med and f0_med > 0 and ref_f0_med > 0:
                f0_semitone_diff = round(abs(12.0 * math.log2(float(f0_med) / float(ref_f0_med))), 3)
        if spk in ref_vectors_by_speaker and embedder is not None:
            try:
                s_arr, s_sr = _read_pcm_for_dsp(wav_path)
                u_vec = embedder.embed(s_arr.astype(np.float32), s_sr)
                if u_vec is not None:
                    ref_cosine = round(float(_cosine_sim(ref_vectors_by_speaker[spk], u_vec)), 4)
            except Exception:
                pass

        if issues:
            naturalness_warnings.append({"id": uid, "speaker": spk, "issues": issues})

        utterance_evaluations.append({
            "id": uid,
            "speaker": spk,
            "mora_count": mora,
            "spoken_mora_per_second": mora_rate,
            "f0_hz_median": f0_med,
            "reference_f0_hz_median": round(ref_f0_med, 2) if ref_f0_med is not None else None,
            "f0_distance_semitones": f0_semitone_diff,
            "reference_embedding_cosine": ref_cosine,
            "frame_step_semitones_median": step_semi,
            "clipped_sample_fraction": clipped,
            "acoustic_naturalness_passed": len(issues) == 0,
            "issues": issues,
            "human_perceptual_verified": False,
        })

    return {
        "evaluated_utterance_count": len(utterance_evaluations),
        "acoustic_naturalness_passed": len(naturalness_warnings) == 0,
        "naturalness_warnings": naturalness_warnings,
        "utterances": utterance_evaluations,
        "human_perceptual_verified": False,
    }


def compare_voices(manifest_path, output, *, embedder=None):
    from audio_mixer import ffmpeg
    from voice_cloning import digest
    source=Path(manifest_path).resolve();request=json.loads(source.read_text())
    if request.get('kind')!='voice_quality_request' or request.get('format_version')!=1:
        raise ValueError('Expected voice_quality_request v1.')
    refs=request.get('references',{});samples=request.get('samples',[])
    if not refs or not samples:raise ValueError('References and samples are required.')
    if any(s in ('UNKNOWN','MIXED','REVIEW','',None) for s in refs):raise ValueError('Known speaker labels required.')
    ids=[r['id'] for r in samples]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample IDs.')
    files={}
    def resolve(value):
        path=(source.parent/value).resolve()
        if str(path) not in files:files[str(path)]=measure_wav(path)
        return str(path)
    references={s:resolve(p) for s,p in refs.items()}
    normalized=[]
    for row in samples:
        if row['speaker'] not in references or not isinstance(row.get('text_ja'),str) or not row['text_ja'].strip():
            raise ValueError('Missing speaker/reference or Japanese text.')
        r={**row}
        if not any(row.get(mode+'_audio') for mode in ('preset','clone')):raise ValueError('At least one generated WAV required.')
        for mode in ('preset','clone'):
            r[mode+'_audio']=resolve(row[mode+'_audio']) if row.get(mode+'_audio') else None
        normalized.append(r)
    identity={'request':request,'files':files,'implementation':digest(__file__),
              'dsp_implementation':digest(voice_profiler.__file__),
              'embedding':embedder.identity if embedder else None}
    key=fingerprint(identity);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);target=root/key[:20]
    if target.exists():
        saved=json.loads((target/'quality.json').read_text())
        if saved.get('identity')!=identity:raise ValueError('Quality cache identity mismatch.')
        for name,sha in saved['artifacts'].items():
            if digest(target/name)!=sha:raise ValueError('Quality artifact changed.')
        return target/'quality.json'
    vectors={};features={}
    for path in files:
        data=ffmpeg(['-i',path,'-vn','-ar','16000','-ac','1','-f','s16le','pipe:1'])
        x=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
        features[path]=acoustic_features(x,16000)
        features[path].pop('asr_estimates',None)
        if embedder:
            vectors[path]=embedder.embed(x)
            if vectors[path] is not None:cosine(vectors[path],vectors[path])
    def similarity(a,b):
        return cosine(vectors[a],vectors[b]) if embedder and vectors.get(a) is not None and vectors.get(b) is not None else None
    rows=[]
    for sample in normalized:
        for mode in ('preset','clone'):
            path=sample[mode+'_audio']
            if not path:continue
            reference=references[sample['speaker']]
            other={s:similarity(path,p) for s,p in references.items() if s!=sample['speaker']}
            rows.append({'id':sample['id'],'speaker':sample['speaker'],'mode':mode,'text_ja':sample['text_ja'],
                         'audio_file':path,'measurement':files[path],'features':features[path],
                         'reference_cosine':similarity(reference,path),'other_reference_cosines':other,
                         'script_characters_per_audio_second':len(''.join(sample['text_ja'].split()))/files[path]['audio_duration'],
                         'generation_wall_seconds':sample.get(mode+'_generation_wall_seconds'),
                         'pronunciation_verified':False,'perceptual_similarity_verified':False})
    consistency=[]
    for speaker in references:
        for mode in ('preset','clone'):
            group=[r for r in rows if r['speaker']==speaker and r['mode']==mode]
            consistency.append({'speaker':speaker,'mode':mode,'sample_count':len(group),
                'pairs':[{'ids':[a['id'],b['id']],'cosine':similarity(a['audio_file'],b['audio_file'])} for a,b in itertools.combinations(group,2)],
                'human_consistency_verified':False,'acceptance_threshold':None})
    result={'kind':'voice_quality_comparison','format_version':1,'identity':identity,
            'references':{s:{'audio_file':p,'measurement':files[p],'features':features[p]} for s,p in references.items()},
            'rows':rows,'consistency':consistency,'embedding_identity':embedder.identity if embedder else None,
            'review_required':True,'is_authorization_record':False,
            'limitations':['Analysis is not an authorization record.','Cosine similarity is not proof of identity or pronunciation.',
                          'No acceptance threshold is fitted to these samples.','Cross-language scores need separate listening interpretation.',
                          'F0/activity are DSP estimates; characters/second is a script-based rate proxy.']}
    tmp=Path(tempfile.mkdtemp(prefix='.quality-',dir=root))
    try:
        copied={}
        for i,path in enumerate(files):
            name=f'audio-{i}.wav';shutil.copyfile(path,tmp/name)
            if digest(tmp/name)!=files[path]['sha256']:raise ValueError('Audio changed during comparison.')
            copied[path]=name
        esc=html.escape
        page=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>Voice quality comparison</title>',
              '<style>body{font:16px/1.7 system-ui;max-width:900px;margin:32px auto;padding:20px}article{border:1px solid #ccc;padding:16px;margin:16px 0}audio{width:100%}</style>',
              '<h1>参考 / Preset / Clone 比較</h1><p>発音・声質・一貫性は未確認。スコアは承認ではありません。</p>']
        for speaker,path in references.items():
            page.append(f'<h2>{esc(speaker)} · Reference</h2><audio controls src="{copied[path]}"></audio>')
        for sample in normalized:
            page.append(f'<article><h2>{esc(sample["id"])} / {esc(sample["speaker"])}</h2><p>{esc(sample["text_ja"])}</p>')
            for mode in ('preset','clone'):
                path=sample[mode+'_audio']
                if not path:page.append(f'<p>{mode}: 未生成・比較保留</p>');continue
                row=next(r for r in rows if r['id']==sample['id'] and r['mode']==mode)
                pitch=row['features']['dsp_estimates']['f0_hz_median'];score=row['reference_cosine']
                page.append(f'<h3>{mode}</h3><audio controls src="{copied[path]}"></audio><p>実測 {files[path]["audio_duration"]:.3f}秒 / F0推定 {pitch} Hz / 参考とのcosine {score}</p>')
            page.append('</article>')
        page.append('<p>同じ文章の発音・欠落・音色・イントネーション・話速と、文章間の声質を実際に聞いて確認してください。</p></html>')
        (tmp/'review.html').write_text(''.join(page),encoding='utf-8')
        result['artifacts']={p.name:digest(p) for p in tmp.iterdir() if p.is_file()}
        atomic_json(tmp/'quality.json',result,overwrite=False);tmp.rename(target)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return target/'quality.json'
