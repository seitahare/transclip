"""Acoustic utterance attribution using existing local models; immutable outputs."""
import argparse
import copy
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from audio_mixer import read_json, sha256_file
from language.cache import atomic_json, fingerprint
from speaker_refinement import (OfflinePyannote, RATE, UNKNOWN, crop, read_audio,
                                validate_inputs)
from speaker_review import build_review, validate_decisions
from voice_profiler import acoustic_features, centroid, cosine, rank_speakers
from word_speaker_alignment import assign_speaker, covered_duration

BASE_DIR = Path(__file__).resolve().parent
SETTINGS = dict(min_exact_seconds=.8, min_short_seconds=.12, min_similarity=.70,
                min_margin=.18, short_similarity=.75, short_margin=.25,
                max_overlap_vote=.08, mixed_overlap_vote=.35,
                min_anchor_count=2, reference_guard_seconds=.10,
                target_embeddings_may_include_overlap=True,
                duration_probe_seconds=[1., 2., 4.], minimum_duration_similarity=.50,
                minimum_speech_evidence_seconds=.40,
                score_calibration='uncalibrated_rules_not_probability')


class AcousticBackend(OfflinePyannote):
    """Frame-level speech-count votes reuse the cached segmentation model."""
    def activity(self, samples):
        import torch
        from pyannote.audio.core.inference import Inference
        from pyannote.core import SlidingWindowFeature
        pipeline = self.load()
        if not pipeline._segmentation.model.specifications.powerset:
            raise ValueError('This adapter requires the existing powerset model.')
        scores = pipeline.get_segmentations({
            'waveform': torch.from_numpy(samples.copy()).float()[None, :], 'sample_rate': RATE})
        counts = np.sum(scores.data > .5, axis=2)
        votes = np.stack([counts >= 1, counts >= 2], axis=2).astype(float)
        aggregate = Inference.aggregate(SlidingWindowFeature(votes, scores.sliding_window),
                                       frames=pipeline._segmentation.model.receptive_field,
                                       hamming=True)
        return {'frame_start': aggregate.sliding_window.start,
                'frame_step': aggregate.sliding_window.step,
                'frame_duration': aggregate.sliding_window.duration,
                'votes': [[float(x) if math.isfinite(x) else None for x in row]
                          for row in aggregate.data],
                'interpretation': 'Weighted chunk votes for >=1 and >=2 active voices; '
                                  'same segmentation model as diarization, not independent truth or probability.'}


def activity_between(activity, a, b):
    step, duration = activity['frame_step'], activity['frame_duration']
    centers = activity['frame_start'] + duration / 2 + np.arange(len(activity['votes'])) * step
    values = np.asarray(activity['votes'], dtype=float)
    # Frame centers, rather than repeated receptive fields, define the sampling grid.
    mask = (centers >= a) & (centers < b)
    selected = values[mask]
    valid = selected[np.isfinite(selected).all(axis=1)]
    coverage = min(1., len(valid) * step / (b-a))
    return {'coverage': coverage, 'speech_vote': float(np.mean(valid[:, 0])) if len(valid) else None,
            'overlap_vote': float(np.mean(valid[:, 1])) if len(valid) else None,
            'overlap_supported_seconds': float(np.sum(valid[:, 1] >= .5) * step),
            'frame_count': len(valid), 'calibrated_probability': False}


def windows_for(a, b, duration):
    windows = [('exact', a, b)]
    if b-a < .8:
        windows += [('left_context', max(0., a-.95), b),
                    ('right_context', a, min(duration, b+.95)),
                    ('center_context', max(0., a-.55), min(duration, b+.55))]
    else:
        windows += [('expanded', max(0., a-.20), min(duration, b+.20))]
        if b-a >= 1.6:
            # Local disagreement reveals sequential speakers as well as mixtures.
            starts = list(np.arange(a, b-1.2+.001, .8))
            starts.append(b-1.2)
            windows += [('local', float(x), min(float(x)+1.2, b)) for x in sorted(set(starts))]
    return [{'scope': scope, 'relative_start': round(x, 6), 'relative_end': round(y, 6)}
            for scope, x, y in windows]


def independent_anchors(analysis, profiles, a, b, settings=SETTINGS):
    """Exclude ALL target/context time from references, then remove duplicate windows."""
    eligible = {(p['speaker_id'], r['relative_start'], r['relative_end'])
                for p in profiles['profiles'] for r in p['reference_candidates']
                if r['quality']['eligible']}
    grouped = {}
    for row in sorted(analysis['embedding_window_evidence'],
                      key=lambda r: -(r.get('speaker_similarity') or -1)):
        s, x, y = row['speaker_id'], row['relative_start'], row['relative_end']
        if (s, x, y) not in eligible or row['embedding'] is None:
            continue
        if x < b+settings['reference_guard_seconds'] and y > a-settings['reference_guard_seconds']:
            continue
        chosen = grouped.setdefault(s, [])
        if any(x < q['relative_end'] and y > q['relative_start'] for q in chosen):
            continue
        chosen.append(row)
    return grouped


def compare_embedding(vector, anchors):
    centers = {s: centroid([r['embedding'] for r in rows]) for s, rows in anchors.items() if rows}
    ranked = rank_speakers(vector, centers)
    if vector is not None:
        for row in ranked:
            peers = anchors[row['speaker_id']]
            scores = [cosine(vector, p['embedding']) for p in peers]
            row.update(reference_count=len(peers), reference_similarity_median=float(np.median(scores)),
                       reference_similarity_min=min(scores), reference_similarity_max=max(scores))
    return ranked


def quality_reasons(features):
    m, d = features['measured'], features['dsp_estimates']
    reasons = []
    if m['rms_amplitude'] < .005: reasons.append('low_energy')
    if m['clipped_fraction'] > .001: reasons.append('clipping')
    if d['energy_activity_fraction'] < .15: reasons.append('insufficient_active_audio')
    # Unvoiced consonants/pauses are common. Lack of F0 alone is not lack of speech.
    if d['voiced_fraction'] < .05 and (m['spectral_flatness_median'] or 0) > .15:
        reasons.append('limited_periodicity_and_noise_like_spectrum')
    if (m['spectral_flatness_median'] or 0) > .45: reasons.append('noise_like_spectrum')
    return reasons


def best_supported(window, settings=SETTINGS, short=False):
    ranked = window['candidates']
    speech = window['activity']['speech_vote']
    # ASR word spans can include pauses. Require enough speech, not 50% occupancy.
    speech_seconds = (speech or 0) * window['features']['measured']['duration_seconds']
    if (len(ranked) < 2 or window['quality_reasons'] or window['activity']['coverage'] < .80
            or speech is None or speech_seconds < settings['minimum_speech_evidence_seconds']
            or window['activity']['overlap_vote'] is None
            or window['activity']['overlap_vote'] > settings['max_overlap_vote']):
        return None
    best, second = ranked[:2]
    similarity = settings['short_similarity' if short else 'min_similarity']
    duration_check = best.get('duration_reference_check')
    if not short and duration_check and duration_check['usable_probe_count'] >= 3:
        similarity = duration_check['required_similarity']
    margin = settings['short_margin' if short else 'min_margin']
    if (best['cosine_similarity'] >= similarity
            and best['cosine_similarity']-second['cosine_similarity'] >= margin
            and best['reference_count'] >= settings['min_anchor_count']
            and best['reference_similarity_median'] >= similarity-.05):
        return best['speaker_id']
    return None


def reference_duration_checks(anchors, samples, backend, cache):
    """Model-labelled consistency probes, never human calibration or test accuracy.

    Each probe is compared with OTHER, nonoverlapping reference windows. Reference
    and probe windows have already been excluded from all target/context audio.
    """
    result = {}
    centers = {s: centroid([r['embedding'] for r in rows]) for s, rows in anchors.items() if rows}
    for s, rows in anchors.items():
        result[s] = {}
        for duration in SETTINGS['duration_probe_seconds']:
            positives = []; impostors = []; probes = []
            for row in rows[:6]:
                a, b = row['relative_start'], row['relative_end']
                if b-a < duration or len(rows) < 3: continue
                start = (a+b-duration)/2; end = start+duration
                cache_key = (round(start, 6), round(end, 6))
                if cache_key not in cache:
                    x = crop(samples, start, end)
                    cache[cache_key] = (backend.embed(x), quality_reasons(acoustic_features(x)))
                vector, bad = cache[cache_key]
                if vector is None or bad: continue
                peers = [p['embedding'] for p in rows if p is not row]
                positive = cosine(vector, centroid(peers))
                negative = max((cosine(vector, c) for other, c in centers.items() if other != s), default=1.)
                positives.append(positive); impostors.append(negative)
                probes.append({'relative_start':start, 'relative_end':end, 'parent_reference':[a,b],
                               'positive_cosine':positive, 'nearest_other_cosine':negative,
                               'peer_reference_count':len(peers)})
            required = max(SETTINGS['minimum_duration_similarity'],
                           (float(np.percentile(positives, 10))+float(np.percentile(impostors, 95)))/2,
                           float(np.percentile(impostors, 95))+.20) if len(positives) >= 3 else SETTINGS['min_similarity']
            result[s][str(duration)] = {'duration_seconds':duration, 'usable_probe_count':len(positives),
                                       'required_similarity':required, 'probes':probes,
                                       'human_calibrated':False, 'labels_source':'existing_model_reference_labels',
                                       'method':'leave-parent-out midpoint of positive p10 and other-speaker p95; also require other p95+0.20 and floor 0.50'}
    return result


def descriptor_comparison(features, profile, anchors):
    """Secondary diagnostics, never a speaker identity classifier by themselves."""
    intervals = {(r['relative_start'], r['relative_end']) for r in anchors}
    peers = [r['features'] for r in profile['analyzed_intervals']
             if (r['relative_start'], r['relative_end']) in intervals]
    result = {}
    for group, key in [('dsp_estimates', 'f0_hz_median'), ('measured', 'spectral_centroid_hz_median'),
                       ('measured', 'spectral_flatness_median'), ('measured', 'zero_crossing_rate_median')]:
        values = [r[group][key] for r in peers if r[group][key] is not None]
        target = features[group][key]
        result[key] = {'target': target, 'reference_median': float(np.median(values)) if values else None,
                       'reference_count': len(values)}
    return result


def conversation_context(rows, index):
    current = rows[index]
    previous = rows[index-1] if index else None
    following = rows[index+1] if index+1 < len(rows) else None
    result = {'previous': {k: previous[k] for k in ('id', 'speaker', 'text_en')} if previous else None,
              'following': {k: following[k] for k in ('id', 'speaker', 'text_en')} if following else None,
              'hypotheses': [], 'used_to_assign_without_audio': False}
    if previous and previous['text_en'].rstrip().endswith('?'):
        result['hypotheses'].append('previous_question_may_invite_turn_change; no fixed questioner ID')
    if previous and current['relative_start']-previous['relative_end'] <= .2:
        result['hypotheses'].append('adjacent_speech_may_continue_or_interject; acoustic_evidence_required')
    if following and following['speaker'] not in UNKNOWN and current['relative_end'] <= following['relative_start']+.05:
        result['hypotheses'].append('following_known_speaker_is_context_only')
    return result


def decide(row, windows, temporal, settings=SETTINGS):
    exact = windows[0]; a, b = row['relative_start'], row['relative_end']
    act = exact['activity']; vote = act['overlap_vote']
    reasons = []; speaker = None; method = 'unresolved'; status = 'unknown'
    hypotheses = []; temporal_overlap = len([c for c in temporal if c['overlap_seconds'] >= .03]) > 1
    support = [best_supported(w, settings, short=b-a < .8) for w in windows]
    if vote is not None and act['coverage'] >= .8 and vote >= settings['mixed_overlap_vote'] and act['overlap_supported_seconds'] >= .12:
        status = 'mixed'; reasons.append('frame_model_supports_multiple_voices_not_separated_transcripts')
        hypotheses.append('A_simultaneous_voices_model_supported')
        if b-a < .8: hypotheses.append('C_short_interjection_possible')
    elif b-a >= settings['min_exact_seconds']:
        other = [(w, s) for w, s in zip(windows[1:], support[1:]) if w['scope'] == 'local']
        disagree = any(s and support[0] and s != support[0] for _, s in other)
        if disagree:
            reasons.append('local_windows_disagree_possible_speaker_change')
            hypotheses.append('B_boundary_or_sequential_speaker_change_possible')
        # At least two views must agree. Local crops must not positively disagree.
        elif support[0] and support.count(support[0]) >= 2:
            speaker = support[0]; status = 'single'; method = 'direct_embedding_multiscale'
        else:
            reasons.append('insufficient_consistent_target_embedding')
    elif b-a >= settings['min_short_seconds']:
        left, right, center = support[1:]
        good = left and left == right == center
        # Never infer a tiny utterance solely from surrounding dominant speech.
        if (good and not exact['quality_reasons'] and act['coverage'] >= .8
                and vote is not None and vote <= settings['max_overlap_vote']
                and act['speech_vote'] >= .6 and not temporal_overlap):
            target_f0 = exact['features']['dsp_estimates']['f0_hz_median']
            flank_f0 = [w['features']['dsp_estimates']['f0_hz_median'] for w in windows[1:]]
            if target_f0 and all(f and .70 <= target_f0/f <= 1.43 for f in flank_f0):
                speaker = left; status = 'single'; method = 'short_acoustic_continuity'
        if speaker is None: reasons.append('short_target_needs_agreeing_flanks_and_pitch')
    else:
        reasons.append('target_too_short_for_reliable_attribution')
    if speaker and temporal_overlap:
        hypotheses.append('B_apparent_turn_overlap_not_supported_by_frame_audio')
    if speaker and temporal and all(c['speaker'] != speaker for c in temporal):
        hypotheses.append('B_or_D_acoustic_temporal_disagreement')
        # Strong target audio can disagree with old timing, but remains reviewable.
    if act['speech_vote'] is not None and act['speech_vote'] < .5:
        hypotheses.append('D_asr_timing_or_non_speech_possible')
    if speaker is None and status != 'mixed': hypotheses.append('E_insufficient_or_conflicting_acoustic_evidence')
    if temporal_overlap and not hypotheses: hypotheses.append('overlap_unresolved')
    reasons += exact['quality_reasons']
    if speaker: reasons.append('uncalibrated_automatic_attribution_requires_review')
    return {'speaker_id': speaker, 'status': status, 'method': method,
            'reasons': sorted(set(reasons)), 'window_support': support,
            'overlap': {'temporal_overlap': temporal_overlap, 'acoustic_activity': act,
                        'hypotheses': hypotheses, 'human_verified': False,
                        'separated_speaker_transcripts_available': False},
            'review_required': True, 'score_calibration': settings['score_calibration']}


def analyze_utterances(source, analysis, profiles, samples, activity, backend):
    results = []; duration = len(samples)/RATE; probe_cache = {}
    for index, row in enumerate(source):
        spans = windows_for(row['relative_start'], row['relative_end'], duration)
        anchors = independent_anchors(analysis, profiles, min(w['relative_start'] for w in spans),
                                      max(w['relative_end'] for w in spans))
        duration_checks = reference_duration_checks(anchors, samples, backend, probe_cache)
        windows = []
        for span in spans:
            a, b = span['relative_start'], span['relative_end']; x = crop(samples, a, b)
            features = acoustic_features(x)
            vector = backend.embed(x) if b-a >= .8 else None
            candidates = compare_embedding(vector, anchors)
            bucket = min(SETTINGS['duration_probe_seconds'], key=lambda d: abs(math.log(max(.8,b-a)/d)))
            for c in candidates:
                check = duration_checks[c['speaker_id']][str(bucket)]
                c['duration_reference_check'] = {k:v for k,v in check.items() if k != 'probes'}
            windows.append({**span, 'features': features, 'quality_reasons': quality_reasons(features),
                            'embedding': vector, 'candidates': candidates,
                            'activity': activity_between(activity, a, b)})
        temporal = assign_speaker(row, analysis['turns'])['speaker_candidates']
        automatic = decide(row, windows, temporal)
        result = {'utterance_id': row['id'], 'text_en': row['text_en'],
                  **{k: row[k] for k in ('start', 'end', 'relative_start', 'relative_end')},
                  'previous_speaker': row['speaker'], 'automatic': automatic,
                  'speaker_id': automatic['speaker_id'], 'status': automatic['status'],
                  'method': automatic['method'], 'review_required': True, 'human_confirmation': None,
                  'temporal_candidates': temporal, 'windows': windows,
                  'overlapping_turn_tracks': [copy.deepcopy(t) for t in analysis['turns']
                                              if t['relative_start'] < row['relative_end'] and t['relative_end'] > row['relative_start']],
                  'duration_reference_checks': duration_checks,
                  'reference_windows': {s: [{k: r[k] for k in ('relative_start', 'relative_end')}
                                             for r in rs] for s, rs in anchors.items()},
                  'descriptor_comparisons': {p['speaker_id']: descriptor_comparison(windows[0]['features'], p, anchors.get(p['speaker_id'], []))
                                             for p in profiles['profiles']},
                  'conversation_context': conversation_context(source, index)}
        results.append(result)
        print(f"Attribution {index+1}/{len(source)}: {row['id']} -> {automatic['speaker_id'] or automatic['status']}", flush=True)
    return results


def verified_human_labels(review_dir, decisions_path, original, analysis):
    if decisions_path is None: return {}, None
    folder = Path(review_dir).resolve()
    verify_bundle(folder)
    bundle = read_json(folder/'speaker_review.json')
    decisions = validate_decisions(bundle, read_json(decisions_path))
    if (bundle['video_id'] != analysis['video_id'] or bundle['clip'] != analysis['clip']
            or bundle['original_utterances_fingerprint'] != fingerprint(original)):
        raise ValueError('Human review does not match original source.')
    items = {r['utterance_id']: r for r in bundle['items']}
    labels = {}
    for d in decisions.get('decisions', []):
        item = items[d['utterance_id']]; path = (folder/item['audio_file']).resolve()
        if not path.is_relative_to(folder) or sha256_file(path) != d['audio_sha256']:
            raise ValueError('Human review audio changed or escapes review directory.')
        labels[d['utterance_id']] = {**d, 'reviewed_by': decisions['reviewed_by'],
                                    'reviewed_at': decisions.get('reviewed_at'),
                                    'source_review_key': bundle['review_key'],
                                    'decisions_sha256': sha256_file(decisions_path)}
    return labels, decisions


def apply_human_labels(results, source, labels):
    for row, original in zip(results, source):
        human = labels.get(row['utterance_id'])
        if human is None and original.get('human_review'):
            # Existing explicitly reviewed sources are authoritative, never automatically relabelled.
            human = {**original['human_review'], 'speaker': original['speaker'],
                     'attribution_status': original.get('attribution_status', 'unknown' if original['speaker'] in UNKNOWN else 'single'),
                     'source': 'existing_human_review'}
        if human is not None:
            row['human_confirmation'] = human
            row['speaker_id'] = human['speaker']
            row['status'] = human.get('attribution_status', 'unknown' if human['speaker'] is None else 'single')
            row['method'] = 'human_review'
        elif (row['status'] == 'unknown' and original['speaker'] not in UNKNOWN
              and 'local_windows_disagree_possible_speaker_change' not in row.get('automatic', {}).get('reasons', [])):
            # Lack of new evidence is not evidence that an existing label is wrong.
            row['speaker_id'] = original['speaker']
            row['status'] = 'single'
            row['method'] = 'previous_assignment_unverified'
    return results


def evaluate(results):
    labelled = [r for r in results if r['human_confirmation'] is not None]
    known = [r for r in labelled if r['human_confirmation']['speaker'] not in UNKNOWN]
    assigned = [r for r in known if r['automatic']['speaker_id'] is not None]
    correct = sum(r['automatic']['speaker_id'] == r['human_confirmation']['speaker'] for r in assigned)
    old_correct = sum(r['previous_speaker'] == r['human_confirmation']['speaker'] for r in known)
    return {'human_label_count': len(labelled), 'known_speaker_label_count': len(known),
            'automatic_assignments_on_known_labels': len(assigned), 'correct_automatic_assignments': correct,
            'selective_accuracy': correct/len(assigned) if assigned else None,
            'coverage_on_known_labels': len(assigned)/len(known) if known else None,
            'previous_correct_on_known_labels': old_correct,
            'new_correct_on_known_labels': correct,
            'thresholds_calibrated': False, 'threshold_selection_used_labels': False,
            'accuracy_improvement_verified': False,
            'limitations': 'Only supplied human labels evaluated; small/nonrandom review set is not population accuracy.'}


def downstream_documents(data, analysis, results, key):
    source = copy.deepcopy(data['utterances']); translated = copy.deepcopy(data['utterance_translations'])
    mapping = {r['utterance_id']: r for r in results}
    for doc in (source, translated):
        for row in doc['utterances']:
            r = mapping[row['id']]
            row.update(original_speaker=row['speaker'], speaker=r['speaker_id'],
                       attribution_status=r['status'], review_required=True,
                       attribution_key=key, attribution_method=r['method'])
            row['review_reasons'] = sorted(set(row.get('review_reasons', []) + r['automatic']['reasons']))
            if r['human_confirmation'] is not None: row['human_review'] = r['human_confirmation']
            for w in row.get('words', []):
                w.update(original_speaker=w.get('speaker'), speaker=r['speaker_id'], review_required=True,
                         assignment_method=r['method'], attribution_scope='utterance_audio_not_isolated_word')
        doc['cache_key'] = fingerprint(doc['utterances'])
    words = copy.deepcopy(data['word_speakers'])
    by_id = {w['word_id']: w for u in source['utterances'] for w in u['words']}
    for i, w in enumerate(words['words'], 1):
        if i in by_id:
            new = by_id[i]
            w.update(original_speaker=w.get('speaker'), speaker=new['speaker'],
                     assignment_method=new['assignment_method'], review_required=True,
                     attribution_scope='utterance_audio_not_isolated_word')
    words['cache_key'] = fingerprint(words['words'])
    timeline = {**copy.deepcopy(analysis), 'analysis_key': key, 'utterances_file': 'utterances.json',
                'attribution_file': 'speaker_attribution.json',
                'unknown_policy': 'Acoustic reassessment; human labels authoritative; mixed speech not merged.',
                'note': 'Original turn hypotheses retained, not verified simultaneous speech.'}
    regeneration = [{'utterance_id': r['utterance_id'], 'previous_speaker': r['previous_speaker'],
                     'new_speaker': r['speaker_id'], 'regeneration_required': r['speaker_id'] != r['previous_speaker'],
                     'block_synthesis': r['speaker_id'] in UNKNOWN or r['status'] == 'mixed',
                     'reason': 'Rebuild script/feedback on speaker change; never relabel cached WAV.'} for r in results]
    return {'utterances.json': source, 'utterance_translations.json': translated,
            'word_speakers.json': words, 'speaker_timeline.json': timeline,
            'selection.json': data['selection'], 'tts_regeneration.json': {'analysis_key': key, 'utterances': regeneration}}


def verify_bundle(folder):
    manifest = read_json(folder/'manifest.json')
    for name, digest in manifest['artifacts'].items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or sha256_file(path) != digest:
            raise ValueError('Analysis artifact changed: '+name)
    return manifest


def run_attribution(video_id, input_dir, analysis_dir, output_dir, backend, *, decisions_path=None, review_dir=None):
    inputs, parent, root = (Path(p).resolve() for p in (input_dir, analysis_dir, output_dir))
    data = validate_inputs(inputs, video_id); parent_manifest = verify_bundle(parent)
    analysis = read_json(parent/'diarization.enhanced.json'); profiles = read_json(parent/'speaker_profiles.json')
    if (analysis['video_id'] != video_id or analysis['clip'] != data['utterances']['clip']
            or analysis['input_fingerprints'] != {k: fingerprint(v) for k, v in data.items()}):
        raise ValueError('Phase 4 analysis does not match original inputs.')
    if analysis['model'] != backend.identity: raise ValueError('Embedding models differ; cached anchors cannot be compared.')
    if sha256_file(inputs/'source.mp4') != parent_manifest['identity']['source_video_sha256']:
        raise ValueError('Source video changed since profiling.')
    labels, decisions = verified_human_labels(review_dir or parent, decisions_path, data['utterances'], analysis)
    input_hashes = {p.name: sha256_file(p) for p in inputs.glob('*.json') if p.stem in data}
    identity = {'video_id': video_id, 'clip': analysis['clip'], 'settings': SETTINGS, 'model': backend.identity,
                'parent_manifest': fingerprint(parent_manifest), 'input_sha256': input_hashes,
                'human_decisions': fingerprint(decisions),
                'implementation': {'speaker_attribution.py':sha256_file(Path(__file__)),
                                   **{n+'.py':sha256_file(Path(sys.modules[n].__file__)) for n in
                                      ('speaker_review','speaker_refinement','voice_profiler','word_speaker_alignment','audio_mixer')}}}
    key = fingerprint(identity); root.mkdir(parents=True, exist_ok=True); target = root/key[:16]
    with (root/'.attribution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            manifest = verify_bundle(target)
            if manifest['identity'] != identity: raise ValueError('Attribution identity mismatch.')
            return target, True
        temp = Path(tempfile.mkdtemp(prefix='.pending-attribution-', dir=root))
        try:
            samples = read_audio(parent/'selected_source.wav')
            print('Reusing Phase 4 references; measuring frame activity...', flush=True)
            activity = backend.activity(samples)
            atomic_json(temp/'frame_activity.json', activity, overwrite=False)
            results = analyze_utterances(data['utterances']['utterances'], analysis, profiles, samples, activity, backend)
            apply_human_labels(results, data['utterances']['utterances'], labels)
            report = {'format_version': 1, 'kind': 'speaker_attribution', 'video_id': video_id,
                      'clip': analysis['clip'], 'analysis_key': key, 'provenance': identity, 'utterances': results}
            docs = downstream_documents(data, analysis, results, key)
            docs['speaker_attribution.json'] = report
            docs['evaluation.json'] = evaluate(results)
            docs['comparison.json'] = {'previous_unknown': sum(r['previous_speaker'] in UNKNOWN for r in results),
                                      'new_unknown': sum(r['status'] == 'unknown' for r in results),
                                      'new_mixed': sum(r['status'] == 'mixed' for r in results),
                                      'new_unassigned_total': sum(r['speaker_id'] in UNKNOWN for r in results),
                                      'automatic_assigned_ids': [r['utterance_id'] for r in results if r['automatic']['speaker_id'] is not None and r['previous_speaker'] in UNKNOWN],
                                      'changed_ids': [r['utterance_id'] for r in results if r['previous_speaker'] != r['speaker_id']],
                                      'overlap_hypotheses': dict(Counter(h for r in results for h in r['automatic']['overlap']['hypotheses'])),
                                      'accuracy_improvement_verified': False}
            docs['speaker_profiles.json'] = profiles
            for name, content in docs.items(): atomic_json(temp/name, content, overwrite=False)
            shutil.copytree(parent/'references', temp/'references')
            review_rows = copy.deepcopy(docs['utterances.json']['utterances'])
            for row, result in zip(review_rows, results):
                row.update(acoustic_candidates=result['windows'][0]['candidates'],
                           temporal_candidates=result['temporal_candidates'], attribution=result)
            build_review(temp, docs['speaker_timeline.json'], review_rows, profiles, samples, data['utterances'])
            dataset = []
            for result in results:
                human = result['human_confirmation']
                if human is None: continue
                path = temp/f"review_audio/utterance_{result['utterance_id']:03d}_exact.wav"
                dataset.append({'video_id': video_id, 'clip': analysis['clip'], 'analysis_key': key,
                                'utterance_id': result['utterance_id'], 'audio_sha256': sha256_file(path),
                                'audio_file': str(path.relative_to(temp)), 'start': result['start'], 'end': result['end'],
                                'human_label': human, 'automatic': result['automatic'],
                                'reference_exclusion': result['reference_windows'],
                                'split': 'evaluation_only_not_used_to_fit_thresholds'})
            (temp/'human_evaluation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in dataset))
            if any(sha256_file(inputs/n) != digest for n, digest in input_hashes.items()):
                raise ValueError('Original input changed during analysis.')
            if sha256_file(inputs/'source.mp4') != parent_manifest['identity']['source_video_sha256']:
                raise ValueError('Source video changed during analysis.')
            verify_bundle(parent)
            atomic_json(temp/'manifest.json', {'identity': identity, 'analysis_key': key,
                        'artifacts': {str(p.relative_to(temp)): sha256_file(p) for p in temp.rglob('*') if p.is_file()}}, overwrite=False)
            os.rename(temp, target)
        finally:
            if temp.exists(): shutil.rmtree(temp)
    return target, False


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline acoustic speaker reassessment with separate human labels.')
    parser.add_argument('video_id'); parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path); parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--decisions', type=Path); parser.add_argument('--review-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        path, reused = run_attribution(args.video_id, args.input_dir or BASE_DIR/'input'/args.video_id,
                                      args.analysis_dir, args.output_dir or BASE_DIR/'output'/args.video_id/'speaker_attribution',
                                      AcousticBackend(), decisions_path=args.decisions, review_dir=args.review_dir)
        print(('Reused: ' if reused else 'Saved: ')+str(path), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'Error: {error}', flush=True); return 1


if __name__ == '__main__': raise SystemExit(main())
