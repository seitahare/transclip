"""Bounded, same-speaker reflow from measured durations and Phase 13 generic timing sync.

Adapted from TRANS sync_group's start=max(source_start, previous_end+pause).
Only adjacent known-speaker groups may move. Other speakers and source overlap
are hard boundaries; unsafe proposals are retained as unresolved diagnostics.
"""
import math
import re


def overlap(a,b):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))


def bounded_placements(rows, *, seconds, max_shift=.5, minimum_pause=.04):
    for name,value,limit in [('seconds',seconds,None),('max_shift',max_shift,2),('minimum_pause',minimum_pause,.5)]:
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (limit is not None and value>limit):
            raise ValueError('Invalid timing '+name)
    if seconds<=0:raise ValueError('Positive timeline duration required.')
    ordered=sorted(rows,key=lambda r:(r['relative_start'],r['id']))
    if len({r['id'] for r in ordered})!=len(ordered):raise ValueError('Duplicate timing entry IDs.')
    starts={r['id']:r['relative_start'] for r in ordered if r.get('generation_status')=='complete'}
    for r in ordered:
        if not 0<=r['relative_start']<r['relative_end']:raise ValueError('Invalid source interval.')
        if r.get('generation_status')=='complete':
            d=r['actual_tts_duration_seconds']
            if type(d) not in (int,float) or not math.isfinite(d) or d<=0:raise ValueError('Measured duration required.')
            if r['speaker'] in (None,'UNKNOWN','MIXED','REVIEW'):raise ValueError('Unknown cannot receive generated placement.')
    # Source overlap is an anchor, including overlap with unresolved identities.
    anchored=set()
    for i,a in enumerate(ordered):
        for b in ordered[i+1:]:
            if a['speaker']!=b['speaker'] and overlap((a['relative_start'],a['relative_end']),(b['relative_start'],b['relative_end']))>0:
                anchored.update((a['id'],b['id']))
    groups=[]
    for r in ordered:
        if (groups and r.get('generation_status')=='complete' and groups[-1][-1].get('generation_status')=='complete'
                and r['speaker']==groups[-1][-1]['speaker'] and r['relative_start']-groups[-1][-1]['relative_end']<=.8):
            groups[-1].append(r)
        else:groups.append([r])
    events=[]
    for group in groups:
        if len(group)<2 or any(r.get('generation_status')!='complete' for r in group):continue
        proposed={};end=None
        for r in group:
            start=r['relative_start'] if end is None else max(r['relative_start'],end+minimum_pause)
            proposed[r['id']]=start;end=start+r['actual_tts_duration_seconds']
        changed=[r for r in group if proposed[r['id']]-r['relative_start']>1e-9]
        if not changed:continue
        reasons=[];ids={r['id'] for r in group}
        if ids & anchored:reasons.append('source_overlap_anchor')
        if any(proposed[r['id']]-r['relative_start']>max_shift+1e-9 for r in group):reasons.append('maximum_shift_exceeded')
        if end>seconds+1e-9:reasons.append('preview_boundary')
        for r in group:
            interval=(proposed[r['id']],proposed[r['id']]+r['actual_tts_duration_seconds'])
            for other in ordered:
                if other['id'] in ids:continue
                other_start=starts.get(other['id'],other['relative_start'])
                other_end=other_start+other['actual_tts_duration_seconds'] if other.get('generation_status')=='complete' else other['relative_end']
                if overlap(interval,(other_start,other_end))>1e-9:
                    reasons.append('intervening_or_other_speaker_boundary')
        if not reasons:starts.update(proposed)
        events.append({'entry_ids':[r['id'] for r in group],'status':'placed' if not reasons else 'unresolved',
                       'proposed_starts':proposed,'reasons':sorted(set(reasons)),
                       'max_shift_seconds':max(proposed[r['id']]-r['relative_start'] for r in group)})
    return {'format_version':1,'policy':'adjacent_same_speaker_bounded_delay','max_shift_seconds':max_shift,
            'minimum_pause_seconds':minimum_pause,'playback_rate':1.0,'starts':starts,'events':events,
            'source_overlap_anchors':sorted(anchored),'waveform_modified':False,
            'review_required':bool(events),'human_timing_verified':False}


def diagnose_collision_budgets(rows, *, seconds, max_shift=.5, minimum_pause=.04, default_mora_per_second=7.0):
    """Phase 9 helper: compute exact duration/mora reductions needed to resolve same-speaker collisions.

    Preserves natural cross-speaker overlaps and never truncates or speed-stretches audio.
    """
    if type(default_mora_per_second) not in (int,float) or not math.isfinite(default_mora_per_second) or default_mora_per_second<=0:
        raise ValueError('Invalid default_mora_per_second')
    placement=bounded_placements(rows,seconds=seconds,max_shift=max_shift,minimum_pause=minimum_pause)
    ordered=sorted(rows,key=lambda r:(r['relative_start'],r['id']))
    starts=placement['starts']
    anchored=set(placement['source_overlap_anchors'])

    by_speaker={}
    for r in ordered:
        if r.get('generation_status')=='complete':
            by_speaker.setdefault(r['speaker'],[]).append(r)

    budgets=[]
    unresolved_pairs=[]
    for speaker,s_rows in sorted(by_speaker.items()):
        for idx,curr in enumerate(s_rows):
            curr_start=starts[curr['id']]
            curr_dur=curr['actual_tts_duration_seconds']
            curr_end=curr_start+curr_dur
            next_same=s_rows[idx+1] if idx+1<len(s_rows) else None
            hard_ceiling=seconds
            barrier_reason='preview_boundary'
            if next_same is not None:
                # How far can next_same legally shift without violating max_shift or anchored source overlap?
                next_max_start=next_same['relative_start'] if (next_same['id'] in anchored or curr['id'] in anchored) else min(seconds,next_same['relative_start']+max_shift)
                # Also check if an intervening turn prevents shifting next_same.
                for other in ordered:
                    if other['id'] in (curr['id'],next_same['id']):continue
                    if curr['relative_end']<=other['relative_start']<=next_same['relative_start']:
                        next_max_start=next_same['relative_start']
                        break
                same_speaker_ceiling=max(curr_start+minimum_pause,next_max_start-minimum_pause)
                if same_speaker_ceiling<hard_ceiling:
                    hard_ceiling=same_speaker_ceiling
                    barrier_reason=f"same_speaker_next_entry:{next_same['id']}"
                actual_next_start=starts[next_same['id']]
                if curr_end>actual_next_start+1e-9:
                    unresolved_pairs.append({
                        'speaker':speaker,
                        'entry_ids':[curr['id'],next_same['id']],
                        'overlap_seconds':round(curr_end-actual_next_start,6),
                        'first_actual_duration_seconds':curr_dur,
                        'first_source_duration_seconds':round(curr['relative_end']-curr['relative_start'],6),
                    })
            max_allowable_duration=max(0.1,hard_ceiling-curr_start)
            reduction_needed=max(0.0,curr_dur-max_allowable_duration)
            pause_sec=float(curr.get('pause_seconds') or 0.0)
            mora_count=curr.get('mora_count')
            if mora_count is None and isinstance(curr.get('japanese_analysis'),dict):
                mora_count=curr['japanese_analysis'].get('mora_count')
            rate=float((curr.get('timing_config') or {}).get('mora_per_second') or default_mora_per_second)
            if mora_count is not None and curr_dur>pause_sec+0.1 and mora_count>0:
                # Use empirical speaking rate of this synthesized utterance for realistic mora target.
                rate=mora_count/(curr_dur-pause_sec)
            target_spoken=max(0.0,min(curr_dur,max_allowable_duration)-pause_sec)
            target_mora=max(1,math.floor(target_spoken*rate)) if reduction_needed>1e-9 else mora_count
            budgets.append({
                'id':curr['id'],
                'speaker':speaker,
                'source_duration_seconds':round(curr['relative_end']-curr['relative_start'],6),
                'actual_tts_duration_seconds':round(curr_dur,6),
                'placement_start_seconds':round(curr_start,6),
                'max_allowable_duration_seconds':round(max_allowable_duration,6),
                'required_reduction_seconds':round(reduction_needed,6),
                'barrier_reason':barrier_reason if reduction_needed>1e-9 else None,
                'current_mora_count':mora_count,
                'target_mora_budget':target_mora,
                'mora_reduction_needed':max(0,(mora_count-target_mora)) if (mora_count is not None and target_mora is not None) else None,
                'effective_mora_per_second':round(rate,3),
                'requires_script_shortening':reduction_needed>1e-9,
            })
    return {
        'kind':'same_speaker_collision_budget_plan',
        'format_version':1,
        'placement_plan':placement,
        'unresolved_same_speaker_collisions':unresolved_pairs,
        'entry_budgets':budgets,
        'requires_regeneration':bool(unresolved_pairs),
        'waveform_modified':False,
    }


# ============================================================================
# Phase 13: Generic Video-to-TTS Speech Timing Synchronization
# ============================================================================

def _estimate_unit_mora_weight(text):
    clean = str(text or '').strip()
    if not clean:
        return 1.0
    try:
        from language.pronunciation import count_mora
        m = count_mora(clean)
        if isinstance(m, int) and m > 0:
            base = float(m)
        else:
            base = float(len(re.sub(r'[\s、。！？!?,.…]', '', clean)))
    except Exception:
        base = float(max(1, len(re.sub(r'[\s、。！？!?,.…]', '', clean))))
    # Account for natural clause/sentence pauses
    comma_pauses = sum(1 for ch in clean if ch in '、，,') * 1.5
    period_pauses = sum(1 for ch in clean if ch in '。！？!?') * 2.5
    return max(1.0, base + comma_pauses + period_pauses)


def align_semantic_units_to_utterance_window(entry):
    """Map clauses / semantic sentence units of an utterance onto its source [start, end] time window."""
    start = float(entry.get('source_start', entry.get('start', entry.get('relative_start', 0.0))))
    end = float(entry.get('source_end', entry.get('end', entry.get('relative_end', start + 1.0))))
    if end <= start:
        end = start + 0.5
    duration = end - start

    sentence_units = entry.get('sentence_units')
    raw_units = []
    if isinstance(sentence_units, list) and sentence_units:
        for u in sentence_units:
            txt = u.get('text_ja') if isinstance(u, dict) else str(u)
            if txt and str(txt).strip():
                raw_units.append(str(txt).strip())

    if not raw_units:
        text_ja = str(entry.get('text_ja') or '').strip()
        if text_ja:
            try:
                from japanese_subtitles import split_japanese_clauses
                raw_units = split_japanese_clauses(text_ja, entry.get('japanese_analysis'))
            except Exception:
                raw_units = [m.group(0) for m in re.finditer(r'[^。！？!?、，]+[。！？!?、，]*', text_ja)] or [text_ja]

    if not raw_units:
        return []

    weights = [_estimate_unit_mora_weight(u) for u in raw_units]
    total_w = sum(weights)
    cursor = start
    aligned = []
    for idx, (txt, w) in enumerate(zip(raw_units, weights)):
        share = w / total_w if total_w > 0 else (1.0 / len(raw_units))
        u_end = end if idx == len(raw_units) - 1 else min(end, cursor + duration * share)
        aligned.append({
            'unit_index': idx,
            'text_ja': txt,
            'weight': round(w, 2),
            'share': round(share, 4),
            'target_start_seconds': round(cursor, 4),
            'target_end_seconds': round(u_end, 4),
            'target_duration_seconds': round(max(0.01, u_end - cursor), 4),
        })
        cursor = u_end
    return aligned


def compute_auto_speech_rate_ratio(
    actual_duration,
    target_slot_duration,
    *,
    min_atempo=0.90,
    max_atempo=1.30,
    deadband=0.03,
):
    """Compute bounded automatic speech rate adjustment ratio (default 0.90..1.30x) to fit target slot.

    - When Japanese TTS is longer than the speaker's utterance window: speeds up within [1.0, max_atempo].
    - When Japanese TTS finishes noticeably earlier than the speaker's window: slows down within [min_atempo, 1.0].
    - Prevents excessive compression/stretching by clamping strictly to [min_atempo, max_atempo].
    """
    for name, val in [('actual_duration', actual_duration), ('target_slot_duration', target_slot_duration),
                      ('min_atempo', min_atempo), ('max_atempo', max_atempo), ('deadband', deadband)]:
        if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
            raise ValueError(f'Invalid {name}: {val}')
    if not (0.5 <= min_atempo <= 1.0 <= max_atempo <= 2.0):
        raise ValueError(f'Invalid atempo bounds [{min_atempo}, {max_atempo}]; expected within [0.5, 2.0] straddling 1.0.')

    ideal_ratio = float(actual_duration) / float(target_slot_duration)
    if abs(ideal_ratio - 1.0) <= float(deadband):
        return {
            'applied_atempo': 1.0,
            'ideal_atempo': round(ideal_ratio, 4),
            'clamped': False,
            'direction': 'none',
            'expected_duration_seconds': round(float(actual_duration), 4),
            'prevented_excessive_stretch': False,
        }

    clamped = ideal_ratio < float(min_atempo) or ideal_ratio > float(max_atempo)
    applied = max(float(min_atempo), min(float(max_atempo), ideal_ratio))
    if abs(applied - 1.0) <= 0.012:
        applied = 1.0
        direction = 'none'
    elif applied > 1.0:
        direction = 'speedup'
    else:
        direction = 'slowdown'

    expected_dur = float(actual_duration) / applied
    return {
        'applied_atempo': round(applied, 4),
        'ideal_atempo': round(ideal_ratio, 4),
        'clamped': bool(clamped),
        'direction': direction,
        'expected_duration_seconds': round(expected_dur, 4),
        'prevented_excessive_stretch': bool(clamped),
    }


def detect_timing_and_coverage_anomalies(
    placed_entries,
    *,
    selection_start=0.0,
    selection_end=30.0,
    source_utterances=None,
    early_finish_threshold=0.45,
    long_silence_threshold=3.0,
    drift_warning_threshold=0.60,
):
    """Automatically detect timing drift, early TTS finishes, missing dubbing during source speech,
    long silences, and audio truncation for any input video.
    """
    sel_a = float(selection_start)
    sel_b = float(selection_end)
    clip_dur = max(0.1, sel_b - sel_a)

    early_finishes = []
    drift_records = []
    truncation_events = []
    tts_windows = []

    for e in placed_entries or []:
        uid = e.get('id', e.get('raw_id'))
        spk = e.get('speaker')
        src_a = float(e.get('source_start', e.get('source_start_seconds', e.get('start', sel_a))))
        src_b = float(e.get('source_end', e.get('source_end_seconds', e.get('end', src_a + 1.0))))
        if sel_a > 0.0 and src_a < sel_a - 1.0:
            src_a += sel_a
            src_b += sel_a
        slot_dur = max(0.05, src_b - src_a)

        place_a = float(e.get('placement_start', e.get('placement_start_seconds', e.get('placement_seconds', src_a))))
        if sel_a > 0.0 and place_a < sel_a - 1.0:
            place_a += sel_a
        act_dur = float(e.get('actual_duration', e.get('actual_tts_duration_seconds', slot_dur)))
        place_b = float(e.get('placement_end', e.get('placement_end_seconds', e.get('end_seconds', place_a + act_dur))))

        rel_a = max(0.0, place_a - sel_a)
        rel_b = max(rel_a, place_b - sel_a)
        tts_windows.append((rel_a, rel_b, uid, spk))

        start_drift = round(place_a - src_a, 4)
        end_drift = round(place_b - src_b, 4)
        early_gap = round(src_b - place_b, 4)
        fit_ratio = round(act_dur / slot_dur, 4)

        drift_rec = {
            'id': uid,
            'speaker': spk,
            'source_start_seconds': round(src_a, 4),
            'source_end_seconds': round(src_b, 4),
            'placement_start_seconds': round(place_a, 4),
            'placement_end_seconds': round(place_b, 4),
            'slot_duration_seconds': round(slot_dur, 4),
            'actual_tts_duration_seconds': round(act_dur, 4),
            'start_drift_seconds': start_drift,
            'end_drift_seconds': end_drift,
            'duration_fit_ratio': fit_ratio,
            'drift_exceeds_threshold': abs(start_drift) > drift_warning_threshold or abs(end_drift) > drift_warning_threshold * 1.5,
        }
        drift_records.append(drift_rec)

        if early_gap >= float(early_finish_threshold):
            early_finishes.append({
                'id': uid,
                'speaker': spk,
                'source_end_seconds': round(src_b, 4),
                'placement_end_seconds': round(place_b, 4),
                'early_finish_seconds': early_gap,
                'duration_fit_ratio': fit_ratio,
                'reason': 'tts_finishes_before_source_speech_ends',
            })

        placed_span = place_b - place_a
        if e.get('truncated_by_mixer') or placed_span < act_dur - 0.01 or place_b > sel_b + 0.02:
            truncation_events.append({
                'id': uid,
                'speaker': spk,
                'actual_tts_duration_seconds': round(act_dur, 4),
                'placed_span_seconds': round(placed_span, 4),
                'placement_end_seconds': round(place_b, 4),
                'selection_end_seconds': round(sel_b, 4),
                'reason': (
                    'truncated_by_mixer'
                    if e.get('truncated_by_mixer')
                    else ('exceeds_clip_boundary' if place_b > sel_b + 0.02 else 'placed_span_shorter_than_wav')
                ),
            })

    # Detect undubbed active source speech intervals
    undubbed_speech = []
    if source_utterances:
        for su in source_utterances:
            s_a = float(su.get('start', su.get('source_start', 0.0)))
            s_b = float(su.get('end', su.get('source_end', s_a)))
            if sel_a > 0.0 and s_a < sel_a - 1.0 and s_b <= clip_dur + 1.0:
                s_a += sel_a
                s_b += sel_a
            inter_a = max(sel_a, s_a)
            inter_b = min(sel_b, s_b)
            if inter_b - inter_a < 0.15:
                continue
            rel_su_a = inter_a - sel_a
            rel_su_b = inter_b - sel_a
            covered_sec = 0.0
            for w_a, w_b, _, _ in tts_windows:
                ov = max(0.0, min(rel_su_b, w_b) - max(rel_su_a, w_a))
                covered_sec = max(covered_sec, ov)
            inter_dur = rel_su_b - rel_su_a
            if covered_sec / max(0.05, inter_dur) < 0.25:
                undubbed_speech.append({
                    'source_id': su.get('id'),
                    'speaker': su.get('speaker'),
                    'source_start_seconds': round(s_a, 4),
                    'source_end_seconds': round(s_b, 4),
                    'clip_intersection_seconds': round(inter_dur, 4),
                    'covered_by_tts_seconds': round(covered_sec, 4),
                    'crosses_clip_boundary': s_b > sel_b + 0.01 or s_a < sel_a - 0.01,
                    'reason': 'source_speaker_talking_without_dubbed_audio',
                })

    # Detect long silence gaps in TTS timeline
    tts_windows.sort(key=lambda x: (x[0], x[1]))
    merged_tts = []
    for w_a, w_b, _, _ in tts_windows:
        if merged_tts and w_a <= merged_tts[-1][1] + 0.01:
            merged_tts[-1] = (merged_tts[-1][0], max(merged_tts[-1][1], w_b))
        else:
            merged_tts.append((w_a, w_b))

    long_silences = []
    cursor = 0.0
    for w_a, w_b in [*merged_tts, (clip_dur, clip_dur)]:
        gap = w_a - cursor
        if gap >= float(long_silence_threshold):
            abs_gap_a = sel_a + cursor
            abs_gap_b = sel_a + w_a
            has_source_speech = any(
                overlap((abs_gap_a, abs_gap_b), (u['source_start_seconds'], u['source_end_seconds'])) > 0.3
                for u in undubbed_speech
            )
            long_silences.append({
                'clip_relative_start_seconds': round(cursor, 4),
                'clip_relative_end_seconds': round(w_a, 4),
                'source_start_seconds': round(abs_gap_a, 4),
                'source_end_seconds': round(abs_gap_b, 4),
                'duration_seconds': round(gap, 4),
                'classification': 'missing_dub_during_source_speech' if has_source_speech else 'natural_pause_or_ambient_interval',
            })
        cursor = max(cursor, w_b)

    mean_start_drift = (
        round(sum(abs(r['start_drift_seconds']) for r in drift_records) / len(drift_records), 4)
        if drift_records else 0.0
    )
    mean_end_drift = (
        round(sum(abs(r['end_drift_seconds']) for r in drift_records) / len(drift_records), 4)
        if drift_records else 0.0
    )

    return {
        'early_tts_finishes': early_finishes,
        'undubbed_active_speech_intervals': undubbed_speech,
        'long_silence_gaps': long_silences,
        'audio_truncation_events': truncation_events,
        'lip_sync_drift_records': drift_records,
        'summary': {
            'utterance_count': len(drift_records),
            'mean_abs_start_drift_seconds': mean_start_drift,
            'mean_abs_end_drift_seconds': mean_end_drift,
            'early_finish_count': len(early_finishes),
            'undubbed_active_speech_count': len(undubbed_speech),
            'long_silence_count': len(long_silences),
            'truncation_count': len(truncation_events),
        },
    }
