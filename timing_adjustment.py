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


# ============================================================================
# Phase 13.1: Word-Level Timestamp Synchronization, Filler/Pause Handling &
#             Clip Boundary Auto-Detection / Extension
# ============================================================================

PURE_VOCAL_FILLERS = frozenset({
    'um', 'uh', 'umm', 'uhh', 'er', 'erm', 'ah', 'ahh', 'hmm', 'hm', 'mm', 'mhm', 'uh-huh', 'uhhuh',
})

DISCOURSE_STARTER_FILLERS = frozenset({
    'so', 'well', 'like', 'right', 'okay', 'ok', 'yeah', 'yes', 'oh', 'hey', 'look', 'listen',
    'anyway', 'basically', 'actually', 'literally', 'mean',
})

MULTIWORD_FILLERS = (
    ('you', 'know'),
    ('i', 'mean'),
    ('sort', 'of'),
    ('kind', 'of'),
    ("let's", 'see'),
)

HESITATION_STARTER_TOKENS = frozenset({
    'and', 'but', 'or', 'so', 'well', 'like', 'right', 'okay', 'ok', 'yeah', 'yes',
    'what', 'about', 'that', 'this', 'if', 'i', "i've", 'we', 'it',
})


def _clean_word_token(raw_word):
    return re.sub(r"^[^A-Za-z0-9']+|[^A-Za-z0-9']+$", '', str(raw_word or '').strip()).lower()


def extract_or_resolve_word_timestamps(entry, *, external_words=None):
    """Extract sorted word-level timestamps for `entry` from embedded words, merged spans, or external word list.

    Falls back to proportional syllable/punctuation estimation only when no word timestamps exist.
    """
    src_start = float(entry.get('source_start', entry.get('start', entry.get('relative_start', 0.0))))
    src_end = float(entry.get('source_end', entry.get('end', entry.get('relative_end', src_start + 1.0))))
    if src_end <= src_start:
        src_end = src_start + 0.5

    raw_words = []
    if isinstance(entry.get('words'), list) and entry['words']:
        raw_words = list(entry['words'])
    elif isinstance(entry.get('source_segment_spans'), list):
        for seg in entry['source_segment_spans']:
            if isinstance(seg.get('words'), list):
                raw_words.extend(seg['words'])

    if not raw_words and external_words:
        for w in external_words:
            w_s = float(w.get('start', 0.0))
            w_e = float(w.get('end', w_s))
            if w_e > src_start - 0.05 and w_s < src_end + 0.05:
                raw_words.append(w)

    normalized = []
    for idx, w in enumerate(raw_words):
        txt = str(w.get('word', w.get('text', ''))).strip()
        if not txt:
            continue
        w_s = float(w.get('start', src_start))
        w_e = float(w.get('end', max(w_s + 0.04, w_s)))
        if w_e < w_s:
            w_e = w_s + 0.04
        normalized.append({
            'word': txt,
            'clean_token': _clean_word_token(txt),
            'start': round(w_s, 4),
            'end': round(w_e, 4),
            'duration': round(max(0.01, w_e - w_s), 4),
            'speaker': w.get('speaker', entry.get('speaker')),
            'word_index': idx,
            'estimated': bool(w.get('estimated', False)),
        })

    if normalized:
        normalized.sort(key=lambda x: (x['start'], x['end'], x['word_index']))
        for i, item in enumerate(normalized):
            item['word_index'] = i
        return normalized

    # Fallback: estimate word timestamps from English text if no word list was provided
    text_en = str(entry.get('text_en', entry.get('text', '')) or '').strip()
    tokens = [t for t in re.split(r'\s+', text_en) if t]
    if not tokens:
        return []
    weights = []
    for tok in tokens:
        clean = _clean_word_token(tok)
        w = max(1.0, float(len(clean)))
        if tok.endswith((',', ';', ':')):
            w += 1.8
        elif tok.endswith(('.', '?', '!', '...')):
            w += 2.5
        weights.append(w)
    total_w = sum(weights) or 1.0
    span = src_end - src_start
    cursor = src_start
    for idx, (tok, w) in enumerate(zip(tokens, weights)):
        dur = span * (w / total_w)
        pause_after = min(0.18, dur * 0.25) if tok.endswith((',', ';', '...')) and idx < len(tokens) - 1 else 0.0
        w_e = src_end if idx == len(tokens) - 1 else min(src_end, cursor + max(0.04, dur - pause_after))
        normalized.append({
            'word': tok,
            'clean_token': _clean_word_token(tok),
            'start': round(cursor, 4),
            'end': round(w_e, 4),
            'duration': round(max(0.01, w_e - cursor), 4),
            'speaker': entry.get('speaker'),
            'word_index': idx,
            'estimated': True,
        })
        cursor = min(src_end, w_e + pause_after)
    return normalized


def estimate_japanese_mora_count(text):
    """Estimate mora count for a Japanese string."""
    return int(round(_estimate_unit_mora_weight(text)))


def classify_english_words_and_pauses(
    words,
    *,
    segment_start=None,
    segment_end=None,
    utterance_start=None,
    utterance_end=None,
    long_pause_threshold=0.55,
    filler_pause_threshold=0.18,
):
    """Distinguish English fillers (So, um, uh, Right, etc.), internal silence/pauses, and substantive content speech.

    Computes:
    - `effective_content_start`: start time of the first substantive content word (preventing premature TTS start on fillers/pauses).
    - `effective_content_end`: end time of the last substantive content word.
    - `internal_pauses`: inter-word gaps >= 0.20s, marking gaps >= `long_pause_threshold` as `is_long_pause`.
    - `speech_clusters`: contiguous word clusters separated by long pauses (>= `long_pause_threshold`).
    """
    if segment_start is None and utterance_start is not None:
        segment_start = utterance_start
    if segment_end is None and utterance_end is not None:
        segment_end = utterance_end

    if not words:
        s0 = float(segment_start or 0.0)
        e0 = float(segment_end if segment_end is not None else s0 + 1.0)
        return {
            'segment_start': round(s0, 4),
            'segment_end': round(e0, 4),
            'first_word_start': round(s0, 4),
            'last_word_end': round(e0, 4),
            'effective_content_start': round(s0, 4),
            'effective_content_end': round(e0, 4),
            'leading_silence_seconds': 0.0,
            'leading_filler_offset_seconds': 0.0,
            'leading_filler_skip_seconds': 0.0,
            'leading_filler_words': [],
            'leading_fillers': [],
            'content_words': [],
            'content_word_count': 0,
            'total_word_count': 0,
            'internal_pauses': [],
            'long_pauses': [],
            'long_pause_count': 0,
            'speech_clusters': [],
            'classified_words': [],
        }

    ordered = [dict(w) for w in words]
    ordered.sort(key=lambda x: (float(x['start']), float(x['end'])))
    seg_s = float(segment_start) if segment_start is not None else float(ordered[0]['start'])
    seg_e = float(segment_end) if segment_end is not None else float(ordered[-1]['end'])
    seg_s = min(seg_s, float(ordered[0]['start']))
    seg_e = max(seg_e, float(ordered[-1]['end']))

    n = len(ordered)
    for i, w in enumerate(ordered):
        w['clean_token'] = _clean_word_token(w.get('word', ''))
        prev_e = float(ordered[i - 1]['end']) if i > 0 else seg_s
        next_s = float(ordered[i + 1]['start']) if i + 1 < n else seg_e
        w['gap_before'] = round(max(0.0, float(w['start']) - prev_e), 4)
        w['gap_after'] = round(max(0.0, next_s - float(w['end'])), 4)
        w['role'] = 'content'
        w['filler_kind'] = None

    # 1. Mark pure vocal fillers anywhere ("um", "uh", "er", "hmm", etc.)
    for w in ordered:
        if w['clean_token'] in PURE_VOCAL_FILLERS:
            w['role'] = 'filler'
            w['filler_kind'] = 'pure_vocal_filler'

    # 2. Mark multi-word fillers ("you know", "i mean", "sort of", "kind of") at start or bounded by pauses/commas
    i = 0
    while i < n - 1:
        pair = (ordered[i]['clean_token'], ordered[i + 1]['clean_token'])
        if pair in MULTIWORD_FILLERS:
            at_leading_edge = all(ordered[k]['role'] == 'filler' for k in range(i))
            has_boundary = (
                at_leading_edge
                or ordered[i]['gap_before'] >= filler_pause_threshold
                or ordered[i + 1]['gap_after'] >= filler_pause_threshold
                or str(ordered[i + 1]['word']).endswith(',')
            )
            if has_boundary and n > 2:
                ordered[i]['role'] = 'filler'
                ordered[i]['filler_kind'] = 'multiword_filler'
                ordered[i + 1]['role'] = 'filler'
                ordered[i + 1]['filler_kind'] = 'multiword_filler'
                i += 2
                continue
        i += 1

    # 3. Mark leading discourse fillers ("So,", "Well,", "Right, so", "Yeah,", "Okay,") when at the start of an utterance
    for idx, w in enumerate(ordered):
        if w['role'] == 'filler':
            continue
        if not all(ordered[k]['role'] == 'filler' for k in range(idx)):
            break
        tok = w['clean_token']
        raw_txt = str(w.get('word', '')).strip()
        remaining_words = n - (idx + 1)
        if remaining_words < 2:
            break
        is_punctuated_starter = raw_txt.endswith((',', '...', '—', '--'))
        has_pause_after = w['gap_after'] >= filler_pause_threshold
        next_is_filler = (idx + 1 < n) and (
            ordered[idx + 1]['clean_token'] in PURE_VOCAL_FILLERS
            or ordered[idx + 1]['clean_token'] in DISCOURSE_STARTER_FILLERS
        )
        if tok in DISCOURSE_STARTER_FILLERS and (is_punctuated_starter or has_pause_after or next_is_filler or idx <= 1):
            w['role'] = 'filler'
            w['filler_kind'] = 'leading_discourse_filler'

    # 4. Build internal pauses and contiguous speech clusters separated by long pauses (>= long_pause_threshold)
    internal_pauses = []
    clusters = []
    curr_cluster = [ordered[0]]
    for idx in range(n - 1):
        curr_w = ordered[idx]
        next_w = ordered[idx + 1]
        gap = round(max(0.0, float(next_w['start']) - float(curr_w['end'])), 4)
        if gap >= 0.20:
            internal_pauses.append({
                'after_word_index': idx,
                'before_word_index': idx + 1,
                'after_word': curr_w['word'],
                'before_word': next_w['word'],
                'start': round(float(curr_w['end']), 4),
                'end': round(float(next_w['start']), 4),
                'duration': gap,
                'is_long_pause': bool(gap >= float(long_pause_threshold)),
            })
        if gap >= float(long_pause_threshold):
            clusters.append(curr_cluster)
            curr_cluster = [next_w]
        else:
            curr_cluster.append(next_w)
    if curr_cluster:
        clusters.append(curr_cluster)

    # 5. Detect short leading hesitation starters before a long pause
    #    Example: "But what about" [1849.36..1849.88] -> 1.14s pause -> "for physical jobs, for assembly jobs..." [1851.02..1859.42]
    if len(clusters) >= 2:
        c0 = clusters[0]
        rest_words = [w for c in clusters[1:] for w in c]
        c0_dur = float(c0[-1]['end']) - float(c0[0]['start'])
        c0_tokens_hesitation = all(
            (w['role'] == 'filler' or w['clean_token'] in HESITATION_STARTER_TOKENS)
            for w in c0
        )
        if len(c0) <= 3 and c0_dur <= 1.25 and len(rest_words) >= max(3, len(c0)) and c0_tokens_hesitation:
            for w in c0:
                if w['role'] != 'filler':
                    w['role'] = 'filler'
                    w['filler_kind'] = 'hesitation_starter_before_long_pause'

    # Summarize speech clusters
    cluster_summaries = []
    for c_idx, c_words in enumerate(clusters):
        c_content = [w for w in c_words if w['role'] == 'content']
        c_start = float(c_words[0]['start'])
        c_end = float(c_words[-1]['end'])
        c_cont_start = float(c_content[0]['start']) if c_content else c_start
        c_cont_end = float(c_content[-1]['end']) if c_content else c_end
        cluster_summaries.append({
            'cluster_index': c_idx,
            'start': round(c_start, 4),
            'end': round(c_end, 4),
            'content_start': round(c_cont_start, 4),
            'content_end': round(c_cont_end, 4),
            'duration': round(max(0.01, c_end - c_start), 4),
            'word_count': len(c_words),
            'content_word_count': len(c_content),
            'is_filler_only': len(c_content) == 0,
            'text_en': ' '.join(w['word'] for w in c_words),
            'words': c_words,
        })

    content_words = [w for w in ordered if w['role'] == 'content']
    leading_fillers = []
    for w in ordered:
        if w['role'] == 'filler':
            leading_fillers.append(w)
        else:
            break

    first_word_start = float(ordered[0]['start'])
    last_word_end = float(ordered[-1]['end'])
    if content_words:
        effective_content_start = float(content_words[0]['start'])
        effective_content_end = float(content_words[-1]['end'])
    else:
        effective_content_start = first_word_start
        effective_content_end = last_word_end

    leading_silence = max(0.0, first_word_start - seg_s)
    leading_filler_offset = max(0.0, effective_content_start - seg_s)
    long_pauses = [p for p in internal_pauses if p['is_long_pause']]
    leading_filler_list = [
        {'word': w['word'], 'start': w['start'], 'end': w['end'], 'filler_kind': w['filler_kind']}
        for w in leading_fillers
    ]

    return {
        'segment_start': round(seg_s, 4),
        'segment_end': round(seg_e, 4),
        'first_word_start': round(first_word_start, 4),
        'last_word_end': round(last_word_end, 4),
        'effective_content_start': round(effective_content_start, 4),
        'effective_content_end': round(effective_content_end, 4),
        'leading_silence_seconds': round(leading_silence, 4),
        'leading_filler_offset_seconds': round(leading_filler_offset, 4),
        'leading_filler_skip_seconds': round(leading_filler_offset, 4),
        'leading_filler_words': leading_filler_list,
        'leading_fillers': leading_filler_list,
        'content_words': content_words,
        'content_word_count': len(content_words),
        'total_word_count': len(ordered),
        'internal_pauses': internal_pauses,
        'long_pauses': long_pauses,
        'long_pause_count': len(long_pauses),
        'speech_clusters': cluster_summaries,
        'classified_words': ordered,
    }


def split_utterance_on_long_pauses_and_align_clauses(
    entry,
    *,
    external_words=None,
    long_pause_threshold=0.55,
):
    """Analyze word-level timestamps of `entry` (including merged utterances) and map Japanese clauses
    onto speech clusters separated by long pauses, preserving internal pause windows.
    """
    words = extract_or_resolve_word_timestamps(entry, external_words=external_words)
    src_start = float(entry.get('source_start', entry.get('start', entry.get('relative_start', 0.0))))
    src_end = float(entry.get('source_end', entry.get('end', entry.get('relative_end', src_start + 1.0))))
    wt_analysis = classify_english_words_and_pauses(
        words,
        segment_start=src_start,
        segment_end=src_end,
        long_pause_threshold=long_pause_threshold,
    )

    sentence_units = entry.get('sentence_units')
    raw_units = []
    if isinstance(sentence_units, list) and sentence_units:
        for u in sentence_units:
            txt = u.get('text_ja') if isinstance(u, dict) else str(u)
            if txt and str(txt).strip():
                raw_units.append(str(txt).strip())

    if not raw_units:
        text_ja = str(entry.get('text_ja', entry.get('translated', entry.get('translated_text', ''))) or '').strip()
        if text_ja:
            try:
                from japanese_subtitles import split_japanese_clauses
                raw_units = split_japanese_clauses(text_ja, entry.get('japanese_analysis'))
            except Exception:
                raw_units = [m.group(0) for m in re.finditer(r'[^。！？!?、，]+[。！？!?、，]*', text_ja)] or [text_ja]

    substantive_clusters = [c for c in wt_analysis['speech_clusters'] if not c['is_filler_only']]
    if not substantive_clusters:
        substantive_clusters = wt_analysis['speech_clusters']

    eff_start = wt_analysis['effective_content_start']
    eff_end = max(eff_start + 0.1, wt_analysis['effective_content_end'])

    if not raw_units:
        return {
            'word_timing_analysis': wt_analysis,
            'timed_semantic_units': [],
            'split_across_long_pauses': False,
        }

    weights = [_estimate_unit_mora_weight(u) for u in raw_units]
    total_w = sum(weights) or 1.0

    timed_units = []
    if len(substantive_clusters) >= 2 and len(raw_units) >= 2:
        cluster_weights = [max(1.0, float(c['content_word_count'] or c['word_count'])) for c in substantive_clusters]
        total_cw = sum(cluster_weights) or 1.0
        cum_u = 0.0
        cluster_to_units = {i: [] for i in range(len(substantive_clusters))}
        cum_c_bounds = []
        acc_c = 0.0
        for cw in cluster_weights:
            acc_c += cw / total_cw
            cum_c_bounds.append(acc_c)

        for u_idx, (u_txt, u_w) in enumerate(zip(raw_units, weights)):
            mid_frac = (cum_u + 0.5 * u_w) / total_w
            cum_u += u_w
            target_c = len(substantive_clusters) - 1
            for c_i, bound in enumerate(cum_c_bounds):
                if mid_frac <= bound + 1e-6:
                    target_c = c_i
                    break
            cluster_to_units[target_c].append((u_idx, u_txt, u_w))

        non_empty_clusters = [c_i for c_i, items in cluster_to_units.items() if items]
        if len(non_empty_clusters) == 1 and len(substantive_clusters) >= 2 and len(raw_units) >= 2:
            cluster_to_units[0] = [(0, raw_units[0], weights[0])]
            cluster_to_units[1] = [(i, raw_units[i], weights[i]) for i in range(1, len(raw_units))]

        for c_i, c_info in enumerate(substantive_clusters):
            assigned = cluster_to_units.get(c_i, [])
            if not assigned:
                continue
            c_s = float(c_info['content_start'])
            c_e = max(c_s + 0.08, float(c_info['content_end']))
            c_dur = c_e - c_s
            c_w_sum = sum(item[2] for item in assigned) or 1.0
            cur = c_s
            for local_i, (u_idx, u_txt, u_w) in enumerate(assigned):
                sh = u_w / c_w_sum
                u_end = c_e if local_i == len(assigned) - 1 else min(c_e, cur + c_dur * sh)
                timed_units.append({
                    'unit_index': u_idx,
                    'cluster_index': c_i,
                    'text_ja': u_txt,
                    'weight': round(u_w, 2),
                    'share': round(u_w / total_w, 4),
                    'start': round(cur, 4),
                    'end': round(u_end, 4),
                    'target_start_seconds': round(cur, 4),
                    'target_end_seconds': round(u_end, 4),
                    'target_duration_seconds': round(max(0.01, u_end - cur), 4),
                    'anchored_to_cluster': True,
                })
                cur = u_end
        split_across_pauses = True
    else:
        dur = eff_end - eff_start
        cur = eff_start
        for u_idx, (u_txt, u_w) in enumerate(zip(raw_units, weights)):
            sh = u_w / total_w
            u_end = eff_end if u_idx == len(raw_units) - 1 else min(eff_end, cur + dur * sh)
            timed_units.append({
                'unit_index': u_idx,
                'cluster_index': 0,
                'text_ja': u_txt,
                'weight': round(u_w, 2),
                'share': round(sh, 4),
                'start': round(cur, 4),
                'end': round(u_end, 4),
                'target_start_seconds': round(cur, 4),
                'target_end_seconds': round(u_end, 4),
                'target_duration_seconds': round(max(0.01, u_end - cur), 4),
                'anchored_to_cluster': False,
            })
            cur = u_end
        split_across_pauses = False

    return {
        'word_timing_analysis': wt_analysis,
        'timed_semantic_units': timed_units,
        'split_across_long_pauses': split_across_pauses,
    }


def compute_word_anchored_placement(
    entry=None,
    *,
    actual_tts_duration=None,
    tts_duration=None,
    raw_start=None,
    raw_end=None,
    effective_content_start=None,
    effective_content_end=None,
    prev_placed_end=None,
    next_anchor_start=None,
    next_barrier=None,
    selection_start=0.0,
    selection_end=30.0,
    sync_anchor='adaptive',
    min_pause=0.05,
    min_atempo=0.90,
    max_atempo=1.30,
    external_words=None,
    long_pause_threshold=0.55,
):
    """Compute word-timestamp-anchored placement and rate adjustment for an utterance."""
    if sync_anchor not in ('content_start', 'end', 'adaptive', 'start'):
        raise ValueError(f"Unsupported sync_anchor: {sync_anchor!r}")
    if actual_tts_duration is None and tts_duration is not None:
        actual_tts_duration = tts_duration
    if next_anchor_start is None and next_barrier is not None:
        next_anchor_start = next_barrier

    if entry is None:
        r_s = float(raw_start if raw_start is not None else selection_start)
        r_e = float(raw_end if raw_end is not None else r_s + 2.0)
        e_s = float(effective_content_start if effective_content_start is not None else r_s)
        e_e = float(effective_content_end if effective_content_end is not None else r_e)
        entry = {
            'start': r_s,
            'end': r_e,
            'source_start': r_s,
            'source_end': r_e,
            'effective_content_start': e_s,
            'effective_content_end': e_e,
            'words': [
                {'word': 'filler', 'start': r_s, 'end': e_s} if e_s > r_s + 0.01 else {'word': 'start', 'start': e_s, 'end': min(e_e, e_s + 0.2)},
                {'word': 'content', 'start': e_s, 'end': e_e},
            ],
        }

    clause_sync = split_utterance_on_long_pauses_and_align_clauses(
        entry,
        external_words=external_words,
        long_pause_threshold=long_pause_threshold,
    )
    wt = clause_sync['word_timing_analysis']
    r_start = float(raw_start if raw_start is not None else entry.get('source_start', entry.get('start', entry.get('relative_start', selection_start))))
    r_end = float(raw_end if raw_end is not None else entry.get('source_end', entry.get('end', entry.get('relative_end', r_start + 1.0))))
    if float(selection_start) > 0.0 and r_start < float(selection_start) - 1.0:
        r_start += float(selection_start)
        r_end += float(selection_start)

    eff_start = float(effective_content_start if effective_content_start is not None else entry.get('effective_content_start', wt['effective_content_start']))
    eff_end = float(effective_content_end if effective_content_end is not None else entry.get('effective_content_end', wt['effective_content_end']))
    if float(selection_start) > 0.0 and eff_start < float(selection_start) - 1.0:
        eff_start += float(selection_start)
        eff_end += float(selection_start)

    eff_start = max(r_start, eff_start)
    eff_end = max(eff_start + 0.10, eff_end)
    content_slot_dur = max(0.10, eff_end - eff_start)

    base_anchor = r_start if sync_anchor == 'start' else eff_start
    if prev_placed_end is not None:
        earliest_legal_start = max(base_anchor, float(prev_placed_end) + float(min_pause))
    else:
        earliest_legal_start = base_anchor

    hard_ceiling = max(float(selection_end), r_end + 5.0) if selection_end == 30.0 and r_end > 30.0 else float(selection_end)
    if next_anchor_start is not None:
        hard_ceiling = min(hard_ceiling, max(earliest_legal_start + 0.15, float(next_anchor_start) - float(min_pause)))

    available_window = max(0.15, hard_ceiling - earliest_legal_start)
    target_slot_for_rate = min(content_slot_dur, available_window) if float(actual_tts_duration) > available_window else content_slot_dur

    rate_info = compute_auto_speech_rate_ratio(
        float(actual_tts_duration),
        max(0.15, target_slot_for_rate),
        min_atempo=min_atempo,
        max_atempo=max_atempo,
    )
    if sync_anchor == 'end' and float(actual_tts_duration) <= content_slot_dur:
        expected_dur = float(actual_tts_duration)
    else:
        expected_dur = float(rate_info['expected_duration_seconds'])

    if sync_anchor == 'end':
        backcalc_start = max(earliest_legal_start, eff_end - expected_dur)
        chosen_start = min(backcalc_start, max(earliest_legal_start, hard_ceiling - expected_dur))
        anchor_mode_used = 'end_backcalculated'
    elif sync_anchor == 'adaptive':
        slack = eff_end - (earliest_legal_start + expected_dur)
        has_leading_hesitation = wt['leading_filler_offset_seconds'] >= 0.15 or wt['long_pause_count'] > 0
        if slack > 0.25 and has_leading_hesitation:
            shift = min(slack * 0.65, 1.20)
            chosen_start = min(earliest_legal_start + shift, max(earliest_legal_start, hard_ceiling - expected_dur))
            anchor_mode_used = 'adaptive_end_weighted'
        else:
            chosen_start = earliest_legal_start
            anchor_mode_used = 'adaptive_content_start'
    elif sync_anchor == 'content_start':
        chosen_start = earliest_legal_start
        anchor_mode_used = 'content_start'
    else:
        chosen_start = earliest_legal_start
        anchor_mode_used = 'raw_start'

    chosen_end = chosen_start + expected_dur
    exceeds_available_window = chosen_end > hard_ceiling + 0.02
    mora_count = int(entry.get('mora_count') or _estimate_unit_mora_weight(entry.get('text_ja', '')))
    empirical_mps = mora_count / max(0.1, float(actual_tts_duration))
    max_feasible_source_dur = available_window * float(max_atempo)
    target_mora_budget = max(1, int(math.floor(max_feasible_source_dur * empirical_mps * 0.92))) if exceeds_available_window else mora_count

    return {
        'raw_source_start_seconds': round(r_start, 4),
        'raw_source_end_seconds': round(r_end, 4),
        'effective_content_start_seconds': round(eff_start, 4),
        'effective_content_end_seconds': round(eff_end, 4),
        'content_slot_duration_seconds': round(content_slot_dur, 4),
        'leading_filler_offset_seconds': wt['leading_filler_offset_seconds'],
        'leading_filler_words': wt['leading_filler_words'],
        'long_pause_count': wt['long_pause_count'],
        'placement_start_seconds': round(chosen_start, 4),
        'placement_end_seconds': round(chosen_end, 4),
        'placed_start': round(chosen_start, 4),
        'placed_end': round(chosen_end, 4),
        'premature_start_prevented': bool(chosen_start >= eff_start - 1e-4),
        'premature_start_prevented_seconds': round(max(0.0, chosen_start - r_start), 4),
        'starts_before_content': bool(chosen_start < eff_start - 1e-4),
        'sync_anchor_requested': sync_anchor,
        'sync_anchor_used': anchor_mode_used,
        'rate_adjustment': rate_info,
        'available_window_seconds': round(available_window, 4),
        'hard_ceiling_seconds': round(hard_ceiling, 4),
        'requires_rebalance': bool(exceeds_available_window),
        'target_mora_budget': target_mora_budget,
        'word_timing_analysis': wt,
        'timed_semantic_units': clause_sync['timed_semantic_units'],
        'split_across_long_pauses': clause_sync['split_across_long_pauses'],
    }


_CONCISE_REWRITE_PATTERNS = (
    (r'^つまり、?', ''),
    (r'基本的には、?', ''),
    (r'というふうに思います', 'と思います'),
    (r'というふうに考えています', 'と考えています'),
    (r'ということになります', 'です'),
    (r'ということになるでしょう', 'になるでしょう'),
    (r'という状況になっています', 'になっています'),
    (r'のではないかと不安に思っているんだと思います', 'のではと心配しています'),
    (r'のではないかと思います', 'と思います'),
    (r'なぜなら、?', ''),
    (r'というのも、?', ''),
    (r'文字どおり、?', ''),
    (r'実際のところ、?', '実際、'),
    (r'いずれにせよ、?', ''),
    (r'しかしながら、?', 'しかし、'),
    (r'それに加えて、?', 'また、'),
    (r'ある意味では、?', 'ある意味、'),
    (r'〜についてはどうですか', 'はどうですか'),
    (r'じゃあ、〜についてはどうですか[？?]?\s*', ''),
)


def concise_rewrite_for_timing_budget(text_ja, *, target_mora_budget=None):
    """Tighten verbose conversational Japanese expressions while strictly preserving numbers, proper nouns, and meaning."""
    original = str(text_ja or '').strip()
    if not original:
        return {
            'original_text_ja': original,
            'rewritten_text_ja': original,
            'rewritten_text': original,
            'changed': False,
            'original_mora_estimate': 0,
            'rewritten_mora_estimate': 0,
            'original_mora_count': 0,
            'rewritten_mora_count': 0,
        }

    rewritten = original
    applied_rules = []
    for pattern, repl in _CONCISE_REWRITE_PATTERNS:
        next_txt = re.sub(pattern, repl, rewritten)
        if next_txt != rewritten:
            applied_rules.append(pattern)
            rewritten = next_txt

    rewritten = re.sub(r'、、+', '、', rewritten)
    rewritten = re.sub(r'^、+', '', rewritten).strip()
    if not rewritten:
        rewritten = original

    orig_nums = re.findall(r'\d+(?:\.\d+)?', original)
    new_nums = re.findall(r'\d+(?:\.\d+)?', rewritten)
    if orig_nums != new_nums:
        rewritten = original
        applied_rules = []

    orig_mora = int(_estimate_unit_mora_weight(original))
    new_mora = int(_estimate_unit_mora_weight(rewritten))
    return {
        'original_text_ja': original,
        'rewritten_text_ja': rewritten,
        'rewritten_text': rewritten,
        'changed': rewritten != original,
        'applied_rules': applied_rules,
        'original_mora_estimate': orig_mora,
        'rewritten_mora_estimate': new_mora,
        'original_mora_count': orig_mora,
        'rewritten_mora_count': new_mora,
        'target_mora_budget': target_mora_budget,
    }


def detect_and_resolve_clip_boundaries(
    utterances,
    *,
    selection_start=None,
    selection_end=None,
    clip_start=None,
    clip_end=None,
    source_video_duration=None,
    auto_extend_clip=False,
    max_extension_seconds=12.0,
    tail_pad_seconds=0.35,
    head_pad_seconds=0.15,
    boundary_padding_seconds=None,
    fade_out_seconds=0.35,
):
    """Detect whether `selection_start` or `selection_end` cuts an utterance or sentence mid-way,
    and optionally extend the clip boundaries within `[0, source_video_duration]` so the sentence completes cleanly.
    """
    if selection_start is None and clip_start is not None:
        selection_start = clip_start
    if selection_end is None and clip_end is not None:
        selection_end = clip_end
    if boundary_padding_seconds is not None:
        tail_pad_seconds = float(boundary_padding_seconds)

    sel_s = float(selection_start)
    sel_e = float(selection_end)
    if sel_e <= sel_s:
        raise ValueError(f'Invalid selection range [{sel_s}, {sel_e}]')

    ordered = sorted(
        [dict(u) for u in (utterances or [])],
        key=lambda u: (float(u.get('start', u.get('source_start', 0.0))), float(u.get('end', u.get('source_end', 0.0)))),
    )

    start_cut_utterances = []
    end_cut_utterances = []
    continuation_utterances = []

    proposed_start = sel_s
    proposed_end = sel_e

    for idx, u in enumerate(ordered):
        u_s = float(u.get('start', u.get('source_start', 0.0)))
        u_e = float(u.get('end', u.get('source_end', u_s)))
        uid = u.get('id', u.get('utterance_id', idx + 1))
        txt_en = str(u.get('text_en', u.get('text', '')) or '').strip()

        if u_s < sel_s - 0.02 and u_e > sel_s + 0.05:
            start_cut_utterances.append({
                'id': uid,
                'utterance_id': uid,
                'speaker': u.get('speaker'),
                'start': round(u_s, 4),
                'end': round(u_e, 4),
                'text_en': txt_en,
                'reason': 'utterance_crosses_clip_start',
            })
            if sel_s - u_s <= float(max_extension_seconds):
                proposed_start = min(proposed_start, max(0.0, u_s - float(head_pad_seconds)))

        if u_s < sel_e - 0.05 and u_e > sel_e + 0.02:
            end_cut_utterances.append({
                'id': uid,
                'utterance_id': uid,
                'speaker': u.get('speaker'),
                'start': round(u_s, 4),
                'end': round(u_e, 4),
                'text_en': txt_en,
                'reason': 'utterance_crosses_clip_end',
            })
            if (u_e - sel_e) <= float(max_extension_seconds):
                proposed_end = max(proposed_end, u_e + float(tail_pad_seconds))

    inside = [
        u for u in ordered
        if float(u.get('start', u.get('source_start', 0.0))) < proposed_end - 0.02
        and float(u.get('end', u.get('source_end', 0.0))) > proposed_start + 0.02
    ]
    if inside:
        last_u = inside[-1]
        last_idx = ordered.index(last_u)
        last_en = str(last_u.get('text_en', last_u.get('text', '')) or '').strip()
        cursor_end = float(last_u.get('end', last_u.get('source_end', proposed_end)))
        curr_txt = last_en
        j = last_idx + 1
        while j < len(ordered) and not re.search(r'[.?!]["\')\]]*$', curr_txt):
            nxt_u = ordered[j]
            nxt_s = float(nxt_u.get('start', nxt_u.get('source_start', cursor_end)))
            nxt_e = float(nxt_u.get('end', nxt_u.get('source_end', nxt_s)))
            if nxt_s - cursor_end > 1.8 or (nxt_e - sel_e) > float(max_extension_seconds):
                break
            nxt_uid = nxt_u.get('id', nxt_u.get('utterance_id'))
            continuation_utterances.append({
                'id': nxt_uid,
                'utterance_id': nxt_uid,
                'speaker': nxt_u.get('speaker'),
                'start': round(nxt_s, 4),
                'end': round(nxt_e, 4),
                'text_en': str(nxt_u.get('text_en', nxt_u.get('text', '')) or '').strip(),
                'reason': 'completes_unfinished_sentence_at_clip_end',
            })
            proposed_end = max(proposed_end, nxt_e + float(tail_pad_seconds))
            cursor_end = nxt_e
            curr_txt = str(nxt_u.get('text_en', nxt_u.get('text', '')) or '').strip()
            j += 1

    if source_video_duration is not None and float(source_video_duration) > 0:
        proposed_end = min(proposed_end, float(source_video_duration))
    proposed_start = max(0.0, proposed_start)

    has_boundary_cut = bool(start_cut_utterances or end_cut_utterances or continuation_utterances)
    extension_applied = bool(auto_extend_clip and has_boundary_cut and (proposed_end > sel_e + 1e-4 or proposed_start < sel_s - 1e-4))
    effective_start = round(proposed_start if extension_applied else sel_s, 4)
    effective_end = round(proposed_end if extension_applied else sel_e, 4)

    end_cuts_all = end_cut_utterances + continuation_utterances
    untranslated = [] if extension_applied else (start_cut_utterances + end_cuts_all)

    return {
        'original_selection': {'start': round(sel_s, 4), 'end': round(sel_e, 4), 'duration': round(sel_e - sel_s, 4)},
        'proposed_selection': {
            'start': round(proposed_start, 4),
            'end': round(proposed_end, 4),
            'duration': round(proposed_end - proposed_start, 4),
            'start_extension_seconds': round(max(0.0, sel_s - proposed_start), 4),
            'end_extension_seconds': round(max(0.0, proposed_end - sel_e), 4),
        },
        'effective_selection': {
            'start': effective_start,
            'end': effective_end,
            'duration': round(effective_end - effective_start, 4),
        },
        'effective_clip_start': effective_start,
        'effective_clip_end': effective_end,
        'clip_was_extended': extension_applied,
        'has_mid_utterance_cut': has_boundary_cut,
        'auto_extend_enabled': bool(auto_extend_clip),
        'extension_applied': extension_applied,
        'start_cut_utterances': start_cut_utterances,
        'end_cut_utterances': end_cut_utterances,
        'sentence_continuation_utterances': continuation_utterances,
        'start_boundary': {
            'cut_detected': bool(start_cut_utterances),
            'reason': start_cut_utterances[0]['reason'] if start_cut_utterances else 'clean',
            'utterances': start_cut_utterances,
        },
        'end_boundary': {
            'cut_detected': bool(end_cuts_all),
            'reason': end_cuts_all[0]['reason'] if end_cuts_all else 'clean',
            'utterances': end_cuts_all,
        },
        'untranslated_boundary_utterances': untranslated,
        'fade_out_seconds': float(fade_out_seconds),
        'unextended_boundary_warning': bool(has_boundary_cut and not extension_applied),
    }


# ============================================================================
# Phase 13.2: Natural Speaking Rate (Slowdown Prohibited by Default),
#             Ordered Overlong TTS Resolution, Intra-Speaker Tempo Smoothing &
#             Time-Stretch Degradation Diagnostics
# ============================================================================

DEFAULT_NORMAL_ATEMPO = 1.00
DEFAULT_MAX_ATEMPO_PHASE13_2 = 1.20
HARD_MAX_ATEMPO_PHASE13_2 = 1.25
MIN_EXCEPTIONAL_SLOWDOWN_ATEMPO = 0.95


def compute_phase13_2_speech_rate_ratio(
    actual_duration,
    target_slot_duration,
    *,
    available_barrier_duration=None,
    allow_slowdown=False,
    min_atempo=1.00,
    exceptional_min_atempo=MIN_EXCEPTIONAL_SLOWDOWN_ATEMPO,
    max_atempo=DEFAULT_MAX_ATEMPO_PHASE13_2,
    hard_max_atempo=HARD_MAX_ATEMPO_PHASE13_2,
    deadband=0.02,
):
    """Phase 13.2 natural speech rate policy:
    - Normal playback: `1.00x`.
    - Allowed speedup: default `<= 1.20x`, configurable up to `<= 1.25x` (`hard_max_atempo`).
    - Automatic slowdown: PROHIBITED by default (`allow_slowdown=False`).
      When generated Japanese TTS is shorter than `target_slot_duration`, plays at `1.00x`
      and leaves the remaining duration as a natural pause (`trailing_pause_seconds`).
    - Exceptional slowdown: only when `allow_slowdown=True`, and strictly clamped to `>= 0.95x`.
    """
    for name, val in [
        ('actual_duration', actual_duration),
        ('target_slot_duration', target_slot_duration),
        ('min_atempo', min_atempo),
        ('max_atempo', max_atempo),
        ('hard_max_atempo', hard_max_atempo),
    ]:
        if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
            raise ValueError(f'Invalid {name}: {val}')

    if float(max_atempo) < 1.0 - 1e-6:
        raise ValueError(f'max_atempo={max_atempo} must be >= 1.0.')

    min_slowdown_floor = max(MIN_EXCEPTIONAL_SLOWDOWN_ATEMPO, float(exceptional_min_atempo))
    if allow_slowdown:
        effective_min = min(1.0, max(min_slowdown_floor, float(min_atempo) if float(min_atempo) < 1.0 else min_slowdown_floor))
    else:
        effective_min = 1.00

    effective_max = min(float(hard_max_atempo), max(1.00, float(max_atempo)))
    act_d = float(actual_duration)
    slot_d = float(target_slot_duration)
    avail_d = float(available_barrier_duration) if available_barrier_duration is not None else slot_d
    budget_d = max(0.15, min(slot_d, avail_d) if avail_d < slot_d else avail_d)

    ideal_ratio = act_d / max(0.10, slot_d)
    barrier_ratio = act_d / budget_d

    if act_d <= budget_d + float(deadband):
        if allow_slowdown and ideal_ratio < 1.0 - float(deadband):
            applied = max(effective_min, ideal_ratio)
            direction = 'exceptional_slowdown' if applied < 0.995 else 'none'
            if direction == 'none':
                applied = 1.0
        else:
            applied = 1.0
            direction = 'none'
        exp_dur = act_d / applied
        trailing_pause = max(0.0, slot_d - exp_dur)
        return {
            'applied_atempo': round(applied, 4),
            'ideal_atempo': round(ideal_ratio, 4),
            'barrier_atempo': round(barrier_ratio, 4),
            'direction': direction,
            'slowdown_applied': bool(applied < 0.999),
            'slowdown_prohibited_by_default': not bool(allow_slowdown),
            'preserved_natural_speed_with_pause': bool(act_d < slot_d and applied == 1.0),
            'trailing_pause_seconds': round(trailing_pause, 4),
            'natural_trailing_pause_seconds': round(trailing_pause, 4),
            'expected_duration_seconds': round(exp_dur, 4),
            'clamped': bool(allow_slowdown and ideal_ratio < effective_min),
            'exceeds_max_speedup': False,
            'exceeded_max_atempo': False,
            'requires_structural_resolution': False,
        }

    # TTS is longer than budget_d -> apply bounded speedup within [1.0, effective_max]
    applied = min(effective_max, max(1.0, barrier_ratio))
    if abs(applied - 1.0) <= 0.01:
        applied = 1.0
        direction = 'none'
    else:
        direction = 'speedup'
    exp_dur = act_d / applied
    still_overflows = bool(exp_dur > budget_d + 0.03 or barrier_ratio > effective_max + 1e-4)
    trailing_p = round(max(0.0, slot_d - exp_dur), 4)
    return {
        'applied_atempo': round(applied, 4),
        'ideal_atempo': round(ideal_ratio, 4),
        'barrier_atempo': round(barrier_ratio, 4),
        'direction': direction,
        'slowdown_applied': False,
        'slowdown_prohibited_by_default': not bool(allow_slowdown),
        'preserved_natural_speed_with_pause': False,
        'trailing_pause_seconds': trailing_p,
        'natural_trailing_pause_seconds': trailing_p,
        'expected_duration_seconds': round(exp_dur, 4),
        'clamped': bool(barrier_ratio > effective_max + 1e-4),
        'exceeds_max_speedup': still_overflows,
        'exceeded_max_atempo': still_overflows,
        'requires_structural_resolution': still_overflows,
    }


def plan_overlong_tts_resolution(
    entry,
    *,
    actual_tts_duration=None,
    raw_tts_duration=None,
    trimmed_tts_duration=None,
    slot_duration=1.0,
    available_window_duration=None,
    downstream_pause_seconds=0.0,
    max_atempo=DEFAULT_MAX_ATEMPO_PHASE13_2,
    max_timing_shift=0.8,
    max_shift_seconds=None,
):
    """Execute Requirement 2.3 ordered resolution when Japanese TTS exceeds its utterance window."""
    text_ja = str(entry.get('text_ja', entry.get('translated_text', entry.get('translated', ''))) or '').strip()
    act_d = float(
        trimmed_tts_duration
        if trimmed_tts_duration is not None
        else (actual_tts_duration if actual_tts_duration is not None else (raw_tts_duration if raw_tts_duration is not None else slot_duration))
    )
    slot_d = max(0.15, float(slot_duration))
    avail_d = (
        max(slot_d, float(available_window_duration))
        if available_window_duration is not None
        else slot_d + max(0.0, min(0.45, float(downstream_pause_seconds or 0.0)))
    )
    eff_max_shift = float(max_shift_seconds if max_shift_seconds is not None else max_timing_shift)
    steps_executed = []

    # Step 1: Inspect translation redundancy
    mora_count = int(entry.get('mora_count') or _estimate_unit_mora_weight(text_ja))
    empirical_mps = mora_count / max(0.1, act_d)
    target_mora_at_1x = max(1, int(math.floor(avail_d * empirical_mps)))
    target_mora_at_max = max(1, int(math.floor(avail_d * float(max_atempo) * empirical_mps)))
    is_overlong_at_1x = act_d > avail_d + 0.03
    steps_executed.append({
        'step': 1,
        'action': 'inspect_translation_redundancy',
        'mora_count': mora_count,
        'target_mora_at_1x': target_mora_at_1x,
        'target_mora_at_max_atempo': target_mora_at_max,
        'is_overlong_at_1x': is_overlong_at_1x,
    })

    # Step 2 & 3: Meaning-preserving concise rewrite + TTS regeneration recommendation
    rewrite_res = concise_rewrite_for_timing_budget(text_ja, target_mora_budget=target_mora_at_1x)
    estimated_rewritten_duration = act_d
    if rewrite_res['changed'] and mora_count > 0:
        estimated_rewritten_duration = round(
            act_d * (float(rewrite_res['rewritten_mora_count']) / float(mora_count)),
            4,
        )
    steps_executed.append({
        'step': 2,
        'action': 'concise_meaning_preserving_translation',
        'changed': rewrite_res['changed'],
        'original_text_ja': text_ja,
        'rewritten_text_ja': rewrite_res['rewritten_text_ja'],
        'original_mora_count': rewrite_res['original_mora_count'],
        'rewritten_mora_count': rewrite_res['rewritten_mora_count'],
    })
    steps_executed.append({
        'step': 3,
        'action': 'regenerate_natural_tts',
        'recommended': bool(is_overlong_at_1x and rewrite_res['changed']),
        'estimated_duration_after_regeneration_seconds': estimated_rewritten_duration,
    })

    # Step 4: Bounded speedup within [1.0, max_atempo]
    rate_res = compute_phase13_2_speech_rate_ratio(
        estimated_rewritten_duration,
        slot_d,
        available_barrier_duration=avail_d,
        allow_slowdown=False,
        max_atempo=max_atempo,
    )
    dur_after_atempo = float(rate_res['expected_duration_seconds'])
    steps_executed.append({
        'step': 4,
        'action': 'bounded_speedup',
        'applied_atempo': rate_res['applied_atempo'],
        'duration_after_atempo_seconds': dur_after_atempo,
    })

    # Step 5: Safe placement micro-adjustment
    overflow_after_atempo = max(0.0, dur_after_atempo - avail_d)
    applied_micro_shift = round(min(overflow_after_atempo, max(0.0, eff_max_shift)), 4)
    residual_overflow = round(max(0.0, overflow_after_atempo - applied_micro_shift), 4)
    steps_executed.append({
        'step': 5,
        'action': 'safe_placement_micro_adjustment',
        'applied_shift_seconds': applied_micro_shift,
        'residual_overflow_seconds': residual_overflow,
    })

    # Step 6: Quality warning if still unresolved
    quality_warning = None
    if residual_overflow > 0.03:
        quality_warning = (
            f"Utterance {entry.get('id', entry.get('utterance_id'))} still exceeds available window by "
            f"{residual_overflow:.2f}s after concise rewrite, {rate_res['applied_atempo']:.2f}x speedup, "
            f"and {applied_micro_shift:.2f}s shift."
        )
    steps_executed.append({
        'step': 6,
        'action': 'quality_warning_check',
        'warning_emitted': quality_warning is not None,
        'warning': quality_warning,
    })

    return {
        'utterance_id': entry.get('id', entry.get('utterance_id')),
        'is_overlong': is_overlong_at_1x,
        'steps_executed': steps_executed,
        'rewrite_result': rewrite_res,
        'rate_adjustment': rate_res,
        'applied_micro_shift_seconds': applied_micro_shift,
        'residual_overflow_seconds': residual_overflow,
        'quality_warning': quality_warning,
    }


def smooth_intra_speaker_speaking_rates(
    entries,
    *,
    max_adjacent_atempo_jump=0.12,
    max_adjacent_delta=None,
    max_atempo=DEFAULT_MAX_ATEMPO_PHASE13_2,
    allow_slowdown=False,
):
    """Detect and smooth abrupt speaking-rate changes across adjacent utterances of the same speaker,
    while respecting intentional source speaker pace changes.
    """
    eff_jump = float(max_adjacent_delta if max_adjacent_delta is not None else max_adjacent_atempo_jump)
    working_entries = [dict(e) for e in (entries or [])]
    by_speaker = {}
    for idx, e in enumerate(working_entries):
        spk = e.get('speaker')
        if spk:
            by_speaker.setdefault(spk, []).append((idx, e))

    anomalies = []
    smoothed_adjustments = []

    for spk, indexed_rows in sorted(by_speaker.items()):
        for k in range(len(indexed_rows) - 1):
            idx_a, a = indexed_rows[k]
            idx_b, b = indexed_rows[k + 1]
            atempo_a = float(a.get('applied_atempo', 1.0))
            atempo_b = float(b.get('applied_atempo', 1.0))
            jump = abs(atempo_b - atempo_a)

            en_words_a = len(str(a.get('text_en', a.get('original_text', ''))).split())
            en_words_b = len(str(b.get('text_en', b.get('original_text', ''))).split())
            src_dur_a = max(0.3, float(a.get('slot_duration', a.get('end', 1.0) - a.get('start', 0.0))))
            src_dur_b = max(0.3, float(b.get('slot_duration', b.get('end', 1.0) - b.get('start', 0.0))))
            en_wps_a = en_words_a / src_dur_a
            en_wps_b = en_words_b / src_dur_b
            source_rate_ratio = max(en_wps_a, en_wps_b) / max(0.5, min(en_wps_a, en_wps_b))
            intentional_source_change = bool(en_words_a > 0 and en_words_b > 0 and source_rate_ratio >= 1.45)

            if jump > eff_jump and not intentional_source_change:
                anomalies.append({
                    'speaker': spk,
                    'first_id': a.get('id', a.get('utterance_id')),
                    'second_id': b.get('id', b.get('utterance_id')),
                    'first_atempo': round(atempo_a, 4),
                    'second_atempo': round(atempo_b, 4),
                    'atempo_jump': round(jump, 4),
                    'intentional_source_change': False,
                })
                if atempo_a > atempo_b + eff_jump:
                    new_a = round(min(float(max_atempo), atempo_b + eff_jump), 4)
                    smoothed_adjustments.append({
                        'utterance_id': a.get('id', a.get('utterance_id')),
                        'old_atempo': round(atempo_a, 4),
                        'smoothed_atempo': new_a,
                    })
                    a['smoothed_atempo'] = new_a
                    a['applied_atempo'] = new_a
                elif atempo_b > atempo_a + eff_jump:
                    new_b = round(min(float(max_atempo), atempo_a + eff_jump), 4)
                    smoothed_adjustments.append({
                        'utterance_id': b.get('id', b.get('utterance_id')),
                        'old_atempo': round(atempo_b, 4),
                        'smoothed_atempo': new_b,
                    })
                    b['smoothed_atempo'] = new_b
                    b['applied_atempo'] = new_b

    return {
        'max_adjacent_atempo_jump_allowed': eff_jump,
        'detected_rate_jump_anomalies': anomalies,
        'smoothed_adjustments': smoothed_adjustments,
        'entries': working_entries,
        'tempo_consistent': len(anomalies) == 0,
    }


def build_timing_quality_report(
    utterance_diagnostic_rows,
    *,
    placements_map=None,
    duration_adjustments=None,
    allow_slowdown=False,
    max_atempo=DEFAULT_MAX_ATEMPO_PHASE13_2,
    final_mixed_audio_path=None,
):
    """Build Requirement 2.1 & 5 `timing_quality_report.json` recording per-utterance raw vs processed WAVs,
    slot durations, `atempo` ratios, resampling counts, trim/concat operations, and degradation root causes.
    """
    from pathlib import Path
    from voice_consistency import _read_mono_wav_float32, compute_acoustic_speaker_embedding

    p_map = dict(placements_map or {})
    records = []
    slowdown_violations = []
    speedup_violations = []
    total_anomalies = 0

    for row in utterance_diagnostic_rows or []:
        uid = row.get('id', row.get('utterance_id'))
        p_info = p_map.get(str(uid)) or p_map.get(uid) or {}
        merged = {**row, **p_info}
        spk = merged.get('speaker', merged.get('speaker_id'))
        raw_wav = merged.get('raw_tts_wav_path') or merged.get('raw_audio_file') or merged.get('original_audio_file') or merged.get('audio_file')
        proc_wav = merged.get('processed_wav_path') or merged.get('active_audio_file') or merged.get('audio_file')
        slot_dur = float(merged.get('slot_duration_seconds', merged.get('slot_duration', merged.get('source_slot_duration_seconds', 1.0))))
        raw_dur = float(merged.get('raw_tts_duration_seconds', merged.get('original_duration_seconds', merged.get('actual_tts_duration_seconds', merged.get('actual_duration', 1.0)))))
        proc_dur = float(merged.get('processed_tts_duration_seconds', merged.get('actual_tts_duration_seconds', merged.get('actual_duration', raw_dur))))
        atempo_val = float(merged.get('applied_atempo', 1.0))
        resampling_count = int(merged.get('resampling_count', 1 if abs(atempo_val - 1.0) > 0.01 else 0))

        raw_ac = compute_acoustic_speaker_embedding(raw_wav) if (raw_wav and Path(raw_wav).is_file()) else {}
        proc_ac = compute_acoustic_speaker_embedding(proc_wav) if (proc_wav and Path(proc_wav).is_file()) else raw_ac

        issues = []
        if not allow_slowdown and atempo_val < 0.999:
            slowdown_violations.append({'utterance_id': uid, 'applied_atempo': atempo_val})
            issues.append(f'slowdown_prohibited:{atempo_val:.3f}')
        if atempo_val > float(max_atempo) + 0.001:
            speedup_violations.append({'utterance_id': uid, 'applied_atempo': atempo_val, 'max_atempo': max_atempo})
            issues.append(f'exceeded_max_atempo:{atempo_val:.3f}')
        if resampling_count > 1:
            issues.append(f'excessive_resampling:{resampling_count}')

        raw_warble = float(raw_ac.get('warble_modulation_index', 0.0))
        proc_warble = float(proc_ac.get('warble_modulation_index', 0.0))
        proc_clicks = int(proc_ac.get('click_noise_count', 0))
        if proc_clicks > 0:
            issues.append(f'click_noise_detected:{proc_clicks}')

        # Check tail truncation (non-zero high amplitude at very end of WAV)
        tail_truncated = False
        if proc_wav and Path(proc_wav).is_file():
            try:
                p_samples, p_sr = _read_mono_wav_float32(proc_wav)
                tail_win = min(len(p_samples), max(8, int(round(0.008 * p_sr))))
                tail_rms = float(np.sqrt(np.mean(p_samples[-tail_win:] ** 2)))
                if tail_rms > 0.12:
                    tail_truncated = True
                    issues.append(f'abrupt_tail_truncation:rms={tail_rms:.3f}')
            except Exception:
                pass

        if atempo_val < 0.98 and proc_warble > raw_warble + 0.12:
            degradation_cause = 'time_stretch_slowdown_warble'
        elif atempo_val > 1.25:
            degradation_cause = 'excessive_time_stretch_speedup'
        elif raw_warble > 0.72:
            degradation_cause = 'raw_tts_generation_instability'
        elif proc_clicks > int(raw_ac.get('click_noise_count', 0)) + 2 or tail_truncated:
            degradation_cause = 'post_processing_trim_or_concat_click'
        else:
            degradation_cause = 'none'

        if issues:
            total_anomalies += 1

        records.append({
            'utterance_id': uid,
            'speaker_id': spk,
            'raw_tts_wav_path': str(raw_wav) if raw_wav else None,
            'processed_wav_path': str(proc_wav) if proc_wav else None,
            'final_mixed_audio_path': str(final_mixed_audio_path) if final_mixed_audio_path else merged.get('final_mixed_audio_path'),
            'source_slot_duration_seconds': round(slot_dur, 4),
            'raw_tts_duration_seconds': round(raw_dur, 4),
            'processed_tts_duration_seconds': round(proc_dur, 4),
            'applied_atempo_ratio': round(atempo_val, 4),
            'resampling_count': resampling_count,
            'trim_and_concat_operations': merged.get('trim_and_concat_operations', {
                'silence_trimmed': bool(merged.get('trim_applied', merged.get('silence_trimmed', False))),
                'trimmed_seconds': round(max(0.0, raw_dur - proc_dur) if abs(atempo_val - 1.0) < 1e-4 else 0.0, 4),
            }),
            'raw_warble_modulation_index': raw_warble,
            'processed_warble_modulation_index': proc_warble,
            'onset_distortion_score': float(proc_ac.get('onset_distortion_score', 0.0)),
            'click_noise_count': proc_clicks,
            'clipped_sample_count': int(proc_ac.get('clipped_sample_count', 0)),
            'tail_truncated': tail_truncated,
            'degradation_cause': degradation_cause,
            'issues': issues,
        })

    tempo_smoothing = smooth_intra_speaker_speaking_rates(
        [
            {
                'id': r['utterance_id'],
                'speaker': r['speaker_id'],
                'applied_atempo': r['applied_atempo_ratio'],
                'slot_duration': r['source_slot_duration_seconds'],
            }
            for r in records
        ],
        max_atempo=max_atempo,
    )

    return {
        'kind': 'timing_quality_report',
        'format_version': 1,
        'policy': {
            'normal_atempo': DEFAULT_NORMAL_ATEMPO,
            'default_max_atempo': DEFAULT_MAX_ATEMPO_PHASE13_2,
            'hard_max_atempo': HARD_MAX_ATEMPO_PHASE13_2,
            'allow_slowdown': bool(allow_slowdown),
            'min_exceptional_slowdown_atempo': MIN_EXCEPTIONAL_SLOWDOWN_ATEMPO,
        },
        'utterance_count': len(records),
        'anomaly_count': total_anomalies,
        'slowdown_violation_count': len(slowdown_violations),
        'speedup_violation_count': len(speedup_violations),
        'slowdown_violations': slowdown_violations,
        'speedup_violations': speedup_violations,
        'intra_speaker_tempo_consistency': tempo_smoothing,
        'utterances': records,
        'all_timing_constraints_satisfied': len(slowdown_violations) == 0 and len(speedup_violations) == 0 and total_anomalies == 0,
    }
