"""Bounded, same-speaker reflow from measured durations; no waveform edits.

Adapted from TRANS sync_group's start=max(source_start, previous_end+pause).
Only adjacent known-speaker groups may move. Other speakers and source overlap
are hard boundaries; unsafe proposals are retained as unresolved diagnostics.
"""
import math


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
