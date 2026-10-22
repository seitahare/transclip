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
        events.append({'entry_ids':[r['id'] for r in group],'status':'unresolved' if reasons else 'placed',
                       'proposed_starts':proposed,'reasons':sorted(set(reasons)),
                       'max_shift_seconds':max(proposed[r['id']]-r['relative_start'] for r in group)})
    return {'format_version':1,'policy':'adjacent_same_speaker_bounded_delay','max_shift_seconds':max_shift,
            'minimum_pause_seconds':minimum_pause,'playback_rate':1.0,'starts':starts,'events':events,
            'source_overlap_anchors':sorted(anchored),'waveform_modified':False,
            'review_required':bool(events),'human_timing_verified':False}
