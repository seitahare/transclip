"""Waveform descriptors and conservative reference selection (no voice cloning)."""
import math
import numpy as np

RATE = 16000
FEATURE_VERSION = 1


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def subtract(intervals, excluded):
    result = []
    for a, b in union(intervals):
        for left, right in union(excluded):
            if right <= a or left >= b:
                continue
            if left > a:
                result.append((a, left))
            a = max(a, right)
            if a >= b:
                break
        if a < b:
            result.append((a, b))
    return result


def overlap_regions(turns):
    return union([(max(a['relative_start'], b['relative_start']), min(a['relative_end'], b['relative_end']))
                  for i, a in enumerate(turns) for b in turns[i+1:]
                  if a['speaker'] != b['speaker'] and max(a['relative_start'], b['relative_start']) < min(a['relative_end'], b['relative_end'])])


def reference_windows(turns, *, excluded=(), minimum=2.5, maximum=8.0, step=4.0, guard=.10):
    """Never bridge silence or include known overlap; long turns yield many candidates."""
    if not 0 < minimum <= maximum or step <= 0 or guard < 0:
        raise ValueError('Invalid reference window parameters.')
    result = []
    for speaker in sorted({t['speaker'] for t in turns if t['speaker'] not in (None,'UNKNOWN','REVIEW')}):
        own = [(t['relative_start'],t['relative_end']) for t in turns if t['speaker']==speaker]
        other = [(t['relative_start'],t['relative_end']) for t in turns if t['speaker']!=speaker]
        for a,b in subtract(own,[*other,*excluded]):
            a,b=a+guard,b-guard
            if b-a < minimum:
                continue
            starts=[a]
            while starts[-1]+maximum < b-1e-6:
                nxt=min(starts[-1]+step,b-maximum)
                if nxt<=starts[-1]+1e-6: break
                starts.append(nxt)
            for left in starts:
                result.append({'speaker_id':speaker,'relative_start':round(left,6),
                               'relative_end':round(min(left+maximum,b),6), 'known_overlap_seconds':0.0})
    return result


def normalize(vector):
    value=np.asarray(vector,dtype=float)
    if value.ndim!=1 or value.size==0 or not np.isfinite(value).all() or np.linalg.norm(value)<1e-8:
        raise ValueError('Invalid embedding vector.')
    return value/np.linalg.norm(value)


def cosine(a,b):
    x,y=normalize(a),normalize(b)
    if x.shape!=y.shape: raise ValueError('Embedding dimensions differ.')
    return float(np.clip(np.dot(x,y),-1,1))


def centroid(vectors):
    if not vectors: return None
    return normalize(np.mean([normalize(v) for v in vectors],axis=0)).tolist()


def rank_speakers(vector,centers):
    if vector is None: return []
    return sorted([{'speaker_id':speaker,'cosine_similarity':cosine(vector,c)} for speaker,c in centers.items() if c is not None],
                  key=lambda x:(-x['cosine_similarity'],x['speaker_id']))


def runs(mask,hop):
    output=[];start=None
    for i,on in enumerate([*mask,False]):
        if on and start is None: start=i
        elif not on and start is not None:
            output.append({'start':round(start*hop,4),'end':round(i*hop,4),'duration':round((i-start)*hop,4)})
            start=None
    return output


def acoustic_features(samples, rate=RATE, words=()):
    """F0/activity are DSP estimates, never labelled ground-truth measurements."""
    x=np.asarray(samples,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or rate<4000:
        raise ValueError('Invalid mono waveform/rate.')
    if np.max(np.abs(x))>1.0001: raise ValueError('Waveform must be normalized PCM.')
    frame=max(1,round(.04*rate));hop=max(1,round(.01*rate))
    padded=np.pad(x,(0,max(0,frame-len(x))))
    frames=np.lib.stride_tricks.sliding_window_view(padded,frame)[::hop].copy()
    rms=np.sqrt(np.mean(frames**2,axis=1))
    activity=rms>max(.003,float(np.percentile(rms,95))*.08)
    centered=frames-np.mean(frames,axis=1,keepdims=True)
    spectrum=np.abs(np.fft.rfft(centered*np.hanning(frame),axis=1))**2
    frequencies=np.fft.rfftfreq(frame,1/rate)
    total=spectrum.sum(axis=1)+1e-20
    cents=(spectrum@frequencies)/total
    flat=np.exp(np.mean(np.log(spectrum+1e-20),axis=1))/(np.mean(spectrum,axis=1)+1e-20)
    zcr=np.mean(np.diff(np.signbit(centered),axis=1),axis=1)
    fft=np.fft.rfft(centered,n=2*frame,axis=1)
    ac=np.fft.irfft(fft*np.conj(fft),n=2*frame,axis=1)[:,:frame]
    energy=np.cumsum(centered**2,axis=1)
    f0=[];periodicity=[]
    low,high=int(rate/450),min(frame-2,int(rate/60))
    for i in np.where(activity)[0]:
        lags=np.arange(low,high+1)
        # Normalize each lag by its two corresponding frame energies.
        left=energy[i,frame-lags-1]
        right=energy[i,-1]-energy[i,lags-1]
        correlations=ac[i,lags]/np.sqrt(np.maximum(left*right,1e-20))
        peaks=np.where((correlations[1:-1]>correlations[:-2])&(correlations[1:-1]>=correlations[2:]))[0]+1
        if len(peaks)==0: continue
        best=float(np.max(correlations[peaks]))
        eligible=peaks[correlations[peaks]>=max(.65,.9*best)]
        if len(eligible)==0: continue
        index=int(eligible[0]);lag=lags[index]
        y0,y1,y2=correlations[index-1:index+2]
        correction=.5*(y0-y2)/(y0-2*y1+y2) if abs(y0-2*y1+y2)>1e-12 else 0
        f0.append(float(rate/(lag+np.clip(correction,-.5,.5))))
        periodicity.append(min(.999999,max(0,float(y1))))
    inactive=runs((~activity).tolist(),hop/rate)
    pauses=[p for p in inactive if p['duration']>=.15]
    speech_runs=runs(activity.tolist(),hop/rate)
    lengths=[r['duration'] for r in speech_runs]
    duration=len(x)/rate
    centers=[(w['relative_start']+w['relative_end'])/2 for w in words]
    gaps=[max(0,b['relative_start']-a['relative_end']) for a,b in zip(words,words[1:])]
    def median(v): return float(np.median(v)) if len(v) else None
    return {'feature_version':FEATURE_VERSION,
            'measured':{'sample_rate':rate,'sample_frames':len(x),'duration_seconds':duration,
                        'peak_amplitude':float(np.max(np.abs(x))),'rms_amplitude':float(np.sqrt(np.mean(x*x))),
                        'clipped_fraction':float(np.mean(np.abs(x)>=32767/32768)),
                        'spectral_centroid_hz_median':median(cents[activity]),
                        'spectral_flatness_median':median(flat[activity]),'zero_crossing_rate_median':median(zcr[activity])},
            'dsp_estimates':{'method':'normalized autocorrelation 40ms / 10ms hop, 60..450Hz',
                             'f0_hz_median':median(f0),'f0_hz_p10':float(np.percentile(f0,10)) if f0 else None,
                             'f0_hz_p90':float(np.percentile(f0,90)) if f0 else None,
                             'f0_hz_std':float(np.std(f0)) if f0 else None,'f0_frame_count':len(f0),
                             'periodicity_median':median(periodicity),'voiced_fraction':len(f0)/len(frames),
                             'energy_activity_fraction':float(np.mean(activity)),
                             'low_energy_pauses':pauses,'pause_seconds':sum(p['duration'] for p in pauses),
                             'rhythm_active_run_cv':float(np.std(lengths)/np.mean(lengths)) if lengths else None,
                             'energy_dynamic_range_db_proxy':float(20*np.log10((np.percentile(rms,90)+1e-8)/(np.percentile(rms,10)+1e-8))),
                             'limitations':['Pitch octave errors possible.','Energy pauses are not linguistic pauses.',
                                            'Dynamic range is not a calibrated SNR or intelligibility measure.']},
            'asr_estimates':{'word_count':len(words),'words_per_second':len(words)/duration,
                             'word_gap_seconds':gaps,'median_interword_center_seconds':median(np.diff(centers)) if len(centers)>1 else None,
                             'source':'existing Whisper word timestamps; transcription unverified'}}


def reference_quality(features, own_similarity, margin, *, overlap_seconds=0, stable_identity=True):
    m,d=features['measured'],features['dsp_estimates']
    reasons=[]
    if overlap_seconds>0: reasons.append('known_overlapping_speech')
    if m['duration_seconds']<2.5: reasons.append('too_short')
    if m['clipped_fraction']>.001: reasons.append('clipping')
    if m['rms_amplitude']<.005: reasons.append('low_level')
    if d['voiced_fraction']<.25: reasons.append('limited_voiced_audio')
    flatness=m['spectral_flatness_median']
    if flatness is not None and flatness>.45: reasons.append('noise_like_spectrum_proxy')
    if own_similarity is None or own_similarity<.55 or margin is None or margin<.10: reasons.append('uncertain_speaker_embedding')
    if not stable_identity: reasons.append('unresolved_global_speaker_identity')
    score=(max(0,own_similarity or 0)+max(0,margin or 0)+d['voiced_fraction']
           + min(m['duration_seconds'],8)/8 - min(1,m['clipped_fraction']*100) - min(1,(flatness or 0)*2))
    return {'eligible':not reasons,'review_required':True,'rejection_reasons':reasons,'ranking_score':score,
            'known_overlap_seconds':overlap_seconds,'background_noise_assessment':'heuristic only; human listening required',
            'noise_like_spectral_flatness_proxy':flatness,
            'intelligibility_verified':False,'clone_permission':'not_verified'}


if __name__=='__main__':
    from speaker_refinement import main
    raise SystemExit(main())
