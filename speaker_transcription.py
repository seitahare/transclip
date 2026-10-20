"""Word/time provenance, transcript comparisons, and a local ASR worker."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time


def tokens(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?",text.casefold())


def align_segments(result, *, clip_start, window_start, window_end, core_start, core_end,
                   speaker_id, source, separation_source):
    """Keep full word times; ownership by midpoint avoids window-boundary duplication."""
    output=[]
    for segment in result.get('segments',[]):
        words=[]; reasons=[]
        for w in segment.get('words',[]):
            a,b=w.get('start'),w.get('end')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not 0<=a<b<=window_end-window_start+.03:
                reasons.append('invalid_word_timestamp');continue
            a,b=a+window_start,b+window_start
            if not core_start <= (a+b)/2 < core_end: continue
            if a<core_start or b>core_end: reasons.append('word_crosses_processing_boundary')
            words.append({'word':w['word'].strip(),'relative_start':a,'relative_end':b,
                          'start':clip_start+a,'end':clip_start+b,'speaker':speaker_id,
                          'asr_probability':w.get('probability'),'review_required':True,
                          'assignment_method':'separated_audio_embedding' if separation_source else 'original_audio_embedding'})
        if not words:continue
        output.append({'speaker_id':speaker_id,'text':' '.join(w['word'] for w in words),'words':words,
                       'relative_start':words[0]['relative_start'],'relative_end':words[-1]['relative_end'],
                       'start':words[0]['start'],'end':words[-1]['end'],
                       'quality':{'avg_logprob':segment.get('avg_logprob'),'no_speech_prob':segment.get('no_speech_prob')},
                       'transcription_source':source,'separation_source':separation_source,
                       'review_required':True,'review_reasons':sorted(set(reasons+['asr_and_speaker_unverified']))})
    return output


def compare_transcript(row, source_words, captions=()):
    """Lexical/time support is a warning signal, never proof of correct recognition."""
    unsupported=[]
    for w in row['words']:
        center=(w['relative_start']+w['relative_end'])/2
        local=[s['word'] for s in source_words if s['relative_start']-.75 <= center <= s['relative_end']+.75]
        local += [s['text'] for s in captions if s['relative_start']-.75 <= center <= s['relative_end']+.75]
        supported=set(tokens(' '.join(local)))
        if not set(tokens(w['word'])).issubset(supported):unsupported.append(w['word'])
    quality=row['quality'];reasons=[]
    if unsupported:reasons.append('words_not_supported_by_original_asr_or_captions')
    if quality.get('no_speech_prob') is not None and quality['no_speech_prob']>.6:reasons.append('asr_no_speech_warning')
    if quality.get('avg_logprob') is not None and quality['avg_logprob'] < -1.:reasons.append('low_asr_logprob')
    words=tokens(row['text'])
    if len(words)>5 and len(set(words))<len(words)/3:reasons.append('repetitive_asr_hallucination_possible')
    return {'unsupported_words':unsupported,'review_reasons':reasons,
            'support_is_ground_truth':False,'automatic_confirmation_allowed':False}


def worker(model_dir,language):
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from faster_whisper import WhisperModel
    from separation_backends import peak_memory_bytes
    model=None
    for line in sys.stdin:
        start=time.perf_counter()
        try:
            request=json.loads(line)
            if model is None:model=WhisperModel(str(model_dir),device='cpu',compute_type='int8',local_files_only=True,cpu_threads=4)
            segments,_=model.transcribe(request['audio_file'],language=language,word_timestamps=True,
                                         vad_filter=False,beam_size=3,condition_on_previous_text=False)
            result={'segments':[{'text':s.text,'start':s.start,'end':s.end,'avg_logprob':s.avg_logprob,
                                 'no_speech_prob':s.no_speech_prob,
                                 'words':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability} for w in s.words or []]}
                                for s in segments], 'elapsed_seconds':time.perf_counter()-start,
                    'peak_memory_bytes':peak_memory_bytes()}
        except Exception as error:result={'error':str(error)}
        print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--language',default='en');a=p.parse_args()
    if not a.worker:p.error('This module is an ASR worker; use speech_separation.py for the pipeline.')
    worker(a.model_dir,a.language)
