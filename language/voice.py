"""Explicit preset, voice-design and authorized-reference synthesis contracts."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VoiceProfile:
    speaker: str
    mode: str = 'preset'
    voice: str | None = None
    reference_audio: str | None = None
    consent_reference: str | None = None
    design_description: str | None = None
    analysis: dict | None = None
    reference_text: str | None = None
    reference_audio_sha256: str | None = None
    reference_text_review: dict | None = None
    source_audio_quality: dict | None = None
    voice_characteristics: dict | None = None

    def __post_init__(self):
        if not self.speaker or self.speaker in ('UNKNOWN','REVIEW','MIXED'):
            raise ValueError('A known speaker ID is required.')
        if self.mode not in {'preset','voice_design','approved_clone','approved_reference'}:
            raise ValueError('Unsupported voice mode.')
        if self.mode in {'approved_clone','approved_reference'} and not (self.reference_audio and self.consent_reference):
            raise ValueError('Reference voice requires audio and documented consent.')
        if self.mode in {'preset','voice_design'} and (self.reference_audio or self.consent_reference):
            raise ValueError('Reference audio/consent cannot be supplied as preset or voice design.')
        if self.mode=='voice_design' and (not self.design_description or not self.design_description.strip()):
            raise ValueError('Voice design requires an explicit synthetic voice description.')
        if self.mode=='preset' and self.design_description:
            raise ValueError('Preset cannot silently use voice design.')


def validate_clone_permission(profile):
    """Check documented authorization binding. Never infer consent from a profile."""
    record=json.loads(Path(profile.consent_reference).read_text())
    digest=hashlib.sha256(Path(profile.reference_audio).read_bytes()).hexdigest()
    if (record.get('authorized') is not True or record.get('speaker_id')!=profile.speaker
            or record.get('reference_audio_sha256')!=digest or record.get('scope')!='voice_cloning'
            or not isinstance(record.get('evidence'),str) or not record['evidence'].strip()
            or not isinstance(record.get('verified_by'),str) or not record['verified_by'].strip()):
        raise ValueError('Voice cloning requires verified authorization for this speaker and exact audio.')
    return record


def synthesize_profile(backend, *, text, profile, output_path):
    """Dispatch without falling back from clone/design to a different method."""
    if profile.mode in {'approved_clone','approved_reference'}:
        validate_clone_permission(profile)
    mode='approved_clone' if profile.mode=='approved_reference' else profile.mode
    if mode not in getattr(backend,'supported_voice_modes',{'preset'}):
        raise ValueError(f'Backend does not implement {mode}; no audio generated.')
    return backend.synthesize(text=text,profile=profile,output_path=output_path)


def profile_for_synthesis(document, speaker, *, preset=None, mode='preset', design_description=None,
                          reference_id=None, consent_reference=None, document_directory=None):
    if document.get('kind')!='speaker_profiles' or document.get('format_version')!=1:
        raise ValueError('Expected speaker_profiles v1.')
    rows=[p for p in document.get('profiles',[]) if p.get('speaker_id')==speaker]
    if len(rows)!=1: raise ValueError('Missing or duplicate speaker profile.')
    source=rows[0]
    analysis={'analysis_key':document['analysis_key'],'video_id':document['video_id'],'clip':document['clip'],
              'speaker_id':speaker,'summary_dsp_estimates':source['summary_dsp_estimates'],
              'model_estimates':source['model_estimates'],'quality':source['quality']}
    reference=None
    if mode in {'approved_clone','approved_reference'}:
        refs=[r for r in source['reference_candidates'] if r['reference_id']==reference_id and r.get('audio_file')]
        if len(refs)!=1 or not refs[0]['quality']['eligible']:
            raise ValueError('Clone requires a selected, eligible reference.')
        path=Path(refs[0]['audio_file'])
        if not path.is_absolute():
            if document_directory is None: raise ValueError('Reference profile directory required.')
            path=Path(document_directory)/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=refs[0]['audio_sha256']:
            raise ValueError('Reference audio hash mismatch.')
        reference=str(path.resolve())
    return VoiceProfile(speaker,mode=mode,voice=preset,design_description=design_description,
                        reference_audio=reference,consent_reference=consent_reference,analysis=analysis)


class ProfiledPresetSynthesizer:
    """Pass acoustic provenance through the existing preset boundary; no imitation."""
    supported_voice_modes={'preset'}

    def __init__(self,backend,document,*,video_id,clip):
        from .cache import fingerprint
        if document.get('video_id')!=video_id or document.get('clip')!=clip:
            raise ValueError('Voice profiles do not match synthesis video/clip.')
        if document.get('kind')!='speaker_profiles' or document.get('format_version')!=1:
            raise ValueError('Expected speaker_profiles v1.')
        ids=[p.get('speaker_id') for p in document.get('profiles',[])]
        if not ids or len(ids)!=len(set(ids)) or any(not isinstance(s,str) or s in ('UNKNOWN','REVIEW','MIXED') for s in ids):
            raise ValueError('Invalid profile speaker IDs.')
        self.backend,self.document=backend,document
        self.identity={**backend.identity,'speaker_profiles_fingerprint':fingerprint(document),
                       'voice_profile_mode':'preset','voice_profile_effect':'analysis provenance only; preset voice unchanged'}

    def synthesize(self,*,text,profile,output_path):
        if profile.mode!='preset': raise ValueError('Preset wrapper cannot clone or design voices.')
        enriched=profile_for_synthesis(self.document,profile.speaker,preset=profile.voice)
        return synthesize_profile(self.backend,text=text,profile=enriched,output_path=output_path)


class SpeechSynthesizer(Protocol):
    identity: dict

    def synthesize(self, *, text: str, profile: VoiceProfile, output_path: str) -> dict:
        """Return actual generation metadata; never truncate or silently change mode."""
        ...
