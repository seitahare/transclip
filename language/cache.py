"""Content-addressed candidate cache and bounded provider calls.

No network provider is installed here. Applications may inject one only after
explicit API authorization. Each successful call is persisted for resumption.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Protocol


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, data, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(name, path)
        else:
            # Atomic create without clobber, including another process's output.
            os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class CandidateProvider(Protocol):
    # Include backend/model/prompt version and settings in identity.
    identity: dict

    def propose(self, request: dict) -> str: ...


class CachedProvider:
    def __init__(self, provider: CandidateProvider, directory, *, max_calls=0):
        if not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative integer.")
        self.provider, self.directory = provider, Path(directory)
        self.max_calls, self.calls = max_calls, 0

    @property
    def identity(self):
        return self.provider.identity

    def propose(self, request):
        key = fingerprint({"format_version": 1, "provider": self.identity, "request": request})
        path = self.directory / (key + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get("key") != key or not isinstance(cached.get("text_ja"), str) or not cached["text_ja"].strip():
                raise ValueError("Corrupt candidate cache.")
            return cached["text_ja"]
        if self.calls >= self.max_calls:
            raise RuntimeError("Candidate call budget exhausted; successful calls remain cached.")
        self.calls += 1
        text = self.provider.propose(request)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provider returned an empty or invalid candidate.")
        atomic_json(path, {"key": key, "text_ja": text.strip()})
        return text.strip()
