"""Text and audio embeddings in one space (CLAP), L2-normalised.

`Embedder` is the interface the index and retrieval use. `ClapEmbedder` wraps
`laion/larger_clap_music`; tests pass a small deterministic embedder instead.
Text embeddings are cached on disk by (model id, text).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_id: str
    sr: int

    def text(self, texts: list[str]) -> np.ndarray: ...
    def audio(self, clips: list[np.ndarray]) -> np.ndarray: ...


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class CachedText:
    """Wrap an embedder so repeated query texts are not re-embedded."""

    def __init__(self, inner: Embedder, cache_dir: str | Path):
        self.inner = inner
        self.model_id, self.sr = inner.model_id, inner.sr
        self.dir = Path(cache_dir) / "text_embeddings"

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model_id}\n{text}".encode()).hexdigest()[:32]
        return self.dir / f"{key}.npy"

    def text(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray | None] = []
        missing = []
        for i, t in enumerate(texts):
            p = self._path(t)
            if p.exists():
                out.append(np.load(p))
            else:
                out.append(None)
                missing.append(i)
        if missing:
            vecs = self.inner.text([texts[i] for i in missing])
            self.dir.mkdir(parents=True, exist_ok=True)
            for i, v in zip(missing, vecs):
                np.save(self._path(texts[i]), v)
                out[i] = v
        return np.stack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        return self.inner.audio(clips)


def clap_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class ClapEmbedder:
    """`laion/larger_clap_music` through transformers `ClapModel` / `ClapProcessor`."""

    def __init__(self, model_id: str = "laion/larger_clap_music", batch_size: int = 16, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.model_id = model_id
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(device).eval()
        # Never hardcode: read the rate the feature extractor expects.
        self.sr = int(self.processor.feature_extractor.sampling_rate)
        self._torch = torch

    def _features(self, out) -> np.ndarray:
        # transformers returns a tensor, or an output object in newer versions.
        t = out if hasattr(out, "detach") else getattr(out, "pooler_output", None)
        if t is None:
            t = out[0]
        return t.detach().float().cpu().numpy()

    def text(self, texts: list[str]) -> np.ndarray:
        vecs = []
        with self._torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inputs = self.processor(text=texts[i:i + self.batch_size], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vecs.append(self._features(self.model.get_text_features(**inputs)))
        return l2(np.concatenate(vecs)) if vecs else np.zeros((0, 512), dtype=np.float32)

    def audio(self, clips: list[np.ndarray]) -> np.ndarray:
        vecs = []
        with self._torch.no_grad():
            for i in range(0, len(clips), self.batch_size):
                batch = [np.asarray(c, dtype=np.float32) for c in clips[i:i + self.batch_size]]
                inputs = self.processor(audio=batch, sampling_rate=self.sr, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vecs.append(self._features(self.model.get_audio_features(**inputs)))
        return l2(np.concatenate(vecs)) if vecs else np.zeros((0, 512), dtype=np.float32)


def make_embedder(model_id: str, batch_size: int, cache_dir: str | Path) -> Embedder | None:
    """CLAP if torch and transformers import and the weights load, else None (D3: tag search)."""
    if not clap_available():
        return None
    try:
        return CachedText(ClapEmbedder(model_id, batch_size), cache_dir)
    except Exception:  # noqa: BLE001  (no weights and no network)
        return None
