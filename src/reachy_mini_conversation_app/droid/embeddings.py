"""Local sentence embeddings (all-MiniLM-L6-v2, Apache-2.0) on ONNX Runtime, for semantic memory search."""

import threading

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from numpy.typing import NDArray
from huggingface_hub import hf_hub_download


_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_MAX_TOKENS = 256
EMBEDDING_DIM = 384


class TextEmbedder:
    """Turn texts into unit-length 384-d vectors; the model downloads once and loads lazily."""

    def __init__(self) -> None:
        """Defer loading until the first ``embed`` call."""
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None

    def _load(self) -> tuple[ort.InferenceSession, Tokenizer]:
        if self._session is None or self._tokenizer is None:
            tokenizer = Tokenizer.from_file(hf_hub_download(_MODEL_REPO, "tokenizer.json", revision=_MODEL_REVISION))
            tokenizer.enable_truncation(_MAX_TOKENS)
            tokenizer.enable_padding()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                hf_hub_download(_MODEL_REPO, "onnx/model.onnx", revision=_MODEL_REVISION),
                options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = tokenizer
        return self._session, self._tokenizer

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return one normalised embedding row per text (mean pooling over tokens)."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        with self._lock:
            session, tokenizer = self._load()
            encodings = tokenizer.encode_batch(texts)
            ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
            mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            outputs = session.run(None, inputs)
        token_embeddings = np.asarray(outputs[0], dtype=np.float32)
        weights = mask[..., None].astype(np.float32)
        pooled = (token_embeddings * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
        normalized: NDArray[np.float32] = (pooled / norms).astype(np.float32)
        return normalized
