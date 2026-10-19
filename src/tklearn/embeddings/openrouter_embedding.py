from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tklearn.embeddings.base import Embedding

__all__ = [
    "OPENROUTER_URL",
    "OpenRouterEmbedding",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# rate limits, timeouts and server errors are worth retrying
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class OpenRouterEmbedding(Embedding):
    """Embeddings from a model served through the OpenRouter API.

    Parameters
    ----------
    name : str
        Model id on OpenRouter, e.g. ``"qwen/qwen3-embedding-8b"``.
    dimensions : int, optional
        Ask for vectors of this size, for models that can shorten them
        (e.g. Qwen3 and OpenAI's text-embedding-3). By default, the model's
        full size.
    api_key : str, optional
        By default ``$OPENROUTER_API_KEY``.
    base_url : str, default=OPENROUTER_URL
        API root; any API with OpenAI's ``/embeddings`` format works.
    batch_size : int, default=256
        Texts per request.
    max_workers : int, default=4
        Requests sent at once.
    max_retries : int, default=6
        Retries of a request after a rate limit, a timeout or a server
        error, waiting 1, 2, 4, ... seconds (or as the server asks).
    timeout : float, default=60
        Seconds to wait for a response.
    """

    def __init__(
        self,
        name: str,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str = OPENROUTER_URL,
        batch_size: int = 256,
        max_workers: int = 4,
        max_retries: int = 6,
        timeout: float = 60,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            msg = "pass api_key or set OPENROUTER_API_KEY"
            raise ValueError(msg)
        self.name = name
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self._api_key = api_key
        self._dim: int | None = None

    def __repr__(self) -> str:
        args = repr(self.name)
        if self.dimensions is not None:
            args += f", dimensions={self.dimensions}"
        return f"{type(self).__name__}({args})"

    @property
    def dim(self) -> int:
        # the API doesn't say, so embed a text once to find out
        if self._dim is None:
            self.encode(["dim"])
        return self._dim

    def encode(self, texts: str | Sequence[str]) -> np.ndarray:
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not texts:
            return np.empty((0, self._dim or 0), dtype=np.float32)
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        with ThreadPoolExecutor(self.max_workers) as pool:
            vectors = np.concatenate(list(pool.map(self._request, batches)))
        self._dim = vectors.shape[1]
        return vectors[0] if single else vectors

    def _request(self, texts: list[str]) -> np.ndarray:
        body = {"model": self.name, "input": texts}
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    payload = json.load(response)
            except urllib.error.HTTPError as e:
                if e.code not in _RETRY_STATUS or attempt == self.max_retries:
                    detail = e.read().decode(errors="replace")[:1000]
                    msg = f"{self.name}: HTTP {e.code} {e.reason}: {detail}"
                    raise RuntimeError(msg) from e
                delay = float(e.headers.get("Retry-After") or 2**attempt)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
                delay = 2**attempt
            else:
                return _vectors(self.name, payload, len(texts))
            time.sleep(delay)
        raise AssertionError("unreachable")


def _vectors(name: str, payload: dict, n: int) -> np.ndarray:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != n:
        msg = f"{name}: expected {n} embeddings, got {str(payload)[:1000]}"
        raise RuntimeError(msg)
    data = sorted(data, key=lambda item: item["index"])
    return np.asarray([item["embedding"] for item in data], dtype=np.float32)
