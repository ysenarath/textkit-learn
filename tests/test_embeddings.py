import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import numpy as np

from tklearn.embeddings import OpenRouterEmbedding

URLOPEN = "tklearn.embeddings.openrouter_embedding.urllib.request.urlopen"
SLEEP = "tklearn.embeddings.openrouter_embedding.time.sleep"


def http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/embeddings",
        code,
        "error",
        headers or {},
        io.BytesIO(body),
    )


class FakeAPI:
    """Answers each request with [len(text), position] per text."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.requests = []

    def __call__(self, request, timeout):
        body = json.loads(request.data)
        self.requests.append((request, body))
        if self.errors:
            raise self.errors.pop(0)
        data = [
            {"index": i, "embedding": [len(text), i]}
            for i, text in enumerate(body["input"])
        ]
        # the API doesn't promise the order of the items
        payload = {"data": data[::-1]}
        return io.BytesIO(json.dumps(payload).encode())


class TestOpenRouterEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = OpenRouterEmbedding(
            "qwen/qwen3-embedding-8b", api_key="secret", batch_size=2
        )

    def test_batches_requests_in_order(self):
        api = FakeAPI()
        with mock.patch(URLOPEN, api):
            vectors = self.embedding.encode(["a", "bb", "ccc", "dddd", "e"])
        np.testing.assert_array_equal(
            vectors, [[1, 0], [2, 1], [3, 0], [4, 1], [1, 0]]
        )
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(
            sorted(len(body["input"]) for _, body in api.requests), [1, 2, 2]
        )
        request, body = api.requests[0]
        self.assertEqual(
            request.full_url, "https://openrouter.ai/api/v1/embeddings"
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(body["model"], "qwen/qwen3-embedding-8b")
        self.assertNotIn("dimensions", body)

    def test_dimensions(self):
        embedding = OpenRouterEmbedding("m", dimensions=1024, api_key="k")
        api = FakeAPI()
        with mock.patch(URLOPEN, api):
            embedding.encode(["a"])
        self.assertEqual(api.requests[0][1]["dimensions"], 1024)
        self.assertEqual(
            repr(embedding), "OpenRouterEmbedding('m', dimensions=1024)"
        )

    def test_single_text_and_dim(self):
        api = FakeAPI()
        with mock.patch(URLOPEN, api):
            self.assertEqual(self.embedding.encode("abc").tolist(), [3, 0])
            self.assertEqual(self.embedding.dim, 2)
        self.assertEqual(len(api.requests), 1)

    def test_retries_rate_limits_and_server_errors(self):
        api = FakeAPI([
            http_error(429, headers={"Retry-After": "3"}),
            http_error(502),
            urllib.error.URLError("connection reset"),
        ])
        with mock.patch(URLOPEN, api), mock.patch(SLEEP) as sleep:
            vectors = self.embedding.encode(["a"])
        self.assertEqual(vectors.tolist(), [[1, 0]])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [3, 2, 4])

    def test_gives_up_after_max_retries(self):
        embedding = OpenRouterEmbedding("m", api_key="k", max_retries=1)
        api = FakeAPI([http_error(429), http_error(429, b"slow down")])
        with mock.patch(URLOPEN, api), mock.patch(SLEEP):
            with self.assertRaisesRegex(RuntimeError, "429.*slow down"):
                embedding.encode(["a"])

    def test_client_errors_are_not_retried(self):
        api = FakeAPI([http_error(400, b'{"error": "unknown model"}')])
        with mock.patch(URLOPEN, api):
            with self.assertRaisesRegex(RuntimeError, "unknown model"):
                self.embedding.encode(["a"])
        self.assertEqual(len(api.requests), 1)

    def test_unexpected_response(self):
        def urlopen(request, timeout):
            return io.BytesIO(b'{"error": {"message": "no provider"}}')

        with mock.patch(URLOPEN, urlopen):
            with self.assertRaisesRegex(RuntimeError, "no provider"):
                self.embedding.encode(["a"])

    def test_api_key(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env"}):
            self.assertEqual(OpenRouterEmbedding("m")._api_key, "env")
        with mock.patch.dict(os.environ, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                OpenRouterEmbedding("m")
        self.assertNotIn("secret", repr(self.embedding))


if __name__ == "__main__":
    unittest.main()
