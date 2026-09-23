"""Scorer backed by a running llama-server; standard library only.

Open-Jev scores each candidate as ``head(h_last)``, where ``h_last`` is the
final-norm hidden state of the last prompt token of the LoRA-merged Qwen
backbone. llama-server started with ``--embeddings --pooling last`` returns
exactly that vector (``embd_normalize: -1`` turns off its L2 normalization),
so the head is a single dot product done here in Python.
"""
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

from jev.api import candidate_prompts


class LlamaServerError(RuntimeError):
    pass


def load_head(path):
    head = json.loads(Path(path).read_text(encoding="utf-8"))
    weight, bias = head["weight"], float(head["bias"])
    if len(weight) != head["hidden_size"] or not all(math.isfinite(w) for w in weight):
        raise ValueError("head.json weight does not match hidden_size or is not finite")
    if not math.isfinite(head["temperature"]) or head["temperature"] <= 0:
        raise ValueError("head.json temperature must be finite and positive")
    return head


class LlamaServerScorer:
    """Implements the ``score(records) -> (logits, input_tokens)`` interface of ``jev.serving``."""

    def __init__(self, head, url="http://127.0.0.1:8792", timeout=600):
        self.head = head
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.weight = [float(w) for w in head["weight"]]
        self.bias = float(head["bias"])
        self.prefix, self.suffix = head["prompt_prefix"], head["prompt_suffix"]
        self.max_length = int(head["max_length"])

    def _post(self, path, body):
        request = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise LlamaServerError(f"llama-server {path} returned {error.code}: {detail}") from None
        except urllib.error.URLError as error:
            raise LlamaServerError(f"cannot reach llama-server at {self.url}: {error.reason}") from None

    def check(self):
        """Fail early if the server is not an embeddings server for the expected model."""
        props = self._get("/props")
        n_embd = self._post("/v1/embeddings", {"input": [[9175]], "embd_normalize": -1})["data"][0]["embedding"]
        if len(n_embd) != self.head["hidden_size"]:
            raise LlamaServerError(
                f"server returns {len(n_embd)}-dim embeddings; head expects {self.head['hidden_size']}. "
                "Is llama-server running the Open-Jev GGUF with --embeddings --pooling last?")
        # A smaller server context (to save RAM) lowers the prompt limit; say so
        # up front instead of failing inside llama-server.
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if isinstance(n_ctx, int) and 0 < n_ctx < self.max_length:
            self.max_length = n_ctx
        return props

    def _get(self, path):
        try:
            with urllib.request.urlopen(self.url + path, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.URLError as error:
            raise LlamaServerError(f"cannot reach llama-server at {self.url}: {error}") from None

    def tokenize(self, text):
        # parse_special: the chat-template markers must become their special tokens,
        # exactly as the Hugging Face tokenizer produces them.
        return self._post("/tokenize", {"content": text, "add_special": False,
                                        "parse_special": True})["tokens"]

    def prompts(self, records):
        return [self.prefix + prompt + self.suffix for record in records for prompt in candidate_prompts(record)]

    def hidden_states(self, token_lists):
        data = self._post("/v1/embeddings", {"input": token_lists, "embd_normalize": -1,
                                             "encoding_format": "float"})["data"]
        rows = sorted(data, key=lambda item: item["index"])
        if len(rows) != len(token_lists):
            raise LlamaServerError("llama-server returned the wrong number of embeddings")
        return [row["embedding"] for row in rows]

    def score(self, records):
        tokens = [self.tokenize(prompt) for prompt in self.prompts(records)]
        longest = max(len(t) for t in tokens)
        if longest > self.max_length:
            raise ValueError(f"Input length {longest} exceeds max_length={self.max_length}; no silent truncation"
                             + (" (raise JEV_CTX, up to 4096)" if self.max_length < self.head["max_length"] else ""))
        scores = [math.fsum(w * h for w, h in zip(self.weight, hidden)) + self.bias
                  for hidden in self.hidden_states(tokens)]
        # Same assembly as jev.model.DecisionModel.forward.
        logits, offset = [], 0
        for record in records:
            count = len(candidate_prompts(record))
            values = scores[offset:offset + count]
            logits.append([0.0, values[0]] if record["kind"] == "noul" else values)
            offset += count
        return logits, sum(len(t) for t in tokens)
