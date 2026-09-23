"""Scorer tests against a stub llama-server; standard library only.

    python -m unittest discover tests
"""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev.serving import Predictor  # noqa: E402
from jev_mobile.scorer import LlamaServerScorer  # noqa: E402

HIDDEN = 4


def fake_tokens(text):
    return [ord(c) % 1000 for c in text]


def fake_embedding(tokens):
    # Deterministic "hidden state": depends on the whole prompt.
    return [float(len(tokens) % 7), float(sum(tokens) % 11), 1.0, -1.0]


class Stub(BaseHTTPRequestHandler):
    n_ctx = 4096

    def log_message(self, *args):
        pass

    def reply(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply({"default_generation_settings": {"n_ctx": self.n_ctx}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            assert body["parse_special"] is True
            return self.reply({"tokens": fake_tokens(body["content"])})
        assert self.path == "/v1/embeddings" and body["embd_normalize"] == -1
        self.reply({"data": [{"index": i, "embedding": fake_embedding(t)}
                             for i, t in reversed(list(enumerate(body["input"])))]})


HEAD = {"model_name": "test", "hidden_size": HIDDEN, "max_length": 4096, "temperature": 1.5,
        "prompt_prefix": "<u>", "prompt_suffix": "</u>", "weight": [0.5, -0.25, 1.0, 0.0], "bias": 0.1}

REQUEST = {"state": "A customer was charged twice.", "questions": {
    "route": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "charges", "security": "access", "technical": None}},
    "dup": {"type": "noul", "instructions": "Duplicate charge?"},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}}}


class ScorerTest(unittest.TestCase):
    def setUp(self):
        Stub.n_ctx = 4096
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.scorer = LlamaServerScorer(HEAD, url=f"http://127.0.0.1:{self.server.server_port}")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def expected(self, prompt):
        hidden = fake_embedding(fake_tokens("<u>" + prompt + "</u>"))
        return sum(w * h for w, h in zip(HEAD["weight"], hidden)) + HEAD["bias"]

    def test_logits_match_decision_model_layout(self):
        from jev.api import candidate_prompts, compile_request
        records = compile_request(REQUEST["state"], REQUEST["questions"])
        rows, tokens = self.scorer.score(records)
        for record, row in zip(records, rows):
            want = [self.expected(p) for p in candidate_prompts(record)]
            if record["kind"] == "noul":
                want = [0.0, want[0]]
            self.assertEqual(len(row), len(want))
            for got, value in zip(row, want):
                self.assertAlmostEqual(got, value, places=9)
        self.assertGreater(tokens, 0)

    def test_predictor_end_to_end(self):
        self.scorer.check()
        result = Predictor(self.scorer, model_name="test", temperature=HEAD["temperature"], batch_size=2).predict(REQUEST)
        answers = result["answers"]
        self.assertEqual(set(answers), {"route", "dup", "urgency"})
        self.assertAlmostEqual(sum(answers["route"]["probabilities"].values()), 1.0)
        self.assertTrue(0 <= answers["dup"]["noul"] <= 1)

    def test_small_server_context_lowers_limit(self):
        Stub.n_ctx = 8
        self.scorer.check()
        from jev.api import compile_request
        with self.assertRaisesRegex(ValueError, "JEV_CTX"):
            self.scorer.score(compile_request(REQUEST["state"], REQUEST["questions"]))


if __name__ == "__main__":
    unittest.main()
