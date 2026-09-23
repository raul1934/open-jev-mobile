#!/usr/bin/env python3
"""Generate parity fixtures for the Kotlin port from the vendored Open-Jev modules.

For each request: the exact candidate prompts (jev.api.candidate_prompts), and
the typed response jev.serving.Predictor produces for deterministic fake
logits. The Kotlin unit tests must reproduce both. Standard library only:

    python android/tools/make_fixtures.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jev.api import candidate_prompts, compile_request  # noqa: E402
from jev.serving import Predictor  # noqa: E402

TEMPERATURE = 1.518796342858676

# Edge cases for the JSON rendering Open-Jev relies on (sort_keys, separators,
# float repr, unicode, escapes, nesting) and for every validation branch.
EDGE_CASES = {
    "edge-rendering": {
        "state": {"z": [1, 2.5, 1e-05, 1e16, 123456789012345678901234567890, -0.0, 0.1, 100.0, True, None],
                  "a": "acentuação 🙂 \"aspas\" \\ barra\ttab\nlinha \u0001",
                  "B": {"nested": {"k": []}, "empty": {}}, "é": 1},
        "questions": {
            "c": {"type": "choice", "instructions": {"task": "pick", "list": [3, 2, 1]},
                  "criteria": {"b": None, "a": {"why": "obj"}, "c": ["x", "y"], "d": "plain"}},
            "s": {"type": "score", "instructions": "rate", "criteria": ["low", {"level": "high"}, ["mid"]]},
            "n": {"type": "noul", "instructions": "yes?", "criteria": {"true": "it is", "false": {"no": 1}}},
        },
    },
    "edge-text-state": {"state": "plain text state", "questions": {
        "only": {"type": "choice", "instructions": "one option", "criteria": {"single": None}}}},
    "edge-list-state": {"state": [1, "two", {"3": 3}], "questions": {
        "q": {"type": "noul", "instructions": "list?"}}},
}

ERROR_CASES = [
    {"state": 5, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": "s", "questions": {}},
    {"state": "s", "questions": {"q": {"type": "other", "instructions": "x"}}},
    {"state": "s", "questions": {"q": {"type": "noul"}}},
    {"state": "s", "questions": {"q": {"type": "noul", "instructions": 3}}},
    {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a"]}}},
    {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": 1}}}},
    {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}},
    {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}}},
    {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "t"}}}},
    {"state": "s", "questions": {"q": "not a mapping"}},
]


def fake_logits(prompt):
    """Deterministic pseudo-logit in [-4, 4) derived from the prompt bytes."""
    digest = hashlib.sha256(prompt.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 * 8 - 4


class FakeScorer:
    def score(self, records):
        rows = []
        for record in records:
            values = [fake_logits(p) for p in candidate_prompts(record)]
            rows.append([0.0, values[0]] if record["kind"] == "noul" else values)
        return rows, 0


def main():
    requests = {}
    for path in sorted((ROOT / "examples").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"state", "questions"} <= data.keys():
            requests[path.name] = data
    requests.update(EDGE_CASES)
    predictor = Predictor(FakeScorer(), model_name="fixture", temperature=TEMPERATURE, batch_size=4)
    cases = []
    for name, request in requests.items():
        records = compile_request(request["state"], request["questions"])
        result = predictor.predict(request)
        cases.append({"name": name, "request": request,
                      "prompts": [p for r in records for p in candidate_prompts(r)],
                      "answers": result["answers"]})
    errors = []
    for request in ERROR_CASES:
        try:
            compile_request(request["state"], request["questions"])
        except ValueError as error:
            errors.append({"request": request, "error": str(error)})
        else:
            raise SystemExit(f"expected an error for {request}")
    out = ROOT / "android" / "app" / "src" / "test" / "resources" / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temperature": TEMPERATURE, "cases": cases, "errors": errors},
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases and {len(errors)} error cases to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
