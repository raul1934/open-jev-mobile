#!/usr/bin/env python3
"""Compare the GGUF pipeline against the original Open-Jev PyTorch model.

1. Runs the released checkpoint exactly as upstream loads it
   (``jev.model.DecisionModel.load`` from an Open-Jev checkout, bf16, CPU)
   on a set of request files and caches its raw logits.
2. For every GGUF given, starts llama-server, scores the same records through
   ``jev_mobile``, and reports: token-ID agreement with the Hugging Face
   tokenizer, max |logit| difference, max |probability| difference after the
   calibration temperature, and how often the top answer agrees.

Usage (from the repository root, on a PC):
    python convert/verify.py --open-jev ../Open-Jev --llama-server /path/to/llama-server \
        models/open-jev-2b-Q4_K_M.gguf models/open-jev-2b-Q8_0.gguf
"""
import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_requests(open_jev, limit):
    files = [open_jev / "configs" / "example-request.json",
             *sorted((open_jev / "examples" / "recipes").glob("*.json")),
             *sorted((open_jev / "examples" / "community").glob("*.json"))]
    requests = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"state", "questions"} <= data.keys():
            requests.append((path.relative_to(open_jev).as_posix(), data))
    return requests[:limit]


def reference_logits(open_jev, checkpoint, requests, cache):
    if cache.is_file():
        return json.loads(cache.read_text())
    sys.path.insert(0, str(open_jev))
    import torch
    from jev.api import candidate_prompts, compile_request
    from jev.model import DecisionModel
    model = DecisionModel.load(checkpoint, device="cpu")
    out = {}
    for name, request in requests:
        records = compile_request(request["state"], request["questions"])
        start = time.perf_counter()
        with torch.inference_mode():
            rows = [row.float().tolist() for row in model(records)]
        prompts = [model.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in records for p in candidate_prompts(r)]
        ids = [model.tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]
        out[name] = {"logits": rows, "token_ids": ids, "seconds": time.perf_counter() - start}
        print(json.dumps({"event": "reference", "request": name, "candidates": len(ids),
                          "seconds": round(out[name]["seconds"], 1)}), flush=True)
    cache.write_text(json.dumps(out))
    del model
    return out


def wait_ready(url, process, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def softmax(row, temperature):
    top = max(row)
    exps = [math.exp((v - top) / temperature) for v in row]
    total = sum(exps)
    return [e / total for e in exps]


def compare(gguf, args, requests, reference):
    sys.path.insert(0, str(ROOT))
    from jev.api import compile_request
    from jev_mobile.scorer import LlamaServerScorer, load_head

    head = load_head(args.head)
    port = args.port
    command = [args.llama_server, "-m", str(gguf), "--embeddings", "--pooling", "last",
               "-c", str(args.ctx), "-b", str(args.ctx), "-ub", str(args.ctx), "-np", "1",
               "--host", "127.0.0.1", "--port", str(port), "-t", str(args.threads)]
    log = open(gguf.with_suffix(".server.log"), "w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        wait_ready(url, process)
        scorer = LlamaServerScorer(head, url=url)
        scorer.check()
        worst_logit = worst_prob = 0.0
        prob_diffs = []
        agree = total = token_mismatch = 0
        seconds = 0.0
        for name, request in requests:
            records = compile_request(request["state"], request["questions"])
            ref = reference[name]
            tokens = [scorer.tokenize(p) for p in scorer.prompts(records)]
            token_mismatch += sum(a != b for a, b in zip(tokens, ref["token_ids"]))
            start = time.perf_counter()
            rows, _ = scorer.score(records)
            seconds += time.perf_counter() - start
            for record, got, want in zip(records, rows, ref["logits"]):
                worst_logit = max(worst_logit, max(abs(a - b) for a, b in zip(got, want)))
                p_got, p_want = softmax(got, head["temperature"]), softmax(want, head["temperature"])
                diff = max(abs(a - b) for a, b in zip(p_got, p_want))
                worst_prob = max(worst_prob, diff)
                prob_diffs.append(diff)
                total += 1
                agree += p_got.index(max(p_got)) == p_want.index(max(p_want))
        return {"gguf": gguf.name, "bytes": gguf.stat().st_size, "questions": total,
                "token_mismatches": token_mismatch, "max_abs_logit_diff": round(worst_logit, 4),
                "max_abs_probability_diff": round(worst_prob, 4),
                "mean_abs_probability_diff": round(sum(prob_diffs) / len(prob_diffs), 4),
                "median_abs_probability_diff": round(sorted(prob_diffs)[len(prob_diffs) // 2], 4),
                "top_answer_agreement": f"{agree}/{total}",
                "gguf_seconds": round(seconds, 1),
                "reference_seconds": round(sum(reference[n]["seconds"] for n, _ in requests), 1)}
    finally:
        process.terminate()
        process.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", nargs="+", type=Path)
    parser.add_argument("--open-jev", type=Path, required=True, help="Open-Jev checkout (for jev.model)")
    parser.add_argument("--checkpoint", type=Path,
                        help="Open-Jev checkpoint dir (default: the pinned package build.py downloaded)")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--head", type=Path, default=ROOT / "models" / "head.json",
                        help="head.json from build.py (the head is the same for every quantization)")
    parser.add_argument("--limit", type=int, default=12, help="number of request files")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port", type=int, default=18792)
    parser.add_argument("--report", type=Path, default=ROOT / "convert" / "verify-report.json")
    args = parser.parse_args()
    requests = load_requests(args.open_jev.resolve(), args.limit)
    cache = args.report.with_name("reference-logits.json")
    if args.checkpoint is None:
        sys.path.insert(0, str(ROOT / "convert"))
        from build import PACKAGE_REPO, PACKAGE_REVISION
        from huggingface_hub import snapshot_download
        args.checkpoint = Path(snapshot_download(PACKAGE_REPO, revision=PACKAGE_REVISION,
                                                 local_files_only=True)) / "package" / "checkpoint"
    reference = reference_logits(args.open_jev.resolve(), args.checkpoint.resolve(), requests, cache)
    results = []
    for gguf in args.gguf:
        result = compare(gguf.resolve(), args, requests, reference)
        print(json.dumps(result), flush=True)
        results.append(result)
    args.report.write_text(json.dumps({"requests": [n for n, _ in requests], "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
