"""Command line: one-off prediction, or the Open-Jev HTTP API on top of llama-server.

    python -m jev_mobile predict examples/01-en-support-routing.json
    python -m jev_mobile serve --port 8791
"""
import argparse
import json
import sys
from pathlib import Path

from jev.server import make_server, strict_json
from jev.serving import Predictor

from .scorer import LlamaServerError, LlamaServerScorer, load_head

ROOT = Path(__file__).resolve().parent.parent


def build_predictor(args):
    head = load_head(args.head)
    scorer = LlamaServerScorer(head, url=args.llama_url)
    scorer.check()
    provenance = {key: head[key] for key in ("package_repo", "package_revision", "base_model",
                                             "base_revision", "llama_cpp") if key in head}
    return Predictor(scorer, model_name=head["model_name"], temperature=head["temperature"],
                     batch_size=args.batch_size, method="lora_decision_head_gguf", provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jev_mobile", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default=str(ROOT / "models" / "head.json"),
                        help="head.json produced by convert/build.py")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8792",
                        help="llama-server started by termux/start.sh")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="candidate sequences per llama-server request")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="score one request file and print the typed answers")
    predict.add_argument("request", help="JSON file with state and questions ('-' for stdin)")
    serve = sub.add_parser("serve", help="expose /v1/inference like `python -m jev.server`")
    serve.add_argument("--host", default="127.0.0.1",
                       help="use 0.0.0.0 to accept requests from other devices on your Wi-Fi")
    serve.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)

    try:
        predictor = build_predictor(args)
    except (LlamaServerError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.command == "predict":
        raw = sys.stdin.buffer.read() if args.request == "-" else Path(args.request).read_bytes()
        result = predictor.predict(strict_json(raw))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    server = make_server(predictor, args.host, args.port, static_root=ROOT / "examples")
    print(json.dumps({"url": f"http://{args.host}:{server.server_port}", "model": predictor.model_name,
                      "method": predictor.method}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
