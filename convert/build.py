#!/usr/bin/env python3
"""Build a small GGUF of Open-Jev-2B plus its decision head, on a PC.

Steps (all pinned, all checked):
  1. download the released Open-Jev-2B package and the exact Qwen3.5-2B base
     revision it was trained on, and verify the package manifest digests;
  2. merge the LoRA adapter into the base weights (W + alpha/r * B @ A, in fp32);
  3. convert the text backbone to GGUF (F16) with llama.cpp's converter;
  4. quantize it (default Q5_K_M, see README for why not Q4_K_M) with llama-quantize;
  5. write head.json: decision-head weight and bias, calibration temperature,
     the exact chat-template wrapper, and provenance.

Needs about 15 GB of free disk (about 10 GB stays: base weights in the Hugging Face
cache, the F16 GGUF in --work, the quantized GGUF) and 8 GB of RAM. No GPU.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_REPO = "ZefanCai/Open-Jev-2B"
PACKAGE_REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
# Same source as the Termux `llama-cpp` package, so converter and runtime agree.
LLAMA_CPP_TAG = "v0.4.1"
SENTINEL = "OPEN_JEV_PROMPT"
ROOT = Path(__file__).resolve().parent.parent


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    """Pinned downloads into the normal Hugging Face cache (shared with verify.py and Open-Jev)."""
    from huggingface_hub import snapshot_download
    package = Path(snapshot_download(PACKAGE_REPO, revision=PACKAGE_REVISION))
    manifest = json.loads((package / "package" / "manifest.json").read_text())
    for name, entry in sorted(manifest["files"].items()):
        path = package / "package" / name
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            sys.exit(f"package file does not match its published manifest: {name}")
    checkpoint = package / "package" / "checkpoint"
    config = json.loads((checkpoint / "model.json").read_text())
    if (config["model_id"], config["revision"]) != (BASE_MODEL, BASE_REVISION):
        sys.exit(f"checkpoint expects {config['model_id']}@{config['revision']}")
    base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION,
                                  ignore_patterns=["*.pth", "*.bin", "*.gguf", "original/*"]))
    log("downloaded", package_files_verified=len(manifest["files"]))
    return checkpoint, base, config


def merge(checkpoint, base, out):
    """Fold the PEFT LoRA into the language-model weights; drop vision and MTP tensors.

    Merged projections are stored in fp32 so the adapter delta is not rounded
    away before llama.cpp writes F16; every other tensor keeps its bf16 bytes.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter_cfg = json.loads((checkpoint / "adapter" / "adapter_config.json").read_text())
    if adapter_cfg.get("use_dora") or adapter_cfg.get("use_rslora") or adapter_cfg.get("lora_bias"):
        sys.exit("unsupported adapter variant (DoRA / rsLoRA / LoRA bias)")
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]

    lora = {}
    with safe_open(checkpoint / "adapter" / "adapter_model.safetensors", "pt") as fh:
        for key in fh.keys():
            # base_model.model.layers.N.<module>.lora_A.weight
            #   -> model.language_model.layers.N.<module>.weight
            stem, part = key.removeprefix("base_model.model.").rsplit(".lora_", 1)
            lora.setdefault(f"model.language_model.{stem}.weight", {})[part[0]] = fh.get_tensor(key)
    if any(set(pair) != {"A", "B"} for pair in lora.values()):
        sys.exit("adapter has an unpaired lora_A / lora_B tensor")

    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(index.values())):
        with safe_open(base / shard, "pt") as fh:
            for key in fh.keys():
                if key.startswith(("model.visual.", "mtp.")):
                    continue
                tensor = fh.get_tensor(key)
                pair = lora.pop(key, None)
                if pair is not None:
                    tensor = tensor.float() + scale * (pair["B"].float() @ pair["A"].float())
                    tensors[key] = tensor.contiguous()
                else:
                    tensors[key] = tensor
    if lora:
        sys.exit(f"{len(lora)} adapter tensors have no base weight, e.g. {next(iter(lora))}")

    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    for item in base.iterdir():
        if item.is_file() and not item.name.startswith("model.safetensors") and item.suffix != ".md":
            shutil.copy2(item, out / item.name)
    log("merged", tensors=len(tensors), lora_scale=scale)


def prompt_wrapper(base):
    """The exact text Open-Jev feeds the model around each candidate prompt."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base)
    text = tokenizer.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    prefix, suffix = text.split(SENTINEL)
    return prefix, suffix


def ensure_llama_cpp(path):
    if not (path / "convert_hf_to_gguf.py").is_file():
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--branch",
                        LLAMA_CPP_TAG, "https://github.com/ggml-org/llama.cpp.git", str(path)], check=True)
    return path


def find_quantize(explicit, llama_cpp):
    candidates = [explicit] if explicit else []
    candidates += [shutil.which("llama-quantize"),
                   *[str(p) for p in llama_cpp.glob("build/bin/**/llama-quantize*")]]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    sys.exit("llama-quantize not found: pass --quantize /path/to/llama-quantize "
             "(it ships in the llama.cpp release zips and in Termux's llama-cpp package)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=ROOT / "build", help="F16 GGUF, llama.cpp checkout and temporary merge")
    parser.add_argument("--out", type=Path, default=ROOT / "models", help="where the GGUF and head.json go")
    parser.add_argument("--llama-cpp", type=Path, help=f"llama.cpp checkout (default: clone {LLAMA_CPP_TAG})")
    parser.add_argument("--quantize", help="path to the llama-quantize binary")
    parser.add_argument("--types", default="Q5_K_M",
                        help="comma-separated llama-quantize types, e.g. Q5_K_M,Q4_K_M (F16 is always kept in --work)")
    args = parser.parse_args()
    work, out = args.work.resolve(), args.out.resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    llama_cpp = ensure_llama_cpp((args.llama_cpp or work / "llama.cpp").resolve())

    checkpoint, base, config = download()
    f16 = work / "open-jev-2b-F16.gguf"
    if not f16.is_file():
        merged = work / "merged"
        if not (merged / "model.safetensors").is_file():
            merge(checkpoint, base, merged)
        subprocess.run([sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged),
                        "--outtype", "f16", "--no-mtp", "--outfile", str(f16)], check=True)
        shutil.rmtree(merged)  # 4+ GB intermediate; the F16 GGUF holds the same weights
    log("converted", gguf=str(f16), bytes=f16.stat().st_size)

    for qtype in [t.strip() for t in args.types.split(",") if t.strip()]:
        target = out / f"open-jev-2b-{qtype}.gguf"
        if qtype.upper() == "F16":
            shutil.copy2(f16, target)
        elif not target.is_file():
            subprocess.run([find_quantize(args.quantize, llama_cpp), str(f16), str(target), qtype], check=True)
        log("quantized", type=qtype, gguf=target.name, bytes=target.stat().st_size, sha256=sha256(target))

    import torch
    head = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    weight, bias = head["weight"].float().reshape(-1), head["bias"].float().reshape(-1)
    temperature = json.loads((checkpoint / "temperature.json").read_text())["temperature"]
    prefix, suffix = prompt_wrapper(base)
    record = {
        "model_name": "Open-Jev-2B-gguf",
        "package_repo": PACKAGE_REPO, "package_revision": PACKAGE_REVISION,
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "llama_cpp": LLAMA_CPP_TAG,
        "hidden_size": weight.numel(), "max_length": config["max_length"],
        "temperature": temperature, "prompt_prefix": prefix, "prompt_suffix": suffix,
        "bias": float(bias[0]), "weight": [float(v) for v in weight],
    }
    # One head for every quantization: it reads the backbone's hidden state, not its weights.
    (out / "head.json").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    log("head", path=str(out / "head.json"))


if __name__ == "__main__":
    main()
