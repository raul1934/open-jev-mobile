#!/usr/bin/env python3
"""Convert intfloat/multilingual-e5-small to GGUF for the llama.cpp in the app.

The model is a BERT encoder with the XLM-RoBERTa SentencePiece vocabulary.
llama.cpp's converter supports that vocabulary (`_xlmroberta_set_vocab`) but
only selects it for XLM-RoBERTa architectures, so this wrapper routes
BertModel checkpoints that ship `sentencepiece.bpe.model` to it. The graph
stays plain BERT (absolute positions from 0), which is what e5 uses.

    python fraud/convert_encoder.py --llama-cpp C:/oj/llama.cpp-v041 --out fraud/model/e5-small-f16.gguf
"""
import argparse
import sys
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outtype", default="f16")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(MODEL_ID, revision=REVISION)

    sys.path.insert(0, str(args.llama_cpp))
    sys.path.insert(1, str(args.llama_cpp / "gguf-py"))
    import conversion.bert as bert

    original = bert.BertModel.set_vocab

    def set_vocab(self):
        if (self.dir_model / "sentencepiece.bpe.model").is_file():
            self._xlmroberta_set_vocab()
            self.gguf_writer.add_token_type_count(self.hparams.get("type_vocab_size", 1))
            return
        original(self)

    bert.BertModel.set_vocab = set_vocab
    import convert_hf_to_gguf
    sys.argv = ["convert_hf_to_gguf.py", model_dir, "--outtype", args.outtype, "--outfile", str(args.out)]
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()
