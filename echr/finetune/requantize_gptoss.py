"""
Re-quantize a merged GPT-OSS bf16 model to MXFP4 using nvidia-modelopt.

The model was already LoRA-merged and saved as bf16 (218GB).
This script loads it, runs MXFP4 calibration, and exports with
export_hf_checkpoint which writes compressed MXFP4 safetensors
(~60GB) that vLLM can load natively on H100 or with DEQUANT=1 on A100.

Skips if FORMAT sentinel already says "mxfp4".

Usage:
    python echr/finetune/requantize_gptoss.py --article 3
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HF_HOME_DEFAULT = "/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--article", type=int, required=True)
    p.add_argument("--model_dir", type=str, default=None,
                   help="Merged bf16 model dir (default: data/models/gptoss_merged_art{N})")
    p.add_argument("--calib_data", type=str,
                   default="data/finetune/art{article}/sft_train.jsonl")
    p.add_argument("--hf_home", type=str, default=HF_HOME_DEFAULT)
    return p.parse_args()


def load_calib_texts(calib_path, tokenizer, n_samples=128, max_len=512):
    import json
    texts = []
    with open(calib_path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    texts.append(m["content"][:2000])
                    break
            if len(texts) >= n_samples:
                break
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    return enc


def main():
    args = parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_home, "transformers")

    model_dir = Path(args.model_dir or f"data/models/gptoss_merged_art{args.article}")
    sentinel = model_dir / "FORMAT"

    if sentinel.exists() and sentinel.read_text().strip() == "mxfp4":
        print(f"Already MXFP4: {model_dir}. Exiting.")
        sys.exit(0)

    current_fmt = sentinel.read_text().strip() if sentinel.exists() else "unknown"
    print(f"=== Requantizing art{args.article}: {current_fmt} → mxfp4 ===", flush=True)
    print(f"  Model dir: {model_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print("Loading bf16 merged model (device_map=auto across 4×A100)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    calib_path = args.calib_data.format(article=args.article)
    if not Path(calib_path).exists():
        calib_path = "data/finetune/sft_train.jsonl"
    print(f"Calibration data: {calib_path}", flush=True)

    calib_enc = load_calib_texts(calib_path, tokenizer)
    device = next(model.parameters()).device
    input_ids = calib_enc["input_ids"].to(device)
    attention_mask = calib_enc["attention_mask"].to(device)

    def forward_loop(m):
        with torch.no_grad():
            for i in range(0, len(input_ids), 8):
                m(input_ids=input_ids[i:i+8], attention_mask=attention_mask[i:i+8])

    print("Running MXFP4 calibration...", flush=True)
    model = mtq.quantize(model, mtq.MXFP4_DEFAULT_CFG, forward_loop)
    print("Quantization done.", flush=True)

    print(f"Exporting MXFP4 checkpoint to {model_dir} ...", flush=True)
    export_hf_checkpoint(model, export_dir=str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    sentinel.write_text("mxfp4")
    print(f"=== Done: art{args.article} exported as MXFP4 ===", flush=True)


if __name__ == "__main__":
    main()
