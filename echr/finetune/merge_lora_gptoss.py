"""
Merge a GPT-OSS-120B LoRA adapter into the base model.

Loads the MXFP4 base model dequantized to bf16, merges the LoRA adapter via
PEFT merge_and_unload(), then attempts to re-quantize to MXFP4 using
nvidia-modelopt. Falls back to saving bf16 if modelopt is unavailable.

Usage:
    python echr/finetune/merge_lora_gptoss.py \
        --article 3 \
        --adapter_dir data/models/gptoss_lora_art3/adapter_final \
        --output_dir data/models/gptoss_merged_art3

Output sentinel: {output_dir}/FORMAT  (contains "mxfp4" or "bf16")
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HF_HOME_DEFAULT = "/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"
BASE_MODEL_ID = "openai/gpt-oss-120b"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--article", type=int, required=True)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hf_home", type=str, default=HF_HOME_DEFAULT)
    p.add_argument("--skip_mxfp4", action="store_true",
                   help="Skip MXFP4 requantization, always save bf16")
    p.add_argument("--calib_data", type=str,
                   default="data/finetune/art{article}/sft_train.jsonl",
                   help="Calibration JSONL for MXFP4 quantization")
    return p.parse_args()


def load_calib_texts(calib_path: str, tokenizer, n_samples: int = 128, max_len: int = 512):
    import json
    texts = []
    with open(calib_path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages", [])
            # Use first user message as calibration text
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

    # Set HF cache
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(args.hf_home, "transformers")

    output_dir = Path(args.output_dir)
    sentinel = output_dir / "FORMAT"

    if sentinel.exists():
        fmt = sentinel.read_text().strip()
        print(f"Sentinel exists: {output_dir} already merged as {fmt}. Exiting.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        print(f"ERROR: adapter_dir not found: {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Merging LoRA art{args.article} ===", flush=True)
    print(f"  Base: {BASE_MODEL_ID}", flush=True)
    print(f"  Adapter: {adapter_dir}", flush=True)
    print(f"  Output: {output_dir}", flush=True)

    # Load base model: MXFP4 → bf16 via dequantize
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import PeftModel

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=os.path.join(args.hf_home, "hub"),
        trust_remote_code=True,
    )

    print("Loading base model (MXFP4 → bf16, device_map=auto)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.path.join(args.hf_home, "hub"),
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging and unloading LoRA...", flush=True)
    model = model.merge_and_unload()
    model.eval()

    # Try MXFP4 requantization
    saved_format = "bf16"
    if not args.skip_mxfp4:
        try:
            import modelopt.torch.quantization as mtq
            print("nvidia-modelopt available — attempting MXFP4 requantization...", flush=True)

            calib_path = args.calib_data.format(article=args.article)
            if not Path(calib_path).exists():
                # Try the combined training file
                calib_path = "data/finetune/sft_train.jsonl"
            print(f"  Calibration data: {calib_path}", flush=True)

            calib_enc = load_calib_texts(calib_path, tokenizer)
            device = next(model.parameters()).device
            input_ids = calib_enc["input_ids"].to(device)
            attention_mask = calib_enc["attention_mask"].to(device)

            def forward_loop(m):
                with torch.no_grad():
                    for i in range(0, len(input_ids), 8):
                        batch_ids = input_ids[i:i+8]
                        batch_mask = attention_mask[i:i+8]
                        m(input_ids=batch_ids, attention_mask=batch_mask)

            model = mtq.quantize(model, mtq.MXFP4_DEFAULT_CFG, forward_loop)
            print("MXFP4 quantization done.", flush=True)
            from modelopt.torch.export import export_hf_checkpoint
            print(f"Exporting MXFP4 checkpoint to {output_dir}...", flush=True)
            export_hf_checkpoint(model, export_dir=str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            sentinel.write_text("mxfp4")
            print(f"=== Done: art{args.article} saved as mxfp4 ===", flush=True)
            sys.exit(0)

        except Exception as e:
            print(f"MXFP4 requantization failed ({e}), saving as bf16.", flush=True)
            saved_format = "bf16"

    print(f"Saving {saved_format} model to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    sentinel.write_text(saved_format)
    print(f"=== Done: art{args.article} saved as {saved_format} ===", flush=True)


if __name__ == "__main__":
    main()
