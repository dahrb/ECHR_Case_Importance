"""Completion-only LoRA fine-tuning for ECHR importance prediction.

Each example is tokenized before it reaches TRL. If it is too long, only the
end of the user/case text is removed; the entire gold assistant completion is
always retained and is the only part contributing to the loss.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

HF_HOME_DEFAULT = "/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenized_example(record: dict[str, Any], tokenizer: Any,
                      max_length: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    messages = record["messages"]
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("Each record must contain exactly one user and one assistant message")

    user_text = messages[0]["content"]
    assistant_text = messages[1]["content"]
    user_tokens = tokenizer.encode(user_text, add_special_tokens=False)

    def render(keep: int) -> tuple[list[int], list[int]]:
        shortened_user = tokenizer.decode(user_tokens[:keep], skip_special_tokens=True)
        prompt_messages = [{"role": "user", "content": shortened_user}]
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat-template prompt is not a prefix of the completed conversation")
        return prompt_ids, full_ids

    prompt_ids, full_ids = render(len(user_tokens))
    original_length = len(full_ids)
    truncated = original_length > max_length
    if truncated:
        low, high = 0, len(user_tokens)
        best: tuple[list[int], list[int]] | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = render(middle)
            if len(candidate[1]) <= max_length:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError(
                f"Gold completion alone does not fit max_length={max_length}; "
                f"filename={record.get('metadata', {}).get('filename')}"
            )
        prompt_ids, full_ids = best

    completion_tokens = len(full_ids) - len(prompt_ids)
    if completion_tokens <= 0 or len(full_ids) > max_length:
        raise AssertionError("Invalid completion mask or sequence length")
    example = {
        "input_ids": full_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * completion_tokens,
    }
    stats = {
        "filename": record.get("metadata", {}).get("filename"),
        "original_tokens": original_length,
        "final_tokens": len(full_ids),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
        "truncated": truncated,
    }
    return example, stats


def prepare_dataset(records: list[dict[str, Any]], tokenizer: Any,
                    max_length: int, name: str) -> tuple[Dataset, dict[str, Any]]:
    examples, rows = zip(*(tokenized_example(r, tokenizer, max_length) for r in records))
    report = {
        "split": name,
        "examples": len(rows),
        "truncated_examples": sum(row["truncated"] for row in rows),
        "max_original_tokens": max(row["original_tokens"] for row in rows),
        "max_final_tokens": max(row["final_tokens"] for row in rows),
        "min_completion_tokens": min(row["completion_tokens"] for row in rows),
        "max_completion_tokens": max(row["completion_tokens"] for row in rows),
        "all_completions_retained": True,
    }
    print(json.dumps(report, indent=2), flush=True)
    return Dataset.from_list(list(examples)), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--hf_home", default=HF_HOME_DEFAULT)
    parser.add_argument("--load_mode", default="auto",
                        choices=["auto", "bnb4bit", "asis", "mxfp4_dequant"])
    parser.add_argument("--reasoning_effort", default=None,
                        choices=["low", "medium", "high"],
                        help="Recorded inference/evaluation setting for GPT-OSS")
    args = parser.parse_args()

    if args.epochs != 2:
        raise ValueError("This thesis rerun is fixed at exactly two epochs")
    if "gpt-oss" in args.model_id.lower() and args.reasoning_effort != "medium":
        raise ValueError("GPT-OSS reruns must record --reasoning_effort medium")

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{args.hf_home}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{args.hf_home}/transformers"
    hf_key = Path(args.hf_home).parent / "LLM_Experiments" / "hf.key"
    if hf_key.exists():
        token = hf_key.read_text().strip()
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, cache_dir=f"{args.hf_home}/hub"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = load_jsonl(args.train_dataset)
    val_records = load_jsonl(args.val_dataset)
    train_ds, train_token_audit = prepare_dataset(
        train_records, tokenizer, args.max_seq_len, "train"
    )
    val_ds, val_token_audit = prepare_dataset(
        val_records, tokenizer, args.max_seq_len, "validation"
    )

    load_mode = args.load_mode
    if load_mode == "auto":
        model_name = args.model_id.lower()
        load_mode = "asis" if "fp8" in model_name else (
            "mxfp4_dequant" if "gpt-oss" in model_name else "bnb4bit"
        )
    print(f"Model: {args.model_id}\nLoad mode: {load_mode}", flush=True)
    common = dict(device_map="auto", trust_remote_code=True,
                  cache_dir=f"{args.hf_home}/hub", attn_implementation="flash_attention_2")
    if load_mode == "bnb4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=quantization,
            torch_dtype=torch.bfloat16, **common
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        checkpoint_kwargs = {"use_reentrant": False}
    elif load_mode == "mxfp4_dequant":
        from transformers import Mxfp4Config
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=Mxfp4Config(dequantize=True),
            torch_dtype=torch.bfloat16, **common
        )
        model.enable_input_require_grads()
        checkpoint_kwargs = {"use_reentrant": True}
    elif load_mode == "asis":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, **common
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.enable_input_require_grads()
        checkpoint_kwargs = {"use_reentrant": False}
    else:
        raise ValueError(load_mode)

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(output_dir), overwrite_output_dir=False,
        num_train_epochs=2, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=checkpoint_kwargs,
        optim="adamw_8bit", learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio, fp16=False, bf16=True,
        logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, report_to="none", dataloader_num_workers=2,
        max_length=args.max_seq_len, packing=False, completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_ds,
        eval_dataset=val_ds, args=config,
    )
    trainer.train()
    adapter_path = output_dir / "adapter_final"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    record = {
        "base_model": args.model_id, "load_mode": load_mode,
        "train_dataset": args.train_dataset, "val_dataset": args.val_dataset,
        "epochs": 2, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": args.lr, "max_seq_len": args.max_seq_len,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_target_modules": LORA_TARGET_MODULES,
        "loss_scope": "assistant_completion_only",
        "truncation_policy": "truncate_case_prompt_end_preserve_full_completion",
        "reasoning_effort_for_inference": args.reasoning_effort,
        "token_audit": {"train": train_token_audit, "validation": val_token_audit},
    }
    (output_dir / "finetune_config.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Adapter saved to {adapter_path}", flush=True)


if __name__ == "__main__":
    main()
