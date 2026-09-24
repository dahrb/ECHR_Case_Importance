"""
QLoRA Fine-tuning for ECHR Case Importance Prediction.

Adapted from ADM_JURIX/ADM/finetune_qwen.py. Supports:
  - nvidia/Llama-3.3-70B-Instruct-FP8
  - openai/gpt-oss-120b

4-bit NF4 QLoRA via BitsAndBytes. LoRA applied to attention + MLP projections.
SFT format: ChatML messages column; only assistant turns are trained on.

Usage (Llama):
    python echr/finetune/finetune_echr.py \\
        --model_id nvidia/Llama-3.3-70B-Instruct-FP8 \\
        --train_dataset data/finetune/sft_train.jsonl \\
        --val_dataset data/finetune/sft_val.jsonl \\
        --output_dir data/models/llama_lora \\
        --epochs 3

Usage (GPT-OSS):
    python echr/finetune/finetune_echr.py \\
        --model_id openai/gpt-oss-120b \\
        --output_dir data/models/gptoss_lora \\
        --epochs 3
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

HF_HOME_DEFAULT = "/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"


class DataCollatorForCompletionOnlyLM:
    """Masks prompt tokens so loss is computed only on completion tokens.
    Drop-in replacement for trl's removed DataCollatorForCompletionOnlyLM (trl>=1.0).
    """
    def __init__(self, response_template, tokenizer, ignore_index=-100):
        self.response_template = list(response_template)
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, features):
        import torch
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                      padding_value=self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        tlen = len(self.response_template)
        for i in range(labels.shape[0]):
            ids = labels[i].tolist()
            # Find last occurrence of response_template tokens
            start = None
            for j in range(len(ids) - tlen, -1, -1):
                if ids[j:j + tlen] == self.response_template:
                    start = j + tlen
                    break
            if start is None:
                labels[i, :] = self.ignore_index
            else:
                labels[i, :start] = self.ignore_index
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LENGTH_DEFAULT = 4096


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nvidia/Llama-3.3-70B-Instruct-FP8")
    parser.add_argument("--train_dataset", type=str, default="data/finetune/sft_train.jsonl")
    parser.add_argument("--val_dataset", type=str, default="data/finetune/sft_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/models/llama_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH_DEFAULT)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--hf_home", type=str, default=HF_HOME_DEFAULT)
    parser.add_argument("--load_mode", type=str, default="auto",
                        choices=["auto", "bnb4bit", "asis", "mxfp4_dequant"],
                        help="auto: FP8→asis, gpt-oss→mxfp4_dequant, else bnb4bit")
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable in-loop eval (avoids OOM on huge MoE models; "
                             "eval logits over full vocab are memory-heavy). "
                             "Adapter selected as final epoch, not best-val.")
    parser.add_argument("--harmony_sft", action="store_true",
                        help="Use harmony-channel SFT: training data has reasoning_content "
                             "(analysis channel) + answer fields; sequences are pre-formatted "
                             "with both <|channel|>analysis and <|channel|>final tokens. "
                             "Use with traces_train.jsonl / traces_val.jsonl from "
                             "gen_reasoning_traces.py. Incompatible with packing.")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{args.hf_home}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{args.hf_home}/transformers"

    hf_key = os.path.join(
        os.path.dirname(args.hf_home), "LLM_Experiments", "hf.key"
    )
    if os.path.exists(hf_key):
        token = open(hf_key).read().strip()
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token

    print(f"Model: {args.model_id}", flush=True)
    print(f"Train: {args.train_dataset}  Val: {args.val_dataset}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=f"{args.hf_home}/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Resolve load mode. Neither of our target checkpoints is plain bf16:
    #   - nvidia/Llama-3.3-70B-Instruct-FP8 → FP8 (ModelOpt/compressed-tensors);
    #     bnb 4-bit cannot re-quantize float8 → load AS-IS, train LoRA on top.
    #   - openai/gpt-oss-120b → MXFP4; dequantize to bf16 for training, then LoRA.
    #   - a genuine bf16 base → standard bnb 4-bit QLoRA.
    load_mode = args.load_mode
    if load_mode == "auto":
        mid = args.model_id.lower()
        if "fp8" in mid:
            load_mode = "asis"
        elif "gpt-oss" in mid or "gpt_oss" in mid:
            load_mode = "mxfp4_dequant"
        else:
            load_mode = "bnb4bit"
    print(f"Load mode: {load_mode}", flush=True)

    common = dict(
        device_map="auto",
        trust_remote_code=True,
        cache_dir=f"{args.hf_home}/hub",
        attn_implementation="flash_attention_2",
    )

    if load_mode == "bnb4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading model in 4-bit NF4 QLoRA...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, **common,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif load_mode == "mxfp4_dequant":
        from transformers import Mxfp4Config
        print("Loading MXFP4 model dequantized to bf16 (LoRA on top)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=Mxfp4Config(dequantize=True),
            torch_dtype=torch.bfloat16, **common,
        )
        # NOTE: gradient checkpointing is enabled by SFTConfig (use_reentrant=True).
        # gpt-oss is MoE — non-reentrant checkpointing raises a metadata-mismatch
        # error because expert routing differs between forward and recompute.
        model.enable_input_require_grads()
    elif load_mode == "asis":
        print("Loading pre-quantized model as-is (LoRA on frozen base)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, **common,
        )
        # Freeze all base params; LoRA adapters (added below) remain trainable.
        for p in model.parameters():
            p.requires_grad_(False)
        model.enable_input_require_grads()
    else:
        raise ValueError(f"Unknown load_mode: {load_mode}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = load_jsonl(args.train_dataset)
    val_data = load_jsonl(args.val_dataset)
    print(f"Train: {len(train_data)}  Val: {len(val_data)}", flush=True)

    data_collator = None  # set below for harmony_sft mode

    if args.harmony_sft:
        # Harmony-channel SFT: training data has reasoning_content + answer fields.
        # We pre-format each sequence as:
        #   <system>...<user>...<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
        #   <|start|>assistant<|channel|>final<|message|>{answer}<|return|>
        # Loss is computed only on tokens after the first <|start|>assistant (both channels).
        print("Harmony SFT mode: pre-formatting sequences with analysis+final channels", flush=True)

        # DataCollatorForCompletionOnlyLM defined at module level (removed in trl>=1.0)

        def _format_harmony(ex: dict) -> dict:
            user_content = ex["messages"][0]["content"]
            reasoning = ex.get("reasoning_content", "") or ""
            answer = ex["answer"]

            # Prompt: system (auto) + user, ending with generation prompt = <|start|>assistant
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=True,
                add_generation_prompt=True,
            )

            # Completion: analysis channel (may be empty) then final channel
            if reasoning.strip():
                completion = (
                    f"<|channel|>analysis<|message|>{reasoning}<|end|>"
                    f"<|start|>assistant<|channel|>final<|message|>{answer}<|return|>"
                )
            else:
                # Fallback: no reasoning trace available → skip analysis channel
                completion = f"<|channel|>final<|message|>{answer}<|return|>"

            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            full_ids = (prompt_ids + completion_ids)[:args.max_seq_len]
            return {"text": tokenizer.decode(full_ids, skip_special_tokens=False)}

        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)
        train_ds = train_ds.map(_format_harmony, remove_columns=train_ds.column_names)
        if not args.no_eval:
            val_ds = val_ds.map(_format_harmony, remove_columns=val_ds.column_names)

        # Mask everything up to (and including) the first <|start|>assistant token;
        # train on both the analysis and final channel content that follow.
        response_template_ids = tokenizer.encode("<|start|>assistant", add_special_tokens=False)
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
        use_packing = False
        dataset_text_field = "text"
    else:
        train_ds = Dataset.from_list([{"messages": ex["messages"]} for ex in train_data])
        val_ds = Dataset.from_list([{"messages": ex["messages"]} for ex in val_data])
        use_packing = True
        dataset_text_field = None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        logging_steps=5,
        eval_strategy="no" if args.no_eval else "epoch",
        save_strategy="epoch",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=(not args.no_eval),
        metric_for_best_model=None if args.no_eval else "eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        max_length=args.max_seq_len,
        dataset_text_field=dataset_text_field,
        packing=use_packing,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None if args.no_eval else val_ds,
        args=sft_config,
        data_collator=data_collator,
        callbacks=None if args.no_eval else [
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    checkpoints = (
        sorted(Path(args.output_dir).glob("checkpoint-*"),
               key=lambda p: int(p.name.split("-")[1]))
        if args.resume_from_checkpoint else []
    )
    resume = checkpoints[-1] if checkpoints else False

    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    adapter_path = os.path.join(args.output_dir, "adapter_final")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved → {adapter_path}", flush=True)

    config_record = {
        "base_model": args.model_id,
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": LORA_TARGET_MODULES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
    }
    with open(os.path.join(args.output_dir, "finetune_config.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
