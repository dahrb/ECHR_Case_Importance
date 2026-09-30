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
from collections import Counter
import json
import os
import sys
from pathlib import Path

import torch

HF_HOME_DEFAULT = "/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"
BASE_MODEL_ID = "openai/gpt-oss-120b"


def restore_native_gptoss_mxfp4_metadata(output_dir: Path, base_config) -> None:
    """Keep ModelOpt's exported MXFP4 tensors on vLLM's native GPT-OSS path.

    ModelOpt writes ``quant_method=modelopt`` metadata.  vLLM routes that to
    its ModelOpt loader, which does not accept MXFP4.  GPT-OSS itself has a
    native MXFP4 loader and selects it from the base model's metadata.
    """
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = base_config.quantization_config
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # This sidecar is ModelOpt-specific and can otherwise be discovered ahead
    # of the corrected config by downstream tooling. Preserve it for audit.
    sidecar = output_dir / "hf_quant_config.json"
    if sidecar.exists():
        sidecar.rename(output_dir / "hf_quant_config.modelopt_export.json")


def chunked_mxfp4_weight_scale(module, weight_name: str):
    """Compute MXFP4 E8M0 scales without materialising the 7-way FP4 LUT tensor.

    ModelOpt's generic helper calls ``MXFP4QTensor.quantize`` just to retain
    its scale output.  For a fused GPT-OSS MoE expert it creates a temporary
    ``[..., 7]`` comparison tensor (>55 GiB on an A100).  Scales only require
    the 32-value block amax, which can be evaluated in small flat chunks.
    """
    weight = getattr(module, weight_name)
    quantizer = getattr(module, f"{weight_name}_weight_quantizer")
    block_size = quantizer.block_sizes[-1]
    flat = weight.view(-1, block_size)
    chunk_rows = int(os.environ.get("MXFP4_SCALE_CHUNK_ROWS", "1000000"))
    scale_chunks = []
    for start in range(0, flat.shape[0], chunk_rows):
        block_amax = flat[start:start + chunk_rows].abs().amax(dim=-1, keepdim=True)
        exponent = torch.ceil(torch.maximum(
            torch.log2(block_amax.float() / 6.0),
            torch.full_like(block_amax.float(), -127.0),
        ))
        scale_chunks.append((exponent + 127).to(torch.uint8))
    return torch.cat(scale_chunks, dim=0).reshape(*weight.shape[:-1], -1)


def chunked_mxfp4_quantize(cls, input: torch.Tensor, block_size: int | None):
    """Bounded-memory equivalent of ``MXFP4QTensor.quantize`` for export."""
    if block_size is None:
        block_size = 32
    original_shape, original_dtype = input.shape, input.dtype
    flat = input.view(-1, block_size)
    chunk_rows = int(os.environ.get("MXFP4_SCALE_CHUNK_ROWS", "1000000"))
    packed_chunks, scale_chunks = [], []
    bounds = cls.E2M1_bounds.to(input.device)
    for start in range(0, flat.shape[0], chunk_rows):
        chunk = flat[start:start + chunk_rows]
        amax = chunk.float().abs().amax(dim=-1, keepdim=True)
        exponent = torch.ceil(torch.maximum(
            torch.log2(amax / cls.E2M1_max), torch.full_like(amax, -127.0)
        ))
        normalized = chunk / torch.exp2(exponent)
        sign_bit = (2 - torch.sign(normalized)) // 2
        ordinal = torch.sum((normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1)
        fp4 = (sign_bit * 0b1000 + ordinal).to(torch.uint8)
        left, right = fp4[..., 0::2], fp4[..., 1::2]
        packed = (right.clone() << 4)
        packed[..., :left.shape[-1]] += left
        packed_chunks.append(packed)
        scale_chunks.append((exponent + 127).to(torch.uint8))
    packed_shape = (*original_shape[:-1], (original_shape[-1] + 1) // 2)
    packed_data = torch.cat(packed_chunks, dim=0).reshape(packed_shape)
    scales = torch.cat(scale_chunks, dim=0)
    return cls(original_shape, original_dtype, packed_data), scales


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--article", type=int, required=True)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hf_home", type=str, default=HF_HOME_DEFAULT)
    p.add_argument("--skip_mxfp4", action="store_true",
                   help="Skip MXFP4 requantization, always save bf16")
    p.add_argument("--calib_data", type=str,
                   default="data/finetune_v2/art{article}/sft_train.jsonl",
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

    # Leaving Accelerate unconstrained lets ``device_map=auto`` pack GPU 0 to
    # ~75 GiB, leaving too little workspace for ModelOpt's MXFP4 exporter.
    # A 68-GiB per-GPU placement keeps the 120B bf16 model resident on four
    # A100s while reserving enough headroom for the export's largest shard.
    gpu_budget = os.environ.get("GPU_MAX_MEMORY", "68GiB")
    max_memory = {index: gpu_budget for index in range(torch.cuda.device_count())}
    print(
        "Loading base model (MXFP4 → bf16, balanced device map; "
        f"max_memory={max_memory})...",
        flush=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=Mxfp4Config(dequantize=True),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        cache_dir=os.path.join(args.hf_home, "hub"),
        trust_remote_code=True,
    )
    placement = Counter(str(device) for device in base_model.hf_device_map.values())
    print(f"Model module placement: {dict(placement)}", flush=True)

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
            # Quantization leaves reclaimable allocator segments on every
            # device.  Release them before export, which needs one additional
            # temporary shard-sized allocation on the most-loaded GPU.
            import gc
            gc.collect()
            for device_index in range(torch.cuda.device_count()):
                with torch.cuda.device(device_index):
                    torch.cuda.empty_cache()
            if os.environ.get("STREAM_MXFP4_EXPORT", "0") == "1":
                # The streaming exporter normally performs these model-wide
                # preparation passes itself.  Do them while Accelerate's
                # original balanced device map is intact, before weights are
                # made meta by CPU offload.  In particular, GPT-OSS's MoE
                # resmoothing needs a real multi-GPU forward.
                from modelopt.torch.export.unified_export_hf import (
                    _prepare_moe_inputs,
                    _warn_on_unsynced_moe_gate_up,
                    requantize_resmooth_fused_llm_layers,
                )
                export_dtype = next(model.parameters()).dtype
                print("Preparing GPT-OSS MoE export before CPU offload...", flush=True)
                _prepare_moe_inputs(model, export_dtype, is_modelopt_qlora=False)
                requantize_resmooth_fused_llm_layers(model)
                _warn_on_unsynced_moe_gate_up(model)

                # ModelOpt detects Accelerate offload hooks and switches to its
                # layer-streaming writer.  This avoids the resident exporter's
                # rank-0 whole-state-dict peak that exceeds four 80GB A100s.
                from accelerate import cpu_offload
                from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
                import modelopt.torch.export.unified_export_hf_streaming as streaming_export

                decoder_layers = LayerActivationCollector.get_decoder_layers(model)
                if not decoder_layers:
                    raise RuntimeError("Cannot discover decoder layers for streaming MXFP4 export")
                for layer in decoder_layers:
                    # Preserve the original balanced GPU placement.  Sending
                    # all four shards to cuda:0 makes GPT-OSS residual paths
                    # cross devices during ModelOpt's dummy forward.
                    layer_device = next(layer.parameters()).device
                    if layer_device.type != "cuda":
                        raise RuntimeError(f"Decoder layer is not CUDA-resident: {layer_device}")
                    cpu_offload(layer, execution_device=layer_device, offload_buffers=True)
                # Offload non-decoder owners too.  A GPT-OSS expert export
                # temporarily needs ~55 GiB of workspace; leaving embedding,
                # heads and dispatch leftovers resident leaves too little room
                # on its assigned A100.  The writer's preliminary passes are
                # already complete, so it never needs ``model.device`` after
                # this point.
                standalone = 0
                decoder_owned = {id(mod) for layer in decoder_layers for mod in layer.modules()}
                for module in model.modules():
                    if id(module) in decoder_owned:
                        continue
                    own_parameters = [p for p in module.parameters(recurse=False) if p is not None]
                    if not own_parameters:
                        continue
                    owner_device = own_parameters[0].device
                    if owner_device.type != "cuda":
                        raise RuntimeError(f"Standalone owner is not CUDA-resident: {owner_device}")
                    cpu_offload(module, execution_device=owner_device, offload_buffers=True)
                    standalone += 1
                # Reuse the pre-offload preparation above.  The upstream
                # streaming function imports these names into its module, so
                # patch only its local bindings rather than site-packages.
                streaming_export._prepare_moe_inputs = lambda *args, **kwargs: None
                streaming_export.requantize_resmooth_fused_llm_layers = lambda *args, **kwargs: None
                streaming_export._warn_on_unsynced_moe_gate_up = lambda *args, **kwargs: None
                # Avoid ModelOpt's unchunked 7-way FP4 lookup temporary for
                # the fused GPT-OSS expert tensors.  This replacement is
                # mathematically the scale half of MXFP4QTensor.quantize.
                import modelopt.torch.export.unified_export_hf as unified_export
                from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor
                original_weight_scale = unified_export.get_weight_scaling_factor

                def export_weight_scale(module, weight_name="weight"):
                    if type(module).__name__ == "QuantGptOssExperts":
                        return chunked_mxfp4_weight_scale(module, weight_name)
                    return original_weight_scale(module, weight_name)

                unified_export.get_weight_scaling_factor = export_weight_scale
                MXFP4QTensor.quantize = classmethod(chunked_mxfp4_quantize)
                print(
                    "Enabling layer-wise CPU offload for streaming MXFP4 export "
                    f"({len(decoder_layers)} decoders, {standalone} standalone owners)...",
                    flush=True,
                )
            print(f"Exporting MXFP4 checkpoint to {output_dir}...", flush=True)
            export_hf_checkpoint(model, export_dir=str(output_dir))
            restore_native_gptoss_mxfp4_metadata(output_dir, base_model.config)
            tokenizer.save_pretrained(str(output_dir))
            sentinel.write_text("mxfp4")
            print(f"=== Done: art{args.article} saved as mxfp4 ===", flush=True)
            sys.exit(0)

        except Exception as e:
            print(f"MXFP4 requantization failed ({e})", flush=True)
            if os.environ.get("REQUIRE_MXFP4", "0") == "1":
                raise
            print("Saving bf16 fallback.", flush=True)
            saved_format = "bf16"

    print(f"Saving {saved_format} model to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    sentinel.write_text(saved_format)
    print(f"=== Done: art{args.article} saved as {saved_format} ===", flush=True)


if __name__ == "__main__":
    main()
