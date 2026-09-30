"""Convert a ModelOpt MXFP4 GPT-OSS export to the native GPT-OSS checkpoint layout.

ModelOpt exports all linears as ``weight``/``weight_scale`` pairs.  GPT-OSS's
native MXFP4 loader instead requires BF16 attention weights plus packed MoE
``*_blocks``/``*_scales`` tensors.  This converter retains the quantized
values, changing only their storage layout.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    """Decode ModelOpt's last-axis MXFP4 packing without large temporary tensors."""
    assert packed.shape[-1] * 2 == scales.shape[-1] * 32
    rows, packed_width = packed.reshape(-1, packed.shape[-1]).shape
    out_width = packed_width * 2
    out = torch.empty((rows, out_width), dtype=torch.bfloat16, device=packed.device)
    lut = FP4_VALUES.to(packed.device)
    flat_packed = packed.reshape(rows, packed_width)
    flat_scales = scales.reshape(rows, scales.shape[-1])
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        p = flat_packed[start:stop]
        codes = torch.empty((stop - start, out_width), dtype=torch.long, device=p.device)
        codes[:, 0::2] = (p & 0x0F).long()
        codes[:, 1::2] = (p >> 4).long()
        sign = 1.0 - 2.0 * ((codes & 0x08) >> 3).float()
        values = sign * lut[codes & 0x07]
        exponents = flat_scales[start:stop].float().repeat_interleave(32, dim=-1) - 127.0
        out[start:stop] = (values * torch.exp2(exponents)).to(torch.bfloat16)
    return out.reshape(*packed.shape[:-1], out_width)


def quant_native_mxfp4(weight: torch.Tensor, chunk_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an ``[..., out, in]`` BF16 tensor into GPT-OSS native MXFP4."""
    assert weight.shape[-1] % 32 == 0
    prefix = weight.shape[:-1]
    width = weight.shape[-1]
    rows = int(torch.tensor(prefix).prod().item())
    flat = weight.reshape(rows, width)
    blocks_out = torch.empty((rows, width // 2), dtype=torch.uint8, device=weight.device)
    scales_out = torch.empty((rows, width // 32), dtype=torch.uint8, device=weight.device)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=weight.device)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        values = flat[start:stop].reshape(-1, 32)
        amax = values.float().abs().amax(dim=-1, keepdim=True)
        exponent = torch.ceil(torch.maximum(torch.log2(amax / 6.0), torch.full_like(amax, -127.0)))
        normalized = values / torch.exp2(exponent)
        sign = ((2 - torch.sign(normalized)) // 2).to(torch.uint8)
        ordinal = torch.sum((normalized.abs().unsqueeze(-1) - bounds) > 0, dim=-1).to(torch.uint8)
        codes = (sign * 8 + ordinal).reshape(stop - start, width)
        blocks_out[start:stop] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales_out[start:stop] = (exponent + 127).to(torch.uint8).reshape(stop - start, width // 32)
    packed = blocks_out.reshape(*prefix, width // 32, 16)
    scales = scales_out.reshape(*prefix, width // 32)
    return packed, scales


def load_source_tensor(source_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    """Load a tensor even when ModelOpt put its companion scale in another shard."""
    with safe_open(source_dir / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def convert_shard(
    source: Path,
    source_dir: Path,
    weight_map: dict[str, str],
    destination: Path,
    chunk_rows: int,
) -> list[str]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        # Own a transformed pair in the shard containing its packed weight.
        # ModelOpt can place the paired scale in a different shard.
        expert_bases = {
            key for key in keys
            if key.endswith(".mlp.experts.gate_up_proj")
            or key.endswith(".mlp.experts.down_proj")
        }
        attention_bases = {
            key
            for key in keys
            if ".self_attn." in key and key.endswith(".weight")
        }
        # Every scale is consumed alongside its owner, so do not copy orphaned
        # scale keys from a different shard into the native artifact.
        skipped = {
            key for key in keys
            if key.endswith("_weight_scale")
            or (".self_attn." in key and key.endswith(".weight_scale"))
        }
        skipped |= expert_bases | attention_bases
        for key in keys:
            if key not in skipped:
                tensors[key] = handle.get_tensor(key)

        for base in sorted(attention_bases):
            packed = handle.get_tensor(base).cuda()
            scales = load_source_tensor(source_dir, weight_map, f"{base}_scale").cuda()
            tensors[base] = dequant_mxfp4(packed, scales, chunk_rows).cpu()
            del packed, scales
            torch.cuda.empty_cache()

        for base in sorted(expert_bases):
            packed = handle.get_tensor(base).cuda()
            scales = load_source_tensor(source_dir, weight_map, f"{base}_weight_scale").cuda()
            logical = dequant_mxfp4(packed, scales, chunk_rows).transpose(-1, -2).contiguous()
            blocks, native_scales = quant_native_mxfp4(logical, chunk_rows)
            native_base = base.replace("gate_up_proj", "gate_up_proj_blocks").replace(
                "down_proj", "down_proj_blocks"
            )
            tensors[native_base] = blocks.cpu()
            tensors[native_base.replace("_blocks", "_scales")] = native_scales.cpu()
            del packed, scales, logical, blocks, native_scales
            torch.cuda.empty_cache()
    save_file(tensors, str(destination), metadata={"format": "pt"})
    return list(tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    for path in args.source.iterdir():
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        if path.is_file():
            shutil.copy2(path, args.output / path.name)

    base_config = json.loads(args.base_config.read_text())
    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = base_config["quantization_config"]
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())
    source_weight_map = source_index["weight_map"]
    filenames = sorted(set(source_weight_map.values()))
    weight_map: dict[str, str] = {}
    for filename in filenames:
        print(f"Converting {filename}", flush=True)
        keys = convert_shard(
            args.source / filename, args.source, source_weight_map,
            args.output / filename, args.chunk_rows,
        )
        weight_map.update({key: filename for key in keys})
    total_size = sum(path.stat().st_size for path in args.output.glob("*.safetensors"))
    (args.output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": weight_map,
    }, indent=2) + "\n")
    (args.output / "FORMAT").write_text("mxfp4")
    print(f"Native MXFP4 checkpoint written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
