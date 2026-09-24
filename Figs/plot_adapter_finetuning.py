#!/usr/bin/env python3
"""Plot publication-ready learning curves for Llama and GPT-OSS adapters.

Llama uses answer-only QLoRA with epoch-end validation. GPT-OSS uses the
currently served Harmony-SFT adapters (reasoning + answer targets), trained
with evaluation disabled. The two objectives are therefore shown in separate
panels and GPT-OSS validation metrics are not inferred.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "Figs"
RUNS = [
    ("Llama 3.3 70B", "Article 3", ROOT / "data/models/llama_lora_art3"),
    ("Llama 3.3 70B", "Article 6", ROOT / "data/models/llama_lora_art6"),
    ("Llama 3.3 70B", "Article 8", ROOT / "data/models/llama_lora_art8"),
    ("GPT-OSS 120B", "Article 3", ROOT / "data/models/gptoss_lora_art3"),
    ("GPT-OSS 120B", "Article 6", ROOT / "data/models/gptoss_lora_art6"),
    ("GPT-OSS 120B", "Article 8", ROOT / "data/models/gptoss_lora_art8"),
]

# Okabe-Ito palette, safe for common colour-vision deficiencies.
COLORS = {
    "Article 3": "#0072B2",
    "Article 6": "#D55E00",
    "Article 8": "#009E73",
}
MARKERS = {"Article 3": "o", "Article 6": "s", "Article 8": "D"}


def final_state(adapter_dir: Path) -> Path:
    checkpoints = sorted(
        adapter_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No trainer state found under {adapter_dir}")
    return checkpoints[-1]


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_run(model: str, adapter: str, adapter_dir: Path) -> tuple[list[dict], dict]:
    state_path = final_state(adapter_dir)
    state = json.loads(state_path.read_text())
    config = json.loads((adapter_dir / "finetune_config.json").read_text())
    rows: list[dict] = []

    for record in state["log_history"]:
        common = {
            "model": model,
            "adapter": adapter,
            "step": int(record["step"]),
            "epoch": float(record["epoch"]),
            "source": str(state_path.relative_to(ROOT)),
        }
        if "loss" in record and finite(record.get("grad_norm")):
            # Article 3's Llama state retains an invalid FP8-base prefix with
            # NaN gradient norms. Requiring a finite gradient rejects it while
            # preserving legitimate early GPT-OSS losses above 5.
            rows.append(
                common
                | {
                    "split": "train",
                    "loss": float(record["loss"]),
                    "token_accuracy": float(record["mean_token_accuracy"]),
                }
            )
        elif "eval_loss" in record and float(record["eval_loss"]) < 5.0:
            rows.append(
                common
                | {
                    "split": "validation",
                    "loss": float(record["eval_loss"]),
                    "token_accuracy": float(record["eval_mean_token_accuracy"]),
                }
            )

    metadata = {
        "model": model,
        "adapter": adapter,
        "training_target": "reasoning + answer" if model.startswith("GPT") else "answer only",
        "train_examples": int(config["train_examples"]),
        "validation_examples": int(config["val_examples"]),
        "epochs": int(config["epochs"]),
        "maximum_steps": int(state["max_steps"]),
        "base_model": config["base_model"],
    }
    return rows, metadata


def write_data(rows: list[dict], metadata: list[dict]) -> None:
    with (OUTPUT_DIR / "adapter_finetuning_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for meta in metadata:
        run = [
            row for row in rows
            if row["model"] == meta["model"] and row["adapter"] == meta["adapter"]
        ]
        train = [row for row in run if row["split"] == "train"]
        valid = [row for row in run if row["split"] == "validation"]
        best = min(valid, key=lambda row: row["loss"]) if valid else None
        summaries.append(
            meta
            | {
                "first_recorded_train_loss": train[0]["loss"],
                "last_recorded_train_loss": train[-1]["loss"],
                "train_loss_reduction_percent": 100
                * (train[0]["loss"] - train[-1]["loss"])
                / train[0]["loss"],
                "final_train_token_accuracy": train[-1]["token_accuracy"],
                "best_validation_loss": "" if best is None else best["loss"],
                "best_validation_epoch": "" if best is None else best["epoch"],
                "best_validation_token_accuracy": "" if best is None else best["token_accuracy"],
            }
        )

    with (OUTPUT_DIR / "adapter_finetuning_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 3.08)
    axis.set_xticks([0, 1, 2, 3])
    axis.set_xlabel("Epoch")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot(rows: list[dict]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.7), constrained_layout=True)
    llama_train, llama_valid, gpt_loss, gpt_accuracy = axes.flat

    for model, adapters in (("Llama 3.3 70B", COLORS), ("GPT-OSS 120B", list(COLORS)[:3])):
        for adapter in adapters:
            train = [
                row for row in rows
                if row["model"] == model and row["adapter"] == adapter
                and row["split"] == "train"
            ]
            if not train:
                continue
            color, marker = COLORS[adapter], MARKERS[adapter]
            loss_axis = llama_train if model.startswith("Llama") else gpt_loss
            loss_axis.plot(
                [row["epoch"] for row in train],
                [row["loss"] for row in train],
                color=color,
                linewidth=1.45,
                marker=marker,
                markersize=3.0,
                markeredgewidth=0,
            )
            if model.startswith("GPT"):
                gpt_accuracy.plot(
                    [row["epoch"] for row in train],
                    [row["token_accuracy"] for row in train],
                    color=color,
                    linewidth=1.45,
                    marker=marker,
                    markersize=3.0,
                    markeredgewidth=0,
                )

    for adapter in COLORS:
        valid = [
            row for row in rows
            if row["model"] == "Llama 3.3 70B" and row["adapter"] == adapter
            and row["split"] == "validation"
        ]
        color, marker = COLORS[adapter], MARKERS[adapter]
        llama_valid.plot(
            [row["epoch"] for row in valid],
            [row["loss"] for row in valid],
            color=color,
            linewidth=1.45,
            marker=marker,
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        best = min(valid, key=lambda row: row["loss"])
        llama_valid.scatter(
            best["epoch"], best["loss"], s=76, marker="*", facecolor=color,
            edgecolor="black", linewidth=0.4, zorder=5,
        )

    for axis in axes.flat:
        style_axis(axis)

    llama_train.set_title("a  Llama: training loss", loc="left", fontweight="bold")
    llama_train.set_ylabel("Cross-entropy loss")
    llama_train.set_ylim(0.65, 2.02)

    llama_valid.set_title("b  Llama: validation loss", loc="left", fontweight="bold")
    llama_valid.set_ylabel("Cross-entropy loss")
    llama_valid.set_ylim(0.94, 1.50)
    llama_valid.set_xticks([1, 2, 3])
    llama_valid.legend(
        handles=[Line2D([0], [0], marker="*", linestyle="none", color="black",
                        markerfacecolor="#777777", markersize=8, label="Best checkpoint")],
        frameon=False,
        loc="upper right",
    )

    gpt_loss.set_title("c  GPT-OSS: training loss", loc="left", fontweight="bold")
    gpt_loss.set_ylabel("Cross-entropy loss (log scale)")
    gpt_loss.set_yscale("log")
    gpt_loss.set_ylim(0.3, 7.5)
    gpt_loss.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 4, 7]))
    gpt_loss.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    gpt_loss.grid(axis="y", which="minor", visible=False)

    gpt_accuracy.set_title("d  GPT-OSS: training token accuracy", loc="left", fontweight="bold")
    gpt_accuracy.set_ylabel("Mean token accuracy")
    gpt_accuracy.set_ylim(0.1, 0.93)

    article_handles = [
        Line2D([0], [0], color=COLORS[name], marker=MARKERS[name], linewidth=1.6,
               markersize=4.2, label=name)
        for name in COLORS
    ]
    fig.legend(
        handles=article_handles, frameon=False, ncol=4, loc="outside upper center",
        handlelength=2.3, columnspacing=1.7,
    )

    fig.savefig(OUTPUT_DIR / "adapter_finetuning_curves.png", dpi=600)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metadata: list[dict] = []
    for model, adapter, directory in RUNS:
        run_rows, run_metadata = load_run(model, adapter, directory)
        rows.extend(run_rows)
        metadata.append(run_metadata)
    write_data(rows, metadata)
    plot(rows)


if __name__ == "__main__":
    main()
