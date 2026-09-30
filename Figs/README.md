# Figure style guide

All figures in this project must be publication-ready and follow these rules:

- Export figures as high-resolution PNG files only.
- Do not include a title or subtitle within the figure; use the manuscript caption instead.
- Do not include grid lines or other background lines.
- Keep the background white and use clear, consistent typography and labels.
- Retain only essential axes and panel headings needed to interpret the figure.

## Baseline confusion matrices

`create_baseline_confusion_matrices.py` produces
`baseline_confusion_matrices.png`, a single 3-by-4 panel figure covering the
zero-shot and few-shot GPT-OSS-120B and Llama-3.3-70B-Instruct (FP8) baselines
for Articles 3, 6, and 8. The plotted counts and valid-coverage audit are saved
to `baseline_confusion_matrix_counts.csv`.

## Experiment 1 confusion matrices

`create_experiment1_confusion_matrices.py` produces
`experiment1_best_confusion_matrices.png`. For each of Articles 3, 6, and 8,
the left panel shows the complete Experiment 1 configuration with the lowest
MAE and the right panel shows the configuration with the highest SRC. The
underlying counts, row percentages, metrics, and source paths are saved to
`experiment1_confusion_matrix_counts.csv`.

## Full results atlas and thesis review

The [results atlas](results_atlas/README.md) contains 15 high-resolution PNG
figures, suggested captions, audited source CSVs and reproduction instructions.
Run `create_results_atlas.py` to regenerate them from the current saved results.
The [narrative review](results_atlas/NARRATIVE_REVIEW.md) compares these results
with the ICAIL paper and documents the newly identified date-cutoff and gold-link
issues that must be considered before making final forecasting claims.
