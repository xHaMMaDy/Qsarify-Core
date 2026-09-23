"""Generate manuscript-supporting tables and figures from benchmark outputs.

This script deliberately consumes existing benchmark artefacts rather than
retraining models.  It produces deterministic, publication-friendly CSV/TeX
tables and PNG figures, together with a manifest that records the input
checksums and plotting configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
MODEL_ORDER = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
MODEL_LABELS = {
    "LogisticRegression": "Logistic regression",
    "RandomForestClassifier": "Random forest",
    "GradientBoostingClassifier": "Gradient boosting",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


_path_anchor = Path(__file__).resolve().parents[1]
if (_path_anchor / "app.py").is_file():
    PROJECT_ROOT = _path_anchor.parent
else:
    PROJECT_ROOT = _path_anchor


def portable_project_path(path: Path) -> str:
    """Return a repository-relative path for portable output manifests."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summary_rows(summary: dict) -> list[dict]:
    rows = []
    aggregate = summary.get("aggregate", {})
    model_names = [name for name in MODEL_ORDER if name in aggregate]
    model_names.extend(name for name in aggregate if name not in model_names)
    for model in model_names:
        values = aggregate[model]
        row = {"model": model, "model_label": MODEL_LABELS.get(model, model)}
        for metric in METRICS:
            if metric not in values:
                raise ValueError(f"Repeated summary is missing metric '{metric}' for {model}.")
            row[f"{metric}_mean"] = float(values[metric]["mean"])
            row[f"{metric}_std"] = float(values[metric]["sample_std"])
        rows.append(row)
    if not rows:
        raise ValueError("Repeated summary contains no model aggregates.")
    return rows


def write_table(rows: list[dict], output_dir: Path) -> Path:
    columns = ["model", "model_label"]
    for metric in METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    frame = pd.DataFrame(rows, columns=columns)
    csv_path = output_dir / "benchmark_summary.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8f")

    display = pd.DataFrame({"Model": [row["model_label"] for row in rows]})
    for metric in METRICS:
        display[METRIC_LABELS[metric]] = [
            f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}" for row in rows
        ]
    tex_path = output_dir / "benchmark_summary.tex"
    tex_path.write_text(
        display.to_latex(index=False, escape=False, column_format="l" + "c" * len(METRICS)),
        encoding="utf-8",
    )
    return csv_path


def _split_label(split_strategy: str) -> tuple[str, str]:
    if split_strategy == "scaffold":
        return "scaffold", "scaffold-disjoint"
    if split_strategy == "random":
        return "random", "stratified random"
    safe = split_strategy.replace(" ", "-").lower()
    return safe, split_strategy


def plot_metric_summary(rows: list[dict], output_dir: Path, split_strategy: str) -> Path:
    file_label, display_label = _split_label(split_strategy)
    models = [row["model_label"] for row in rows]
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = ["#475569", "#be123c", "#0f766e"]
    for index, row in enumerate(rows):
        means = [row[f"{metric}_mean"] for metric in METRICS]
        errors = [row[f"{metric}_std"] for metric in METRICS]
        ax.bar(
            x + (index - (len(rows) - 1) / 2) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=models[index],
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xticks(x, [METRIC_LABELS[metric] for metric in METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± sample SD)")
    ax.set_title(f"Repeated {display_label} benchmark")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    path = output_dir / f"benchmark_metrics_{file_label}_repeated.png"
    fig.savefig(path, dpi=220, metadata={"Software": "QSARify publication-output generator"})
    plt.close(fig)
    return path


def plot_seed_curves(results: dict, output_dir: Path, seed: str, split_strategy: str) -> tuple[Path, Path]:
    file_label, display_label = _split_label(split_strategy)
    roc_fig, roc_ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    pr_fig, pr_ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    colors = ["#475569", "#be123c", "#0f766e"]
    for index, model in enumerate(MODEL_ORDER):
        if model not in results:
            continue
        payload = results[model]
        roc = payload.get("roc_curve", {})
        pr = payload.get("pr_curve", {})
        label = f"{MODEL_LABELS.get(model, model)} (AUC={payload['roc_auc']:.3f})"
        roc_ax.plot(roc["fpr"], roc["tpr"], color=colors[index % len(colors)], label=label)
        pr_label = f"{MODEL_LABELS.get(model, model)} (AUC={payload['pr_auc']:.3f})"
        pr_ax.plot(pr["recall"], pr["precision"], color=colors[index % len(colors)], label=pr_label)

    roc_ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC curves ({display_label} seed {seed})")
    roc_ax.grid(alpha=0.25)
    roc_ax.legend(frameon=False, fontsize=8, loc="lower right")
    roc_path = output_dir / f"roc_curves_{file_label}_seed_{seed}.png"
    roc_fig.savefig(roc_path, dpi=220, metadata={"Software": "QSARify publication-output generator"})
    plt.close(roc_fig)

    pr_ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall curves ({display_label} seed {seed})")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.05)
    pr_ax.grid(alpha=0.25)
    pr_ax.legend(frameon=False, fontsize=8, loc="lower left")
    pr_path = output_dir / f"pr_curves_{file_label}_seed_{seed}.png"
    pr_fig.savefig(pr_path, dpi=220, metadata={"Software": "QSARify publication-output generator"})
    plt.close(pr_fig)
    return roc_path, pr_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeated-summary", type=Path, required=True)
    parser.add_argument("--seed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.repeated_summary.resolve()
    seed_results_path = args.seed_results.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(summary_path)
    seed_results = load_json(seed_results_path)
    rows = summary_rows(summary)
    split_strategy = summary.get("split_strategy", "unknown")
    write_table(rows, output_dir)
    plot_metric_summary(rows, output_dir, split_strategy)
    seed = seed_results_path.parent.name.removeprefix("seed-")
    plot_seed_curves(seed_results, output_dir, seed, split_strategy)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "repeated_summary": {"path": portable_project_path(summary_path), "sha256": sha256_file(summary_path)},
            "seed_results": {"path": portable_project_path(seed_results_path), "sha256": sha256_file(seed_results_path)},
        },
        "configuration": {
            "metrics": METRICS,
            "model_order": MODEL_ORDER,
            "seed": seed,
            "split_strategy": split_strategy,
            "matplotlib_backend": "Agg",
        },
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
