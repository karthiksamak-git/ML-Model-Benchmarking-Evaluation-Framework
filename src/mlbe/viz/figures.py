"""Figure generation from the exported web JSON (single source of truth).

Colorblind-safe palette (Okabe-Ito), fixed style, 200 dpi PNG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # grey
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "figure.autolayout": True,
        }
    )


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _color(model: str) -> str:
    return PALETTE[abs(hash(model)) % len(PALETTE)] if False else PALETTE[sum(map(ord, model)) % len(PALETTE)]


def leaderboard_ci(leaderboard: dict[str, Any], out: Path) -> Path:
    rows = leaderboard["overall"]
    rows = sorted(rows, key=lambda r: r["accuracy"])
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(rows) + 1.6))
    y = np.arange(len(rows))
    vals = [r["accuracy"] for r in rows]
    err = np.array([[r["accuracy"] - r["ci_low"], r["ci_high"] - r["accuracy"]] for r in rows]).T
    ax.barh(y, vals, xerr=np.clip(err, 0, None), color=[_color(r["model"]) for r in rows], alpha=0.85, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r.get("label", r["model"]) for r in rows])
    for i, r in enumerate(rows):
        ax.text(r["accuracy"] + 0.005, i, f"{r['accuracy']:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean CV accuracy (bootstrap 95% CI over folds)")
    lo = max(0.0, min(vals) - 0.08)
    ax.set_xlim(lo, min(1.005, max(r["ci_high"] for r in rows) + 0.03))
    ax.set_title("Leaderboard — mean accuracy across datasets")
    return _save(fig, out, "leaderboard_ci.png")


def critical_difference(stats: dict[str, Any], out: Path) -> Path:
    cd = stats["friedman"].get("critical_difference")
    ranks = stats["friedman"].get("avg_ranks", [])
    models = stats["friedman"].get("models", [])
    if not cd or not ranks:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "Not enough complete datasets/models for Friedman + Nemenyi", ha="center")
        ax.axis("off")
        return _save(fig, out, "critical_difference.png")
    order = np.argsort(ranks)
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(models) + 1.4))
    lo, hi = min(ranks) - 0.4, max(ranks) + cd + 0.6
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, len(models) - 0.2)
    ax.grid(axis="x", alpha=0.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Average rank (lower is better)")
    # CD bar
    ax.plot([hi - cd, hi], [len(models) - 0.7] * 2, color="black", lw=1.5)
    ax.text(hi - cd / 2, len(models) - 0.55, f"CD = {cd:.2f}", ha="center", fontsize=8)
    for row, idx in enumerate(order):
        model = models[idx]
        x = ranks[idx]
        ax.plot([x, hi - 0.1], [row, row], color="#CCCCCC", lw=0.8, zorder=0)
        ax.scatter([x], [row], s=70, color=_color(model), zorder=2)
        ax.text(x, row + 0.18, model, ha="center", fontsize=8)
    ax.set_title("Critical-difference diagram (Friedman + Nemenyi, α=0.05)")
    return _save(fig, out, "critical_difference.png")


def pareto(pareto_data: dict[str, Any], out: Path) -> Path:
    datasets = pareto_data["datasets"]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=False):
        pts = ds["points"]
        for p in pts:
            color = PALETTE[0] if p["pareto"] else "#BBBBBB"
            ax.scatter(p["latency_p50_ms"], p["accuracy"], color=color, s=40, zorder=2)
            ax.annotate(
                p["model"],
                (p["latency_p50_ms"], p["accuracy"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Inference latency p50 (ms, log)")
        ax.set_ylabel("Accuracy")
        ax.set_title(ds["dataset"])
    axes[0][0].legend(["Pareto front", "dominated"], loc="lower right", fontsize=7)
    fig.suptitle("Accuracy vs latency Pareto front (blue = non-dominated)", fontsize=11, fontweight="bold")
    return _save(fig, out, "pareto.png")


def learning_curves(gen: dict[str, Any], out: Path) -> Path:
    rows = gen.get("learning_curves", [])
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(3.6 * max(1, len(datasets)), 3.0), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=False):
        sub = [r for r in rows if r["dataset"] == ds]
        agg: dict[float, list[float]] = {}
        for r in sub:
            agg.setdefault(r["fraction"], []).append(r["accuracy"])
        xs = sorted(agg)
        ys = [float(np.mean(agg[x])) for x in xs]
        ax.plot([x * 100 for x in xs], ys, marker="o", color=PALETTE[0])
        ax.set_xlabel("% of training fold used")
        ax.set_ylabel("Fold accuracy")
        ax.set_title(f"{ds} — {sub[0]['model'] if sub else ''}")
    fig.suptitle("Learning curves", fontsize=11, fontweight="bold")
    return _save(fig, out, "learning_curves.png")


def robustness_curves(gen: dict[str, Any], out: Path) -> Path:
    rows = gen.get("robustness", [])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for (ds, kind), color in zip(
        sorted({(r["dataset"], r["kind"]) for r in rows}),
        PALETTE,
        strict=False,
    ):
        sub = sorted([r for r in rows if r["dataset"] == ds and r["kind"] == kind], key=lambda r: r["level"])
        ax.plot(
            [r["level"] for r in sub],
            [r["accuracy"] for r in sub],
            marker="o",
            label=f"{ds} / {kind}",
            color=color,
        )
    ax.set_xlabel("Noise level (noise = std multiples, shift = std shift)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Robustness under Gaussian noise and covariate shift")
    return _save(fig, out, "robustness.png")


def calibration_plots(errors: dict[str, Any], out: Path) -> Path:
    results = errors["results"]
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0), squeeze=False)
    for ax, r in zip(axes[0], results, strict=False):
        bins = r["calibration"]["bins"]
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
        ax.plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color=PALETTE[1],
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{r['dataset']} (ECE {r['calibration']['ece']:.3f})")
    fig.suptitle("Reliability diagrams", fontsize=11, fontweight="bold")
    return _save(fig, out, "calibration.png")


def confusion_matrices(errors: dict[str, Any], out: Path) -> Path:
    results = errors["results"]
    n = max(1, len(results))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for ax, r in zip(axes[0], results, strict=False):
        cm = np.asarray(r["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r["labels"])))
        ax.set_yticks(range(len(r["labels"])))
        ax.set_xticklabels(r["labels"], fontsize=7)
        ax.set_yticklabels(r["labels"], fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(r["dataset"])
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, label="Row-normalised rate")
    fig.suptitle("Confusion matrices (row-normalised)", fontsize=11, fontweight="bold")
    return _save(fig, out, "confusion_matrices.png")


def rubric_bars(dataset_rubric: dict[str, Any], model_rubric: dict[str, Any], out: Path) -> Path:
    checks = dataset_rubric["checks"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ids = [c["id"] for c in checks]
    values = [c["score"] for c in checks]
    colors = [PALETTE[0] if c["severity"] == "info" else PALETTE[1] if c["severity"] == "warn" else PALETTE[5] for c in checks]
    axes[0].barh(ids, values, color=colors)
    axes[0].set_xlim(0, 1)
    axes[0].set_title(f"Dataset-quality rubric — {dataset_rubric.get('dataset', 'aggregate')} ({dataset_rubric['score']})")
    axes[0].invert_yaxis()
    models = model_rubric.get("models", [])
    if models:
        best = max(models, key=lambda r: r["score"])
        mids = [c["id"] for c in best["checks"]]
        mvals = [c["score"] for c in best["checks"]]
        mcolors = [PALETTE[0] if c["severity"] == "info" else PALETTE[1] if c["severity"] == "warn" else PALETTE[5] for c in best["checks"]]
        axes[1].barh(mids, mvals, color=mcolors)
        axes[1].set_xlim(0, 1)
        axes[1].set_title(f"Model-correctness rubric — {best['dataset']}/{best['model']} ({best['score']})")
        axes[1].invert_yaxis()
    fig.suptitle("Rubric check scores (blue pass, orange warn, red fail)", fontsize=11, fontweight="bold")
    return _save(fig, out, "rubric_bars.png")


def rubric_validation(validation: dict[str, Any], out: Path) -> Path:
    rows = validation["rows"]
    names = [r["defect"] for r in rows]
    values = [r["overall_score"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows) + 1.8))
    colors = [PALETTE[2] if r["detected"] else PALETTE[1] if r["detected"] is False else PALETTE[7] for r in rows]
    axes[0].barh(names, values, color=colors)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Overall rubric score")
    axes[0].set_title("Planted-defect rubric scores")
    axes[0].invert_yaxis()
    noise = [r for r in rows if "label_noise" in r]
    if noise:
        x = np.arange(len(noise))
        w = 0.36
        axes[1].bar(x - w / 2, [r["label_noise"]["planted_rate"] for r in noise], w, label="planted", color=PALETTE[3])
        axes[1].bar(x + w / 2, [r["label_noise"]["estimated_rate"] for r in noise], w, label="estimated", color=PALETTE[0])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([r["defect"].replace("label_noise_", "") + "%" for r in noise])
        axes[1].set_ylabel("Noise rate")
        axes[1].legend()
        axes[1].set_title("Label-noise estimate vs planted rate")
    fig.suptitle("Rubric validation on planted defects (green = detected)", fontsize=11, fontweight="bold")
    return _save(fig, out, "rubric_validation.png")


def hpo_plots(hpo: dict[str, Any], out: Path) -> Path:
    studies = hpo.get("studies", [])
    fig, axes = plt.subplots(1, max(1, len(studies)), figsize=(3.8 * max(1, len(studies)), 3.2), squeeze=False)
    for ax, study in zip(axes[0], studies, strict=False):
        hist = [h for h in study["history"] if h["value"] is not None]
        ax.plot([h["trial"] for h in hist], [h["value"] for h in hist], marker="o", color=PALETTE[0], ms=3)
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV accuracy")
        ax.set_title(f"{study['model']} — best {study['best_cv_accuracy']:.3f}")
    fig.suptitle("HPO optimisation history", fontsize=11, fontweight="bold")
    p1 = _save(fig, out, "hpo_history.png")

    fig, axes = plt.subplots(1, max(1, len(studies)), figsize=(4.0 * max(1, len(studies)), 3.0), squeeze=False)
    for ax, study in zip(axes[0], studies, strict=False):
        imp = study.get("importance") or {}
        items = sorted(imp.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color=PALETTE[2])
        ax.set_xlabel("Importance")
        ax.set_title(study["model"])
    fig.suptitle("Hyperparameter importance", fontsize=11, fontweight="bold")
    _save(fig, out, "hpo_importance.png")
    return p1


def generate_all(data_dir: str | Path, out_dir: str | Path) -> list[Path]:
    _style()
    data = Path(data_dir)
    out = Path(out_dir)
    paths = []
    load = lambda name: json.loads((data / name).read_text(encoding="utf-8"))  # noqa: E731
    paths.append(leaderboard_ci(load("leaderboard.json"), out))
    paths.append(critical_difference(load("stats.json"), out))
    paths.append(pareto(load("pareto.json"), out))
    paths.append(learning_curves(load("generalization.json"), out))
    paths.append(robustness_curves(load("generalization.json"), out))
    paths.append(calibration_plots(load("error_analysis.json"), out))
    paths.append(confusion_matrices(load("error_analysis.json"), out))
    paths.append(rubric_bars(load("rubric_dataset.json"), load("rubric_model.json"), out))
    paths.append(rubric_validation(load("rubric_validation.json"), out))
    paths.append(hpo_plots(load("hpo.json"), out))
    return paths
