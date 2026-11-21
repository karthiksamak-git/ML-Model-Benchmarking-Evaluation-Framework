"""Auto-generated markdown findings from real error-analysis numbers.

This module only formats values that were computed from actual predictions; it
never inserts a claim that is not backed by the input dictionaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def findings_markdown(results: list[dict[str, Any]], env: dict[str, Any] | None = None) -> str:
    """Render per-dataset findings strictly from computed numbers."""
    lines = ["# Error analysis findings", ""]
    if env:
        lines += [
            f"Environment: {env.get('machine')} / {env.get('processor')} / "
            f"python {env.get('python')}, {env.get('ram_gb')} GB RAM, "
            f"git `{env.get('git')}`.",
            "",
        ]
    for r in results:
        cm = np.asarray(r["confusion_matrix"])
        n = max(1, int(cm.sum()))
        off = int(cm.sum() - np.trace(cm))
        lines.append(f"## {r['dataset']} — {r['model']}")
        lines.append("")
        lines.append(
            f"- Test accuracy **{r['accuracy']:.3f}** on {r['n_test']} samples; "
            f"{off} of {n} predictions wrong ({off / n:.1%})."
        )
        if r["most_confused_pairs"]:
            p = r["most_confused_pairs"][0]
            lines.append(
                f"- Most confused pair: true **{p['true']}** predicted as **{p['predicted']}** "
                f"({p['count']} cases)."
            )
        weakest = min(r["per_class"], key=lambda c: c["recall"])
        lines.append(
            f"- Weakest class: **{weakest['class']}** with recall {weakest['recall']:.3f} "
            f"over {weakest['support']} samples."
        )
        cal = r["calibration"]
        lines.append(f"- Calibration: ECE {cal['ece']:.4f} across {len(cal['bins'])} reliability bins.")
        if r["top_confident_errors"]:
            e = r["top_confident_errors"][0]
            lines.append(
                f"- Most confident error: sample #{e['index']}, true {e['true']}, "
                f"predicted {e['predicted']} at confidence {e['confidence']:.3f}."
            )
        if r["error_clusters"]["clusters"]:
            biggest = max(r["error_clusters"]["clusters"], key=lambda c: c["n"])
            feats = ", ".join(
                f"{f['feature']} (z={f['mean_z']:+.2f})" for f in biggest["defining_features"]
            )
            lines.append(
                f"- Error clusters: {len(r['error_clusters']['clusters'])}; largest holds "
                f"{biggest['n']} errors ({biggest['share']:.0%}), driven by {feats}."
            )
        if r["slices"]:
            worst_slice = min(r["slices"], key=lambda s: s["accuracy"])
            lines.append(
                f"- Weakest slice: **{worst_slice['feature']}** quartile "
                f"{worst_slice['quartile']} accuracy {worst_slice['accuracy']:.3f} "
                f"(n={worst_slice['n']})."
            )
        lines.append("")
    return "\n".join(lines)


def write_findings(
    results: list[dict[str, Any]], path: str, env: dict[str, Any] | None = None
) -> str:
    markdown = findings_markdown(results, env)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    return markdown
