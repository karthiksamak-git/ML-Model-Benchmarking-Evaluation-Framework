"""Command line interface: `mlbe benchmark|hpo|analyze|rubric|repro|export|figures|all`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

from .analysis.errors import analyze_predictions
from .analysis.report import findings_markdown
from .benchmark.generalization import learning_curve, robustness_analysis
from .benchmark.runner import run_benchmark
from .config import PROJECT_ROOT, benchmark_config, dump_json, hpo_config, rubric_config
from .data.registry import load_dataset
from .data.splits import make_splits
from .env_info import env_info
from .export.web_artifacts import export_all
from .hpo.search import run_hpo
from .models.registry import build_model
from .rubrics.dataset_quality import evaluate_dataset_quality
from .rubrics.model_correctness import evaluate_model_correctness
from .rubrics.report import validate_dataset_rubric, write_rubric_outputs
from .seed import seed_everything
from .tracking import mlflow_utils as mu
from .tracking.queries import benchmark_parents, experiment_id
from .tracking.repro import reproduce_run, sample_run_id
from .viz.figures import generate_all as generate_figures


def _quick_cfg(args: argparse.Namespace):
    overrides = list(args.override or [])
    if args.quick:
        cfg_path = PROJECT_ROOT / "configs" / "benchmark.yaml"
        from .config import Config

        base = Config.load(cfg_path)
        q = base.quick
        overrides += [
            f"cv.n_splits={q.n_splits}",
            f"cv.n_repeats={q.n_repeats}",
            f"performance.latency_repeats={min(10, base.performance.latency_repeats)}",
        ]
    return benchmark_config(overrides=overrides)


def _artifacts() -> Path:
    path = PROJECT_ROOT / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _best_models(cfg: Any) -> dict[str, dict[str, Any]]:
    client = MlflowClient()
    exp_id = experiment_id(client, str(cfg.paths.experiment))
    parents = benchmark_parents(client, exp_id)
    best: dict[str, dict[str, Any]] = {}
    for p in parents:
        acc = float(p["metrics"].get("accuracy_mean", -1))
        if p["dataset"] not in best or acc > best[p["dataset"]]["accuracy"]:
            best[p["dataset"]] = {
                "model": p["model"],
                "run_id": p["run_id"],
                "accuracy": acc,
                "gap": float(p["metrics"].get("gap_accuracy_mean", 0.0)),
            }
    return best


def cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    summary = run_benchmark(
        cfg,
        quick=args.quick,
        max_datasets=args.max_datasets,
        only_models=args.models,
    )
    print(f"benchmark: {len(summary.parents)} parent runs in {summary.elapsed_s:.1f}s")
    for p in sorted(summary.parents, key=lambda r: (r["dataset"], -r["metrics"].get("accuracy_mean", 0)))[:30]:
        print(f"  {p['dataset']:14s} {p['model']:7s} acc={p['metrics'].get('accuracy_mean', float('nan')):.4f}")
    return summary.to_dict()


def cmd_hpo(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    hcfg = hpo_config()
    summary = run_hpo(cfg, hcfg, quick=args.quick, models=args.models, dataset=args.dataset)
    out = _artifacts() / "hpo" / "hpo.json"
    dump_json(summary, out)
    for s in summary["studies"]:
        print(
            f"hpo {s['model']:5s}: best CV {s['best_cv_accuracy']:.4f}, "
            f"tuned test {s['tuned_test_accuracy']:.4f} vs default {s['default_test_accuracy']:.4f} "
            f"({s['n_completed']}/{s['n_trials']} trials, {s['elapsed_s']:.0f}s)"
        )
    return summary


def cmd_analyze(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    mu.init_tracking(cfg)
    best = _best_models(cfg)
    results = []
    curves = []
    robustness = []
    seed_everything(int(cfg.seed))
    for name, info in sorted(best.items()):
        ds = load_dataset(name, cfg.datasets[name], seed=int(cfg.seed))
        splits = make_splits(
            ds.y,
            n_splits=int(cfg.cv.n_splits),
            n_repeats=int(cfg.cv.n_repeats),
            seed=int(cfg.seed),
            dataset=name,
        )
        split = splits[0]
        fold_seed = int(cfg.seed) + 1000 * split.repeat + split.fold
        seed_everything(fold_seed)
        model = build_model(
            info["model"],
            seed=fold_seed,
            n_classes=ds.n_classes,
            input_dim=ds.n_features,
            image_shape=ds.image_shape if ds.is_image else None,
        )
        model.fit(ds.X[split.train_idx], ds.y[split.train_idx])
        proba = model.predict_proba(ds.X[split.test_idx])
        pred = model.predict(ds.X[split.test_idx])
        result = analyze_predictions(
            name,
            info["model"],
            ds.X[split.test_idx],
            ds.y[split.test_idx],
            pred,
            proba,
            np.unique(ds.y),
            ds.feature_names,
            seed=int(cfg.seed),
        )
        result["source_run_id"] = info["run_id"]
        results.append(result)
        fractions = (0.5, 1.0) if args.quick else (0.2, 0.4, 0.6, 0.8, 1.0)
        curves.extend(
            learning_curve(
                ds.X,
                ds.y,
                info["model"],
                seed=fold_seed,
                fractions=fractions,
                cv_folds=3,
                dataset_name=name,
            )
        )
        levels = (0.0, 0.5) if args.quick else (0.0, 0.1, 0.25, 0.5, 1.0)
        robustness.extend(
            robustness_analysis(
                ds.X[split.train_idx],
                ds.y[split.train_idx],
                ds.X[split.test_idx],
                ds.y[split.test_idx],
                info["model"],
                seed=fold_seed,
                noise_levels=levels,
                shift_levels=(0.0, 0.5) if args.quick else (0.0, 0.5, 1.0, 2.0),
                dataset_name=name,
            )
        )
        print(f"analyze {name}/{info['model']}: accuracy {result['accuracy']:.4f}, {len(result['top_confident_errors'])} confident errors")

    env = env_info()
    markdown = findings_markdown(results, env)
    payload = {
        "results": results,
        "generalization": {"learning_curves": curves, "robustness": robustness},
        "findings_markdown": markdown,
        "source_run_ids": [info["run_id"] for info in best.values()],
        "env": env,
    }
    out_dir = _artifacts() / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(payload, out_dir / "analysis.json")
    (out_dir / "findings.md").write_text(markdown, encoding="utf-8")
    with mu.run("analysis", tags={"kind": "analysis_artifact"}) as run:
        mu.log_env_tags(env)
        mu.log_json_artifact(payload, "analysis.json")
        import mlflow

        mlflow.log_artifact(str(out_dir / "findings.md"))
        run_id = run.info.run_id
    print(f"analysis: {len(results)} datasets, {len(curves)} learning-curve points, run {run_id}")
    return payload


def cmd_rubric(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    rcfg = rubric_config()
    mu.init_tracking(cfg)
    best = _best_models(cfg)
    dataset_reports = []
    model_reports = []
    seed_everything(int(cfg.seed))
    for name, info in sorted(best.items()):
        ds = load_dataset(name, cfg.datasets[name], seed=int(cfg.seed))
        splits = make_splits(
            ds.y,
            n_splits=int(cfg.cv.n_splits),
            n_repeats=int(cfg.cv.n_repeats),
            seed=int(cfg.seed),
            dataset=name,
        )
        split = splits[0]
        report = evaluate_dataset_quality(
            ds.X[split.train_idx],
            ds.y[split.train_idx],
            ds.X[split.test_idx],
            ds.y[split.test_idx],
            rcfg,
            feature_names=ds.feature_names,
            seed=int(cfg.seed),
        )
        report["dataset"] = name
        dataset_reports.append(report)
        model_report = evaluate_model_correctness(
            name,
            ds.X,
            ds.y,
            splits,
            info["model"],
            cfg,
            rcfg,
            seed=int(cfg.seed),
        )
        model_reports.append(model_report)
        print(
            f"rubric {name}: dataset {report['score']} ({report['grade']}), "
            f"model {model_report['score']} ({model_report['grade']}) [{info['model']}]"
        )

    print("validating rubric on planted defects...")
    validation = validate_dataset_rubric(rcfg, seed=int(cfg.seed))
    print(
        f"validation: {validation['summary']['n_detected']}/{validation['summary']['n_defect_datasets']} "
        f"defects detected, clean score {validation['summary']['clean_score']}"
    )
    out_dir = _artifacts() / "rubric"
    paths = write_rubric_outputs(validation, dataset_reports, model_reports, rcfg, out_dir)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    payload["source_run_ids"] = [info["run_id"] for info in best.values()]
    dump_json(payload, paths["json"])
    with mu.run("rubric", tags={"kind": "rubric_artifact"}) as run:
        mu.log_env_tags(env_info())
        mu.log_json_artifact(payload, "rubric_report.json")
        import mlflow

        mlflow.log_artifact(str(paths["markdown"]))
        mlflow.log_artifact(str(paths["validation_md"]))
        run_id = run.info.run_id
    print(f"rubric outputs: {paths['json']} (run {run_id})")
    return payload


def cmd_repro(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    run_id = args.run_id or sample_run_id(cfg)
    result = reproduce_run(run_id, cfg)
    payload = {"pass": result, "fail_demo": None, "checked_at_unix": time.time()}
    if args.demo_failure:
        payload["fail_demo"] = reproduce_run(run_id, cfg, seed_offset=1, dry_run=True)
    out_dir = _artifacts() / "repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(payload, out_dir / "repro.json")
    print(
        f"repro {run_id}: {result['status']} (max |Δ| = {result['max_abs_diff']:.3g}, tol {result['tolerance']:.1g})"
    )
    if payload["fail_demo"]:
        d = payload["fail_demo"]
        print(f"repro failure demo (seed+1): {d['status']} (max |Δ| = {d['max_abs_diff']:.3g})")
    return payload


def cmd_export(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    info = export_all(cfg)
    print(f"exported {len(info['files'])} files, {info['total_bytes'] / 1024:.1f} KiB total -> {info['web_dir']}")
    for name, meta in info["files"].items():
        print(f"  {name:24s} {meta['bytes'] / 1024:8.1f} KiB")
    return info


def cmd_figures(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _quick_cfg(args)
    web = PROJECT_ROOT / cfg.paths.web_data
    paths = generate_figures(web, PROJECT_ROOT / cfg.paths.figures)
    total = sum(p.stat().st_size for p in paths)
    print(f"figures: {len(paths)} PNGs, {total / 1024:.0f} KiB -> {PROJECT_ROOT / cfg.paths.figures}")
    return {"figures": [str(p) for p in paths], "total_bytes": total}


def cmd_all(args: argparse.Namespace) -> None:
    started = time.time()
    cmd_benchmark(args)
    cmd_hpo(args)
    cmd_analyze(args)
    cmd_rubric(args)
    repro_args = argparse.Namespace(**vars(args))
    repro_args.demo_failure = True
    repro_args.run_id = getattr(args, "run_id", None)
    repro_args.sample = True
    cmd_repro(repro_args)
    cmd_export(args)
    cmd_figures(args)
    print(f"all: pipeline finished in {time.time() - started:.1f}s")
    print("Dashboard data is static JSON in web/public/data; no Python runs on Vercel.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlbe", description="ML benchmarking & evaluation framework")
    parser.add_argument("--override", action="append", help="config override, e.g. cv.n_splits=3")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--quick", action="store_true", help="fast mode for CI/tests")
        return p

    b = common(sub.add_parser("benchmark", help="run the benchmark suite"))
    b.add_argument("--max-datasets", type=int, default=None)
    b.add_argument("--models", nargs="*", default=None)
    b.set_defaults(func=cmd_benchmark)

    h = common(sub.add_parser("hpo", help="run Optuna studies"))
    h.add_argument("--models", nargs="*", default=None)
    h.add_argument("--dataset", default="breast_cancer")
    h.set_defaults(func=cmd_hpo)

    a = common(sub.add_parser("analyze", help="error analysis + generalization"))
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("rubric", help="evaluate dataset/model rubrics and validate on planted defects")
    r.add_argument("--quick", action="store_true")
    r.set_defaults(func=cmd_rubric)

    rp = common(sub.add_parser("repro", help="reproduce a logged run"))
    rp.add_argument("--run-id", default=None)
    rp.add_argument("--sample", action="store_true", help="pick the best logged fold run")
    rp.add_argument("--demo-failure", action="store_true", help="also show a seed-changed failure")
    rp.set_defaults(func=cmd_repro)

    e = sub.add_parser("export", help="export MLflow + artifacts to web JSON")
    e.add_argument("--quick", action="store_true")
    e.set_defaults(func=cmd_export)

    f = sub.add_parser("figures", help="generate figures from exported JSON")
    f.add_argument("--quick", action="store_true")
    f.set_defaults(func=cmd_figures)

    al = common(sub.add_parser("all", help="run the full pipeline"))
    al.add_argument("--max-datasets", type=int, default=None)
    al.add_argument("--models", nargs="*", default=None)
    al.add_argument("--dataset", default="breast_cancer")
    al.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "quick"):
        args.quick = False
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
