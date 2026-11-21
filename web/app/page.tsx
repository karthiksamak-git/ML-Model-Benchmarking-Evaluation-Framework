import Link from "next/link";
import { loadData, formatNum } from "@/lib/data";
import { Badge, Card, PageHeader, Provenance, ScoreBadge, Section, Stat } from "@/components/ui";

export default function OverviewPage() {
  const leaderboard = loadData("leaderboard.json");
  const rubric = loadData("rubric_dataset.json");
  const validation = loadData("rubric_validation.json");
  const env = loadData("env.json");
  const repro = loadData("repro.json");
  const stats = loadData("stats.json");

  const best = leaderboard.overall?.[0];
  const top = (leaderboard.overall ?? []).slice(0, 5);
  const datasets = Object.entries(env.datasets ?? {}) as [string, any][];
  const reproPass = repro?.pass?.status;
  const detection = validation?.summary;

  return (
    <div>
      <PageHeader
        title="ML model benchmarking & evaluation framework"
        lead={
          <>
            Models are compared with repeated stratified cross-validation on identical splits, with corrected
            resampled t-tests, Wilcoxon and McNemar tests plus bootstrap intervals — never a single-split
            leaderboard. Every run is tracked in MLflow, and two hand-written rubrics score dataset quality and
            model correctness against configurable thresholds. All numbers below come from real runs exported to
            static JSON; nothing is computed on Vercel.
          </>
        }
      />

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat
          label="Top-ranked model"
          value={best ? best.label : "—"}
          hint={best ? `${formatNum(best.accuracy, 4)} mean accuracy · ${best.n_datasets} datasets` : undefined}
          tone="good"
        />
        <Stat
          label="Rubric score"
          value={rubric?.summary?.overall_score?.toFixed(1) ?? "—"}
          hint={`grade ${rubric?.summary?.overall_grade ?? "—"} · dataset ${rubric?.summary?.dataset_quality_score?.toFixed(
            1,
          )} / model ${rubric?.summary?.model_correctness_score?.toFixed(1)}`}
        />
        <Stat
          label="Planted defects detected"
          value={detection ? `${detection.n_detected}/${detection.n_defect_datasets}` : "—"}
          hint={`clean dataset scores ${detection?.clean_score ?? "—"} · noisy-label P/R ${formatNum(
            detection?.label_noise_mean_precision,
            2,
          )}/${formatNum(detection?.label_noise_mean_recall, 2)}`}
          tone={detection?.detection_rate === 1 ? "good" : "warn"}
        />
        <Stat
          label="Reproducibility"
          value={reproPass ? reproPass.toUpperCase() : "—"}
          hint={
            repro?.pass
              ? `same seed max |Δ| = ${repro.pass.max_abs_diff.toExponential(1)} (tol ${repro.pass.tolerance.toExponential(1)}); seed+1 demo ${repro.fail_demo?.status}`
              : undefined
          }
          tone={reproPass === "pass" ? "good" : "bad"}
        />
      </div>

      <Section
        title="Headline leaderboard"
        subtitle="Weighted multi-criteria ranking (accuracy 50%, generalization 25%, performance 25%) across all datasets."
        action={
          <Link href="/leaderboard" className="text-sm font-medium text-blue-700 hover:underline">
            Explore and re-weight →
          </Link>
        }
      >
        <Card className="overflow-x-auto p-0">
          <table className="data w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th>#</th>
                <th>Model</th>
                <th>Mean accuracy (95% CI)</th>
                <th>Generalization gap</th>
                <th>Latency p50</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {top.map((row: any, i: number) => (
                <tr key={row.model}>
                  <td className="tabular text-slate-400">{i + 1}</td>
                  <td className="font-medium text-slate-900">{row.label}</td>
                  <td className="tabular">
                    {formatNum(row.accuracy, 4)}{" "}
                    <span className="text-xs text-slate-400">
                      [{formatNum(row.ci_low, 3)}, {formatNum(row.ci_high, 3)}]
                    </span>
                  </td>
                  <td className="tabular">{formatNum(row.gap, 3)}</td>
                  <td className="tabular">{formatNum(row.latency_p50_ms, 2)} ms</td>
                  <td className="tabular font-semibold">{formatNum(row.score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </Section>

      <div className="mt-8 grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="text-sm font-semibold text-slate-900">Datasets</h3>
          <p className="mt-1 text-xs text-slate-500">
            {env.cv?.n_splits}×{env.cv?.n_repeats} repeated stratified CV, seed {env.cv?.seed}. CNN only runs on
            the image form.
          </p>
          <table className="data mt-3 w-full text-sm">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Samples</th>
                <th>Features</th>
                <th>Classes</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map(([name, meta]) => (
                <tr key={name}>
                  <td className="font-medium text-slate-900">{name}</td>
                  <td className="tabular">{meta.n_samples}</td>
                  <td className="tabular">{meta.n_features}</td>
                  <td className="tabular">{meta.n_classes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-900">What was tested</h3>
          <div className="mt-2 space-y-2 text-xs leading-relaxed text-slate-600">
            <p>
              <strong>{stats.pairwise ? Object.keys(stats.pairwise).length : 0} datasets</strong> × up to 7 models,
              with per-fold MLflow runs, out-of-fold predictions for paired tests, train-vs-test gaps, latency,
              throughput, memory and model size.
            </p>
            <p>
              Omnibus test: Friedman with Nemenyi critical difference (α=0.05).{" "}
              {stats.friedman?.p_value !== undefined
                ? `Friedman χ²=${formatNum(stats.friedman.statistic, 2)}, p=${formatNum(stats.friedman.p_value, 4)}.`
                : null}
            </p>
            <p>
              Rubric validation is run against planted defects (label noise 5/10/20%, duplicates, conflicting
              labels, leaky feature, 15% missing values, outliers, covariate shift, imbalance, train/test
              overlap). Each detection is traceable to the exported check evidence.
            </p>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="blue">CPU-first</Badge>
            <Badge tone="blue">SQLite MLflow</Badge>
            <Badge tone="blue">Repeated stratified CV</Badge>
            <Badge tone="blue">Deterministic seeds</Badge>
            <Badge tone="blue">No fabricated numbers</Badge>
          </div>
          <div className="mt-3">
            <ScoreBadge score={rubric?.summary?.overall_score ?? 0} grade={rubric?.summary?.overall_grade} />
          </div>
        </Card>
      </div>

      <Provenance provenance={leaderboard.provenance} />
    </div>
  );
}
