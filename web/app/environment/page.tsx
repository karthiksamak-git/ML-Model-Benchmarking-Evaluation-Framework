import { formatNum, loadData } from "@/lib/data";
import { Badge, Card, PageHeader, Provenance, Section, Stat } from "@/components/ui";

export default function EnvironmentPage() {
  const env = loadData("env.json");
  const repro = loadData("repro.json");
  const leaderboard = loadData("leaderboard.json");
  const info = env.environment ?? {};
  const versions = info.versions ?? {};

  return (
    <div>
      <PageHeader
        title="Environment, reproducibility & methodology"
        lead={
          <>
            Every run records its seed, library versions, CPU, RAM, OS, thread settings and git hash as MLflow
            tags. Model comparison uses repeated stratified CV with identical splits; the rubric thresholds are
            configurable and validated against planted defects. Training happens locally or in CI — this site is a
            static export.
          </>
        }
      />

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="CPU" value={info.processor ?? "—"} hint={`${info.cpu_count_logical ?? "?"} logical cores`} />
        <Stat label="RAM" value={`${info.ram_gb ?? "?"} GB`} />
        <Stat label="Accelerator" value={info.accelerator ?? "cpu"} hint={info.gpu_available ? "CUDA available" : "CPU-only run"} />
        <Stat label="Python" value={info.python ?? "—"} hint={info.platform} />
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="text-sm font-semibold text-slate-900">Library versions (pinned)</h3>
          <table className="data mt-2 w-full text-sm">
            <tbody>
              {Object.entries(versions).map(([pkg, version]) => (
                <tr key={pkg}>
                  <td className="font-mono text-xs text-slate-600">{pkg}</td>
                  <td className="tabular">{String(version)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-slate-900">Run fingerprint</h3>
          <dl className="mt-2 space-y-2 text-xs">
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">Git hash</dt>
              <dd className="font-mono">{info.git}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">Threads (torch)</dt>
              <dd className="tabular">{info.threads?.torch}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">OMP / MKL</dt>
              <dd className="tabular">
                {info.threads?.OMP_NUM_THREADS} / {info.threads?.MKL_NUM_THREADS}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">CV scheme</dt>
              <dd>
                {env.cv?.n_splits}×{env.cv?.n_repeats} repeated stratified (seed {env.cv?.seed})
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500">Dataset subsampling</dt>
              <dd>stratified, deterministic</dd>
            </div>
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="green">deterministic seeds</Badge>
            <Badge tone="green">identical folds for all models</Badge>
            <Badge tone="green">pitched pipeline inside folds</Badge>
          </div>
        </Card>
      </div>

      <Section title="Reproducibility check" subtitle="Re-run a logged fold from its recorded params, seed and split ID.">
        {repro?.pass ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900">Same seed</h3>
                <Badge tone={repro.pass.status === "pass" ? "green" : "rose"}>{repro.pass.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {repro.pass.model} on {repro.pass.dataset}, split {repro.pass.split_id}
              </p>
              <table className="data mt-2 w-full text-xs">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Logged</th>
                    <th>Re-run</th>
                    <th>|Δ|</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(repro.pass.metrics_compared ?? {}).map(([k, v]: [string, any]) => (
                    <tr key={k}>
                      <td className="font-mono text-[10px] text-slate-600">{k}</td>
                      <td className="tabular">{formatNum(v.logged, 6)}</td>
                      <td className="tabular">{formatNum(v.rerun, 6)}</td>
                      <td className="tabular">{v.abs_diff.toExponential(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-slate-500">
                Tolerance {repro.pass.tolerance.toExponential(1)}; max |Δ| {repro.pass.max_abs_diff.toExponential(1)}.
              </p>
            </Card>
            <Card>
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900">Deliberately changed seed</h3>
                <Badge tone={repro.fail_demo?.status === "fail" ? "amber" : "green"}>
                  {repro.fail_demo?.status ?? "—"}
                </Badge>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Seed offset +1 changes the fitted model, so the check must fail. Max |Δ| ={" "}
                {formatNum(repro.fail_demo?.max_abs_diff, 6)} against tolerance{" "}
                {formatNum(repro.fail_demo?.tolerance, 10)}. This demonstrates that the check can detect
                irreproducibility rather than rubber-stamping.
              </p>
              <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
                <p className="font-medium text-slate-800">Scope of each phase</p>
                <ul className="mt-1 list-disc space-y-1 pl-4">
                  <li>Benchmark: {leaderboard.rows?.length ?? 0} (dataset, model) parent runs.</li>
                  <li>HPO: TPE + median pruning, nested MLflow trials.</li>
                  <li>Rubric validation: planted defects with ground-truth manifests.</li>
                  <li>Errors: per-class, calibration, slices, clusters, learning curves, robustness.</li>
                </ul>
              </div>
            </Card>
          </div>
        ) : (
          <Card>No reproducibility export found.</Card>
        )}
      </Section>

      <Section title="Methodology notes" subtitle="What is measured, and what the numbers do not mean.">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <h3 className="text-sm font-semibold text-slate-900">Statistics</h3>
            <ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-relaxed text-slate-600">
              <li>Repeated stratified k-fold; every model sees the same fold indices (checksum logged).</li>
              <li>Nadeau–Bengio corrected resampled t-test accounts for overlapping training sets.</li>
              <li>Wilcoxon signed-rank as a nonparametric paired check; Holm correction within datasets.</li>
              <li>Bootstrap intervals resample folds, not samples — they reflect CV variability.</li>
              <li>Friedman + Nemenyi across datasets; McNemar on the shared first split.</li>
            </ul>
          </Card>
          <Card>
            <h3 className="text-sm font-semibold text-slate-900">Limitations</h3>
            <ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-relaxed text-slate-600">
              <li>Small sklearn datasets on a laptop CPU; absolute accuracy is not a claim about production data.</li>
              <li>Label-noise detection is a confident-learning heuristic, not ground truth; it is validated on
                planted noise with documented tolerance.</li>
              <li>Outlier flags include legitimate long tails; the rule requires two of three methods on the full
                pipeline and is configurable.</li>
              <li>CNN results use 8×8 digits, so they are not comparable to full-resolution benchmarks.</li>
              <li>The browser checker covers {8} of the dataset checks and renormalises weights.</li>
            </ul>
          </Card>
        </div>
      </Section>

      <Provenance provenance={env.provenance} />
    </div>
  );
}
