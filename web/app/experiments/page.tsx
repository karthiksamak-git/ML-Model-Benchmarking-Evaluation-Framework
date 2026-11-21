import { formatNum, loadData } from "@/lib/data";
import { Badge, Card, PageHeader, Provenance, Section } from "@/components/ui";
import HpoChart from "@/components/HpoChart";

export default function ExperimentsPage() {
  const hpo = loadData("hpo.json");
  const runs = loadData("runs.json");
  const studies = hpo.studies ?? [];
  const totalRuns = runs.runs?.length ?? 0;
  const datasets = [...new Set((runs.runs ?? []).map((r: any) => r.dataset))];

  return (
    <div>
      <PageHeader
        title="Experiments & hyperparameter optimisation"
        lead={
          <>
            Optuna (TPE + median pruning, seeded) searches each space with k-fold CV <em>inside</em> an 80%
            training split; the held-out 20% is evaluated once for the tuned-vs-default comparison, so the reported
            improvement is not selection-biased. Every trial is a nested MLflow run with its own params, fold
            scores and pruning state.
          </>
        }
      />

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">Benchmark fold runs</div>
          <div className="mt-1 text-2xl font-semibold tabular">{totalRuns}</div>
          <div className="text-xs text-slate-500">{datasets.length} datasets</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">HPO studies</div>
          <div className="mt-1 text-2xl font-semibold tabular">{studies.length}</div>
          <div className="text-xs text-slate-500">TPE + MedianPruner, seeded sampler</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">Protocol</div>
          <div className="mt-1 text-sm font-medium text-slate-800">CV-then-held-out</div>
          <div className="text-xs text-slate-500">{studies[0]?.protocol ?? "—"}</div>
        </Card>
      </div>

      <Section title="Studies" subtitle="Best parameters, unbiased test comparison, importance and trial history.">
        <div className="space-y-4">
          {studies.map((s: any) => (
            <Card key={s.study_run_id}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    {s.model} <span className="font-normal text-slate-400">on {s.dataset}</span>
                  </h3>
                  <p className="text-xs text-slate-500">
                    {s.n_completed}/{s.n_trials} trials completed · {formatNum(s.elapsed_s, 0)}s · seed{" "}
                    {s.history?.[0] ? "logged" : "—"}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2 text-xs">
                  <Badge tone="blue">best CV {formatNum(s.best_cv_accuracy, 4)}</Badge>
                  <Badge tone={s.test_improvement > 0 ? "green" : "slate"}>
                    tuned {formatNum(s.tuned_test_accuracy, 4)} vs default {formatNum(s.default_test_accuracy, 4)}
                  </Badge>
                </div>
              </div>

              <div className="mt-3 grid gap-4 lg:grid-cols-[1fr_1.4fr]">
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Best parameters</h4>
                  <table className="data mt-1 w-full text-xs">
                    <tbody>
                      {Object.entries(s.best_params ?? {}).map(([k, v]) => (
                        <tr key={k}>
                          <td className="font-mono text-slate-600">{k}</td>
                          <td className="tabular text-slate-900">
                            {typeof v === "number" ? Number(v).toPrecision(5) : String(v)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="mt-2 text-xs text-slate-500">
                    Tuned CV 95% CI [{formatNum(s.best_cv_ci?.[0], 4)}, {formatNum(s.best_cv_ci?.[1], 4)}]; paired
                    bootstrap Δ CI [{formatNum(s.paired_bootstrap_ci?.[0], 4)},{" "}
                    {formatNum(s.paired_bootstrap_ci?.[1], 4)}]; McNemar p ={" "}
                    {formatNum(s.mcnemar?.p_value, 4)}.
                  </div>
                </div>
                <HpoChart study={s} />
              </div>

              <details className="mt-3">
                <summary className="cursor-pointer text-xs font-medium text-slate-500">
                  Trial table ({s.history?.length ?? 0})
                </summary>
                <div className="mt-2 max-h-64 overflow-auto">
                  <table className="data w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>
                        <th>Trial</th>
                        <th>State</th>
                        <th>CV accuracy</th>
                        <th>Params</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(s.history ?? []).map((h: any) => (
                        <tr key={h.trial}>
                          <td className="tabular">{h.trial}</td>
                          <td>{h.state}</td>
                          <td className="tabular">{h.value === null ? "—" : formatNum(h.value, 4)}</td>
                          <td className="max-w-[30rem] whitespace-normal font-mono text-[10px] text-slate-500">
                            {JSON.stringify(h.params)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </Card>
          ))}
        </div>
      </Section>

      <Section title="Run log" subtitle="Compact per-fold records (all runs are in MLflow; this is a browser-friendly sample).">
        <Card className="overflow-x-auto">
          <table className="data w-full text-xs">
            <thead className="bg-slate-50">
              <tr>
                <th>Dataset</th>
                <th>Model</th>
                <th>Repeat</th>
                <th>Fold</th>
                <th>Seed</th>
                <th>Accuracy</th>
                <th>Log loss</th>
                <th>ECE</th>
                <th>Run</th>
              </tr>
            </thead>
            <tbody>
              {(runs.runs ?? []).slice(0, 400).map((r: any, i: number) => (
                <tr key={i}>
                  <td>{r.dataset}</td>
                  <td>{r.model}</td>
                  <td className="tabular">{r.repeat}</td>
                  <td className="tabular">{r.fold}</td>
                  <td className="tabular">{r.seed}</td>
                  <td className="tabular">{formatNum(r.metrics?.accuracy, 4)}</td>
                  <td className="tabular">{formatNum(r.metrics?.log_loss, 4)}</td>
                  <td className="tabular">{formatNum(r.metrics?.ece, 4)}</td>
                  <td className="font-mono text-[10px] text-slate-400">{String(r.run_id).slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </Section>

      <Provenance provenance={hpo.provenance} />
    </div>
  );
}
