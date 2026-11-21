import { formatNum, loadData } from "@/lib/data";
import { Badge, Card, PageHeader, Provenance, Section } from "@/components/ui";

function pCell(p: number) {
  if (p < 0.001) return "bg-emerald-100 text-emerald-900";
  if (p < 0.01) return "bg-emerald-50 text-emerald-800";
  if (p < 0.05) return "bg-amber-50 text-amber-800";
  return "bg-slate-50 text-slate-500";
}

export default function StatisticsPage() {
  const stats = loadData("stats.json");
  const datasets = Object.keys(stats.pairwise ?? {});
  const friedman = stats.friedman ?? {};
  const models: string[] = friedman.models ?? [];

  return (
    <div>
      <PageHeader
        title="Statistical comparison"
        lead={
          <>
            Comparisons use the corrected resampled t-test (Nadeau–Bengio variance correction for overlapping
            training sets), Wilcoxon signed-rank on paired folds and paired bootstrap intervals. Within each
            dataset, t-test p-values are Holm-corrected. Across datasets, a Friedman test with Nemenyi post-hoc
            critical differences tests whether rankings are consistent. McNemar's exact test uses paired
            predictions on the same held-out split.
          </>
        }
      />

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">Friedman χ²</div>
          <div className="mt-1 text-2xl font-semibold tabular">{formatNum(friedman.statistic, 2)}</div>
          <div className="text-xs text-slate-500">p = {formatNum(friedman.p_value, 4)}</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">Nemenyi CD (α=0.05)</div>
          <div className="mt-1 text-2xl font-semibold tabular">{formatNum(friedman.critical_difference, 2)}</div>
          <div className="text-xs text-slate-500">avg ranks over {friedman.n_datasets ?? 0} datasets</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">Datasets compared</div>
          <div className="mt-1 text-2xl font-semibold tabular">{datasets.length}</div>
          <div className="text-xs text-slate-500">{models.length} models in the complete matrix</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-slate-500">McNemar pairs</div>
          <div className="mt-1 text-2xl font-semibold tabular">{stats.mcnemar?.length ?? 0}</div>
          <div className="text-xs text-slate-500">on the shared first split</div>
        </Card>
      </div>

      <Section
        title="Average ranks"
        subtitle="Lower is better. Models are ranked within each dataset, then averaged."
      >
        <Card>
          <div className="flex flex-wrap gap-2">
            {models.map((m: string, i: number) => (
              <div key={m} className="rounded-lg border border-slate-200 px-3 py-2">
                <div className="text-xs text-slate-500">{m}</div>
                <div className="text-lg font-semibold tabular">{formatNum(friedman.avg_ranks?.[i], 2)}</div>
              </div>
            ))}
          </div>
          {friedman.p_value !== undefined ? (
            <p className="mt-3 text-xs text-slate-500">
              {friedman.p_value < 0.05
                ? "The Friedman test rejects equal performance across models; Nemenyi pairs below show which differences exceed the critical distance."
                : "The Friedman test does not reject equal performance across models at α=0.05; treat rank differences with caution."}
            </p>
          ) : null}
        </Card>
      </Section>

      <Section
        title="Pairwise significance matrices"
        subtitle="Cell = Holm-corrected corrected-t p-value for the row model vs the column model. Green < 0.05."
      >
        <div className="grid gap-5 xl:grid-cols-2">
          {datasets.map((ds) => {
            const pairs = stats.pairwise[ds] as any[];
            const names = [...new Set(pairs.flatMap((p) => [p.model_a, p.model_b]))].sort();
            const lookup = new Map(pairs.map((p) => [`${p.model_a}|${p.model_b}`, p]));
            return (
              <Card key={ds} className="overflow-x-auto">
                <h3 className="text-sm font-semibold text-slate-900">{ds}</h3>
                <table className="mt-2 border-collapse text-xs">
                  <thead>
                    <tr>
                      <th />
                      {names.map((m) => (
                        <th key={m} className="px-1.5 py-1 font-medium text-slate-500">
                          {m}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {names.map((a) => (
                      <tr key={a}>
                        <th className="pr-2 text-right font-medium text-slate-600">{a}</th>
                        {names.map((b) => {
                          if (a === b)
                            return (
                              <td key={b} className="px-1.5 py-1 text-center text-slate-300">
                                ·
                              </td>
                            );
                          const pair = lookup.get(`${a}|${b}`) ?? lookup.get(`${b}|${a}`);
                          if (!pair) return <td key={b} />;
                          return (
                            <td key={b} className={`px-1.5 py-1 text-center tabular ${pCell(pair.t_p_value_holm)}`}>
                              {pair.t_p_value_holm < 0.001 ? "<.001" : pair.t_p_value_holm.toFixed(3)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-[11px] text-slate-500">
                  {pairs.length} pairs · method: {stats.tests?.multiple_comparison}
                </p>
              </Card>
            );
          })}
        </div>
      </Section>

      <Section title="Nemenyi post-hoc" subtitle="Pairs whose average-rank difference exceeds the critical distance.">
        <Card className="overflow-x-auto">
          <table className="data w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th>Model A</th>
                <th>Model B</th>
                <th>Rank difference</th>
                <th>Significant</th>
              </tr>
            </thead>
            <tbody>
              {(friedman.nemenyi_pairs ?? []).map((p: any, i: number) => (
                <tr key={i}>
                  <td>{p.model_a}</td>
                  <td>{p.model_b}</td>
                  <td className="tabular">{formatNum(p.rank_diff, 2)}</td>
                  <td>{p.significant ? <Badge tone="green">yes</Badge> : <Badge>no</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </Section>

      <Section title="McNemar on paired predictions" subtitle="Exact binomial test on discordant predictions from the shared first split.">
        <Card className="overflow-x-auto">
          <table className="data w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th>Dataset</th>
                <th>Model A</th>
                <th>Model B</th>
                <th>A only correct</th>
                <th>B only correct</th>
                <th>p-value</th>
              </tr>
            </thead>
            <tbody>
              {(stats.mcnemar ?? []).map((m: any, i: number) => (
                <tr key={i}>
                  <td>{m.dataset}</td>
                  <td>{m.model_a}</td>
                  <td>{m.model_b}</td>
                  <td className="tabular">{m.b_only_a_correct}</td>
                  <td className="tabular">{m.c_only_b_correct}</td>
                  <td className={`tabular ${m.p_value < 0.05 ? "font-semibold text-emerald-700" : ""}`}>
                    {m.p_value < 0.001 ? "<.001" : m.p_value.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </Section>

      <Provenance provenance={stats.provenance} />
    </div>
  );
}
