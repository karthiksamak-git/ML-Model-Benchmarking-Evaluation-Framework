"use client";

import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Row = {
  model: string;
  label: string;
  dataset: string;
  accuracy: number;
  accuracy_std: number;
  ci_low: number;
  ci_high: number;
  gap: number;
  latency_p50_ms: number;
  model_size_mb: number;
  param_count: number;
};

type Weights = { accuracy: number; generalization: number; performance: number };

const PRESETS: { name: string; weights: Weights }[] = [
  { name: "Balanced", weights: { accuracy: 50, generalization: 25, performance: 25 } },
  { name: "Accuracy", weights: { accuracy: 80, generalization: 15, performance: 5 } },
  { name: "Deployable", weights: { accuracy: 40, generalization: 20, performance: 40 } },
];

function normalize(w: Weights) {
  const total = w.accuracy + w.generalization + w.performance || 1;
  return {
    accuracy: w.accuracy / total,
    generalization: w.generalization / total,
    performance: w.performance / total,
  };
}

/** Same scoring as src/mlbe/stats/ranking.py. */
export function scoreRow(row: Row, w: Weights) {
  const n = normalize(w);
  const generalization = Math.min(1, Math.max(0, 1 - Math.abs(row.gap)));
  const performance = 1 / (1 + Math.max(row.latency_p50_ms, 1e-9) / 10);
  return (
    100 *
    (n.accuracy * row.accuracy + n.generalization * generalization + n.performance * performance)
  );
}

export default function Leaderboard({
  rows,
  overall,
  pareto,
}: {
  rows: Row[];
  overall: Row[];
  pareto: { datasets: { dataset: string; points: any[] }[] };
}) {
  const [weights, setWeights] = useState<Weights>(PRESETS[0].weights);
  const [dataset, setDataset] = useState("overall");
  const [sortKey, setSortKey] = useState("score");
  const [sortDir, setSortDir] = useState<-1 | 1>(-1);

  const datasets = useMemo(() => [...new Set(rows.map((r) => r.dataset))].sort(), [rows]);
  const base = dataset === "overall" ? overall : rows.filter((r) => r.dataset === dataset);
  const scored = useMemo(
    () =>
      base
        .map((r) => {
          const accuracy = r.accuracy;
          const generalization = Math.min(1, Math.max(0, 1 - Math.abs(r.gap)));
          const performance = 1 / (1 + Math.max(r.latency_p50_ms, 1e-9) / 10);
          return { ...r, score: scoreRow(r, weights), criteria: { accuracy, generalization, performance } };
        })
        .sort((a, b) => {
          const av = (a as any)[sortKey] ?? 0;
          const bv = (b as any)[sortKey] ?? 0;
          const cmp = typeof av === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
          return cmp * sortDir;
        }),
    [base, weights, sortKey, sortDir],
  );

  const paretoSet = useMemo(() => {
    const ds = dataset === "overall" ? datasets[0] : dataset;
    const points = pareto.datasets.find((d) => d.dataset === ds)?.points ?? [];
    return { dataset: ds, points };
  }, [dataset, datasets, pareto]);

  function sortBy(key: string) {
    if (key === sortKey) setSortDir((d) => (d === -1 ? 1 : -1));
    else {
      setSortKey(key);
      setSortDir(-1);
    }
  }

  const header = (key: string, label: string) => (
    <th className="cursor-pointer select-none hover:text-slate-900" onClick={() => sortBy(key)}>
      {label}
      {sortKey === key ? <span className="ml-1 text-slate-400">{sortDir === -1 ? "▼" : "▲"}</span> : null}
    </th>
  );

  return (
    <div className="space-y-6">
      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Re-weight the ranking</h2>
            <p className="text-xs text-slate-500">
              Criteria are normalised exactly like the Python ranking: accuracy, 1 − |train−test gap|, and
              1&nbsp;/&nbsp;(1 + latency/10&nbsp;ms). Move a slider to re-rank live.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => setWeights(p.weights)}
                className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {(
            [
              ["accuracy", "Accuracy"],
              ["generalization", "Generalization"],
              ["performance", "Performance"],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="block text-xs font-medium text-slate-600">
              <span className="flex justify-between">
                <span>{label}</span>
                <span className="tabular text-slate-900">{weights[key]}</span>
              </span>
              <input
                type="range"
                min={0}
                max={100}
                value={weights[key]}
                onChange={(e) => setWeights({ ...weights, [key]: Number(e.target.value) })}
                className="mt-2 w-full accent-slate-900"
              />
            </label>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="font-medium text-slate-600">Dataset:</span>
          <select
            className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            <option value="overall">all datasets (mean)</option>
            {datasets.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <span className="ml-2 text-slate-400">
            Weights normalised to {Object.values(normalize(weights)).map((v) => v.toFixed(2)).join(" / ")}
          </span>
        </div>
      </div>

      <div className="card overflow-x-auto p-0">
        <table className="data w-full text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="w-12">#</th>
              {header("label", "Model")}
              {header("accuracy", "Accuracy (95% CI)")}
              {header("gap", "Gen. gap")}
              {header("latency_p50_ms", "Latency p50")}
              {header("model_size_mb", "Size")}
              {header("score", "Weighted score")}
            </tr>
          </thead>
          <tbody>
            {scored.map((r, i) => (
              <tr key={r.model + r.dataset}>
                <td className="tabular text-slate-400">{i + 1}</td>
                <td className="font-medium text-slate-900">{r.label}</td>
                <td className="tabular">
                  {r.accuracy.toFixed(4)}{" "}
                  <span className="text-xs text-slate-400">
                    [{r.ci_low.toFixed(3)}, {r.ci_high.toFixed(3)}]
                  </span>
                </td>
                <td className="tabular">{r.gap.toFixed(3)}</td>
                <td className="tabular">{r.latency_p50_ms.toFixed(2)} ms</td>
                <td className="tabular">{r.model_size_mb < 0.01 ? "<0.01" : r.model_size_mb.toFixed(2)} MB</td>
                <td className="tabular font-semibold">{r.score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card p-4">
        <div className="mb-2 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold text-slate-900">Accuracy vs latency Pareto front</h3>
          <span className="text-xs text-slate-500">{paretoSet.dataset}</span>
        </div>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                type="number"
                dataKey="latency_p50_ms"
                name="latency p50 (ms)"
                scale="log"
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{ value: "Inference latency p50 (ms, log scale)", position: "insideBottom", offset: -10 }}
              />
              <YAxis
                type="number"
                dataKey="accuracy"
                name="accuracy"
                domain={["auto", "auto"]}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{ value: "Accuracy", angle: -90, position: "insideLeft" }}
              />
              <Tooltip
                formatter={(value: any, name: any) => [Number(value).toFixed(4), String(name)]}
                labelFormatter={() => ""}
                content={({ payload }) => {
                  const p = payload?.[0]?.payload;
                  if (!p) return null;
                  return (
                    <div className="rounded-lg border border-slate-200 bg-white p-2 text-xs shadow">
                      <div className="font-semibold text-slate-900">{p.label}</div>
                      <div>accuracy {p.accuracy.toFixed(4)}</div>
                      <div>latency {p.latency_p50_ms.toFixed(3)} ms</div>
                      <div>size {p.model_size_mb.toFixed(2)} MB</div>
                      <div className={p.pareto ? "text-emerald-600" : "text-slate-400"}>
                        {p.pareto ? "on Pareto front" : "dominated"}
                      </div>
                    </div>
                  );
                }}
              />
              <Scatter data={paretoSet.points} name="models">
                {paretoSet.points.map((p) => (
                  <Cell key={p.id} fill={p.pareto ? "#0072B2" : "#cbd5e1"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-1 text-xs text-slate-500">
          Blue points are non-dominated in (accuracy ↑, latency ↓, size ↓). Pareto membership is computed in
          Python and exported; the chart does not infer it.
        </p>
      </div>
    </div>
  );
}
