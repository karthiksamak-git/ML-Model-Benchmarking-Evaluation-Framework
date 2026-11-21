"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function HpoChart({ study }: { study: any }) {
  const history = (study.history ?? [])
    .filter((h: any) => h.value !== null)
    .map((h: any) => ({ trial: h.trial + 1, accuracy: h.value }));
  const importance = Object.entries(study.importance ?? {})
    .map(([name, value]) => ({ name, value: Number(value) }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Optimisation history</h4>
        <div className="mt-1 h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 8, right: 8, bottom: 8, left: -18 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="trial" tick={{ fontSize: 10 }} />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(3)} />
              <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
              <Line type="monotone" dataKey="accuracy" stroke="#0072B2" dot={{ r: 2 }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Parameter importance</h4>
        <div className="mt-1 h-56">
          {importance.length ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance} layout="vertical" margin={{ top: 8, right: 12, bottom: 8, left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={80} />
                <Tooltip formatter={(v: any) => Number(v).toFixed(3)} />
                <Bar dataKey="value" fill="#009E73" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="pt-16 text-center text-xs text-slate-400">Not enough completed trials for importance.</p>
          )}
        </div>
      </div>
    </div>
  );
}
