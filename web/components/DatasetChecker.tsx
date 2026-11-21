"use client";

import { useMemo, useState } from "react";
import { evaluateDataset, parseCsv, type DatasetReport, type RubricConfig } from "@/lib/rubric";

const SAMPLE = `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,,1.4,0.2,0
5.4,3.9,1.7,0.4,1
4.6,3.4,1.4,0.3,1
5.0,3.4,1.5,0.2,1
4.4,2.9,1.4,0.2,1
4.9,3.1,1.5,0.2,1
7.0,3.2,4.7,1.4,2
6.4,3.2,4.5,1.5,2
6.9,3.1,4.9,1.5,2
6.9,3.1,4.9,1.5,2
6.9,3.1,4.9,1.5,2
7.0,3.2,4.7,1.4,2`;

function tone(severity: string) {
  return severity === "info"
    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : severity === "warn"
      ? "bg-amber-50 text-amber-800 border-amber-200"
      : "bg-rose-50 text-rose-700 border-rose-200";
}

function evidenceText(evidence: Record<string, unknown>): string {
  return Object.entries(evidence)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .slice(0, 6)
    .map(([k, v]) => `${k}=${typeof v === "number" ? Number(v).toPrecision(4) : String(v)}`)
    .join(" · ");
}

export default function DatasetChecker({ config }: { config: RubricConfig }) {
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const parsed = useMemo(() => (text.trim() ? parseCsv(text) : null), [text]);
  const report: DatasetReport | null = useMemo(() => {
    if (!parsed || parsed.dataset.rows.length === 0) return null;
    try {
      return evaluateDataset(parsed.dataset, config);
    } catch (e) {
      return null;
    }
  }, [parsed, config]);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setFileName(file.name);
    setError(null);
    try {
      setText(await file.text());
    } catch {
      setError("Could not read that file as text.");
    }
  }

  return (
    <div className="card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Check your own dataset (in the browser)</h3>
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-500">
            Paste numeric CSV or load a file. The last column is treated as the label. Checks run locally with the
            same thresholds and scoring formulas as the Python rubric; model-dependent checks (label noise,
            leakage, covariate shift) and the three-method outlier vote need the full pipeline.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
            onClick={() => {
              setText(SAMPLE);
              setFileName("example.csv");
            }}
          >
            Load example
          </button>
          <label className="cursor-pointer rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-slate-700">
            Upload CSV
            <input
              type="file"
              accept=".csv,text/csv,text/plain"
              className="hidden"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </label>
        </div>
      </div>

      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setFileName(null);
        }}
        rows={8}
        spellCheck={false}
        placeholder="paste CSV here — header row plus numeric columns, last column = label"
        className="mt-3 w-full rounded-lg border border-slate-300 bg-slate-50 p-3 font-mono text-xs text-slate-800 outline-none focus:border-slate-500"
      />

      {parsed?.errors.length ? (
        <p className="mt-2 text-xs text-amber-700">{parsed.errors.join(" ")}</p>
      ) : null}
      {error ? <p className="mt-2 text-xs text-rose-700">{error}</p> : null}

      {parsed && report ? (
        <div className="mt-4">
          <div className="flex flex-wrap items-center gap-4">
            <div>
              <div className="text-xs uppercase tracking-wide text-slate-500">Weighted score</div>
              <div className="text-3xl font-semibold tabular text-slate-900">
                {report.score.toFixed(1)}
                <span className="ml-2 text-base text-slate-400">{report.grade}</span>
              </div>
            </div>
            <div className="text-xs text-slate-500">
              <div>
                {parsed.dataset.rows.length} rows · {parsed.dataset.featureNames.length} features · label{" "}
                <code>{parsed.dataset.targetName}</code>
                {fileName ? ` · ${fileName}` : ""}
              </div>
              <div>
                Browser coverage: {(report.coverage * 100).toFixed(0)}% of the rubric weight (the rest needs a
                trained model).
              </div>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="data w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th>Check</th>
                  <th>Score</th>
                  <th>Severity</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {report.checks.map((c) => (
                  <tr key={c.id}>
                    <td className="font-medium text-slate-900">{c.name}</td>
                    <td className="tabular">{c.score.toFixed(3)}</td>
                    <td>
                      <span className={`inline-flex rounded-full border px-2 py-0.5 text-xs ${tone(c.severity)}`}>
                        {c.severity}
                      </span>
                    </td>
                    <td className="max-w-[34rem] whitespace-normal text-xs text-slate-500">
                      {evidenceText(c.evidence)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Severity bands: info ≥ {config.severity.info_min}, warn ≥ {config.severity.warn_min}, otherwise fail.
            The Python implementation validates these exact formulas against planted defects; see the validation
            table above.
          </p>
        </div>
      ) : (
        <p className="mt-3 text-xs text-slate-400">Load the example or paste data to see checks and a score.</p>
      )}
    </div>
  );
}
