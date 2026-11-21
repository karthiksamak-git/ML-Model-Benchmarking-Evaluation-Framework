import fs from "node:fs";
import path from "node:path";

/** Load an exported static JSON artifact. Runs at build time only. */
export function loadData<T = any>(name: string): T {
  const file = path.join(process.cwd(), "public", "data", name);
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

export function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function provenanceLine(provenance: any): string {
  if (!provenance) return "";
  const sha = String(provenance.git_hash ?? "unknown").slice(0, 12);
  const when = String(provenance.generated_at ?? "").replace("T", " ").replace("+00:00", " UTC");
  return `git ${sha} · exported ${when} · ${provenance.source_run_ids?.length ?? 0} MLflow runs`;
}
