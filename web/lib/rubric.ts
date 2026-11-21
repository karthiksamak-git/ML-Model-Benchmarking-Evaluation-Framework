/**
 * Browser-side dataset-quality rubric.
 *
 * This is a TypeScript port of the checks in
 * `src/mlbe/rubrics/dataset_quality.py` that can run without a model. The
 * weighted score and severity bands match the Python implementation exactly;
 * golden vectors in `tests/golden/rubric_golden.json` are generated from the
 * Python code and verified by `npm run test:golden`.
 *
 * Checks that need model predictions (label noise, train/test leakage,
 * covariate shift) and the three-method outlier vote are not reproduced in the
 * browser. The browser adds an IQR-only outlier heuristic that is labelled as
 * such in the UI.
 */

export type Severity = "info" | "warn" | "fail";

export interface Thresholds {
  [key: string]: number;
}

export interface RubricConfig {
  severity: { info_min: number; warn_min: number };
  dataset_quality: {
    weights: Record<string, number>;
    thresholds: Thresholds;
  };
}

export interface CheckResult {
  id: string;
  name: string;
  score: number;
  severity: Severity;
  evidence: Record<string, number | string | null | Record<string, number>>;
  recommendation: string;
}

export interface Dataset {
  featureNames: string[];
  rows: (number | null)[][];
  target: (number | null)[] | null;
  targetName: string | null;
}

export function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export function scoreLowerBetter(value: number, ok: number, warn: number, fail: number): number {
  if (!Number.isFinite(value)) return 1;
  if (value <= ok) return 1;
  if (value <= warn) {
    const span = Math.max(warn - ok, 1e-12);
    return 1 - (0.15 * (value - ok)) / span;
  }
  if (value <= fail) {
    const span = Math.max(fail - warn, 1e-12);
    return 0.85 - (0.3 * (value - warn)) / span;
  }
  const span = Math.max(Math.abs(fail), 1e-9);
  return Math.max(0, 0.55 - 0.55 * Math.min(1, (value - fail) / span));
}

export function scoreHigherBetter(value: number, ok: number, warn: number, fail: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value >= ok) return 1;
  if (value >= warn) {
    const span = Math.max(ok - warn, 1e-12);
    return 1 - (0.15 * (ok - value)) / span;
  }
  if (value >= fail) {
    const span = Math.max(warn - fail, 1e-12);
    return 0.85 - (0.3 * (warn - value)) / span;
  }
  const span = Math.max(Math.abs(fail), 1e-9);
  return Math.max(0, 0.55 - 0.55 * Math.min(1, (fail - value) / span));
}

export function severityFor(score: number, cfg: RubricConfig): Severity {
  if (score >= cfg.severity.info_min) return "info";
  if (score >= cfg.severity.warn_min) return "warn";
  return "fail";
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function pearson(a: number[], b: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  if (da <= 0 || db <= 0) return 0;
  return num / Math.sqrt(da * db);
}

function uniqueLabels(target: (number | null)[]): number[] {
  const set = new Set<number>();
  for (const v of target) if (v !== null && Number.isFinite(v)) set.add(Number(v));
  return [...set].sort((a, b) => a - b);
}

function rowKey(row: (number | null)[]): string {
  return row.map((v) => (v === null || !Number.isFinite(v) ? "NaN" : String(v))).join("|");
}

export function checkMissing(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  const total = ds.rows.length * ds.featureNames.length;
  let missing = 0;
  const perCol = new Array(ds.featureNames.length).fill(0);
  for (const row of ds.rows) {
    row.forEach((v, j) => {
      if (v === null || !Number.isFinite(v)) {
        missing += 1;
        perCol[j] += 1;
      }
    });
  }
  const rate = total ? missing / total : 0;
  const top = ds.featureNames
    .map((name, j) => ({ column: name, missing_rate: ds.rows.length ? perCol[j] / ds.rows.length : 0 }))
    .filter((d) => d.missing_rate > 0)
    .sort((a, b) => b.missing_rate - a.missing_rate)
    .slice(0, 5);
  const score = scoreLowerBetter(rate, 0, t.missing_rate_warn, t.missing_rate_fail);
  return {
    id: "missing_values",
    name: "Missing values",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      overall_missing_rate: rate,
      columns_with_missing: perCol.filter((c) => c > 0).length,
      n_columns: ds.featureNames.length,
      top_columns: top as unknown as Record<string, number>,
    },
    recommendation: "Impute or drop missing cells; if missingness is informative, encode it explicitly.",
  };
}

export function checkConstantColumns(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  const constant: string[] = [];
  const near: string[] = [];
  ds.featureNames.forEach((name, j) => {
    const col = ds.rows.map((r) => r[j]).filter((v): v is number => v !== null && Number.isFinite(v));
    if (!col.length) {
      constant.push(name);
      return;
    }
    const counts = new Map<number, number>();
    col.forEach((v) => counts.set(v, (counts.get(v) ?? 0) + 1));
    const unique = counts.size;
    const maxCount = Math.max(...counts.values());
    if (unique <= t.constant_unique_values) constant.push(name);
    else if (maxCount / col.length >= 1 - t.near_constant_share) near.push(name);
  });
  const share = (constant.length + near.length) / Math.max(1, ds.featureNames.length);
  const score = scoreLowerBetter(share, 0, 0.05, 0.2);
  return {
    id: "constant_columns",
    name: "Constant / near-constant columns",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      n_constant: constant.length,
      n_near_constant: near.length,
      share,
      constant_columns: constant.slice(0, 10).join(", "),
      near_constant_columns: near.slice(0, 10).join(", "),
    },
    recommendation: "Drop constant columns; they carry no signal and inflate dimensionality.",
  };
}

export function checkDuplicates(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  const groups = new Map<string, number[]>();
  ds.rows.forEach((row, i) => {
    const key = rowKey(row);
    const list = groups.get(key);
    if (list) list.push(i);
    else groups.set(key, [i]);
  });
  let duplicates = 0;
  groups.forEach((idx) => {
    duplicates += Math.max(0, idx.length - 1);
  });
  const rate = duplicates / Math.max(1, ds.rows.length);
  const score = scoreLowerBetter(rate, 0, t.duplicate_rate_warn, t.duplicate_rate_fail);
  return {
    id: "duplicates",
    name: "Exact duplicate rows",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      n_rows: ds.rows.length,
      n_unique_rows: groups.size,
      n_duplicate_rows: duplicates,
      duplicate_rate: rate,
    },
    recommendation: "De-duplicate before splitting, otherwise identical rows leak across folds.",
  };
}

export function checkConflictingLabels(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  if (!ds.target) {
    return {
      id: "conflicting_labels",
      name: "Duplicate rows with conflicting labels",
      score: 1,
      severity: "info",
      evidence: { status: "skipped (no target column selected)" },
      recommendation: "Select the label column to check for conflicting duplicates.",
    };
  }
  const groups = new Map<string, number[]>();
  ds.rows.forEach((row, i) => {
    const key = rowKey(row);
    const list = groups.get(key);
    if (list) list.push(i);
    else groups.set(key, [i]);
  });
  let groupsConflict = 0;
  let conflicts = 0;
  groups.forEach((idx) => {
    if (idx.length < 2) return;
    const labels = new Set(idx.map((i) => ds.target?.[i]));
    if (labels.size > 1) {
      groupsConflict += 1;
      conflicts += idx.length - 1;
    }
  });
  const rate = conflicts / Math.max(1, ds.rows.length);
  const score = scoreLowerBetter(rate, 0, t.conflict_rate_warn, t.conflict_rate_fail);
  return {
    id: "conflicting_labels",
    name: "Duplicate rows with conflicting labels",
    score,
    severity: severityFor(score, cfg),
    evidence: { n_conflicting_groups: groupsConflict, n_conflicting_rows: conflicts, rate },
    recommendation: "Resolve conflicting labels (relabel or drop); they cap achievable accuracy.",
  };
}

export function checkClassImbalance(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  if (!ds.target) {
    return {
      id: "class_imbalance",
      name: "Class imbalance",
      score: 1,
      severity: "info",
      evidence: { status: "skipped (no target column selected)" },
      recommendation: "Select the label column to assess class balance.",
    };
  }
  const counts = new Map<number, number>();
  ds.target.forEach((v) => {
    if (v !== null && Number.isFinite(v)) counts.set(Number(v), (counts.get(Number(v)) ?? 0) + 1);
  });
  const values = [...counts.values()];
  const total = values.reduce((a, b) => a + b, 0);
  const ratio = values.length ? Math.max(...values) / Math.max(1, Math.min(...values)) : 1;
  const minority = total ? Math.min(...values) / total : 1;
  const score = Math.min(
    scoreLowerBetter(ratio, 1, t.imbalance_ratio_warn, t.imbalance_ratio_fail),
    scoreHigherBetter(minority, 0.25, t.minority_share_warn, t.minority_share_fail),
  );
  return {
    id: "class_imbalance",
    name: "Class imbalance",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      n_classes: values.length,
      imbalance_ratio: ratio,
      minority_share: minority,
      class_counts: JSON.stringify([...counts.entries()].sort((a, b) => a[0] - b[0])),
    },
    recommendation: "Use stratified splits and class weights; report balanced accuracy and per-class recall.",
  };
}

export function checkLowInformation(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  if (!ds.target) {
    return {
      id: "low_information",
      name: "Low-information features",
      score: 1,
      severity: "info",
      evidence: { status: "skipped (no target column selected)" },
      recommendation: "Select the label column to score feature information.",
    };
  }
  const labels = uniqueLabels(ds.target);
  const scores: number[] = ds.featureNames.map((_, j) => {
    const pairs = ds.rows
      .map((row, i) => [row[j], ds.target?.[i] ?? null] as [number | null, number | null])
      .filter((p): p is [number, number] => p[0] !== null && Number.isFinite(p[0]) && p[1] !== null);
    if (!pairs.length) return 0;
    const col = pairs.map((p) => p[0]);
    if (new Set(col).size <= 1) return 0;
    if (labels.length === 2) {
      return Math.abs(pearson(col, pairs.map((p) => p[1])));
    }
    return Math.max(...labels.map((c) => Math.abs(pearson(col, pairs.map((p) => (p[1] === c ? 1 : 0))))));
  });
  const share = scores.filter((s) => s < t.low_info_abs_corr).length / Math.max(1, ds.featureNames.length);
  const score = scoreLowerBetter(share, 0.1, t.low_info_share_warn, 0.8);
  const low = ds.featureNames.filter((_, j) => scores[j] < t.low_info_abs_corr).slice(0, 12);
  return {
    id: "low_information",
    name: "Low-information features",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      share_below_threshold: share,
      threshold_abs_correlation: t.low_info_abs_corr,
      low_information_features: low.join(", "),
    },
    recommendation: "Confirm with multivariate importance before dropping; univariate filters miss interactions.",
  };
}

export function checkSampleAdequacy(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  const n = ds.rows.length;
  const d = Math.max(1, ds.featureNames.length);
  const perFeature = n / d;
  let minClass = 0;
  if (ds.target) {
    const counts = new Map<number, number>();
    ds.target.forEach((v) => {
      if (v !== null && Number.isFinite(v)) counts.set(Number(v), (counts.get(Number(v)) ?? 0) + 1);
    });
    const values = [...counts.values()];
    minClass = values.length ? Math.min(...values) : 0;
  }
  const score = Math.min(
    scoreHigherBetter(perFeature, t.min_samples_per_feature, t.min_samples_per_feature / 2, 2),
    ds.target
      ? scoreHigherBetter(minClass, t.min_samples_per_class, t.min_samples_per_class / 2, 5)
      : 1,
  );
  return {
    id: "sample_adequacy",
    name: "Sample-size adequacy",
    score,
    severity: severityFor(score, cfg),
    evidence: {
      n_train: n,
      n_features: ds.featureNames.length,
      samples_per_feature: perFeature,
      min_samples_per_class: minClass,
    },
    recommendation: "Collect more data (especially minority classes) or reduce dimensionality/regularise.",
  };
}

/** Browser-only heuristic (IQR fences); not part of the Python golden vectors. */
export function checkOutliersIqr(ds: Dataset, cfg: RubricConfig): CheckResult {
  const t = cfg.dataset_quality.thresholds;
  const factor = t.outlier_iqr_factor;
  const flagged = new Array(ds.rows.length).fill(false);
  ds.featureNames.forEach((_, j) => {
    const col = ds.rows.map((r) => r[j]).filter((v): v is number => v !== null && Number.isFinite(v));
    if (col.length < 4) return;
    const sorted = [...col].sort((a, b) => a - b);
    const q = (p: number) => {
      const pos = (sorted.length - 1) * p;
      const lo = Math.floor(pos);
      const hi = Math.ceil(pos);
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    };
    const q1 = q(0.25);
    const q3 = q(0.75);
    const iqr = q3 - q1;
    if (iqr === 0) return;
    ds.rows.forEach((row, i) => {
      const v = row[j];
      if (v !== null && Number.isFinite(v) && (v < q1 - factor * iqr || v > q3 + factor * iqr)) flagged[i] = true;
    });
  });
  const rate = flagged.filter(Boolean).length / Math.max(1, ds.rows.length);
  const score = scoreLowerBetter(rate, 0, t.outlier_rate_warn, t.outlier_rate_fail);
  return {
    id: "outliers_iqr",
    name: "Outliers (IQR heuristic, browser)",
    score,
    severity: severityFor(score, cfg),
    evidence: { outlier_rate: rate, n_flagged: flagged.filter(Boolean).length, note: "IQR fences only; the full pipeline votes across IQR, robust z and Isolation Forest." },
    recommendation: "Inspect flagged rows; winsorise or use models robust to heavy tails.",
  };
}

export const BROWSER_CHECKS = [
  checkMissing,
  checkConstantColumns,
  checkDuplicates,
  checkConflictingLabels,
  checkClassImbalance,
  checkLowInformation,
  checkSampleAdequacy,
  checkOutliersIqr,
];

export interface DatasetReport {
  checks: CheckResult[];
  score: number;
  grade: string;
  severity: Severity;
  coverage: number;
}

export function evaluateDataset(ds: Dataset, cfg: RubricConfig): DatasetReport {
  const checks = BROWSER_CHECKS.map((fn) => fn(ds, cfg));
  const weights = cfg.dataset_quality.weights;
  const available = checks.filter((c) => !String(c.evidence.status ?? "").startsWith("skipped"));
  const totalWeight = available.reduce((s, c) => s + (weights[c.id] ?? 0), 0);
  const score = totalWeight
    ? (100 * available.reduce((s, c) => s + (weights[c.id] ?? 0) * c.score, 0)) / totalWeight
    : 0;
  const severity: Severity = checks.some((c) => c.severity === "fail")
    ? "fail"
    : checks.some((c) => c.severity === "warn")
      ? "warn"
      : "info";
  const grade = score >= 90 ? "A" : score >= 80 ? "B" : score >= 70 ? "C" : score >= 60 ? "D" : "F";
  return { checks, score, grade, severity, coverage: totalWeight / Math.max(1e-9, Object.values(weights).reduce((a, b) => a + b, 0)) };
}

export interface ParsedCsv {
  dataset: Dataset;
  errors: string[];
}

/** Minimal, dependency-free CSV parser for numeric feature columns. */
export function parseCsv(text: string): ParsedCsv {
  const errors: string[] = [];
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    return { dataset: { featureNames: [], rows: [], target: null, targetName: null }, errors: ["Need a header row and at least one data row."] };
  }
  const splitLine = (line: string): string[] => {
    const out: string[] = [];
    let cur = "";
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') {
          cur += '"';
          i += 1;
        } else quoted = !quoted;
      } else if (ch === "," && !quoted) {
        out.push(cur.trim());
        cur = "";
      } else cur += ch;
    }
    out.push(cur.trim());
    return out;
  };
  const header = splitLine(lines[0]);
  const raw = lines.slice(1).map(splitLine);
  const width = header.length;
  if (raw.some((r) => r.length !== width)) errors.push("Some rows have a different number of columns than the header.");
  const parsed: (number | null)[][] = raw.map((r) =>
    header.map((_, j) => {
      const cell = r[j];
      if (cell === undefined || cell === "" || cell.toLowerCase() === "na" || cell.toLowerCase() === "nan") return null;
      const v = Number(cell);
      return Number.isFinite(v) ? v : null;
    }),
  );
  const targetIdx = Math.max(0, header.length - 1);
  const targetName = header[targetIdx];
  const featureIdx = header.map((_, j) => j).filter((j) => j !== targetIdx);
  const dataset: Dataset = {
    featureNames: featureIdx.map((j) => header[j]),
    rows: parsed.map((r) => featureIdx.map((j) => r[j])),
    target: parsed.map((r) => r[targetIdx]),
    targetName,
  };
  return { dataset, errors };
}
