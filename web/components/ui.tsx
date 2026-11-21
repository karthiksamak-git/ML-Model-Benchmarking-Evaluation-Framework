import Link from "next/link";
import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card p-5 ${className}`}>{children}</div>;
}

export function Section({
  title,
  subtitle,
  children,
  action,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="mt-8">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-900">{title}</h2>
          {subtitle ? <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const tones = {
    default: "text-slate-900",
    good: "text-emerald-700",
    warn: "text-amber-700",
    bad: "text-rose-700",
  } as const;
  return (
    <div className="card p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular ${tones[tone]}`}>{value}</div>
      {hint ? <div className="mt-1 text-xs text-slate-500">{hint}</div> : null}
    </div>
  );
}

export function Badge({
  children,
  tone = "slate",
}: {
  children: ReactNode;
  tone?: "slate" | "green" | "amber" | "rose" | "blue";
}) {
  const tones = {
    slate: "bg-slate-100 text-slate-700 border-slate-200",
    green: "bg-emerald-50 text-emerald-700 border-emerald-200",
    amber: "bg-amber-50 text-amber-800 border-amber-200",
    rose: "bg-rose-50 text-rose-700 border-rose-200",
    blue: "bg-blue-50 text-blue-700 border-blue-200",
  } as const;
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function SeverityBadge({ severity }: { severity: string }) {
  const tone = severity === "info" ? "green" : severity === "warn" ? "amber" : "rose";
  return <Badge tone={tone}>{severity}</Badge>;
}

export function ScoreBadge({ score, grade }: { score: number; grade?: string }) {
  const tone = score >= 85 ? "green" : score >= 60 ? "amber" : "rose";
  return (
    <Badge tone={tone}>
      {score.toFixed(1)}
      {grade ? ` · ${grade}` : ""}
    </Badge>
  );
}

export function Provenance({ provenance }: { provenance: any }) {
  if (!provenance) return null;
  const sha = String(provenance.git_hash ?? "unknown").slice(0, 12);
  const when = String(provenance.generated_at ?? "").replace("T", " ").replace("+00:00", " UTC");
  return (
    <p className="mt-6 text-xs leading-relaxed text-slate-400">
      Every number on this page is read from MLflow runs exported by <code>mlbe export</code>. Git{" "}
      <code>{sha}</code> · exported {when} · {provenance.source_run_ids?.length ?? 0} source runs ·{" "}
      {provenance.mlflow_experiment}
    </p>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-slate-300 p-6 text-sm text-slate-500">{children}</div>;
}

export function PageHeader({ title, lead }: { title: string; lead: ReactNode }) {
  return (
    <div className="border-b border-slate-200 pb-5">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
      <p className="mt-2 max-w-4xl text-sm leading-relaxed text-slate-600">{lead}</p>
    </div>
  );
}

export function LinkButton({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      className="inline-flex items-center rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
    >
      {children}
    </Link>
  );
}
