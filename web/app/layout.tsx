import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ml-bench-eval — model benchmarking & evaluation",
  description:
    "Repeated stratified CV benchmarks, statistical comparison, error analysis and automated dataset/model rubrics, exported from real MLflow runs.",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/statistics", label: "Statistics" },
  { href: "/experiments", label: "Experiments" },
  { href: "/errors", label: "Error analysis" },
  { href: "/rubrics", label: "Rubrics" },
  { href: "/environment", label: "Environment" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-3">
            <Link href="/" className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-900 text-xs font-bold text-white">
                mb
              </span>
              <span className="text-sm font-semibold tracking-tight text-slate-900">ml-bench-eval</span>
              <span className="hidden text-xs text-slate-400 sm:inline">benchmark · statistics · rubrics</span>
            </Link>
            <nav className="flex flex-wrap gap-1 text-sm">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-2.5 py-1.5 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
        <footer className="mx-auto mt-12 max-w-7xl border-t border-slate-200 px-5 py-8 text-xs leading-relaxed text-slate-500">
          Static Next.js dashboard. Training, MLflow tracking, statistical tests and rubric validation run locally
          or in CI — never on Vercel. Scores are produced by <code>mlbe export</code> from logged runs; the
          browser-side dataset checker is a tested TypeScript port of the Python checks.
        </footer>
      </body>
    </html>
  );
}
