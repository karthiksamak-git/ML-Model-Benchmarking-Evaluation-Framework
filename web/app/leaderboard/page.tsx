import { loadData } from "@/lib/data";
import { PageHeader, Provenance } from "@/components/ui";
import Leaderboard from "@/components/Leaderboard";

export default function LeaderboardPage() {
  const leaderboard = loadData("leaderboard.json");
  const pareto = loadData("pareto.json");

  return (
    <div>
      <PageHeader
        title="Leaderboard"
        lead={
          <>
            Mean accuracy over identical repeated-CV folds with bootstrap 95% intervals. Re-weight accuracy,
            generalization and computational performance to see how the ranking changes — the scoring formula is
            the same one implemented in Python (<code>stats/ranking.py</code>). Rows are per dataset or averaged
            across datasets; the Pareto chart uses exported (accuracy, latency, size) points.
          </>
        }
      />
      <div className="mt-6">
        <Leaderboard rows={leaderboard.rows} overall={leaderboard.overall} pareto={pareto} />
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Interval construction: 2,000-sample bootstrap over the {leaderboard.rows?.[0]?.n_folds ?? "k"} per-fold
        accuracy values. A CI describes fold variability, not a guarantee on new data.
      </p>
      <Provenance provenance={leaderboard.provenance} />
    </div>
  );
}
