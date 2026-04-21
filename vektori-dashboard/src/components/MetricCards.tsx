"use client";

import type { DashboardData } from "@/app/page";

type Props = { summary: DashboardData["summary"] };

type CardProps = {
  label: string;
  value: string;
  sub: string;
  accent: string;
  highlight?: boolean;
};

function Card({ label, value, sub, accent, highlight }: CardProps) {
  return (
    <div
      className="rounded-xl p-5 flex flex-col gap-2 relative overflow-hidden"
      style={{
        background: highlight ? "linear-gradient(135deg, #0f172a, #1e1b4b)" : "#0d1117",
        border: `1px solid ${highlight ? "#312e81" : "#1e293b"}`,
      }}
    >
      <div className="text-xs font-medium uppercase tracking-widest" style={{ color: "#475569" }}>
        {label}
      </div>
      <div className="text-4xl font-bold tracking-tight" style={{ color: accent }}>
        {value}
      </div>
      <div className="text-xs" style={{ color: "#64748b" }}>
        {sub}
      </div>
      {/* accent line */}
      <div
        className="absolute bottom-0 left-0 right-0 h-0.5"
        style={{ background: `linear-gradient(90deg, ${accent}88, transparent)` }}
      />
    </div>
  );
}

export default function MetricCards({ summary }: Props) {
  const j = summary.judge;
  const lat = summary.latency_ms;

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      <Card
        label="Context Hit Rate"
        value={`${Math.round(j.context_has_answer_rate * 100)}%`}
        sub={`Answer in retrieved context · ${j.n_judged} judged`}
        accent="#22d3ee"
        highlight
      />
      <Card
        label="QA Failure Rate"
        value={`${Math.round((j.qa_failure / j.n_judged) * 100)}%`}
        sub={`${j.qa_failure}/${j.n_judged} had context, model still failed`}
        accent="#f87171"
      />
      <Card
        label="Retrieval Failure"
        value={`${Math.round((j.retrieval_failure / j.n_judged) * 100)}%`}
        sub={`Only ${j.retrieval_failure} cases — retrieval works`}
        accent="#4ade80"
      />
      <Card
        label="F1 Score"
        value={summary.f1_avg.toFixed(3)}
        sub={`Exact match ${(summary.exact_match_rate * 100).toFixed(1)}% · ${summary.total_questions.toLocaleString()} Qs`}
        accent="#a78bfa"
      />
    </div>
  );
}
