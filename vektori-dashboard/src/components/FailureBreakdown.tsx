"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import type { DashboardData } from "@/app/page";

type Props = { summary: DashboardData["summary"] };

const VERDICT_COLORS: Record<string, string> = {
  CORRECT: "#4ade80",
  PARTIALLY_CORRECT: "#facc15",
  WRONG: "#f87171",
  ABSTAINED: "#475569",
};

const TYPE_LABELS: Record<string, string> = {
  "1": "Single-hop",
  "3": "Inference",
  "4": "Multi-fact",
};

export default function FailureBreakdown({ summary }: Props) {
  const j = summary.judge;

  const verdictData = [
    { name: "Correct", value: j.correct, color: VERDICT_COLORS.CORRECT },
    { name: "Partial", value: j.partially_correct, color: VERDICT_COLORS.PARTIALLY_CORRECT },
    { name: "Wrong", value: j.wrong, color: VERDICT_COLORS.WRONG },
    { name: "Abstained", value: j.abstained, color: VERDICT_COLORS.ABSTAINED },
  ];

  const failureModeData = [
    {
      name: "QA Failure",
      value: j.qa_failure,
      desc: "Context had answer — model failed",
      color: "#f87171",
    },
    {
      name: "Retrieval Failure",
      value: j.retrieval_failure,
      desc: "Answer not in retrieved context",
      color: "#fb923c",
    },
  ];

  const typeData = Object.entries(summary.by_type).map(([type, stats]) => ({
    name: TYPE_LABELS[type] || `Type ${type}`,
    f1: Number((stats.f1_avg * 100).toFixed(1)),
    exact: Number((stats.exact_match_rate * 100).toFixed(1)),
    total: stats.total,
  }));

  const lat = summary.latency_ms;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* Verdict pie */}
      <div
        className="rounded-xl p-6"
        style={{ background: "#0d1117", border: "1px solid #1e293b" }}
      >
        <h3 className="text-sm font-semibold mb-4 uppercase tracking-widest" style={{ color: "#64748b" }}>
          Verdict Distribution
        </h3>
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie
              data={verdictData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={75}
              paddingAngle={3}
              dataKey="value"
            >
              {verdictData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8 }}
              labelStyle={{ color: "#94a3b8" }}
              itemStyle={{ color: "#e2e8f0" }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="grid grid-cols-2 gap-2 mt-2">
          {verdictData.map((d) => (
            <div key={d.name} className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: d.color }} />
              <span style={{ color: "#94a3b8" }}>
                {d.name}: <span className="text-white font-medium">{d.value}</span>
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Failure modes */}
      <div
        className="rounded-xl p-6"
        style={{ background: "#0d1117", border: "1px solid #1e293b" }}
      >
        <h3 className="text-sm font-semibold mb-4 uppercase tracking-widest" style={{ color: "#64748b" }}>
          Failure Modes
        </h3>
        <div className="space-y-4 mt-2">
          {failureModeData.map((d) => (
            <div key={d.name}>
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm font-medium text-white">{d.name}</span>
                <span className="text-sm font-bold" style={{ color: d.color }}>
                  {d.value}
                </span>
              </div>
              <div className="w-full rounded-full h-2" style={{ background: "#1e293b" }}>
                <div
                  className="h-2 rounded-full transition-all"
                  style={{
                    width: `${(d.value / j.n_judged) * 100}%`,
                    background: d.color,
                  }}
                />
              </div>
              <p className="text-xs mt-1" style={{ color: "#475569" }}>
                {d.desc}
              </p>
            </div>
          ))}
        </div>
        <div
          className="mt-6 rounded-lg p-3"
          style={{ background: "#0f172a", border: "1px solid #1e293b" }}
        >
          <p className="text-xs" style={{ color: "#818cf8" }}>
            <span className="font-semibold">
              {Math.round((j.qa_failure / Math.max(j.retrieval_failure, 1)))}×
            </span>{" "}
            more likely to fail at synthesis than retrieval
          </p>
        </div>
      </div>

      {/* By question type */}
      <div
        className="rounded-xl p-6"
        style={{ background: "#0d1117", border: "1px solid #1e293b" }}
      >
        <h3 className="text-sm font-semibold mb-4 uppercase tracking-widest" style={{ color: "#64748b" }}>
          F1 by Question Type
        </h3>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={typeData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8 }}
              labelStyle={{ color: "#94a3b8" }}
              itemStyle={{ color: "#e2e8f0" }}
              formatter={(v) => [`${v}%`]}
            />
            <Bar dataKey="f1" name="F1 %" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 space-y-2">
          {typeData.map((t) => (
            <div key={t.name} className="flex justify-between text-xs">
              <span style={{ color: "#64748b" }}>{t.name}</span>
              <span style={{ color: "#94a3b8" }}>
                {t.total} Qs · F1 {t.f1}% · EM {t.exact}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
