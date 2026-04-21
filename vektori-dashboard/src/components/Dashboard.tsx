"use client";

import { useState } from "react";
import type { DashboardData, JudgeCase } from "@/app/page";
import MetricCards from "./MetricCards";
import FailureBreakdown from "./FailureBreakdown";
import CaseExplorer from "./CaseExplorer";

export default function Dashboard({ data }: { data: DashboardData }) {
  const [selectedVerdict, setSelectedVerdict] = useState<string>("QA_FAILURE");

  const { summary, config, judge_cases } = data;

  return (
    <div className="min-h-screen" style={{ background: "#0a0a0f", color: "#e2e8f0" }}>
      {/* Header */}
      <header
        className="border-b px-8 py-5 flex items-center justify-between"
        style={{ borderColor: "#1e293b", background: "#0d1117" }}
      >
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
              style={{ background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)" }}
            >
              V
            </div>
            <span className="text-white font-semibold text-lg tracking-tight">vektori</span>
          </div>
          <span style={{ color: "#334155" }}>·</span>
          <span style={{ color: "#64748b" }} className="text-sm">
            Benchmark Dashboard
          </span>
        </div>
        <div className="flex items-center gap-6 text-xs" style={{ color: "#475569" }}>
          <span>
            Model:{" "}
            <span style={{ color: "#94a3b8" }}>
              {config.eval_model.replace("gemini:", "")}
            </span>
          </span>
          <span>
            Embedding:{" "}
            <span style={{ color: "#94a3b8" }}>
              {config.embedding_model.replace("cloudflare:", "")}
            </span>
          </span>
          <span>
            Depth:{" "}
            <span style={{ color: "#94a3b8" }}>{config.retrieval_depth.toUpperCase()}</span>
          </span>
          <span
            className="px-2 py-0.5 rounded text-xs font-medium"
            style={{ background: "#0f2027", color: "#22d3ee", border: "1px solid #164e63" }}
          >
            LoCoMo · {summary.total_questions.toLocaleString()} questions
          </span>
        </div>
      </header>

      <main className="px-8 py-8 max-w-7xl mx-auto space-y-8">
        {/* The insight callout */}
        <div
          className="rounded-xl px-6 py-4 flex items-center gap-4"
          style={{
            background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
            border: "1px solid #312e81",
          }}
        >
          <div
            className="w-2 h-10 rounded-full flex-shrink-0"
            style={{ background: "linear-gradient(180deg, #6366f1, #8b5cf6)" }}
          />
          <div>
            <p className="text-white font-semibold text-base">
              The bottleneck is not retrieval — it&apos;s synthesis.
            </p>
            <p className="text-sm mt-0.5" style={{ color: "#818cf8" }}>
              Context contained the answer in <strong>98%</strong> of questions. The model failed
              to extract it in <strong>33%</strong> of judged cases.
            </p>
          </div>
        </div>

        {/* Metric Cards */}
        <MetricCards summary={summary} />

        {/* Charts row */}
        <FailureBreakdown summary={summary} />

        {/* Case Explorer */}
        <CaseExplorer
          cases={judge_cases}
          selectedVerdict={selectedVerdict}
          onVerdictChange={setSelectedVerdict}
        />
      </main>
    </div>
  );
}
