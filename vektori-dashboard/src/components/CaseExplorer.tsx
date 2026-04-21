"use client";

import { useState } from "react";
import type { JudgeCase } from "@/app/page";

type Props = {
  cases: JudgeCase[];
  selectedVerdict: string;
  onVerdictChange: (v: string) => void;
};

const VERDICT_STYLES: Record<string, { bg: string; text: string; border: string; label: string }> = {
  CORRECT: { bg: "#052e16", text: "#4ade80", border: "#166534", label: "Correct" },
  PARTIALLY_CORRECT: { bg: "#1c1400", text: "#facc15", border: "#713f12", label: "Partial" },
  WRONG: { bg: "#1c0a0a", text: "#f87171", border: "#7f1d1d", label: "Wrong" },
  ABSTAINED: { bg: "#0f172a", text: "#64748b", border: "#334155", label: "Abstained" },
};

const FILTER_OPTIONS = [
  { key: "QA_FAILURE", label: "QA Failures", desc: "Context had answer — model failed" },
  { key: "WRONG", label: "All Wrong", desc: "Any wrong verdict" },
  { key: "CORRECT", label: "Correct", desc: "Model got it right" },
  { key: "ALL", label: "All Cases", desc: "Show everything" },
];

function ContextSection({ context }: { context: string }) {
  const [expanded, setExpanded] = useState(false);
  const preview = context.slice(0, 400);

  return (
    <div>
      <div
        className="text-xs rounded-lg p-3 font-mono leading-relaxed"
        style={{
          background: "#050810",
          border: "1px solid #1e293b",
          color: "#94a3b8",
          whiteSpace: "pre-wrap",
          maxHeight: expanded ? "none" : "120px",
          overflow: "hidden",
        }}
      >
        {expanded ? context : preview + (context.length > 400 ? "…" : "")}
      </div>
      {context.length > 400 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs mt-1 hover:underline"
          style={{ color: "#6366f1" }}
        >
          {expanded ? "Show less" : `Show full context (${Math.round(context.length / 4)} tokens est.)`}
        </button>
      )}
    </div>
  );
}

function CaseCard({ c, index }: { c: JudgeCase; index: number }) {
  const [open, setOpen] = useState(index === 0);
  const vs = VERDICT_STYLES[c.verdict] || VERDICT_STYLES.ABSTAINED;
  const isSynthesisFailure = c.context_has_answer && c.verdict === "WRONG";

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: `1px solid ${open ? "#334155" : "#1e293b"}`, background: "#0d1117" }}
    >
      {/* Card header — always visible */}
      <button
        className="w-full text-left px-5 py-4 flex items-start gap-4 hover:bg-white/[0.02] transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex-shrink-0 mt-0.5">
          <span
            className="text-xs font-semibold px-2 py-1 rounded"
            style={{ background: vs.bg, color: vs.text, border: `1px solid ${vs.border}` }}
          >
            {vs.label}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-white font-medium leading-snug">{c.question}</p>
          <div className="flex items-center gap-4 mt-1.5 flex-wrap">
            {isSynthesisFailure && (
              <span
                className="text-xs font-medium px-2 py-0.5 rounded"
                style={{ background: "#1e0a0a", color: "#f87171", border: "1px solid #7f1d1d" }}
              >
                ⚡ Synthesis failure — context had answer
              </span>
            )}
            {c.context_has_answer && c.verdict !== "WRONG" && (
              <span className="text-xs" style={{ color: "#4ade80" }}>
                ✓ Context hit
              </span>
            )}
            <span className="text-xs" style={{ color: "#475569" }}>
              Type {c.question_type} · F1 {c.f1.toFixed(2)} ·{" "}
              {c.retrieval_ms.toFixed(0)}ms retrieval · {c.qa_ms.toFixed(0)}ms QA
            </span>
          </div>
        </div>
        <span className="text-xs flex-shrink-0 mt-1" style={{ color: "#334155" }}>
          {open ? "▲" : "▼"}
        </span>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="px-5 pb-5 space-y-4" style={{ borderTop: "1px solid #1e293b" }}>
          {/* Answer comparison */}
          <div className="grid grid-cols-1 gap-3 mt-4 sm:grid-cols-2">
            <div>
              <p className="text-xs font-medium uppercase tracking-widest mb-1.5" style={{ color: "#475569" }}>
                Model said
              </p>
              <div
                className="rounded-lg px-3 py-2 text-sm"
                style={{
                  background: c.verdict === "CORRECT" ? "#052e16" : "#1c0a0a",
                  border: `1px solid ${c.verdict === "CORRECT" ? "#166534" : "#7f1d1d"}`,
                  color: c.verdict === "CORRECT" ? "#4ade80" : "#f87171",
                }}
              >
                {c.model_answer || <span style={{ color: "#475569", fontStyle: "italic" }}>No answer / abstained</span>}
              </div>
            </div>
            <div>
              <p className="text-xs font-medium uppercase tracking-widest mb-1.5" style={{ color: "#475569" }}>
                Expected
              </p>
              <div
                className="rounded-lg px-3 py-2 text-sm"
                style={{ background: "#052e16", border: "1px solid #166534", color: "#4ade80" }}
              >
                {c.expected_answer}
              </div>
            </div>
          </div>

          {/* Judge explanation */}
          {c.judge_explanation && (
            <div
              className="rounded-lg px-4 py-3"
              style={{ background: "#0f172a", border: "1px solid #1e293b" }}
            >
              <p className="text-xs font-medium uppercase tracking-widest mb-1" style={{ color: "#475569" }}>
                Judge explanation
              </p>
              <p className="text-sm" style={{ color: "#94a3b8" }}>
                {c.judge_explanation}
              </p>
            </div>
          )}

          {/* Retrieval stats */}
          <div className="flex gap-4 flex-wrap">
            {[
              { label: "Facts", value: c.facts_retrieved },
              { label: "Episodes", value: c.episodes_retrieved },
              { label: "Sentences", value: c.sentences_retrieved },
            ].map((s) => (
              <div
                key={s.label}
                className="text-xs rounded-lg px-3 py-2"
                style={{ background: "#050810", border: "1px solid #1e293b" }}
              >
                <span style={{ color: "#475569" }}>{s.label}: </span>
                <span className="font-semibold" style={{ color: "#818cf8" }}>
                  {s.value}
                </span>
              </div>
            ))}
            <div
              className="text-xs rounded-lg px-3 py-2"
              style={{ background: "#050810", border: "1px solid #1e293b" }}
            >
              <span style={{ color: "#475569" }}>Context has answer: </span>
              <span
                className="font-semibold"
                style={{ color: c.context_has_answer ? "#4ade80" : "#f87171" }}
              >
                {c.context_has_answer ? "YES" : "NO"}
              </span>
            </div>
          </div>

          {/* Retrieved context */}
          <div>
            <p className="text-xs font-medium uppercase tracking-widest mb-1.5" style={{ color: "#475569" }}>
              Retrieved context
            </p>
            <ContextSection context={c.retrieved_context} />
          </div>
        </div>
      )}
    </div>
  );
}

export default function CaseExplorer({ cases, selectedVerdict, onVerdictChange }: Props) {
  const filtered = cases.filter((c) => {
    if (selectedVerdict === "ALL") return true;
    if (selectedVerdict === "QA_FAILURE") return c.failure_mode === "QA_FAILURE";
    return c.verdict === selectedVerdict;
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-5 flex-wrap gap-4">
        <div>
          <h2 className="text-base font-semibold text-white">Case Explorer</h2>
          <p className="text-xs mt-0.5" style={{ color: "#475569" }}>
            {filtered.length} cases · click any card to expand
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => onVerdictChange(opt.key)}
              className="text-xs px-3 py-1.5 rounded-lg font-medium transition-all"
              style={
                selectedVerdict === opt.key
                  ? {
                      background: "#1e1b4b",
                      color: "#a5b4fc",
                      border: "1px solid #4338ca",
                    }
                  : {
                      background: "#0d1117",
                      color: "#475569",
                      border: "1px solid #1e293b",
                    }
              }
              title={opt.desc}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div
          className="rounded-xl p-8 text-center"
          style={{ background: "#0d1117", border: "1px solid #1e293b" }}
        >
          <p style={{ color: "#475569" }}>No cases match this filter.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.slice(0, 20).map((c, i) => (
            <CaseCard key={c.question_id} c={c} index={i} />
          ))}
          {filtered.length > 20 && (
            <p className="text-xs text-center py-2" style={{ color: "#475569" }}>
              Showing 20 of {filtered.length} cases
            </p>
          )}
        </div>
      )}
    </div>
  );
}
