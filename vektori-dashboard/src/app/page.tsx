import dashboardData from "@/data/dashboard_data.json";
import Dashboard from "@/components/Dashboard";

export default function Home() {
  return <Dashboard data={dashboardData as DashboardData} />;
}

export type JudgeCase = {
  question_id: string;
  question: string;
  question_type: string;
  expected_answer: string;
  model_answer: string;
  retrieved_context: string;
  facts_retrieved: number;
  sentences_retrieved: number;
  episodes_retrieved: number;
  retrieval_ms: number;
  qa_ms: number;
  f1: number;
  exact_match: boolean;
  verdict: "CORRECT" | "PARTIALLY_CORRECT" | "WRONG" | "ABSTAINED";
  context_has_answer: boolean;
  failure_mode: string | null;
  judge_explanation: string;
  speaker_a: string;
  speaker_b: string;
};

export type DashboardData = {
  summary: {
    total_questions: number;
    f1_avg: number;
    exact_match_rate: number;
    by_type: Record<string, { total: number; f1_avg: number; exact_match_rate: number }>;
    judge: {
      n_judged: number;
      correct: number;
      partially_correct: number;
      wrong: number;
      abstained: number;
      correct_rate: number;
      combined_rate: number;
      context_has_answer_rate: number;
      qa_failure: number;
      retrieval_failure: number;
      by_type: Record<string, { total: number; correct: number; partial: number; ctx_ok: number }>;
    };
    retrieval_counts_avg: { facts: number; sentences: number; episodes: number };
    latency_ms: {
      retrieval_ms: { avg: number; p95: number };
      qa_ms: { avg: number; p95: number };
      total_question_ms: { avg: number; p95: number };
    };
  };
  config: {
    embedding_model: string;
    extraction_model: string;
    eval_model: string;
    retrieval_depth: string;
    top_k: number;
  };
  judge_cases: JudgeCase[];
};
