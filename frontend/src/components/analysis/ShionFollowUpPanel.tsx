"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, HelpCircle, RefreshCw, Send } from "lucide-react";
import { apiClient } from "../../lib/api";
import type { JudgmentAssetCandidate } from "../../lib/shionReview";

type AnswerStatus = "confirmed" | "partial" | "concern";

type FollowupQuestion = {
  id: string;
  category: string;
  question: string;
  reason: string;
  hypothesis: string;
};

type FollowupAnswer = {
  question_id: string;
  status: AnswerStatus;
  note: string;
};

type FollowupSummary = {
  baseline_decision?: string;
  updated_decision?: string;
  change_reason?: string;
  approval_conditions?: string[];
  ringi_comment?: string;
  score_changed?: boolean;
};

type FollowupSession = {
  followup_id: string;
  case_id: string;
  baseline_decision: string;
  questions: FollowupQuestion[];
  answers: FollowupAnswer[];
  summary: FollowupSummary;
  status: "questions_ready" | "answered" | "outcome_linked" | string;
  outcome_status?: string;
};

const ANSWER_OPTIONS: { value: AnswerStatus; label: string; tone: string }[] = [
  { value: "confirmed", label: "確認できた", tone: "border-emerald-300 bg-emerald-50 text-emerald-800" },
  { value: "partial", label: "一部確認", tone: "border-amber-300 bg-amber-50 text-amber-800" },
  { value: "concern", label: "未確認・懸念あり", tone: "border-rose-300 bg-rose-50 text-rose-800" },
];

export default function ShionFollowUpPanel({
  caseId,
  reviewId,
  formSnapshot,
  resultSnapshot,
  judgmentAssets,
}: {
  caseId: string;
  reviewId?: number;
  formSnapshot: Record<string, any>;
  resultSnapshot: Record<string, any>;
  judgmentAssets: JudgmentAssetCandidate[];
}) {
  const [session, setSession] = useState<FollowupSession | null>(null);
  const [answers, setAnswers] = useState<Record<string, { status?: AnswerStatus; note: string }>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!caseId) return;
    let active = true;
    apiClient.get("/api/shion-followups", { params: { case_id: caseId, limit: 1 } })
      .then((response) => {
        if (!active) return;
        const current = Array.isArray(response.data?.followups) ? response.data.followups[0] : null;
        if (!current) return;
        setSession(current);
        setAnswers(Object.fromEntries((current.answers || []).map((answer: FollowupAnswer) => [
          answer.question_id,
          { status: answer.status, note: answer.note || "" },
        ])));
      })
      .catch(() => undefined);
    return () => { active = false; };
  }, [caseId]);

  const allAnswered = useMemo(
    () => Boolean(session?.questions.length) && session!.questions.every((question) => Boolean(answers[question.id]?.status)),
    [answers, session],
  );

  const createQuestions = async () => {
    if (!caseId || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await apiClient.post("/api/shion-followups", {
        case_id: caseId,
        review_id: reviewId || null,
        form_snapshot: formSnapshot,
        result_snapshot: resultSnapshot,
        judgment_assets: judgmentAssets.slice(0, 3),
      });
      setSession(response.data?.followup || null);
      setAnswers({});
    } catch (requestError) {
      console.error("Shion follow-up question creation failed", requestError);
      setError("追加確認を作れませんでした。API接続を確認してください。");
    } finally {
      setLoading(false);
    }
  };

  const submitAnswers = async () => {
    if (!session || !allAnswered || loading) return;
    setLoading(true);
    setError("");
    try {
      const payload = session.questions.map((question) => ({
        question_id: question.id,
        status: answers[question.id]?.status,
        note: answers[question.id]?.note || "",
      }));
      const response = await apiClient.post(`/api/shion-followups/${session.followup_id}/answers`, { answers: payload });
      setSession(response.data?.followup || session);
    } catch (requestError) {
      console.error("Shion follow-up answer save failed", requestError);
      setError("回答を保存できませんでした。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50/70 p-3 sm:p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h4 className="flex items-center gap-2 text-xs font-black text-indigo-950">
            <HelpCircle className="h-4 w-4 text-indigo-600" />
            紫苑の追加確認モード
          </h4>
          <p className="mt-1 text-[11px] font-bold leading-5 text-indigo-700">
            不明・曖昧・影響が大きい点だけを最大3問に絞ります。回答でスコアは変えず、判断理由と承認条件を更新します。
          </p>
        </div>
        <button
          type="button"
          onClick={createQuestions}
          disabled={loading || !caseId}
          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-[11px] font-black text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          {session ? "質問を作り直す" : "重要な3問を出す"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-lg bg-rose-50 p-2 text-[11px] font-bold text-rose-700">{error}</p>}

      {session?.questions.length ? (
        <div className="mt-4 space-y-3">
          {session.questions.map((question, index) => (
            <div key={question.id} className="rounded-xl border border-indigo-100 bg-white p-3">
              <div className="flex items-start gap-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-[10px] font-black text-indigo-700">{index + 1}</span>
                <div>
                  <div className="text-[10px] font-black text-indigo-500">{question.category}</div>
                  <p className="mt-0.5 text-xs font-black leading-5 text-slate-800">{question.question}</p>
                  <p className="mt-1 text-[10px] font-medium leading-4 text-slate-500">理由: {question.reason}</p>
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {ANSWER_OPTIONS.map((option) => {
                  const selected = answers[question.id]?.status === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setAnswers((current) => ({
                        ...current,
                        [question.id]: { status: option.value, note: current[question.id]?.note || "" },
                      }))}
                      className={`rounded-lg border px-2.5 py-1.5 text-[10px] font-black ${selected ? option.tone : "border-slate-200 bg-white text-slate-500"}`}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
              <textarea
                value={answers[question.id]?.note || ""}
                onChange={(event) => setAnswers((current) => ({
                  ...current,
                  [question.id]: { status: current[question.id]?.status, note: event.target.value },
                }))}
                rows={2}
                maxLength={2000}
                placeholder="回答の根拠・営業確認内容（任意）"
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-medium text-slate-700 outline-none focus:border-indigo-300"
              />
            </div>
          ))}

          <button
            type="button"
            onClick={submitAnswers}
            disabled={!allAnswered || loading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-black text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className="h-3.5 w-3.5" />
            {loading ? "判断を更新中" : "回答を反映して判断を更新"}
          </button>
          {!allAnswered && <p className="text-center text-[10px] font-bold text-slate-400">すべての質問の状態を選ぶと反映できます</p>}
        </div>
      ) : (
        <p className="mt-3 rounded-lg border border-dashed border-indigo-200 bg-white/70 p-3 text-[11px] font-bold text-indigo-600">
          紫苑レビューのあとに「重要な3問を出す」を押してください。
        </p>
      )}

      {session?.summary?.updated_decision && (
        <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-3">
          <div className="flex items-center gap-2 text-xs font-black text-emerald-900">
            {session.summary.updated_decision === "追加確認を継続" ? <AlertTriangle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
            更新後: {session.summary.updated_decision}
          </div>
          <p className="mt-1 text-[11px] font-bold leading-5 text-emerald-800">
            更新前: {session.summary.baseline_decision || session.baseline_decision} ／ {session.summary.change_reason}
          </p>
          <div className="mt-2 rounded-lg bg-white p-2 text-[11px] font-medium leading-5 text-slate-700">
            {session.summary.ringi_comment}
          </div>
          {!!session.summary.approval_conditions?.length && (
            <ul className="mt-2 space-y-1 text-[10px] font-bold leading-4 text-slate-600">
              {session.summary.approval_conditions.map((condition) => <li key={condition}>・{condition}</li>)}
            </ul>
          )}
          <p className="mt-2 text-[10px] font-black text-emerald-700">
            結果登録時にこの確認内容を照合します。スコア変更: なし
            {session.outcome_status ? ` ／ 結果: ${session.outcome_status}` : ""}
          </p>
        </div>
      )}
    </div>
  );
}
