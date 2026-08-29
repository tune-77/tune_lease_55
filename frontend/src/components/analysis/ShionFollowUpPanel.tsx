"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, BarChart3, CheckCircle2, HelpCircle, RefreshCw, Send } from "lucide-react";
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

type ImpactLabel = "decision_changed" | "risk_prevented" | "outcome_matched" | "evidence_strengthened" | "not_helpful";

type ImpactFeedback = {
  question_id: string;
  impact_label: ImpactLabel;
  impact_label_text?: string;
  note?: string;
};

type ImpactQuestionAnalytics = {
  question_id: string;
  category: string;
  question: string;
  asked_count: number;
  labeled_count: number;
  direct_impact_count: number;
  decision_changed_count: number;
  risk_prevented_count: number;
  warning_match_count: number;
  usefulness_rate: number | null;
  evidence_level: string;
};

type ImpactAnalytics = {
  session_count: number;
  outcome_linked_session_count: number;
  feedback_count: number;
  minimum_comparable_feedback: number;
  questions: ImpactQuestionAnalytics[];
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
  impact_feedback?: ImpactFeedback[];
};

const ANSWER_OPTIONS: { value: AnswerStatus; label: string; tone: string }[] = [
  { value: "confirmed", label: "確認できた", tone: "border-emerald-300 bg-emerald-50 text-emerald-800" },
  { value: "partial", label: "一部確認", tone: "border-amber-300 bg-amber-50 text-amber-800" },
  { value: "concern", label: "未確認・懸念あり", tone: "border-rose-300 bg-rose-50 text-rose-800" },
];

const IMPACT_OPTIONS: { value: ImpactLabel; label: string; selectedTone: string }[] = [
  { value: "decision_changed", label: "判断・条件を変えた", selectedTone: "border-indigo-400 bg-indigo-100 text-indigo-800" },
  { value: "risk_prevented", label: "事故・見落とし防止", selectedTone: "border-rose-400 bg-rose-100 text-rose-800" },
  { value: "outcome_matched", label: "懸念が結果に表れた", selectedTone: "border-amber-400 bg-amber-100 text-amber-800" },
  { value: "evidence_strengthened", label: "根拠を補強", selectedTone: "border-emerald-400 bg-emerald-100 text-emerald-800" },
  { value: "not_helpful", label: "役立たなかった", selectedTone: "border-slate-400 bg-slate-200 text-slate-700" },
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
  const [analytics, setAnalytics] = useState<ImpactAnalytics | null>(null);
  const [impactSaving, setImpactSaving] = useState("");
  const [impactError, setImpactError] = useState("");
  const locked = Boolean(session?.status.startsWith("outcome_linked"));

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

  useEffect(() => {
    let active = true;
    apiClient.get("/api/shion-followups-analytics", { params: { limit: 10 } })
      .then((response) => {
        if (active) setAnalytics(response.data || null);
      })
      .catch(() => undefined);
    return () => { active = false; };
  }, [session?.impact_feedback]);

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

  const saveImpact = async (questionId: string, impactLabel: ImpactLabel) => {
    if (!session || !locked || impactSaving) return;
    setImpactSaving(questionId);
    setImpactError("");
    try {
      const response = await apiClient.post(`/api/shion-followups/${session.followup_id}/impact-feedback`, {
        entries: [{ question_id: questionId, impact_label: impactLabel, note: "" }],
      });
      const saved = Array.isArray(response.data?.impact_feedback) ? response.data.impact_feedback[0] : null;
      if (saved) {
        setSession((current) => current ? {
          ...current,
          impact_feedback: [
            ...(current.impact_feedback || []).filter((entry) => entry.question_id !== questionId),
            saved,
          ],
        } : current);
      }
    } catch (requestError) {
      console.error("Shion follow-up impact feedback save failed", requestError);
      setImpactError("質問の効果を保存できませんでした。");
    } finally {
      setImpactSaving("");
    }
  };

  const analyticsRows = (analytics?.questions || []).filter(
    (question) => question.labeled_count > 0 || question.warning_match_count > 0,
  ).slice(0, 3);

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
          disabled={loading || !caseId || locked}
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
                      disabled={locked}
                      onClick={() => setAnswers((current) => ({
                        ...current,
                        [question.id]: { status: option.value, note: current[question.id]?.note || "" },
                      }))}
                      className={`rounded-lg border px-2.5 py-1.5 text-[10px] font-black disabled:cursor-not-allowed disabled:opacity-60 ${selected ? option.tone : "border-slate-200 bg-white text-slate-500"}`}
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
                disabled={locked}
                placeholder="回答の根拠・営業確認内容（任意）"
                className="mt-2 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-medium text-slate-700 outline-none focus:border-indigo-300 disabled:cursor-not-allowed disabled:opacity-60"
              />
            </div>
          ))}

          <button
            type="button"
            onClick={submitAnswers}
            disabled={!allAnswered || loading || locked}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-black text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className="h-3.5 w-3.5" />
            {loading ? "判断を更新中" : "回答を反映して判断を更新"}
          </button>
          {!allAnswered && <p className="text-center text-[10px] font-bold text-slate-400">すべての質問の状態を選ぶと反映できます</p>}
          {locked && (
            <p className="rounded-lg border border-slate-200 bg-slate-100 p-2 text-center text-[10px] font-black text-slate-600">
              結果登録済みのため、この確認セッションは編集できません。
              {session.status === "outcome_linked_unanswered" ? " 未回答のまま終了した記録として保持します。" : ""}
            </p>
          )}

          {locked && session.answers.length > 0 && (
            <div className="rounded-xl border border-violet-200 bg-violet-50 p-3">
              <div className="text-[11px] font-black text-violet-900">結果を見て、この質問は何に役立ちましたか？</div>
              <p className="mt-1 text-[10px] font-bold leading-4 text-violet-700">
                人間の評価だけを蓄積します。審査スコアや判断資産は自動変更しません。
              </p>
              <div className="mt-3 space-y-3">
                {session.questions.map((question) => {
                  const selected = session.impact_feedback?.find((entry) => entry.question_id === question.id)?.impact_label;
                  return (
                    <div key={`impact-${question.id}`} className="rounded-lg border border-violet-100 bg-white p-2.5">
                      <p className="text-[10px] font-black leading-4 text-slate-700">{question.category}: {question.question}</p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {IMPACT_OPTIONS.map((option) => (
                          <button
                            key={option.value}
                            type="button"
                            disabled={Boolean(impactSaving)}
                            onClick={() => saveImpact(question.id, option.value)}
                            className={`rounded-lg border px-2 py-1.5 text-[9px] font-black disabled:opacity-50 ${selected === option.value ? option.selectedTone : "border-slate-200 bg-white text-slate-500"}`}
                          >
                            {impactSaving === question.id ? "保存中" : option.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
              {impactError && <p className="mt-2 text-[10px] font-black text-rose-700">{impactError}</p>}
            </div>
          )}
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

      {analytics && (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-3">
          <div className="flex items-center gap-2 text-xs font-black text-slate-900">
            <BarChart3 className="h-4 w-4 text-indigo-600" />
            追加確認の効果分析
          </div>
          <p className="mt-1 text-[10px] font-bold leading-4 text-slate-500">
            結果連携 {analytics.outcome_linked_session_count}件 ／ 人間評価 {analytics.feedback_count}件。
            5評価未満は暫定で、順位を確定しません。
          </p>
          {analyticsRows.length ? (
            <div className="mt-3 space-y-2">
              {analyticsRows.map((question) => (
                <div key={`analytics-${question.question_id}`} className="rounded-lg bg-slate-50 p-2.5">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-[10px] font-black leading-4 text-slate-700">{question.category}: {question.question}</p>
                    <span className="shrink-0 rounded-full bg-white px-2 py-1 text-[9px] font-black text-slate-500">{question.evidence_level}</span>
                  </div>
                  <p className="mt-1 text-[9px] font-bold text-slate-500">
                    判断・条件変更 {question.decision_changed_count} ／ 事故・見落とし防止 {question.risk_prevented_count} ／ 懸念と結果一致 {question.warning_match_count}
                    {question.usefulness_rate !== null ? ` ／ 有用評価 ${Math.round(question.usefulness_rate * 100)}%` : ""}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 rounded-lg border border-dashed border-slate-200 p-3 text-[10px] font-bold text-slate-500">
              結果登録後に質問の効果を選ぶと、ここへ質問別の実績が蓄積されます。
            </p>
          )}
        </div>
      )}
    </div>
  );
}
