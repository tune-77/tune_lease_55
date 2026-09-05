"use client";
import { useState, type ReactNode } from "react";
import { Activity, Bot, Brain, ChevronDown, MessageSquare } from "lucide-react";
import {
  judgmentAssetHighlightTerms,
  normalizeReviewText,
  buildShionThoughtProcessSteps,
  SHION_REVIEW_IMAGE,
  type JudgmentAssetCandidate,
  type QRiskBreakdown,
  type ShionReviewFeedback,
  type ShionScreeningReview,
} from "../../lib/shionReview";
import ShionFollowUpPanel from "./ShionFollowUpPanel";
import QRiskBreakdownChart from "./QRiskBreakdownChart";

const renderPlainReviewTextWithHighlights = (
  text: string,
  candidates: JudgmentAssetCandidate[],
  keyPrefix: string,
) => {
  const assetTerms = judgmentAssetHighlightTerms(candidates);
  if (!assetTerms.length) return text;
  const pattern = new RegExp(`(${assetTerms.map((item) => item.term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
  const byTerm = new Map(assetTerms.map((item) => [item.term, item]));
  return text.split(pattern).map((part, index): ReactNode => {
    const asset = byTerm.get(part);
    if (!asset) {
      return <span key={`${keyPrefix}-plain-${index}`}>{part}</span>;
    }
    return (
      <span
        key={`${keyPrefix}-asset-${index}`}
        className={`mx-0.5 inline rounded-md border px-1.5 py-0.5 font-black ${
          asset.canonical
            ? "border-emerald-200 bg-emerald-100 text-emerald-950"
            : "border-amber-200 bg-amber-100 text-amber-950"
        }`}
      >
        <span className={`mr-1 rounded px-1 py-0.5 text-[10px] text-white ${asset.canonical ? "bg-emerald-600" : "bg-amber-500"}`}>
          {asset.canonical ? "正規判断資産" : "昇格候補"}
        </span>
        {part}
      </span>
    );
  });
};

const renderReviewTextWithHighlights = (
  text: string,
  candidates: JudgmentAssetCandidate[] = [],
) => {
  const parts = text.split(/(判断資産出典\s*[:：][^\n]*|JA-[A-Za-z0-9_-]{6,})/g).filter((part) => part !== "");
  if (!parts.length) return null;
  return parts.map((part, index): ReactNode => {
    if (/^判断資産出典\s*[:：]/.test(part)) {
      const isCanonical = part.includes("正規") || part.includes("JA-cr-");
      return (
        <span
          key={`asset-source-${index}`}
          className={`my-1 block rounded-xl border px-3 py-2 text-xs font-black leading-6 shadow-sm ${
            isCanonical
              ? "border-emerald-200 bg-emerald-100 text-emerald-950"
              : "border-amber-200 bg-amber-100 text-amber-950"
          }`}
        >
          <span className={`mr-2 rounded-full px-2 py-0.5 text-[10px] text-white ${isCanonical ? "bg-emerald-600" : "bg-amber-500"}`}>
            {isCanonical ? "正規判断資産" : "昇格候補"}
          </span>
          {part}
        </span>
      );
    }
    if (/^JA-[A-Za-z0-9_-]{6,}$/.test(part)) {
      const isCanonical = part.startsWith("JA-cr-");
      return (
        <span
          key={`asset-id-${index}`}
          className={`mx-0.5 inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-black leading-none ${
            isCanonical
              ? "border-emerald-300 bg-emerald-100 text-emerald-900"
              : "border-amber-300 bg-amber-100 text-amber-900"
          }`}
        >
          {part}
        </span>
      );
    }
    return (
      <span key={`review-text-${index}`}>
        {renderPlainReviewTextWithHighlights(part, candidates, `review-text-${index}`)}
      </span>
    );
  });
};

export function ShionScreeningReviewCard({
  review,
  loading,
  error,
  onReview,
  onFeedback,
  feedbackSaving,
  judgmentAssetCandidates,
  result,
  formData,
}: {
  review: ShionScreeningReview | null;
  loading: boolean;
  error: string;
  onReview: () => void;
  onFeedback: (feedback: ShionReviewFeedback) => void;
  feedbackSaving: boolean;
  judgmentAssetCandidates: JudgmentAssetCandidate[];
  result: Record<string, any> | null;
  formData: Record<string, any>;
}) {
  const feedbackOptions: { key: ShionReviewFeedback; label: string }[] = [
    { key: "specific", label: "具体的" },
    { key: "thin", label: "薄い" },
    { key: "discomfort_hit", label: "違和感○" },
    { key: "over_inferred", label: "推測強い" },
    { key: "useful", label: "使えた" },
    { key: "needs_fix", label: "修正" },
    { key: "wrong", label: "違った" },
  ];
  const [showThoughtProcess, setShowThoughtProcess] = useState(false);
  const thoughtProcessSteps = result
    ? buildShionThoughtProcessSteps(result, judgmentAssetCandidates, review)
    : [];
  // LLM 応答が取れず buildShionReviewFallback の定型文を表示している状態。
  // エラーは握り潰して本文を出しているため、紫苑が書いた文と区別できるようバッジで明示する。
  const isFallback = review?.vertexStatus === "fallback";

  return (
    <section className="overflow-hidden rounded-2xl border border-violet-200 bg-white shadow-sm">
      <div className="grid gap-0 lg:grid-cols-[150px_minmax(0,1fr)]">
        <div className="relative min-h-36 bg-violet-950">
          <img src={SHION_REVIEW_IMAGE} alt="審査レビュー中の紫苑" className="h-full w-full object-cover object-top opacity-95" />
          <div className="absolute inset-x-0 bottom-0 bg-violet-950/80 px-3 py-2 text-center text-[10px] font-black tracking-[0.25em] text-violet-100">
            SHION REVIEW
          </div>
        </div>
        <div className="p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h3 className="flex items-center gap-2 text-sm font-black text-violet-950">
                <Bot className="h-4 w-4 text-violet-600" />
                紫苑レビュー
              </h3>
              <p className="mt-1 text-xs font-bold leading-relaxed text-violet-700">
                点数の説明ではなく、違和感・承認条件・稟議に残す一文へ変換します。判断資産出典は色付きで表示します。
              </p>
            </div>
            <button
              type="button"
              onClick={onReview}
              disabled={loading}
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-xs font-black text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-violet-300"
            >
              {loading ? <Activity className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}
              {review ? "再レビュー" : "紫苑レビュー生成"}
            </button>
          </div>

          <div className="mt-4 rounded-xl border border-violet-100 bg-violet-50/70 p-4">
            {loading ? (
              <div className="flex min-h-28 items-center justify-center gap-2 text-sm font-black text-violet-700">
                <Activity className="h-5 w-5 animate-spin" />
                紫苑が審査結果を読み直しています
              </div>
            ) : error ? (
              <p className="text-sm font-bold leading-7 text-rose-700">{error}</p>
            ) : review ? (
              <>
                {isFallback && (
                  <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-[11px] font-bold leading-5 text-amber-900">
                    簡易生成: 紫苑からの応答が取得できなかったため、案件情報と判断資産から組み立てた定型の下書きを表示しています。「再レビュー」で生成し直せます。
                  </div>
                )}
                <div className="space-y-2 text-sm font-medium leading-7 text-slate-800">
                  {normalizeReviewText(review.reply).split(/\n{2,}/).map((block, index) => (
                    <p key={index} className="whitespace-pre-wrap">
                      {renderReviewTextWithHighlights(block, judgmentAssetCandidates)}
                    </p>
                  ))}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-black text-violet-700">
                  <span className="rounded-full bg-white px-2.5 py-1">記憶 {review.memoryRefs}件</span>
                  <span className="rounded-full bg-white px-2.5 py-1">知識 {review.knowledgeRefs}件</span>
                  <span className="rounded-full bg-white px-2.5 py-1">同一性 {review.identityUsed ? "ON" : "OFF"}</span>
                  <span className={`rounded-full px-2.5 py-1 ${review.vertexUsed ? "bg-teal-100 text-teal-700" : "bg-white"}`}>
                    Vertex {review.vertexUsed ? "ON" : review.vertexStatus || "OFF"}
                  </span>
                  <span className={`rounded-full px-2.5 py-1 ${review.vertexAnswerUsed ? "bg-cyan-100 text-cyan-800" : "bg-white"}`}>
                    Answer {review.vertexAnswerUsed ? "ON" : review.vertexAnswerStatus || "OFF"}
                    {typeof review.groundingScore === "number" ? ` / 根拠${Math.round(review.groundingScore * 100)}%` : ""}
                  </span>
                  {review.lowSupportClaimCount ? (
                    <span className="rounded-full bg-amber-100 px-2.5 py-1 text-amber-800">低根拠 {review.lowSupportClaimCount}件</span>
                  ) : null}
                  {review.savedId && <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-700">経験保存済 #{review.savedId}</span>}
                </div>
                {review.vertexRefs?.length ? (
                  <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-black text-teal-700">
                    {review.vertexRefs.slice(0, 3).map((ref, index) => (
                      <span key={`${ref}-${index}`} className="rounded-full bg-white px-2.5 py-1">
                        {ref.split("/").pop() || ref}
                      </span>
                    ))}
                  </div>
                ) : null}
                <QRiskBreakdownChart breakdown={(result?.q_risk_breakdown as QRiskBreakdown | undefined) ?? null} />
                <ShionFollowUpPanel
                  caseId={String(result?.case_id || formData.company_no || formData.company_name || "")}
                  reviewId={review.savedId}
                  formSnapshot={formData}
                  resultSnapshot={result || {}}
                  judgmentAssets={judgmentAssetCandidates}
                />
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-violet-100 pt-3">
                  <span className="text-[11px] font-black text-violet-500">紫苑レビュー評価</span>
                  {feedbackOptions.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      onClick={() => onFeedback(option.key)}
                      disabled={!review.savedId || feedbackSaving}
                      className={`rounded-lg border px-3 py-1.5 text-[11px] font-black transition disabled:cursor-not-allowed disabled:opacity-50 ${
                        review.userFeedback === option.key
                          ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                          : "border-violet-100 bg-white text-violet-700 hover:bg-violet-100"
                      }`}
                    >
                      {feedbackSaving && review.userFeedback === option.key ? "保存中" : option.label}
                    </button>
                  ))}
                  {!review.savedId && (
                    <span className="text-[11px] font-bold text-slate-400">レビュー保存後に評価できます</span>
                  )}
                </div>
                {thoughtProcessSteps.length > 0 && (
                  <div className="mt-3 border-t border-violet-100 pt-3">
                    <button
                      type="button"
                      onClick={() => setShowThoughtProcess((prev) => !prev)}
                      className="inline-flex items-center gap-1.5 text-[11px] font-black text-violet-600 hover:text-violet-800"
                    >
                      <Brain className="h-3.5 w-3.5" />
                      紫苑の思考プロセスを見る
                      <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showThoughtProcess ? "rotate-180" : ""}`} />
                    </button>
                    {showThoughtProcess && (
                      <ol className="mt-3 space-y-2">
                        {thoughtProcessSteps.map((step, index) => (
                          <li key={step.title} className="rounded-lg border border-violet-100 bg-white p-2.5">
                            <div className="flex items-center gap-2 text-[11px] font-black text-violet-700">
                              <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-violet-100 text-[10px] text-violet-700">
                                {index + 1}
                              </span>
                              {step.title}
                            </div>
                            <ul className="mt-1.5 space-y-1 pl-6 text-[11px] font-medium leading-relaxed text-slate-600">
                              {step.items.map((item, itemIndex) => (
                                <li key={itemIndex} className="list-disc">{item}</li>
                              ))}
                            </ul>
                          </li>
                        ))}
                      </ol>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="min-h-20 text-sm font-bold leading-7 text-violet-700">
                審査実行後に、紫苑がこの案件をレビューします。境界案件では、点数よりも「何を条件に残すか」を優先して見ます。
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
