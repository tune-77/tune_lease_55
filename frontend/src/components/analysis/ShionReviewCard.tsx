"use client";
import { useState, type MouseEvent as ReactMouseEvent, type ReactNode } from "react";
import { Activity, Bot, Brain, ChevronDown, Database, MessageSquare } from "lucide-react";
import {
  FEEDBACK_LABELS,
  judgmentAssetHighlightTerms,
  normalizeReviewText,
  validPastCompanyName,
  buildShionThoughtProcessSteps,
  SHION_REVIEW_IMAGE,
  type JudgmentAssetCandidate,
  type PastCompanyHighlight,
  type ShionReviewFeedback,
  type ShionScreeningReview,
} from "../../lib/shionReview";

const POPUP_WIDTH_PX = 320;
const POPUP_MAX_HEIGHT_PX = 340;

function PastCompanyPopupRow({ label, value }: { label: string; value?: string }) {
  if (!value) return null;
  return (
    <span className="block">
      <span className="mr-1.5 inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-black text-slate-500">{label}</span>
      <span className="text-[11px] font-bold leading-5 text-slate-700">{value}</span>
    </span>
  );
}

function PastCompanyHighlightBadge({ highlight }: { highlight: PastCompanyHighlight }) {
  const [popupPos, setPopupPos] = useState<{ left: number; top?: number; bottom?: number } | null>(null);
  const experienceCase = highlight.experienceCase;
  const pastReview = highlight.pastReview;
  const hasDetail = Boolean(experienceCase || pastReview);

  const handleMouseEnter = (event: ReactMouseEvent<HTMLSpanElement>) => {
    if (!hasDetail) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPUP_WIDTH_PX - 8));
    const showAbove = rect.bottom + POPUP_MAX_HEIGHT_PX + 12 > window.innerHeight && rect.top > POPUP_MAX_HEIGHT_PX + 12;
    setPopupPos(
      showAbove
        ? { left, bottom: window.innerHeight - rect.top + 6 }
        : { left, top: rect.bottom + 6 },
    );
  };

  return (
    <span
      className={`inline-flex items-center gap-1 rounded bg-cyan-50 px-1.5 py-0.5 font-black text-cyan-800 ring-1 ring-cyan-200 ${hasDetail ? "cursor-help" : ""}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setPopupPos(null)}
    >
      {highlight.name}
      <span className="rounded bg-white px-1 text-[10px] font-black text-cyan-600">
        {highlight.label}
      </span>
      {popupPos && hasDetail && (
        <span
          className="fixed z-[120] block overflow-y-auto rounded-xl border border-cyan-200 bg-white p-3 text-left font-medium shadow-xl"
          style={{ left: popupPos.left, top: popupPos.top, bottom: popupPos.bottom, width: POPUP_WIDTH_PX, maxHeight: POPUP_MAX_HEIGHT_PX }}
        >
          <span className="block border-b border-slate-100 pb-1.5 text-xs font-black text-cyan-900">
            {highlight.name}
            <span className="ml-1.5 rounded bg-cyan-50 px-1.5 py-0.5 text-[10px] text-cyan-600 ring-1 ring-cyan-200">{highlight.label}</span>
          </span>
          {experienceCase && (
            <span className="mt-2 block space-y-1.5">
              <PastCompanyPopupRow label="期間・業種" value={[experienceCase.period, experienceCase.industry].filter(Boolean).join(" / ")} />
              <PastCompanyPopupRow
                label="スコア・判断"
                value={[`${experienceCase.score.toFixed(1)}点`, experienceCase.decision, experienceCase.outcome].filter(Boolean).join(" / ")}
              />
              <PastCompanyPopupRow
                label="類似度"
                value={experienceCase.similarityScore
                  ? `${Math.round(experienceCase.similarityScore)}（${(experienceCase.similarityReasons || []).join("・") || "理由未計算"}）`
                  : ""}
              />
              <PastCompanyPopupRow label="似ている点" value={experienceCase.similarity} />
              <PastCompanyPopupRow label="当時の対応" value={experienceCase.actionTaken} />
              <PastCompanyPopupRow label="得た教訓" value={experienceCase.lesson} />
              <PastCompanyPopupRow label="今回との差分" value={experienceCase.difference} />
            </span>
          )}
          {pastReview && (
            <span className="mt-2 block space-y-1.5">
              <PastCompanyPopupRow label="業種" value={pastReview.industry_sub} />
              <PastCompanyPopupRow
                label="スコア・判定"
                value={[
                  pastReview.score != null ? `${Number(pastReview.score).toFixed(1)}点` : "",
                  pastReview.hantei || "",
                ].filter(Boolean).join(" / ")}
              />
              <PastCompanyPopupRow
                label="人間評価"
                value={pastReview.user_feedback ? FEEDBACK_LABELS[pastReview.user_feedback] : "未評価"}
              />
              <PastCompanyPopupRow
                label="過去レビュー"
                value={(() => {
                  const preview = normalizeReviewText(pastReview.review_text || "");
                  return preview ? `${preview.slice(0, 220)}${preview.length > 220 ? "…" : ""}` : "";
                })()}
              />
            </span>
          )}
        </span>
      )}
    </span>
  );
}

export function PastCompanyReferenceStrip({ companies }: { companies: PastCompanyHighlight[] }) {
  const visibleCompanies = Array.from(new Map(companies.map((item) => [item.name.trim(), item])).values())
    .filter((item) => validPastCompanyName(item.name))
    .slice(0, 3);
  if (!visibleCompanies.length) return null;
  return (
    <div className="mb-3 rounded-xl border border-cyan-200 bg-cyan-50/80 p-3">
      <div className="mb-2 flex items-center gap-2 text-[11px] font-black text-cyan-900">
        <Database className="h-3.5 w-3.5" />
        参照した過去取引事例
      </div>
      <div className="grid gap-2 md:grid-cols-3">
        {visibleCompanies.map((highlight) => {
          const caseDetail = highlight.experienceCase;
          const reviewDetail = highlight.pastReview;
          const score = caseDetail?.score ?? reviewDetail?.score;
          const decision = caseDetail?.decision || reviewDetail?.hantei || "";
          const lesson = caseDetail?.lesson || reviewDetail?.review_text || "";
          return (
            <div key={highlight.name} className="rounded-lg border border-cyan-100 bg-white p-2 shadow-sm">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs font-black text-cyan-950">{highlight.name}</span>
                <span className="rounded bg-cyan-100 px-1.5 py-0.5 text-[10px] font-black text-cyan-700">
                  {highlight.label}
                </span>
              </div>
              <p className="mt-1 text-[11px] font-bold leading-5 text-slate-700">
                {[caseDetail?.industry || reviewDetail?.industry_sub, score != null ? `${Number(score).toFixed(1)}点` : "", decision]
                  .filter(Boolean)
                  .join(" / ")}
              </p>
              {lesson && (
                <p className="mt-1 line-clamp-2 text-[11px] font-medium leading-5 text-slate-600">
                  {String(lesson)}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const highlightTextByCompanies = (text: string, companies: PastCompanyHighlight[]) => {
  const highlights = Array.from(new Map(companies.map((item) => [item.name.trim(), item])).values())
    .filter((item) => validPastCompanyName(item.name))
    .sort((a, b) => b.name.length - a.name.length);
  if (!highlights.length) return text;
  const names = highlights.map((item) => item.name);
  const highlightByName = new Map(highlights.map((item) => [item.name, item]));
  const pattern = new RegExp(`(${names.map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
  return text.split(pattern).map((part, index) => {
    const highlight = highlightByName.get(part);
    return highlight && names.includes(part) ? (
      <PastCompanyHighlightBadge key={`${part}-${index}`} highlight={highlight} />
    ) : part;
  });
};

const renderPlainReviewTextWithHighlights = (
  text: string,
  companies: PastCompanyHighlight[],
  candidates: JudgmentAssetCandidate[],
  keyPrefix: string,
) => {
  const assetTerms = judgmentAssetHighlightTerms(candidates);
  if (!assetTerms.length) return highlightTextByCompanies(text, companies);
  const pattern = new RegExp(`(${assetTerms.map((item) => item.term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
  const byTerm = new Map(assetTerms.map((item) => [item.term, item]));
  return text.split(pattern).map((part, index): ReactNode => {
    const asset = byTerm.get(part);
    if (!asset) {
      return <span key={`${keyPrefix}-plain-${index}`}>{highlightTextByCompanies(part, companies)}</span>;
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
  companies: PastCompanyHighlight[],
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
        {renderPlainReviewTextWithHighlights(part, companies, candidates, `review-text-${index}`)}
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
  pastCompanies,
  judgmentAssetCandidates,
  result,
}: {
  review: ShionScreeningReview | null;
  loading: boolean;
  error: string;
  onReview: () => void;
  onFeedback: (feedback: ShionReviewFeedback) => void;
  feedbackSaving: boolean;
  pastCompanies: PastCompanyHighlight[];
  judgmentAssetCandidates: JudgmentAssetCandidate[];
  result: Record<string, any> | null;
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
    ? buildShionThoughtProcessSteps(result, judgmentAssetCandidates, pastCompanies, review)
    : [];

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
                点数の説明ではなく、違和感・承認条件・稟議に残す一文へ変換します。過去案件名と判断資産出典は色付きで表示します。
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
                <PastCompanyReferenceStrip companies={pastCompanies} />
                <div className="space-y-2 text-sm font-medium leading-7 text-slate-800">
                  {normalizeReviewText(review.reply).split(/\n{2,}/).map((block, index) => (
                    <p key={index} className="whitespace-pre-wrap">
                      {renderReviewTextWithHighlights(block, pastCompanies, judgmentAssetCandidates)}
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
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-violet-100 pt-3">
                  <span className="text-[11px] font-black text-violet-500">人間評価</span>
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
                    <span className="text-[11px] font-bold text-slate-400">経験保存後に評価できます</span>
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
