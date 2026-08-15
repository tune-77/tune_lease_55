"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  RefreshCw,
  ShieldCheck,
  UserCheck,
  XCircle,
} from "lucide-react";

import { apiClient } from "@/lib/api";

type ReviewStatus = "candidate" | "approved" | "rejected";

type JudgmentCandidate = {
  id: number;
  case_id: string;
  recorded_at: string;
  source: string;
  model_decision: string;
  human_decision: string;
  target_label: number;
  reason: string;
  score: number | null;
  input_snapshot: Record<string, unknown>;
  evidence_snapshot: Record<string, unknown>;
  review_status: ReviewStatus;
};

type CandidateResponse = {
  items: JudgmentCandidate[];
  approved_only: boolean;
};

type FeedbackSummary = {
  total: number;
  candidates: number;
  approved: number;
};

type PredictionErrorDecision = "adopted" | "revised" | "held" | "rejected";

type PredictionErrorCandidate = {
  candidate_id: string;
  case_id: string;
  target_belief: string;
  error_type: string;
  severity: string;
  repeat_count: number;
  status: string;
  suggested_update: string;
  review_decision?: PredictionErrorDecision;
  review_note?: string;
  reviewed_at?: string;
  evidence: {
    main_concern?: string;
    recommended_action?: string;
    user_feedback?: string;
    feedback_note?: string;
  };
  next_action: {
    kind: string;
    requires_human_review: boolean;
    no_automatic_scoring_or_rag_change: boolean;
  };
  created_at: string;
};

type PredictionErrorSummary = {
  total_events: number;
  total_candidates: number;
  ready_for_review: number;
  review_counts: Record<string, number>;
  by_error_type: Record<string, number>;
  guardrail: string;
};

type PredictionErrorResponse = {
  summary: PredictionErrorSummary;
  candidates: PredictionErrorCandidate[];
};

type PromotionCandidate = {
  id: string;
  candidate_type: string;
  research_topic: string;
  claim: string;
  effective_claim: string;
  edited_claim?: string;
  evidence_path?: string;
  promotion_status: string;
  verified_status: string;
  use_count: number;
  useful_count: number;
  neutral_count: number;
  rejected_count: number;
  edit_count: number;
  last_feedback_at?: string;
  verification_note?: string;
  score: number;
  already_active_statement?: boolean;
};

type PromotionResponse = {
  count: number;
  active_count: number;
  promotion_policy: string;
  candidates: PromotionCandidate[];
};

const STATUS_STYLE: Record<ReviewStatus, string> = {
  candidate: "border-amber-200 bg-amber-50 text-amber-700",
  approved: "border-emerald-200 bg-emerald-50 text-emerald-700",
  rejected: "border-rose-200 bg-rose-50 text-rose-700",
};

const STATUS_LABEL: Record<ReviewStatus, string> = {
  candidate: "レビュー待ち",
  approved: "教材として承認",
  rejected: "却下",
};

const PREDICTION_DECISION_LABEL: Record<PredictionErrorDecision, string> = {
  adopted: "採用",
  revised: "修正して採用",
  held: "保留",
  rejected: "却下",
};

const PREDICTION_ERROR_LABEL: Record<string, string> = {
  risk_overestimated: "リスク過大評価",
  risk_underestimated: "リスク過小評価",
  condition_setting_miss: "条件設定ズレ",
  confirmation_gap: "確認不足",
  sales_context_miss: "営業/競合文脈ズレ",
  explanation_gap: "説明不足",
  general_prediction_error: "分類未確定",
};

const PREDICTION_STATUS_STYLE: Record<string, string> = {
  candidate: "border-slate-200 bg-slate-50 text-slate-700",
  ready_for_review: "border-blue-200 bg-blue-50 text-blue-700",
  adopted: "border-emerald-200 bg-emerald-50 text-emerald-700",
  revised: "border-cyan-200 bg-cyan-50 text-cyan-700",
  held: "border-amber-200 bg-amber-50 text-amber-700",
  rejected: "border-rose-200 bg-rose-50 text-rose-700",
};

const FIELD_LABELS: Record<string, string> = {
  industry_major: "業種大分類",
  industry_sub: "業種",
  grade: "格付",
  customer_type: "顧客区分",
  nenshu: "年商",
  acquisition_cost: "取得価額",
  lease_amount: "リース金額",
  lease_term: "リース期間",
  asset_name: "物件",
  score: "スコア",
};

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

export default function JudgmentReviewPage() {
  const [items, setItems] = useState<JudgmentCandidate[]>([]);
  const [summary, setSummary] = useState<FeedbackSummary | null>(null);
  const [predictionItems, setPredictionItems] = useState<PredictionErrorCandidate[]>([]);
  const [predictionSummary, setPredictionSummary] = useState<PredictionErrorSummary | null>(null);
  const [promotionItems, setPromotionItems] = useState<PromotionCandidate[]>([]);
  const [promotionSummary, setPromotionSummary] = useState<PromotionResponse | null>(null);
  const [filter, setFilter] = useState<ReviewStatus | "all">("candidate");
  const [predictionFilter, setPredictionFilter] = useState<string>("ready_for_review");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState<Record<number, boolean>>({});
  const [predictionActionLoading, setPredictionActionLoading] = useState<Record<string, boolean>>({});
  const [promotionActionLoading, setPromotionActionLoading] = useState<Record<string, boolean>>({});
  const [promotionNotes, setPromotionNotes] = useState<Record<string, string>>({});
  const [predictionNotes, setPredictionNotes] = useState<Record<string, string>>({});
  const [predictionEdits, setPredictionEdits] = useState<Record<string, string>>({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [candidateResponse, summaryResponse, predictionResponse, promotionResponse] = await Promise.all([
        apiClient.get<CandidateResponse>("/api/judgment-feedback/candidates"),
        apiClient.get<FeedbackSummary>("/api/judgment-feedback/summary"),
        apiClient.get<PredictionErrorResponse>(
          "/api/judgment-assets/prediction-errors/candidates",
          { params: { limit: 100 } },
        ),
        apiClient.get<PromotionResponse>("/api/judgment-assets/promotion-candidates", {
          params: { limit: 30 },
        }),
      ]);
      setItems(candidateResponse.data.items ?? []);
      setSummary(summaryResponse.data);
      setPredictionItems(predictionResponse.data.candidates ?? []);
      setPredictionSummary(predictionResponse.data.summary ?? null);
      setPromotionItems(promotionResponse.data.candidates ?? []);
      setPromotionSummary(promotionResponse.data ?? null);
    } catch {
      setError("判断差分候補を読み込めませんでした。APIの状態を確認してください。");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const visibleItems = useMemo(
    () =>
      items.filter(
        (item) => filter === "all" || item.review_status === filter,
      ),
    [filter, items],
  );

  const rejectedCount = items.filter(
    (item) => item.review_status === "rejected",
  ).length;

  const visiblePredictionItems = useMemo(
    () =>
      predictionItems.filter((item) => (
        predictionFilter === "all" || item.status === predictionFilter
      )),
    [predictionFilter, predictionItems],
  );

  const promotionReadyCount = promotionItems.filter(
    (item) => item.promotion_status !== "held",
  ).length;

  const handleReview = useCallback(
    async (item: JudgmentCandidate, reviewStatus: "approved" | "rejected") => {
      setActionLoading((current) => ({ ...current, [item.id]: true }));
      setError("");
      try {
        await apiClient.post(`/api/judgment-feedback/${item.id}/review`, {
          review_status: reviewStatus,
        });
        setItems((current) =>
          current.map((candidate) =>
            candidate.id === item.id
              ? { ...candidate, review_status: reviewStatus }
              : candidate,
          ),
        );
        const summaryResponse = await apiClient.get<FeedbackSummary>(
          "/api/judgment-feedback/summary",
        );
        setSummary(summaryResponse.data);
      } catch {
        setError("レビュー結果を保存できませんでした。もう一度お試しください。");
      } finally {
        setActionLoading((current) => ({ ...current, [item.id]: false }));
      }
    },
    [],
  );

  const handlePredictionReview = useCallback(
    async (item: PredictionErrorCandidate, decision: PredictionErrorDecision) => {
      setPredictionActionLoading((current) => ({ ...current, [item.candidate_id]: true }));
      setError("");
      try {
        await apiClient.post(
          `/api/judgment-assets/prediction-errors/candidates/${encodeURIComponent(item.candidate_id)}/review`,
          {
            decision,
            note: predictionNotes[item.candidate_id] || "",
            edited_update: predictionEdits[item.candidate_id] || "",
          },
        );
        setPredictionItems((current) =>
          current.map((candidate) =>
            candidate.candidate_id === item.candidate_id
              ? {
                  ...candidate,
                  status: decision,
                  review_decision: decision,
                  review_note: predictionNotes[item.candidate_id] || "",
                  suggested_update: predictionEdits[item.candidate_id] || candidate.suggested_update,
                }
              : candidate,
          ),
        );
        const response = await apiClient.get<PredictionErrorResponse>(
          "/api/judgment-assets/prediction-errors/candidates",
          { params: { limit: 100 } },
        );
        setPredictionSummary(response.data.summary ?? null);
      } catch {
        setError("予測誤差レビューを保存できませんでした。もう一度お試しください。");
      } finally {
        setPredictionActionLoading((current) => ({ ...current, [item.candidate_id]: false }));
      }
    },
    [predictionEdits, predictionNotes],
  );

  const refreshPromotionCandidates = useCallback(async () => {
    const response = await apiClient.get<PromotionResponse>("/api/judgment-assets/promotion-candidates", {
      params: { limit: 30 },
    });
    setPromotionItems(response.data.candidates ?? []);
    setPromotionSummary(response.data ?? null);
  }, []);

  const handlePromotionAction = useCallback(
    async (item: PromotionCandidate, action: "promote" | "hold" | "reject") => {
      setPromotionActionLoading((current) => ({ ...current, [item.id]: true }));
      setError("");
      try {
        if (action === "promote") {
          await apiClient.post(
            `/api/judgment-assets/promotion-candidates/${encodeURIComponent(item.id)}/promote`,
          );
        } else {
          await apiClient.post(
            `/api/judgment-assets/promotion-candidates/${encodeURIComponent(item.id)}/review`,
            {
              action,
              comment: promotionNotes[item.id] || "",
            },
          );
        }
        await refreshPromotionCandidates();
      } catch {
        setError("判断資産候補のレビュー結果を保存できませんでした。もう一度お試しください。");
      } finally {
        setPromotionActionLoading((current) => ({ ...current, [item.id]: false }));
      }
    },
    [promotionNotes, refreshPromotionCandidates],
  );

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-6xl space-y-5">
        <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white">
                <UserCheck className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-black text-slate-900">
                  実案件差分レビュー
                </h1>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  紫苑の判断と担当者判断の差を確認し、理由が妥当な事例だけを教材として承認します。
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={fetchData}
              disabled={loading}
              className="ml-auto inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              更新
            </button>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <SummaryCard
              label="レビュー待ち"
              value={summary?.candidates ?? 0}
              color="amber"
            />
            <SummaryCard
              label="承認済み教材"
              value={summary?.approved ?? 0}
              color="emerald"
            />
            <SummaryCard label="却下" value={rejectedCount} color="rose" />
          </div>
        </header>

        <section className="rounded-2xl border border-blue-200 bg-blue-50 p-4">
          <div className="flex gap-3">
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-blue-700" />
            <div>
              <p className="text-sm font-black text-blue-900">承認時の扱い</p>
              <p className="mt-1 text-sm leading-6 text-blue-800">
                承認した事例は紫苑の審査回答に「確認観点」として反映されます。
                自動承認・自動否決のルールにはせず、個別案件の事実確認を優先します。
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-amber-500 text-white">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-black text-slate-900">
                  判断資産 Promotion 候補
                </h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  紫苑レビューで人間が「効いた」「修正」と返した判断だけを、正規判断資産へ昇格する前に確認します。
                </p>
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <SummaryCard
              label="正規判断資産"
              value={promotionSummary?.active_count ?? 0}
              color="emerald"
            />
            <SummaryCard
              label="昇格候補"
              value={promotionSummary?.count ?? promotionItems.length}
              color="amber"
            />
            <SummaryCard
              label="今すぐ確認"
              value={promotionReadyCount}
              color="rose"
            />
          </div>

          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">
              安全線
            </p>
            <p className="mt-1 text-sm leading-6 text-slate-700">
              昇格しても自動承認・自動否決には接続しません。紫苑の回答時に参照する「確認観点」として扱います。
            </p>
          </div>

          {loading ? null : promotionItems.length === 0 ? (
            <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center">
              <CheckCircle2 className="mx-auto h-8 w-8 text-emerald-500" />
              <p className="mt-3 font-black text-slate-800">
                昇格候補はありません
              </p>
              <p className="mt-1 text-sm text-slate-500">
                審査分析で判断資産候補に評価が返ると、ここに表示されます。
              </p>
            </div>
          ) : (
            <div className="mt-5 space-y-4">
              {promotionItems.map((item) => {
                const displayClaim = item.edited_claim || item.effective_claim || item.claim;
                const isBusy = !!promotionActionLoading[item.id];
                return (
                  <article
                    key={item.id}
                    className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-700">
                        JA-{item.id.slice(0, 8)}
                      </span>
                      <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-xs font-black text-amber-700">
                        {item.candidate_type}
                      </span>
                      <span className="rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-black text-blue-700">
                        score {item.score}
                      </span>
                      {item.already_active_statement && (
                        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-black text-emerald-700">
                          既存資産と同文
                        </span>
                      )}
                    </div>

                    <p className="mt-4 whitespace-pre-wrap text-sm font-bold leading-7 text-slate-900">
                      {displayClaim}
                    </p>

                    <div className="mt-4 grid gap-3 md:grid-cols-2">
                      <InfoBlock label="研究トピック" value={item.research_topic || "-"} />
                      <InfoBlock label="状態" value={`${item.promotion_status} / ${item.verified_status}`} />
                      <InfoBlock label="利用実績" value={`使用 ${item.use_count} / 効いた ${item.useful_count} / 修正 ${item.edit_count} / 違う ${item.rejected_count}`} />
                      <InfoBlock label="出典" value={item.evidence_path || "-"} />
                    </div>

                    {item.verification_note && (
                      <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-4">
                        <p className="text-xs font-black uppercase tracking-wide text-blue-700">
                          検証メモ
                        </p>
                        <p className="mt-2 whitespace-pre-wrap text-sm font-bold leading-6 text-blue-900">
                          {item.verification_note}
                        </p>
                      </div>
                    )}

                    <label className="mt-4 block">
                      <span className="text-xs font-black text-slate-500">レビュー メモ</span>
                      <input
                        value={promotionNotes[item.id] || ""}
                        onChange={(event) =>
                          setPromotionNotes((current) => ({
                            ...current,
                            [item.id]: event.target.value,
                          }))
                        }
                        className="mt-2 w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium text-slate-800 outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
                        placeholder="昇格・保留・却下の理由"
                      />
                    </label>

                    <div className="mt-5 flex flex-col gap-2 border-t border-slate-200 pt-4 sm:flex-row sm:justify-end">
                      <button
                        type="button"
                        onClick={() => handlePromotionAction(item, "reject")}
                        disabled={isBusy}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-rose-200 bg-white px-4 py-2.5 text-sm font-black text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                      >
                        <XCircle className="h-4 w-4" />
                        却下
                      </button>
                      <button
                        type="button"
                        onClick={() => handlePromotionAction(item, "hold")}
                        disabled={isBusy}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-amber-200 bg-white px-4 py-2.5 text-sm font-black text-amber-700 hover:bg-amber-50 disabled:opacity-50"
                      >
                        <AlertCircle className="h-4 w-4" />
                        保留
                      </button>
                      <button
                        type="button"
                        onClick={() => handlePromotionAction(item, "promote")}
                        disabled={isBusy}
                        className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-black text-white hover:bg-emerald-700 disabled:opacity-50"
                      >
                        <CheckCircle2 className="h-4 w-4" />
                        正規資産へ昇格
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white">
                <ClipboardCheck className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-black text-slate-900">
                  予測誤差レビュー
                </h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  結果登録から作られた「外れ方」を確認し、次回の質問・条件・説明に変換する候補だけを残します。
                </p>
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <SummaryCard
              label="予測誤差イベント"
              value={predictionSummary?.total_events ?? 0}
              color="amber"
            />
            <SummaryCard
              label="レビュー可能"
              value={predictionSummary?.ready_for_review ?? 0}
              color="emerald"
            />
            <SummaryCard
              label="却下済み"
              value={predictionSummary?.review_counts?.rejected ?? 0}
              color="rose"
            />
          </div>

          <div className="mt-5 flex flex-wrap gap-2">
            {(
              [
                ["ready_for_review", "レビュー可能"],
                ["candidate", "蓄積中"],
                ["adopted", "採用"],
                ["revised", "修正採用"],
                ["held", "保留"],
                ["rejected", "却下"],
                ["all", "すべて"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setPredictionFilter(value)}
                className={`rounded-full border px-4 py-2 text-sm font-bold transition-colors ${
                  predictionFilter === value
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-black uppercase tracking-wide text-slate-500">
              安全線
            </p>
            <p className="mt-1 text-sm leading-6 text-slate-700">
              採用してもスコアリング、RAG順位、プロンプト、正規判断資産は自動変更しません。
              ここでは「次に見直すべき信念候補」を選別します。
            </p>
          </div>

          {loading ? null : visiblePredictionItems.length === 0 ? (
            <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center">
              <CheckCircle2 className="mx-auto h-8 w-8 text-emerald-500" />
              <p className="mt-3 font-black text-slate-800">
                該当する予測誤差候補はありません
              </p>
            </div>
          ) : (
            <div className="mt-5 space-y-4">
              {visiblePredictionItems.map((item) => {
                const edited = predictionEdits[item.candidate_id] ?? item.suggested_update;
                return (
                  <article
                    key={item.candidate_id}
                    className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full border px-2.5 py-1 text-xs font-black ${
                          PREDICTION_STATUS_STYLE[item.status] || PREDICTION_STATUS_STYLE.candidate
                        }`}
                      >
                        {PREDICTION_DECISION_LABEL[item.status as PredictionErrorDecision] || (item.status === "ready_for_review" ? "レビュー可能" : "蓄積中")}
                      </span>
                      <span className="text-xs font-bold text-slate-400">
                        {item.candidate_id}
                      </span>
                      <span className="text-xs font-bold text-slate-500">
                        repeat {item.repeat_count}
                      </span>
                      <span className="text-xs font-bold text-slate-500">
                        {PREDICTION_ERROR_LABEL[item.error_type] || item.error_type}
                      </span>
                    </div>

                    <div className="mt-4 grid gap-3 md:grid-cols-2">
                      <InfoBlock label="更新対象の信念" value={item.target_belief} />
                      <InfoBlock label="次アクション" value={item.next_action?.kind || "-"} />
                      <InfoBlock label="当時の主懸念" value={item.evidence?.main_concern || "-"} />
                      <InfoBlock label="当時の推奨" value={item.evidence?.recommended_action || "-"} />
                    </div>

                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500">
                        観測されたズレ
                      </p>
                      <p className="mt-2 whitespace-pre-wrap text-sm font-medium leading-6 text-slate-800">
                        {item.evidence?.feedback_note || item.evidence?.user_feedback || "-"}
                      </p>
                    </div>

                    <label className="mt-4 block">
                      <span className="text-xs font-black text-slate-500">修正後の更新候補</span>
                      <textarea
                        value={edited}
                        onChange={(event) =>
                          setPredictionEdits((current) => ({
                            ...current,
                            [item.candidate_id]: event.target.value,
                          }))
                        }
                        className="mt-2 min-h-24 w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium leading-6 text-slate-800 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                      />
                    </label>

                    <label className="mt-3 block">
                      <span className="text-xs font-black text-slate-500">レビュー メモ</span>
                      <input
                        value={predictionNotes[item.candidate_id] || ""}
                        onChange={(event) =>
                          setPredictionNotes((current) => ({
                            ...current,
                            [item.candidate_id]: event.target.value,
                          }))
                        }
                        className="mt-2 w-full rounded-xl border border-slate-300 bg-white p-3 text-sm font-medium text-slate-800 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
                        placeholder="採用・保留・却下の理由"
                      />
                    </label>

                    {!item.review_decision && item.status !== "adopted" && item.status !== "revised" && item.status !== "held" && item.status !== "rejected" && (
                      <div className="mt-5 flex flex-col gap-2 border-t border-slate-200 pt-4 sm:flex-row sm:justify-end">
                        {(
                          [
                            ["rejected", "却下"],
                            ["held", "保留"],
                            ["revised", "修正して採用"],
                            ["adopted", "採用"],
                          ] as const
                        ).map(([decision, label]) => (
                          <button
                            key={decision}
                            type="button"
                            onClick={() => handlePredictionReview(item, decision)}
                            disabled={predictionActionLoading[item.candidate_id]}
                            className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-black disabled:opacity-50 ${
                              decision === "rejected"
                                ? "border border-rose-200 bg-white text-rose-700 hover:bg-rose-50"
                                : decision === "held"
                                  ? "border border-amber-200 bg-white text-amber-700 hover:bg-amber-50"
                                  : "bg-blue-600 text-white hover:bg-blue-700"
                            }`}
                          >
                            {decision === "rejected" ? <XCircle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                            {label}
                          </button>
                        ))}
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <div className="flex flex-wrap gap-2">
          {(
            [
              ["candidate", "レビュー待ち"],
              ["approved", "承認済み"],
              ["rejected", "却下"],
              ["all", "すべて"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value)}
              className={`rounded-full border px-4 py-2 text-sm font-bold transition-colors ${
                filter === value
                  ? "border-slate-900 bg-slate-900 text-white"
                  : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        )}

        {loading ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-sm font-bold text-slate-500">
            判断差分を読み込み中...
          </div>
        ) : visibleItems.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center">
            <CheckCircle2 className="mx-auto h-8 w-8 text-emerald-500" />
            <p className="mt-3 font-black text-slate-800">
              {filter === "candidate"
                ? "現在、レビュー待ちの判断差分はありません"
                : "該当する判断差分はありません"}
            </p>
            <p className="mt-1 text-sm text-slate-500">
              討論審査または軍師AIで判断変更を記録すると、ここに表示されます。
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {visibleItems.map((item) => (
              <article
                key={item.id}
                className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
              >
                <div className="flex flex-col gap-3 md:flex-row md:items-start">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full border px-2.5 py-1 text-xs font-black ${STATUS_STYLE[item.review_status]}`}
                      >
                        {STATUS_LABEL[item.review_status]}
                      </span>
                      <span className="text-xs font-bold text-slate-400">
                        #{item.id} / {item.source}
                      </span>
                      {item.score !== null && (
                        <span className="text-xs font-bold text-slate-500">
                          スコア {item.score.toFixed(1)}
                        </span>
                      )}
                    </div>

                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <Decision label="紫苑" value={item.model_decision} />
                      <ArrowRight className="h-5 w-5 text-slate-400" />
                      <Decision label="担当者" value={item.human_decision} human />
                    </div>

                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                      <p className="text-xs font-black uppercase tracking-wide text-slate-500">
                        変更理由
                      </p>
                      <p className="mt-2 whitespace-pre-wrap text-sm font-medium leading-6 text-slate-800">
                        {item.reason}
                      </p>
                    </div>
                  </div>
                </div>

                <Snapshot
                  title="匿名化された案件情報"
                  values={item.input_snapshot}
                />
                <Snapshot
                  title="判断根拠の記録"
                  values={item.evidence_snapshot}
                  collapsed
                />

                {item.review_status === "candidate" && (
                  <div className="mt-5 flex flex-col gap-2 border-t border-slate-200 pt-4 sm:flex-row sm:justify-end">
                    <button
                      type="button"
                      onClick={() => handleReview(item, "rejected")}
                      disabled={actionLoading[item.id]}
                      className="inline-flex items-center justify-center gap-2 rounded-lg border border-rose-200 bg-white px-4 py-2.5 text-sm font-black text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                    >
                      <XCircle className="h-4 w-4" />
                      教材にしない
                    </button>
                    <button
                      type="button"
                      onClick={() => handleReview(item, "approved")}
                      disabled={actionLoading[item.id]}
                      className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-black text-white hover:bg-emerald-700 disabled:opacity-50"
                    >
                      <CheckCircle2 className="h-4 w-4" />
                      教材として承認
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

function InfoBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
      <p className="text-[11px] font-black text-slate-400">{label}</p>
      <p className="mt-1 whitespace-pre-wrap break-words text-sm font-bold text-slate-700">
        {value || "-"}
      </p>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "amber" | "emerald" | "rose";
}) {
  const styles = {
    amber: "border-amber-200 bg-amber-50 text-amber-800",
    emerald: "border-emerald-200 bg-emerald-50 text-emerald-800",
    rose: "border-rose-200 bg-rose-50 text-rose-800",
  };
  return (
    <div className={`rounded-xl border p-4 ${styles[color]}`}>
      <p className="text-xs font-black">{label}</p>
      <p className="mt-1 text-2xl font-black">{value}</p>
    </div>
  );
}

function Decision({
  label,
  value,
  human = false,
}: {
  label: string;
  value: string;
  human?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border px-4 py-3 ${
        human
          ? "border-emerald-200 bg-emerald-50"
          : "border-violet-200 bg-violet-50"
      }`}
    >
      <p className="text-xs font-black text-slate-500">{label}の判断</p>
      <p className="mt-1 text-lg font-black text-slate-900">{value}</p>
    </div>
  );
}

function Snapshot({
  title,
  values,
  collapsed = false,
}: {
  title: string;
  values: Record<string, unknown>;
  collapsed?: boolean;
}) {
  const entries = Object.entries(values || {}).filter(
    ([, value]) => value !== null && value !== undefined && value !== "",
  );
  if (entries.length === 0) return null;

  const content = (
    <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="rounded-lg bg-slate-50 px-3 py-2">
          <p className="text-[11px] font-black text-slate-400">
            {FIELD_LABELS[key] || key}
          </p>
          <p className="mt-1 whitespace-pre-wrap break-words text-sm font-bold text-slate-700">
            {formatValue(value)}
          </p>
        </div>
      ))}
    </div>
  );

  if (collapsed) {
    return (
      <details className="mt-4 rounded-xl border border-slate-200 p-4">
        <summary className="cursor-pointer text-sm font-black text-slate-700">
          {title}
        </summary>
        {content}
      </details>
    );
  }

  return (
    <section className="mt-4">
      <h2 className="text-sm font-black text-slate-700">{title}</h2>
      {content}
    </section>
  );
}
