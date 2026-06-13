"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
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
  const [filter, setFilter] = useState<ReviewStatus | "all">("candidate");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState<Record<number, boolean>>({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [candidateResponse, summaryResponse] = await Promise.all([
        apiClient.get<CandidateResponse>("/api/judgment-feedback/candidates"),
        apiClient.get<FeedbackSummary>("/api/judgment-feedback/summary"),
      ]);
      setItems(candidateResponse.data.items ?? []);
      setSummary(summaryResponse.data);
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
