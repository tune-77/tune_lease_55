"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  FileSearch,
  PauseCircle,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";

import { apiClient } from "@/lib/api";

type ReturnStatus = "candidate" | "approved" | "held" | "rejected";
type ReturnKind = "score_input" | "ocr_result" | "shion_review" | "judgment_asset";
type FilterStatus = ReturnStatus | "all";
type FilterKind = ReturnKind | "all";

type ReturnItem = {
  id: number;
  kind: ReturnKind;
  title: string;
  source_id: string;
  created_at: string;
  review_status: ReturnStatus;
  review_note: string;
  reviewed_at: string;
  preview: string;
};

type ReturnResponse = {
  db_path: string;
  items: ReturnItem[];
  summary: Record<ReturnStatus | "total", number>;
};

const STATUS_LABEL: Record<ReturnStatus, string> = {
  candidate: "検疫待ち",
  approved: "承認済み",
  held: "保留",
  rejected: "破棄",
};

const STATUS_STYLE: Record<ReturnStatus, string> = {
  candidate: "border-amber-200 bg-amber-50 text-amber-800",
  approved: "border-emerald-200 bg-emerald-50 text-emerald-800",
  held: "border-sky-200 bg-sky-50 text-sky-800",
  rejected: "border-rose-200 bg-rose-50 text-rose-800",
};

const KIND_LABEL: Record<ReturnKind, string> = {
  score_input: "審査入力",
  ocr_result: "決算書OCR",
  shion_review: "紫苑レビュー",
  judgment_asset: "判断資産",
};

export default function CloudRunReturnReviewPage() {
  const [items, setItems] = useState<ReturnItem[]>([]);
  const [summary, setSummary] = useState<ReturnResponse["summary"] | null>(null);
  const [dbPath, setDbPath] = useState("");
  const [status, setStatus] = useState<FilterStatus>("candidate");
  const [kind, setKind] = useState<FilterKind>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiClient.get<ReturnResponse>("/api/cloudrun-return-review", {
        params: { status, kind, limit: 150 },
      });
      setItems(response.data.items ?? []);
      setSummary(response.data.summary ?? null);
      setDbPath(response.data.db_path ?? "");
    } catch {
      setError("Cloud Run帰還データを読み込めませんでした。APIと隔離DBの状態を確認してください。");
    } finally {
      setLoading(false);
    }
  }, [kind, status]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const totals = useMemo(
    () => ({
      candidate: summary?.candidate ?? 0,
      approved: summary?.approved ?? 0,
      held: summary?.held ?? 0,
      rejected: summary?.rejected ?? 0,
      total: summary?.total ?? 0,
    }),
    [summary],
  );

  const handleReview = useCallback(
    async (item: ReturnItem, nextStatus: Exclude<ReturnStatus, "candidate">) => {
      const key = `${item.kind}:${item.id}`;
      setActionLoading((current) => ({ ...current, [key]: true }));
      setError("");
      try {
        const response = await apiClient.patch<{ item: ReturnItem }>(
          `/api/cloudrun-return-review/${item.kind}/${item.id}`,
          {
            review_status: nextStatus,
            note:
              nextStatus === "approved"
                ? "demo.db昇格候補として承認"
                : nextStatus === "held"
                  ? "内容確認のため保留"
                  : "demo.dbへ昇格しない",
          },
        );
        const updated = response.data.item;
        setItems((current) =>
          current.map((candidate) =>
            candidate.kind === item.kind && candidate.id === item.id ? updated : candidate,
          ),
        );
        await fetchData();
      } catch {
        setError("レビュー結果を保存できませんでした。もう一度お試しください。");
      } finally {
        setActionLoading((current) => ({ ...current, [key]: false }));
      }
    },
    [fetchData],
  );

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-6xl space-y-5">
        <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-teal-600 text-white">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-black text-slate-900">
                  Cloud Run 帰還データ検疫
                </h1>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  デモ環境で積んだ審査入力・OCR・紫苑レビュー・人間の反応を隔離DBで確認し、demo.dbへ戻す候補だけを承認します。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 lg:ml-auto">
              <Link
                href="/system-overview"
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 hover:bg-slate-100"
              >
                <Database className="h-4 w-4" />
                仕組みを見る
              </Link>
              <button
                type="button"
                onClick={fetchData}
                disabled={loading}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                更新
              </button>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <SummaryCard label="検疫待ち" value={totals.candidate} color="amber" />
            <SummaryCard label="承認済み" value={totals.approved} color="emerald" />
            <SummaryCard label="保留" value={totals.held} color="sky" />
            <SummaryCard label="破棄" value={totals.rejected} color="rose" />
            <SummaryCard label="総件数" value={totals.total} color="slate" />
          </div>
        </header>

        <section className="rounded-2xl border border-teal-200 bg-teal-50 p-4">
          <div className="flex gap-3">
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-teal-700" />
            <div>
              <p className="text-sm font-black text-teal-950">安全な承認方式</p>
              <p className="mt-1 text-sm leading-6 text-teal-800">
                この画面の承認は隔離DB内の印付けです。ここでは
                <span className="font-black"> data/lease_data.db へ直接書き込みません</span>。
                承認済みだけを後で data/demo.db への昇格対象にします。
              </p>
              {dbPath && (
                <p className="mt-2 break-all font-mono text-xs text-teal-700">
                  quarantine: {dbPath}
                </p>
              )}
            </div>
          </div>
        </section>

        <div className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm md:flex-row md:items-center md:justify-between">
          <FilterGroup
            label="状態"
            value={status}
            options={[
              ["candidate", "検疫待ち"],
              ["approved", "承認済み"],
              ["held", "保留"],
              ["rejected", "破棄"],
              ["all", "すべて"],
            ]}
            onChange={(value) => setStatus(value as FilterStatus)}
          />
          <FilterGroup
            label="種類"
            value={kind}
            options={[
              ["all", "すべて"],
              ["score_input", "審査入力"],
              ["ocr_result", "OCR"],
              ["shion_review", "紫苑レビュー"],
              ["judgment_asset", "判断資産"],
            ]}
            onChange={(value) => setKind(value as FilterKind)}
          />
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        )}

        {loading ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-sm font-bold text-slate-500">
            Cloud Run帰還データを読み込み中...
          </div>
        ) : items.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center">
            <FileSearch className="mx-auto h-8 w-8 text-slate-400" />
            <p className="mt-3 font-black text-slate-800">
              該当する帰還データはありません
            </p>
            <p className="mt-1 text-sm text-slate-500">
              GCS同期後に審査入力・OCR・紫苑レビュー・判断資産候補がここへ表示されます。
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {items.map((item) => {
              const loadingKey = `${item.kind}:${item.id}`;
              return (
                <article
                  key={`${item.kind}:${item.id}`}
                  className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
                >
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`rounded-full border px-2.5 py-1 text-xs font-black ${STATUS_STYLE[item.review_status]}`}>
                          {STATUS_LABEL[item.review_status]}
                        </span>
                        <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-600">
                          {KIND_LABEL[item.kind]}
                        </span>
                        <span className="text-xs font-bold text-slate-400">
                          #{item.id}
                          {item.source_id ? ` / ${item.source_id}` : ""}
                        </span>
                      </div>
                      <h2 className="mt-3 text-lg font-black text-slate-900">
                        {item.title}
                      </h2>
                      <p className="mt-1 text-xs font-bold text-slate-400">
                        {item.created_at || "created_at未記録"}
                        {item.reviewed_at ? ` / reviewed ${item.reviewed_at}` : ""}
                      </p>
                    </div>
                    {item.review_status === "candidate" && (
                      <div className="flex flex-col gap-2 sm:flex-row md:shrink-0">
                        <button
                          type="button"
                          onClick={() => handleReview(item, "rejected")}
                          disabled={actionLoading[loadingKey]}
                          className="inline-flex items-center justify-center gap-2 rounded-lg border border-rose-200 bg-white px-3 py-2 text-sm font-black text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                        >
                          <XCircle className="h-4 w-4" />
                          破棄
                        </button>
                        <button
                          type="button"
                          onClick={() => handleReview(item, "held")}
                          disabled={actionLoading[loadingKey]}
                          className="inline-flex items-center justify-center gap-2 rounded-lg border border-sky-200 bg-white px-3 py-2 text-sm font-black text-sky-700 hover:bg-sky-50 disabled:opacity-50"
                        >
                          <PauseCircle className="h-4 w-4" />
                          保留
                        </button>
                        <button
                          type="button"
                          onClick={() => handleReview(item, "approved")}
                          disabled={actionLoading[loadingKey]}
                          className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-black text-white hover:bg-emerald-700 disabled:opacity-50"
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          承認
                        </button>
                      </div>
                    )}
                  </div>
                  {item.review_note && (
                    <p className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-bold text-slate-600">
                      {item.review_note}
                    </p>
                  )}
                  <pre className="mt-4 max-h-80 overflow-auto rounded-xl border border-slate-200 bg-slate-950 p-4 text-xs leading-5 text-slate-100">
                    {item.preview || "previewなし"}
                  </pre>
                </article>
              );
            })}
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
  color: "amber" | "emerald" | "sky" | "rose" | "slate";
}) {
  const styles = {
    amber: "border-amber-200 bg-amber-50 text-amber-800",
    emerald: "border-emerald-200 bg-emerald-50 text-emerald-800",
    sky: "border-sky-200 bg-sky-50 text-sky-800",
    rose: "border-rose-200 bg-rose-50 text-rose-800",
    slate: "border-slate-200 bg-slate-100 text-slate-800",
  };
  return (
    <div className={`rounded-xl border p-4 ${styles[color]}`}>
      <p className="text-xs font-black">{label}</p>
      <p className="mt-1 text-2xl font-black">{value}</p>
    </div>
  );
}

function FilterGroup({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly (readonly [string, string])[];
  onChange: (value: string) => void;
}) {
  return (
    <div>
      <p className="mb-2 text-xs font-black text-slate-500">{label}</p>
      <div className="flex flex-wrap gap-2">
        {options.map(([optionValue, optionLabel]) => (
          <button
            key={optionValue}
            type="button"
            onClick={() => onChange(optionValue)}
            className={`rounded-full border px-3 py-1.5 text-xs font-black transition-colors ${
              value === optionValue
                ? "border-slate-900 bg-slate-900 text-white"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-100"
            }`}
          >
            {optionLabel}
          </button>
        ))}
      </div>
    </div>
  );
}
