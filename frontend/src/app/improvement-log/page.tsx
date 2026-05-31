"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  AlertCircle,
  CheckCircle2,
  ClipboardList,
  RefreshCw,
  Search,
  ShieldCheck,
  Wrench,
} from "lucide-react";

type ImprovementItem = {
  id: string;
  title: string;
  status: string;
  priority?: string;
  category?: string;
  recommended_order?: number;
  canonical_key?: string;
  group_id?: string;
  duplicate_count?: number;
  reason?: string;
  auto_fix_policy?: { reason?: string; risk?: string };
};

type ImprovementLog = {
  date: string;
  generated_at: string;
  status: string;
  approved: number;
  auto_fix_candidates: number;
  needs_review: number;
  rejected: number;
  applied: number;
  items: ImprovementItem[];
  obsidian_compliance?: {
    status?: string;
    violations?: unknown[];
    route_sensitive_ids?: string[];
  };
  source?: string;
};

const STATUS_LABELS: Record<string, { label: string; className: string }> = {
  APPROVED: { label: "承認", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  AUTO_FIX_CANDIDATE: { label: "自動修正候補", className: "bg-blue-50 text-blue-700 border-blue-200" },
  NEEDS_REVIEW: { label: "要確認", className: "bg-amber-50 text-amber-700 border-amber-200" },
  needs_review: { label: "要確認", className: "bg-amber-50 text-amber-700 border-amber-200" },
  REJECTED: { label: "拒否", className: "bg-rose-50 text-rose-700 border-rose-200" },
  APPLIED: { label: "適用済", className: "bg-slate-100 text-slate-700 border-slate-300" },
  SKIPPED: { label: "スキップ", className: "bg-slate-50 text-slate-500 border-slate-200" },
};

const CATEGORY_LABELS: Record<string, string> = {
  quick_ui: "UI",
  obsidian_chat: "Obsidian/Chat",
  logic_light: "軽量ロジック",
  data_quality: "運用品質",
  db_api: "DB/API",
  external: "外部連携",
  infra: "インフラ",
  planning: "仕様整理",
};

export default function ImprovementLogPage() {
  const [data, setData] = useState<ImprovementLog | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("ALL");

  const fetchLog = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get<ImprovementLog>("/api/improvement-log");
      setData(res.data);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLog();
  }, [fetchLog]);

  const filteredItems = useMemo(() => {
    const items = data?.items ?? [];
    return items.filter((item) => {
      const matchesStatus = status === "ALL" || item.status === status || (status === "NEEDS_REVIEW" && item.status === "needs_review");
      const needle = query.trim().toLowerCase();
      const matchesQuery =
        !needle ||
        item.id.toLowerCase().includes(needle) ||
        (item.title || "").toLowerCase().includes(needle) ||
        (item.canonical_key || "").toLowerCase().includes(needle);
      return matchesStatus && matchesQuery;
    });
  }, [data?.items, query, status]);

  const obsidianStatus = data?.obsidian_compliance?.status || "unknown";
  const obsidianViolations = data?.obsidian_compliance?.violations?.length || 0;

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-6xl space-y-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900 text-white">
              <ClipboardList className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">改善パイプライン ログ</h1>
              <p className="text-sm text-slate-500">
                {data?.date ? `最終実行: ${data.date}` : "最新の改善レポートを読み込みます"}
              </p>
            </div>
          </div>
          <button
            onClick={fetchLog}
            className="ml-auto inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            <RefreshCw className="h-4 w-4" />
            更新
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-5">
          <Stat label="適用済" value={data?.applied ?? 0} icon={<CheckCircle2 className="h-4 w-4" />} />
          <Stat label="承認" value={data?.approved ?? 0} icon={<CheckCircle2 className="h-4 w-4" />} />
          <Stat label="自動修正候補" value={data?.auto_fix_candidates ?? 0} icon={<Wrench className="h-4 w-4" />} />
          <Stat label="要確認" value={data?.needs_review ?? 0} icon={<AlertCircle className="h-4 w-4" />} />
          <Stat label="拒否" value={data?.rejected ?? 0} icon={<AlertCircle className="h-4 w-4" />} />
        </div>

        <section className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="ID・タイトル・canonical_keyで検索"
                className="w-full rounded-md border border-slate-300 py-2 pl-9 pr-3 text-sm outline-none focus:border-slate-500"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {["ALL", "AUTO_FIX_CANDIDATE", "NEEDS_REVIEW", "REJECTED", "APPLIED"].map((key) => (
                <button
                  key={key}
                  onClick={() => setStatus(key)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    status === key ? "bg-slate-900 text-white" : "border border-slate-300 bg-white text-slate-600"
                  }`}
                >
                  {key === "ALL" ? "すべて" : STATUS_LABELS[key]?.label || key}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1">
              <ShieldCheck className="h-3.5 w-3.5" />
              Obsidian: {obsidianStatus} / violations {obsidianViolations}
            </span>
            {data?.source && <span className="rounded-full bg-slate-100 px-2 py-1">{data.source}</span>}
            <span className="rounded-full bg-slate-100 px-2 py-1">{filteredItems.length}件表示</span>
          </div>
        </section>

        <section className="overflow-hidden rounded-lg border border-slate-200 bg-white">
          {loading ? (
            <div className="p-10 text-center text-sm text-slate-500">読み込み中...</div>
          ) : filteredItems.length === 0 ? (
            <div className="p-10 text-center text-sm text-slate-500">該当する改善案がありません</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[900px] text-sm">
                <thead className="bg-slate-100 text-left text-xs text-slate-500">
                  <tr>
                    <th className="px-4 py-3">順</th>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">タイトル</th>
                    <th className="px-4 py-3">分類</th>
                    <th className="px-4 py-3">状態</th>
                    <th className="px-4 py-3">理由</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredItems.map((item) => {
                    const statusStyle = STATUS_LABELS[item.status] || {
                      label: item.status || "-",
                      className: "bg-slate-50 text-slate-600 border-slate-200",
                    };
                    return (
                      <tr key={`${item.id}-${item.status}`} className="align-top hover:bg-slate-50">
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">{item.recommended_order ?? "-"}</td>
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">{item.id}</td>
                        <td className="px-4 py-3">
                          <div className="font-medium text-slate-800">{item.title || "-"}</div>
                          <div className="mt-1 text-xs text-slate-400">
                            {item.canonical_key || "-"}
                            {item.duplicate_count ? ` / duplicates ${item.duplicate_count}` : ""}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs text-slate-600">
                          {CATEGORY_LABELS[item.category || ""] || item.category || "-"}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex rounded-full border px-2 py-1 text-xs font-semibold ${statusStyle.className}`}>
                            {statusStyle.label}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs leading-relaxed text-slate-600">
                          {item.auto_fix_policy?.reason || item.reason || "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function Stat({ label, value, icon }: { label: string; value: number; icon: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
    </div>
  );
}
