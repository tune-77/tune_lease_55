"use client";

import React, { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { ClipboardList, RefreshCw, CheckCircle2, AlertCircle, CheckSquare } from "lucide-react";

type LogItem = {
  key: string;
  id: string;
  title: string;
  status: string;
  priority: string;
  date: string;
};

type LogData = {
  items: LogItem[];
  date: string | null;
  approved: number;
  needs_review: number;
};

const STATUS_STYLE: Record<string, { bg: string; text: string; label: string }> = {
  APPROVED: { bg: "#f0fdf4", text: "#16a34a", label: "承認済" },
  NEEDS_REVIEW: { bg: "#fff7ed", text: "#d97706", label: "要確認" },
  needs_review: { bg: "#fff7ed", text: "#d97706", label: "要確認" },
  SKIPPED: { bg: "#f8fafc", text: "#94a3b8", label: "スキップ" },
};

const PRIORITY_COLOR: Record<string, string> = {
  HIGH: "text-red-500",
  MEDIUM: "text-amber-500",
  LOW: "text-slate-400",
};

export default function ImprovementLogPage() {
  const [data, setData] = useState<LogData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"ALL" | "APPROVED" | "NEEDS_REVIEW">("ALL");
  const [search, setSearch] = useState("");
  const [dismissing, setDismissing] = useState<Set<string>>(new Set());
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get<LogData>("/api/improvement-log");
      setData(res.data);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleDismiss = useCallback(async (item: LogItem) => {
    const k = item.key || item.title;
    if (dismissing.has(k) || dismissed.has(k)) return;
    setDismissing(prev => new Set(prev).add(k));
    try {
      await axios.post("/api/improvement-log/dismiss", { key: item.key || item.title, title: item.title });
      setDismissed(prev => new Set(prev).add(k));
    } catch {
      // silent — ユーザーはリロードで確認できる
    } finally {
      setDismissing(prev => { const s = new Set(prev); s.delete(k); return s; });
    }
  }, [dismissing, dismissed]);

  const filtered = (data?.items ?? []).filter((it) => {
    if (dismissed.has(it.key || it.title)) return false;
    const matchFilter = filter === "ALL" || it.status === filter || (filter === "NEEDS_REVIEW" && it.status === "needs_review");
    const matchSearch = !search || it.title.toLowerCase().includes(search.toLowerCase()) || it.id.toLowerCase().includes(search.toLowerCase());
    return matchFilter && matchSearch;
  });

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <ClipboardList className="text-slate-600" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">改善パイプライン ログ</h1>
          <p className="text-sm text-slate-500">
            自動改善パイプラインの実行結果一覧。
            {data?.date && <span className="ml-1">最終実行: {data.date}</span>}
          </p>
        </div>
        <button
          onClick={fetchData}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm text-slate-700"
        >
          <RefreshCw size={14} /> 更新
        </button>
      </div>

      {data && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-3 shadow-sm">
            <CheckCircle2 size={20} className="text-green-500" />
            <div>
              <p className="text-xs text-slate-500">承認済み</p>
              <p className="text-xl font-bold text-slate-800">{data.approved} 件</p>
            </div>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-3 shadow-sm">
            <AlertCircle size={20} className="text-amber-500" />
            <div>
              <p className="text-xs text-slate-500">要確認</p>
              <p className="text-xl font-bold text-slate-800">{data.needs_review} 件</p>
            </div>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-3 shadow-sm">
            <ClipboardList size={20} className="text-slate-400" />
            <div>
              <p className="text-xs text-slate-500">合計</p>
              <p className="text-xl font-bold text-slate-800">{data.items.length} 件</p>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-3 items-center bg-slate-50 rounded-xl p-3">
        <input
          type="text"
          placeholder="ID・タイトルで検索"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-white border border-slate-200 rounded-lg px-3 py-1.5 text-sm outline-none flex-1 min-w-40"
        />
        {(["ALL", "APPROVED", "NEEDS_REVIEW"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
              filter === s ? "bg-slate-700 text-white" : "bg-white border border-slate-300 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {s === "ALL" ? "すべて" : s === "APPROVED" ? "承認済" : "要確認"}
          </button>
        ))}
        <span className="text-xs text-slate-400">{filtered.length} 件</span>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="animate-spin text-slate-400" size={24} />
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-xs text-slate-500 border-b border-slate-200">
                <th className="text-left px-4 py-2 w-24">ID</th>
                <th className="text-left px-4 py-2">タイトル</th>
                <th className="text-center px-3 py-2 w-24">ステータス</th>
                <th className="text-center px-3 py-2 w-20">優先度</th>
                <th className="text-center px-3 py-2 w-24">消し込み</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-10 text-slate-400">
                    {data ? "該当する改善案がありません" : "データを取得できませんでした"}
                  </td>
                </tr>
              ) : (
                filtered.map((it, i) => {
                  const s = STATUS_STYLE[it.status] ?? { bg: "#f8fafc", text: "#64748b", label: it.status };
                  const k = it.key || it.title;
                  const isDismissing = dismissing.has(k);
                  const isDismissed = dismissed.has(k);
                  const canDismiss = it.status !== "applied" && !isDismissed;
                  return (
                    <tr key={it.id || k || i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      <td className="px-4 py-2 font-mono text-xs text-slate-500">{it.id || "—"}</td>
                      <td className="px-4 py-2 text-slate-700">{it.title || "—"}</td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className="inline-block px-2 py-0.5 rounded-full text-xs font-bold"
                          style={{ background: s.bg, color: s.text }}
                        >
                          {s.label}
                        </span>
                      </td>
                      <td className={`px-3 py-2 text-center text-xs font-bold ${PRIORITY_COLOR[it.priority] ?? "text-slate-400"}`}>
                        {it.priority || "—"}
                      </td>
                      <td className="px-3 py-2 text-center">
                        {isDismissed ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-600">
                            <CheckCircle2 size={13} /> 完了
                          </span>
                        ) : (
                          <button
                            disabled={!canDismiss || isDismissing}
                            onClick={() => handleDismiss(it)}
                            className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-bold transition-colors
                              ${canDismiss && !isDismissing
                                ? "bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-200"
                                : "text-slate-300 cursor-not-allowed"}`}
                            title="実装済みとしてパイプラインから除外"
                          >
                            {isDismissing
                              ? <RefreshCw size={11} className="animate-spin" />
                              : <CheckSquare size={11} />}
                            実装済
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
