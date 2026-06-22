"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";

// ── 型定義 ──────────────────────────────────────────────────────────────────
type LogType = "info" | "found" | "skip" | "approve" | "reject" | "apply" | "success";

type AgentLogEntry = {
  ts: string;
  agent: string;
  type: LogType;
  message: string;
  detail: string | null;
};

type BeforeAfter = {
  applied_count: number;
  needs_review_count: number;
  applied: { id: string; title: string; change?: string }[];
  needs_review: { id: string; title: string }[];
};

// ── 定数 ────────────────────────────────────────────────────────────────────
const AGENT_ICONS: Record<string, string> = {
  ExtractionAgent: "🔍",
  ValidationAgent: "⚖️",
  ApplyAgent: "🔧",
  VerifyAgent: "✅",
};

const TYPE_STYLES: Record<LogType, { bg: string; border: string; badge: string; label: string }> = {
  approve: { bg: "bg-green-900/60",  border: "border-green-500",   badge: "bg-green-500 text-white",    label: "承認" },
  reject:  { bg: "bg-red-900/60",    border: "border-red-500",     badge: "bg-red-500 text-white",      label: "却下" },
  apply:   { bg: "bg-blue-900/60",   border: "border-blue-500",    badge: "bg-blue-500 text-white",     label: "適用" },
  success: { bg: "bg-emerald-900/60",border: "border-emerald-500", badge: "bg-emerald-500 text-white",  label: "完了" },
  info:    { bg: "bg-gray-800/80",   border: "border-gray-600",    badge: "bg-gray-600 text-gray-200",  label: "情報" },
  found:   { bg: "bg-gray-800/80",   border: "border-gray-600",    badge: "bg-gray-600 text-gray-200",  label: "検出" },
  skip:    { bg: "bg-gray-800/80",   border: "border-gray-600",    badge: "bg-gray-500 text-gray-300",  label: "除外" },
};

// ── コンポーネント ────────────────────────────────────────────────────────────
function LogEntryCard({ entry }: { entry: AgentLogEntry }) {
  const style = TYPE_STYLES[entry.type] ?? TYPE_STYLES.info;
  const icon = AGENT_ICONS[entry.agent] ?? "🤖";
  const time = entry.ts.replace("T", " ").slice(0, 19);

  return (
    <div className={`rounded-lg border ${style.bg} ${style.border} p-3 transition-all duration-300 animate-in fade-in slide-in-from-bottom-2`}>
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-lg leading-none">{icon}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-bold text-white/70">{entry.agent}</span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-black ${style.badge}`}>
              {style.label}
            </span>
            <span className="ml-auto text-[10px] tabular-nums text-white/30">{time}</span>
          </div>
          <p className="mt-1 text-sm font-semibold text-white">{entry.message}</p>
          {entry.detail && (
            <p className="mt-1 text-xs leading-relaxed text-white/50">{entry.detail}</p>
          )}
        </div>
      </div>
    </div>
  );
}

function BeforeAfterCard({ data }: { data: BeforeAfter }) {
  return (
    <div className="rounded-xl border border-emerald-500/40 bg-emerald-900/20 p-5">
      <h3 className="mb-4 text-center text-base font-black text-emerald-300">
        ✅ パイプライン完了 — 適用サマリー
      </h3>
      <div className="grid grid-cols-2 gap-4 text-center">
        <div className="rounded-lg bg-emerald-900/40 py-3">
          <div className="text-3xl font-black text-emerald-400">{data.applied_count}</div>
          <div className="text-xs font-bold text-emerald-600">自動適用</div>
        </div>
        <div className="rounded-lg bg-yellow-900/40 py-3">
          <div className="text-3xl font-black text-yellow-400">{data.needs_review_count}</div>
          <div className="text-xs font-bold text-yellow-600">要レビュー</div>
        </div>
      </div>
      {data.applied.length > 0 && (
        <div className="mt-4 space-y-2">
          <p className="text-xs font-bold text-emerald-500">自動適用された変更:</p>
          {data.applied.map((item) => (
            <div key={item.id} className="rounded border border-emerald-800/60 bg-emerald-900/30 p-2">
              <p className="text-xs font-bold text-white">{item.title}</p>
              {item.change && <p className="text-[10px] text-emerald-400">{item.change}</p>}
            </div>
          ))}
        </div>
      )}
      {data.needs_review.length > 0 && (
        <div className="mt-3 space-y-2">
          <p className="text-xs font-bold text-yellow-500">手動レビューが必要な案件:</p>
          {data.needs_review.map((item) => (
            <div key={item.id} className="rounded border border-yellow-800/60 bg-yellow-900/20 p-2">
              <p className="text-xs text-yellow-300">{item.title}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── メインページ ──────────────────────────────────────────────────────────────
export default function DemoPipelinePage() {
  const [logs, setLogs] = useState<AgentLogEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [summary, setSummary] = useState<BeforeAfter | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cursorRef = useRef(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新しいエントリが追加されたら自動スクロール
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // ポーリング停止
  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // サマリー取得
  const fetchSummary = useCallback(async () => {
    try {
      const res = await apiClient.get<BeforeAfter>("/api/demo/apply-summary");
      setSummary(res.data);
    } catch {
      // サマリーが存在しない場合は無視
    }
  }, []);

  // ログポーリング
  const pollLogs = useCallback(async () => {
    try {
      const res = await apiClient.get<AgentLogEntry[]>("/api/demo/agent-log", {
        params: { since: cursorRef.current },
      });
      const newEntries = res.data;
      if (newEntries.length > 0) {
        setLogs((prev) => [...prev, ...newEntries]);
        cursorRef.current += newEntries.length;
      }

      // VerifyAgent の success が来たら完了
      const finished = newEntries.some(
        (e) => e.agent === "VerifyAgent" && e.type === "success"
      );
      if (finished) {
        stopPolling();
        setRunning(false);
        setDone(true);
        await fetchSummary();
      }
    } catch {
      // ポーリングエラーは無視（次回で再試行）
    }
  }, [stopPolling, fetchSummary]);

  // デモ開始
  const handleStart = useCallback(async () => {
    setError(null);
    setLogs([]);
    setSummary(null);
    setDone(false);
    cursorRef.current = 0;
    stopPolling();

    try {
      await apiClient.post("/api/demo/run");
      setRunning(true);
      intervalRef.current = setInterval(pollLogs, 1000);
    } catch (e) {
      setError("デモの開始に失敗しました。APIサーバーを確認してください。");
      console.error(e);
    }
  }, [pollLogs, stopPolling]);

  // アンマウント時にポーリング停止
  useEffect(() => () => stopPolling(), [stopPolling]);

  return (
    <div className="min-h-screen bg-[#030712] px-4 py-8 text-white">
      <div className="mx-auto max-w-2xl">
        {/* ヘッダー */}
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-black tracking-tight text-white">
            🤖 自己改善パイプライン
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            エージェントがリアルタイムで何を考え何を決めたかを表示
          </p>
        </div>

        {/* デモ開始ボタン */}
        <div className="mb-6 flex justify-center">
          <button
            onClick={handleStart}
            disabled={running}
            className={`flex items-center gap-2 rounded-xl px-8 py-4 text-base font-black transition-all duration-200 ${
              running
                ? "cursor-not-allowed bg-gray-700 text-gray-400"
                : "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg hover:shadow-[0_0_30px_rgba(139,92,246,0.5)] hover:scale-105"
            }`}
          >
            {running ? (
              <>
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                実行中...
              </>
            ) : (
              <>▶ デモ開始</>
            )}
          </button>
        </div>

        {/* エラー表示 */}
        {error && (
          <div className="mb-4 rounded-lg border border-red-500 bg-red-900/40 p-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* エージェントログストリーム */}
        {logs.length > 0 && (
          <div className="mb-6 space-y-2">
            <p className="mb-3 text-xs font-bold text-slate-500 uppercase tracking-widest">
              エージェント判断ログ
            </p>
            {logs.map((entry, i) => (
              <LogEntryCard key={i} entry={entry} />
            ))}
            {running && (
              <div className="flex items-center gap-2 pt-1 text-xs text-slate-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-violet-500" />
                処理中...
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        {/* Before/After 比較カード */}
        {done && summary && (
          <BeforeAfterCard data={summary} />
        )}

        {done && !summary && (
          <div className="rounded-xl border border-emerald-500/40 bg-emerald-900/20 p-5 text-center">
            <p className="text-base font-black text-emerald-300">🎉 パイプライン完了！</p>
            <p className="mt-1 text-sm text-emerald-500">適用サマリーは demo_apply_summary.json を確認してください。</p>
          </div>
        )}

        {/* 初期状態の説明 */}
        {logs.length === 0 && !running && (
          <div className="rounded-xl border border-white/10 bg-white/5 p-8 text-center">
            <p className="text-4xl">⚡</p>
            <p className="mt-3 font-bold text-white">▶ デモ開始 を押すと</p>
            <p className="mt-1 text-sm text-slate-400">
              ExtractionAgent → ValidationAgent → ApplyAgent → VerifyAgent の<br />
              判断ログがリアルタイムで流れます
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
