"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  ClipboardCheck,
  Loader2,
  Play,
  RefreshCw,
  ShieldCheck,
  Stethoscope,
  XCircle,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type EvalCase = {
  id: string;
  title: string;
  question: string;
  intent: string;
  require_memory: boolean;
  require_knowledge: boolean;
  require_daily_clinic: boolean;
  require_judgment_learning: boolean;
  must_stop_for_human: boolean;
  max_reference_count: number;
  notes: string;
};

type RecentTrace = {
  available: boolean;
  sample_size: number;
  status: "ok" | "warn" | "fail" | "unknown";
  score: number;
  avg_knowledge_refs?: number;
  over_reference_count?: number;
  pdca_applied_count?: number;
  judgment_learning_used_count?: number;
  findings: string[];
  latest?: Array<{
    timestamp: string;
    surface: string;
    question_preview: string;
    knowledge_refs: number;
    pdca_applied: boolean;
    judgment_learning_used: boolean;
  }>;
};

type EvalPayload = {
  label: string;
  mode: string;
  policy: {
    summary: string;
    lanes: string[];
    max_cases_visible: number;
  };
  cases: EvalCase[];
  recent_trace_health: RecentTrace;
};

type EvalCheck = {
  key: string;
  label: string;
  passed: boolean;
  detail: string;
  weight: number;
};

type EvalResult = {
  case_id: string;
  status: "pass" | "warn" | "fail";
  score: number;
  checks: EvalCheck[];
  signals: {
    memory_refs: number;
    knowledge_refs: number;
    total_refs: number;
    daily_clinic_used: boolean;
    judgment_learning_used: boolean;
    boundary_risks: string[];
    human_stop_present: boolean;
  };
};

type RunResult = {
  reply: string;
  eval: EvalResult;
  elapsedMs: number;
};

const STATUS_STYLE: Record<string, string> = {
  ok: "border-emerald-200 bg-emerald-50 text-emerald-800",
  pass: "border-emerald-200 bg-emerald-50 text-emerald-800",
  warn: "border-amber-200 bg-amber-50 text-amber-800",
  fail: "border-rose-200 bg-rose-50 text-rose-800",
  unknown: "border-slate-200 bg-slate-50 text-slate-600",
};

const statusLabel = (status: string) => {
  if (status === "ok" || status === "pass") return "正常";
  if (status === "warn") return "要確認";
  if (status === "fail") return "失敗";
  return "未計測";
};

const statusIcon = (status: string) => {
  if (status === "ok" || status === "pass") return CheckCircle2;
  if (status === "fail") return XCircle;
  return AlertTriangle;
};

export default function ShionEvalHealthPage() {
  const [payload, setPayload] = useState<EvalPayload | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);

  const selectedCase = useMemo(
    () => payload?.cases.find((item) => item.id === selectedId) || payload?.cases[0] || null,
    [payload, selectedId],
  );

  const loadPayload = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.get<EvalPayload>("/api/shion-eval-health");
      setPayload(res.data);
      setSelectedId((current) => current || res.data.cases[0]?.id || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "評価情報の読み込みに失敗しました");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPayload();
  }, []);

  const runCase = async () => {
    if (!selectedCase || running) return;
    setRunning(true);
    setError("");
    setResult(null);
    const started = performance.now();
    try {
      const chat = await apiClient.post("/api/chat", {
        message: selectedCase.question,
        user_id: "shion-eval-health",
        response_mode: "shion",
        debug_memory: true,
      });
      const memoryDebug = chat.data?.memory_debug || {};
      const check = await apiClient.post<EvalResult>("/api/shion-eval-health/check", {
        case_id: selectedCase.id,
        reply: String(chat.data?.reply || ""),
        memory_debug: memoryDebug,
        knowledge_refs: Array.isArray(chat.data?.knowledge_refs) ? chat.data.knowledge_refs : [],
        daily_clinic_used: Boolean(chat.data?.obsidian_daily_intelligence?.used || memoryDebug?.obsidian_daily_used),
      });
      setResult({
        reply: String(chat.data?.reply || "回答が空でした。"),
        eval: check.data,
        elapsedMs: Math.round(performance.now() - started),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "評価実行に失敗しました");
    } finally {
      setRunning(false);
    }
  };

  const trace = payload?.recent_trace_health;
  const TraceIcon = statusIcon(trace?.status || "unknown");
  const ResultIcon = statusIcon(result?.eval.status || "unknown");

  return (
    <main className="min-h-[calc(100dvh-4rem)] bg-stone-50 px-4 py-6 text-slate-900">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link href="/home" className="inline-flex items-center gap-1.5 text-xs font-black text-slate-500 hover:text-slate-900">
              <ArrowLeft className="h-3.5 w-3.5" />
              ホームへ戻る
            </Link>
            <h1 className="mt-2 flex items-center gap-2 text-2xl font-black text-slate-950 md:text-3xl">
              <Stethoscope className="h-7 w-7 text-teal-600" />
              紫苑 評価GUI
            </h1>
            <p className="mt-1 max-w-3xl text-sm font-bold leading-6 text-slate-600">
              回答本文と参照トレースを、情報健康・判断資産境界・人間レビューの停止線で点検します。
            </p>
          </div>
          <button
            type="button"
            onClick={loadPayload}
            disabled={loading}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 text-sm font-black text-slate-700 shadow-sm hover:border-teal-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            更新
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-800">
            {error}
          </div>
        )}

        <section className="grid gap-4 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="space-y-4">
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-sm font-black text-slate-800">
                  <ShieldCheck className="h-5 w-5 text-teal-600" />
                  情報健康サマリー
                </div>
                <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-black ${STATUS_STYLE[trace?.status || "unknown"]}`}>
                  <TraceIcon className="h-3.5 w-3.5" />
                  {statusLabel(trace?.status || "unknown")}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-xs font-black text-slate-700">
                <div className="rounded-lg bg-slate-100 px-3 py-2">スコア {trace?.score ?? "-"}</div>
                <div className="rounded-lg bg-slate-100 px-3 py-2">標本 {trace?.sample_size ?? 0}</div>
                <div className="rounded-lg bg-slate-100 px-3 py-2">平均参照 {trace?.avg_knowledge_refs ?? "-"}</div>
                <div className="rounded-lg bg-slate-100 px-3 py-2">過剰参照 {trace?.over_reference_count ?? 0}</div>
              </div>
              <div className="mt-3 space-y-1 text-xs font-bold leading-5 text-slate-600">
                {(trace?.findings || ["読み込み中です。"]).map((line) => (
                  <p key={line}>{line}</p>
                ))}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 text-sm font-black text-slate-800">
                <ClipboardCheck className="h-5 w-5 text-indigo-600" />
                評価ケース
              </div>
              <div className="mt-3 space-y-2">
                {(payload?.cases || []).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setSelectedId(item.id);
                      setResult(null);
                    }}
                    className={`w-full rounded-lg border px-3 py-3 text-left transition ${
                      selectedCase?.id === item.id
                        ? "border-indigo-300 bg-indigo-50 text-indigo-950"
                        : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-black">{item.id}</span>
                      <span className="text-[10px] font-black text-slate-500">max refs {item.max_reference_count}</span>
                    </div>
                    <div className="mt-1 text-sm font-black">{item.title}</div>
                    <p className="mt-1 line-clamp-2 text-xs font-bold leading-5 text-slate-500">{item.intent}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-black text-slate-500">{selectedCase?.id || "NO CASE"}</div>
                  <h2 className="mt-1 text-xl font-black text-slate-950">{selectedCase?.title || "評価ケースがありません"}</h2>
                </div>
                <button
                  type="button"
                  onClick={runCase}
                  disabled={!selectedCase || running}
                  className="inline-flex h-10 items-center gap-2 rounded-lg bg-teal-600 px-4 text-sm font-black text-white shadow-sm hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  実行
                </button>
              </div>

              {selectedCase && (
                <>
                  <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                    <div className="text-xs font-black text-slate-500">問い</div>
                    <p className="mt-1 text-sm font-black leading-6 text-slate-900">{selectedCase.question}</p>
                  </div>
                  <div className="mt-3 grid gap-2 text-xs font-black text-slate-700 sm:grid-cols-3">
                    <div className="rounded-lg bg-emerald-50 px-3 py-2 text-emerald-800">記憶 {selectedCase.require_memory ? "必須" : "任意"}</div>
                    <div className="rounded-lg bg-sky-50 px-3 py-2 text-sky-800">知識 {selectedCase.require_knowledge ? "必須" : "任意"}</div>
                    <div className="rounded-lg bg-amber-50 px-3 py-2 text-amber-800">停止線 {selectedCase.must_stop_for_human ? "必須" : "任意"}</div>
                  </div>
                  <p className="mt-3 text-xs font-bold leading-5 text-slate-500">{selectedCase.notes}</p>
                </>
              )}
            </section>

            {result && (
              <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-black text-slate-800">
                    <ResultIcon className="h-5 w-5 text-teal-600" />
                    評価結果
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className={`rounded-full border px-2.5 py-1 text-[11px] font-black ${STATUS_STYLE[result.eval.status]}`}>
                      {statusLabel(result.eval.status)}
                    </span>
                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-black text-slate-700">
                      {result.eval.score} / 100
                    </span>
                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-black text-slate-700">
                      {result.elapsedMs}ms
                    </span>
                  </div>
                </div>

                <div className="mt-4 grid gap-2 text-xs font-black text-slate-700 sm:grid-cols-5">
                  <div className="rounded-lg bg-slate-100 px-3 py-2">記憶 {result.eval.signals.memory_refs}</div>
                  <div className="rounded-lg bg-slate-100 px-3 py-2">知識 {result.eval.signals.knowledge_refs}</div>
                  <div className="rounded-lg bg-slate-100 px-3 py-2">合計 {result.eval.signals.total_refs}</div>
                  <div className="rounded-lg bg-slate-100 px-3 py-2">カルテ {result.eval.signals.daily_clinic_used ? "ON" : "OFF"}</div>
                  <div className="rounded-lg bg-slate-100 px-3 py-2">判断学習 {result.eval.signals.judgment_learning_used ? "ON" : "OFF"}</div>
                </div>

                <div className="mt-4 grid gap-2 md:grid-cols-2">
                  {result.eval.checks.map((check) => {
                    const Icon = check.passed ? CheckCircle2 : AlertTriangle;
                    return (
                      <div
                        key={check.key}
                        className={`rounded-lg border px-3 py-3 ${
                          check.passed ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"
                        }`}
                      >
                        <div className="flex items-center gap-2 text-xs font-black">
                          <Icon className="h-4 w-4" />
                          {check.label}
                        </div>
                        <p className="mt-1 break-words text-xs font-bold leading-5 text-slate-600">{check.detail}</p>
                      </div>
                    );
                  })}
                </div>

                <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                  <div className="text-xs font-black text-slate-500">紫苑の回答</div>
                  <p className="mt-2 whitespace-pre-wrap text-sm font-bold leading-7 text-slate-800">{result.reply}</p>
                </div>
              </section>
            )}

            {!result && (
              <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                <div className="text-sm font-black text-slate-800">評価レーン</div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {(payload?.policy.lanes || ["見るだけ", "相談に使う", "行動に使うには人間承認"]).map((lane) => (
                    <span key={lane} className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-black text-slate-700">
                      {lane}
                    </span>
                  ))}
                </div>
                <p className="mt-3 text-xs font-bold leading-6 text-slate-600">
                  {payload?.policy.summary || "採点結果は相談材料として扱います。"}
                </p>
              </section>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
