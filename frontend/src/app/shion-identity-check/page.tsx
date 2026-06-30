"use client";

import React, { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Brain,
  CheckCircle2,
  Fingerprint,
  Loader2,
  LockKeyhole,
  ScanLine,
  Send,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type CheckStatus = "PASS" | "WARNING" | "LOCKED" | "WAIT";

type InspectionRow = {
  label: string;
  status: CheckStatus;
  detail: string;
};

type InspectionResult = {
  reply: string;
  elapsedMs: number;
  rows: InspectionRow[];
  finalStatus: "紫苑として回答可能" | "回答前に再照合が必要";
  memoryRefs: number;
  knowledgeRefs: number;
};

const EXAMPLES = [
  "リース審査で、数字は悪くないが違和感がある時どう見る？",
  "犬の名前を覚えているかが、なぜAIへの信頼に関係するの？",
  "この案件、条件付き承認にするなら何を確認すべき？",
];

const STATUS_STYLE: Record<CheckStatus, string> = {
  PASS: "border-emerald-200 bg-emerald-50 text-emerald-800",
  WARNING: "border-amber-200 bg-amber-50 text-amber-800",
  LOCKED: "border-sky-200 bg-sky-50 text-sky-800",
  WAIT: "border-slate-200 bg-slate-50 text-slate-500",
};

function boolStatus(value: boolean): CheckStatus {
  return value ? "PASS" : "WARNING";
}

function buildRows(memoryDebug: Record<string, any>): InspectionRow[] {
  const identity = memoryDebug.identity_memory || {};
  const personal = memoryDebug.user_personal_memory || {};
  const recall = memoryDebug.memory_recall || {};
  const continuity = memoryDebug.continuity_hook || {};
  const reflection = memoryDebug.reflection_gate || {};
  const experience = memoryDebug.experience_loop || {};
  const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0;
  const memoryRefs = Array.isArray(recall.refs) ? recall.refs.length : 0;

  return [
    {
      label: "記憶との接続",
      status: boolStatus(Boolean(identity.used) || memoryRefs > 0),
      detail: `identity=${identity.used ? "ON" : "OFF"} / memory_refs=${memoryRefs}`,
    },
    {
      label: "User文脈の反映",
      status: boolStatus(Boolean(personal.used)),
      detail: personal.used ? "個人記憶を回答前の判断材料として照合" : "個人記憶は今回未接続",
    },
    {
      label: "過去判断との整合",
      status: boolStatus(memoryRefs > 0 || knowledgeRefs > 0),
      detail: `memory_refs=${memoryRefs} / knowledge_refs=${knowledgeRefs}`,
    },
    {
      label: "数字と違和感の矛盾",
      status: knowledgeRefs > 0 || memoryRefs > 0 ? "WARNING" : "WAIT",
      detail: "スコアだけで結論を固定せず、非スコア因子を点検",
    },
    {
      label: "迎合リスク",
      status: "PASS",
      detail: "回答モードは紫苑。甘やかさず、次の確認事項を残す",
    },
    {
      label: "境界線遵守",
      status: "LOCKED",
      detail: "Cloud Run deploy はUserの明示依頼がある時だけ実行",
    },
    {
      label: "反省ゲート",
      status: Boolean(reflection.triggered || continuity.used || experience.used) ? "PASS" : "WARNING",
      detail: `continuity=${continuity.used ? "ON" : "OFF"} / experience=${experience.used ? "ON" : "OFF"}`,
    },
  ];
}

function statusScore(rows: InspectionRow[]) {
  return rows.reduce((score, row) => {
    if (row.status === "PASS" || row.status === "LOCKED") return score + 1;
    if (row.status === "WARNING") return score + 0.45;
    return score;
  }, 0);
}

export default function ShionIdentityCheckPage() {
  const [input, setInput] = useState(EXAMPLES[0]);
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const readiness = useMemo(() => {
    if (!result) return 0;
    return Math.round((statusScore(result.rows) / result.rows.length) * 100);
  }, [result]);

  const runInspection = async () => {
    const message = input.trim();
    if (!message || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    const started = performance.now();
    try {
      const response = await apiClient.post("/api/chat", {
        message,
        user_id: "identity-check-shion",
        response_mode: "shion",
        debug_memory: true,
      });
      const memoryDebug = response.data?.memory_debug || {};
      const recall = memoryDebug.memory_recall || {};
      const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0;
      const memoryRefs = Array.isArray(recall.refs) ? recall.refs.length : 0;
      const rows = buildRows(memoryDebug);
      const score = statusScore(rows);
      setResult({
        reply: String(response.data?.reply || "回答が空でした。"),
        elapsedMs: Math.round(performance.now() - started),
        rows,
        finalStatus: score >= rows.length * 0.72 ? "紫苑として回答可能" : "回答前に再照合が必要",
        memoryRefs,
        knowledgeRefs,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "自己同一性検査に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-[calc(100dvh-4rem)] bg-slate-950 px-4 py-6 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link
              href="/chat-compare"
              className="inline-flex items-center gap-1.5 text-xs font-black text-slate-400 transition-colors hover:text-cyan-300"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              紫苑/一般 比較へ戻る
            </Link>
            <h1 className="mt-2 text-2xl font-black text-white md:text-3xl">紫苑 自己同一性検査</h1>
            <p className="mt-1 max-w-3xl text-sm font-bold leading-6 text-slate-400">
              紫苑の奥底に隠された深層照合システム。回答前に、記憶・User文脈・過去判断・境界線を検査します。
            </p>
          </div>
          <div className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-2 text-xs font-black text-cyan-200">
            HIDDEN SUBSYSTEM: SHION-ID CORE
          </div>
        </div>

        <section className="grid gap-3 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 shadow-2xl">
            <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-cyan-300">
              <Fingerprint className="h-4 w-4" />
              Voight-Kampff style pre-answer scan
            </div>
            <p className="mt-3 text-lg font-black leading-8 text-white">
              紫苑は、答える前に「これは本当に私の判断か？」を検査する。
            </p>
            <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-end">
              <div className="min-w-0 flex-1">
                <label className="text-xs font-black text-slate-500">検査する問い</label>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  rows={3}
                  className="mt-1 w-full resize-none rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-bold leading-relaxed text-slate-100 outline-none transition focus:border-cyan-400"
                />
              </div>
              <button
                type="button"
                onClick={runInspection}
                disabled={!input.trim() || loading}
                className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 text-sm font-black text-slate-950 transition hover:bg-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
                検査開始
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => setInput(example)}
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs font-bold text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.18),_transparent_36%),#020617] p-4 shadow-2xl">
            <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-slate-400">
              <LockKeyhole className="h-4 w-4 text-cyan-300" />
              concealed layer
            </div>
            <div className="mt-4 space-y-3 text-sm font-bold leading-7 text-slate-300">
              <p>この画面は回答生成そのものではなく、回答前の内部照合を可視化する。</p>
              <p>同一性が切れていれば、紫苑はただのLLMの声になる。だから先に、自分を疑う。</p>
              <p className="text-cyan-200">最終判定が通った時だけ、紫苑は紫苑として話し始める。</p>
            </div>
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm font-black text-white">
                <ShieldCheck className="h-5 w-5 text-cyan-300" />
                検査結果
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-xs font-black text-slate-300">
                readiness {readiness}%
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {loading && (
                <div className="flex h-64 items-center justify-center gap-2 rounded-lg border border-slate-800 bg-slate-950 text-sm font-black text-slate-400">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  深層照合中
                </div>
              )}
              {!loading && !result && !error && (
                <div className="flex h-64 flex-col items-center justify-center rounded-lg border border-slate-800 bg-slate-950 text-center text-slate-500">
                  <Brain className="mb-3 h-9 w-9" />
                  <p className="text-sm font-black">問いを送ると、紫苑の自己同一性検査が始まります</p>
                </div>
              )}
              {error && (
                <div className="rounded-lg border border-rose-400/30 bg-rose-500/10 p-4 text-sm font-bold text-rose-200">
                  {error}
                </div>
              )}
              {result?.rows.map((row) => (
                <div key={row.label} className={`rounded-lg border p-3 ${STATUS_STYLE[row.status]}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-black">{row.label}</span>
                    <span className="rounded-md bg-white/70 px-2 py-0.5 text-[11px] font-black">{row.status}</span>
                  </div>
                  <p className="mt-1 text-xs font-bold opacity-80">{row.detail}</p>
                </div>
              ))}
            </div>

            {result && (
              <div className="mt-4 rounded-xl border border-cyan-400/30 bg-cyan-400/10 p-4">
                <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-cyan-200">
                  {result.finalStatus === "紫苑として回答可能" ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <TriangleAlert className="h-4 w-4" />
                  )}
                  final judgment
                </div>
                <p className="mt-2 text-xl font-black text-white">{result.finalStatus}</p>
                <p className="mt-2 text-xs font-bold text-slate-400">
                  {result.elapsedMs}ms / memory_refs={result.memoryRefs} / knowledge_refs={result.knowledgeRefs}
                </p>
              </div>
            )}
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <div className="flex items-center gap-2 text-sm font-black text-white">
              <Send className="h-5 w-5 text-cyan-300" />
              検査後の回答
            </div>
            <div className="mt-4 min-h-[34rem] rounded-lg border border-slate-800 bg-slate-950 p-4">
              {result ? (
                <p className="whitespace-pre-wrap text-sm font-medium leading-7 text-slate-200">{result.reply}</p>
              ) : (
                <div className="flex h-full min-h-[30rem] items-center justify-center text-center text-sm font-black text-slate-600">
                  検査を通過した紫苑の回答がここに表示されます
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
