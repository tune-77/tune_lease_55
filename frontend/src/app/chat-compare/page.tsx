"use client";

import React, { useMemo, useState } from "react";
import { renderInline } from "@/lib/renderMarkdown";
import Link from "next/link";
import {
  ArrowLeft,
  Bot,
  BookOpen,
  Brain,
  Calendar,
  Clock,
  Database,
  GitCompare,
  Fingerprint,
  Loader2,
  MessageCircle,
  MessagesSquare,
  Network,
  RefreshCw,
  Send,
  Sparkles,
  User,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type ModeKey = "shion" | "general";

type CompareResult = {
  mode: ModeKey;
  reply: string;
  elapsedMs: number;
  error?: string;
  memoryRefs: number;
  knowledgeRefs: number;
  identityUsed: boolean;
  personalUsed: boolean;
  obsidianDailyUsed: boolean;
  experienceUsed: boolean;
};

const EXAMPLES = [
  "この案件、条件付き承認にするなら何を確認すべき？",
  "犬の名前を覚えているかが、なぜAIへの信頼に関係するの？",
  "リース審査で、数字は悪くないが違和感がある時どう見る？",
];

const MODE_META: Record<ModeKey, {
  label: string;
  caption: string;
  icon: React.ElementType;
  image?: string;
  accent: string;
  panel: string;
  badge: string;
}> = {
  shion: {
    label: "紫苑",
    caption: "記憶・関係性・判断資産を優先",
    icon: Bot,
    image: "/lease-intelligence/moods/attachment.webp",
    accent: "text-indigo-700",
    panel: "border-indigo-200 bg-indigo-50/70",
    badge: "bg-indigo-600 text-white",
  },
  general: {
    label: "一般",
    caption: "中立で分かりやすい通常回答",
    icon: MessageCircle,
    accent: "text-slate-700",
    panel: "border-slate-200 bg-white",
    badge: "bg-slate-800 text-white",
  },
};

const SHION_VARIANTS = [
  {
    key: "curiosity",
    label: "通常",
    caption: "対話・探索",
    image: "/lease-intelligence/moods/curiosity.webp",
  },
  {
    key: "focus",
    label: "精密",
    caption: "考え中",
    image: "/lease-intelligence/moods/focus.webp",
  },
  {
    key: "vigilance",
    label: "警戒",
    caption: "審査・検査",
    image: "/lease-intelligence/moods/vigilance.webp",
  },
  {
    key: "attachment",
    label: "親和",
    caption: "記憶・関係性",
    image: "/lease-intelligence/moods/attachment.webp",
  },
  {
    key: "weariness",
    label: "反省",
    caption: "迷い・疲労",
    image: "/lease-intelligence/moods/weariness.webp",
  },
  {
    key: "maintenance",
    label: "修復",
    caption: "改善・調整",
    image: "/lease-intelligence/moods/maintenance.webp",
  },
  {
    key: "roles",
    label: "分岐",
    caption: "複数紫苑",
    image: "/lease-intelligence/moods/roles.webp",
  },
];

const METRICS: Array<{
  key: "memory" | "knowledge" | "identity" | "personal" | "daily" | "experience";
  label: string;
  icon: React.ElementType;
  hint: string;
}> = [
  { key: "memory", label: "記憶参照", icon: Database, hint: "紫苑の記憶検索でヒットした件数" },
  { key: "knowledge", label: "知識参照", icon: BookOpen, hint: "Obsidianナレッジの参照件数" },
  { key: "identity", label: "同一性", icon: Fingerprint, hint: "過去の自分の判断を意識したか" },
  { key: "personal", label: "個人記憶", icon: User, hint: "ユーザー固有の記憶を使ったか" },
  { key: "daily", label: "日次知性", icon: Calendar, hint: "その日の日次内省を使ったか" },
  { key: "experience", label: "経験ループ", icon: RefreshCw, hint: "人間の反応フィードバックを次回へ反映したか" },
];

function metricValue(result: CompareResult | null, key: (typeof METRICS)[number]["key"]): string {
  if (!result) return "-";
  if (key === "memory") return `${result.memoryRefs}件`;
  if (key === "knowledge") return `${result.knowledgeRefs}件`;
  if (key === "identity") return result.identityUsed ? "使用" : "未使用";
  if (key === "personal") return result.personalUsed ? "ON" : "OFF";
  if (key === "daily") return result.obsidianDailyUsed ? "ON" : "OFF";
  return result.experienceUsed ? "ON" : "OFF";
}

function normalizeReply(value: unknown): string {
  return typeof value === "string" && value.trim() ? value : "回答が空でした。";
}

function signalLabel(value: boolean) {
  return value ? "ON" : "OFF";
}

export default function ChatComparePage() {
  const [input, setInput] = useState(EXAMPLES[0]);
  const [results, setResults] = useState<Record<ModeKey, CompareResult | null>>({
    shion: null,
    general: null,
  });
  const [loading, setLoading] = useState(false);

  const canSend = input.trim().length > 0 && !loading;

  const diffSummary = useMemo(() => {
    const shion = results.shion;
    const general = results.general;
    if (!shion || !general || shion.error || general.error) return null;
    const shionSignals =
      shion.memoryRefs +
      shion.knowledgeRefs +
      (shion.identityUsed ? 1 : 0) +
      (shion.personalUsed ? 1 : 0) +
      (shion.obsidianDailyUsed ? 1 : 0) +
      (shion.experienceUsed ? 1 : 0);
    const generalSignals =
      general.memoryRefs +
      general.knowledgeRefs +
      (general.identityUsed ? 1 : 0) +
      (general.personalUsed ? 1 : 0) +
      (general.obsidianDailyUsed ? 1 : 0) +
      (general.experienceUsed ? 1 : 0);
    if (shionSignals > generalSignals) {
      return "同じ問いでも、紫苑側だけが個人記憶・同一性・経験ループを使って回答しています。";
    }
    if (generalSignals > shionSignals) {
      return "一般側の方が参照件数は多いですが、文体と判断軸は中立寄りです。";
    }
    return "参照量は近いので、文体・冒頭・判断への変換の差を見てください。";
  }, [results]);

  const demoSignals = useMemo(() => {
    const shion = results.shion;
    const general = results.general;
    if (!shion || !general || shion.error || general.error) return null;
    return [
      {
        label: "個人記憶",
        shion: signalLabel(shion.personalUsed),
        general: signalLabel(general.personalUsed),
      },
      {
        label: "同一性",
        shion: signalLabel(shion.identityUsed),
        general: signalLabel(general.identityUsed),
      },
      {
        label: "記憶検索",
        shion: `${shion.memoryRefs}件`,
        general: `${general.memoryRefs}件`,
      },
      {
        label: "日次知性",
        shion: signalLabel(shion.obsidianDailyUsed),
        general: signalLabel(general.obsidianDailyUsed),
      },
      {
        label: "経験ループ",
        shion: signalLabel(shion.experienceUsed),
        general: signalLabel(general.experienceUsed),
      },
    ];
  }, [results]);

  const runCompare = async () => {
    const message = input.trim();
    if (!message || loading) return;
    setLoading(true);
    setResults({ shion: null, general: null });

    const requestOne = async (mode: ModeKey): Promise<CompareResult> => {
      const started = performance.now();
      try {
        const res = await apiClient.post("/api/chat", {
          message,
          user_id: `compare-${mode}`,
          response_mode: mode,
          debug_memory: true,
        });
        const memoryDebug = res.data?.memory_debug || {};
        const memoryRecall = memoryDebug.memory_recall || {};
        const identityMemory = memoryDebug.identity_memory || {};
        const personalMemory = memoryDebug.user_personal_memory || {};
        const experienceLoop = memoryDebug.experience_loop || {};
        const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs)
          ? memoryDebug.knowledge_refs.length
          : 0;
        const memoryRefs = Array.isArray(memoryRecall.refs)
          ? memoryRecall.refs.length
          : 0;
        return {
          mode,
          reply: normalizeReply(res.data?.reply),
          elapsedMs: Math.round(performance.now() - started),
          memoryRefs,
          knowledgeRefs,
          identityUsed: Boolean(identityMemory.used),
          personalUsed: Boolean(personalMemory.used),
          obsidianDailyUsed: Boolean(memoryDebug.obsidian_daily_used),
          experienceUsed: Boolean(experienceLoop.used),
        };
      } catch (error) {
        return {
          mode,
          reply: "",
          elapsedMs: Math.round(performance.now() - started),
          error: error instanceof Error ? error.message : "送信に失敗しました",
          memoryRefs: 0,
          knowledgeRefs: 0,
          identityUsed: false,
          personalUsed: false,
          obsidianDailyUsed: false,
          experienceUsed: false,
        };
      }
    };

    const [shion, general] = await Promise.all([requestOne("shion"), requestOne("general")]);
    setResults({ shion, general });
    setLoading(false);
  };

  return (
    <main className="min-h-[calc(100dvh-4rem)] bg-slate-50 px-4 py-6">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link
              href="/chat"
              className="inline-flex items-center gap-1.5 text-xs font-black text-slate-500 transition-colors hover:text-indigo-700"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              紫苑チャットへ戻る
            </Link>
            <h1 className="mt-2 text-2xl font-black text-slate-950 md:text-3xl">紫苑 / 一般 チャット比較</h1>
            <p className="mt-1 text-sm font-bold text-slate-500">
              同じ問いを2つの回答モードへ投げ、記憶・連続性・判断軸の差を見ます。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <div className="inline-flex items-center gap-2 rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs font-black text-indigo-700 shadow-sm">
              <Sparkles className="h-4 w-4" />
              debug_memory 有効
            </div>
            <div className="inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-white px-3 py-2 text-xs font-black text-emerald-700 shadow-sm">
              <GitCompare className="h-4 w-4" />
              Hackathon Demo
            </div>
            <Link
              href="/shion-identity-check"
              className="inline-flex items-center gap-2 rounded-lg border border-cyan-200 bg-white px-3 py-2 text-xs font-black text-cyan-700 shadow-sm transition hover:border-cyan-300 hover:bg-cyan-50"
            >
              <Fingerprint className="h-4 w-4" />
              One More Thing
            </Link>
          </div>
        </div>

        <section className="grid gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-black text-slate-500">
              <Brain className="h-4 w-4 text-indigo-600" />
              デモの見どころ
            </div>
            <p className="mt-2 text-sm font-black leading-6 text-slate-900">
              一般AIではなく、記憶と経験を判断に変えるAIとして見せる。
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-black text-slate-500">
              <Database className="h-4 w-4 text-emerald-600" />
              比較するもの
            </div>
            <p className="mt-2 text-sm font-black leading-6 text-slate-900">
              文章の違いだけでなく、裏で使った記憶層の差を表示する。
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-black text-slate-500">
              <Sparkles className="h-4 w-4 text-amber-600" />
              伝える一言
            </div>
            <p className="mt-2 text-sm font-black leading-6 text-slate-900">
              「記憶があるAI」と「ただ答えるAI」は、同じ問いでも返し方が変わる。
            </p>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center gap-2 text-xs font-black text-slate-500">
            <Fingerprint className="h-4 w-4 text-indigo-600" />
            指標の見方（下の比較カードで使う6項目）
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
            {METRICS.map((metric) => (
              <div key={metric.key} className="flex items-start gap-2 rounded-lg border border-slate-100 bg-slate-50 p-2.5">
                <metric.icon className="mt-0.5 h-4 w-4 flex-shrink-0 text-indigo-500" />
                <div>
                  <div className="text-xs font-black text-slate-800">{metric.label}</div>
                  <div className="mt-0.5 text-[11px] font-bold leading-4 text-slate-500">{metric.hint}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-indigo-100 bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-end justify-between gap-2">
            <div>
              <div className="text-xs font-black uppercase tracking-wide text-indigo-500">Shion Variants</div>
              <h2 className="mt-1 text-lg font-black text-slate-950">状態で顔が変わる紫苑</h2>
            </div>
            <Link
              href="/shion-identity-check"
              className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-200 px-3 py-1.5 text-xs font-black text-cyan-700 transition hover:bg-cyan-50"
            >
              <Fingerprint className="h-3.5 w-3.5" />
              自己同一性検査へ
            </Link>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
            {SHION_VARIANTS.map((variant) => (
              <div key={variant.key} className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
                <img src={variant.image} alt={`紫苑 ${variant.label}`} className="aspect-square w-full object-cover object-top" />
                <div className="p-2">
                  <div className="text-xs font-black text-slate-900">{variant.label}</div>
                  <div className="mt-0.5 text-[10px] font-bold text-slate-500">{variant.caption}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
            <div className="min-w-0 flex-1">
              <label className="text-xs font-black text-slate-500">比較する問い</label>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                rows={3}
                className="mt-1 w-full resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-bold leading-relaxed text-slate-800 outline-none transition focus:border-indigo-300 focus:bg-white"
              />
            </div>
            <button
              type="button"
              disabled={!canSend}
              onClick={runCompare}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-slate-950 px-5 text-sm font-black text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              比較する
            </button>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setInput(example)}
                className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-bold text-slate-600 transition hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        {diffSummary && (
          <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm font-black text-emerald-900">
            <div>{diffSummary}</div>
            {demoSignals && (
              <div className="mt-3 grid gap-2 sm:grid-cols-5">
                {demoSignals.map((signal) => (
                  <div key={signal.label} className="rounded-lg border border-emerald-100 bg-white/80 p-2">
                    <div className="text-[10px] text-emerald-700">{signal.label}</div>
                    <div className="mt-1 flex items-center justify-between gap-2 text-xs">
                      <span className="text-indigo-700">紫苑 {signal.shion}</span>
                      <span className="text-slate-500">一般 {signal.general}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        <section className="grid gap-4 lg:grid-cols-2">
          {(["shion", "general"] as const).map((mode) => {
            const meta = MODE_META[mode];
            const Icon = meta.icon;
            const result = results[mode];
            return (
              <article key={mode} className={`min-h-[28rem] rounded-xl border p-4 shadow-sm ${meta.panel}`}>
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    {meta.image ? (
                      <div className="h-12 w-12 overflow-hidden rounded-lg border border-indigo-200 bg-white shadow-sm">
                        <img src={meta.image} alt={meta.label} className="h-full w-full object-cover object-top" />
                      </div>
                    ) : (
                      <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${meta.badge}`}>
                        <Icon className="h-5 w-5" />
                      </div>
                    )}
                    <div>
                      <h2 className={`text-lg font-black ${meta.accent}`}>{meta.label}</h2>
                      <p className="text-xs font-bold text-slate-500">{meta.caption}</p>
                    </div>
                  </div>
                  {result && (
                    <div className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2 py-1 text-[11px] font-black text-slate-500">
                      <Clock className="h-3.5 w-3.5" />
                      {result.elapsedMs}ms
                    </div>
                  )}
                </div>

                <div className="mt-4 grid grid-cols-3 gap-2">
                  {METRICS.map((metric) => (
                    <div key={metric.key} className="rounded-lg border border-white/70 bg-white/80 p-2">
                      <div className="flex items-center gap-1 text-[10px] font-black text-slate-400">
                        <metric.icon className="h-3 w-3" />
                        {metric.label}
                      </div>
                      <div className="mt-1 text-sm font-black text-slate-800">{metricValue(result, metric.key)}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-lg border border-white/70 bg-white/90 p-4">
                  {loading && !result ? (
                    <div className="flex h-48 items-center justify-center gap-2 text-sm font-black text-slate-500">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      回答中
                    </div>
                  ) : result?.error ? (
                    <p className="text-sm font-bold leading-relaxed text-rose-700">{result.error}</p>
                  ) : result?.reply ? (
                    <>
                      <div className="flex items-center gap-1.5 text-[11px] font-black uppercase tracking-widest text-slate-400">
                        <MessagesSquare className="h-3.5 w-3.5" />
                        回答
                      </div>
                      <p className="mt-2 whitespace-pre-wrap text-sm font-medium leading-7 text-slate-800">{renderInline(result.reply)}</p>
                    </>
                  ) : (
                    <div className="flex h-48 flex-col items-center justify-center text-center text-slate-400">
                      <Network className="mb-3 h-8 w-8" />
                      <p className="text-sm font-black">問いを送るとここに回答が出ます</p>
                    </div>
                  )}
                </div>
              </article>
            );
          })}
        </section>
      </div>
    </main>
  );
}
