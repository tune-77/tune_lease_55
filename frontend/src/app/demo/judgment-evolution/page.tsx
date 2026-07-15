"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  BookOpenCheck,
  Brain,
  CheckCircle2,
  Clock3,
  GitBranch,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  UserCheck,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Snapshot = {
  cases: number;
  corrections: number;
  assets: number;
  reuse: number;
  hitRate: number;
  reviewDepth: number;
};

type TimelineEvent = {
  at: number;
  title: string;
  body: string;
  tag: string;
};

const DURATION_MS = 18000;

const timelineEvents: TimelineEvent[] = [
  {
    at: 80,
    title: "初期レビュー",
    body: "財務・返済原資・物件価値を一般的に確認するだけで、論点がまだ粗い。",
    tag: "before",
  },
  {
    at: 210,
    title: "人間が修正",
    body: "補助金未採択時の返済余力は、承認条件として別建てで残すべきだと教える。",
    tag: "human",
  },
  {
    at: 430,
    title: "判断資産化",
    body: "迷いと修正を、次回使える確認条件として保存する。丸写しではなく応用対象にする。",
    tag: "asset",
  },
  {
    at: 690,
    title: "類似案件で再利用",
    body: "工作機械・新規先・補助金依存の組み合わせで、過去の確認条件が再び呼び出される。",
    tag: "reuse",
  },
  {
    at: 910,
    title: "見る場所が変わる",
    body: "スコアだけでなく、資金使途、競合、未採択時の代替原資を先に確認するようになる。",
    tag: "after",
  },
];

const chartData = [
  { label: "0", assets: 3, reuse: 0, corrections: 0 },
  { label: "200", assets: 18, reuse: 7, corrections: 21 },
  { label: "400", assets: 36, reuse: 22, corrections: 43 },
  { label: "600", assets: 55, reuse: 48, corrections: 68 },
  { label: "800", assets: 72, reuse: 83, corrections: 94 },
  { label: "1000", assets: 91, reuse: 126, corrections: 121 },
];

const beforeChecks = [
  "財務内容を確認",
  "返済原資を確認",
  "物件価値を確認",
];

const afterChecks = [
  "補助金未採択時の代替返済計画を条件化",
  "更新投資なら稼働率と粗利改善の根拠を月次で確認",
  "新規先は競合見積と導入目的の説明整合性を先に見る",
];

const finalLines = [
  "AIが賢くなったのではなく、人間の判断が消えずに戻ってきた。",
  "紫苑は、リース審査判断の Human-in-the-loop DevOps です。",
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function easeOutCubic(x: number) {
  return 1 - Math.pow(1 - x, 3);
}

function makeSnapshot(progress: number): Snapshot {
  const eased = easeOutCubic(progress);
  return {
    cases: Math.round(1000 * progress),
    corrections: Math.round(121 * eased),
    assets: Math.round(3 + 88 * eased),
    reuse: Math.round(126 * Math.pow(progress, 1.28)),
    hitRate: Math.round(34 + 41 * eased),
    reviewDepth: Math.round(28 + 57 * eased),
  };
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("ja-JP").format(value);
}

function MetricCard({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className={`border bg-white p-4 shadow-sm ${tone}`}>
      <div className="text-[11px] font-black uppercase tracking-widest text-slate-500">{label}</div>
      <div className="mt-2 text-3xl font-black text-slate-950">{value}</div>
      <div className="mt-1 min-h-10 text-xs font-bold leading-5 text-slate-600">{detail}</div>
    </div>
  );
}

function Checklist({ items, strong }: { items: string[]; strong?: boolean }) {
  return (
    <div className="space-y-3">
      {items.map((item) => (
        <div
          key={item}
          className={`flex items-start gap-3 border p-3 ${
            strong ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-slate-50"
          }`}
        >
          <CheckCircle2 className={`mt-0.5 h-4 w-4 flex-shrink-0 ${strong ? "text-emerald-700" : "text-slate-400"}`} />
          <span className="text-sm font-bold leading-6 text-slate-800">{item}</span>
        </div>
      ))}
    </div>
  );
}

export default function JudgmentEvolutionDemoPage() {
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let start: number | null = null;

    const tick = (now: number) => {
      if (start == null) start = now - progress * DURATION_MS;
      const next = clamp((now - start) / DURATION_MS, 0, 1);
      setProgress(next);
      if (next < 1) {
        frame = window.requestAnimationFrame(tick);
      } else {
        setPlaying(false);
      }
    };

    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, progress]);

  const snapshot = useMemo(() => makeSnapshot(progress), [progress]);
  const visibleEvents = timelineEvents.filter((event) => snapshot.cases >= event.at);
  const activeEvent = visibleEvents[visibleEvents.length - 1] ?? timelineEvents[0];
  const afterUnlocked = progress > 0.68;
  const complete = progress >= 1;

  const play = () => {
    setStarted(true);
    if (progress >= 1) setProgress(0);
    setPlaying(true);
  };

  const reset = () => {
    setPlaying(false);
    setStarted(false);
    setProgress(0);
  };

  return (
    <main className="min-h-screen bg-[#f8fafc] text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-7 md:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-4xl">
              <div className="inline-flex items-center gap-2 border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-black text-amber-900">
                <Sparkles className="h-4 w-4" />
                1000件早送りデモ
              </div>
              <h1 className="mt-4 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                人間の判断が消えないDevOps
              </h1>
              <p className="mt-4 max-w-3xl text-base font-bold leading-8 text-slate-600">
                1件の審査画面では伝わりにくい紫苑の価値を、審査経験が蓄積して判断が変化する時間軸で見せます。
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={playing ? () => setPlaying(false) : play}
                className="inline-flex items-center gap-2 bg-slate-950 px-4 py-3 text-sm font-black text-white hover:bg-slate-800"
              >
                {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {playing ? "一時停止" : started ? "再開" : "再生"}
              </button>
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 border border-slate-300 bg-white px-4 py-3 text-sm font-black text-slate-800 hover:bg-slate-50"
              >
                <RotateCcw className="h-4 w-4" />
                リセット
              </button>
              <button
                onClick={() => {
                  setStarted(true);
                  setPlaying(false);
                  setProgress(1);
                }}
                className="inline-flex items-center gap-2 border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-black text-amber-900 hover:bg-amber-100"
              >
                <Zap className="h-4 w-4" />
                1000件へ
              </button>
              <Link
                href="/screening"
                className="inline-flex items-center gap-2 border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-black text-emerald-900 hover:bg-emerald-100"
              >
                実審査画面
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 py-6 md:px-8 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-[11px] font-black uppercase tracking-widest text-slate-500">Fast-forward</div>
              <div className="mt-1 text-xl font-black text-slate-950">
                {formatNumber(snapshot.cases)} / 1,000 件の審査経験
              </div>
            </div>
            <div className="inline-flex items-center gap-2 border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-black text-slate-700">
              <Clock3 className="h-4 w-4 text-slate-500" />
              録画用 18秒
            </div>
          </div>

          <div className="mt-5 h-4 bg-slate-100">
            <div
              className="h-full bg-[linear-gradient(90deg,#0f172a,#0f766e,#d97706)] transition-[width] duration-100"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Human corrections"
              value={formatNumber(snapshot.corrections)}
              detail="人間が違和感・修正・承認条件を戻した回数"
              tone="border-sky-200"
            />
            <MetricCard
              label="Judgment assets"
              value={formatNumber(snapshot.assets)}
              detail="次回案件で使える確認条件・判断の型"
              tone="border-emerald-200"
            />
            <MetricCard
              label="Reuse"
              value={formatNumber(snapshot.reuse)}
              detail="類似案件で過去判断が呼び戻された回数"
              tone="border-amber-200"
            />
            <MetricCard
              label="Useful signal"
              value={`${snapshot.hitRate}%`}
              detail="人間feedbackで効いた/修正後に使えた割合"
              tone="border-rose-200"
            />
          </div>

          <div className="mt-6 h-[260px] border border-slate-200 bg-slate-50 p-3">
            {mounted ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="label" tick={{ fontSize: 12, fontWeight: 700 }} />
                  <YAxis tick={{ fontSize: 12, fontWeight: 700 }} />
                  <Tooltip />
                  <Area type="monotone" dataKey="assets" name="判断資産" stroke="#047857" fill="#a7f3d0" strokeWidth={2} />
                  <Area type="monotone" dataKey="reuse" name="再利用" stroke="#b45309" fill="#fde68a" strokeWidth={2} />
                  <Area type="monotone" dataKey="corrections" name="人間修正" stroke="#0369a1" fill="#bae6fd" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-sm font-black text-slate-400">
                chart loading
              </div>
            )}
          </div>
        </div>

        <div className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center bg-slate-950 text-white">
              <GitBranch className="h-5 w-5" />
            </div>
            <div>
              <div className="text-[11px] font-black uppercase tracking-widest text-slate-500">Loop event</div>
              <h2 className="mt-1 text-xl font-black text-slate-950">{activeEvent.title}</h2>
              <p className="mt-2 min-h-14 text-sm font-bold leading-7 text-slate-600">{activeEvent.body}</p>
            </div>
          </div>

          <div className="mt-6 space-y-3">
            {timelineEvents.map((event) => {
              const visible = snapshot.cases >= event.at;
              const active = activeEvent.title === event.title;
              return (
                <div
                  key={event.title}
                  className={`grid grid-cols-[64px_1fr] gap-3 border p-3 transition-colors ${
                    active
                      ? "border-slate-950 bg-slate-950 text-white"
                      : visible
                        ? "border-emerald-200 bg-emerald-50 text-slate-900"
                        : "border-slate-200 bg-slate-50 text-slate-400"
                  }`}
                >
                  <div className="text-xs font-black">{event.at}件</div>
                  <div>
                    <div className="text-sm font-black">{event.title}</div>
                    <div className={`mt-1 text-xs font-bold leading-5 ${active ? "text-slate-200" : "text-slate-500"}`}>
                      {event.tag}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-6 md:px-8 lg:grid-cols-2">
        <div className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <Brain className="h-5 w-5 text-slate-500" />
            <h2 className="text-xl font-black text-slate-950">Before: 一般的な審査AI</h2>
          </div>
          <p className="mt-3 min-h-12 text-sm font-bold leading-7 text-slate-600">
            初期状態では、紫苑も普通のリスク確認に近い。正しいが、案件の芯にはまだ届かない。
          </p>
          <div className="mt-4">
            <Checklist items={beforeChecks} />
          </div>
        </div>

        <div className={`border p-5 shadow-sm transition-colors ${afterUnlocked ? "border-emerald-300 bg-white" : "border-slate-200 bg-slate-100"}`}>
          <div className="flex items-center gap-3">
            <BookOpenCheck className={`h-5 w-5 ${afterUnlocked ? "text-emerald-700" : "text-slate-400"}`} />
            <h2 className="text-xl font-black text-slate-950">After: 判断資産が戻る紫苑</h2>
          </div>
          <p className="mt-3 min-h-12 text-sm font-bold leading-7 text-slate-600">
            人間の修正が蓄積すると、同じ案件でも最初に見る場所が変わる。
          </p>
          <div className={`mt-4 transition-opacity ${afterUnlocked ? "opacity-100" : "opacity-35"}`}>
            <Checklist items={afterChecks} strong />
          </div>
          {afterUnlocked && (
            <div className="mt-4 border border-emerald-200 bg-emerald-50 p-4 text-sm font-black leading-7 text-emerald-950">
              判断資産出典: JA-demo-asset-life / JA-demo-subsidy-cashflow / JA-demo-new-customer
            </div>
          )}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-10 md:px-8">
        <div className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <UserCheck className="h-5 w-5 text-sky-700" />
              <h2 className="text-lg font-black">非エンジニア起点の必然性</h2>
            </div>
            <p className="mt-3 text-sm font-bold leading-7 text-slate-600">
              コードのDevOpsではなく、現場で消えていく判断をDevOpsする。リース審査の痛みを知る人間が、AIに判断を代行させるのではなく、判断を教え続けるためのデモです。
            </p>
          </div>

          <div className={`border p-5 shadow-sm transition-colors ${complete ? "border-slate-950 bg-slate-950 text-white" : "border-slate-200 bg-white text-slate-950"}`}>
            <div className="flex items-center gap-3">
              <ShieldCheck className={`h-5 w-5 ${complete ? "text-amber-300" : "text-slate-500"}`} />
              <h2 className="text-lg font-black">締めの一言</h2>
            </div>
            <div className="mt-4 space-y-3">
              {finalLines.map((line) => (
                <p key={line} className={`text-xl font-black leading-9 ${complete ? "text-white" : "text-slate-400"}`}>
                  {line}
                </p>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
