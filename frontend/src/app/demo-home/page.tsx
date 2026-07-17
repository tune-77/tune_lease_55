"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  BookOpenCheck,
  Brain,
  Check,
  Cloud,
  Database,
  FileText,
  GitBranch,
  HeartHandshake,
  Home,
  MessageSquareText,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  ThumbsUp,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type CloudStatus = {
  ready?: boolean;
  db?: { backend?: string; available?: boolean };
  gcs_vault?: { enabled?: boolean; bucket?: string; markdown_count?: number };
};

type DashboardStats = {
  analysis?: { closed_count?: number; avg_score_borrower?: number | null };
  lease_news_reflection?: {
    indexed_notes?: number;
    knowledge_source_count?: number;
    knowledge_sources?: string[];
    current_question?: string;
  };
  lease_news_focus?: {
    theme_summary?: string;
    focus_lines?: string[];
  };
};

type PromptFeedbackSummary = {
  summary?: { total?: number; response_changed_rate?: number };
};

type JudgmentFeedbackSummary = {
  total?: number;
  approved?: number;
  needs_review?: number;
};

type LiveData = {
  cloud?: CloudStatus;
  dashboard?: DashboardStats;
  prompt?: PromptFeedbackSummary;
  judgment?: JudgmentFeedbackSummary;
};

const fallback: Required<LiveData> = {
  cloud: {
    ready: false,
    db: { backend: "Cloud SQL", available: false },
    gcs_vault: { enabled: true, bucket: "tune-lease-55-data", markdown_count: 0 },
  },
  dashboard: {
    analysis: { closed_count: 142, avg_score_borrower: 74.2 },
    lease_news_reflection: {
      indexed_notes: 0,
      knowledge_source_count: 0,
      knowledge_sources: [],
      current_question: "この案件で、過去の判断資産は何を変えるか",
    },
    lease_news_focus: {
      theme_summary: "工作機械・補助金・更新投資の論点を案件判断へ戻す。",
      focus_lines: [
        "補助金は採択前提にしすぎず、未採択時の返済余力を見る。",
        "更新設備は既存工程の稼働率と粗利改善の見込みを確認する。",
        "新規先は代表者説明と資金繰り資料の整合性を重く見る。",
      ],
    },
  },
  prompt: { summary: { total: 0, response_changed_rate: 0 } },
  judgment: { total: 0, approved: 0, needs_review: 0 },
};

const loopSteps = [
  {
    key: "input",
    label: "審査入力",
    body: "企業情報・物件・営業メモを入れる",
    icon: MessageSquareText,
  },
  {
    key: "recall",
    label: "記憶照会",
    body: "過去判断・Research・判断資産を呼び戻す",
    icon: BookOpenCheck,
  },
  {
    key: "judge",
    label: "紫苑レビュー",
    body: "確認質問・承認条件・反証を出す",
    icon: Brain,
  },
  {
    key: "feedback",
    label: "人間評価",
    body: "役に立った / 要修正 / 違うを返す",
    icon: ThumbsUp,
  },
  {
    key: "learn",
    label: "検疫・昇格",
    body: "承認済みだけ次回判断へ戻す",
    icon: GitBranch,
  },
];

const decisionLines = [
  "Cloud Run版は入力受付とデモ実行に限定し、本体DBへ直接昇格しない。",
  "ローカル紫苑側で検疫・人間レビュー・昇格処理を行う。",
  "承認済みの判断だけを、次の紫苑レビューと審査判断へ戻す。",
];

function n(value?: number | null) {
  return new Intl.NumberFormat("ja-JP").format(value ?? 0);
}

function pct(value?: number | null) {
  if (value == null || Number.isNaN(value)) return "0%";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

function InfoStrip({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="border-t border-stone-200 px-4 py-4 sm:border-l sm:border-t-0 first:border-l-0">
      <p className="text-[11px] font-black uppercase tracking-widest text-stone-500">{label}</p>
      <p className="mt-1 text-2xl font-black text-stone-950">{value}</p>
      <p className="mt-1 text-xs leading-5 text-stone-500">{detail}</p>
    </div>
  );
}

function WarmPill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/25 bg-white/15 px-3 py-1 text-xs font-bold text-white shadow-sm backdrop-blur-md">
      {children}
    </span>
  );
}

export default function DemoHomePage() {
  const [data, setData] = useState<LiveData>({});

  useEffect(() => {
    let alive = true;
    async function load() {
      const [cloud, dashboard, prompt, judgment] = await Promise.allSettled([
        apiClient.get<CloudStatus>("/api/system/cloud-status"),
        apiClient.get<DashboardStats>("/api/dashboard/stats"),
        apiClient.get<PromptFeedbackSummary>("/api/prompt-feedback/summary"),
        apiClient.get<JudgmentFeedbackSummary>("/api/judgment-feedback/summary"),
      ]);
      if (!alive) return;
      setData({
        cloud: cloud.status === "fulfilled" ? cloud.value.data : undefined,
        dashboard: dashboard.status === "fulfilled" ? dashboard.value.data : undefined,
        prompt: prompt.status === "fulfilled" ? prompt.value.data : undefined,
        judgment: judgment.status === "fulfilled" ? judgment.value.data : undefined,
      });
    }
    load();
    return () => {
      alive = false;
    };
  }, []);

  const live = useMemo(
    () => ({
      cloud: data.cloud ?? fallback.cloud,
      dashboard: data.dashboard ?? fallback.dashboard,
      prompt: data.prompt ?? fallback.prompt,
      judgment: data.judgment ?? fallback.judgment,
    }),
    [data],
  );

  const reflection = live.dashboard.lease_news_reflection ?? fallback.dashboard.lease_news_reflection;
  const focus = live.dashboard.lease_news_focus ?? fallback.dashboard.lease_news_focus;
  const notes = live.cloud.gcs_vault?.markdown_count || reflection?.indexed_notes || 0;
  const referenced = reflection?.knowledge_source_count || 0;
  const feedback = (live.prompt.summary?.total || 0) + (live.judgment.total || 0);
  const sources = (reflection?.knowledge_sources || []).slice(0, 3);

  return (
    <main className="min-h-screen bg-[#f7f4ee] text-stone-950">
      <section className="relative min-h-[92vh] overflow-hidden">
        <img
          src="/lease-grumble/characters/lease-intelligence-girl.jpg"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
        />
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(18,18,16,0.86),rgba(18,18,16,0.56)_42%,rgba(18,18,16,0.22)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_22%_28%,rgba(244,214,159,0.28),transparent_34%),radial-gradient(circle_at_70%_18%,rgba(20,184,166,0.16),transparent_30%)]" />

        <div className="relative z-10 mx-auto flex min-h-[92vh] max-w-7xl flex-col px-4 py-5 sm:px-6 lg:px-8">
          <header className="flex items-center justify-between gap-4">
            <Link href="/demo" className="inline-flex items-center gap-2 text-sm font-black text-white">
              <span className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-white/15 backdrop-blur">
                <Home className="h-4 w-4" />
              </span>
              AURION / SHION
            </Link>
            <div className="hidden items-center gap-2 md:flex">
              <WarmPill>
                <Cloud className="h-3.5 w-3.5" />
                {live.cloud.ready ? "Cloud Run ready" : "Cloud Run standby"}
              </WarmPill>
              <WarmPill>
                <Database className="h-3.5 w-3.5" />
                {live.cloud.db?.backend || "DB"}
              </WarmPill>
            </div>
          </header>

          <div className="grid flex-1 items-center gap-10 py-10 lg:grid-cols-[0.92fr_1.08fr]">
            <div className="max-w-3xl">
              <div className="mb-5 flex flex-wrap gap-2">
                <WarmPill>
                  <Sparkles className="h-3.5 w-3.5 text-[#f7d78b]" />
                  DevOps × AI Agent
                </WarmPill>
                <WarmPill>
                  <ShieldCheck className="h-3.5 w-3.5 text-emerald-200" />
                  判断資産を再利用
                </WarmPill>
              </div>
              <p className="text-xs font-black uppercase tracking-[0.34em] text-[#f7d78b]">
                Lease Intelligence
              </p>
              <h1 className="mt-4 text-5xl font-black leading-[0.98] tracking-tight text-white sm:text-6xl lg:text-7xl">
                紫苑
              </h1>
              <p className="mt-4 max-w-xl text-2xl font-black leading-tight text-white sm:text-3xl">
                判断を記憶し、検疫して、次の審査へ戻すAI。
              </p>
              <p className="mt-5 max-w-2xl text-base leading-8 text-stone-100">
                案件、過去メモ、ニュース、人間評価を判断資産へ変換します。Cloud Runは実証フィールド、ローカル紫苑は検疫・昇格・本体保護を担う運用基盤です。
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/screening"
                  className="inline-flex items-center gap-2 rounded-md bg-white px-5 py-3 text-sm font-black text-stone-950 shadow-xl shadow-black/20 transition hover:bg-[#fff7e6]"
                >
                  <FileText className="h-4 w-4" />
                  審査・分析を開く
                </Link>
                <Link
                  href="/shion-memory-system"
                  className="inline-flex items-center gap-2 rounded-md border border-[#f7d78b]/60 bg-[#f7d78b]/20 px-5 py-3 text-sm font-black text-white backdrop-blur transition hover:bg-[#f7d78b]/30"
                >
                  記憶システムを見る
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  href="/system-overview"
                  className="inline-flex items-center gap-2 rounded-md border border-white/30 bg-white/10 px-5 py-3 text-sm font-black text-white backdrop-blur transition hover:bg-white/20"
                >
                  検疫・昇格を見る
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  href="/improvement-log"
                  className="inline-flex items-center gap-2 rounded-md border border-orange-200/60 bg-orange-300/20 px-5 py-3 text-sm font-black text-white backdrop-blur transition hover:bg-orange-300/30"
                >
                  改善PMレポート
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </div>

            <div className="rounded-[28px] border border-white/20 bg-white/90 p-4 shadow-2xl shadow-black/30 backdrop-blur-md">
              <div className="rounded-[20px] border border-stone-200 bg-[#fffdf8] p-5">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-[11px] font-black uppercase tracking-widest text-stone-500">Main Demo Route</p>
                    <h2 className="mt-2 text-2xl font-black text-stone-950">審査から判断資産化まで</h2>
                    <p className="mt-1 text-sm font-bold text-stone-500">Cloud Run実証フィールド + ローカル検疫昇格</p>
                  </div>
                  <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-black text-emerald-800">
                    <ShieldCheck className="h-3.5 w-3.5" />
                    本体DB保護
                  </span>
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-3">
                  {[
                    ["Cloud Run", "入力受付"],
                    ["ローカル", "検疫・昇格"],
                    ["記憶", "次回判断へ戻す"],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-lg border border-stone-200 bg-white px-3 py-3">
                      <p className="text-[11px] font-black text-stone-500">{label}</p>
                      <p className="mt-1 text-sm font-black text-stone-950">{value}</p>
                    </div>
                  ))}
                </div>

                <p className="mt-4 rounded-lg bg-[#f6efe3] px-4 py-3 text-sm leading-7 text-stone-700">
                  審査入力で紫苑レビューを作り、人間評価を残し、ローカル側で検疫してから承認済みの判断だけを次の案件へ戻します。
                </p>

                <div className="mt-5 grid gap-2 sm:grid-cols-5">
                  {loopSteps.map((step, index) => {
                    const Icon = step.icon;
                    return (
                      <div
                        key={step.key}
                        className="min-h-28 rounded-lg border border-stone-200 bg-white px-2 py-3 text-center text-stone-800"
                      >
                        <Icon className="mx-auto h-5 w-5 text-teal-700" />
                        <span className="mt-2 block text-xs font-black">{index + 1}. {step.label}</span>
                        <span className="mt-2 block text-[11px] font-bold leading-5 text-stone-500">{step.body}</span>
                      </div>
                    );
                  })}
                </div>

                <div className="mt-5 rounded-xl border border-stone-200 bg-white p-4">
                  <div className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-teal-700" />
                    <p className="text-sm font-black text-stone-950">紫苑の判断メモ</p>
                  </div>
                  <div className="mt-3 space-y-2">
                    {decisionLines.map((line) => (
                      <p key={line} className="flex gap-2 text-sm leading-7 text-stone-700">
                        <Check className="mt-1 h-4 w-4 shrink-0 text-emerald-700" />
                        {line}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="pb-2 text-center text-xs font-bold text-white/70">
            scroll
          </div>
        </div>
      </section>

      <section className="border-b border-stone-200 bg-white">
        <div className="mx-auto grid max-w-7xl px-4 sm:grid-cols-4 sm:px-6 lg:px-8">
          <InfoStrip label="Knowledge" value={`${n(notes)}件`} detail="GCS Vault / Markdown snapshot" />
          <InfoStrip label="Referenced" value={`${n(referenced)}件`} detail="直近の回答で想起された判断資産" />
          <InfoStrip label="Feedback" value={`${n(feedback)}件`} detail="人の修正とprompt改善の信号" />
          <InfoStrip label="Response Change" value={pct(live.prompt.summary?.response_changed_rate)} detail="改善が回答へ効いた比率" />
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-4 py-10 sm:px-6 lg:grid-cols-[1fr_1fr_1fr] lg:px-8">
        <div className="rounded-xl border border-stone-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <BookOpenCheck className="h-5 w-5 text-teal-700" />
            <h2 className="text-lg font-black text-stone-950">参照した判断資産</h2>
          </div>
          <div className="mt-4 space-y-3">
            {sources.length > 0 ? (
              sources.map((source) => (
                <p key={source} className="truncate rounded-md bg-teal-50 px-3 py-2 text-sm font-bold text-teal-900">
                  {source}
                </p>
              ))
            ) : (
              <p className="text-sm leading-7 text-stone-600">
                Cloud Runに載ったGCS知識数と、チャット実行後の参照元がここに出ます。
              </p>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-stone-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-5 w-5 text-rose-700" />
            <h2 className="text-lg font-black text-stone-950">今日の論点</h2>
          </div>
          <p className="mt-4 text-sm font-bold leading-7 text-stone-800">
            {focus?.theme_summary || reflection?.current_question}
          </p>
          <div className="mt-4 space-y-2">
            {(focus?.focus_lines || fallback.dashboard.lease_news_focus?.focus_lines || []).slice(0, 3).map((line) => (
              <p key={line} className="border-l-2 border-rose-300 pl-3 text-xs leading-6 text-stone-600">
                {line}
              </p>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-stone-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <HeartHandshake className="h-5 w-5 text-amber-700" />
            <h2 className="text-lg font-black text-stone-950">人の判断が戻る場所</h2>
          </div>
          <div className="mt-4 space-y-3">
            {[
              ["判断feedback", `${n(live.judgment.total)}件`],
              ["承認済み候補", `${n(live.judgment.approved)}件`],
              ["要レビュー", `${n(live.judgment.needs_review)}件`],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between border-b border-stone-100 pb-2 last:border-b-0">
                <span className="text-sm font-bold text-stone-600">{label}</span>
                <span className="text-sm font-black text-stone-950">{value}</span>
              </div>
            ))}
          </div>
          <Link
            href="/demo/knowledge-loop"
            className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-md bg-stone-950 px-4 py-3 text-sm font-black text-white hover:bg-stone-800"
          >
            ループを詳しく見る
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      <section className="bg-[#171512] px-4 py-10 text-white sm:px-6 lg:px-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-black uppercase tracking-[0.28em] text-[#f7d78b]">Next action</p>
            <h2 className="mt-2 text-2xl font-black">デモ本番は、この入口から始める。</h2>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link href="/screening" className="inline-flex items-center gap-2 rounded-md bg-white px-4 py-3 text-sm font-black text-stone-950">
              <FileText className="h-4 w-4" />
              実案件を審査
            </Link>
            <Link href="/system-overview" className="inline-flex items-center gap-2 rounded-md border border-white/20 px-4 py-3 text-sm font-black text-white">
              <ArrowRight className="h-4 w-4" />
              システム全体
            </Link>
            <Link href="/improvement-log" className="inline-flex items-center gap-2 rounded-md border border-orange-200/40 px-4 py-3 text-sm font-black text-white">
              <ArrowRight className="h-4 w-4" />
              改善PMレポート
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
