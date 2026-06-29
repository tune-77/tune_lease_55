"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  BookOpenCheck,
  Brain,
  CheckCircle2,
  Cloud,
  Database,
  FileSearch,
  GitBranch,
  Loader2,
  MessageSquareText,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  ThumbsUp,
  TriangleAlert,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type CloudStatus = {
  status?: string;
  ready?: boolean;
  db?: {
    backend?: string;
    available?: boolean;
    database_url_configured?: boolean;
    local_db_exists?: boolean;
    error?: string;
  };
  gcs_vault?: {
    enabled?: boolean;
    bucket?: string;
    prefix?: string;
    local_dir?: string;
    local_dir_exists?: boolean;
    markdown_count?: number;
    latest_local_mtime?: number | null;
  };
  cloud_run?: {
    service?: string;
    revision?: string;
    configuration?: string;
  };
};

type DashboardStats = {
  analysis?: {
    closed_count?: number;
    avg_score_borrower?: number | null;
  };
  lease_news_reflection?: {
    knowledge_available?: boolean;
    knowledge_scope?: string;
    indexed_notes?: number;
    knowledge_source_count?: number;
    knowledge_sources?: string[];
    current_question?: string;
    thought_lines?: string[];
  };
  lease_news_focus?: {
    available?: boolean;
    note_date?: string;
    theme_summary?: string;
    focus_lines?: string[];
  };
};

type PromptFeedbackSummary = {
  summary?: {
    total?: number;
    pdca_rate?: number;
    response_changed_rate?: number;
  };
};

type JudgmentFeedbackSummary = {
  total?: number;
  approved?: number;
  needs_review?: number;
  by_source?: Record<string, number>;
};

type LoadState = {
  cloud?: CloudStatus;
  dashboard?: DashboardStats;
  prompt?: PromptFeedbackSummary;
  judgment?: JudgmentFeedbackSummary;
};

const fallback: Required<LoadState> = {
  cloud: {
    status: "degraded",
    ready: false,
    db: { backend: "unknown", available: false, database_url_configured: false },
    gcs_vault: {
      enabled: true,
      bucket: "tune-lease-55-data",
      prefix: "vault/",
      markdown_count: 0,
    },
    cloud_run: { service: "Cloud Run" },
  },
  dashboard: {
    analysis: { closed_count: 142, avg_score_borrower: 74.2 },
    lease_news_reflection: {
      knowledge_available: false,
      indexed_notes: 0,
      knowledge_source_count: 0,
      knowledge_scope: "Cloud Runから取得できる知識状態を確認中",
      knowledge_sources: [],
      current_question: "この案件で、過去の判断資産は何を変えるか",
      thought_lines: [
        "案件入力をただ採点せず、過去の稟議・ニュース・改善ログへ照会する。",
        "人が押したfeedbackを次回の回答改善候補として残す。",
      ],
    },
    lease_news_focus: {
      available: false,
      theme_summary: "リースニュースと審査論点を接続する準備中",
      focus_lines: [],
    },
  },
  prompt: { summary: { total: 0, pdca_rate: 0, response_changed_rate: 0 } },
  judgment: { total: 0, approved: 0, needs_review: 0, by_source: {} },
};

const demoCase = {
  company: "地域製造業の設備更新案件",
  asset: "CNC工作機械",
  amount: "3,800万円",
  concern: "新規先、粗利改善途上、補助金採択前",
};

function pct(value?: number | null) {
  if (value == null || Number.isNaN(value)) return "0%";
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(normalized)}%`;
}

function numberText(value?: number | null) {
  return new Intl.NumberFormat("ja-JP").format(value ?? 0);
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold ${
        ok
          ? "bg-emerald-100 text-emerald-800"
          : "bg-amber-100 text-amber-800"
      }`}
    >
      {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <TriangleAlert className="h-3.5 w-3.5" />}
      {label}
    </span>
  );
}

const stageBase =
  "min-h-[168px] border border-slate-200 bg-white p-4 shadow-sm transition-colors";

const boundaryLanes = [
  {
    title: "Local Obsidian",
    role: "判断資産の正本",
    body: "Daily note、Private Reflection、判断メモ、作業ログはローカル/iCloud Vaultへ保存する。",
    footer: "Private notes stay local",
    tone: "border-emerald-200 bg-emerald-50 text-emerald-950",
  },
  {
    title: "Cloud Run",
    role: "外部入口とデモ実行",
    body: "審査入力、AIチャット、feedbackを受け、Cloud SQL/GCSへappend-onlyで記録する。",
    footer: "Intake, not source of truth",
    tone: "border-sky-200 bg-sky-50 text-sky-950",
  },
  {
    title: "GCS Vault",
    role: "選抜知識コピー",
    body: "Cloud Runが読むMarkdownだけを同期する。Cloud SQL要約や入力ログは再アップロードしない。",
    footer: "Curated knowledge only",
    tone: "border-amber-200 bg-amber-50 text-amber-950",
  },
];

export default function KnowledgeLoopDemoPage() {
  const [data, setData] = useState<LoadState>({});
  const [loading, setLoading] = useState(true);
  const [selectedStage, setSelectedStage] = useState(1);

  useEffect(() => {
    let alive = true;
    async function load() {
      setLoading(true);
      const [cloudRes, dashboardRes, promptRes, judgmentRes] = await Promise.allSettled([
        apiClient.get<CloudStatus>("/api/system/cloud-status"),
        apiClient.get<DashboardStats>("/api/dashboard/stats"),
        apiClient.get<PromptFeedbackSummary>("/api/prompt-feedback/summary"),
        apiClient.get<JudgmentFeedbackSummary>("/api/judgment-feedback/summary"),
      ]);
      if (!alive) return;
      setData({
        cloud: cloudRes.status === "fulfilled" ? cloudRes.value.data : undefined,
        dashboard: dashboardRes.status === "fulfilled" ? dashboardRes.value.data : undefined,
        prompt: promptRes.status === "fulfilled" ? promptRes.value.data : undefined,
        judgment: judgmentRes.status === "fulfilled" ? judgmentRes.value.data : undefined,
      });
      setLoading(false);
    }
    load();
    return () => {
      alive = false;
    };
  }, []);

  const view = useMemo(
    () => ({
      cloud: data.cloud ?? fallback.cloud,
      dashboard: data.dashboard ?? fallback.dashboard,
      prompt: data.prompt ?? fallback.prompt,
      judgment: data.judgment ?? fallback.judgment,
    }),
    [data],
  );

  const reflection = view.dashboard.lease_news_reflection ?? fallback.dashboard.lease_news_reflection;
  const focus = view.dashboard.lease_news_focus ?? fallback.dashboard.lease_news_focus;
  const cloudReady = Boolean(view.cloud.ready);
  const gcsEnabled = Boolean(view.cloud.gcs_vault?.enabled);
  const markdownCount = view.cloud.gcs_vault?.markdown_count ?? reflection?.indexed_notes ?? 0;
  const sourceCount = reflection?.knowledge_source_count ?? 0;
  const feedbackTotal = (view.judgment.total ?? 0) + (view.prompt.summary?.total ?? 0);
  const loopHealth = cloudReady && gcsEnabled ? "正常" : cloudReady ? "一部確認中" : "確認中";
  const loopHealthDetail = cloudReady && gcsEnabled
    ? "Cloud Runと選抜知識コピーが接続されています"
    : "ローカル表示または一部APIの状態を確認しています";

  const stages = [
    {
      title: "案件入力",
      subtitle: "人の判断材料",
      Icon: MessageSquareText,
      tone: "border-sky-200 bg-sky-50",
      body: `${demoCase.company} / ${demoCase.asset} / ${demoCase.amount}`,
      detail: demoCase.concern,
    },
    {
      title: "知識照会",
      subtitle: "過去メモ・ニュース・判断基準",
      Icon: FileSearch,
      tone: "border-emerald-200 bg-emerald-50",
      body: `${numberText(markdownCount)}件のMarkdownスナップショット`,
      detail: sourceCount > 0 ? `直近回答で${sourceCount}件を参照` : "Cloud Run上のGCS Vault状態を表示",
    },
    {
      title: "審査判断",
      subtitle: "根拠つきコメント生成",
      Icon: Brain,
      tone: "border-indigo-200 bg-indigo-50",
      body: `平均スコア ${Math.round(view.dashboard.analysis?.avg_score_borrower ?? 74)}`,
      detail: `${numberText(view.dashboard.analysis?.closed_count ?? 0)}件の過去案件を背景に比較`,
    },
    {
      title: "Feedback",
      subtitle: "人の修正を保存",
      Icon: ThumbsUp,
      tone: "border-amber-200 bg-amber-50",
      body: `${numberText(feedbackTotal)}件の改善信号`,
      detail: `判断feedback ${numberText(view.judgment.total ?? 0)} / prompt改善 ${numberText(view.prompt.summary?.total ?? 0)}`,
    },
    {
      title: "次回へ反映",
      subtitle: "知識ループ継続",
      Icon: GitBranch,
      tone: "border-rose-200 bg-rose-50",
      body: `回答変化率 ${pct(view.prompt.summary?.response_changed_rate)}`,
      detail: "改善ログと判断履歴を次の案件文脈へ戻す",
    },
  ];

  const selected = stages[selectedStage];
  const selectedSources = (reflection?.knowledge_sources ?? []).slice(0, 4);
  const thoughtLines = (reflection?.thought_lines ?? []).slice(0, 3);
  const focusLines = (focus?.focus_lines ?? []).slice(0, 3);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-xs font-black text-white">
                  <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
                  Hackathon Demo
                </span>
                <StatusPill ok={cloudReady} label={cloudReady ? "Cloud Run ready" : "Cloud Run確認中"} />
                <StatusPill ok={gcsEnabled} label={gcsEnabled ? "GCS知識接続" : "ローカル知識表示"} />
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl">
                知識ループ確認
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
                紫苑が「案件入力 → 判断資産の参照 → 審査判断 → 人の修正 → 次回反映」を回せているかを見る画面です。
                まず上の状態を見て、次に5段階のどこで止まっているかを確認します。
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Link
                href="/lease-intelligence"
                className="inline-flex items-center gap-2 rounded-md bg-slate-950 px-4 py-2 text-sm font-bold text-white hover:bg-slate-800"
              >
                <Brain className="h-4 w-4" />
                紫苑と試す
              </Link>
              <Link
                href="/knowledge-space"
                className="inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-700 hover:bg-slate-100"
              >
                <BookOpenCheck className="h-4 w-4" />
                知識マップ
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
                  Current Loop Status
                </p>
                <h2 className="mt-2 text-2xl font-black text-slate-950">
                  {loopHealth}
                </h2>
                <p className="mt-1 text-sm leading-6 text-slate-600">{loopHealthDetail}</p>
              </div>
              {loading && (
                <span className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-bold text-slate-600">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  live data loading
                </span>
              )}
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <div className="border-l-4 border-emerald-400 bg-emerald-50 px-4 py-3">
                <p className="text-xs font-black text-emerald-900">1. 知識はあるか</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{numberText(markdownCount)}件</p>
                <p className="mt-1 text-xs text-slate-600">参照可能なMarkdown</p>
              </div>
              <div className="border-l-4 border-indigo-400 bg-indigo-50 px-4 py-3">
                <p className="text-xs font-black text-indigo-900">2. 使われたか</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{numberText(sourceCount)}件</p>
                <p className="mt-1 text-xs text-slate-600">直近回答の参照数</p>
              </div>
              <div className="border-l-4 border-amber-400 bg-amber-50 px-4 py-3">
                <p className="text-xs font-black text-amber-900">3. 戻せるか</p>
                <p className="mt-1 text-2xl font-black text-slate-950">{numberText(feedbackTotal)}件</p>
                <p className="mt-1 text-xs text-slate-600">feedback信号</p>
              </div>
            </div>
          </div>
          <div className="border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
              Data Connection
            </p>
            <div className="mt-4 grid gap-3 text-sm">
              <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-3">
                <span className="font-bold text-slate-600">DB</span>
                <span className="font-black text-slate-950">{view.cloud.db?.backend ?? "unknown"}</span>
              </div>
              <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-3">
                <span className="font-bold text-slate-600">Cloud Run</span>
                <span className="font-black text-slate-950">{cloudReady ? "ready" : "checking"}</span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="font-bold text-slate-600">GCS Vault</span>
                <span className="font-black text-slate-950">{gcsEnabled ? "connected" : "local"}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-8 sm:px-6 lg:px-8">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">Loop Flow</p>
            <h2 className="mt-1 text-xl font-black text-slate-950">どこで止まっているかを見る</h2>
          </div>
          <p className="hidden text-xs font-bold text-slate-500 md:block">クリックすると下に根拠が出ます</p>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {stages.map((stage, index) => {
            const Icon = stage.Icon;
            const active = selectedStage === index;
            return (
              <button
                key={stage.title}
                type="button"
                onClick={() => setSelectedStage(index)}
                className={`${stageBase} ${stage.tone} text-left ${
                  active ? "ring-2 ring-slate-950" : "hover:border-slate-400"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
                      Step {index + 1}
                    </p>
                    <h2 className="mt-2 text-lg font-black text-slate-950">{stage.title}</h2>
                    <p className="mt-1 text-xs font-bold text-slate-500">{stage.subtitle}</p>
                  </div>
                  <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-white text-slate-800 shadow-sm">
                    <Icon className="h-5 w-5" />
                  </span>
                </div>
                <p className="mt-5 text-sm font-bold leading-6 text-slate-900">{stage.body}</p>
                <p className="mt-2 text-xs leading-5 text-slate-600">{stage.detail}</p>
              </button>
            );
          })}
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-4 pb-10 sm:px-6 lg:grid-cols-[1.2fr_0.8fr] lg:px-8">
        <div className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
                Selected Step
              </p>
              <h2 className="mt-2 text-2xl font-black text-slate-950">{selected.title}</h2>
              <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-600">{selected.detail}</p>
            </div>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-xs font-black text-slate-500">
                <Cloud className="h-4 w-4 text-sky-600" />
                Cloud Run
              </div>
              <p className="mt-3 text-sm font-bold text-slate-900">
                {view.cloud.cloud_run?.service || "service未取得"}
              </p>
              <p className="mt-1 break-all text-xs leading-5 text-slate-500">
                {view.cloud.cloud_run?.revision || "revisionは実環境で表示"}
              </p>
            </div>
            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-xs font-black text-slate-500">
                <Database className="h-4 w-4 text-emerald-600" />
                Knowledge Store
              </div>
              <p className="mt-3 text-sm font-bold text-slate-900">
                gs://{view.cloud.gcs_vault?.bucket || "tune-lease-55-data"}/{view.cloud.gcs_vault?.prefix || "vault/"}
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                {view.cloud.gcs_vault?.local_dir || "/tmp/gcs_vault"} に同期
              </p>
            </div>
            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-xs font-black text-slate-500">
                <ShieldCheck className="h-4 w-4 text-rose-600" />
                Writeback
              </div>
              <p className="mt-3 text-sm font-bold text-slate-900">
                append-only event log
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                Cloud Run入力はGCSへ追記し、ローカル夜間処理が正本へ反映
              </p>
            </div>
          </div>

          <div className="mt-6 border-t border-slate-200 pt-5">
            <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
              Demo Case
            </p>
            <div className="mt-3 grid gap-3 md:grid-cols-4">
              {[
                ["取引先", demoCase.company],
                ["物件", demoCase.asset],
                ["金額", demoCase.amount],
                ["論点", demoCase.concern],
              ].map(([label, value]) => (
                <div key={label} className="border border-slate-200 bg-white p-3">
                  <p className="text-[11px] font-black text-slate-500">{label}</p>
                  <p className="mt-1 text-sm font-bold leading-6 text-slate-900">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <aside className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">
                Evidence
              </p>
              <h2 className="mt-2 text-xl font-black text-slate-950">確認すべき根拠</h2>
            </div>
            <Activity className="h-6 w-6 text-slate-500" />
          </div>

          <div className="mt-5 space-y-4">
            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2">
                <RefreshCw className="h-4 w-4 text-cyan-700" />
                <p className="text-sm font-black text-slate-900">今日の論点</p>
              </div>
              <p className="mt-2 text-xs leading-6 text-slate-600">
                {focus?.theme_summary || reflection?.current_question || "審査論点を読み込み中"}
              </p>
              {focusLines.length > 0 && (
                <div className="mt-3 space-y-2">
                  {focusLines.map((line) => (
                    <p key={line} className="border-l-2 border-cyan-400 pl-3 text-xs leading-5 text-slate-700">
                      {line}
                    </p>
                  ))}
                </div>
              )}
            </div>

            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2">
                <BookOpenCheck className="h-4 w-4 text-emerald-700" />
                <p className="text-sm font-black text-slate-900">参照された判断資産</p>
              </div>
              {selectedSources.length > 0 ? (
                <div className="mt-3 space-y-2">
                  {selectedSources.map((source) => (
                    <p key={source} className="truncate text-xs font-bold text-emerald-800">
                      {source}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="mt-2 text-xs leading-6 text-slate-600">
                  GCS Vaultの件数は表示済み。回答単位の参照元は、紫苑チャット実行後にここへ出ます。
                </p>
              )}
            </div>

            <div className="border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-indigo-700" />
                <p className="text-sm font-black text-slate-900">紫苑の内省</p>
              </div>
              <div className="mt-3 space-y-2">
                {(thoughtLines.length
                  ? thoughtLines
                  : fallback.dashboard.lease_news_reflection?.thought_lines ?? []
                ).map((line) => (
                  <p key={line} className="text-xs leading-6 text-slate-700">
                    {line}
                  </p>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </section>

      <section className="border-y border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="grid gap-6 lg:grid-cols-[0.74fr_1.26fr] lg:items-start">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-xs font-black text-white">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-300" />
                Security Boundary
              </div>
              <h2 className="mt-4 text-2xl font-black tracking-tight text-slate-950">
                Obsidian正本とクラウド実行を分ける
              </h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">
                金融・審査ナレッジでは、全部をクラウドRAGへ流さないことが価値になります。Cloud Runは入力を受け、ローカルMacが夜に要約・機密除去・知識化し、公開してよい知識だけをGCS Vaultへ戻します。
              </p>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              {boundaryLanes.map((lane) => (
                <div key={lane.title} className={`border p-4 ${lane.tone}`}>
                  <p className="text-[11px] font-black uppercase tracking-widest opacity-70">{lane.title}</p>
                  <h3 className="mt-2 text-base font-black">{lane.role}</h3>
                  <p className="mt-3 min-h-20 text-xs leading-6">{lane.body}</p>
                  <p className="mt-4 border-t border-current/20 pt-3 text-[11px] font-black uppercase tracking-widest opacity-70">
                    {lane.footer}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
