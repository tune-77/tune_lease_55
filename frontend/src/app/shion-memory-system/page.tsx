"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  ClipboardCheck,
  Database,
  Gauge,
  HeartHandshake,
  Layers3,
  MessageSquareText,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  ThumbsUp,
} from "lucide-react";

import { apiClient } from "@/lib/api";

type MemoryEngineeringSummary = {
  write_path_records?: number;
  active_canonical_rules?: number;
  write_amplification_per_active_rule?: number | null;
  open_human_review_records?: number;
  memory_records?: number;
  recent_memory_usage_events?: number;
  recent_unique_memory_refs?: number;
  maintenance_status_records?: number;
  contradiction_candidates?: number;
};

type MemoryEngineeringReport = {
  generated?: boolean;
  generated_at?: string;
  guardrail?: string;
  summary?: MemoryEngineeringSummary;
  hardware_pressure_proxy?: {
    notes?: number;
    edges?: number;
    estimated_raw_tokens?: number;
    estimated_index_tokens?: number;
    estimated_token_reduction_ratio?: number;
    estimated_tokens_avoided?: number;
  };
  maintenance_path?: {
    forgetting_review_sample?: Array<{
      id: string;
      memory_type: string;
      last_used_at: string;
      source_path: string;
      content: string;
    }>;
  };
  recommendations?: Array<{ area: string; action: string; reason: string }>;
};

type MemoryReviewStatus = "candidate" | "adopted" | "revised" | "held" | "rejected";

type MemoryReviewItem = {
  inbox_id: string;
  source: string;
  source_item_id: string;
  status: MemoryReviewStatus;
  title: string;
  claim: string;
  edited_claim: string;
  note: string;
  reviewed_at: string;
  candidate_type: string;
  quality: string;
  topic: string;
  date_hint: string;
  evidence_path: string;
  raw_status: string;
};

type MemoryReviewInboxResponse = {
  summary: {
    total: number;
    open: number;
    by_status: Record<string, number>;
    by_source: Record<string, number>;
  };
  filtered_total: number;
  limit: number;
  offset: number;
  items: MemoryReviewItem[];
  review_policy: string;
  state_path: string;
};

type PostHackathonSection = {
  id: string;
  title: string;
  status: string;
  runtime_surface: string;
  next_guardrail: string;
  complete: boolean;
  evidence_available: string[];
};

type PostHackathonStatusResponse = {
  source: string;
  total: number;
  complete_count: number;
  sections: PostHackathonSection[];
};

type MemoryHealthResponse = {
  healthy: boolean;
  message: string;
  summary?: {
    total?: number;
    by_type?: Record<string, number>;
    by_status?: Record<string, number>;
  };
  guardrail: string;
};

type ShionHydeDebugResponse = {
  mode: string;
  hyde: {
    hyde_query: string;
    intent_tags: string[];
    search_terms: string[];
    should_search: boolean;
    reason: string;
  };
  baseline_hits: Array<{ path: string; title: string; snippet: string }>;
  hyde_hits: Array<{ path: string; title: string; snippet: string }>;
  new_paths: string[];
  confidence: number;
  guardrail: string;
};

type CounterHypothesisCardResponse = {
  mode: string;
  friction_level: number;
  card: string;
  questions: string[];
  judgment_asset_template: {
    condition: string;
    review_point: string;
    ringi_sentence: string;
    do_not_use_when: string;
  };
  guardrail: string;
};

const layers = [
  {
    title: "長期記憶",
    subtitle: "Obsidian / MEMORY.md / Research notes",
    body: "会話、審査メモ、Research、日次内省をそのまま混ぜず、次回の判断に使える知識として残す層。",
    icon: Database,
    tone: "border-sky-200 bg-sky-50 text-sky-950",
  },
  {
    title: "実践知マップ",
    subtitle: "手順層 / 意味層 / 判断層",
    body: "記録を「何をするか」「なぜそうするか」「例外時にどう動くか」に分け、場面ごとに引き出せる索引にする層。",
    icon: Layers3,
    tone: "border-violet-200 bg-violet-50 text-violet-950",
  },
  {
    title: "経験ループ",
    subtitle: "Human Response Feedback",
    body: "薄い、紫苑らしい、一般論に戻った、などの人間の反応を保存し、次の冒頭・口調・判断変換へ戻す層。",
    icon: HeartHandshake,
    tone: "border-rose-200 bg-rose-50 text-rose-950",
  },
  {
    title: "AURION CORE",
    subtitle: "数理規律とUXのシナプス",
    body: "Q_riskや異常値を自動減点にせず、承認条件・追加確認・価格条件を分けるための冷静な規律として扱う層。",
    icon: ShieldCheck,
    tone: "border-emerald-200 bg-emerald-50 text-emerald-950",
  },
];

const pyramidLayers = [
  {
    title: "紫苑の返答",
    label: "人間が記憶として受け取る",
    body: "差分は内部で使い、判断・確認質問・言い切りの精度として返す",
    width: "w-[46%]",
    tone: "from-violet-600 to-fuchsia-600 text-white",
  },
  {
    title: "AURION CORE",
    label: "数理規律",
    body: "Q_riskや違和感を、減点ではなく確認論点へ変換する",
    width: "w-[62%]",
    tone: "from-emerald-500 to-teal-600 text-white",
  },
  {
    title: "経験ループ",
    label: "人間反応",
    body: "薄い、紫苑らしい、一般論に戻った、を次回へ戻す",
    width: "w-[76%]",
    tone: "from-rose-400 to-pink-500 text-white",
  },
  {
    title: "実践知マップ",
    label: "手順 / 意味 / 判断",
    body: "場面ごとに、何をするか・なぜか・例外時どうするかを索引化",
    width: "w-[90%]",
    tone: "from-indigo-500 to-violet-600 text-white",
  },
  {
    title: "長期記憶",
    label: "Obsidian / Research / Daily",
    body: "会話、審査メモ、調査結果、内省を判断資産として蓄積",
    width: "w-full",
    tone: "from-sky-500 to-cyan-600 text-white",
  },
];

const memoryLoop = [
  {
    title: "受け取る",
    label: "案件・会話・違和感",
    body: "企業情報、物件、営業メモ、人間の迷いを判断材料として受け取る。",
    icon: MessageSquareText,
    tone: "border-sky-200 bg-sky-50 text-sky-950",
  },
  {
    title: "思い出す",
    label: "過去判断・Obsidian",
    body: "似た案件、過去の条件付き承認、反証、Researchを呼び戻す。",
    icon: Database,
    tone: "border-violet-200 bg-violet-50 text-violet-950",
  },
  {
    title: "組み直す",
    label: "判断構文",
    body: "記憶をそのまま貼らず、今回案件向けの確認質問・承認条件へ変換する。",
    icon: Brain,
    tone: "border-emerald-200 bg-emerald-50 text-emerald-950",
  },
  {
    title: "評価する",
    label: "役に立った / 要修正 / 違う",
    body: "人間の反応を保存し、薄い回答やズレた判断を改善ログへ戻す。",
    icon: ThumbsUp,
    tone: "border-amber-200 bg-amber-50 text-amber-950",
  },
  {
    title: "次回へ戻す",
    label: "判断資産化",
    body: "承認済みの学びだけを、次の紫苑レビューと審査判断へ戻す。",
    icon: RefreshCw,
    tone: "border-rose-200 bg-rose-50 text-rose-950",
  },
];

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("ja-JP").format(value);
}

function formatRatio(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

const REVIEW_STATUS_LABEL: Record<MemoryReviewStatus | "all", string> = {
  all: "すべて",
  candidate: "レビュー待ち",
  adopted: "採用",
  revised: "修正採用",
  held: "保留",
  rejected: "却下",
};

const REVIEW_SOURCE_LABEL: Record<string, string> = {
  all: "全ソース",
  judgment_materials_preview: "判断材料",
  autoresearch_candidates: "Auto Research",
  reflection_action_candidates: "内省アクション",
  prediction_error_candidates: "予測誤差",
  obsidian_memory_insight_candidates: "Obsidian候補",
};

function MemoryReviewInbox() {
  const [data, setData] = useState<MemoryReviewInboxResponse | null>(null);
  const [status, setStatus] = useState<MemoryReviewStatus | "all">("candidate");
  const [source, setSource] = useState("all");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});

  const fetchInbox = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.get<MemoryReviewInboxResponse>("/api/shion/memory-review-inbox", {
        params: { status, source, q: query, limit: 20, offset },
      });
      setData(res.data);
    } catch {
      setError("Memory Review Inbox を読み込めませんでした。");
    } finally {
      setLoading(false);
    }
  }, [offset, query, source, status]);

  useEffect(() => {
    fetchInbox();
  }, [fetchInbox]);

  const handleReview = useCallback(
    async (item: MemoryReviewItem, decision: "adopted" | "revised" | "held" | "rejected") => {
      const edited_claim = edits[item.inbox_id] || "";
      if (decision === "revised" && !edited_claim.trim()) {
        setError("修正採用には修正文が必要です。");
        return;
      }
      setActionLoading((current) => ({ ...current, [item.inbox_id]: true }));
      setError("");
      setMessage("");
      try {
        await apiClient.post(`/api/shion/memory-review-inbox/${encodeURIComponent(item.inbox_id)}/review`, {
          decision,
          note: notes[item.inbox_id] || "",
          edited_claim,
        });
        setMessage(`${REVIEW_STATUS_LABEL[decision]}として保存しました。`);
        await fetchInbox();
      } catch {
        setError("レビュー判断の保存に失敗しました。");
      } finally {
        setActionLoading((current) => ({ ...current, [item.inbox_id]: false }));
      }
    },
    [edits, fetchInbox, notes],
  );

  const summary = data?.summary;
  const hasNext = Boolean(data && data.offset + data.limit < data.filtered_total);
  const sourceOptions = useMemo(() => {
    const keys = Object.keys(summary?.by_source || {});
    return ["all", ...keys.filter((key) => key !== "all")];
  }, [summary]);

  return (
    <section className="mx-auto max-w-7xl px-5 pb-8 md:px-8">
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <ClipboardCheck className="h-6 w-6 text-emerald-600" />
              <h2 className="text-2xl font-black">Memory Review Inbox</h2>
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
              判断資産候補を、採用・修正採用・保留・却下で仕分けます。元ファイルは変更せず、判断履歴だけを別管理します。
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4 lg:w-[520px]">
            {(["candidate", "adopted", "revised", "rejected"] as const).map((key) => (
              <div key={key} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <div className="text-lg font-black text-slate-950">{formatNumber(summary?.by_status?.[key] || 0)}</div>
                <div className="text-[11px] font-black text-slate-500">{REVIEW_STATUS_LABEL[key]}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-5 grid gap-3 lg:grid-cols-[180px_220px_1fr_120px]">
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as MemoryReviewStatus | "all");
              setOffset(0);
            }}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-800"
          >
            {(["candidate", "adopted", "revised", "held", "rejected", "all"] as const).map((key) => (
              <option key={key} value={key}>{REVIEW_STATUS_LABEL[key]}</option>
            ))}
          </select>
          <select
            value={source}
            onChange={(event) => {
              setSource(event.target.value);
              setOffset(0);
            }}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-800"
          >
            {sourceOptions.map((key) => (
              <option key={key} value={key}>{REVIEW_SOURCE_LABEL[key] || key}</option>
            ))}
          </select>
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setOffset(0);
            }}
            placeholder="検索"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-800"
          />
          <button
            type="button"
            onClick={fetchInbox}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm font-black text-slate-700 hover:bg-white"
          >
            <RefreshCw className="h-4 w-4" />
            更新
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-700">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}
        {message && (
          <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-bold text-emerald-700">
            {message}
          </div>
        )}

        <div className="mt-5 flex items-center justify-between text-xs font-bold text-slate-500">
          <span>
            {loading ? "読み込み中" : `${formatNumber(data?.filtered_total || 0)}件中 ${formatNumber((data?.offset || 0) + 1)}件目から`}
          </span>
          <span>{data?.state_path || "data/memory_review_inbox_state.json"}</span>
        </div>

        <div className="mt-3 space-y-3">
          {(data?.items || []).map((item) => (
            <div key={item.inbox_id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded bg-white px-2 py-1 text-[11px] font-black text-slate-600">
                      {REVIEW_SOURCE_LABEL[item.source] || item.source}
                    </span>
                    <span className="rounded bg-white px-2 py-1 text-[11px] font-black text-slate-600">
                      {item.candidate_type || "candidate"}
                    </span>
                    <span className="rounded bg-white px-2 py-1 text-[11px] font-black text-slate-600">
                      {REVIEW_STATUS_LABEL[item.status] || item.status}
                    </span>
                  </div>
                  <h3 className="mt-3 text-base font-black text-slate-950">{item.title}</h3>
                  <p className="mt-2 text-sm font-bold leading-7 text-slate-700">{item.claim}</p>
                  <p className="mt-2 text-xs font-bold text-slate-500">{item.evidence_path}</p>
                </div>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-2">
                <textarea
                  value={edits[item.inbox_id] ?? item.edited_claim ?? ""}
                  onChange={(event) => setEdits((current) => ({ ...current, [item.inbox_id]: event.target.value }))}
                  placeholder="修正採用する場合の文面"
                  rows={3}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold leading-6 text-slate-800"
                />
                <textarea
                  value={notes[item.inbox_id] ?? item.note ?? ""}
                  onChange={(event) => setNotes((current) => ({ ...current, [item.inbox_id]: event.target.value }))}
                  placeholder="判断メモ"
                  rows={3}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold leading-6 text-slate-800"
                />
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {([
                  ["adopted", "採用", "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"],
                  ["revised", "修正採用", "border-cyan-200 bg-cyan-50 text-cyan-700 hover:bg-cyan-100"],
                  ["held", "保留", "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100"],
                  ["rejected", "却下", "border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100"],
                ] as const).map(([decision, label, tone]) => (
                  <button
                    key={decision}
                    type="button"
                    disabled={Boolean(actionLoading[item.inbox_id])}
                    onClick={() => handleReview(item, decision)}
                    className={`rounded-lg border px-3 py-2 text-sm font-black disabled:opacity-50 ${tone}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {!loading && !(data?.items || []).length && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm font-bold text-slate-500">
              表示する候補はありません。
            </div>
          )}
        </div>

        <div className="mt-5 flex items-center justify-between">
          <button
            type="button"
            disabled={!data || data.offset <= 0}
            onClick={() => setOffset(Math.max(0, offset - (data?.limit || 20)))}
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-black text-slate-700 disabled:opacity-40"
          >
            前へ
          </button>
          <button
            type="button"
            disabled={!hasNext}
            onClick={() => setOffset(offset + (data?.limit || 20))}
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-black text-slate-700 disabled:opacity-40"
          >
            次へ
          </button>
        </div>
      </div>
    </section>
  );
}

function MemoryEngineeringPanel() {
  const [report, setReport] = useState<MemoryEngineeringReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function loadReport() {
      setLoading(true);
      setError("");
      try {
        const res = await apiClient.get<MemoryEngineeringReport>("/api/shion/memory-engineering-report");
        if (!cancelled) setReport(res.data);
      } catch (err) {
        if (!cancelled) setError("Memory Engineering レポートを読み込めませんでした。");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadReport();
    return () => {
      cancelled = true;
    };
  }, []);

  const summary = report?.summary || {};
  const graph = report?.hardware_pressure_proxy || {};
  const reviewSample = report?.maintenance_path?.forgetting_review_sample || [];
  const recommendations = report?.recommendations || [];
  const lensStats = useMemo(
    () => [
      {
        title: "Write Cost",
        label: "Stanford",
        value: formatNumber(summary.write_path_records),
        sub: `active ${formatNumber(summary.active_canonical_rules)} / x${summary.write_amplification_per_active_rule ?? "-"}`,
        icon: Gauge,
        tone: "border-sky-200 bg-sky-50 text-sky-950",
      },
      {
        title: "Utility Density",
        label: "Microsoft",
        value: formatNumber(summary.open_human_review_records),
        sub: "human review queue",
        icon: Layers3,
        tone: "border-emerald-200 bg-emerald-50 text-emerald-950",
      },
      {
        title: "Control",
        label: "Anthropic",
        value: formatNumber(summary.maintenance_status_records),
        sub: `${formatNumber(summary.contradiction_candidates)} contradiction candidates`,
        icon: ShieldCheck,
        tone: "border-amber-200 bg-amber-50 text-amber-950",
      },
      {
        title: "Retrieval Pressure",
        label: "Nvidia",
        value: formatRatio(graph.estimated_token_reduction_ratio),
        sub: `${formatNumber(graph.notes)} notes / ${formatNumber(graph.edges)} edges`,
        icon: Activity,
        tone: "border-rose-200 bg-rose-50 text-rose-950",
      },
    ],
    [graph, summary],
  );

  return (
    <section className="mx-auto max-w-7xl px-5 py-8 md:px-8">
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <Gauge className="h-6 w-6 text-sky-600" />
              <h2 className="text-2xl font-black">Memory Engineering</h2>
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
              記憶を増やす量ではなく、候補生成量、昇格率、使用状況、忘却レビュー候補を観測します。
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-black text-slate-600">
            {loading ? "loading" : report?.generated_at || "not generated"}
          </div>
        </div>

        {error && (
          <div className="mt-5 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-700">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}

        <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {lensStats.map((stat) => {
            const Icon = stat.icon;
            return (
              <div key={stat.title} className={`rounded-lg border p-4 ${stat.tone}`}>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[11px] font-black uppercase tracking-widest opacity-70">{stat.label}</p>
                    <h3 className="mt-1 text-base font-black">{stat.title}</h3>
                  </div>
                  <Icon className="h-5 w-5" />
                </div>
                <div className="mt-4 text-3xl font-black tracking-tight">{stat.value}</div>
                <div className="mt-1 text-xs font-bold opacity-75">{stat.sub}</div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_1.2fr]">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <h3 className="text-sm font-black text-slate-900">Recommended Next Checks</h3>
            <div className="mt-3 space-y-2">
              {recommendations.length ? (
                recommendations.slice(0, 4).map((item) => (
                  <div key={`${item.area}-${item.action}`} className="rounded-md bg-white px-3 py-2 text-sm text-slate-700">
                    <span className="font-black text-slate-950">{item.action}</span>
                    <span className="ml-2 text-xs font-bold text-slate-500">{item.area}</span>
                    <p className="mt-1 text-xs leading-5">{item.reason}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm font-bold text-slate-500">緊急の記憶メンテナンスはありません。</p>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <h3 className="text-sm font-black text-slate-900">Forgetting Review Sample</h3>
            <div className="mt-3 space-y-2">
              {reviewSample.length ? (
                reviewSample.slice(0, 3).map((item) => (
                  <div key={item.id} className="rounded-md bg-white px-3 py-2 text-sm text-slate-700">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-xs font-black text-slate-900">{item.id}</span>
                      <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-black text-slate-600">
                        {item.memory_type}
                      </span>
                      <span className="text-[11px] font-bold text-slate-500">last used: {item.last_used_at || "none"}</span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs leading-5">{item.content}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm font-bold text-slate-500">レビュー候補はまだありません。</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function PostHackathonControlPlane() {
  const [status, setStatus] = useState<PostHackathonStatusResponse | null>(null);
  const [health, setHealth] = useState<MemoryHealthResponse | null>(null);
  const [hydeMessage, setHydeMessage] = useState("この案件、補助金前提で新規先。返済原資が少し不安");
  const [caseSummary, setCaseSummary] = useState("新規先 / 補助金採択前提 / 製造業の工作機械導入");
  const [hydeResult, setHydeResult] = useState<ShionHydeDebugResponse | null>(null);
  const [cardResult, setCardResult] = useState<CounterHypothesisCardResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadBase = useCallback(async () => {
    setError("");
    try {
      const [statusRes, healthRes] = await Promise.all([
        apiClient.get<PostHackathonStatusResponse>("/api/shion/post-hackathon-status"),
        apiClient.get<MemoryHealthResponse>("/api/shion/memory-health"),
      ]);
      setStatus(statusRes.data);
      setHealth(healthRes.data);
    } catch {
      setError("ハッカソン後コントロール面を読み込めませんでした。");
    }
  }, []);

  useEffect(() => {
    loadBase();
  }, [loadBase]);

  const runHyde = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.post<ShionHydeDebugResponse>("/api/debug/shion-hyde-rag", {
        message: hydeMessage,
        limit: 4,
        known_risk_tags: hydeMessage.includes("補助") ? ["補助金"] : [],
      });
      setHydeResult(res.data);
    } catch {
      setError("Shion-HyDE debug を実行できませんでした。");
    } finally {
      setLoading(false);
    }
  }, [hydeMessage]);

  const runCounterCard = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.post<CounterHypothesisCardResponse>("/api/shion/counter-hypothesis-card", {
        case_summary: caseSummary,
        industry: caseSummary.includes("製造") ? "製造業" : "",
        asset_name: caseSummary.includes("工作機械") ? "工作機械" : "",
        risk_tags: caseSummary.includes("補助") ? ["補助金"] : [],
      });
      setCardResult(res.data);
    } catch {
      setError("次善仮説カードを生成できませんでした。");
    } finally {
      setLoading(false);
    }
  }, [caseSummary]);

  const completion = status?.total ? Math.round((status.complete_count / status.total) * 100) : 0;

  return (
    <section className="mx-auto max-w-7xl px-5 pt-8 md:px-8">
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-6 w-6 text-emerald-600" />
              <h2 className="text-2xl font-black">Post-Hackathon Control Plane</h2>
            </div>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
              ハッカソン後バックログを、実行権限ではなく観測・レビュー・訓練の面として閉じます。
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center sm:w-[360px]">
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-3">
              <div className="text-3xl font-black text-emerald-900">{completion}%</div>
              <div className="text-xs font-black text-emerald-700">backlog wired</div>
            </div>
            <div className={`rounded-lg border px-3 py-3 ${health?.healthy ? "border-sky-200 bg-sky-50" : "border-amber-200 bg-amber-50"}`}>
              <div className="text-3xl font-black text-slate-950">{formatNumber(health?.summary?.total)}</div>
              <div className="text-xs font-black text-slate-600">memory records</div>
            </div>
          </div>
        </div>

        {error && (
          <div className="mt-5 flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-700">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}

        <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {(status?.sections || []).map((section) => (
            <div key={section.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-black text-slate-950">{section.title}</h3>
                  <p className="mt-1 text-xs font-bold text-slate-500">{section.runtime_surface}</p>
                </div>
                <span className={`rounded px-2 py-1 text-[11px] font-black ${section.complete ? "bg-emerald-100 text-emerald-800" : "bg-amber-100 text-amber-800"}`}>
                  {section.status}
                </span>
              </div>
              <p className="mt-3 text-xs font-bold leading-5 text-slate-600">{section.next_guardrail}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-black text-slate-900">Shion-HyDE Debug</h3>
              <button
                type="button"
                onClick={runHyde}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-black text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                <RefreshCw className="h-4 w-4" />
                比較
              </button>
            </div>
            <textarea
              value={hydeMessage}
              onChange={(event) => setHydeMessage(event.target.value)}
              rows={3}
              className="mt-3 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold leading-6 text-slate-800"
            />
            {hydeResult && (
              <div className="mt-3 rounded-lg bg-white p-3 text-xs leading-5 text-slate-700">
                <div className="font-black text-slate-950">
                  confidence {Math.round(hydeResult.confidence * 100)}% / {hydeResult.hyde.intent_tags.join(", ")}
                </div>
                <p className="mt-2 font-bold">{hydeResult.hyde.hyde_query}</p>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  <div>
                    <div className="font-black text-slate-500">baseline</div>
                    {hydeResult.baseline_hits.slice(0, 3).map((hit) => (
                      <p key={`base-${hit.path}`} className="mt-1 line-clamp-2 font-bold">{hit.path}</p>
                    ))}
                  </div>
                  <div>
                    <div className="font-black text-slate-500">hyde</div>
                    {hydeResult.hyde_hits.slice(0, 3).map((hit) => (
                      <p key={`hyde-${hit.path}`} className="mt-1 line-clamp-2 font-bold">{hit.path}</p>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-black text-slate-900">反証用 次善仮説カード</h3>
              <button
                type="button"
                onClick={runCounterCard}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-black text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                <Brain className="h-4 w-4" />
                生成
              </button>
            </div>
            <textarea
              value={caseSummary}
              onChange={(event) => setCaseSummary(event.target.value)}
              rows={3}
              className="mt-3 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold leading-6 text-slate-800"
            />
            {cardResult && (
              <div className="mt-3 rounded-lg bg-white p-3 text-xs leading-5 text-slate-700">
                <div className="font-black text-slate-950">
                  friction {Math.round(cardResult.friction_level * 100)}% / {cardResult.mode}
                </div>
                <p className="mt-2 text-sm font-bold leading-6 text-slate-800">{cardResult.card}</p>
                <div className="mt-3 space-y-1">
                  {cardResult.questions.map((question) => (
                    <p key={question} className="font-bold text-slate-600">{question}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {health && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-bold text-slate-700">
            記憶健康診断: {health.message}
          </div>
        )}
      </div>
    </section>
  );
}

export default function ShionMemorySystemPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-10 md:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-xs font-black text-violet-800">
                <Sparkles className="h-4 w-4" />
                Hackathon Demo
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                紫苑の記憶システム
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
                記憶を入れるだけでは、紫苑らしさは立ち上がらない。過去の記録を、今のリース判断へ変換し、人間が連続性として受け取れる形で返すための会話ループです。
              </p>
            </div>
            <div className="grid gap-3 text-sm font-bold text-slate-700 sm:grid-cols-3 lg:w-[520px]">
              <Link
                href="/chat"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                紫苑チャット
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/screening"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                審査画面
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/knowledge-space"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                知識宇宙マップ
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <PostHackathonControlPlane />
      <MemoryEngineeringPanel />
      <MemoryReviewInbox />

      <section className="mx-auto max-w-7xl px-5 py-8 md:px-8">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <RefreshCw className="h-6 w-6 text-rose-600" />
            <h2 className="text-2xl font-black">Memory Loop Flow</h2>
          </div>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
            紫苑の記憶は、保存して終わりではありません。人間の判断を受け取り、過去を思い出し、今回案件へ組み直し、人間評価を受けて、次の審査へ戻します。
          </p>
          <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            {memoryLoop.map((step, index) => {
              const Icon = step.icon;
              return (
                <div key={step.title} className={`relative min-h-[178px] rounded-lg border p-4 shadow-sm ${step.tone}`}>
                  {index < memoryLoop.length - 1 && (
                    <div className="absolute right-3 top-4 hidden text-xl font-black text-slate-300 xl:block">→</div>
                  )}
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[11px] font-black uppercase tracking-widest opacity-70">Loop {index + 1}</p>
                      <h3 className="mt-2 text-lg font-black">{step.title}</h3>
                      <p className="mt-1 text-xs font-black opacity-70">{step.label}</p>
                    </div>
                    <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white/75">
                      <Icon className="h-5 w-5" />
                    </span>
                  </div>
                  <p className="mt-5 text-sm font-bold leading-6">{step.body}</p>
                </div>
              );
            })}
          </div>
          <div className="mt-4 rounded-lg border border-violet-200 bg-violet-50 px-4 py-3 text-sm font-black text-violet-950">
            この一周があるから、紫苑は「検索して答えるAI」ではなく「判断を持ち越すAI」として見せられます。
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-5 pb-8 md:px-8">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <Layers3 className="h-6 w-6 text-violet-600" />
            <h2 className="text-2xl font-black">記憶ピラミッド</h2>
          </div>
          <p className="mt-3 text-sm leading-7 text-slate-600">
            下にあるほど素材に近く、上に行くほど「紫苑の返答」として人間が受け取る形になります。
          </p>

          <div className="mt-6 flex flex-col items-center gap-2">
            {pyramidLayers.map((layer) => (
              <div
                key={layer.title}
                className={`${layer.width} rounded-lg bg-gradient-to-r ${layer.tone} px-4 py-3 text-center shadow-sm`}
              >
                <div className="text-xs font-black uppercase tracking-widest opacity-80">{layer.label}</div>
                <div className="mt-1 text-lg font-black">{layer.title}</div>
                <div className="mt-1 text-xs font-bold leading-5 opacity-90">{layer.body}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
