"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  Database,
  ExternalLink,
  Globe2,
  Loader2,
  RefreshCw,
  Search,
  Settings2,
  SlidersHorizontal,
  TriangleAlert,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type VertexRef = {
  title?: string;
  source_path?: string;
  doc_id?: string;
  link?: string;
  excerpt?: string;
  uri?: string;
  snippet?: string;
};

type VertexDebugResponse = {
  query?: string;
  config?: Record<string, unknown>;
  controls?: {
    preferred_terms?: string[];
    bury_terms?: string[];
    mode?: string;
  };
  search?: {
    used?: boolean;
    status?: string;
    summary?: string;
    refs?: string[];
    results?: VertexRef[];
    error?: string;
  };
  answer?: {
    used?: boolean;
    status?: string;
    answer_text?: string;
    answer_state?: string;
    grounding_score?: number | string | null;
    grounding_score_source?: string;
    grounding_supports?: Array<{
      claim_text?: string;
      support_score?: number | string | null;
      citation_indices?: number[];
    }>;
    low_support_claim_count?: number;
    support_count?: number;
    related_questions?: unknown[];
    refs?: string[];
    search_results?: VertexRef[];
    error?: string;
  };
  google_grounding?: {
    used?: boolean;
    status?: string;
    text?: string;
    sources?: Array<{ title?: string; uri?: string }>;
    error?: string;
  };
};

type WidgetConfig = {
  configured?: boolean;
  config_id?: string;
  placeholder?: string;
  snippet?: string;
  setup_note?: string;
};

const examples = [
  "補助金前提の工作機械リースで確認すべきこと",
  "再リース案件では耐用年数と残価をどう確認するべきか",
  "運送業のリース審査で見るべき業界リスク",
  "返済余力と資金繰りの異常兆候をどう確認するか",
];

function StatusPill({ status, used }: { status?: string; used?: boolean }) {
  const ok = status === "ok" && used !== false;
  const skipped = status === "disabled" || status === "not_configured" || status === "auth_unavailable";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-bold ${
        ok
          ? "bg-emerald-50 text-emerald-700"
          : skipped
            ? "bg-amber-50 text-amber-700"
            : "bg-rose-50 text-rose-700"
      }`}
    >
      {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <TriangleAlert className="h-3.5 w-3.5" />}
      {status || "not_run"}
    </span>
  );
}

function ResultList({ items }: { items?: VertexRef[] }) {
  if (!items?.length) return <p className="text-sm font-semibold text-slate-500">参照なし</p>;
  return (
    <div className="space-y-3">
      {items.slice(0, 8).map((item, index) => {
        const label = item.source_path || item.title || item.uri || item.link || item.doc_id || `result-${index + 1}`;
        const href = item.uri || item.link || "";
        return (
          <div key={`${label}-${index}`} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <p className="min-w-0 break-words text-sm font-black text-slate-900">{label}</p>
              {href ? (
                <a href={href} target="_blank" rel="noreferrer" className="shrink-0 text-slate-400 hover:text-teal-700">
                  <ExternalLink className="h-4 w-4" />
                </a>
              ) : null}
            </div>
            {(item.excerpt || item.snippet) && (
              <p className="mt-2 text-xs font-semibold leading-6 text-slate-600">{item.excerpt || item.snippet}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Section({
  title,
  icon: Icon,
  status,
  used,
  children,
}: {
  title: string;
  icon: React.ElementType;
  status?: string;
  used?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-slate-200 py-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-teal-700" />
          <h2 className="text-lg font-black text-slate-950">{title}</h2>
        </div>
        <StatusPill status={status} used={used} />
      </div>
      {children}
    </section>
  );
}

export default function VertexSearchDebugPage() {
  const [query, setQuery] = useState(examples[0]);
  const [pageSize, setPageSize] = useState(5);
  const [includeSearch, setIncludeSearch] = useState(true);
  const [includeAnswer, setIncludeAnswer] = useState(true);
  const [includeGoogle, setIncludeGoogle] = useState(false);
  const [includeGroundingSupports, setIncludeGroundingSupports] = useState(true);
  const [groundingFilteringLevel, setGroundingFilteringLevel] = useState("");
  const [filterExpression, setFilterExpression] = useState("");
  const [data, setData] = useState<VertexDebugResponse | null>(null);
  const [widget, setWidget] = useState<WidgetConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadWidget = async () => {
    try {
      const res = await apiClient.get<WidgetConfig>("/api/vertex-search/widget-config");
      setWidget(res.data);
    } catch {
      setWidget({ configured: false, setup_note: "Widget設定を取得できませんでした。" });
    }
  };

  useEffect(() => {
    loadWidget();
  }, []);

  const run = async () => {
    const text = query.trim();
    if (!text) return;
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.post<VertexDebugResponse>("/api/vertex-search/debug", {
        query: text,
        page_size: pageSize,
        include_search: includeSearch,
        include_answer: includeAnswer,
        include_google_grounding: includeGoogle,
        include_grounding_supports: includeGroundingSupports,
        grounding_filtering_level: groundingFilteringLevel || null,
        filter_expression: filterExpression.trim() || null,
      });
      setData(res.data);
    } catch {
      setError("Vertexデバッグを実行できませんでした。APIサーバー、認証、クレジット状態を確認してください。");
    } finally {
      setLoading(false);
    }
  };

  const configRows = useMemo(() => Object.entries(data?.config || {}), [data]);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-7 md:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 rounded-md bg-teal-50 px-3 py-1 text-xs font-black text-teal-800">
                <Database className="h-4 w-4" />
                Vertex AI Search
              </div>
              <h1 className="mt-3 text-3xl font-black tracking-tight md:text-4xl">Vertex知識検索デバッグ</h1>
              <p className="mt-3 max-w-3xl text-sm font-bold leading-7 text-slate-600">
                既存Obsidian RAGを主軸に、Vertex Search、Answer API、Google Search groundingを同じ質問で確認します。
              </p>
            </div>
            <button
              onClick={run}
              disabled={loading || !query.trim()}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-slate-950 px-5 text-sm font-black text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              実行
            </button>
          </div>
        </div>
      </section>

      <div className="mx-auto grid max-w-7xl gap-6 px-5 py-6 md:px-8 lg:grid-cols-[380px_1fr]">
        <aside className="h-fit rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <Settings2 className="h-5 w-5 text-teal-700" />
            <h2 className="text-base font-black">実行条件</h2>
          </div>

          <label className="mt-5 block text-xs font-black text-slate-500">質問</label>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={5}
            className="mt-2 w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-3 text-sm font-semibold leading-6 outline-none focus:border-teal-500"
          />

          <div className="mt-4 grid gap-2">
            {examples.map((example) => (
              <button
                key={example}
                onClick={() => setQuery(example)}
                className="rounded-lg border border-slate-200 px-3 py-2 text-left text-xs font-bold leading-5 text-slate-600 hover:border-teal-300 hover:text-teal-800"
              >
                {example}
              </button>
            ))}
          </div>

          <div className="mt-5 grid gap-3">
            <label className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 px-3 py-2.5 text-sm font-bold">
              <span className="flex items-center gap-2">
                <Search className="h-4 w-4 text-slate-500" />
                Search
              </span>
              <input type="checkbox" checked={includeSearch} onChange={(event) => setIncludeSearch(event.target.checked)} />
            </label>
            <label className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 px-3 py-2.5 text-sm font-bold">
              <span className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-slate-500" />
                Support score
              </span>
              <input type="checkbox" checked={includeGroundingSupports} onChange={(event) => setIncludeGroundingSupports(event.target.checked)} />
            </label>
            <label className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 px-3 py-2.5 text-sm font-bold">
              <span className="flex items-center gap-2">
                <BrainCircuit className="h-4 w-4 text-slate-500" />
                Answer
              </span>
              <input type="checkbox" checked={includeAnswer} onChange={(event) => setIncludeAnswer(event.target.checked)} />
            </label>
            <label className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 px-3 py-2.5 text-sm font-bold">
              <span className="flex items-center gap-2">
                <Globe2 className="h-4 w-4 text-slate-500" />
                Google Search
              </span>
              <input type="checkbox" checked={includeGoogle} onChange={(event) => setIncludeGoogle(event.target.checked)} />
            </label>
          </div>

          <label className="mt-5 block text-xs font-black text-slate-500">件数</label>
          <input
            type="number"
            min={1}
            max={10}
            value={pageSize}
            onChange={(event) => setPageSize(Math.max(1, Math.min(10, Number(event.target.value) || 5)))}
            className="mt-2 h-10 w-full rounded-lg border border-slate-200 px-3 text-sm font-bold outline-none focus:border-teal-500"
          />

          <label className="mt-5 block text-xs font-black text-slate-500">filter</label>
          <input
            value={filterExpression}
            onChange={(event) => setFilterExpression(event.target.value)}
            placeholder='例: source_path: ANY("Research")'
            className="mt-2 h-10 w-full rounded-lg border border-slate-200 px-3 text-xs font-semibold outline-none focus:border-teal-500"
          />

          <label className="mt-5 block text-xs font-black text-slate-500">grounding filtering</label>
          <select
            value={groundingFilteringLevel}
            onChange={(event) => setGroundingFilteringLevel(event.target.value)}
            className="mt-2 h-10 w-full rounded-lg border border-slate-200 px-3 text-xs font-bold outline-none focus:border-teal-500"
          >
            <option value="">指定なし</option>
            <option value="FILTERING_LEVEL_LOW">LOW</option>
            <option value="FILTERING_LEVEL_HIGH">HIGH</option>
          </select>

          <div className="mt-5 rounded-lg bg-slate-50 p-3">
            <p className="text-xs font-black text-slate-500">Widget</p>
            <p className="mt-2 break-words text-xs font-semibold leading-5 text-slate-700">
              {widget?.configured ? widget.snippet : widget?.setup_note || "設定確認中"}
            </p>
          </div>
        </aside>

        <div className="rounded-lg border border-slate-200 bg-white px-5 shadow-sm">
          {error && (
            <div className="mt-5 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-bold text-rose-700">
              {error}
            </div>
          )}

          <section className="py-6">
            <div className="mb-4 flex items-center gap-2">
              <SlidersHorizontal className="h-5 w-5 text-teal-700" />
              <h2 className="text-lg font-black">検索制御</h2>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg bg-slate-50 p-4">
                <p className="text-xs font-black text-slate-500">優先</p>
                <p className="mt-2 text-sm font-bold leading-6 text-slate-800">
                  {(data?.controls?.preferred_terms || []).join(" / ") || "未実行"}
                </p>
              </div>
              <div className="rounded-lg bg-slate-50 p-4">
                <p className="text-xs font-black text-slate-500">抑制</p>
                <p className="mt-2 text-sm font-bold leading-6 text-slate-800">
                  {(data?.controls?.bury_terms || []).join(" / ") || "未実行"}
                </p>
              </div>
            </div>
            {configRows.length > 0 && (
              <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {configRows.map(([key, value]) => (
                  <div key={key} className="rounded-lg border border-slate-200 px-3 py-2">
                    <p className="text-[11px] font-black uppercase text-slate-400">{key}</p>
                    <p className="mt-1 break-words text-xs font-bold text-slate-700">{String(value)}</p>
                  </div>
                ))}
              </div>
            )}
          </section>

          <Section title="Search API" icon={Search} status={data?.search?.status} used={data?.search?.used}>
            {data?.search?.error && <p className="mb-3 text-sm font-bold text-rose-700">{data.search.error}</p>}
            {data?.search?.summary && <p className="mb-4 text-sm font-semibold leading-7 text-slate-700">{data.search.summary}</p>}
            <ResultList items={data?.search?.results} />
          </Section>

          <Section title="Answer API" icon={BrainCircuit} status={data?.answer?.status} used={data?.answer?.used}>
            {data?.answer?.error && <p className="mb-3 text-sm font-bold text-rose-700">{data.answer.error}</p>}
            {data?.answer && (
              <div className="mb-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg bg-cyan-50 p-3">
                  <p className="text-[11px] font-black text-cyan-700">Grounding score</p>
                  <p className="mt-1 text-lg font-black text-slate-950">{data.answer.grounding_score ?? "未返却"}</p>
                  {data.answer.grounding_score_source && (
                    <p className="mt-1 text-[11px] font-bold text-cyan-700">{data.answer.grounding_score_source}</p>
                  )}
                </div>
                <div className="rounded-lg bg-slate-50 p-3">
                  <p className="text-[11px] font-black text-slate-500">Support claims</p>
                  <p className="mt-1 text-lg font-black text-slate-950">{data.answer.support_count ?? 0}</p>
                </div>
                <div className="rounded-lg bg-amber-50 p-3">
                  <p className="text-[11px] font-black text-amber-700">Low support</p>
                  <p className="mt-1 text-lg font-black text-slate-950">{data.answer.low_support_claim_count ?? 0}</p>
                </div>
              </div>
            )}
            {data?.answer?.answer_text && (
              <p className="mb-4 whitespace-pre-wrap text-sm font-semibold leading-7 text-slate-800">{data.answer.answer_text}</p>
            )}
            {data?.answer?.grounding_supports?.length ? (
              <div className="mb-4 space-y-2">
                {data.answer.grounding_supports.slice(0, 5).map((support, index) => (
                  <div key={`${support.claim_text || "claim"}-${index}`} className="rounded-lg border border-cyan-100 bg-white p-3">
                    <p className="text-[11px] font-black text-cyan-700">support {support.support_score ?? "n/a"}</p>
                    <p className="mt-1 text-xs font-semibold leading-6 text-slate-700">{support.claim_text || "claim textなし"}</p>
                  </div>
                ))}
              </div>
            ) : null}
            <ResultList items={data?.answer?.search_results} />
          </Section>

          <Section title="Google Search Grounding" icon={Globe2} status={data?.google_grounding?.status} used={data?.google_grounding?.used}>
            {data?.google_grounding?.error && <p className="mb-3 text-sm font-bold text-rose-700">{data.google_grounding.error}</p>}
            {data?.google_grounding?.text && (
              <p className="mb-4 whitespace-pre-wrap text-sm font-semibold leading-7 text-slate-800">{data.google_grounding.text}</p>
            )}
            <ResultList items={(data?.google_grounding?.sources || []).map((source) => ({ title: source.title, uri: source.uri }))} />
          </Section>
        </div>
      </div>
    </main>
  );
}
