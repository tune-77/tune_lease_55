"use client";

import React, { useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  AlertTriangle,
  ArrowUpRight,
  Brain,
  ExternalLink,
  FileText,
  Lightbulb,
  Newspaper,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

type NewsFocus = {
  available?: boolean;
  note_date?: string;
  headline?: string;
  theme_summary?: string;
  bucket_summary?: string;
  tag_summary?: string;
  focus_lines?: string[];
  memo_lines?: string[];
  metrics_lines?: string[];
  article_titles?: string[];
};

type RecentNewsItem = {
  date?: string;
  title?: string;
  summary_lines?: string[];
  usage_memo?: string;
  tags?: string[];
  importance?: string;
  source?: string;
  article_url?: string;
  file_path?: string;
};

type ClassifiedArticle = {
  date?: string;
  title?: string;
  summary_lines?: string[];
  usage_memo?: string;
  source?: string;
  article_url?: string;
  importance?: string;
  impact_direction?: string;
};

type ClassifiedCategory = {
  axis: string;
  axis_label: string;
  category: string;
  article_count: number;
  trend: string;
  key_points?: string[];
  lease_implications?: {
    direction?: string;
    repayment_capacity?: string;
    residual_value?: string;
    business_opportunity?: string;
  };
  recommended_checks?: string[];
  articles?: ClassifiedArticle[];
};

type ClassifiedAxis = {
  axis: string;
  label: string;
  category_count: number;
  article_count: number;
  categories: ClassifiedCategory[];
};

type ClassifiedSummary = {
  available?: boolean;
  generated_at?: string;
  article_count?: number;
  axes?: ClassifiedAxis[];
  top_insights?: Array<{
    label?: string;
    trend?: string;
    repayment_capacity?: string;
  }>;
};

type VertexTrendSummary = {
  available?: boolean;
  generated_at?: string;
  source?: string;
  trend_title?: string;
  overall_summary?: string;
  trend_lines?: string[];
  caution_points?: string[];
  screening_actions?: string[];
  watch_categories?: Array<{
    label?: string;
    count?: number;
    reason?: string;
  }>;
  source_articles?: Array<{
    date?: string;
    title?: string;
    source?: string;
    article_url?: string;
  }>;
  vertex?: {
    used?: boolean;
    status?: string;
    answer_text?: string;
    grounding_score?: number | string | null;
    low_support_claim_count?: number;
    refs?: Array<{
      title?: string;
      source?: string;
      snippet?: string;
    }>;
  };
};

const axisOrder = ["industry", "social", "finance"];

function formatDate(value?: string) {
  if (!value) return "";
  return value.slice(0, 10);
}

export default function NewsDashboardPage() {
  const [focus, setFocus] = useState<NewsFocus | null>(null);
  const [recentNews, setRecentNews] = useState<RecentNewsItem[]>([]);
  const [summary, setSummary] = useState<ClassifiedSummary | null>(null);
  const [trendSummary, setTrendSummary] = useState<VertexTrendSummary | null>(null);
  const [selectedAxis, setSelectedAxis] = useState("industry");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const loadNews = async (refresh = false) => {
    setError("");
    setRefreshing(refresh);
    try {
      const [focusRes, recentRes, summaryRes, trendRes] = await Promise.all([
        apiClient.get("/api/lease-news/focus"),
        apiClient.get("/api/lease-news/recent?limit=8"),
        apiClient.get(`/api/lease-news/classified-summary${refresh ? "?refresh=true" : ""}`),
        apiClient.get(`/api/lease-news/trend-summary${refresh ? "?refresh=true" : ""}`),
      ]);
      setFocus(focusRes.data || null);
      setRecentNews(recentRes.data?.items || []);
      setSummary(summaryRes.data || null);
      setTrendSummary(trendRes.data || null);
    } catch (err) {
      console.error("Failed to load news dashboard", err);
      setError("ニュース情報を取得できませんでした。API と Obsidian Vault の同期状態を確認してください。");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadNews();
  }, []);

  const axes = useMemo(() => {
    const items = summary?.axes || [];
    return [...items].sort((a, b) => {
      const ai = axisOrder.indexOf(a.axis);
      const bi = axisOrder.indexOf(b.axis);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
  }, [summary]);

  const activeAxis = axes.find((axis) => axis.axis === selectedAxis) || axes[0];
  const activeCategory =
    activeAxis?.categories?.find((category) => category.category === selectedCategory) ||
    activeAxis?.categories?.[0];

  useEffect(() => {
    if (!activeAxis) return;
    if (activeAxis.axis !== selectedAxis) {
      setSelectedAxis(activeAxis.axis);
      setSelectedCategory(activeAxis.categories?.[0]?.category || "");
      return;
    }
    if (activeCategory?.category && activeCategory.category !== selectedCategory) {
      setSelectedCategory(activeCategory.category);
    }
  }, [activeAxis, activeCategory?.category, selectedAxis, selectedCategory]);

  return (
    <div className="min-h-screen bg-slate-50 px-4 py-6 text-slate-900 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-black uppercase tracking-[0.18em] text-sky-600">
              <Newspaper className="h-4 w-4" />
              Lease News Intelligence
            </div>
            <h1 className="mt-2 text-2xl font-black tracking-tight text-slate-950 sm:text-3xl">
              ニュース審査ダッシュボード
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-600">
              業界ニュースを業種別・社会情勢・金融情報に分類し、返済能力、残価リスク、事業機会への示唆として確認します。
            </p>
          </div>
          <button
            onClick={() => loadNews(true)}
            disabled={refreshing}
            className="inline-flex w-fit items-center gap-2 rounded-lg bg-slate-950 px-4 py-2 text-sm font-black text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            更新
          </button>
        </header>

        {error && (
          <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {loading ? (
          <div className="rounded-lg border border-slate-200 bg-white p-8 text-sm font-bold text-slate-500">
            ニュース情報を読み込み中です。
          </div>
        ) : (
          <>
            <section className="rounded-lg border border-sky-200 bg-white p-5">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-sky-600">
                    <Sparkles className="h-4 w-4" />
                    Vertex Assisted Summary
                  </div>
                  <h2 className="mt-1 text-xl font-black text-slate-950">
                    {trendSummary?.trend_title || "最近ニュース診断"}
                  </h2>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full bg-sky-50 px-3 py-1 text-xs font-black text-sky-700">
                    Vertex: {trendSummary?.vertex?.used ? "利用" : trendSummary?.vertex?.status || "未利用"}
                  </span>
                  {trendSummary?.generated_at && (
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-500">
                      {formatDate(trendSummary.generated_at)}
                    </span>
                  )}
                </div>
              </div>

              {!trendSummary?.available ? (
                <p className="mt-4 rounded-lg bg-slate-50 p-4 text-sm text-slate-500">
                  最近ニュースの傾向サマリーはまだありません。ニュース収集後に表示されます。
                </p>
              ) : (
                <div className="mt-5 grid grid-cols-1 gap-4 xl:grid-cols-[1.25fr_0.75fr]">
                  <div className="space-y-4">
                    <div className="rounded-lg bg-slate-50 p-4">
                      <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">要約</p>
                      <p className="mt-2 text-sm font-bold leading-relaxed text-slate-800">
                        {trendSummary.overall_summary}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                      {(trendSummary.trend_lines || []).slice(0, 4).map((line, index) => (
                        <div key={index} className="rounded-lg border border-slate-200 bg-white p-3">
                          <div className="flex items-center gap-2">
                            <Lightbulb className="h-3.5 w-3.5 text-sky-500" />
                            <p className="text-[10px] font-black text-sky-600">傾向 {index + 1}</p>
                          </div>
                          <p className="mt-2 text-xs leading-relaxed text-slate-700">{line}</p>
                        </div>
                      ))}
                    </div>
                    {(trendSummary.screening_actions || []).length > 0 && (
                      <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-4">
                        <p className="text-[10px] font-black uppercase tracking-widest text-emerald-700">
                          審査アクション
                        </p>
                        <div className="mt-2 space-y-1.5">
                          {(trendSummary.screening_actions || []).slice(0, 5).map((line, index) => (
                            <p key={index} className="text-xs leading-relaxed text-emerald-950">{line}</p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                      <div className="flex items-center gap-2">
                        <ShieldAlert className="h-4 w-4 text-amber-700" />
                        <p className="text-[10px] font-black uppercase tracking-widest text-amber-700">注意点</p>
                      </div>
                      <div className="mt-2 space-y-2">
                        {(trendSummary.caution_points || []).slice(0, 5).map((line, index) => (
                          <p key={index} className="text-xs font-bold leading-relaxed text-amber-950">{line}</p>
                        ))}
                      </div>
                    </div>

                    {(trendSummary.watch_categories || []).length > 0 && (
                      <div className="rounded-lg border border-slate-200 bg-white p-4">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">重点カテゴリ</p>
                        <div className="mt-3 space-y-2">
                          {(trendSummary.watch_categories || []).slice(0, 4).map((item, index) => (
                            <div key={`${item.label}-${index}`} className="rounded-lg bg-slate-50 p-3">
                              <div className="flex items-center justify-between gap-3">
                                <p className="text-xs font-black text-slate-900">{item.label}</p>
                                <span className="rounded-full bg-white px-2 py-1 text-[10px] font-black text-slate-500">
                                  {item.count || 0}件
                                </span>
                              </div>
                              <p className="mt-1 text-xs leading-relaxed text-slate-600">{item.reason}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {(trendSummary.vertex?.refs || []).length > 0 && (
                      <details className="rounded-lg border border-slate-200 bg-white p-4">
                        <summary className="cursor-pointer text-sm font-black text-slate-800">Vertex参照を見る</summary>
                        <div className="mt-3 space-y-3">
                          {(trendSummary.vertex?.refs || []).slice(0, 5).map((ref, index) => (
                            <div key={`${ref.source}-${index}`}>
                              <p className="text-xs font-black text-slate-900">{ref.title || ref.source}</p>
                              {ref.snippet && <p className="mt-1 text-xs leading-relaxed text-slate-600">{ref.snippet}</p>}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              )}
            </section>

            <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2 rounded-lg border border-slate-200 bg-white p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-black uppercase tracking-widest text-sky-600">Focus</p>
                    <h2 className="mt-1 text-lg font-black text-slate-900">
                      {focus?.headline || focus?.theme_summary || "業界リスクニュースの注目論点"}
                    </h2>
                  </div>
                  {focus?.note_date && (
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-500">
                      {focus.note_date}
                    </span>
                  )}
                </div>

                {focus?.available ? (
                  <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                    {(focus.focus_lines || []).slice(0, 4).map((line, index) => (
                      <div key={index} className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                        <p className="text-[10px] font-black text-sky-600">論点 {index + 1}</p>
                        <p className="mt-1 text-sm font-bold leading-relaxed text-slate-800">{line}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-4 text-sm text-slate-500">注目論点はまだありません。</p>
                )}

                {(focus?.memo_lines || []).length > 0 && (
                  <div className="mt-4 rounded-lg border border-amber-100 bg-amber-50 p-3">
                    <p className="text-[10px] font-black uppercase tracking-widest text-amber-700">審査メモ</p>
                    <div className="mt-2 space-y-1.5">
                      {(focus?.memo_lines || []).slice(0, 3).map((line, index) => (
                        <p key={index} className="text-xs leading-relaxed text-amber-950">{line}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5">
                <p className="text-xs font-black uppercase tracking-widest text-emerald-600">Coverage</p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="rounded-lg bg-emerald-50 p-3">
                    <p className="text-[10px] font-black text-emerald-700">分類記事</p>
                    <p className="mt-1 text-2xl font-black text-slate-950">{summary?.article_count || 0}</p>
                  </div>
                  <div className="rounded-lg bg-sky-50 p-3">
                    <p className="text-[10px] font-black text-sky-700">軸</p>
                    <p className="mt-1 text-2xl font-black text-slate-950">{axes.length}</p>
                  </div>
                </div>
                {summary?.generated_at && (
                  <p className="mt-3 text-xs font-bold text-slate-500">生成: {formatDate(summary.generated_at)}</p>
                )}
                {(focus?.metrics_lines || []).length > 0 && (
                  <div className="mt-4 space-y-2">
                    {(focus?.metrics_lines || []).slice(0, 4).map((line, index) => (
                      <p key={index} className="rounded-lg bg-slate-50 px-3 py-2 text-xs font-bold text-slate-700">
                        {line}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <p className="text-xs font-black uppercase tracking-widest text-emerald-600">Classified Summary</p>
                  <h2 className="mt-1 text-xl font-black text-slate-950">分類別の審査示唆</h2>
                </div>
                <div className="flex flex-wrap gap-2">
                  {axes.map((axis) => (
                    <button
                      key={axis.axis}
                      onClick={() => {
                        setSelectedAxis(axis.axis);
                        setSelectedCategory(axis.categories?.[0]?.category || "");
                      }}
                      className={`rounded-lg px-3 py-2 text-xs font-black transition-colors ${
                        activeAxis?.axis === axis.axis
                          ? "bg-slate-950 text-white"
                          : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                      }`}
                    >
                      {axis.label} {axis.article_count}
                    </button>
                  ))}
                </div>
              </div>

              {!summary?.available || axes.length === 0 ? (
                <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-5 text-sm text-slate-500">
                  分類済みニュースサマリーがまだありません。ニュース収集後に更新してください。
                </div>
              ) : (
                <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-[260px_1fr]">
                  <aside className="space-y-2">
                    {(activeAxis?.categories || []).map((category) => (
                      <button
                        key={category.category}
                        onClick={() => setSelectedCategory(category.category)}
                        className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm font-black transition-colors ${
                          activeCategory?.category === category.category
                            ? "bg-emerald-600 text-white"
                            : "bg-emerald-50 text-emerald-800 hover:bg-emerald-100"
                        }`}
                      >
                        <span>{category.category}</span>
                        <span className="text-xs opacity-80">{category.article_count}</span>
                      </button>
                    ))}
                  </aside>

                  {activeCategory && (
                    <article className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <p className="text-[10px] font-black uppercase tracking-widest text-emerald-600">
                            {activeCategory.axis_label}
                          </p>
                          <h3 className="mt-1 text-lg font-black text-slate-950">{activeCategory.category}</h3>
                        </div>
                        <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-slate-500">
                          {activeCategory.article_count}件
                        </span>
                      </div>
                      <p className="mt-3 text-sm font-bold leading-relaxed text-slate-800">{activeCategory.trend}</p>

                      <div className="mt-4 grid grid-cols-1 gap-3 xl:grid-cols-3">
                        <InsightCard title="返済能力" text={activeCategory.lease_implications?.repayment_capacity} />
                        <InsightCard title="残価リスク" text={activeCategory.lease_implications?.residual_value} />
                        <InsightCard title="事業機会" text={activeCategory.lease_implications?.business_opportunity} />
                      </div>

                      {(activeCategory.recommended_checks || []).length > 0 && (
                        <div className="mt-4 rounded-lg border border-amber-100 bg-amber-50 p-3">
                          <p className="text-[10px] font-black uppercase tracking-widest text-amber-700">確認ポイント</p>
                          <div className="mt-2 space-y-1.5">
                            {(activeCategory.recommended_checks || []).slice(0, 4).map((line, index) => (
                              <p key={index} className="text-xs leading-relaxed text-amber-950">{line}</p>
                            ))}
                          </div>
                        </div>
                      )}

                      {(activeCategory.articles || []).length > 0 && (
                        <details className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
                          <summary className="cursor-pointer text-sm font-black text-slate-800">元記事を見る</summary>
                          <div className="mt-3 divide-y divide-slate-100">
                            {(activeCategory.articles || []).slice(0, 8).map((article, index) => (
                              <div key={`${article.title}-${index}`} className="py-3">
                                <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
                                  <p className="text-sm font-black leading-relaxed text-slate-900">
                                    {article.title || "ニュース"}
                                  </p>
                                  {article.article_url && (
                                    <a
                                      href={article.article_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="inline-flex shrink-0 items-center gap-1 text-xs font-black text-sky-600 hover:text-sky-800"
                                    >
                                      元記事
                                      <ExternalLink className="h-3 w-3" />
                                    </a>
                                  )}
                                </div>
                                {article.summary_lines?.[0] && (
                                  <p className="mt-1 text-xs leading-relaxed text-slate-600">{article.summary_lines[0]}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </article>
                  )}
                </div>
              )}
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-sky-500" />
                <h2 className="text-lg font-black text-slate-950">最近のニュース</h2>
              </div>
              {recentNews.length === 0 ? (
                <p className="mt-4 text-sm text-slate-500">ニュースダイジェストがまだありません。</p>
              ) : (
                <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
                  {recentNews.map((item, index) => (
                    <div key={`${item.title}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-[10px] font-black text-slate-400">{formatDate(item.date)}</p>
                          <h3 className="mt-1 text-sm font-black leading-relaxed text-slate-900">{item.title || "ニュース"}</h3>
                        </div>
                        {item.article_url && (
                          <a
                            href={item.article_url}
                            target="_blank"
                            rel="noreferrer"
                            className="rounded-lg bg-white p-2 text-sky-600 hover:text-sky-800"
                          >
                            <ArrowUpRight className="h-4 w-4" />
                          </a>
                        )}
                      </div>
                      {item.summary_lines?.[0] && (
                        <p className="mt-2 text-xs leading-relaxed text-slate-600">{item.summary_lines[0]}</p>
                      )}
                      {(item.tags || []).length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {(item.tags || []).slice(0, 4).map((tag) => (
                            <span key={tag} className="rounded-full bg-white px-2 py-1 text-[10px] font-bold text-slate-500">
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  );
}

function InsightCard({ title, text }: { title: string; text?: string }) {
  return (
    <div className="rounded-lg bg-white p-3">
      <div className="flex items-center gap-2">
        <Brain className="h-3.5 w-3.5 text-emerald-500" />
        <p className="text-[10px] font-black text-slate-400">{title}</p>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-700">{text || "個社条件で確認する。"}</p>
    </div>
  );
}
