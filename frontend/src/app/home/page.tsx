"use client";
import React, { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import {
  BarChart3,
  TrendingUp,
  Users,
  Target,
  Activity,
  CheckCircle,
  XCircle,
  Settings,
  X,
  ScrollText,
  Newspaper,
  Plus,
  Send,
  Loader2,
  ExternalLink,
  Tag,
  AlertTriangle,
  Brain,
  MessageCircle,
} from 'lucide-react';
import { normalizePrefecture } from '@/lib/prefecture';
import { formatLocalDateKey } from '@/lib/date';

type TopDriver = {
  label?: string;
  coef?: number;
  direction?: string;
};

type DashboardAnalysis = {
  closed_count?: number;
  avg_financials?: Record<string, number | string | null>;
  tag_ranking?: Array<[string, number]>;
  top3_drivers?: TopDriver[];
  qualitative_summary?: {
    avg_weighted?: number | null;
  } | null;
  avg_score_borrower?: number | null;
};

type RecentCase = {
  timestamp?: string;
  final_status?: string;
  industry_major?: string;
  industry_sub?: string;
  result?: {
    score?: number | null;
    hantei?: string;
  };
};

type GapItem = {
  id?: string;
  title?: string;
  priority?: string;
  category?: string;
  impact?: string;
  recommended_action?: string;
  evidence?: string[];
};

type DashboardStats = {
  analysis?: DashboardAnalysis;
  recent_cases?: RecentCase[];
  improvement_highlights?: {
    available?: boolean;
    date?: string;
    generated_at?: string;
    status?: string;
    source?: string;
    counts?: {
      applied?: number;
      auto_fix_candidates?: number;
      needs_review?: number;
      rejected?: number;
    };
    items?: Array<{
      id?: string;
      title?: string;
      status?: string;
      priority?: string;
      reason?: string;
      category?: string;
      canonical_key?: string;
    }>;
  };
  lease_system_gaps?: {
    available?: boolean;
    generated_at?: string;
    mode?: string;
    source?: string;
    counts?: Record<string, number>;
    items?: GapItem[];
  };
  lease_news_focus?: {
    available?: boolean;
    note_path?: string;
    note_date?: string;
    profile?: string;
    theme_summary?: string;
    bucket_summary?: string;
    tag_summary?: string;
    focus_lines?: string[];
    memo_lines?: string[];
    metrics_lines?: string[];
    article_titles?: string[];
    headline?: string;
  };
  lease_news_reflection?: {
    available?: boolean;
    note_path?: string;
    note_date?: string;
    theme_summary?: string;
    tag_summary?: string;
    headline?: string;
    thought_lines?: string[];
    tomorrow_lines?: string[];
    illustration_url?: string;
    continuity_days?: number;
    dominant_mood?: string;
    self_narrative?: string;
    current_question?: string;
    memory_excerpt?: string;
    user_understanding?: string;
    user_curiosity?: string;
    user_interests?: string[];
    observed_days?: number;
    primary_goal?: string;
    secondary_goal?: string;
    ultimate_goal?: string;
    ultimate_goal_status?: string;
    knowledge_available?: boolean;
    knowledge_scope?: string;
    indexed_notes?: number;
    knowledge_source_count?: number;
    knowledge_sources?: string[];
  };
  lease_news_brief?: {
    available?: boolean;
    prefecture?: string;
    region?: string;
    geo_context?: string;
    national_headline?: string;
    national_focus_lines?: string[];
    regional_available?: boolean;
    regional_title?: string;
    regional_summary_lines?: string[];
    regional_usage_memo?: string;
    regional_tags?: string[];
    regional_source?: string;
    opening_line?: string;
    question_line?: string;
    note_date?: string;
    note_path?: string;
  };
};

const HOME_SHION_CHAT_DRAFT_KEY = "home-shion-chat-draft";

type NewsSummaryItem = {
  date: string;
  title: string;
  summary_lines: string[];
  usage_memo: string;
  summary_codes?: string[];
  usage_codes?: string[];
  key_phrases?: string[];
  tags: string[];
  region: string;
  importance: string;
  source: string;
  article_url: string;
  file_path: string;
  week: string;
  month: string;
};

type HomePanelSettings = {
  showKpis: boolean;
  showHighlights: boolean;
  showNews: boolean;
  showRecentCases: boolean;
  showNewsDigest: boolean;
  showGaps: boolean;
};

const HOME_SETTINGS_KEY = "home-dashboard-panel-settings";
const DEFAULT_PANEL_SETTINGS: HomePanelSettings = {
  showKpis: true,
  showHighlights: true,
  showNews: false,
  showRecentCases: true,
  showNewsDigest: true,
  showGaps: true,
};

export default function HomeDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [panelSettings, setPanelSettings] = useState<HomePanelSettings>(DEFAULT_PANEL_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [leaseNewsFocus, setLeaseNewsFocus] = useState<DashboardStats["lease_news_focus"] | null>(null);
  const leaseNewsReflection = stats?.lease_news_reflection;
  const [leaseNewsBrief, setLeaseNewsBrief] = useState<DashboardStats["lease_news_brief"] | null>(null);
  const [newsPrefecture, setNewsPrefecture] = useState("");
  const [showDailyNewsBrief, setShowDailyNewsBrief] = useState(false);
  const newsPrefectureReadyRef = useRef(false);
  const briefRequestSeqRef = useRef(0);

  const [recentNews, setRecentNews] = useState<NewsSummaryItem[]>([]);
  const [newsFormOpen, setNewsFormOpen] = useState(false);
  const [newsUrl, setNewsUrl] = useState("");
  const [newsBody, setNewsBody] = useState("");
  const [newsSubmitting, setNewsSubmitting] = useState(false);
  const [homeShionMessage, setHomeShionMessage] = useState("");
  const [newsResult, setNewsResult] = useState<{
    title: string;
    summary_lines: string[];
    usage_memo: string;
    summary_codes?: string[];
    usage_codes?: string[];
    key_phrases?: string[];
    tags: string[];
    importance: string;
  } | null>(null);

  useEffect(() => {
    // 画面マウント時にめぶきちゃんを更新
    triggerMebuki('guide', '紫苑の審査室を開きました。\n昨日からの変化と、今日の判断材料を一緒に見ていきます。');

    try {
      const raw = window.localStorage.getItem(HOME_SETTINGS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<HomePanelSettings>;
        setPanelSettings((prev) => ({ ...prev, ...parsed }));
      }
    } catch {
      // ignore
    }

    let initialPrefecture = "";
    try {
      const rawPref = window.localStorage.getItem("lease-news-prefecture-hint") || "";
      if (rawPref) {
        initialPrefecture = rawPref;
        setNewsPrefecture(rawPref);
      }
    } catch {
      // ignore
    }

    const fetchStats = async () => {
      try {
        const res = await apiClient.get(`/api/dashboard/stats`);
        setStats(res.data);
      } catch (err) {
        console.error("Failed to load dashboard stats", err);
      } finally {
        setLoading(false);
      }
    };
    const fetchRecentNews = async () => {
      try {
        const res = await apiClient.get(`/api/lease-news/recent?limit=5`);
        setRecentNews(res.data.items || []);
      } catch {
        // ignore
      }
    };
    const fetchLeaseNewsFocus = async () => {
      try {
        const res = await apiClient.get(`/api/lease-news/focus`);
        setLeaseNewsFocus(res.data || null);
      } catch {
        // ignore
      }
    };
    fetchStats();
    fetchRecentNews();
    fetchLeaseNewsFocus();
    const activityKey = `lease-intelligence-activity:home:${formatLocalDateKey(new Date())}`;
    if (!window.sessionStorage.getItem(activityKey)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "home",
        action: "page_view",
        event_id: activityKey,
      }).then(() => window.sessionStorage.setItem(activityKey, "1")).catch(() => {});
    }
    newsPrefectureReadyRef.current = true;
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(HOME_SETTINGS_KEY, JSON.stringify(panelSettings));
    } catch {
      // ignore
    }
  }, [panelSettings]);

  useEffect(() => {
    if (!newsPrefectureReadyRef.current) return;
    // 1キーストロークごとの API 呼び出しを避けるためデバウンスする
    const timer = window.setTimeout(() => {
      const normalized = normalizePrefecture(newsPrefecture);
      const nextPrefecture = normalized || "";
      try {
        if (nextPrefecture) {
          window.localStorage.setItem("lease-news-prefecture-hint", nextPrefecture);
        } else {
          window.localStorage.removeItem("lease-news-prefecture-hint");
        }
      } catch {
        // ignore
      }
      loadLeaseNewsBrief(nextPrefecture);
    }, 400);
    return () => window.clearTimeout(timer);
  }, [newsPrefecture]);

  const analysis = stats?.analysis;
  const recentCases = stats?.recent_cases || [];
  const improvementHighlights = stats?.improvement_highlights?.items || [];
  const improvementCounts = stats?.improvement_highlights?.counts;
  const gapHighlights = stats?.lease_system_gaps?.items || [];
  const gapCounts = stats?.lease_system_gaps?.counts || {};
  const newsFocus = leaseNewsFocus ?? stats?.lease_news_focus;

  const avgScoreBorrower = analysis?.avg_score_borrower ?? null;

  const togglePanel = (key: keyof HomePanelSettings) => {
    setPanelSettings((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const loadLeaseNewsBrief = async (prefectureHint: string) => {
    const seq = ++briefRequestSeqRef.current;
    try {
      const res = await apiClient.get(`/api/lease-news/brief`, {
        params: {
          prefecture: prefectureHint,
        },
      });
      // 後発リクエストが既にある場合、古いレスポンスで上書きしない
      if (seq !== briefRequestSeqRef.current) return;
      setLeaseNewsBrief(res.data || null);
      const showKey = `lease-news-brief-seen-${formatLocalDateKey()}`;
      const seen = window.localStorage.getItem(showKey);
      const available = Boolean(res.data?.available);
      // 一度表示したら入力編集中に閉じない（既読判定は自動表示の初回のみ）
      setShowDailyNewsBrief((prev) => prev || (!seen && available));
      if (!seen && available) {
        window.localStorage.setItem(showKey, "1");
      }
    } catch {
      if (seq !== briefRequestSeqRef.current) return;
      setLeaseNewsBrief(null);
    }
  };

  const handleNewsSubmit = async () => {
    if (!newsUrl.trim() && !newsBody.trim()) return;
    setNewsSubmitting(true);
    setNewsResult(null);
    try {
      const res = await apiClient.post(`/api/lease-news/summarize`, {
        url: newsUrl.trim(),
        body_text: newsBody.trim(),
      });
      setNewsResult(res.data);
      setNewsUrl("");
      setNewsBody("");
      const updated = await apiClient.get(`/api/lease-news/recent?limit=5`);
      setRecentNews(updated.data.items || []);
    } catch (err) {
      console.error("News summarization failed", err);
    } finally {
      setNewsSubmitting(false);
    }
  };

  const openShionChatFromHome = () => {
    const message = homeShionMessage.trim() || "こんにちは。今日は何から始める？";
    window.localStorage.setItem(HOME_SHION_CHAT_DRAFT_KEY, message);
    window.location.href = "/chat";
  };

  return (
    <div className="p-4 sm:p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 sm:mb-12 relative overflow-hidden bg-[linear-gradient(135deg,#4c1d95_0%,#1e3a8a_48%,#064e3b_100%)] rounded-2xl sm:rounded-[2.5rem] p-6 sm:p-10 text-white shadow-2xl shadow-violet-500/20 group">
        <button
          onClick={() => setSettingsOpen(true)}
          className="absolute top-4 right-4 z-20 inline-flex items-center gap-1.5 rounded-xl bg-white/20 backdrop-blur-md border border-white/30 px-3 py-2 text-sm font-bold text-white hover:bg-white/30 transition-colors"
        >
          <Settings className="h-4 w-4" />
          カスタマイズ
        </button>
        <div className="relative z-10 max-w-2xl">
          <p className="mb-3 text-[11px] font-black uppercase tracking-[0.28em] text-violet-100/75">
            SHION JUDGMENT ROOM
          </p>
          <h1 className="text-2xl sm:text-4xl font-black mb-4 flex items-center gap-3 sm:gap-4">
            <span className="bg-white/20 p-2 sm:p-3 rounded-xl sm:rounded-2xl backdrop-blur-md">
              <Brain className="w-8 h-8 text-white" />
            </span>
            紫苑の審査室
          </h1>
          <p className="text-blue-100 text-sm sm:text-lg font-bold leading-relaxed mb-6 sm:mb-8 opacity-90">
            おかえりなさい。昨日からの変化、今日の案件、残しておくべき判断をここに集めました。<br/>
            一般論ではなく、Userのリース判断資産として読み返せる形で返します。
          </p>
          <div className="flex flex-wrap gap-3">
            {[
              { label: "審査分析", href: "/screening" },
              { label: "紫苑チャット", href: "/chat" },
              { label: "帰還データ検疫", href: "/cloudrun-return-review" },
            ].map((item) => (
              <button
                key={item.href}
                type="button"
                onClick={() => window.location.href = item.href}
                className="rounded-xl border border-white/25 bg-white/15 px-4 py-2 text-xs font-black text-white backdrop-blur-md transition-colors hover:bg-white/25"
              >
                {item.label}
              </button>
            ))}
          </div>
          <div className="mt-4 max-w-xl rounded-2xl border border-white/25 bg-white/15 p-2 backdrop-blur-md">
            <div className="mb-1.5 flex items-center gap-2 px-2 text-[11px] font-black uppercase tracking-wide text-blue-100">
              <MessageCircle className="h-3.5 w-3.5" />
              Shion Quick Chat
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={homeShionMessage}
                onChange={(e) => setHomeShionMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key !== "Enter") return;
                  e.preventDefault();
                  openShionChatFromHome();
                }}
                placeholder="こんにちは。今日は何がしたいですか？"
                className="min-w-0 flex-1 rounded-xl border border-white/20 bg-white/90 px-3 py-2 text-sm font-bold text-slate-800 outline-none placeholder:text-slate-400 focus:ring-2 focus:ring-white/70"
              />
              <button
                type="button"
                onClick={openShionChatFromHome}
                className="inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-white px-3 py-2 text-xs font-black text-indigo-700 shadow-sm transition-colors hover:bg-blue-50"
              >
                <Send className="h-4 w-4" />
                聞く
              </button>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            <div className="bg-white/10 backdrop-blur-md border border-white/20 px-3 sm:px-6 py-2 sm:py-3 rounded-xl sm:rounded-2xl">
              <div className="text-[10px] font-black uppercase tracking-widest text-blue-200">System Status</div>
              <div className="text-xs sm:text-sm font-black flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                {loading ? "ONLINE & SYNCING" : "ONLINE & SYNCED"}
              </div>
            </div>
            <div className="bg-white/10 backdrop-blur-md border border-white/20 px-3 sm:px-6 py-2 sm:py-3 rounded-xl sm:rounded-2xl">
              <div className="text-[10px] font-black uppercase tracking-widest text-blue-200">Memory Loop</div>
              <div className="text-xs sm:text-sm font-black">SHION CONTINUITY ACTIVE</div>
            </div>
          </div>
        </div>

        {/* めぶきちゃんの画像 */}
        <div className="absolute right-0 bottom-0 h-full w-[40%] hidden lg:block select-none pointer-events-none">
          <div className="relative h-full w-full">
            <img 
              src="/mebuki.png" 
              alt="Mebuki" 
              className="absolute bottom-0 right-10 h-[110%] object-contain drop-shadow-[0_20px_50px_rgba(0,0,0,0.3)] group-hover:scale-105 transition-transform duration-700 ease-out"
            />
          </div>
        </div>
        
        {/* 装飾用背景 */}
        <div className="absolute top-[-20%] right-[-10%] w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-20%] left-[-10%] w-64 h-64 bg-blue-400/20 rounded-full blur-3xl"></div>
      </div>

      {showDailyNewsBrief && leaseNewsBrief?.available && (
        <section className="mb-6 rounded-2xl border border-amber-200 bg-amber-50/90 p-4 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-xs font-black uppercase tracking-wide text-amber-700">今日のニュースブリーフ</div>
              <h2 className="mt-1 text-sm font-black text-amber-950">
                {leaseNewsBrief.opening_line || "今日はこのようなニュースがあります。"}
              </h2>
              <p className="mt-1 text-xs font-bold leading-relaxed text-amber-800">
                {leaseNewsBrief.question_line || "この案件で、今日は何を先に確認しますか？"}
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:min-w-56">
              <input
                type="text"
                value={newsPrefecture}
                onChange={(e) => setNewsPrefecture(e.target.value)}
                placeholder="取引地域（例: 大阪府）"
                className="w-full rounded-xl border border-amber-300 bg-white px-3 py-2 text-xs font-bold text-slate-700 outline-none placeholder:text-slate-400"
              />
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => loadLeaseNewsBrief(normalizePrefecture(newsPrefecture))}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
                >
                  更新
                </button>
                <button
                  type="button"
                  onClick={() => window.location.href = "/chat"}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
                >
                  AICHATで相談
                </button>
              </div>
            </div>
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            <div className="rounded-xl border border-white/80 bg-white/80 p-3">
              <div className="text-[10px] font-black uppercase tracking-wide text-slate-400">全国論点</div>
              <p className="mt-1 text-xs font-bold leading-relaxed text-slate-700">
                {leaseNewsBrief.national_headline || leaseNewsFocus?.headline || "最新ニュースの論点を表示します"}
              </p>
              {leaseNewsBrief.national_focus_lines?.length ? (
                <ul className="mt-2 space-y-1">
                  {leaseNewsBrief.national_focus_lines.slice(0, 3).map((line, i) => (
                    <li key={i} className="text-[11px] leading-relaxed text-slate-600">・{line}</li>
                  ))}
                </ul>
              ) : null}
            </div>
            <div className="rounded-xl border border-white/80 bg-white/80 p-3">
              <div className="text-[10px] font-black uppercase tracking-wide text-slate-400">地域論点</div>
              {leaseNewsBrief.regional_available ? (
                <>
                  <p className="mt-1 text-xs font-bold leading-relaxed text-slate-700">
                    {leaseNewsBrief.regional_title}
                  </p>
                  {leaseNewsBrief.regional_summary_lines?.length ? (
                    <ul className="mt-2 space-y-1">
                      {leaseNewsBrief.regional_summary_lines.slice(0, 3).map((line, i) => (
                        <li key={i} className="text-[11px] leading-relaxed text-slate-600">・{line}</li>
                      ))}
                    </ul>
                  ) : leaseNewsBrief.regional_usage_memo ? (
                    <p className="mt-2 text-[11px] leading-relaxed text-slate-600">{leaseNewsBrief.regional_usage_memo}</p>
                  ) : null}
                </>
              ) : (
                <p className="mt-1 text-xs text-slate-500">取引地域を入れると地域論点も出ます。</p>
              )}
            </div>
          </div>
        </section>
      )}


      {!analysis && (
        <div className="bg-amber-50 border border-amber-200 p-6 rounded-2xl flex items-start gap-4">
          <TrendingUp className="w-8 h-8 text-amber-500 shrink-0" />
          <div>
            <h3 className="font-bold text-amber-800 text-lg">成約データが不足しています</h3>
            <p className="text-amber-700 mt-1">成約データが5件以上貯まると、AIによる成約要因分析・実績集計がこの画面に表示されます。「結果登録」画面から最終ステータスを登録してください。</p>
          </div>
        </div>
      )}

      <div className="space-y-6 mt-6">
        {panelSettings.showKpis && analysis && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">総成約数 (分析対象)</p>
                  <h3 className="text-4xl font-black text-slate-800 mt-2">{analysis.closed_count} <span className="text-lg font-bold text-slate-400">件</span></h3>
                </div>
                <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center">
                  <Target className="w-6 h-6" />
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">成約平均 定性スコア</p>
                  <h3 className="text-4xl font-black text-indigo-700 mt-2">
                    {analysis.qualitative_summary?.avg_weighted?.toFixed(1) || "-"}
                    <span className="text-lg font-bold text-slate-400"> / 100</span>
                  </h3>
                </div>
                <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center">
                  <Users className="w-6 h-6" />
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">成約平均 信用スコア</p>
                  <h3 className="text-4xl font-black text-emerald-600 mt-2">
                    {avgScoreBorrower !== null ? avgScoreBorrower.toFixed(1) : "-"}
                    <span className="text-lg font-bold text-slate-400"> %</span>
                  </h3>
                </div>
                <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center">
                  <BarChart3 className="w-6 h-6" />
                </div>
              </div>
            </div>
          </div>
        )}

        {(panelSettings.showHighlights || panelSettings.showNews) && (
          <div className={`grid grid-cols-1 gap-6 ${panelSettings.showHighlights && panelSettings.showNews ? "lg:grid-cols-2" : ""}`}>
            {panelSettings.showHighlights && (
              <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <div className="flex items-center justify-between gap-3 mb-4">
                  <div>
                    <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                      <ScrollText className="text-amber-500 w-5 h-5" />
                      最新の改善項目
                    </h3>
                    <p className="text-xs text-slate-500 font-bold mt-1">
                      朝に見る候補は最大3件まで
                    </p>
                  </div>
                  {stats?.improvement_highlights?.source && (
                    <span className="text-[10px] font-bold text-slate-400 break-all max-w-40 text-right">
                      {stats.improvement_highlights.source}
                    </span>
                  )}
                </div>
                {improvementHighlights.length === 0 ? (
                  <p className="text-sm text-slate-500">改善候補がまだありません。</p>
                ) : (
                  <div className="space-y-3">
                    {improvementHighlights.map((item) => (
                      <div key={item.id || item.canonical_key || item.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-black text-white">{item.id || "-"}</span>
                          <span className="rounded-full bg-amber-100 px-2.5 py-1 text-[10px] font-black text-amber-700">{item.status || "要確認"}</span>
                          {item.priority && <span className="rounded-full bg-slate-200 px-2.5 py-1 text-[10px] font-black text-slate-600">{item.priority}</span>}
                        </div>
                        <div className="mt-2 font-bold text-slate-800">{item.title || "-"}</div>
                        <p className="mt-1 text-xs leading-relaxed text-slate-600">{item.reason || "理由なし"}</p>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {panelSettings.showNews && (
              <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <div className="flex items-center justify-between gap-3 mb-4">
                  <div>
                    <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                      <Newspaper className="text-sky-500 w-5 h-5" />
                      リースニュースの注目論点
                    </h3>
                    <p className="text-xs text-slate-500 font-bold mt-1">
                      {newsFocus?.headline || "最新ニュースの論点を表示"}
                    </p>
                  </div>
                  {newsFocus?.note_date && (
                    <span className="text-[10px] font-bold text-slate-400">{newsFocus.note_date}</span>
                  )}
                </div>
                {!newsFocus?.available ? (
                  <p className="text-sm text-slate-500">ニュース要約がまだありません。</p>
                ) : (
                  <div className="space-y-4">
                    {(newsFocus.theme_summary || newsFocus.bucket_summary || newsFocus.tag_summary) && (
                      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
                        {newsFocus.theme_summary && <p><span className="font-bold">主なテーマ:</span> {newsFocus.theme_summary}</p>}
                        {newsFocus.bucket_summary && <p className="mt-1"><span className="font-bold">収集セット:</span> {newsFocus.bucket_summary}</p>}
                        {newsFocus.tag_summary && <p className="mt-1"><span className="font-bold">重点タグ:</span> {newsFocus.tag_summary}</p>}
                      </div>
                    )}
                    <div className="space-y-2">
                      {(newsFocus.focus_lines || []).slice(0, 4).map((line, index) => (
                        <div key={index} className="flex items-start gap-2 rounded-xl border border-slate-200 bg-white p-3">
                          <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-sky-100 text-[10px] font-black text-sky-700">
                            {index + 1}
                          </span>
                          <p className="text-sm leading-relaxed text-slate-700">{line}</p>
                        </div>
                      ))}
                    </div>
                    {newsFocus.note_path && (
                      <p className="break-all text-[11px] text-slate-400">{newsFocus.note_path}</p>
                    )}
                    {(newsFocus.article_titles || []).length > 0 && (
                      <div className="rounded-xl border border-sky-100 bg-sky-50 p-4">
                        <p className="mb-2 text-[10px] font-black uppercase tracking-widest text-sky-600">注目記事</p>
                        <div className="space-y-2">
                          {(newsFocus.article_titles || []).slice(0, 3).map((title, index) => (
                            <p key={index} className="text-xs font-bold leading-relaxed text-slate-700">{title}</p>
                          ))}
                        </div>
                      </div>
                    )}
                    {(newsFocus.memo_lines || []).length > 0 && (
                      <div className="rounded-xl border border-amber-100 bg-amber-50 p-4">
                        <p className="mb-2 text-[10px] font-black uppercase tracking-widest text-amber-600">審査メモ</p>
                        <div className="space-y-1.5">
                          {(newsFocus.memo_lines || []).slice(0, 3).map((line, index) => (
                            <p key={index} className="text-xs leading-relaxed text-amber-900">{line}</p>
                          ))}
                        </div>
                      </div>
                    )}
                    {(newsFocus.metrics_lines || []).length > 0 && (
                      <div className="grid grid-cols-2 gap-2">
                        {(newsFocus.metrics_lines || []).slice(0, 4).map((line, index) => {
                          const [label, value] = line.split(":").map((part) => part.trim());
                          return (
                            <div key={index} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                              <p className="text-[10px] font-black text-slate-400">{label || "指標"}</p>
                              <p className="mt-1 text-sm font-black text-slate-700">{value || line}</p>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </section>
            )}

            {panelSettings.showNews && leaseNewsReflection?.available && (
              <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <div className="flex items-center justify-between gap-3 mb-4">
                  <div>
                    <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                      <ScrollText className="text-amber-500 w-5 h-5" />
                      紫苑の観察ログ
                    </h3>
                    <p className="text-xs text-slate-500 font-bold mt-1">
                      {leaseNewsReflection.headline || "ニュースを見て考えたこと"}
                    </p>
                  </div>
                  {leaseNewsReflection.note_date && (
                    <span className="text-[10px] font-bold text-slate-400">{leaseNewsReflection.note_date}</span>
                  )}
                </div>
                <div className="space-y-4">
                  {leaseNewsReflection.illustration_url && (
                    <img
                      src={leaseNewsReflection.illustration_url}
                      alt="紫苑の観察ログ"
                      className="aspect-[16/9] w-full rounded-xl border border-amber-100 object-cover"
                    />
                  )}
                  {(leaseNewsReflection.theme_summary || leaseNewsReflection.tag_summary) && (
                    <div className="rounded-xl border border-amber-100 bg-amber-50 p-4 text-sm text-amber-900">
                      {leaseNewsReflection.theme_summary && <p><span className="font-bold">テーマ:</span> {leaseNewsReflection.theme_summary}</p>}
                      {leaseNewsReflection.tag_summary && <p className="mt-1"><span className="font-bold">重点タグ:</span> {leaseNewsReflection.tag_summary}</p>}
                    </div>
                  )}
                  {(leaseNewsReflection.self_narrative || leaseNewsReflection.current_question) && (
                    <div className="rounded-xl border border-violet-200 bg-violet-50 p-4">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <p className="flex items-center gap-2 text-sm font-black text-violet-900">
                          <Brain className="h-4 w-4" />
                          判断の持ち越し
                        </p>
                      </div>
                      {(leaseNewsReflection.ultimate_goal || leaseNewsReflection.primary_goal || leaseNewsReflection.secondary_goal) && (
                        <div className="mb-3 rounded-lg border border-violet-200 bg-white p-3 text-xs leading-relaxed text-slate-700">
                          {leaseNewsReflection.ultimate_goal && (
                            <p>
                              <span className="font-bold text-violet-900">最終目標:</span> {leaseNewsReflection.ultimate_goal}
                              {leaseNewsReflection.ultimate_goal_status && (
                                <span className="ml-2 text-[10px] text-violet-600">({leaseNewsReflection.ultimate_goal_status})</span>
                              )}
                            </p>
                          )}
                          {leaseNewsReflection.primary_goal && (
                            <p className="mt-2"><span className="font-bold text-violet-800">第一目標:</span> {leaseNewsReflection.primary_goal}</p>
                          )}
                          {leaseNewsReflection.secondary_goal && (
                            <p className="mt-1"><span className="font-bold text-violet-800">第二目標:</span> {leaseNewsReflection.secondary_goal}</p>
                          )}
                        </div>
                      )}
                      {leaseNewsReflection.dominant_mood && (
                        <p className="text-xs text-violet-800">
                          <span className="font-bold">いま強い気分:</span> {leaseNewsReflection.dominant_mood}
                        </p>
                      )}
                      {leaseNewsReflection.self_narrative && (
                        <p className="mt-2 text-sm leading-relaxed text-slate-700">{leaseNewsReflection.self_narrative}</p>
                      )}
                      {leaseNewsReflection.memory_excerpt && (
                        <p className="mt-3 rounded-lg bg-white p-3 text-xs leading-relaxed text-slate-600">
                          <span className="font-bold text-violet-800">思い出したこと:</span> {leaseNewsReflection.memory_excerpt}
                        </p>
                      )}
                      {leaseNewsReflection.current_question && (
                        <p className="mt-3 text-xs leading-relaxed text-violet-900">
                          <span className="font-bold">明日へ持ち越す問い:</span> {leaseNewsReflection.current_question}
                        </p>
                      )}
                      {(leaseNewsReflection.user_understanding || leaseNewsReflection.user_curiosity) && (
                        <div className="mt-3 border-t border-violet-200 pt-3">
                          <div className="flex flex-wrap items-center gap-2">
                            {(leaseNewsReflection.user_interests || []).map((interest) => (
                              <span key={interest} className="rounded-full bg-white px-2 py-1 text-[10px] font-bold text-violet-700">
                                {interest}
                              </span>
                            ))}
                          </div>
                          {leaseNewsReflection.user_understanding && (
                            <p className="mt-2 text-xs leading-relaxed text-slate-700">
                              <span className="font-bold text-violet-800">あなたへの暫定理解:</span> {leaseNewsReflection.user_understanding}
                            </p>
                          )}
                          {leaseNewsReflection.user_curiosity && (
                            <p className="mt-2 text-xs leading-relaxed text-violet-900">
                              <span className="font-bold">あなたについて知りたいこと:</span> {leaseNewsReflection.user_curiosity}
                            </p>
                          )}
                        </div>
                      )}
                      <div className="mt-3 border-t border-violet-200 pt-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[10px] font-black text-violet-700">Obsidian知識接続</span>
                          <span className={`rounded-full px-2 py-1 text-[10px] font-bold ${
                            leaseNewsReflection.knowledge_available
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-slate-100 text-slate-600"
                          }`}>
                            {leaseNewsReflection.knowledge_available ? "接続済み" : "未接続"}
                          </span>
                          <span className="text-[10px] text-violet-700">
                            検索可能 {leaseNewsReflection.indexed_notes || 0}件 / 当日参照 {leaseNewsReflection.knowledge_source_count || 0}件
                          </span>
                        </div>
                        {leaseNewsReflection.knowledge_scope && (
                          <p className="mt-2 text-[10px] leading-relaxed text-slate-600">
                            {leaseNewsReflection.knowledge_scope}
                          </p>
                        )}
                        {(leaseNewsReflection.knowledge_sources || []).length > 0 && (
                          <p className="mt-2 text-[10px] leading-relaxed text-violet-800">
                            参照: {(leaseNewsReflection.knowledge_sources || []).slice(0, 3).join(" / ")}
                          </p>
                        )}
                      </div>
                      <p className="mt-3 text-[10px] text-violet-500">
                        アプリ内の行動種別・回数・関心カテゴリだけを使用します。質問本文や個人属性は記憶しません。
                      </p>
                    </div>
                  )}
                  {(leaseNewsReflection.thought_lines || []).length > 0 && (
                    <div className="space-y-2">
                      {(leaseNewsReflection.thought_lines || []).slice(0, 4).map((line, index) => (
                        <div key={index} className="flex items-start gap-2 rounded-xl border border-amber-100 bg-white p-3">
                          <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-black text-amber-700">
                            {index + 1}
                          </span>
                          <p className="text-sm leading-relaxed text-slate-700">{line}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  {(leaseNewsReflection.tomorrow_lines || []).length > 0 && (
                    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                      <p className="mb-2 text-[10px] font-black uppercase tracking-widest text-slate-500">明日見ること</p>
                      <div className="space-y-1.5">
                        {(leaseNewsReflection.tomorrow_lines || []).slice(0, 3).map((line, index) => (
                          <p key={index} className="text-xs leading-relaxed text-slate-700">{line}</p>
                        ))}
                      </div>
                    </div>
                  )}
                  {leaseNewsReflection.note_path && (
                    <p className="break-all text-[11px] text-slate-400">{leaseNewsReflection.note_path}</p>
                  )}
                </div>
              </section>
            )}
          </div>
        )}

        {panelSettings.showGaps && (
          <section className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-800 text-lg flex items-center gap-2">
                  <AlertTriangle className="text-rose-500 w-5 h-5" />
                  不足項目・改善診断
                </h3>
                <p className="text-xs text-slate-500 font-bold mt-1">
                  本体非連動の読み取り専用診断。詳細は改善ログへ。
                </p>
              </div>
              <div className="flex flex-wrap gap-2 text-[10px] font-black">
                <span className="rounded-full bg-rose-100 px-2.5 py-1 text-rose-700">Critical {gapCounts.critical || 0}</span>
                <span className="rounded-full bg-amber-100 px-2.5 py-1 text-amber-700">High {gapCounts.high || 0}</span>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-600">Total {gapHighlights.length}</span>
              </div>
            </div>
            {gapHighlights.length === 0 ? (
              <p className="text-sm text-slate-500">不足項目レポートがまだありません。</p>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                {gapHighlights.slice(0, 3).map((item) => (
                  <div key={item.id || item.title} className="rounded-xl border border-slate-200 bg-slate-50 p-4 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-[11px] font-black text-slate-500">{item.id || "-"}</span>
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-black ${gapPriorityClass(item.priority)}`}>
                        {item.priority || "unknown"}
                      </span>
                      {item.category && (
                        <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-black text-slate-500">
                          {item.category}
                        </span>
                      )}
                    </div>
                    <h4 className="mt-2 text-sm font-black leading-snug text-slate-800">{item.title || "-"}</h4>
                    {item.impact && (
                      <p className="mt-2 text-xs leading-relaxed text-slate-600">{item.impact}</p>
                    )}
                    {item.recommended_action && (
                      <p className="mt-2 text-[11px] leading-relaxed text-slate-700">
                        <span className="font-black">次:</span> {item.recommended_action}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {analysis && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 成約要因トップ3 */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-800 text-lg mb-4 flex items-center gap-2">
                <TrendingUp className="text-rose-500 w-5 h-5" />
                成約要因トップ3ドライバー
              </h3>
              <p className="text-xs text-slate-500 font-bold mb-6">成約に最も寄与している因子（回帰分析結果）</p>

              <div className="space-y-4">
                {analysis.top3_drivers?.map((d: TopDriver, index: number) => (
                  <div key={index} className="flex items-center gap-3 sm:gap-4 bg-slate-50 p-3 sm:p-4 rounded-xl border border-slate-100 min-w-0">
                    <div className="w-8 h-8 shrink-0 bg-slate-800 text-white rounded-full flex items-center justify-center font-black">
                      {index + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-bold text-slate-700 truncate" title={d.label}>{d.label}</div>
                      <div className="text-[11px] sm:text-xs text-slate-500 mt-0.5 truncate" title={`回帰係数: ${(d.coef || 0).toFixed(4)}`}>
                        回帰係数: <span className="tabular-nums">{(d.coef || 0).toFixed(4)}</span>
                      </div>
                    </div>
                    <div className={`shrink-0 text-xs sm:text-sm font-bold px-2 sm:px-3 py-1 rounded border ${
                      d.direction === 'プラス' ? 'bg-green-50 text-green-700 border-green-200' : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {d.direction}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 成約案件 平均財務数値 */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 h-96 flex flex-col">
              <h3 className="font-bold text-slate-800 text-lg mb-6 flex items-center gap-2">
                <BarChart3 className="text-blue-500 w-5 h-5" />
                成約案件の平均財務主要指標
              </h3>

              <div className="flex-1 overflow-auto border rounded-xl border-slate-200 bg-slate-50">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-100/80 sticky top-0 font-bold text-slate-600">
                    <tr>
                      <th className="px-4 py-3 border-b">指標名</th>
                      <th className="px-4 py-3 border-b text-right">成約平均値</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.avg_financials && Object.entries(analysis.avg_financials)
                      .map(([k, v]: [string, number | string | null]) => (
                      <tr key={k} className="border-b last:border-b-0 hover:bg-white transition-colors bg-white">
                        <td className="px-4 py-3 font-semibold text-slate-700">{k}</td>
                        <td className="px-4 py-3 text-right font-black text-indigo-700">
                          {typeof v === 'number' ? v.toLocaleString('ja-JP', { maximumFractionDigits: 1 }) : v}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ニュースダイジェスト */}
      {panelSettings.showNewsDigest && (
        <section className="mt-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-black text-2xl text-slate-800 border-l-4 border-sky-600 pl-3 flex items-center gap-2">
              <Newspaper className="w-6 h-6 text-sky-500" />
              最新リースニュース
            </h3>
            <button
              onClick={() => setNewsFormOpen((prev) => !prev)}
              className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-4 py-2 text-sm font-bold text-white hover:bg-sky-700 transition-colors"
            >
              <Plus className="w-4 h-4" />
              ニュースを追加
            </button>
          </div>

          {newsFormOpen && (
            <div className="mb-6 rounded-2xl border border-sky-200 bg-sky-50 p-6">
              <h4 className="font-bold text-slate-800 mb-4">ニュース要約を作成</h4>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-bold text-slate-600 mb-1">URL</label>
                  <input
                    type="url"
                    value={newsUrl}
                    onChange={(e) => setNewsUrl(e.target.value)}
                    placeholder="https://example.com/news/..."
                    className="w-full rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm focus:border-sky-500 focus:ring-1 focus:ring-sky-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-bold text-slate-600 mb-1">または本文テキスト</label>
                  <textarea
                    value={newsBody}
                    onChange={(e) => setNewsBody(e.target.value)}
                    placeholder="ニュース記事の本文をペースト..."
                    rows={4}
                    className="w-full rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm focus:border-sky-500 focus:ring-1 focus:ring-sky-500 outline-none resize-none"
                  />
                </div>
                <button
                  onClick={handleNewsSubmit}
                  disabled={newsSubmitting || (!newsUrl.trim() && !newsBody.trim())}
                  className="inline-flex items-center gap-2 rounded-xl bg-sky-600 px-6 py-2.5 text-sm font-bold text-white hover:bg-sky-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {newsSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  {newsSubmitting ? "AI要約中..." : "AI要約して保存"}
                </button>
              </div>

              {newsResult && (
                <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                  <p className="font-bold text-emerald-800 mb-2">保存しました: {newsResult.title}</p>
                  <ul className="text-sm text-emerald-700 space-y-1">
                    {newsResult.summary_lines.map((line, i) => (
                      <li key={i}>• {line}</li>
                    ))}
                  </ul>
                  {(newsResult.key_phrases || []).length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {(newsResult.key_phrases || []).slice(0, 5).map((phrase, i) => (
                        <span key={i} className="rounded-full bg-white px-2 py-0.5 text-[10px] font-bold text-emerald-700 border border-emerald-200">
                          {phrase}
                        </span>
                      ))}
                    </div>
                  )}
                  <p className="mt-2 text-xs text-emerald-600">{newsResult.usage_memo}</p>
                </div>
              )}
            </div>
          )}

          {recentNews.length === 0 ? (
            <p className="text-sm text-slate-500">ニュースダイジェストがまだありません。「ニュースを追加」ボタンからニュースを登録してください。</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recentNews.map((item, i) => {
                const importanceBg =
                  item.importance === "高" ? "bg-rose-100 text-rose-700 border-rose-200" :
                  item.importance === "中" ? "bg-amber-100 text-amber-700 border-amber-200" :
                  "bg-slate-100 text-slate-600 border-slate-200";
                const regionBg =
                  item.region === "米国" ? "bg-blue-100 text-blue-700" :
                  item.region === "欧州" ? "bg-violet-100 text-violet-700" :
                  item.region === "アジア" ? "bg-emerald-100 text-emerald-700" :
                  "bg-sky-100 text-sky-700";
                return (
                  <div key={i} className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 hover:shadow-md transition-shadow flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-400">{item.date}</span>
                        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full ${regionBg}`}>
                          {item.region || "国内"}
                        </span>
                      </div>
                      <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${importanceBg}`}>
                        {item.importance === "高" ? "重要" : item.importance === "中" ? "注目" : "通常"}
                      </span>
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm mb-3 leading-snug">{item.title}</h4>
                    <div className="space-y-1.5 mb-3 flex-1">
                      {item.summary_lines.map((line, j) => (
                        <p key={j} className="text-xs text-slate-600 leading-relaxed flex items-start gap-1.5">
                          <span className="mt-0.5 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-sky-100 text-[9px] font-black text-sky-700">
                            {j + 1}
                          </span>
                          {line}
                        </p>
                      ))}
                    </div>
                    {item.usage_memo && (
                      <p className="text-[11px] text-amber-700 bg-amber-50 rounded-lg px-3 py-2 mb-3 border border-amber-100">
                        {item.usage_memo}
                      </p>
                    )}
                    {(item.key_phrases || []).length > 0 && (
                      <div className="mb-3 flex flex-wrap gap-1.5">
                        {(item.key_phrases || []).slice(0, 5).map((phrase, k) => (
                          <span key={k} className="rounded-full bg-slate-50 px-2 py-0.5 text-[10px] font-bold text-slate-500 border border-slate-200">
                            {phrase}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="flex flex-wrap gap-1.5 mt-auto">
                      {item.tags.map((tag, k) => (
                        <span key={k} className="inline-flex items-center gap-1 text-[10px] font-bold text-slate-500 bg-slate-100 rounded-full px-2 py-0.5">
                          <Tag className="w-2.5 h-2.5" />
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-2 flex-wrap">
                      {item.source && item.source !== "手動入力" && (
                        <span className="text-[10px] font-bold text-slate-400">{item.source}</span>
                      )}
                      {item.article_url && (
                        <a
                          href={item.article_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-[11px] text-sky-600 hover:text-sky-800 font-bold ml-auto"
                        >
                          <ExternalLink className="w-3 h-3" />
                          元記事を見る →
                        </a>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {/* 最近の案件一覧 */}
      {panelSettings.showRecentCases && (
        <>
          <h3 className="font-black text-2xl text-slate-800 mt-12 mb-6 border-l-4 border-blue-600 pl-3">📋 最新の案件履歴</h3>
      
          {recentCases.length === 0 ? (
            <p className="text-slate-500 font-bold">まだ案件履歴がありません。審査画面からデータを入力・実行してください。</p>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {recentCases.slice(0, 10).map((c: RecentCase, i: number) => {
                const isClosed = c.final_status === "成約";
                const isLost = c.final_status === "失注";
                return (
                  <div key={i} className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 hover:shadow-md transition-shadow relative overflow-hidden group">
                    <div className={`absolute top-0 left-0 w-2 h-full ${isClosed ? 'bg-green-500' : isLost ? 'bg-rose-500' : 'bg-slate-300'}`}></div>

                    <div className="flex justify-between items-start mb-3 ml-2">
                      <div className="flex items-center gap-2">
                        {isClosed ? <CheckCircle className="text-green-500 w-5 h-5" /> : isLost ? <XCircle className="text-rose-500 w-5 h-5" /> : <Activity className="text-slate-400 w-5 h-5" />}
                        <span className="text-xs font-bold text-slate-400">{c.timestamp?.slice(0, 16)}</span>
                      </div>
                      <div className="text-xl font-black text-slate-800">
                        <span className="text-xs font-bold text-slate-400 mr-2">審査スコア</span>
                        {c.result?.score != null ? c.result.score.toFixed(0) : "-"} <span className="text-sm">点</span>
                      </div>
                    </div>

                    <h4 className="font-bold text-lg text-slate-700 ml-2 mb-1">{c.industry_major || "不明な業種"} <span className="text-sm text-slate-500">- {c.industry_sub}</span></h4>
                    <div className="flex items-center gap-3 ml-2 mt-3">
                      <span className={`text-[10px] uppercase font-black px-2 py-1 rounded bg-slate-100 ${isClosed ? 'text-green-700' : isLost ? 'text-rose-700' : 'text-slate-500'}`}>
                        {c.final_status || "未登録"}
                      </span>
                      <span className="text-xs font-bold text-slate-500 bg-slate-50 px-2 py-1 rounded border">
                        判定: {c.result?.hantei || "不明"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* Slide-over customization panel */}
      {settingsOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40"
          onClick={() => setSettingsOpen(false)}
        />
      )}
      <div
        className={`fixed top-0 right-0 h-full w-80 bg-white shadow-2xl z-50 flex flex-col transition-transform duration-300 ease-in-out ${
          settingsOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between p-4 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-slate-600" />
            <h2 className="font-black text-slate-800 text-base">ホームのカスタマイズ</h2>
          </div>
          <button
            onClick={() => setSettingsOpen(false)}
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <p className="text-xs text-slate-500 mb-4">表示するパネルを切り替えます。設定はこのブラウザに保存されます。</p>
          <div className="space-y-3">
            {[
              { key: "showKpis", label: "KPI", desc: "総成約数と平均指標" },
              { key: "showHighlights", label: "改善項目", desc: "最新の改善候補" },
              { key: "showGaps", label: "不足項目", desc: "改善診断の上位3件" },
              { key: "showNews", label: "リースニュース", desc: "必要な時だけ表示する論点" },
              { key: "showNewsDigest", label: "ニュースダイジェスト", desc: "AI要約ニュース" },
              { key: "showRecentCases", label: "案件履歴", desc: "最近の成約・失注" },
            ].map((item) => {
              const checked = panelSettings[item.key as keyof HomePanelSettings];
              return (
                <label
                  key={item.key}
                  className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 cursor-pointer hover:bg-slate-100"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => togglePanel(item.key as keyof HomePanelSettings)}
                    className="mt-0.5 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="flex-1">
                    <span className="block text-sm font-bold text-slate-700">{item.label}</span>
                    <span className="block text-xs text-slate-500">{item.desc}</span>
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function gapPriorityClass(priority?: string) {
  const key = String(priority || "").toLowerCase();
  if (key === "critical") return "bg-rose-100 text-rose-800";
  if (key === "high") return "bg-amber-100 text-amber-800";
  if (key === "medium") return "bg-sky-100 text-sky-800";
  return "bg-slate-200 text-slate-700";
}
