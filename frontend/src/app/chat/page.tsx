"use client";

import React, { useState, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api";
import { Send, Trash2, Loader2, MessageCircle, Bot, User, NotebookPen, Mic, Network, Database, ChevronDown, ChevronUp, Lightbulb, Volume2, VolumeX } from "lucide-react";
import { extractPrefectureFromText, normalizePrefecture } from "@/lib/prefecture";
import { formatLocalDateKey } from "@/lib/date";

interface ChatMessage {
  id: number;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

type ChatContext = {
  score?: number;
  hantei?: string;
  score_borrower?: number;
  company_name?: string;
  asset_name?: string;
  asset_location?: string;
  prefecture?: string;
  industry_sub?: string;
  industry_major?: string;
  sales_dept?: string;
  quantum_risk?: number;
  case_id?: string;
};

const SHION_AVATAR = "/lease-intelligence/moods/curiosity.webp";
const SHION_THINKING_AVATAR = "/lease-intelligence/moods/focus.webp";

type LeaseNewsFocus = {
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

type LeaseNewsBrief = {
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

type SpeechRecognitionResultLike = ArrayLike<{ transcript: string }>;

interface SpeechRecognitionEventLike {
  results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;
type SpeechWindow = Window & {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
};

const getSpeechRecognition = (): SpeechRecognitionConstructor | null => {
  if (typeof window === "undefined") return null;
  const speechWindow = window as SpeechWindow;
  return speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition || null;
};

const normalizeMessageContent = (content: string) =>
  (content || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\t/g, "  ")
    .trim();

const renderInline = (text: string) => {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
};

const renderAssistantContent = (content: string) => {
  const lines = normalizeMessageContent(content).split("\n");
  const blocks: React.ReactNode[] = [];
  let listItems: string[] = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(
      <p key={`p-${blocks.length}`} className="mb-2 last:mb-0">
        {renderInline(paragraph.join(" "))}
      </p>
    );
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="mb-2 list-disc pl-5 space-y-1 last:mb-0">
        {listItems.map((item, i) => (
          <li key={i}>{renderInline(item)}</li>
        ))}
      </ul>
    );
    listItems = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    const bullet = line.match(/^[-*•]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      flushParagraph();
      listItems.push((bullet?.[1] || numbered?.[1] || "").trim());
      continue;
    }
    flushList();
    paragraph.push(line);
  }
  flushParagraph();
  flushList();
  return blocks.length ? blocks : <p>{normalizeMessageContent(content)}</p>;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [savingObsidian, setSavingObsidian] = useState(false);
  const [saveToast, setSaveToast] = useState<string | null>(null);
  const [showSubtitle, setShowSubtitle] = useState(true);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [recentCases, setRecentCases] = useState<{ id: string; company_name: string; score: number | null; final_status: string }[]>([]);
  const [showCasesPanel, setShowCasesPanel] = useState(false);
  const [improvementMode, setImprovementMode] = useState(false);
  const [answerMode, setAnswerMode] = useState<"shion" | "general">("shion");
  const [leaseNewsFocus, setLeaseNewsFocus] = useState<LeaseNewsFocus | null>(null);
  const [leaseNewsBrief, setLeaseNewsBrief] = useState<LeaseNewsBrief | null>(null);
  const [chatContext, setChatContext] = useState<ChatContext>({});
  const [newsPrefecture, setNewsPrefecture] = useState("");
  const [showDailyNewsBrief, setShowDailyNewsBrief] = useState(false);
  const [newsPrefectureReady, setNewsPrefectureReady] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const briefRequestSeqRef = useRef(0);

  const userId = "default";

  const speakText = (text: string) => {
    if (!speechEnabled || typeof window === "undefined" || !window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "ja-JP";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  };

  const scrollToBottom = () => {
    const el = messageListRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  useEffect(() => {
    loadHistory().then(() => {
      const raw = window.localStorage.getItem("lease-gunshi-context");
      if (raw) {
        window.localStorage.removeItem("lease-gunshi-context");
        try {
          const ctx = JSON.parse(raw) as ChatContext;
          setChatContext(ctx);
          const derivedPrefecture = normalizePrefecture(
            ctx.prefecture || extractPrefectureFromText(ctx.asset_location || "")
          );
          const cachedPrefecture = window.localStorage.getItem("lease-news-prefecture-hint") || "";
          const nextPrefecture = derivedPrefecture || cachedPrefecture;
          if (nextPrefecture) {
            setNewsPrefecture(nextPrefecture);
          }
          const lines: string[] = [
            `【審査結果の相談】${ctx.company_name ? ` ${ctx.company_name}` : ""}`,
            `・物件: ${ctx.asset_name ?? "—"}`,
            `・業種: ${ctx.industry_sub ?? "—"}`,
            `・営業部: ${ctx.sales_dept ?? "—"}`,
            `・総合スコア: ${ctx.score != null ? ctx.score.toFixed(1) + "点" : "—"}`,
            `・判定: ${ctx.hantei ?? "—"}`,
            `・借手スコア: ${ctx.score_borrower != null ? ctx.score_borrower.toFixed(1) + "点" : "—"}`,
          ];
          if (ctx.asset_location) {
            lines.push(`・設置場所: ${ctx.asset_location}`);
          }
          if (ctx.quantum_risk != null) {
            lines.push(`・量子リスク: ${ctx.quantum_risk.toFixed(1)}`);
          }
          lines.push("", "この案件について、審査上のポイントや懸念点を教えてください。");
          const autoMessage = lines.join("\n");
          window.setTimeout(() => sendMessageWithText(autoMessage), 400);
        } catch {
          // JSON parse失敗は無視
        }
      }
      setNewsPrefectureReady(true);
    });
    const timer = setTimeout(() => setShowSubtitle(false), 5000);
    setVoiceSupported(Boolean(getSpeechRecognition()));
    apiClient.get("/api/cases?limit=8&sort=desc")
      .then(res => setRecentCases(res.data || []))
      .catch(() => {});
    apiClient.get("/api/lease-news/focus")
      .then((res) => setLeaseNewsFocus(res.data || null))
      .catch(() => {});
    const activityDate = new Date().toLocaleDateString("sv-SE");
    const activityKey = `lease-intelligence-activity:chat:${activityDate}`;
    if (!window.sessionStorage.getItem(activityKey)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "chat",
        action: "page_view",
        event_id: activityKey,
      }).then(() => window.sessionStorage.setItem(activityKey, "1")).catch(() => {});
    }
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!newsPrefectureReady) return;
    // 1キーストロークごとの API 呼び出しを避けるためデバウンスする
    const timer = window.setTimeout(() => {
      const normalized = normalizePrefecture(newsPrefecture);
      const nextPrefecture = normalized || "";
      const cachedIndustry = chatContext.industry_sub || chatContext.industry_major || "";
      if (nextPrefecture) {
        window.localStorage.setItem("lease-news-prefecture-hint", nextPrefecture);
      } else {
        window.localStorage.removeItem("lease-news-prefecture-hint");
      }
      loadLeaseNewsBrief(nextPrefecture, cachedIndustry);
    }, 400);
    return () => window.clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newsPrefecture, chatContext.industry_sub, chatContext.industry_major]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await apiClient.get("/api/chat/history", {
        params: { user_id: userId, limit: 50 },
      });
      setMessages(res.data.messages || []);
    } catch {
      // 履歴取得失敗は無視して空スタートにする
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadLeaseNewsBrief = async (prefectureHint: string, industryHint: string) => {
    const seq = ++briefRequestSeqRef.current;
    try {
      const res = await apiClient.get("/api/lease-news/brief", {
        params: {
          prefecture: prefectureHint,
          industry: industryHint,
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

  const sendMessageWithText = async (text: string) => {
    if (!text.trim() || loading) return;
    const optimisticUser: ChatMessage = {
      id: Date.now(),
      user_id: userId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUser]);
    setLoading(true);
    try {
      const res = await apiClient.post("/api/chat", {
        message: text,
        user_id: userId,
        prefecture: normalizePrefecture(newsPrefecture),
        industry: chatContext.industry_sub || chatContext.industry_major || "",
        response_mode: answerMode,
      });
      if (res.data?.lease_news_focus) {
        setLeaseNewsFocus(res.data.lease_news_focus);
      }
      if (res.data?.lease_news_brief) {
        setLeaseNewsBrief(res.data.lease_news_brief);
      }
      const reply = res.data.reply as string;
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          user_id: userId,
          role: "assistant",
          content: reply,
          created_at: new Date().toISOString(),
        },
      ]);
      speakText(reply);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          user_id: userId,
          role: "assistant",
          content: "エラーが発生しました。もう一度お試しください。",
          created_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const optimisticUser: ChatMessage = {
      id: Date.now(),
      user_id: userId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUser]);
    setInput("");
    setLoading(true);

    try {
      const res = await apiClient.post("/api/chat", {
        message: text,
        user_id: userId,
        intent: improvementMode ? "improvement" : undefined,
        prefecture: normalizePrefecture(newsPrefecture),
        industry: chatContext.industry_sub || chatContext.industry_major || "",
        response_mode: answerMode,
      });
      if (res.data?.lease_news_focus) {
        setLeaseNewsFocus(res.data.lease_news_focus);
      }
      if (res.data?.lease_news_brief) {
        setLeaseNewsBrief(res.data.lease_news_brief);
      }
      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        user_id: userId,
        role: "assistant",
        content: res.data.reply,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      speakText(res.data.reply as string);
      if (improvementMode && res.data.improvement_saved) {
        setSaveToast("改善メモに登録しました");
        setTimeout(() => setSaveToast(null), 2000);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          user_id: userId,
          role: "assistant",
          content: "エラーが発生しました。もう一度お試しください。",
          created_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  const startVoiceInput = () => {
    if (!voiceSupported || loading) return;
    if (listening) {
      recognitionRef.current?.stop?.();
      setListening(false);
      return;
    }

    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) return;
    const recognition = new SpeechRecognition();
    recognition.lang = "ja-JP";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results || [])
        .map((result) => result?.[0]?.transcript || "")
        .join("")
        .trim();
      if (!transcript) return;
      setInput((prev) => `${prev}${prev.trim() ? "\n" : ""}${transcript}`);
      window.setTimeout(resizeTextarea, 0);
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    setListening(true);
    recognition.start();
  };

  const saveToObsidian = async () => {
    if (savingObsidian) return;
    setSavingObsidian(true);
    try {
      await apiClient.post("/api/chat/save-to-obsidian", { user_id: userId });
      setSaveToast("Obsidianに保存しました ✅");
      setTimeout(() => setSaveToast(null), 2000);
    } catch {
      setSaveToast("保存に失敗しました ❌");
      setTimeout(() => setSaveToast(null), 2000);
    } finally {
      setSavingObsidian(false);
    }
  };

  const clearHistory = async () => {
    if (!confirm("会話履歴を全て削除しますか？")) return;
    try {
      await apiClient.delete("/api/chat/history", { params: { user_id: userId } });
      setMessages([]);
    } catch {
      alert("削除に失敗しました");
    }
  };

  const insertCaseContext = (c: { company_name: string; score: number | null; final_status: string }) => {
    const snippet = `[案件参照] ${c.company_name || "名称なし"} スコア:${c.score != null ? Math.round(c.score) : "—"} ステータス:${c.final_status} `;
    setInput(prev => prev ? `${prev}\n${snippet}` : snippet);
    setShowCasesPanel(false);
    textareaRef.current?.focus();
    window.setTimeout(resizeTextarea, 0);
  };

  const openKnowledgeEvidence = (query: string) => {
    const trimmed = query.replace(/\s+/g, " ").trim().slice(0, 120);
    if (!trimmed) return;
    window.localStorage.setItem("knowledge-space-evidence", trimmed);
    window.open(`/knowledge-space?focus=${encodeURIComponent(trimmed)}`, "_blank");
  };

  const formatTime = (iso: string) => {
    try {
      return new Date(iso).toLocaleTimeString("ja-JP", {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  return (
    <div className="flex flex-col h-[calc(100dvh-4rem)] max-w-3xl mx-auto px-4 py-6 overflow-hidden">
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
            <MessageCircle className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black text-slate-800">AIアドバイザー</h1>
            <p
              className={`text-xs text-slate-500 transition-all duration-700 overflow-hidden ${
                showSubtitle ? "opacity-100 max-h-10" : "opacity-0 max-h-0"
              }`}
            >
              リース審査の相棒・毎日相談できます
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {saveToast && (
            <span className="text-xs font-bold text-emerald-600 bg-emerald-50 border border-emerald-200 rounded-lg px-2 py-1 whitespace-nowrap">
              {saveToast}
            </span>
          )}
          <button
            onClick={saveToObsidian}
            disabled={savingObsidian}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-emerald-600 transition-colors px-3 py-1.5 rounded-lg hover:bg-emerald-50 disabled:opacity-50"
          >
            {savingObsidian ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <NotebookPen className="w-3.5 h-3.5" />
            )}
            <span>Obsidianに保存</span>
          </button>
          <button
            onClick={clearHistory}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-red-500 transition-colors px-3 py-1.5 rounded-lg hover:bg-red-50"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>履歴削除</span>
          </button>
        </div>
      </div>

      {showDailyNewsBrief && leaseNewsBrief?.available && (
        <section className="flex-shrink-0 mb-3 rounded-2xl border border-amber-200 bg-amber-50/90 p-4 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-xs font-black uppercase tracking-wide text-amber-700">今日のニュースブリーフ</div>
              <h2 className="mt-1 text-sm font-black text-amber-950">
                {leaseNewsBrief.opening_line || "今日はこのようなニュースがあります。"}
              </h2>
              <p className="mt-1 text-xs font-bold leading-relaxed text-amber-800">
                {leaseNewsBrief.question_line || "この案件で、今日は何を先に確認しますか？"}
              </p>
              {leaseNewsBrief.geo_context && (
                <p className="mt-2 text-[11px] font-bold leading-relaxed text-amber-700 whitespace-pre-wrap">
                  {leaseNewsBrief.geo_context}
                </p>
              )}
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
                  onClick={() => {
                    const nextPref = normalizePrefecture(newsPrefecture);
                    loadLeaseNewsBrief(nextPref, chatContext.industry_sub || chatContext.industry_major || "");
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
                >
                  更新
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const prompt = leaseNewsBrief.question_line || "今日のニュースで審査上気にする点は？";
                    setInput((prev) => (prev ? `${prev}\n${prompt}` : prompt));
                    textareaRef.current?.focus();
                    window.setTimeout(resizeTextarea, 0);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
                >
                  この論点で相談
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

      {leaseNewsFocus?.available && (
        <section className="flex-shrink-0 mb-3 rounded-2xl border border-amber-200 bg-amber-50/80 p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-xs font-black uppercase tracking-wide text-amber-700">最新リースニュース</div>
              <h2 className="mt-1 text-sm font-black text-amber-950">
                {leaseNewsFocus.headline || leaseNewsFocus.theme_summary || "最新ニュースの注目論点"}
              </h2>
              {leaseNewsFocus.note_date && (
                <p className="mt-1 text-[11px] text-amber-700">更新: {leaseNewsFocus.note_date}</p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  const snippet = leaseNewsFocus.focus_lines?.slice(0, 3).map((line) => `- ${line}`).join("\n") || "";
                  if (!snippet) return;
                  setInput((prev) => (prev ? `${prev}\n${snippet}` : snippet));
                  textareaRef.current?.focus();
                  window.setTimeout(resizeTextarea, 0);
                }}
                className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
              >
                入力に追加
              </button>
              <button
                type="button"
                onClick={() => openKnowledgeEvidence(leaseNewsFocus.headline || leaseNewsFocus.theme_summary || leaseNewsFocus.tag_summary || "")}
                className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-bold text-amber-800 hover:bg-amber-100 transition-colors"
              >
                根拠ルート
              </button>
            </div>
          </div>
          {leaseNewsFocus.focus_lines?.length ? (
            <div className="mt-3 grid gap-1.5">
              {leaseNewsFocus.focus_lines.slice(0, 3).map((line, i) => (
                <p key={i} className="text-xs leading-relaxed text-amber-900">
                  ・{line}
                </p>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-xs text-amber-800">ニュースの注目論点を取得しました。</p>
          )}
        </section>
      )}

      {/* DB案件クイック参照パネル (REV-130) */}
      {recentCases.length > 0 && (
        <div className="flex-shrink-0 mb-2">
          <button
            onClick={() => setShowCasesPanel(p => !p)}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-blue-600 transition-colors px-2 py-1 rounded-lg hover:bg-blue-50"
          >
            <Database className="w-3.5 h-3.5" />
            <span>案件DB参照</span>
            {showCasesPanel ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
          {showCasesPanel && (
            <div className="mt-1 p-2 bg-white border border-slate-200 rounded-xl shadow-sm flex flex-wrap gap-1.5">
              {recentCases.map(c => (
                <button
                  key={c.id}
                  onClick={() => insertCaseContext(c)}
                  className={`text-xs px-2.5 py-1 rounded-full font-bold border transition-all hover:shadow-sm
                    ${c.final_status === '成約' ? 'bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100' :
                      c.final_status === '失注' ? 'bg-rose-50 border-rose-200 text-rose-700 hover:bg-rose-100' :
                      'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'}`}
                >
                  {c.company_name || "名称なし"} {c.score != null ? `(${Math.round(c.score)})` : ""}
                </button>
              ))}
              <p className="w-full text-[10px] text-slate-400 px-1 pt-0.5">クリックでチャットに案件情報を挿入します</p>
            </div>
          )}
        </div>
      )}

      {/* メッセージエリア */}
      <div ref={messageListRef} className="flex-1 min-h-0 overflow-y-auto overscroll-contain space-y-4 pr-1 pb-4">
        {historyLoading ? (
          <div className="flex justify-center items-center h-32">
            <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
          </div>
        ) : messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <Bot className="w-12 h-12 text-slate-300 mb-3" />
            <p className="text-slate-500 font-bold">こんにちは！</p>
            <p className="text-sm text-slate-400 mt-1">
              リース審査や業界動向について何でも聞いてください。
            </p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="w-8 h-8 overflow-hidden rounded-full border border-indigo-200 bg-indigo-50 flex-shrink-0 mt-0.5 shadow">
                  <img src={SHION_AVATAR} alt="紫苑" className="h-full w-full object-cover object-top" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-tr-sm"
                    : "bg-white text-slate-800 border border-slate-200 rounded-tl-sm"
                }`}
              >
                <div className="text-sm leading-relaxed break-words">
                  {msg.role === "assistant" ? (
                    renderAssistantContent(msg.content)
                  ) : (
                    <p className="whitespace-pre-wrap">{normalizeMessageContent(msg.content)}</p>
                  )}
                </div>
                <p
                  className={`text-[10px] mt-1.5 ${
                    msg.role === "user" ? "text-blue-200 text-right" : "text-slate-400"
                  }`}
                >
                  {formatTime(msg.created_at)}
                </p>
                {msg.role === "assistant" && (
                  <button
                    onClick={() => {
                      const previousUser = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
                      openKnowledgeEvidence(previousUser?.content || msg.content);
                    }}
                    className="mt-2 inline-flex items-center gap-1 rounded-md border border-cyan-200/40 bg-cyan-50 px-2 py-1 text-[11px] font-black text-cyan-700 transition hover:bg-cyan-100"
                  >
                    <Network className="h-3 w-3" />
                    根拠ルート
                  </button>
                )}
              </div>
              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5 shadow text-[10px] font-black text-white">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))
        )}

        {loading && (
          <div className="flex gap-3 justify-start">
            <div className="w-8 h-8 overflow-hidden rounded-full border border-indigo-200 bg-indigo-50 flex-shrink-0 mt-0.5 shadow">
              <img src={SHION_THINKING_AVATAR} alt="思考中の紫苑" className="h-full w-full object-cover object-top" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
              <div className="flex gap-1.5 items-center h-5">
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 入力エリア */}
      <div className={`flex-shrink-0 rounded-2xl shadow-lg border p-2 ${improvementMode ? "bg-amber-50 border-amber-200" : "bg-white border-slate-200"}`}>
        <div className="mb-2 flex items-center justify-between gap-2 px-1">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setImprovementMode((current) => !current)}
              className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-black transition-colors ${
                improvementMode
                  ? "bg-amber-500 text-white shadow-sm"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              <Lightbulb className="h-3.5 w-3.5" />
              改善メモ
            </button>
            {improvementMode && (
              <span className="text-[11px] font-bold text-amber-700">
                送信すると Improvement Log に保存
              </span>
            )}
          </div>
          <div className="flex shrink-0 rounded-lg border border-slate-200 bg-slate-100 p-0.5">
            {[
              {
                key: "shion",
                label: "紫苑",
                icon: Bot,
                activeClass: "bg-indigo-600 text-white shadow-sm",
                inactiveClass: "text-indigo-500 hover:bg-white",
              },
              {
                key: "general",
                label: "一般",
                icon: MessageCircle,
                activeClass: "bg-slate-800 text-white shadow-sm",
                inactiveClass: "text-slate-500 hover:bg-white",
              },
            ].map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setAnswerMode(item.key as "shion" | "general")}
                title={item.key === "shion" ? "紫苑モード: 記憶と関係性を優先" : "一般モード: 通常コメント"}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-black transition-colors ${
                  answerMode === item.key
                    ? item.activeClass
                    : item.inactiveClass
                }`}
              >
                <item.icon className="h-3.5 w-3.5" />
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={improvementMode ? "改善したい点を入力（例: この画面の導線が分かりにくい）" : "メッセージを入力（Enterで送信 / Shift+Enterで改行）"}
            rows={1}
            disabled={loading}
            className="flex-1 resize-none bg-transparent outline-none text-sm text-slate-800 placeholder:text-slate-400 px-2 py-2 max-h-40 overflow-y-auto leading-relaxed"
            style={{ minHeight: "2.5rem" }}
            onInput={resizeTextarea}
          />
          <button
            type="button"
            onClick={startVoiceInput}
            disabled={!voiceSupported || loading}
            title={voiceSupported ? (listening ? "録音中（クリックで停止）" : "音声入力") : "このブラウザは音声入力に未対応です"}
            className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
              listening
                ? "bg-rose-600 text-white animate-pulse"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 disabled:bg-slate-100 disabled:text-slate-300"
            }`}
          >
            <Mic className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setSpeechEnabled((v) => !v)}
            title={speechEnabled ? "音声読み上げON（クリックでOFF）" : "音声読み上げOFF（クリックでON）"}
            className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
              speechEnabled
                ? "bg-blue-500 text-white shadow-sm"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200"
            }`}
          >
            {speechEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className={`w-10 h-10 disabled:bg-slate-300 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
              improvementMode ? "bg-amber-500 hover:bg-amber-600" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            ) : (
              <Send className="w-4 h-4 text-white" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
