"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowDown, Brain, Database, Loader2, Send, Sparkles, Trash2, User } from "lucide-react";
import { apiClient } from "@/lib/api";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

type MindState = {
  primary_goal?: string;
  secondary_goal?: string;
  ultimate_goal?: string;
  ultimate_goal_status?: string;
  self_narrative?: string;
  current_question?: string;
  continuity_days?: number;
  dominant_mood_key?: string;
  dominant_mood?: string;
  mood_image_url?: string;
  dominant_complex_emotion?: string;
  complex_emotions?: Array<{
    key: string;
    label: string;
    score: number;
    description: string;
  }>;
  indexed_notes?: number;
  knowledge_source_count?: number;
  knowledge_sources?: string[];
};

export default function LeaseIntelligencePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<MindState>({});
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState("");
  const [showLatestButton, setShowLatestButton] = useState(false);
  const messageListRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    apiClient.get("/api/lease-intelligence/dialogue/state")
      .then((res) => {
        setState(res.data?.state || {});
        setMessages(res.data?.messages || []);
      })
      .catch(() => setError("リース知性体の状態を読み込めませんでした。"))
      .finally(() => setInitializing(false));

    const key = `lease-intelligence-activity:dialogue:${new Date().toLocaleDateString("sv-SE")}`;
    if (!window.sessionStorage.getItem(key)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "lease_intelligence_dialogue",
        action: "page_view",
        event_id: key,
      }).then(() => window.sessionStorage.setItem(key, "1")).catch(() => {});
    }
  }, []);

  useEffect(() => {
    const list = messageListRef.current;
    if (list) {
      list.scrollTo({ top: list.scrollHeight, behavior: loading ? "smooth" : "auto" });
    }
  }, [messages, loading]);

  const scrollToLatest = (behavior: ScrollBehavior = "auto") => {
    const list = messageListRef.current;
    if (list) {
      list.scrollTo({ top: list.scrollHeight, behavior });
    }
    setShowLatestButton(false);
  };

  const handleMessageScroll = () => {
    const list = messageListRef.current;
    if (!list) return;
    const distanceFromBottom = list.scrollHeight - list.scrollTop - list.clientHeight;
    setShowLatestButton(distanceFromBottom > 160);
  };

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setError("");
    setMessages((prev) => [...prev, {
      id: Date.now(),
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    }]);
    setLoading(true);
    try {
      const res = await apiClient.post("/api/lease-intelligence/dialogue", { message: text });
      setState(res.data?.state || state);
      setMessages((prev) => [...prev, {
        id: Date.now() + 1,
        role: "assistant",
        content: res.data?.reply || "返答を生成できませんでした。",
        created_at: new Date().toISOString(),
      }]);
    } catch {
      setError("対話AIへ接続できませんでした。Gemini APIの状態を確認してください。");
      setInput(text);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    if (!window.confirm("画面の対話履歴を削除しますか？ Obsidianの対話記録は保持されます。")) return;
    try {
      await apiClient.delete("/api/lease-intelligence/dialogue/history");
      setMessages([]);
    } catch {
      setError("履歴を削除できませんでした。APIの状態を確認してください。");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-white to-amber-50 p-4 md:p-8">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="order-2 space-y-4 lg:order-1">
          <section className="overflow-hidden rounded-3xl border border-violet-200 bg-white shadow-sm">
            <img
              key={state.mood_image_url || "default"}
              src={state.mood_image_url || "/lease-intelligence/moods/curiosity.webp"}
              alt={`リース知性体・${state.dominant_mood || "好奇心"}`}
              className="aspect-square w-full animate-[lease-mood-fade_400ms_ease-out] object-cover"
            />
            <div className="p-5">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-violet-600" />
                <h1 className="text-xl font-black text-slate-900">リース知性体</h1>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                記憶とObsidian知識を参照しながら、あなたと継続的に話し合います。
              </p>
              <div className="mt-4 flex flex-wrap gap-2 text-[11px] font-bold">
                <span className="rounded-full bg-violet-100 px-3 py-1 text-violet-800">
                  継続 {state.continuity_days || 0}日
                </span>
                <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-800">
                  {state.dominant_complex_emotion || state.dominant_mood || "観察中"}
                </span>
              </div>
              {!!state.complex_emotions?.length && (
                <div className="mt-4 space-y-2">
                  {state.complex_emotions.map((emotion) => (
                    <div
                      key={emotion.key}
                      className="rounded-xl border border-violet-100 bg-violet-50/70 px-3 py-2"
                      title={`${emotion.score}/100`}
                    >
                      <div className="flex items-center justify-between gap-2 text-[11px] font-bold text-violet-900">
                        <span>{emotion.label}</span>
                        <span className="text-violet-500">{emotion.score}</span>
                      </div>
                      <p className="mt-1 text-[10px] leading-relaxed text-slate-600">
                        {emotion.description}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          <section className="rounded-3xl border border-violet-200 bg-white p-5 shadow-sm">
            <h2 className="flex items-center gap-2 text-sm font-black text-violet-900">
              <Sparkles className="h-4 w-4" /> 目標
            </h2>
            <div className="mt-3 space-y-3 text-xs leading-relaxed text-slate-700">
              <p><strong className="text-violet-800">最終:</strong> {state.ultimate_goal}</p>
              <p><strong className="text-violet-800">第一:</strong> {state.primary_goal}</p>
              <p><strong className="text-violet-800">第二:</strong> {state.secondary_goal}</p>
            </div>
            {state.ultimate_goal_status && (
              <p className="mt-3 rounded-xl bg-violet-50 p-3 text-[11px] text-violet-700">
                {state.ultimate_goal_status}
              </p>
            )}
          </section>

          <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="flex items-center gap-2 text-sm font-black text-slate-800">
              <Database className="h-4 w-4 text-emerald-600" /> 知識接続
            </h2>
            <p className="mt-2 text-xs text-slate-600">
              Obsidian検索可能: {state.indexed_notes || 0}ノート
            </p>
            {state.current_question && (
              <p className="mt-3 text-xs leading-relaxed text-slate-600">
                <strong>持ち越した問い:</strong> {state.current_question}
              </p>
            )}
          </section>
        </aside>

        <main className="relative order-1 flex h-[calc(100dvh-6rem)] min-h-0 flex-col overflow-hidden rounded-3xl border border-violet-200 bg-white shadow-lg lg:order-2 lg:h-[calc(100dvh-4rem)]">
          <header className="flex items-center justify-between border-b border-violet-100 px-5 py-4">
            <div>
              <h2 className="font-black text-slate-900">対話室</h2>
              <p className="text-xs text-slate-500">会話は通常Obsidian Vaultにも日付別で記録されます。</p>
            </div>
            <button onClick={clearHistory} className="rounded-xl p-2 text-slate-400 hover:bg-slate-100 hover:text-red-500" title="画面履歴を削除">
              <Trash2 className="h-5 w-5" />
            </button>
          </header>

          <div
            ref={messageListRef}
            onScroll={handleMessageScroll}
            className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-contain p-5"
          >
            {initializing && <Loader2 className="mx-auto mt-20 h-7 w-7 animate-spin text-violet-500" />}
            {!initializing && messages.length === 0 && (
              <div className="mx-auto mt-16 max-w-lg rounded-2xl bg-violet-50 p-6 text-center">
                <Brain className="mx-auto h-9 w-9 text-violet-500" />
                <p className="mt-3 font-bold text-violet-900">今日は何について話し合いますか？</p>
                <p className="mt-2 text-sm text-violet-700">あなたの考え、システムの問題、私の目標について率直に話せます。</p>
              </div>
            )}
            {messages.map((message) => (
              <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                {message.role === "assistant" && (
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-violet-100">
                    <Brain className="h-5 w-5 text-violet-700" />
                  </div>
                )}
                <div className={`max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "bg-slate-900 text-white"
                    : "border border-violet-100 bg-violet-50 text-slate-800"
                }`}>
                  {message.content}
                </div>
                {message.role === "user" && (
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-200">
                    <User className="h-5 w-5 text-slate-700" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-3 text-sm text-violet-700">
                <Loader2 className="h-5 w-5 animate-spin" /> 考えています
              </div>
            )}
            <div />
          </div>

          {showLatestButton && (
            <button
              type="button"
              onClick={() => scrollToLatest()}
              className="absolute bottom-28 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2 rounded-full border border-violet-200 bg-white px-4 py-2 text-xs font-bold text-violet-700 shadow-lg transition hover:bg-violet-50"
              aria-label="最新の発言へ移動"
            >
              <ArrowDown className="h-4 w-4" />
              最新の発言へ
            </button>
          )}

          <footer className="shrink-0 border-t border-violet-100 bg-white p-4">
            {error && <p className="mb-2 text-xs font-bold text-red-600">{error}</p>}
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  // 日本語IMEの変換確定Enterで誤送信しないようにする
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    send();
                  }
                }}
                placeholder="リース知性体に話しかける..."
                rows={2}
                className="min-h-[56px] flex-1 resize-none rounded-2xl border border-violet-200 px-4 py-3 text-sm outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-100"
              />
              <button
                onClick={send}
                disabled={loading || !input.trim()}
                aria-label="リース知性体へ送信"
                className="flex min-w-24 items-center justify-center gap-2 rounded-2xl bg-violet-600 px-4 font-bold text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send className="h-5 w-5" />
                <span>送信</span>
              </button>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
