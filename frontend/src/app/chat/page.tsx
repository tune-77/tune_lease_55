"use client";

import React, { useState, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api";
import { Send, Trash2, Loader2, MessageCircle, Bot, User, NotebookPen, Mic, Network, Lightbulb } from "lucide-react";

interface ChatMessage {
  id: number;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

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
  const [improvementMode, setImprovementMode] = useState(false);
  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  const userId = "default";

  const scrollToBottom = () => {
    const el = messageListRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  useEffect(() => {
    loadHistory();
    const timer = setTimeout(() => setShowSubtitle(false), 5000);
    setVoiceSupported(Boolean(getSpeechRecognition()));
    return () => clearTimeout(timer);
  }, []);

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
      });
      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        user_id: userId,
        role: "assistant",
        content: res.data.reply,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
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
    <div className="flex flex-col h-[calc(100vh-4rem)] max-w-3xl mx-auto px-4 py-6 overflow-hidden">
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
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0 mt-0.5 shadow">
                  <Bot className="w-4 h-4 text-white" />
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
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0 mt-0.5 shadow">
              <Bot className="w-4 h-4 text-white" />
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
            title={voiceSupported ? "音声入力" : "このブラウザは音声入力に未対応です"}
            className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
              listening
                ? "bg-rose-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 disabled:bg-slate-100 disabled:text-slate-300"
            }`}
          >
            <Mic className="w-4 h-4" />
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
