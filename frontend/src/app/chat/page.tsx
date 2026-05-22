"use client";

import React, { useState, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api";
import { Send, Trash2, Loader2, MessageCircle, Bot, User, NotebookPen } from "lucide-react";

interface ChatMessage {
  id: number;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [savingObsidian, setSavingObsidian] = useState(false);
  const [saveToast, setSaveToast] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const userId = "default";

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    loadHistory();
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
      });
      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        user_id: userId,
        role: "assistant",
        content: res.data.reply,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
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
    <div className="flex flex-col h-[calc(100vh-4rem)] max-w-3xl mx-auto px-4 py-6">
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
            <MessageCircle className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black text-slate-800">AIアドバイザー</h1>
            <p className="text-xs text-slate-500">リース審査の相棒・毎日相談できます</p>
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
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 pb-4">
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
          messages.map((msg) => (
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
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                <p
                  className={`text-[10px] mt-1.5 ${
                    msg.role === "user" ? "text-blue-200 text-right" : "text-slate-400"
                  }`}
                >
                  {formatTime(msg.created_at)}
                </p>
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
        <div ref={bottomRef} />
      </div>

      {/* 入力エリア */}
      <div className="flex-shrink-0 bg-white border border-slate-200 rounded-2xl shadow-lg p-2 flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="メッセージを入力（Enterで送信 / Shift+Enterで改行）"
          rows={1}
          disabled={loading}
          className="flex-1 resize-none bg-transparent outline-none text-sm text-slate-800 placeholder:text-slate-400 px-2 py-2 max-h-40 overflow-y-auto leading-relaxed"
          style={{ minHeight: "2.5rem" }}
          onInput={(e) => {
            const el = e.currentTarget;
            el.style.height = "auto";
            el.style.height = `${el.scrollHeight}px`;
          }}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="w-10 h-10 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 rounded-xl flex items-center justify-center transition-colors flex-shrink-0"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 text-white animate-spin" />
          ) : (
            <Send className="w-4 h-4 text-white" />
          )}
        </button>
      </div>
    </div>
  );
}
