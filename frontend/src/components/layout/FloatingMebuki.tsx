"use client";
import React, { useState, useEffect, useRef } from 'react';
import { apiClient } from "@/lib/api";
import { Send, X, Loader2, NotebookPen, Lightbulb } from "lucide-react";
import { usePathname } from "next/navigation";

const YANAMI_BOT_MESSAGES = [
  "システム稼働中。いつでもサポートします！",
  "[観察メモ] 今日の案件では、先に返済原資の説明を固めると判断が速くなりそうです。",
  "[観察メモ] 稟議では、否決理由より先に条件付きで通せる形を探します。",
  "[観察メモ] 運送業の案件は、燃料費と人件費の変化を一緒に見ると筋が見えます。",
  "[観察メモ] DSCRが薄い案件は、月次資金繰りと追加担保の確認を先に置きます。",
  "[観察メモ] キャッシュフロー計算書は、利益より先に返済余力を見る入口になります。",
  "[観察メモ] 減価償却費の足し戻しは、設備更新の実態とセットで読む必要があります。",
  "[観察メモ] 競合金利を見る時は、価格だけでなく条件差も一緒に残します。"
];

interface ChatMessage {
  id: number;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

const MEBUKI_USER_ID = "mebuki-default";

const cleanMebukiText = (text: string) => (
  String(text || "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/`([^`]+)`/g, "$1")
    .trim()
);

export default function FloatingMebuki() {
  const pathname = usePathname();
  const suppressPassiveBubble = pathname === "/demo/knowledge-loop";
  const [mebukiState, setMebukiState] = useState<'guide' | 'approve' | 'challenge' | 'reject'>('guide');
  const [bubbleMessage, setBubbleMessage] = useState("システム稼働中。いつでもサポートします！");
  const [isBubbleVisible, setIsBubbleVisible] = useState(!suppressPassiveBubble);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const eventOverrideRef = useRef<boolean>(false);
  const worldViewAckedRef = useRef<boolean>(false);

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [savingObsidian, setSavingObsidian] = useState(false);
  const [improvementMode, setImprovementMode] = useState(false);
  const [saveToast, setSaveToast] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (suppressPassiveBubble && !eventOverrideRef.current) {
      setIsBubbleVisible(false);
    }
  }, [suppressPassiveBubble]);

  // 吹き出しが表示されたら5秒後に自動非表示
  useEffect(() => {
    if (!isBubbleVisible) return;
    const messageLength = cleanMebukiText(bubbleMessage).length;
    const visibleMs = Math.min(22000, Math.max(8000, messageLength * 120));
    const timer = setTimeout(() => setIsBubbleVisible(false), visibleMs);
    return () => clearTimeout(timer);
  }, [isBubbleVisible, bubbleMessage]);

  // カスタムイベントでめぶきの状態を制御する
  useEffect(() => {
    const handleMebukiEvent = (e: any) => {
      const { type, text } = e.detail;
      setMebukiState(type);
      setBubbleMessage(cleanMebukiText(text));
      setIsBubbleVisible(true);
      eventOverrideRef.current = true;

      setTimeout(() => {
        eventOverrideRef.current = false;
      }, 15000);
    };

    window.addEventListener('mebuki-action', handleMebukiEvent);

    // ランダムな観察メモ（約5分に1回）
    const boyakiInterval = setInterval(() => {
      if (!suppressPassiveBubble && !eventOverrideRef.current && !isChatOpen) {
        const randomIndex = Math.floor(Math.random() * YANAMI_BOT_MESSAGES.length);
        setBubbleMessage(cleanMebukiText(YANAMI_BOT_MESSAGES[randomIndex]));
        setMebukiState('guide');
        setIsBubbleVisible(true);
      }
    }, 300000);

    return () => {
      window.removeEventListener('mebuki-action', handleMebukiEvent);
      clearInterval(boyakiInterval);
    };
  }, [isChatOpen, suppressPassiveBubble]);

  // world_view 更新チェック（マウント時と10分ごと）
  useEffect(() => {
    const checkWorldView = async () => {
      if (suppressPassiveBubble) return;
      if (worldViewAckedRef.current) return;
      try {
        const res = await apiClient.get("/api/world-view-status");
        if (res.data.has_update && !worldViewAckedRef.current) {
          const summary: string = res.data.summary || "";
          const preview = summary.slice(0, 30) + (summary.length > 30 ? "…" : "");
          setBubbleMessage(`🌏 世界認識が更新されました：${preview}`);
          setMebukiState('guide');
          setIsBubbleVisible(true);
          worldViewAckedRef.current = true;
          apiClient.post("/api/world-view-ack").catch(() => {});
        }
      } catch {
        // 失敗は無視
      }
    };

    checkWorldView();
    const interval = setInterval(checkWorldView, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [suppressPassiveBubble]);

  // チャットパネル開閉
  const handleMebukiClick = () => {
    if (isChatOpen) {
      setIsChatOpen(false);
    } else {
      setIsChatOpen(true);
      loadHistory();
    }
  };

  const closeChat = () => setIsChatOpen(false);

  // 履歴読み込み
  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await apiClient.get("/api/chat/history", {
        params: { user_id: MEBUKI_USER_ID, limit: 50 },
      });
      setMessages(res.data.messages || []);
    } catch {
      // 失敗は無視して空スタート
    } finally {
      setHistoryLoading(false);
    }
  };

  // メッセージ送信
  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const optimisticUser: ChatMessage = {
      id: Date.now(),
      user_id: MEBUKI_USER_ID,
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
        user_id: MEBUKI_USER_ID,
        intent: improvementMode ? "improvement" : undefined,
        response_mode: "general",
        caller: "mebuki",
      });
      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        user_id: MEBUKI_USER_ID,
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
          user_id: MEBUKI_USER_ID,
          role: "assistant",
          content: "エラーが発生しました。もう一度お試しください。",
          created_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const saveToObsidian = async () => {
    if (savingObsidian) return;
    setSavingObsidian(true);
    try {
      await apiClient.post("/api/chat/save-to-obsidian", { user_id: MEBUKI_USER_ID });
      setSaveToast("Obsidianに保存しました ✅");
      setTimeout(() => setSaveToast(null), 2000);
    } catch {
      setSaveToast("保存に失敗しました ❌");
      setTimeout(() => setSaveToast(null), 2000);
    } finally {
      setSavingObsidian(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
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

  // スクロール
  useEffect(() => {
    if (isChatOpen) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isChatOpen]);

  if (
    pathname === "/chat" ||
    pathname === "/chat-compare" ||
    pathname === "/lease-intelligence" ||
    pathname === "/multi-shion-demo" ||
    pathname === "/voice-chat"
  ) {
    return null;
  }

  return (
    <div className="fixed bottom-[calc(env(safe-area-inset-bottom)+0.5rem)] right-4 sm:right-6 z-50 flex flex-col items-end justify-end pointer-events-none">

      {/* ── チャットパネル ── */}
      {isChatOpen && (
        <div className="pointer-events-auto mb-3 h-[min(78dvh,38rem)] w-[min(24rem,calc(100vw-1rem))] bg-white rounded-2xl shadow-2xl border border-slate-200 flex flex-col overflow-hidden">
          {/* ヘッダー */}
          <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 flex-shrink-0">
            <div className="flex items-center gap-2">
              <img
                src={`/mebuki/${mebukiState}.png`}
                alt="めぶきちゃん"
                className="w-7 h-7 rounded-full border-2 border-white object-cover bg-emerald-100"
              />
              <span className="text-white font-black text-sm">💬 めぶきちゃん</span>
            </div>
            <div className="flex items-center gap-2">
              {/* Obsidian保存トースト */}
              {saveToast && (
                <span className="text-white text-[10px] font-bold bg-white/20 rounded-lg px-2 py-0.5 whitespace-nowrap">
                  {saveToast}
                </span>
              )}
              <button
                onClick={saveToObsidian}
                disabled={savingObsidian}
                className="text-white/80 hover:text-white transition-colors disabled:opacity-50"
                aria-label="Obsidianに保存"
                title="Obsidianに保存"
              >
                {savingObsidian ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <NotebookPen className="w-4 h-4" />
                )}
              </button>
              <button
                onClick={closeChat}
                className="text-white/80 hover:text-white transition-colors"
                aria-label="閉じる"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* メッセージエリア */}
          <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain p-3 space-y-3 bg-slate-50">
            {historyLoading ? (
              <div className="flex justify-center items-center h-full">
                <Loader2 className="w-5 h-5 text-emerald-500 animate-spin" />
              </div>
            ) : messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center px-4">
                <img
                  src={`/mebuki/guide.png`}
                  alt="めぶきちゃん"
                  className="w-12 h-12 rounded-full border-2 border-emerald-300 object-cover bg-emerald-100 mb-2"
                />
                <p className="text-slate-600 font-bold text-xs">こんにちは！めぶきちゃんです🌿</p>
                <p className="text-slate-400 text-xs mt-1">リース審査のことなら何でも相談してください！</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <img
                      src={`/mebuki/${mebukiState}.png`}
                      alt="めぶきちゃん"
                      className="w-6 h-6 rounded-full border border-emerald-300 object-cover bg-emerald-100 flex-shrink-0 mt-0.5"
                    />
                  )}
                  <div
                    className={`max-w-[88%] rounded-2xl px-3 py-2 shadow-sm text-xs leading-relaxed ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white rounded-tr-sm"
                        : "bg-white text-slate-800 border border-slate-200 rounded-tl-sm"
                    }`}
                  >
                    <p className="whitespace-pre-wrap break-words">{cleanMebukiText(msg.content)}</p>
                    <p className={`text-[9px] mt-1 ${msg.role === "user" ? "text-blue-200 text-right" : "text-slate-400"}`}>
                      {formatTime(msg.created_at)}
                    </p>
                  </div>
                </div>
              ))
            )}

            {/* ローディングドット */}
            {loading && (
              <div className="flex gap-2 justify-start">
                <img
                  src={`/mebuki/${mebukiState}.png`}
                  alt="めぶきちゃん"
                  className="w-6 h-6 rounded-full border border-emerald-300 object-cover bg-emerald-100 flex-shrink-0 mt-0.5"
                />
                <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-3 py-2 shadow-sm">
                  <div className="flex gap-1 items-center h-4">
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* 入力エリア */}
          <div className={`flex-shrink-0 border-t p-2 ${improvementMode ? "bg-amber-50 border-amber-200" : "bg-white border-slate-200"}`}>
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => setImprovementMode((current) => !current)}
                className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-black transition-colors ${
                  improvementMode
                    ? "bg-amber-500 text-white shadow-sm"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                <Lightbulb className="h-3 w-3" />
                改善メモ
              </button>
              {improvementMode && (
                <span className="text-[9px] font-bold text-amber-700">Logに保存</span>
              )}
            </div>
            <div className="flex gap-2 items-end">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={improvementMode ? "改善したい点を入力" : "メッセージを入力（Enter送信）"}
                rows={1}
                disabled={loading}
                className="flex-1 resize-none bg-transparent outline-none text-xs text-slate-800 placeholder:text-slate-400 px-2 py-1.5 max-h-24 overflow-y-auto leading-relaxed"
                style={{ minHeight: "2rem" }}
                onInput={(e) => {
                  const el = e.currentTarget;
                  el.style.height = "auto";
                  el.style.height = `${el.scrollHeight}px`;
                }}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className={`w-8 h-8 disabled:bg-slate-300 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
                  improvementMode ? "bg-amber-500 hover:bg-amber-600" : "bg-emerald-500 hover:bg-emerald-600"
                }`}
              >
                {loading ? (
                  <Loader2 className="w-3.5 h-3.5 text-white animate-spin" />
                ) : (
                  <Send className="w-3.5 h-3.5 text-white" />
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── 従来の吹き出し（チャットが閉じているときのみ） ── */}
      <div className="flex items-end justify-end pointer-events-none">
        {!suppressPassiveBubble && !isChatOpen && (
          <div
            className={`max-h-[42dvh] w-[min(18rem,calc(100vw-6rem))] overflow-y-auto bg-white text-slate-800 p-3 sm:p-4 rounded-2xl rounded-br-none shadow-2xl border-2 border-amber-200 text-xs sm:text-sm font-bold leading-relaxed mb-4 mr-1 sm:mb-6 sm:mr-2 pointer-events-auto transition-all duration-300 transform origin-bottom-right whitespace-pre-wrap break-words ${isBubbleVisible ? 'scale-100 opacity-100' : 'scale-75 opacity-0'}`}
          >
            {cleanMebukiText(bubbleMessage)}
          </div>
        )}

        {/* めぶきちゃん画像 */}
        <div
          className={`relative pointer-events-auto cursor-pointer hover:scale-105 transition-transform drop-shadow-2xl ${
            suppressPassiveBubble ? "w-16 h-16 sm:w-20 sm:h-20" : "w-20 h-20 sm:w-32 sm:h-32"
          }`}
          onClick={handleMebukiClick}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`/mebuki/${mebukiState}.png`}
            alt="めぶきちゃん"
            className="w-full h-full object-cover rounded-full border-4 border-white shadow-lg bg-emerald-100"
          />

          {/* オンラインバッジ */}
          <div className="absolute bottom-2 right-2 w-5 h-5 bg-green-500 rounded-full border-2 border-white shadow-sm skeleton-pulse flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-full opacity-60"></div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ユーティリティ関数（どこからでも呼べる）
export const triggerMebuki = (type: 'guide' | 'approve' | 'challenge' | 'reject', text: string) => {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('mebuki-action', { detail: { type, text: cleanMebukiText(text) } }));
  }
};
