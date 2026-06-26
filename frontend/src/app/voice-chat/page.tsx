"use client";

import React, { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  BookOpenCheck,
  Brain,
  CheckCircle2,
  Keyboard,
  Loader2,
  Mic,
  MicOff,
  Send,
  Square,
  Trash2,
  User,
  Volume2,
  VolumeX,
  Waves,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type SpeechRecognitionInstance = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: (() => void) | null;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type KnowledgeRef = {
  doc_id: string;
  obsidian_ref: string;
  file_name: string;
  rank_score?: number;
};

type VoiceMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
  knowledge_refs?: KnowledgeRef[];
  query?: string;
};

type VoiceState = "idle" | "listening" | "processing" | "speaking";

// 無音が続いた場合に「話し終わり」と判定するまでの待機時間（ms）
// ブラウザ内蔵VADのデフォルト（~500ms）より大幅に延長して途中切断を防ぐ
const VAD_SILENCE_DELAY_MS = 2500;

const stateText: Record<VoiceState, string> = {
  idle: "声を待っています",
  listening: "聞いています",
  processing: "紫苑が考えています",
  speaking: "紫苑が話しています",
};

const starterPrompts = [
  "この工作機械案件、補助金採択前だけどどう見る？",
  "新規先で不安な時、何を確認すべき？",
  "この案件を条件付き承認にするなら、条件は何？",
];

function cleanForSpeech(text: string) {
  return text
    .replace(/[\*#_`~+\[\]{}]/g, "")
    .replace(/https?:\/\/\S+/g, "リンク")
    .replace(/\s+/g, " ")
    .trim();
}

function formatTime(date: Date) {
  return date.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
}

function Waveform({ state }: { state: VoiceState }) {
  const active = state === "listening" || state === "speaking" || state === "processing";
  return (
    <div className="flex h-24 items-center justify-center gap-1.5">
      {Array.from({ length: 18 }).map((_, index) => (
        <span
          key={index}
          className={`w-1.5 rounded-full ${
            state === "listening"
              ? "bg-teal-500"
              : state === "speaking"
                ? "bg-rose-500"
                : state === "processing"
                  ? "bg-violet-500"
                  : "bg-stone-300"
          }`}
          style={{
            height: active ? `${18 + ((index * 17) % 42)}px` : `${10 + (index % 3) * 6}px`,
            animation: active ? `voicePulse ${760 + index * 28}ms ease-in-out infinite alternate` : "none",
            opacity: active ? 0.95 : 0.55,
          }}
        />
      ))}
      <style jsx>{`
        @keyframes voicePulse {
          from {
            transform: scaleY(0.45);
          }
          to {
            transform: scaleY(1);
          }
        }
      `}</style>
    </div>
  );
}

export default function VoiceChatPage() {
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [input, setInput] = useState("");
  const [state, setState] = useState<VoiceState>("idle");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const [continuous, setContinuous] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [supported, setSupported] = useState({ recognition: true, synthesis: true });

  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const internalStopRef = useRef(false);
  const stateRef = useRef<VoiceState>("idle");
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingFinalRef = useRef<string>("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, transcript]);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const SpeechRecognitionApi =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const hasRecognition = Boolean(SpeechRecognitionApi);
    const hasSynthesis = typeof window !== "undefined" && Boolean(window.speechSynthesis);
    setSupported({ recognition: hasRecognition, synthesis: hasSynthesis });
    if (!hasRecognition) {
      setError("このブラウザは音声認識に対応していません。ChromeまたはSafariで試してください。");
      return;
    }

    const rec = new SpeechRecognitionApi() as SpeechRecognitionInstance;
    rec.lang = "ja-JP";
    rec.continuous = true;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      setState("listening");
      setError("");
      setTranscript("");
      internalStopRef.current = false;
      pendingFinalRef.current = "";
    };

    rec.onresult = (event: any) => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const text = event.results[i][0]?.transcript || "";
        if (event.results[i].isFinal) pendingFinalRef.current += text;
        else interim += text;
      }
      setTranscript(pendingFinalRef.current || interim);
      if (pendingFinalRef.current.trim()) {
        silenceTimerRef.current = setTimeout(() => {
          const toSubmit = pendingFinalRef.current.trim();
          pendingFinalRef.current = "";
          if (toSubmit) {
            internalStopRef.current = true;
            rec.stop();
            void submitText(toSubmit, { fromVoice: true });
          }
        }, VAD_SILENCE_DELAY_MS);
      }
    };

    rec.onerror = (event: any) => {
      if (event.error === "no-speech") {
        setState("idle");
        setTranscript("");
        return;
      }
      if (event.error === "not-allowed") {
        setError("マイクの使用が許可されていません。ブラウザの権限を確認してください。");
      } else {
        setError(`音声認識エラー: ${event.error}`);
      }
      setState("idle");
    };

    rec.onend = () => {
      if (stateRef.current === "listening") setState("idle");
      setTranscript("");
    };

    recognitionRef.current = rec;
    return () => {
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      internalStopRef.current = true;
      rec.abort();
      window.speechSynthesis?.cancel();
    };
    // recognition callbacks intentionally read latest behavior through refs/state setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastAssistantRefs = useMemo(() => {
    const assistant = [...messages].reverse().find((msg) => msg.role === "assistant");
    return assistant?.knowledge_refs || [];
  }, [messages]);

  const speak = (text: string) => {
    if (!supported.synthesis || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const spoken = cleanForSpeech(text);
    if (!spoken) return;
    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = "ja-JP";
    utterance.rate = 0.98;
    utterance.pitch = 1.02;
    utterance.volume = 1;
    utterance.onstart = () => setState("speaking");
    utterance.onend = () => {
      setState("idle");
      if (continuous && recognitionRef.current && !internalStopRef.current) {
        window.setTimeout(() => startListening(), 350);
      }
    };
    utterance.onerror = () => setState("idle");
    window.speechSynthesis.speak(utterance);
  };

  const startListening = () => {
    if (!recognitionRef.current) {
      setError("音声認識を初期化できませんでした。");
      return;
    }
    window.speechSynthesis?.cancel();
    internalStopRef.current = false;
    try {
      recognitionRef.current.start();
    } catch {
      // Already started. Keep current state.
    }
  };

  const stopAll = () => {
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    pendingFinalRef.current = "";
    internalStopRef.current = true;
    recognitionRef.current?.abort();
    window.speechSynthesis?.cancel();
    setState("idle");
    setTranscript("");
  };

  async function submitText(text: string, options?: { fromVoice?: boolean }) {
    const trimmed = text.trim();
    if (!trimmed || stateRef.current === "processing") return;
    setError("");
    setInput("");
    setTranscript("");
    internalStopRef.current = true;

    const userMsg: VoiceMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      text: trimmed,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setState("processing");

    try {
      const res = await apiClient.post("/api/lease-intelligence/dialogue", {
        message: trimmed,
        caller: options?.fromVoice ? "voice-chat" : "voice-chat-keyboard",
      });
      const reply = String(res.data?.reply || "返答を生成できませんでした。");
      const refs = (res.data?.knowledge_refs || []) as KnowledgeRef[];
      const assistantMsg: VoiceMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: reply,
        timestamp: new Date(),
        knowledge_refs: refs.length ? refs : undefined,
        query: trimmed,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      internalStopRef.current = false;
      if (autoSpeak) speak(reply);
      else setState("idle");
    } catch {
      setError("紫苑APIへ接続できませんでした。APIサーバーまたはGemini設定を確認してください。");
      setState("idle");
    }
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitText(input);
  };

  const clear = () => {
    stopAll();
    setMessages([]);
    setError("");
  };

  return (
    <main className="min-h-screen bg-[#f7f4ee] text-stone-950">
      <section className="border-b border-stone-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 sm:px-6 lg:flex-row lg:items-end lg:justify-between lg:px-8">
          <div>
            <p className="text-xs font-black uppercase tracking-[0.3em] text-teal-700">
              Real-time Voice Dialogue
            </p>
            <h1 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">
              紫苑と声で話す
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-7 text-stone-600">
              マイク入力をリアルタイムに文字化し、紫苑の回答を読み上げます。過去メモや判断資産を参照した場合は、右側に根拠として表示します。
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link
              href="/demo-home"
              className="inline-flex items-center gap-2 rounded-md border border-stone-300 bg-white px-4 py-2 text-sm font-black text-stone-700 hover:bg-stone-100"
            >
              デモホーム
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/lease-intelligence"
              className="inline-flex items-center gap-2 rounded-md bg-stone-950 px-4 py-2 text-sm font-black text-white hover:bg-stone-800"
            >
              通常チャット
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[0.9fr_1.1fr_0.85fr] lg:px-8">
        <aside className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] font-black uppercase tracking-widest text-stone-500">Voice Console</p>
              <h2 className="mt-1 text-xl font-black">会話の状態</h2>
            </div>
            <span className="inline-flex h-11 w-11 items-center justify-center rounded-full bg-teal-50 text-teal-700">
              <Waves className="h-5 w-5" />
            </span>
          </div>

          <div className="mt-5 rounded-2xl border border-stone-200 bg-[#fffdf8] p-4">
            <Waveform state={state} />
            <div className="mt-2 text-center">
              <span
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-black ${
                  state === "listening"
                    ? "bg-teal-100 text-teal-800"
                    : state === "processing"
                      ? "bg-violet-100 text-violet-800"
                      : state === "speaking"
                        ? "bg-rose-100 text-rose-800"
                        : "bg-stone-100 text-stone-700"
                }`}
              >
                {state === "processing" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {stateText[state]}
              </span>
            </div>
          </div>

          <div className="mt-5 flex justify-center">
            <button
              type="button"
              onClick={state === "listening" ? stopAll : startListening}
              disabled={!supported.recognition || state === "processing"}
              className={`relative flex h-32 w-32 flex-col items-center justify-center rounded-full border-4 text-sm font-black shadow-xl transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-60 ${
                state === "listening"
                  ? "border-teal-300 bg-teal-600 text-white shadow-teal-900/20"
                  : state === "speaking"
                    ? "border-rose-300 bg-rose-600 text-white shadow-rose-900/20"
                    : "border-stone-200 bg-white text-stone-700 hover:border-teal-300"
              }`}
            >
              {state === "listening" ? <MicOff className="h-9 w-9" /> : <Mic className="h-9 w-9" />}
              <span className="mt-2">{state === "listening" ? "止める" : "話す"}</span>
            </button>
          </div>

          <div className="mt-5 space-y-3 rounded-xl border border-stone-200 bg-stone-50 p-4">
            <label className="flex items-center justify-between gap-3 text-sm font-bold text-stone-700">
              連続会話
              <input
                type="checkbox"
                checked={continuous}
                onChange={(event) => setContinuous(event.target.checked)}
                className="h-5 w-5 accent-teal-700"
              />
            </label>
            <label className="flex items-center justify-between gap-3 text-sm font-bold text-stone-700">
              自動読み上げ
              <input
                type="checkbox"
                checked={autoSpeak}
                onChange={(event) => setAutoSpeak(event.target.checked)}
                className="h-5 w-5 accent-teal-700"
              />
            </label>
            <button
              type="button"
              onClick={() => setShowKeyboard((value) => !value)}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-white px-3 py-2 text-sm font-black text-stone-700 shadow-sm hover:bg-stone-100"
            >
              <Keyboard className="h-4 w-4" />
              {showKeyboard ? "キーボードを閉じる" : "キーボード入力"}
            </button>
            <button
              type="button"
              onClick={stopAll}
              className="flex w-full items-center justify-center gap-2 rounded-md border border-stone-300 px-3 py-2 text-sm font-black text-stone-700 hover:bg-white"
            >
              <Square className="h-4 w-4" />
              停止
            </button>
          </div>
        </aside>

        <section className="flex min-h-[680px] flex-col rounded-2xl border border-stone-200 bg-white shadow-sm">
          <div className="flex items-center justify-between border-b border-stone-200 px-5 py-4">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-teal-700" />
              <h2 className="text-lg font-black">会話</h2>
              <span className="rounded-full bg-stone-100 px-2 py-1 text-[11px] font-black text-stone-500">
                {messages.length} messages
              </span>
            </div>
            <button
              type="button"
              onClick={clear}
              className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-bold text-stone-500 hover:bg-stone-100"
            >
              <Trash2 className="h-4 w-4" />
              clear
            </button>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto bg-[#fbf8f1] p-5">
            {messages.length === 0 && (
              <div className="flex h-full min-h-[360px] flex-col items-center justify-center text-center">
                <div className="inline-flex h-16 w-16 items-center justify-center rounded-full bg-white text-teal-700 shadow-sm">
                  <Mic className="h-7 w-7" />
                </div>
                <p className="mt-4 text-lg font-black text-stone-900">声で案件相談を始める</p>
                <p className="mt-2 max-w-sm text-sm leading-7 text-stone-500">
                  下の例文を押すか、マイクで話しかけてください。紫苑が短く答え、必要なら判断資産を想起します。
                </p>
                <div className="mt-5 flex flex-col gap-2">
                  {starterPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => void submitText(prompt)}
                      className="rounded-md border border-stone-200 bg-white px-4 py-2 text-left text-sm font-bold text-stone-700 hover:border-teal-300 hover:bg-teal-50"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "assistant" && (
                  <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-600 to-emerald-700 text-white">
                    <Brain className="h-4 w-4" />
                  </div>
                )}
                <div
                  className={`max-w-[82%] rounded-2xl px-4 py-3 shadow-sm ${
                    message.role === "user"
                      ? "rounded-tr-sm bg-stone-950 text-white"
                      : "rounded-tl-sm border border-stone-200 bg-white text-stone-800"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm leading-7">{message.text}</p>
                  <div className={`mt-2 flex items-center gap-2 text-[10px] ${
                    message.role === "user" ? "text-white/55" : "text-stone-400"
                  }`}>
                    <span>{formatTime(message.timestamp)}</span>
                    {message.role === "assistant" && (
                      <button
                        type="button"
                        onClick={() => speak(message.text)}
                        className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-stone-100"
                      >
                        <Volume2 className="h-3 w-3" />
                        replay
                      </button>
                    )}
                  </div>
                </div>
                {message.role === "user" && (
                  <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-stone-200 text-stone-700">
                    <User className="h-4 w-4" />
                  </div>
                )}
              </div>
            ))}

            {state === "listening" && transcript && (
              <div className="flex justify-end gap-3">
                <div className="max-w-[82%] rounded-2xl rounded-tr-sm border border-dashed border-teal-300 bg-teal-50 px-4 py-3 text-sm leading-7 text-teal-900">
                  {transcript}
                  <span className="ml-1 inline-block h-4 w-1 animate-pulse bg-teal-500 align-middle" />
                </div>
                <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-teal-100 text-teal-700">
                  <User className="h-4 w-4" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {showKeyboard && (
            <form onSubmit={onSubmit} className="flex gap-2 border-t border-stone-200 bg-white p-4">
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="紫苑に聞くことを入力"
                className="min-w-0 flex-1 rounded-md border border-stone-300 bg-stone-50 px-4 py-3 text-sm outline-none focus:border-teal-500 focus:bg-white"
              />
              <button
                type="submit"
                disabled={!input.trim() || state === "processing"}
                className="inline-flex items-center gap-2 rounded-md bg-teal-700 px-4 py-3 text-sm font-black text-white disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
                送信
              </button>
            </form>
          )}

          {error && (
            <div className="border-t border-rose-200 bg-rose-50 px-5 py-3 text-sm font-bold text-rose-800">
              {error}
            </div>
          )}
        </section>

        <aside className="space-y-5">
          <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <BookOpenCheck className="h-5 w-5 text-emerald-700" />
              <h2 className="text-lg font-black">想起した判断資産</h2>
            </div>
            <div className="mt-4 space-y-2">
              {lastAssistantRefs.length > 0 ? (
                lastAssistantRefs.slice(0, 5).map((ref) => (
                  <div key={`${ref.doc_id}-${ref.obsidian_ref}`} className="rounded-lg border border-emerald-100 bg-emerald-50 px-3 py-2">
                    <p className="truncate text-sm font-black text-emerald-900">{ref.file_name || ref.doc_id}</p>
                    <p className="mt-1 truncate text-xs text-emerald-700">{ref.obsidian_ref}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm leading-7 text-stone-600">
                  紫苑がRAG/Obsidian由来の知識を参照した場合、ここに出典が出ます。
                </p>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-teal-700" />
              <h2 className="text-lg font-black">デモで伝わること</h2>
            </div>
            <div className="mt-4 space-y-3 text-sm leading-7 text-stone-700">
              <p>声で案件相談できるので、審査員がすぐ体験できます。</p>
              <p>回答を読み上げるため、チャットより「隣で相談している」感覚が出ます。</p>
              <p>右側に参照知識を出すことで、ただの音声AIではなく判断資産AIとして見せられます。</p>
            </div>
          </div>

          <div className="rounded-2xl border border-stone-200 bg-[#171512] p-5 text-white shadow-sm">
            <div className="flex items-center gap-2">
              {autoSpeak ? <Volume2 className="h-5 w-5 text-amber-300" /> : <VolumeX className="h-5 w-5 text-stone-400" />}
              <h2 className="text-lg font-black">次の段階</h2>
            </div>
            <p className="mt-4 text-sm leading-7 text-stone-200">
              次はSSEで「知識検索中」「参照メモ」「回答生成」を分けて流すと、さらにリアルタイム感が強くなります。
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
