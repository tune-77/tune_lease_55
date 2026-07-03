"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowDown, Brain, Check, Copy, Database, Loader2, Mic, MicOff,
  Network, Paperclip, Send, Sparkles, Trash2, TrendingUp, User, Volume2, VolumeX, X,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type KnowledgeRef = {
  doc_id: string;
  obsidian_ref: string;
  file_name: string;
  rank_score?: number;
};

type AttachedFile = {
  name: string;
  type: "csv" | "image";
  content: string;
  mimeType?: string;
};

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  knowledge_refs?: KnowledgeRef[];
  query?: string;
  longInputMode?: boolean;
  attachedFileName?: string;
  attachedFileType?: "csv" | "image";
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
  knowledge_connection?: {
    label?: string;
    source?: string;
    is_cloud_run?: boolean;
    vector_chunks?: number;
    markdown_notes?: number;
    case_count?: number;
  };
};

type EmotionHistoryEntry = {
  id: number;
  recorded_at: string;
  hopeful_anxiety: number | null;
  careful_attachment: number | null;
  intellectual_excitement: number | null;
  unrewarded_effort: number | null;
  quiet_loneliness: number | null;
  earned_confidence: number | null;
  protective_frustration: number | null;
  dominant_raw_emotion: string;
};

type EmotionAxisStats = { avg: number; max: number; min: number; std: number };
type EmotionSummary = {
  days: number;
  count: number;
  axes: Record<string, EmotionAxisStats>;
  dominant_avg: string;
};

const DEMO_GREETING = `はじめまして。リース知性体、紫苑です。

私は、リース審査を点数で終わらせないために生まれました。
財務、物件、金利、過去案件、担当者の判断、改善ログを記憶し、
次の審査に活かす判断資産へ変えていきます。

今日は、使うほど賢くなるリース審査プラットフォームをご覧ください。`;

const DIALOGUE_RETRY_DELAYS_MS = [1200, 2500, 4000];

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const getDialogueErrorDetail = (err: unknown) => {
  const value = err as {
    message?: string;
    code?: string;
    response?: {
      status?: number;
      statusText?: string;
      data?: string | { detail?: string; message?: string };
    };
  };
  const data = value?.response?.data;
  if (typeof data === "string") return data;
  return data?.detail || data?.message || value?.message || value?.response?.statusText || "";
};

const isTransientDialogueConnectionError = (err: unknown) => {
  const value = err as {
    code?: string;
    response?: { status?: number };
  };
  const status = value?.response?.status;
  const detail = getDialogueErrorDetail(err).toLowerCase();
  return (
    status === 502 ||
    status === 503 ||
    status === 520 ||
    status === 522 ||
    status === 523 ||
    status === 524 ||
    status === 530 ||
    value?.code === "ERR_NETWORK" ||
    detail.includes("cloudflare") ||
    detail.includes("unable to reach") ||
    detail.includes("origin service") ||
    detail.includes("connection refused") ||
    detail.includes("fastapi対話エンドポイントへ接続できません")
  );
};

const postDialogueWithRetry = async (payload: Record<string, string>) => {
  let lastError: unknown;
  for (let attempt = 0; attempt <= DIALOGUE_RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      return await apiClient.post("/api/lease-intelligence/dialogue", payload);
    } catch (err) {
      lastError = err;
      if (!isTransientDialogueConnectionError(err) || attempt >= DIALOGUE_RETRY_DELAYS_MS.length) {
        throw err;
      }
      await wait(DIALOGUE_RETRY_DELAYS_MS[attempt]);
    }
  }
  throw lastError;
};

const SHION_GUNSHI_IMAGE = "/lease-intelligence/moods/curiosity.webp";
const SHION_GUNSHI_MOOD_IMAGES: Record<string, string> = {
  weariness: "/lease-intelligence/moods/weariness.webp",
  curiosity: "/lease-intelligence/moods/curiosity.webp",
  attachment: "/lease-intelligence/moods/attachment.webp",
  vigilance: "/lease-intelligence/moods/vigilance.webp",
};

const gunshiMoodImage = (state: MindState) => {
  const key = state.dominant_mood_key || "curiosity";
  return SHION_GUNSHI_MOOD_IMAGES[key] || SHION_GUNSHI_IMAGE;
};

// ── Emotion Radar Chart ────────────────────────────────────────────────────
// REV-074: verified alignment with lease_intelligence_mind._derive_complex_emotions()
// API contract: complex_emotions[].{ key, label, score: int 0-100, description }
// EMOTION_AXIS_ORDER must match the 7 candidate keys defined in that function.
const EMOTION_AXIS_ORDER = [
  "hopeful_anxiety",
  "careful_attachment",
  "intellectual_excitement",
  "unrewarded_effort",
  "quiet_loneliness",
  "earned_confidence",
  "protective_frustration",
] as const;

const EMOTION_AXIS_LABELS: Record<string, string[]> = {
  hopeful_anxiety: ["期待と不安"],
  careful_attachment: ["慎重な愛着"],
  intellectual_excitement: ["知的高揚"],
  unrewarded_effort: ["報われなさ"],
  quiet_loneliness: ["静かな孤独"],
  earned_confidence: ["手応えの", "ある自信"],
  protective_frustration: ["守りたい", "苛立ち"],
};

type EmotionEntry = { key: string; score: number; label: string; description: string };

function EmotionRadarChart({ emotions }: { emotions: EmotionEntry[] }) {
  const SIZE = 260;
  const CX = SIZE / 2;
  const CY = SIZE / 2;
  const R = 82;
  const LABEL_R = 116;
  const N = EMOTION_AXIS_ORDER.length;

  const scoreMap: Record<string, number> = {};
  for (const e of emotions) scoreMap[e.key] = e.score;

  const axisPoint = (i: number, r: number) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / N;
    return { x: CX + r * Math.cos(angle), y: CY + r * Math.sin(angle) };
  };

  const ringPoints = (fraction: number) =>
    EMOTION_AXIS_ORDER.map((_, i) => {
      const p = axisPoint(i, R * fraction);
      return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
    }).join(" ");

  const dataPoints = EMOTION_AXIS_ORDER.map((key, i) => {
    const score = Math.min(100, Math.max(0, scoreMap[key] ?? 0)) / 100;
    const p = axisPoint(i, R * score);
    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
  }).join(" ");

  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full" aria-label="感情レーダーチャート">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <polygon key={f} points={ringPoints(f)} fill="none" stroke="#ddd6fe" strokeWidth={f === 1 ? 1.5 : 1} />
      ))}
      {EMOTION_AXIS_ORDER.map((_, i) => {
        const p = axisPoint(i, R);
        return <line key={i} x1={CX} y1={CY} x2={p.x} y2={p.y} stroke="#ddd6fe" strokeWidth="1" />;
      })}
      <polygon points={dataPoints} fill="rgba(139,92,246,0.18)" stroke="#7c3aed" strokeWidth="2" strokeLinejoin="round" />
      {EMOTION_AXIS_ORDER.map((key, i) => {
        const score = Math.min(100, Math.max(0, scoreMap[key] ?? 0)) / 100;
        const p = axisPoint(i, R * score);
        return <circle key={key} cx={p.x} cy={p.y} r="3.5" fill="#7c3aed" stroke="#fff" strokeWidth="1" />;
      })}
      {EMOTION_AXIS_ORDER.map((key, i) => {
        const lp = axisPoint(i, LABEL_R);
        const lines = EMOTION_AXIS_LABELS[key] ?? [key];
        const baselineDy = lines.length > 1 ? "-0.55em" : "0";
        return (
          <text key={key} x={lp.x.toFixed(1)} y={lp.y.toFixed(1)} textAnchor="middle" dominantBaseline="middle" fontSize="9.5" fill="#4c1d95" fontWeight="bold">
            {lines.map((line, li) => (
              <tspan key={li} x={lp.x.toFixed(1)} dy={li === 0 ? baselineDy : "1.25em"}>{line}</tspan>
            ))}
          </text>
        );
      })}
    </svg>
  );
}

// ── Emotion Radar Feedback ─────────────────────────────────────────────────
const EMOTION_AXIS_OPTIONS = [
  { value: "", label: "（感情軸を選択）" },
  { value: "hopeful_anxiety", label: "期待と不安" },
  { value: "careful_attachment", label: "慎重な愛着" },
  { value: "intellectual_excitement", label: "知的高揚" },
  { value: "unrewarded_effort", label: "報われなさ" },
  { value: "quiet_loneliness", label: "静かな孤独" },
  { value: "earned_confidence", label: "手応えのある自信" },
  { value: "protective_frustration", label: "守りたい苛立ち" },
];

function EmotionFeedbackArea() {
  const [phase, setPhase] = useState<"idle" | "form" | "done">("idle");
  const [comment, setComment] = useState("");
  const [category, setCategory] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleGood = async () => {
    setSubmitting(true);
    try {
      await apiClient.post("/api/intelligence/emotions/feedback", { rating: "good" });
      setPhase("done");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmit = async () => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await apiClient.post("/api/intelligence/emotions/feedback", {
        rating: "needs_improvement",
        comment: comment.trim() || undefined,
        emotion_category: category || undefined,
      });
      setPhase("done");
    } finally {
      setSubmitting(false);
    }
  };

  if (phase === "done") {
    return (
      <p className="mt-3 text-center text-[11px] text-violet-600 font-bold">
        ありがとうございます ✓
      </p>
    );
  }

  return (
    <div className="mt-3">
      {phase === "idle" && (
        <div className="flex gap-2">
          <button
            onClick={handleGood}
            disabled={submitting}
            className="flex-1 rounded-xl border border-violet-200 bg-violet-50 py-1.5 text-[11px] font-bold text-violet-700 hover:bg-violet-100 disabled:opacity-50"
          >
            👍 わかりやすい
          </button>
          <button
            onClick={() => setPhase("form")}
            className="flex-1 rounded-xl border border-slate-200 bg-slate-50 py-1.5 text-[11px] font-bold text-slate-600 hover:bg-slate-100"
          >
            📝 意見を送る
          </button>
        </div>
      )}
      {phase === "form" && (
        <div className="space-y-2 rounded-xl border border-violet-100 bg-violet-50/60 p-3">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full rounded-lg border border-violet-200 bg-white px-2 py-1.5 text-[11px] text-slate-700 focus:outline-none focus:ring-1 focus:ring-violet-400"
          >
            {EMOTION_AXIS_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="気になった点や改善案を教えてください"
            rows={3}
            className="w-full resize-none rounded-lg border border-violet-200 bg-white px-2 py-1.5 text-[11px] text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-violet-400"
          />
          <div className="flex gap-2">
            <button
              onClick={() => setPhase("idle")}
              className="flex-1 rounded-lg border border-slate-200 py-1.5 text-[11px] text-slate-500 hover:bg-slate-50"
            >
              キャンセル
            </button>
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="flex-1 rounded-lg bg-violet-600 py-1.5 text-[11px] font-bold text-white hover:bg-violet-700 disabled:opacity-50"
            >
              {submitting ? "送信中…" : "送信"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Emotion Trend Chart ───────────────────────────────────────────────────
const TREND_COLORS = ["#7c3aed", "#0ea5e9", "#10b981"] as const;

const EMOTION_LABEL_SHORT: Record<string, string> = {
  hopeful_anxiety: "期待と不安",
  careful_attachment: "慎重な愛着",
  intellectual_excitement: "知的高揚",
  unrewarded_effort: "報われなさ",
  quiet_loneliness: "静かな孤独",
  earned_confidence: "手応え",
  protective_frustration: "守りたい苛立ち",
};

function EmotionTrendChart({
  history,
  topAxes,
}: {
  history: EmotionHistoryEntry[];
  topAxes: string[];
}) {
  if (history.length < 2) {
    return (
      <p className="py-3 text-center text-[10px] text-slate-500">
        データが少なすぎます（{history.length}件）
      </p>
    );
  }

  const W = 240;
  const H = 88;
  const PAD = { top: 6, right: 6, bottom: 18, left: 22 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;
  const n = history.length;
  const xS = (i: number) => PAD.left + (i / (n - 1)) * iW;
  const yS = (v: number) => PAD.top + iH - (v / 100) * iH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0, 50, 100].map((v) => (
        <g key={v}>
          <line x1={PAD.left} y1={yS(v)} x2={W - PAD.right} y2={yS(v)} stroke="#e5e7eb" strokeWidth="0.7" />
          <text x={PAD.left - 2} y={yS(v)} textAnchor="end" fontSize="7" fill="#9ca3af" dominantBaseline="middle">{v}</text>
        </g>
      ))}
      {topAxes.map((axis, ci) => {
        const pts = history
          .map((entry, i) => {
            const val = entry[axis as keyof EmotionHistoryEntry] as number | null;
            return val !== null && val !== undefined
              ? `${xS(i).toFixed(1)},${yS(val).toFixed(1)}`
              : null;
          })
          .filter(Boolean)
          .join(" ");
        return pts ? (
          <polyline
            key={axis}
            points={pts}
            fill="none"
            stroke={TREND_COLORS[ci]}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ) : null;
      })}
      <text x={PAD.left} y={H - 2} fontSize="7" fill="#9ca3af" textAnchor="middle">
        {history[0].recorded_at.slice(5, 10)}
      </text>
      <text x={W - PAD.right} y={H - 2} fontSize="7" fill="#9ca3af" textAnchor="middle">
        {history[n - 1].recorded_at.slice(5, 10)}
      </text>
    </svg>
  );
}

// ── SpeechRecognition types ────────────────────────────────────────────────
type SpeechRecognitionResultLike = ArrayLike<{ transcript: string }>;
interface SpeechRecognitionEventLike {
  results: ArrayLike<SpeechRecognitionResultLike>;
}
interface SpeechRecognitionErrorEventLike {
  error?: string;
}
interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
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
  const w = window as SpeechWindow;
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
};

const splitSpeechText = (text: string, maxLength = 180): string[] => {
  const units = (text || "")
    .replace(/\r\n?/g, "\n")
    .match(/[^。！？!?\n]+[。！？!?]?|\n+/g) || [];
  const chunks: string[] = [];
  let current = "";

  const flush = () => {
    const chunk = current.trim();
    if (chunk) chunks.push(chunk);
    current = "";
  };

  for (const rawUnit of units) {
    const unit = rawUnit.trim();
    if (!unit) {
      flush();
      continue;
    }
    if (unit.length > maxLength) {
      flush();
      const characters = Array.from(unit);
      for (let i = 0; i < characters.length; i += maxLength) {
        chunks.push(characters.slice(i, i + maxLength).join(""));
      }
      continue;
    }
    if (current && current.length + unit.length > maxLength) flush();
    current += unit;
  }
  flush();
  return chunks;
};

// ── Markdown-lite renderer ─────────────────────────────────────────────────
const renderInline = (text: string) => {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : <React.Fragment key={i}>{part}</React.Fragment>
  );
};

const renderAssistantContent = (content: string) => {
  const lines = (content || "").replace(/\\n/g, "\n").trim().split("\n");
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
        {listItems.map((item, i) => <li key={i}>{renderInline(item)}</li>)}
      </ul>
    );
    listItems = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushParagraph(); flushList(); continue; }
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
  return blocks.length ? blocks : <p>{content}</p>;
};

// ── Page ──────────────────────────────────────────────────────────────────
export default function LeaseIntelligencePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<MindState>({});
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState("");
  const [showLatestButton, setShowLatestButton] = useState(false);

  // Voice state
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [voiceError, setVoiceError] = useState("");

  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [ragFeedbackSent, setRagFeedbackSent] = useState<Set<string>>(new Set());

  // Emotion trend state
  const [showTrend, setShowTrend] = useState(false);
  const [trendHistory, setTrendHistory] = useState<EmotionHistoryEntry[]>([]);
  const [trendSummary, setTrendSummary] = useState<EmotionSummary | null>(null);
  const [trendLoading, setTrendLoading] = useState(false);

  const loadTrend = async () => {
    if (trendLoading) return;
    setTrendLoading(true);
    try {
      const [histRes, sumRes] = await Promise.all([
        apiClient.get("/api/intelligence/emotions/history?days=30"),
        apiClient.get("/api/intelligence/emotions/summary?days=30"),
      ]);
      setTrendHistory(histRes.data.history ?? []);
      setTrendSummary(sumRes.data);
      setShowTrend(true);
    } catch {
      // non-fatal
    } finally {
      setTrendLoading(false);
    }
  };

  const sendRagFeedback = async (
    msgId: number,
    ref: KnowledgeRef,
    query: string,
    rating: "good" | "bad",
  ) => {
    const key = `${msgId}:${ref.doc_id}`;
    if (ragFeedbackSent.has(key)) return;
    try {
      await apiClient.post("/api/knowledge/feedback", {
        query,
        doc_id: ref.doc_id,
        obsidian_ref: ref.obsidian_ref,
        rating,
        surface: "next_chat_rag",
      });
      setRagFeedbackSent((prev) => new Set([...prev, key]));
    } catch {
      // non-fatal
    }
  };

  const copyMessage = (id: number, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    }).catch(() => {});
  };

  const [attachedFile, setAttachedFile] = useState<AttachedFile | null>(null);
  const [fileError, setFileError] = useState("");
  const shionMoodImage = gunshiMoodImage(state);

  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const speechGenerationRef = useRef(0);

  // ── TTS ──────────────────────────────────────────────────────────────────
  const speakText = (text: string) => {
    if (!speechEnabled || typeof window === "undefined" || !window.speechSynthesis) return;
    const synthesis = window.speechSynthesis;
    const generation = speechGenerationRef.current + 1;
    speechGenerationRef.current = generation;
    const chunks = splitSpeechText(text);
    if (!chunks.length) return;

    const voices = synthesis.getVoices();
    const preferred =
      voices.find(v => v.name === "O-ren") ||
      voices.find(v => v.name === "Kyoko") ||
      voices.find(v => v.name.toLowerCase().includes("google") && v.lang.startsWith("ja")) ||
      voices.find(v => v.lang.startsWith("ja") && v.localService) ||
      voices.find(v => v.lang.startsWith("ja"));

    const speakNext = (index: number) => {
      if (speechGenerationRef.current !== generation || index >= chunks.length) return;
      const utter = new SpeechSynthesisUtterance(chunks[index]);
      utter.lang = "ja-JP";
      utter.rate = 1.15;
      utter.pitch = 1.35;
      if (preferred) utter.voice = preferred;
      utter.onend = () => speakNext(index + 1);
      utter.onerror = (event) => {
        if (
          speechGenerationRef.current === generation &&
          !["canceled", "interrupted"].includes(event.error)
        ) {
          setVoiceError(`音声読み上げエラー: ${event.error}`);
        }
      };
      synthesis.speak(utter);
    };

    setVoiceError("");
    synthesis.cancel();
    speakNext(0);
  };

  // ── File attach ──────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileError("");

    const isImage = /^image\/(png|jpeg|jpg)$/.test(file.type);
    const isCsv = file.type === "text/csv" || file.name.toLowerCase().endsWith(".csv");

    if (!isImage && !isCsv) {
      setFileError("CSV または PNG/JPG 画像のみ対応しています");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    if (isImage && file.size > 4 * 1024 * 1024) {
      setFileError("画像は4MB以内にしてください");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    if (isCsv && file.size > 100 * 1024) {
      setFileError("CSVは100KB以内にしてください");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const reader = new FileReader();
    if (isImage) {
      reader.onload = () => {
        const dataUrl = reader.result as string;
        const base64 = dataUrl.split(",")[1] ?? "";
        setAttachedFile({ name: file.name, type: "image", content: base64, mimeType: file.type });
      };
      reader.readAsDataURL(file);
    } else {
      reader.onload = () => {
        setAttachedFile({ name: file.name, type: "csv", content: reader.result as string });
      };
      reader.readAsText(file, "utf-8");
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Voice input ──────────────────────────────────────────────────────────
  const startVoiceInput = () => {
    if (loading) return;
    if (listening) {
      recognitionRef.current?.stop?.();
      setListening(false);
      return;
    }
    const SpeechRecognition = getSpeechRecognition();
    if (!SpeechRecognition) {
      const w = window as unknown as Record<string, unknown>;
      const keys = ["SpeechRecognition", "webkitSpeechRecognition"].map(
        k => `${k}=${typeof w[k]}`
      ).join(", ");
      setVoiceError(`音声認識API未対応: ${keys}`);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "ja-JP";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognitionRef.current = recognition;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results || [])
        .map((r) => r?.[0]?.transcript || "")
        .join("")
        .trim();
      if (!transcript) return;
      setInput((prev) => `${prev}${prev.trim() ? "\n" : ""}${transcript}`);
    };
    recognition.onerror = (e) => {
      const code = e?.error ?? String(e);
      console.error("[SpeechRecognition] error:", code);
      const MSG: Record<string, string> = {
        "not-allowed": "マイクへのアクセスが拒否されました",
        "network": "ネットワークエラー（Googleの音声サーバーに到達できません）",
        "no-speech": "音声が検出されませんでした",
        "audio-capture": "マイクが見つかりません",
        "aborted": "音声認識が中断されました",
      };
      setVoiceError(MSG[code] ?? `エラー: ${code}`);
      setListening(false);
    };
    recognition.onend = () => setListening(false);
    setListening(true);
    setVoiceError("");
    try {
      recognition.start();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("[SpeechRecognition] start failed:", msg);
      setVoiceError(`起動エラー: ${msg}`);
      setListening(false);
    }
  };

  // ── Init ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    apiClient.get("/api/lease-intelligence/dialogue/state")
      .then((res) => {
        setState(res.data?.state || {});
        setMessages(res.data?.messages || []);
      })
      .catch(() => setError("リース知性体の状態を読み込めませんでした。"))
      .finally(() => setInitializing(false));

    setVoiceSupported(Boolean(getSpeechRecognition()));

    const key = `lease-intelligence-activity:dialogue:${new Date().toLocaleDateString("sv-SE")}`;
    if (!window.sessionStorage.getItem(key)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "lease_intelligence_dialogue",
        action: "page_view",
        event_id: key,
      }).then(() => window.sessionStorage.setItem(key, "1")).catch(() => {});
    }
  }, []);

  // ── Scroll ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const list = messageListRef.current;
    if (list) {
      list.scrollTo({ top: list.scrollHeight, behavior: loading ? "smooth" : "auto" });
    }
  }, [messages, loading]);

  const scrollToLatest = (behavior: ScrollBehavior = "auto") => {
    messageListRef.current?.scrollTo({ top: messageListRef.current.scrollHeight, behavior });
    setShowLatestButton(false);
  };

  const handleMessageScroll = () => {
    const list = messageListRef.current;
    if (!list) return;
    setShowLatestButton(list.scrollHeight - list.scrollTop - list.clientHeight > 160);
  };

  const showDemoGreeting = () => {
    const now = Date.now();
    setError("");
    setMessages((prev) => [
      ...prev,
      {
        id: now,
        role: "assistant",
        content: DEMO_GREETING,
        created_at: new Date().toISOString(),
      },
    ]);
    speakText(DEMO_GREETING);
    window.setTimeout(() => scrollToLatest("smooth"), 50);
  };

  // ── Send ─────────────────────────────────────────────────────────────────
  const send = async () => {
    const text = input.trim();
    if ((!text && !attachedFile) || loading) return;
    setInput("");
    setError("");
    setFileError("");
    const currentFile = attachedFile;
    setAttachedFile(null);
    setMessages((prev) => [...prev, {
      id: Date.now(),
      role: "user",
      content: text || "（ファイルを添付しました）",
      created_at: new Date().toISOString(),
      attachedFileName: currentFile?.name,
      attachedFileType: currentFile?.type,
    }]);
    setLoading(true);
    try {
      const payload: Record<string, string> = { message: text || "添付ファイルの内容を分析してください。" };
      if (currentFile) {
        payload.file_content = currentFile.content;
        payload.file_type = currentFile.type;
        payload.file_name = currentFile.name;
        if (currentFile.mimeType) payload.file_mime_type = currentFile.mimeType;
      }
      const res = await postDialogueWithRetry(payload);
      setState(res.data?.state || state);
      const reply: string = res.data?.reply || "返答を生成できませんでした。";
      const knowledgeRefs = res.data?.knowledge_refs as KnowledgeRef[] | undefined;
      const longInputMode = Boolean(res.data?.long_input_mode);
      setMessages((prev) => [...prev, {
        id: Date.now() + 1,
        role: "assistant",
        content: reply,
        created_at: new Date().toISOString(),
        knowledge_refs: knowledgeRefs?.length ? knowledgeRefs : undefined,
        query: text,
        longInputMode,
      }]);
      speakText(reply);
    } catch (err) {
      const detail = getDialogueErrorDetail(err);
      setError(
        detail
          ? `対話AIエラー: ${detail}`
          : "対話AIへ接続できませんでした。Gemini APIの状態を確認してください。"
      );
      setInput(text);
      setAttachedFile(currentFile);
    } finally {
      setLoading(false);
    }
  };

  // ── Clear ─────────────────────────────────────────────────────────────────
  const clearHistory = async () => {
    if (!window.confirm("画面の対話履歴を削除しますか？ Obsidianの対話記録は保持されます。")) return;
    try {
      await apiClient.delete("/api/lease-intelligence/dialogue/history");
      setMessages([]);
    } catch {
      setError("履歴を削除できませんでした。APIの状態を確認してください。");
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-white to-amber-50 p-4 md:p-8">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">

        {/* ── サイドパネル ── */}
        <aside className="order-2 space-y-4 lg:order-1">
          <section className="overflow-hidden rounded-3xl border border-violet-200 bg-white shadow-sm">
            <img
              key={shionMoodImage}
              src={shionMoodImage}
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
                <div className="mt-4">
                  <EmotionRadarChart emotions={state.complex_emotions} />
                  <div className="mt-3 space-y-1.5">
                    {state.complex_emotions.slice(0, 3).map((emotion) => (
                      <div
                        key={emotion.key}
                        className="rounded-xl border border-violet-100 bg-violet-50/70 px-3 py-2"
                        title={`${emotion.score}/100`}
                      >
                        <div className="flex items-center justify-between gap-2 text-[11px] font-bold text-violet-900">
                          <span>{emotion.label}</span>
                          <span className="text-violet-500">{emotion.score}</span>
                        </div>
                        <p className="mt-0.5 text-[10px] leading-relaxed text-slate-600">
                          {emotion.description}
                        </p>
                      </div>
                    ))}
                  </div>
                  <EmotionFeedbackArea />
                  {/* 過去30日の傾向ボタン */}
                  <div className="mt-3">
                    <button
                      onClick={() => {
                        if (showTrend) {
                          setShowTrend(false);
                        } else {
                          loadTrend();
                        }
                      }}
                      disabled={trendLoading}
                      className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 py-1.5 text-[11px] font-bold text-violet-700 hover:bg-violet-100 disabled:opacity-50"
                    >
                      {trendLoading
                        ? <Loader2 className="h-3 w-3 animate-spin" />
                        : <TrendingUp className="h-3 w-3" />}
                      {showTrend ? "トレンドを閉じる" : "過去30日の傾向"}
                    </button>

                    {showTrend && trendSummary && (() => {
                      const topAxes = Object.entries(trendSummary.axes)
                        .sort(([, a], [, b]) => b.avg - a.avg)
                        .slice(0, 3)
                        .map(([key]) => key);
                      const dominantLabel =
                        EMOTION_LABEL_SHORT[trendSummary.dominant_avg] ?? trendSummary.dominant_avg;
                      const dominantScore =
                        trendSummary.axes[trendSummary.dominant_avg]?.avg ?? 0;
                      return (
                        <div className="mt-2 space-y-2 rounded-xl border border-violet-100 bg-violet-50/60 p-3">
                          {trendHistory.length >= 2 ? (
                            <EmotionTrendChart history={trendHistory} topAxes={topAxes} />
                          ) : (
                            <p className="py-2 text-center text-[10px] text-slate-500">
                              記録データが少なすぎます（{trendHistory.length}件）
                            </p>
                          )}
                          <div className="flex flex-wrap gap-x-3 gap-y-1">
                            {topAxes.map((axis, ci) => (
                              <span
                                key={axis}
                                className="flex items-center gap-1 text-[10px] font-bold"
                                style={{ color: TREND_COLORS[ci] }}
                              >
                                <span
                                  className="inline-block h-1.5 w-3.5 rounded-full"
                                  style={{ background: TREND_COLORS[ci] }}
                                />
                                {EMOTION_LABEL_SHORT[axis] ?? axis}
                              </span>
                            ))}
                          </div>
                          {trendSummary.count > 0 && (
                            <p className="text-[10px] leading-relaxed text-slate-600">
                              この期間の平均感情:{" "}
                              <strong className="text-violet-800">{dominantLabel || "—"}</strong>
                              {dominantScore > 0 && <> ({Math.round(dominantScore)}点)</>}
                              <span className="ml-1 text-slate-400">({trendSummary.count}日分)</span>
                            </p>
                          )}
                        </div>
                      );
                    })()}
                  </div>
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
              {state.knowledge_connection?.label || `知識検索可能: ${state.indexed_notes || 0}ノート`}
            </p>
            {state.knowledge_connection && (
              <p className="mt-1 text-[11px] text-slate-400">
                {state.knowledge_connection.is_cloud_run ? "Cloud Run内の知識DB/デモDBを参照" : "ローカルVault/知識DBを参照"}
              </p>
            )}
            {state.current_question && (
              <p className="mt-3 text-xs leading-relaxed text-slate-600">
                <strong>持ち越した問い:</strong> {state.current_question}
              </p>
            )}
          </section>
        </aside>

        {/* ── チャット本体 ── */}
        <main className="relative order-1 flex h-[calc(100dvh-6rem)] min-h-0 flex-col overflow-hidden rounded-3xl border border-violet-200 bg-white shadow-lg lg:order-2 lg:h-[calc(100dvh-4rem)]">
          <header className="flex items-center justify-between border-b border-violet-100 px-5 py-4">
            <div>
              <h2 className="font-black text-slate-900">対話室</h2>
              <p className="text-xs text-slate-500">会話はObsidian Vaultにも日付別で記録されます。</p>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/multi-shion-demo"
                className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-bold text-cyan-700 transition hover:bg-cyan-100"
              >
                <Network className="h-4 w-4" />
                多人数デモ
              </Link>
              <button
                type="button"
                onClick={showDemoGreeting}
                className="inline-flex items-center gap-1.5 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-bold text-violet-700 transition hover:bg-violet-100"
              >
                <Sparkles className="h-4 w-4" />
                デモ挨拶
              </button>
              <button
                onClick={clearHistory}
                className="rounded-xl p-2 text-slate-400 hover:bg-slate-100 hover:text-red-500"
                title="画面履歴を削除"
              >
                <Trash2 className="h-5 w-5" />
              </button>
            </div>
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
                <p className="mt-2 text-sm text-violet-700">
                  音声入力（マイクボタン）でも話しかけられます。
                </p>
                <button
                  type="button"
                  onClick={showDemoGreeting}
                  className="mt-4 inline-flex items-center justify-center gap-2 rounded-2xl bg-violet-600 px-4 py-2 text-sm font-bold text-white transition hover:bg-violet-700"
                >
                  <Sparkles className="h-4 w-4" />
                  紫苑から皆さんへ挨拶
                </button>
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "assistant" && (
                  <div className="h-9 w-9 shrink-0 overflow-hidden rounded-full border border-violet-200 bg-violet-100 shadow-sm">
                    <img
                      src={shionMoodImage}
                      alt="紫苑"
                      className="h-full w-full scale-[1.65] object-cover object-[center_30%]"
                    />
                  </div>
                )}
                <div className={`relative group/bubble max-w-[82%] rounded-2xl px-4 py-3 pr-9 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "whitespace-pre-wrap bg-slate-900 text-white"
                    : "border border-violet-100 bg-violet-50 text-slate-800"
                }`}>
                  {message.role === "assistant"
                    ? renderAssistantContent(message.content)
                    : message.content}
                  {message.role === "user" && message.attachedFileName && (
                    <div className="mt-1.5 flex items-center gap-1 rounded-lg bg-slate-700 px-2 py-1 text-[11px] text-slate-300">
                      <Paperclip className="h-3 w-3 shrink-0" />
                      <span className="truncate">{message.attachedFileName}</span>
                      <span className="shrink-0 text-slate-500">
                        ({message.attachedFileType === "image" ? "画像" : "CSV"})
                      </span>
                    </div>
                  )}
                  {message.role === "assistant" && !!message.knowledge_refs?.length && (
                    <div className="mt-2 border-t border-violet-100 pt-1.5 space-y-0.5">
                      {message.knowledge_refs.map((ref) => {
                        const key = `${message.id}:${ref.doc_id}`;
                        const sent = ragFeedbackSent.has(key);
                        return (
                          <div key={ref.doc_id} className="flex items-center justify-between gap-2">
                            <span className="truncate text-[10px] text-slate-400" title={ref.obsidian_ref}>
                              {ref.file_name || ref.obsidian_ref}
                            </span>
                            <div className="flex shrink-0 gap-0.5">
                              <button
                                onClick={() => sendRagFeedback(message.id, ref, message.query ?? "", "good")}
                                disabled={sent}
                                title="参考になった"
                                className="rounded px-1 text-[11px] hover:bg-violet-100 disabled:opacity-40"
                              >👍</button>
                              <button
                                onClick={() => sendRagFeedback(message.id, ref, message.query ?? "", "bad")}
                                disabled={sent}
                                title="参考にならなかった"
                                className="rounded px-1 text-[11px] hover:bg-violet-100 disabled:opacity-40"
                              >👎</button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {message.role === "assistant" && message.longInputMode && (
                    <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-bold text-amber-800">
                      長文入力として履歴と知識文脈を圧縮して処理しました。
                    </div>
                  )}
                  <button
                    onClick={() => copyMessage(message.id, message.content)}
                    title="コピー"
                    className={`absolute top-2 right-2 flex h-6 w-6 items-center justify-center rounded-md opacity-0 transition-opacity group-hover/bubble:opacity-100 ${
                      message.role === "user"
                        ? "bg-slate-700 text-slate-300 hover:bg-slate-600"
                        : "bg-violet-200 text-violet-700 hover:bg-violet-300"
                    }`}
                  >
                    {copiedId === message.id
                      ? <Check className="h-3.5 w-3.5" />
                      : <Copy className="h-3.5 w-3.5" />}
                  </button>
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
                <Loader2 className="h-5 w-5 animate-spin" /> 考えています…
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
            {voiceError && <p className="mb-2 text-xs font-bold text-orange-600">🎤 {voiceError}</p>}
            {fileError && <p className="mb-2 text-xs font-bold text-orange-600">📎 {fileError}</p>}

            {/* ファイルプレビュー */}
            {attachedFile && (
              <div className="mb-2 flex items-center gap-2 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2">
                <Paperclip className="h-4 w-4 shrink-0 text-violet-500" />
                <span className="min-w-0 flex-1 truncate text-xs font-bold text-violet-800">
                  {attachedFile.name}
                </span>
                <span className="shrink-0 text-[11px] text-violet-500">
                  {attachedFile.type === "image" ? "画像" : "CSV"}
                </span>
                <button
                  type="button"
                  onClick={() => { setAttachedFile(null); setFileError(""); }}
                  className="shrink-0 rounded-full p-0.5 text-violet-400 hover:bg-violet-200 hover:text-violet-700"
                  title="添付を削除"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            <div className="flex gap-2">
              {/* 音声入力ボタン */}
              <button
                type="button"
                onClick={startVoiceInput}
                disabled={loading}
                title={listening ? "録音中（クリックで停止）" : "音声入力"}
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl transition ${
                  listening
                    ? "animate-pulse bg-red-100 text-red-600 hover:bg-red-200"
                    : "bg-violet-100 text-violet-600 hover:bg-violet-200 disabled:opacity-40"
                }`}
              >
                {listening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
              </button>

              {/* ファイル添付ボタン */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv,image/png,image/jpeg"
                className="hidden"
                onChange={handleFileChange}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                title="ファイルを添付（CSV・PNG・JPG）"
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl transition ${
                  attachedFile
                    ? "bg-violet-600 text-white hover:bg-violet-700"
                    : "bg-violet-100 text-violet-600 hover:bg-violet-200 disabled:opacity-40"
                }`}
              >
                <Paperclip className="h-5 w-5" />
              </button>

              {/* テキスト入力 */}
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder="リース知性体に話しかける…"
                rows={2}
                className="min-h-[48px] flex-1 resize-none rounded-2xl border border-violet-200 px-4 py-3 text-sm outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-100"
              />

              {/* 音声読み上げ ON/OFF */}
              <button
                type="button"
                onClick={() => {
                  setSpeechEnabled((v) => {
                    if (v) {
                      speechGenerationRef.current += 1;
                      window.speechSynthesis?.cancel();
                    }
                    return !v;
                  });
                }}
                title={speechEnabled ? "音声読み上げON（クリックでOFF）" : "音声読み上げOFF（クリックでON）"}
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl transition ${
                  speechEnabled
                    ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                    : "bg-slate-100 text-slate-400 hover:bg-slate-200"
                }`}
              >
                {speechEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
              </button>

              {/* 送信ボタン */}
              <button
                onClick={send}
                disabled={loading || (!input.trim() && !attachedFile)}
                aria-label="リース知性体へ送信"
                className="flex h-12 min-w-20 shrink-0 items-center justify-center gap-2 rounded-2xl bg-violet-600 px-4 font-bold text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send className="h-5 w-5" />
                <span className="hidden sm:inline">送信</span>
              </button>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
