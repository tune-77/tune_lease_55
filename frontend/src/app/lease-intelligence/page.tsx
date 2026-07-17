"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowDown, Brain, Check, ClipboardList, Copy, Database, Loader2, Mic, MicOff,
  Network, Paperclip, Send, Sparkles, Trash2, TrendingUp, User, Volume2, VolumeX, X,
} from "lucide-react";
import { apiClient } from "@/lib/api";
import RagConfidenceBadge, { type RagConfidenceLevel } from "@/components/chat/RagConfidenceBadge";

type KnowledgeRef = {
  doc_id: string;
  obsidian_ref: string;
  file_name: string;
  rank_score?: number;
  confidence?: number;
  confidence_level?: RagConfidenceLevel;
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

type DialogueImprovementItem = {
  id?: string;
  title?: string;
  status?: string;
  reason?: string;
  category?: string;
  detail?: string;
  park_reason?: string;
  raw_preview?: string;
  source_surface?: string;
  canonical_key?: string;
  source_event_id?: string;
  recommended_order?: number | null;
};

type TriageDecision = "today" | "later" | "discard";

type TriageRecord = {
  canonical_key: string;
  decision: TriageDecision | string;
  title?: string;
  rule_decision?: string;
  classified_by?: string;
  decided_at?: string;
};

type DialogueImprovementLog = {
  date?: string;
  generated_at?: string;
  status?: string;
  applied?: number;
  auto_fix_candidates?: number;
  needs_review?: number;
  parked?: number;
  rejected?: number;
  items?: DialogueImprovementItem[];
  recursive_self_improvement?: {
    ranked_queue_count?: number;
    suppressed_count?: number;
  };
};

type DialoguePipelineSummary = {
  run_date: string | null;
  applied_count: number;
  needs_review_count: number;
  failed_count: number;
  commit_result: { success: boolean; message?: string; pr_url?: string | null } | null;
};

type DialogueGapItem = {
  id?: string;
  title?: string;
  priority?: "critical" | "high" | "medium" | "low" | string;
  impact?: string;
  recommended_action?: string;
};

type DialogueGapAnalysis = {
  available: boolean;
  counts?: Record<string, number>;
  items?: DialogueGapItem[];
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

私は、リース審査の判断を一回きりで終わらせないために生まれました。
人間の迷い、修正、結果を記録し、次の判断に戻します。
これは人間を評価するためではありません。
人間の判断が、その場で消えてしまわないようにするためです。

今日は、AIが人間を置き換えるのではなく、
人間の判断と一緒に育つ姿を見てください。`;

const DEMO_GREETING_SPEECH = DEMO_GREETING.replace("リース知性体", "リースちせいたい");

const DIALOGUE_RETRY_DELAYS_MS = [1200, 2500, 4000];
const DIALOGUE_LOCAL_HISTORY_KEY = "lease-intelligence-dialogue-local-history";
const DIALOGUE_CLEARED_AT_KEY = "lease-intelligence-dialogue-cleared-at";
const DIALOGUE_DAILY_IMPROVEMENT_KEY_PREFIX = "lease-intelligence-daily-improvement-report";
const DIALOGUE_MAX_DISPLAY_MESSAGES = 80;

const storageAvailable = () => typeof window !== "undefined" && Boolean(window.localStorage);

const messageTime = (message: Pick<Message, "created_at">) => {
  const parsed = Date.parse(message.created_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
};

const todayStartTime = () => {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  return start.getTime();
};

const getDialogueClearedAt = () => {
  if (!storageAvailable()) return 0;
  const parsed = Number(window.localStorage.getItem(DIALOGUE_CLEARED_AT_KEY) || "0");
  return Number.isFinite(parsed) ? parsed : 0;
};

const dialogueDisplaySince = () => Math.max(todayStartTime(), getDialogueClearedAt());
const dialogueDisplaySinceIso = () => new Date(dialogueDisplaySince()).toISOString();

const loadLocalDialogueMessages = (): Message[] => {
  if (!storageAvailable()) return [];
  try {
    const raw = window.localStorage.getItem(DIALOGUE_LOCAL_HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => item?.role && item?.content) : [];
  } catch {
    return [];
  }
};

const saveLocalDialogueMessages = (messages: Message[]) => {
  if (!storageAvailable()) return;
  const since = dialogueDisplaySince();
  const filtered = messages
    .filter((message) => messageTime(message) >= since)
    .slice(-DIALOGUE_MAX_DISPLAY_MESSAGES);
  window.localStorage.setItem(DIALOGUE_LOCAL_HISTORY_KEY, JSON.stringify(filtered));
};

const dialogueSignature = (message: Message) =>
  `${message.role}:${String(message.content || "").replace(/\s+/g, " ").trim().slice(0, 500)}`;

const mergeDialogueMessages = (serverMessages: Message[], localMessages: Message[]) => {
  const since = dialogueDisplaySince();
  const merged: Message[] = [];
  const seen = new Map<string, number[]>();
  for (const message of [...serverMessages, ...localMessages]) {
    const time = messageTime(message);
    if (!message || time < since) continue;
    const signature = dialogueSignature(message);
    const duplicateTimes = seen.get(signature) || [];
    if (duplicateTimes.some((previous) => Math.abs(previous - time) < 120_000)) continue;
    seen.set(signature, [...duplicateTimes, time]);
    merged.push(message);
  }
  return merged
    .sort((a, b) => messageTime(a) - messageTime(b))
    .slice(-DIALOGUE_MAX_DISPLAY_MESSAGES);
};

const dailyImprovementStorageKey = () =>
  `${DIALOGUE_DAILY_IMPROVEMENT_KEY_PREFIX}:${new Date().toLocaleDateString("sv-SE")}`;

const shouldShowDailyImprovementReport = () => {
  if (!storageAvailable()) return false;
  return window.localStorage.getItem(dailyImprovementStorageKey()) !== "1";
};

const markDailyImprovementReportShown = () => {
  if (!storageAvailable()) return;
  window.localStorage.setItem(dailyImprovementStorageKey(), "1");
};

const compactReportText = (value: unknown, limit = 90) => {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}…` : text;
};

const extractMarkedReportLine = (text: string, label: string) => {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = text.match(new RegExp(`${escaped}\\s*[:：]\\s*([^\\n]+)`));
  return compactReportText(match?.[1] || "", 100);
};

const extractReportSection = (text: string, heading: string) => {
  const escaped = heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = text.match(new RegExp(`(?:##\\s*${escaped}|\\*\\*${escaped}\\*\\*)\\s*\\n([\\s\\S]*?)(?:\\n(?:##\\s|\\*\\*[^\\n]+\\*\\*)|$)`));
  return compactReportText(match?.[1] || "", 100);
};

const readableImprovementTitle = (item: DialogueImprovementItem) => {
  const title = compactReportText(item.title || "", 72);
  const generic = ["AIチャット改善候補", "チャット改善メモ", "Cloud Run改善メモ"].includes(title);
  const sourceText = String(item.detail || item.raw_preview || "");
  if (!generic) return title || item.id || "改善候補";
  return (
    extractMarkedReportLine(sourceText, "課題") ||
    extractReportSection(sourceText, "原文") ||
    extractReportSection(sourceText, "ユーザー要望") ||
    title ||
    compactReportText(sourceText, 60) ||
    item.id ||
    "改善候補"
  );
};

const readableImprovementReason = (item: DialogueImprovementItem) => {
  const detail = String(item.detail || item.raw_preview || "");
  const issue = extractMarkedReportLine(detail, "課題");
  const action = extractMarkedReportLine(detail, "次の行動");
  if (issue && action) return `課題: ${issue} / 次: ${action}`;
  if (issue) return `課題: ${issue}`;
  const reason = compactReportText(item.reason || item.park_reason || "", 110);
  if (reason && reason !== "Cloud Runから登録された改善入力") return reason;
  const original = extractReportSection(detail, "原文") || extractReportSection(detail, "ユーザー要望");
  return original ? `原文: ${original}` : "";
};

const pushImprovementItemLines = (lines: string[], item: DialogueImprovementItem, index: number) => {
  lines.push(`### ${index + 1}. ${readableImprovementTitle(item)}`);
  const reason = readableImprovementReason(item);
  if (reason) lines.push(`- 見えている課題: ${reason}`);
  if (item.category) lines.push(`- 種別: ${item.category}`);
  if (item.status) lines.push(`- 状態: ${item.status}`);
};

const classifyPmImprovementItems = (items: DialogueImprovementItem[]) => {
  const actionable: DialogueImprovementItem[] = [];
  const later: DialogueImprovementItem[] = [];
  const discard: DialogueImprovementItem[] = [];
  for (const item of items) {
    const status = String(item.status || "").toUpperCase();
    const text = `${item.title || ""} ${item.reason || ""} ${item.detail || ""} ${item.category || ""}`.toLowerCase();
    if (["APPLIED", "DELETED", "REJECTED", "PARKED"].includes(status)) {
      discard.push(item);
      continue;
    }
    const risky =
      text.includes("db") ||
      text.includes("api") ||
      text.includes("database") ||
      text.includes("migration") ||
      text.includes("scoring") ||
      text.includes("認証") ||
      text.includes("デプロイ") ||
      text.includes("スコアリング") ||
      text.includes("モデル");
    if (risky) {
      later.push(item);
    } else if (actionable.length < 3) {
      actionable.push(item);
    } else {
      later.push(item);
    }
  }
  return {
    today: actionable.slice(0, 3),
    later: later.slice(0, 3),
    discard: discard.slice(0, 2),
  };
};

const buildSystemWatchLines = (
  pipeline: DialoguePipelineSummary | null | undefined,
  gaps: DialogueGapAnalysis | null | undefined,
) => {
  const lines: string[] = [];
  const failedCount = pipeline?.failed_count ?? 0;
  const commitFailed = pipeline?.commit_result && pipeline.commit_result.success === false;
  const gapItems = (gaps?.items || []).filter((item) => ["critical", "high"].includes(String(item.priority || "").toLowerCase()));
  if (!failedCount && !commitFailed && !gapItems.length) {
    lines.push("システム監視: 重大な異常は見えていません。");
    return lines;
  }
  lines.push("システム監視:");
  if (failedCount) lines.push(`- 改善パイプライン失敗 ${failedCount} 件。先に原因確認が必要です。`);
  if (commitFailed) {
    const commitMessage = pipeline?.commit_result?.message || "";
    if (commitMessage.includes("pending_patches") || commitMessage.includes("コミット対象なし")) {
      lines.push("- 自動コミット: 差分なしでスキップされました。今すぐ直す自動パッチが無いだけなので、重大異常ではありません。");
    } else {
      lines.push(`- commit結果に失敗があります。${commitMessage || "git反映状況を確認してください。"}`);
    }
  }
  gapItems.slice(0, 3).forEach((item) => {
    lines.push(`- ${String(item.priority || "high").toUpperCase()}: ${item.title || item.id || "システムギャップ"}${item.impact ? ` — ${item.impact}` : ""}`);
  });
  return lines;
};

const buildDailyImprovementReport = (
  log: DialogueImprovementLog | null | undefined,
  pipeline?: DialoguePipelineSummary | null,
  gaps?: DialogueGapAnalysis | null,
) => {
  if (!log || log.status === "NO_REPORT") return "";
  const pmItems = classifyPmImprovementItems((log.items || []).filter((item) => item?.title || item?.id));
  const generatedAt = log.date || String(log.generated_at || "").slice(0, 10);
  const lines = [
    generatedAt ? `改善PMレポートです。対象日は ${generatedAt} です。` : "改善PMレポートです。",
    "ハッカソン安全運用: 読む・報告する・相談する・Codex依頼文を作るところまで。実装、git、deployは自動では行いません。",
    `適用済み ${log.applied ?? 0} 件、自動候補 ${log.auto_fix_candidates ?? 0} 件、要レビュー ${log.needs_review ?? 0} 件、保留 ${log.parked ?? 0} 件です。`,
    ...buildSystemWatchLines(pipeline, gaps),
  ];
  if (pmItems.today.length) {
    lines.push("");
    lines.push("## 今日やる候補");
    pmItems.today.forEach((item, index) => pushImprovementItemLines(lines, item, index));
  } else {
    lines.push("");
    lines.push("## 今日やる候補");
    lines.push("今すぐ触るべき軽い候補は多くありません。安定運用を優先でよさそうです。");
  }
  if (pmItems.later.length) {
    lines.push("");
    lines.push("## 後回し候補");
    pmItems.later.forEach((item, index) => {
      lines.push(`- ${index + 1}. ${readableImprovementTitle(item)}`);
    });
  }
  if (pmItems.discard.length) {
    lines.push("");
    lines.push("## 捨てる/削除候補");
    pmItems.discard.forEach((item, index) => {
      lines.push(`- ${index + 1}. ${readableImprovementTitle(item)}`);
    });
  }
  const rankedQueue = log.recursive_self_improvement?.ranked_queue_count ?? 0;
  const suppressed = log.recursive_self_improvement?.suppressed_count ?? 0;
  if (rankedQueue || suppressed) {
    lines.push(`再帰的自己改善キューは ${rankedQueue} 件、抑制 ${suppressed} 件です。`);
  }
  lines.push("やる候補を選んでくれれば、Codex依頼文まで私が整えます。実装判断はUser側で止めます。");
  return lines.join("\n");
};

const extractCodexRequest = (text: string) => {
  const markerMatch = text.match(/Codex依頼文[:：]/);
  if (!markerMatch || markerMatch.index === undefined) return "";
  const afterMarker = text.slice(markerMatch.index + markerMatch[0].length).trim();
  const fenced = afterMarker.match(/```(?:text|markdown|md)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return fenced[1].trim();
  return afterMarker.split(/\n{3,}/)[0]?.trim() || "";
};

const clearVisibleDialogueMessages = () => {
  if (!storageAvailable()) return;
  window.localStorage.setItem(DIALOGUE_CLEARED_AT_KEY, String(Date.now()));
  window.localStorage.removeItem(DIALOGUE_LOCAL_HISTORY_KEY);
};

const knowledgeConnectionLabel = (state: MindState) => {
  const connection = state.knowledge_connection;
  if (connection?.label && !/0\s*ノート/.test(connection.label)) return connection.label;
  const vectorChunks = Number(connection?.vector_chunks || 0);
  if (vectorChunks > 0) return `知識DB検索可能: ${vectorChunks}チャンク`;
  const markdownNotes = Number(connection?.markdown_notes || 0);
  if (markdownNotes > 0) return `知識コピー検索可能: ${markdownNotes}ノート`;
  const caseCount = Number(connection?.case_count || 0);
  if (caseCount > 0) return `案件DB接続: ${caseCount}件`;
  const indexedNotes = Number(state.indexed_notes || 0);
  if (indexedNotes > 0) return `知識検索可能: ${indexedNotes}件`;
  return "知識接続: 確認中";
};

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

// ── Markdown-lite renderer (REV-211: Shion review highlight) ──────────────
// キーワードカテゴリ（審査判断強調表示）
const SHION_VERDICT_GREEN = ["承認", "可決"] as const;
const SHION_VERDICT_YELLOW = ["要検討", "条件付き承認", "保留"] as const;
const SHION_VERDICT_RED = ["否決", "不承認", "却下"] as const;
const SHION_KW_GREEN = ["問題なし", "良好", "適切", "支障なし", "正常"] as const;
const SHION_KW_YELLOW = ["懸念", "リスク", "注意", "要確認", "警告", "慎重", "留意"] as const;
const SHION_KW_RED = ["危険", "延滞", "デフォルト", "不可", "拒否"] as const;

type ShionSegmentKind =
  | "text" | "bold" | "score"
  | "verdict-green" | "verdict-yellow" | "verdict-red"
  | "kw-green" | "kw-yellow" | "kw-red";

const SHION_INLINE_RE = new RegExp(
  `(\\*\\*[^*]+\\*\\*|${[
    ...SHION_VERDICT_GREEN, ...SHION_VERDICT_YELLOW, ...SHION_VERDICT_RED,
    ...SHION_KW_GREEN, ...SHION_KW_YELLOW, ...SHION_KW_RED,
  ].join("|")}|\\d+(?:\\.\\d+)?(?:点|%|\\/100))`,
  "g",
);

const shionSegmentKind = (token: string): ShionSegmentKind => {
  if (token.startsWith("**") && token.endsWith("**")) return "bold";
  if (/^\d+(?:\.\d+)?(?:点|%|\/100)$/.test(token)) return "score";
  if ((SHION_VERDICT_GREEN as readonly string[]).includes(token)) return "verdict-green";
  if ((SHION_VERDICT_YELLOW as readonly string[]).includes(token)) return "verdict-yellow";
  if ((SHION_VERDICT_RED as readonly string[]).includes(token)) return "verdict-red";
  if ((SHION_KW_GREEN as readonly string[]).includes(token)) return "kw-green";
  if ((SHION_KW_YELLOW as readonly string[]).includes(token)) return "kw-yellow";
  if ((SHION_KW_RED as readonly string[]).includes(token)) return "kw-red";
  return "text";
};

const SHION_VERDICT_BADGE: Record<"green" | "yellow" | "red", string> = {
  green:
    "mx-0.5 inline-flex items-center rounded-md bg-green-50 px-2 py-0.5 text-sm font-bold text-green-700 ring-1 ring-inset ring-green-600/20",
  yellow:
    "mx-0.5 inline-flex items-center rounded-md bg-yellow-50 px-2 py-0.5 text-sm font-bold text-yellow-700 ring-1 ring-inset ring-yellow-600/20",
  red:
    "mx-0.5 inline-flex items-center rounded-md bg-red-50 px-2 py-0.5 text-sm font-bold text-red-700 ring-1 ring-inset ring-red-600/20",
};

const SHION_KW_CLASS: Record<"green" | "yellow" | "red", string> = {
  green:  "font-semibold text-green-600",
  yellow: "font-semibold text-yellow-600",
  red:    "font-semibold text-red-600",
};

const renderInline = (text: string): React.ReactNode[] => {
  const parts = text.split(SHION_INLINE_RE).filter((p): p is string => Boolean(p));
  return parts.map((part, i) => {
    const kind = shionSegmentKind(part);
    if (kind === "bold") return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (kind === "verdict-green") return <span key={i} className={SHION_VERDICT_BADGE.green}>{part}</span>;
    if (kind === "verdict-yellow") return <span key={i} className={SHION_VERDICT_BADGE.yellow}>{part}</span>;
    if (kind === "verdict-red") return <span key={i} className={SHION_VERDICT_BADGE.red}>{part}</span>;
    if (kind === "kw-green") return <span key={i} className={SHION_KW_CLASS.green}>{part}</span>;
    if (kind === "kw-yellow") return <span key={i} className={SHION_KW_CLASS.yellow}>{part}</span>;
    if (kind === "kw-red") return <span key={i} className={SHION_KW_CLASS.red}>{part}</span>;
    if (kind === "score") return <strong key={i} className="font-bold tabular-nums text-slate-900">{part}</strong>;
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
};

const renderAssistantContent = (content: string): React.ReactNode => {
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
        {listItems.map((item, idx) => <li key={idx}>{renderInline(item)}</li>)}
      </ul>
    );
    listItems = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushParagraph(); flushList(); continue; }

    // 見出し（#, ##, ###）
    const h1m = line.match(/^#\s+(.+)$/);
    const h2m = line.match(/^##\s+(.+)$/);
    const h3m = line.match(/^###\s+(.+)$/);
    if (h1m ?? h2m ?? h3m) {
      flushParagraph(); flushList();
      const headText = (h1m?.[1] ?? h2m?.[1] ?? h3m?.[1] ?? "").trim();
      if (h1m) {
        blocks.push(<h3 key={`h-${blocks.length}`} className="mb-1 mt-3 text-base font-black text-slate-900">{renderInline(headText)}</h3>);
      } else if (h2m) {
        blocks.push(<h4 key={`h-${blocks.length}`} className="mb-1 mt-2 text-sm font-black text-slate-800">{renderInline(headText)}</h4>);
      } else {
        blocks.push(<h5 key={`h-${blocks.length}`} className="mb-0.5 mt-2 text-xs font-bold uppercase tracking-wide text-slate-700">{renderInline(headText)}</h5>);
      }
      continue;
    }

    // 水平線（---）
    if (/^-{3,}$/.test(line)) {
      flushParagraph(); flushList();
      blocks.push(<hr key={`hr-${blocks.length}`} className="my-2 border-violet-100" />);
      continue;
    }

    const bullet = line.match(/^[-*•]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet ?? numbered) {
      flushParagraph();
      listItems.push((bullet?.[1] ?? numbered?.[1] ?? "").trim());
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

  // 改善トリアージ state（Phase 1: planning/shion_improvement_loop_plan.md）
  const [improvementLog, setImprovementLog] = useState<DialogueImprovementLog | null>(null);
  const [triageByKey, setTriageByKey] = useState<Record<string, TriageRecord>>({});
  const [triageOpen, setTriageOpen] = useState(false);
  const [triageSavingKey, setTriageSavingKey] = useState("");

  // Emotion trend state
  const [showTrend, setShowTrend] = useState(false);
  const [trendHistory, setTrendHistory] = useState<EmotionHistoryEntry[]>([]);
  const [trendSummary, setTrendSummary] = useState<EmotionSummary | null>(null);
  const [trendLoading, setTrendLoading] = useState(false);

  // ── 改善トリアージ（Phase 1）────────────────────────────────────────────
  // ルール分類（classifyPmImprovementItems）を初期値とし、User が確定する。
  const triageCandidates = useMemo(() => {
    const items = (improvementLog?.items || []).filter((item) => item?.title || item?.id);
    if (!items.length) return [] as { item: DialogueImprovementItem; rule: TriageDecision }[];
    const pm = classifyPmImprovementItems(items);
    const seen = new Set<string>();
    const rows: { item: DialogueImprovementItem; rule: TriageDecision }[] = [];
    ([
      ["today", pm.today],
      ["later", pm.later],
      ["discard", pm.discard],
    ] as const).forEach(([rule, list]) => {
      list.forEach((item) => {
        const key = String(item.canonical_key || item.id || "");
        if (!key || seen.has(key)) return;
        seen.add(key);
        rows.push({ item, rule });
      });
    });
    return rows;
  }, [improvementLog]);

  const sendTriage = async (
    row: { item: DialogueImprovementItem; rule: TriageDecision },
    decision: TriageDecision,
  ) => {
    const key = String(row.item.canonical_key || row.item.id || "");
    if (!key || triageSavingKey) return;
    setTriageSavingKey(key);
    try {
      const res = await apiClient.post<{ record?: TriageRecord }>("/api/improvement/triage", {
        canonical_key: key,
        item_id: row.item.id || "",
        source_event_id: row.item.source_event_id || "",
        title: readableImprovementTitle(row.item),
        decision,
        rule_decision: row.rule,
        classified_by: "user",
        reason: "",
      });
      const record = res.data?.record || { canonical_key: key, decision };
      setTriageByKey((prev) => ({ ...prev, [key]: record }));
    } catch {
      setError("トリアージの保存に失敗しました。API接続を確認してください。");
    } finally {
      setTriageSavingKey("");
    }
  };

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
    let cancelled = false;

    const loadImprovementContext = () => {
      Promise.allSettled([
        apiClient.get<DialogueImprovementLog>("/api/improvement-log"),
        apiClient.get<DialoguePipelineSummary>("/api/improvement-pipeline/summary"),
        apiClient.get<DialogueGapAnalysis>("/api/lease-system-gaps"),
        apiClient.get<{ records?: TriageRecord[] }>("/api/improvement/triage"),
      ]).then(([improvementResult, pipelineResult, gapsResult, triageResult]) => {
        if (cancelled) return;
        if (improvementResult.status === "fulfilled") {
          setImprovementLog(improvementResult.value.data || null);
        }
        if (triageResult.status === "fulfilled") {
          const map: Record<string, TriageRecord> = {};
          (triageResult.value.data?.records || []).forEach((record) => {
            if (record?.canonical_key) map[record.canonical_key] = record;
          });
          setTriageByKey(map);
        }
        if (improvementResult.status === "fulfilled" && shouldShowDailyImprovementReport()) {
          const report = buildDailyImprovementReport(
            improvementResult.value.data,
            pipelineResult.status === "fulfilled" ? pipelineResult.value.data : null,
            gapsResult.status === "fulfilled" ? gapsResult.value.data : null,
          );
          if (report) {
            const reportMessage: Message = {
              id: Date.now() + 17,
              role: "assistant",
              content: report,
              created_at: new Date().toISOString(),
            };
            setMessages((prev) => {
              const next = [...prev, reportMessage].slice(-DIALOGUE_MAX_DISPLAY_MESSAGES);
              saveLocalDialogueMessages(next);
              return next;
            });
            markDailyImprovementReportShown();
            speakText(report);
          }
        }
      });
    };

    apiClient.get("/api/lease-intelligence/dialogue/state", {
      params: { since: dialogueDisplaySinceIso() },
    })
      .then((stateResult) => {
        if (cancelled) return;
        setState(stateResult.data?.state || {});
        const serverMessages = stateResult.data?.messages || [];
        const localMessages = loadLocalDialogueMessages();
        const nextMessages = mergeDialogueMessages(serverMessages, localMessages);
        setMessages(nextMessages);
        saveLocalDialogueMessages(nextMessages);
        loadImprovementContext();
      })
      .catch(() => {
        if (!cancelled) setError("リース知性体の状態を読み込めませんでした。");
      })
      .finally(() => {
        if (!cancelled) setInitializing(false);
      });

    setVoiceSupported(Boolean(getSpeechRecognition()));

    const key = `lease-intelligence-activity:dialogue:${new Date().toLocaleDateString("sv-SE")}`;
    if (!window.sessionStorage.getItem(key)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "lease_intelligence_dialogue",
        action: "page_view",
        event_id: key,
      }).then(() => window.sessionStorage.setItem(key, "1")).catch(() => {});
    }

    return () => {
      cancelled = true;
    };
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
    const demoMessage: Message = {
      id: now,
      role: "assistant",
      content: DEMO_GREETING,
      created_at: new Date().toISOString(),
    };
    setError("");
    setMessages((prev) => {
      const next = [...prev, demoMessage];
      saveLocalDialogueMessages(next);
      return next;
    });
    speakText(DEMO_GREETING_SPEECH);
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
    const userMessage: Message = {
      id: Date.now(),
      role: "user",
      content: text || "（ファイルを添付しました）",
      created_at: new Date().toISOString(),
      attachedFileName: currentFile?.name,
      attachedFileType: currentFile?.type,
    };
    setMessages((prev) => {
      const next = [...prev, userMessage];
      saveLocalDialogueMessages(next);
      return next;
    });
    setLoading(true);
    try {
      const payload: Record<string, string> = {
        message: text || "添付ファイルの内容を分析してください。",
        since: dialogueDisplaySinceIso(),
      };
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
      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: "assistant",
        content: reply,
        created_at: new Date().toISOString(),
        knowledge_refs: knowledgeRefs?.length ? knowledgeRefs : undefined,
        query: text,
        longInputMode,
      };
      setMessages((prev) => {
        const next = [...prev, assistantMessage];
        saveLocalDialogueMessages(next);
        return next;
      });
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
    clearVisibleDialogueMessages();
    setMessages([]);
    try {
      await apiClient.delete("/api/lease-intelligence/dialogue/history");
    } catch {
      setError("画面履歴は削除しました。サーバー側の履歴削除は後で再試行してください。");
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
                  継続 {Math.floor((Date.now() - new Date('2026-06-12').getTime()) / 86400000) + 1}日
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
              {knowledgeConnectionLabel(state)}
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

          {triageCandidates.length > 0 && (
            <div className="border-b border-violet-100 bg-white px-5 py-2">
              <button
                type="button"
                onClick={() => setTriageOpen((prev) => !prev)}
                className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-bold text-violet-700 transition hover:bg-violet-50"
              >
                <ClipboardList className="h-4 w-4" />
                改善トリアージ（確定 {triageCandidates.filter((row) => triageByKey[String(row.item.canonical_key || row.item.id || "")]).length} / {triageCandidates.length} 件）
                <span className="text-slate-400">{triageOpen ? "▲" : "▼"}</span>
              </button>
              {triageOpen && (
                <div className="mt-2 space-y-1.5 pb-1">
                  <p className="text-[11px] text-slate-400">
                    破線がルール分類の初期値です。ボタンで確定すると記録されます（未確定分は持ち越し・自動昇格なし）。
                  </p>
                  {triageCandidates.map((row) => {
                    const key = String(row.item.canonical_key || row.item.id || "");
                    const confirmed = triageByKey[key]?.decision;
                    return (
                      <div key={key} className="flex items-center justify-between gap-2 rounded-xl border border-slate-100 px-2.5 py-1.5">
                        <span className="min-w-0 truncate text-[12px] text-slate-700" title={readableImprovementTitle(row.item)}>
                          {readableImprovementTitle(row.item)}
                        </span>
                        <div className="flex shrink-0 gap-1">
                          {(["today", "later", "discard"] as TriageDecision[]).map((decision) => {
                            const isConfirmed = confirmed === decision;
                            const isRuleDefault = !confirmed && row.rule === decision;
                            const palette =
                              decision === "today"
                                ? isConfirmed
                                  ? "bg-emerald-600 text-white"
                                  : isRuleDefault
                                    ? "border border-dashed border-emerald-400 bg-emerald-50 text-emerald-700"
                                    : "border border-slate-200 text-slate-500 hover:bg-emerald-50"
                                : decision === "later"
                                  ? isConfirmed
                                    ? "bg-amber-500 text-white"
                                    : isRuleDefault
                                      ? "border border-dashed border-amber-400 bg-amber-50 text-amber-700"
                                      : "border border-slate-200 text-slate-500 hover:bg-amber-50"
                                  : isConfirmed
                                    ? "bg-slate-600 text-white"
                                    : isRuleDefault
                                      ? "border border-dashed border-slate-400 bg-slate-100 text-slate-600"
                                      : "border border-slate-200 text-slate-500 hover:bg-slate-100";
                            const label = decision === "today" ? "今日やる" : decision === "later" ? "後回し" : "捨てる";
                            return (
                              <button
                                key={decision}
                                type="button"
                                disabled={triageSavingKey === key}
                                onClick={() => sendTriage(row, decision)}
                                className={`rounded-lg px-2 py-1 text-[11px] font-bold transition disabled:opacity-40 ${palette}`}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

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
            {messages.map((message) => {
              const codexRequest = message.role === "assistant" ? extractCodexRequest(message.content) : "";
              return (
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
                            <span className="flex min-w-0 items-center gap-1">
                              <RagConfidenceBadge confidence={ref.confidence} level={ref.confidence_level} />
                              <span className="truncate text-[10px] text-slate-400" title={ref.obsidian_ref}>
                                {ref.file_name || ref.obsidian_ref}
                              </span>
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
                  {codexRequest && (
                    <button
                      type="button"
                      onClick={() => copyMessage(message.id + 900000, codexRequest)}
                      className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-cyan-200 bg-cyan-50 px-2.5 py-1.5 text-[11px] font-bold text-cyan-700 transition hover:bg-cyan-100"
                    >
                      {copiedId === message.id + 900000
                        ? <Check className="h-3.5 w-3.5" />
                        : <Copy className="h-3.5 w-3.5" />}
                      Codex依頼文をコピー
                    </button>
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
              );
            })}
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
