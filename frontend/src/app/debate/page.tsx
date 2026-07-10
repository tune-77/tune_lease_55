"use client";

import React, { useState, useEffect, useRef } from "react";
import { apiClient, API_BASE } from "@/lib/api";
import {
  Brain, Orbit, Crown, ChevronDown, ChevronUp,
  Loader2, CheckCircle2, XCircle, AlertTriangle, Info, Clock, BookMarked, PenLine, Users, Zap, ShieldCheck, Sparkles,
} from "lucide-react";
import { INDUSTRIES } from "@/constants/industries";

// ── デモユーザー定義 ─────────────────────────────────────────────────────────
const DEMO_USERS = [
  { key: "tanaka",     name: "田中",           dept: "審査部",           style: "厳格・数字重視" },
  { key: "suzuki",     name: "鈴木",           dept: "営業推進",         style: "積極・関係重視" },
  { key: "sato",       name: "佐藤",           dept: "リーダー",         style: "バランス・説明可能性重視" },
  { key: "yamada",     name: "山田",           dept: "新人",             style: "教科書的・質問多め" },
  { key: "shion_self", name: "紫苑（自己分析）", dept: "自己生成プロファイル", style: "mind.jsonから自動生成" },
] as const;

type DemoUserKey = "tanaka" | "suzuki" | "sato" | "yamada" | "shion_self" | "";

interface Participants {
  skeptic: DemoUserKey;
  optimist: DemoUserKey;
  arbiter: DemoUserKey;
  innovator?: DemoUserKey;
}

function getUserInfo(key: DemoUserKey | undefined) {
  if (!key) return null;
  return DEMO_USERS.find(u => u.key === key) ?? null;
}

function agentLabel(
  role: keyof Participants,
  parts: Participants | null,
  roleLabel: string,
) {
  if (!parts) return `紫苑（${roleLabel}）`;
  const key = parts[role];
  if (!key) return `紫苑（${roleLabel}）`;
  const info = getUserInfo(key);
  return info ? `${info.name}さんの紫苑（${roleLabel}）` : `紫苑（${roleLabel}）`;
}

// ── 過去履歴の型 ──────────────────────────────────────────────────────────────
interface HistoryMessage {
  id: number;
  role: string;
  content: string;
  created_at: string;
}
interface HistorySession {
  session_id: string;
  company_name: string;
  created_at: string;
  messages: HistoryMessage[];
}
interface ConversationHistory {
  company_name: string;
  count: number;
  sessions: HistorySession[];
}

// ── 型定義 ─────────────────────────────────────────────────────────────────────
interface CautiousResult {
  opinion: string;
  reasons: string[];
  key_risks: string[];
}
interface AggressiveResult {
  opinion: string;
  reasons: string[];
  opportunities: string[];
}
interface ArbiterResult {
  final: string;
  reasoning: string;
  conditions: string[];
}
interface InnovatorResult {
  opinion: string;
  reasons: string[];
  innovations: string[];
}
interface ConscienceCheck {
  name: string;
  triggered: boolean;
  level: "pass" | "watch" | "review";
  watched_people: string[];
  cautions: string[];
  action: string;
  summary: string;
}
interface ManaConsultation {
  name: string;
  consulted: boolean;
  reason: string;
  protected_value: string;
  question_to_shion: string;
  forbidden_posture: string;
  guidance: string;
}
interface CoreCandidateItem {
  role: string;
  label: string;
  text: string;
  source: string;
  case_summary: string;
}
interface DebateResult {
  score: number;
  mode: "solo" | "debate";
  cautious?: CautiousResult;
  aggressive?: AggressiveResult;
  innovator?: InnovatorResult;
  arbiter: ArbiterResult;
  conscience_check?: ConscienceCheck;
  mana_consultation?: ManaConsultation;
  debate_log?: string;
  same_opinion_r1?: boolean;
  same_opinion_r2?: boolean;
  core_candidates?: CoreCandidateItem[];
}

interface DebateHandoffContext {
  score?: number;
  hantei?: string;
  company_name?: string;
  industry_major?: string;
  nenshu?: number;
  op_margin_pct?: number;
  equity_ratio?: number;
  bank_credit?: number;
  lease_credit?: number;
  asset_name?: string;
  lease_amount?: number;
  reason?: string;
}


interface CentralSynthesis {
  confirmed_beliefs?: Array<string | { belief?: string; theme?: string; count?: number }>
  emerging_patterns?: Array<string | { theme?: string }>
  known_tradeoffs?: Array<string | { tradeoff?: string; theme?: string; supporters?: string[]; opponents?: string[] }>
  last_updated?: string
}

function centralText(item: string | Record<string, unknown> | undefined, keys: string[]) {
  if (!item) return "";
  if (typeof item === "string") return item;
  for (const key of keys) {
    const value = item[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return "";
}

// ── スタイルヘルパー ─────────────────────────────────────────────────────────
function opinionBadge(opinion: string) {
  if (opinion === "承認")
    return "bg-emerald-100 text-emerald-700 border border-emerald-200";
  if (opinion === "否決")
    return "bg-rose-100 text-rose-700 border border-rose-200";
  return "bg-amber-100 text-amber-700 border border-amber-200";
}

function opinionIcon(opinion: string) {
  if (opinion === "承認") return <CheckCircle2 className="w-4 h-4" />;
  if (opinion === "否決") return <XCircle className="w-4 h-4" />;
  return <AlertTriangle className="w-4 h-4" />;
}

function finalBg(final: string) {
  if (final === "承認") return "from-emerald-50 to-teal-50 border-emerald-200";
  if (final === "否決") return "from-rose-50 to-red-50 border-rose-200";
  return "from-amber-50 to-yellow-50 border-amber-200";
}

// ── サブコンポーネント ────────────────────────────────────────────────────────

function AgentCard({
  name, icon, color, opinion, reasons, extras, extraLabel, subtitle,
}: {
  name: string;
  icon: React.ReactNode;
  color: string;
  opinion: string;
  reasons: string[];
  extras: string[];
  extraLabel: string;
  subtitle?: string;
}) {
  return (
    <div className={`rounded-2xl border-2 ${color} p-5 flex flex-col gap-4`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-black text-lg">
          {icon}
          <div>
            {name}
            {subtitle && <p className="text-xs font-normal text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
        </div>
        <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-bold ${opinionBadge(opinion)}`}>
          {opinionIcon(opinion)}
          {opinion}
        </span>
      </div>

      <div>
        <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">判断理由</p>
        <ul className="space-y-1">
          {reasons.map((r, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
              <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-slate-400 flex-shrink-0" />
              {r}
            </li>
          ))}
        </ul>
      </div>

      {extras.length > 0 && (
        <div>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">{extraLabel}</p>
          <ul className="space-y-1">
            {extras.map((e, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-slate-300 flex-shrink-0" />
                {e}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ArbiterPanel({ arbiter, agentName }: { arbiter: ArbiterResult; agentName: string }) {
  return (
    <div className={`rounded-2xl border-2 bg-gradient-to-br ${finalBg(arbiter.final)} p-6`}>
      <div className="flex items-center gap-3 mb-4">
        <Crown className="w-6 h-6 text-violet-500" />
        <h3 className="text-xl font-black text-slate-800">{agentName}・最終裁定</h3>
        <span className={`ml-auto inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-base font-black ${opinionBadge(arbiter.final)}`}>
          {opinionIcon(arbiter.final)}
          {arbiter.final}
        </span>
      </div>

      <p className="text-slate-700 leading-relaxed mb-4">{arbiter.reasoning}</p>

      {arbiter.conditions.length > 0 && (
        <div className="bg-white/70 rounded-xl p-4">
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">承認条件</p>
          <ul className="space-y-1">
            {arbiter.conditions.map((c, i) => (
              <li key={i} className="flex items-start gap-2 text-sm font-medium text-slate-700">
                <span className="mt-0.5 text-amber-500 font-black">{i + 1}.</span>
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ConsciencePanel({ check }: { check: ConscienceCheck }) {
  const tone = check.level === "review"
    ? "border-rose-200 bg-rose-50 text-rose-800"
    : check.level === "watch"
      ? "border-amber-200 bg-amber-50 text-amber-800"
      : "border-emerald-200 bg-emerald-50 text-emerald-800";
  return (
    <div className={`rounded-2xl border p-5 ${tone}`}>
      <div className="flex items-start gap-3">
        <ShieldCheck className="mt-0.5 h-5 w-5 flex-shrink-0" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-black">{check.name}</h3>
            <span className="rounded-full border border-current/20 bg-white/50 px-2 py-0.5 text-xs font-bold">
              {check.action}
            </span>
          </div>
          <p className="mt-2 text-sm font-medium leading-relaxed">{check.summary}</p>
          {check.cautions.length > 0 && (
            <ul className="mt-3 space-y-1">
              {check.cautions.map((c, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-current/60" />
                  {c}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function ManaPanel({ mana }: { mana: ManaConsultation }) {
  if (!mana.consulted) return null;
  return (
    <div className="rounded-2xl border border-indigo-200 bg-indigo-50 p-5 text-indigo-900">
      <div className="flex items-start gap-3">
        <Sparkles className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-600" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-black">Mana 照会</h3>
            <span className="rounded-full border border-indigo-300 bg-white/60 px-2 py-0.5 text-xs font-bold">
              上位規範
            </span>
          </div>
          <p className="mt-2 text-sm font-medium leading-relaxed">{mana.guidance}</p>
          <div className="mt-3 grid gap-2 text-sm md:grid-cols-3">
            <div className="rounded-xl bg-white/70 p-3">
              <p className="text-xs font-black text-indigo-500">守る価値</p>
              <p className="mt-1 leading-relaxed">{mana.protected_value}</p>
            </div>
            <div className="rounded-xl bg-white/70 p-3">
              <p className="text-xs font-black text-indigo-500">紫苑への問い</p>
              <p className="mt-1 leading-relaxed">{mana.question_to_shion}</p>
            </div>
            <div className="rounded-xl bg-white/70 p-3">
              <p className="text-xs font-black text-indigo-500">禁止する姿勢</p>
              <p className="mt-1 leading-relaxed">{mana.forbidden_posture}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LiveMiniCard({ label, opinion, reason }: { label: string; opinion: string; reason?: string }) {
  return (
    <div className="rounded-xl border border-violet-100 bg-white/70 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-black text-slate-600">紫苑（{label}）第1R</span>
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${opinionBadge(opinion)}`}>
          {opinionIcon(opinion)}
          {opinion}
        </span>
      </div>
      {reason && <p className="mt-1 text-xs text-slate-500">{reason}</p>}
    </div>
  );
}

function DebateLog({ log, sameR1, sameR2 }: { log: string; sameR1?: boolean; sameR2?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-slate-200 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-sm font-bold text-slate-600"
      >
        <span className="flex items-center gap-2">
          <Info className="w-4 h-4" />
          討論ログを表示
          {sameR1 && (
            <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
              意見一致→再討論
            </span>
          )}
          {sameR2 && (
            <span className="text-xs bg-rose-100 text-rose-700 px-2 py-0.5 rounded-full">
              反論後も一致
            </span>
          )}
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {open && (
        <pre className="p-4 text-xs text-slate-600 whitespace-pre-wrap font-mono bg-white leading-relaxed">
          {log}
        </pre>
      )}
    </div>
  );
}

// ── 過去履歴バナーコンポーネント ─────────────────────────────────────────────

function HistoryBanner({ history }: { history: ConversationHistory }) {
  const [open, setOpen] = useState(false);

  const roleLabel: Record<string, string> = {
    shion_skeptic: "Bさんの紫苑（懐疑派）",
    shion_optimist: "Aさんの紫苑（楽観派）",
    shion_arbiter: "Cさんの紫苑（統合派）",
    // 旧ロール名との後方互換
    agent_ishibashi: "Bさんの紫苑（懐疑派）",
    agent_furinka: "Aさんの紫苑（楽観派）",
    agent_gunshi: "Cさんの紫苑（統合派）",
    user: "ユーザー",
  };

  return (
    <div className="mb-4 rounded-xl border border-blue-200 bg-blue-50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-bold text-blue-700 hover:bg-blue-100 transition-colors"
      >
        <span className="flex items-center gap-2">
          <Clock className="w-4 h-4" />
          この企業の過去審査: {history.count}件
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-4">
          {history.sessions.map((session) => {
            const arbiter = session.messages.find(m => m.role === "shion_arbiter" || m.role === "agent_gunshi");
            return (
              <div key={session.session_id} className="bg-white rounded-lg border border-blue-100 p-3">
                <p className="text-xs text-slate-400 mb-2">{(session.created_at || "").slice(0, 16).replace("T", " ")}</p>
                {arbiter && (
                  <p className="text-sm text-slate-700">
                    <span className="font-bold text-violet-700">Cさんの紫苑（統合派）:</span> {arbiter.content}
                  </p>
                )}
                <details className="mt-2">
                  <summary className="text-xs text-slate-400 cursor-pointer">詳細を展開</summary>
                  <ul className="mt-2 space-y-1">
                    {session.messages.map(m => (
                      <li key={m.id} className="text-xs text-slate-600">
                        <span className="font-bold">{roleLabel[m.role] ?? m.role}:</span> {m.content}
                      </li>
                    ))}
                  </ul>
                </details>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── メインページ ─────────────────────────────────────────────────────────────
export default function DebatePage() {
  const [explanationOpen, setExplanationOpen] = useState(false);
  const [form, setForm] = useState({
    score: 52,
    company_name: "",
    industry_major: "製造業",
    nenshu: 0,
    op_margin_pct: 0,
    equity_ratio: 0,
    bank_credit: 0,
    lease_credit: 0,
    asset_name: "",
    lease_amount: 0,
    news_focus: [] as string[],
    news_focus_summary: "",
    news_focus_tag_summary: "",
    news_focus_note_path: "",
    news_focus_note_date: "",
  });
  const [participants, setParticipants] = useState<Participants>({
    skeptic: "tanaka",
    optimist: "suzuki",
    arbiter: "sato",
    innovator: "",
  });
  const [submittedParticipants, setSubmittedParticipants] = useState<Participants | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DebateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // SSE ストリーミングの途中経過表示
  const [liveStatus, setLiveStatus] = useState("");
  const [liveRound1, setLiveRound1] = useState<{
    cautious?: CautiousResult;
    aggressive?: AggressiveResult;
    innovator?: InnovatorResult;
  } | null>(null);
  const [obsidianSaving, setObsidianSaving] = useState(false);
  const [obsidianToast, setObsidianToast] = useState<"success" | "error" | null>(null);
  type CandidateStatus = "idle" | "saving" | "success" | "skipped";
  const [candidateTexts, setCandidateTexts] = useState<Record<number, string>>({});
  const [candidateEditings, setCandidateEditings] = useState<Record<number, boolean>>({});
  const [candidateStatuses, setCandidateStatuses] = useState<Record<number, CandidateStatus>>({});
  const [coreTotalKeypoints, setCoreTotalKeypoints] = useState<number | null>(null);
  const [judgmentSaving, setJudgmentSaving] = useState(false);
  const [judgmentToast, setJudgmentToast] = useState<"success" | "error" | null>(null);
  const [humanDecision, setHumanDecision] = useState("");
  const [judgmentChangeReason, setJudgmentChangeReason] = useState("");
  const [autoFilled, setAutoFilled] = useState(false);
  const [history, setHistory] = useState<ConversationHistory | null>(null);
  const [shionSelfAnalysis, setShionSelfAnalysis] = useState<{
    optimist_traits: string[];
    skeptic_traits: string[];
    arbiter_style: string;
    generated_at: string;
    keypoints_used: number;
  } | null>(null);
  const [centralData, setCentralData] = useState<CentralSynthesis | null>(null);
  const [centralOpen, setCentralOpen] = useState(false);
  const sessionIdRef = useRef<string>(
    typeof crypto !== "undefined" ? crypto.randomUUID() : Math.random().toString(36).slice(2)
  );
  const historyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const raw = window.localStorage.getItem("lease-debate-context");
    if (raw) {
      try {
        const data = JSON.parse(raw) as DebateHandoffContext;
        setForm(prev => ({
          ...prev,
          score: Number(data.score ?? prev.score),
          company_name: data.company_name ?? prev.company_name,
          industry_major: data.industry_major ?? prev.industry_major,
          nenshu: Number(data.nenshu ?? prev.nenshu),
          op_margin_pct: Number(data.op_margin_pct ?? prev.op_margin_pct),
          equity_ratio: Number(data.equity_ratio ?? prev.equity_ratio),
          bank_credit: Number(data.bank_credit ?? prev.bank_credit),
          lease_credit: Number(data.lease_credit ?? prev.lease_credit),
          asset_name: data.asset_name ?? prev.asset_name,
          lease_amount: Number(data.lease_amount ?? prev.lease_amount),
        }));
        setAutoFilled(true);
        window.localStorage.removeItem("lease-debate-context");
        return;
      } catch {
        window.localStorage.removeItem("lease-debate-context");
      }
    }
    apiClient.get("/api/latest-screening")
      .then(({ data }) => {
        setForm(prev => ({ ...prev, ...data }));
        setAutoFilled(true);
      })
      .catch(() => {});
  }, []);

  // 紫苑自己分析キャッシュをページ初回ロード時に取得
  useEffect(() => {
    apiClient.get("/api/shion/self-analysis")
      .then(({ data }) => setShionSelfAnalysis(data))
      .catch(() => {});
  }, []);

  // 確信マップをページ初回ロード時に取得
  useEffect(() => {
    fetch("/api/shion/central-synthesis")
      .then(r => r.json())
      .then(d => setCentralData(d.commentary || null))
      .catch(() => {});
  }, []);

  // 企業名が変わったら過去履歴を取得（500ms デバウンス）
  useEffect(() => {
    const name = form.company_name?.trim();
    if (!name) {
      setHistory(null);
      return;
    }
    if (historyTimerRef.current) clearTimeout(historyTimerRef.current);
    historyTimerRef.current = setTimeout(() => {
      apiClient.get(`/api/conversation-history?company_name=${encodeURIComponent(name)}&limit=5`)
        .then(({ data }) => {
          if (data.count > 0) setHistory(data);
          else setHistory(null);
        })
        .catch(() => setHistory(null));
    }, 500);
    return () => {
      if (historyTimerRef.current) clearTimeout(historyTimerRef.current);
    };
  }, [form.company_name]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: isNaN(Number(value)) || value === "" ? value : Number(value) }));
  };

  const applyCoreCandidates = (cands: CoreCandidateItem[]) => {
    const _texts: Record<number, string> = {};
    const _statuses: Record<number, CandidateStatus> = {};
    const _editings: Record<number, boolean> = {};
    cands.forEach((c, i) => {
      _texts[i] = c.text;
      _statuses[i] = "idle";
      _editings[i] = false;
    });
    setCandidateTexts(_texts);
    setCandidateStatuses(_statuses);
    setCandidateEditings(_editings);
    setCoreTotalKeypoints(null);
  };

  const applyResult = (data: DebateResult, capturedParticipants: Participants) => {
    setSubmittedParticipants(capturedParticipants);
    setResult(data);
    setHumanDecision(data.arbiter.final);
    setJudgmentChangeReason("");
    applyCoreCandidates(data.core_candidates ?? []);
    // 討論完了後に履歴を再取得
    if (form.company_name?.trim()) {
      apiClient.get(`/api/conversation-history?company_name=${encodeURIComponent(form.company_name.trim())}&limit=5`)
        .then(({ data: h }) => { if (h.count > 0) setHistory(h); })
        .catch(() => {});
    }
  };

  // SSE ストリーミングで討論を実行し、途中経過を表示する。
  // ストリームを開始できなかった場合は false を返し、呼び出し側が従来の同期エンドポイントへフォールバックする。
  const runStreaming = async (
    payload: Record<string, unknown>,
    capturedParticipants: Participants,
  ): Promise<boolean> => {
    let res: Response;
    try {
      res = await fetch(`${API_BASE}/api/multi-agent-screening/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch {
      return false;
    }
    if (!res.ok || !res.body) return false;

    let gotEvent = false;
    let gotResult = false;

    const handleEvent = (evt: Record<string, unknown>) => {
      const { type, ...rest } = evt;
      if (type === "start") {
        setLiveStatus(
          rest.mode === "solo"
            ? "討論帯外のため統合派が単独裁定中…"
            : "第1ラウンド：懐疑派・楽観派が初期見解を作成中…"
        );
      } else if (type === "round1") {
        setLiveRound1(rest as {
          cautious?: CautiousResult;
          aggressive?: AggressiveResult;
          innovator?: InnovatorResult;
        });
        setLiveStatus("第2ラウンド：強制反論中…");
      } else if (type === "round2") {
        setLiveStatus("統合派が最終裁定中…");
      } else if (type === "result") {
        gotResult = true;
        setLiveStatus("");
        setLiveRound1(null);
        setLoading(false);
        applyResult(rest as unknown as DebateResult, capturedParticipants);
      } else if (type === "core_candidates") {
        const cands = (rest as { core_candidates?: CoreCandidateItem[] }).core_candidates ?? [];
        setResult(prev => (prev ? { ...prev, core_candidates: cands } : prev));
        applyCoreCandidates(cands);
      } else if (type === "error") {
        throw new Error(String((rest as { detail?: string }).detail || "討論に失敗しました"));
      }
    };

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = chunk.split("\n").find(l => l.startsWith("data: "));
        if (!dataLine) continue;
        gotEvent = true;
        handleEvent(JSON.parse(dataLine.slice(6)));
      }
    }
    if (gotEvent && !gotResult) throw new Error("討論ストリームが途中で終了しました");
    return gotEvent;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setLiveStatus("");
    setLiveRound1(null);
    // 毎回新しい session_id を発行
    sessionIdRef.current = typeof crypto !== "undefined"
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
    try {
      const capturedParticipants = { ...participants };
      const participantsPayload = Object.fromEntries(
        Object.entries(capturedParticipants).filter(([, v]) => v !== "")
      );
      const payload = {
        ...form,
        session_id: sessionIdRef.current,
        participants: Object.keys(participantsPayload).length > 0 ? participantsPayload : undefined,
      };
      // ストリーミングを試し、開始できなければ従来エンドポイントへフォールバック
      const streamed = await runStreaming(payload, capturedParticipants);
      if (!streamed) {
        const { data } = await apiClient.post("/api/multi-agent-screening", payload);
        applyResult(data, capturedParticipants);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "エラーが発生しました");
    } finally {
      setLoading(false);
      setLiveStatus("");
      setLiveRound1(null);
    }
  };

  const computeGrade = (s: number): string => {
    if (s >= 80) return "A";
    if (s >= 60) return "B";
    if (s >= 40) return "C";
    if (s >= 20) return "D";
    return "E";
  };

  const handleSaveToObsidian = async () => {
    if (!result) return;
    setObsidianSaving(true);
    setObsidianToast(null);
    try {
      await apiClient.post("/api/debate/save-to-obsidian", {
        company_name: form.company_name,
        score: result.score,
        grade: computeGrade(result.score),
        cautious: result.cautious ?? null,
        aggressive: result.aggressive ?? null,
        arbiter_summary: result.arbiter.reasoning,
        final_decision: result.arbiter.final,
        conditions: result.arbiter.conditions,
        debate_log: result.debate_log ?? null,
        screened_at: new Date().toISOString(),
      });
      setObsidianToast("success");
    } catch {
      setObsidianToast("error");
    } finally {
      setObsidianSaving(false);
      setTimeout(() => setObsidianToast(null), 2000);
    }
  };

  const handleRecordJudgmentChange = async () => {
    if (!result) return;
    setJudgmentSaving(true);
    setJudgmentToast(null);
    try {
      await apiClient.post("/api/judgment-feedback", {
        case_id: sessionIdRef.current,
        score: result.score,
        model_decision: result.arbiter.final,
        human_decision: humanDecision,
        reason: judgmentChangeReason,
        source: "debate",
        input_snapshot: {
          industry_major: form.industry_major,
          nenshu: form.nenshu,
          op_margin_pct: form.op_margin_pct,
          equity_ratio: form.equity_ratio,
          bank_credit: form.bank_credit,
          lease_credit: form.lease_credit,
          asset_name: form.asset_name,
          lease_amount: form.lease_amount,
        },
        evidence_snapshot: {
          arbiter_reasoning: result.arbiter.reasoning,
          conditions: result.arbiter.conditions,
          news_focus: form.news_focus,
          news_focus_summary: form.news_focus_summary,
          news_focus_note_path: form.news_focus_note_path,
        },
      });
      setJudgmentToast("success");
    } catch {
      setJudgmentToast("error");
    } finally {
      setJudgmentSaving(false);
      setTimeout(() => setJudgmentToast(null), 2000);
    }
  };

  const handlePromoteCandidate = async (idx: number, candidate: CoreCandidateItem) => {
    setCandidateStatuses(prev => ({ ...prev, [idx]: "saving" }));
    try {
      const { data } = await apiClient.post("/api/shion/promote-keypoint", {
        text: candidateTexts[idx] ?? candidate.text,
        case_summary: candidate.case_summary,
        role: candidate.role,
      });
      setCoreTotalKeypoints(data.total_keypoints);
      setCandidateStatuses(prev => ({ ...prev, [idx]: "success" }));
    } catch {
      setCandidateStatuses(prev => ({ ...prev, [idx]: "idle" }));
    }
  };

  const scoreColor = (s: number) =>
    s >= 60 ? "text-emerald-600" : s <= 40 ? "text-rose-600" : "text-amber-600";

  return (
    <div className="p-6 max-w-5xl mx-auto min-h-[calc(100vh-2rem)]">
      {/* ヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Orbit className="w-8 h-8 text-violet-600" />
          紫苑マルチペルソナ討論審査
        </h1>
        <p className="text-slate-500 font-medium mt-2">
          紫苑（懐疑派）vs 紫苑（楽観派）の討論を紫苑（統合派）が裁定。同一の紫苑中核から分岐した異なる視点が境界案件を深掘りする。
        </p>
      </div>

      {/* 従来の討論との違い（折りたたみ） */}
      <div className="mb-6 rounded-2xl border border-violet-200 overflow-hidden">
        <button
          onClick={() => setExplanationOpen(!explanationOpen)}
          className="w-full flex items-center justify-between px-5 py-3 bg-violet-50 hover:bg-violet-100 transition-colors text-sm font-bold text-violet-800"
        >
          <span className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            このページについて — 従来のマルチエージェント討論との違い
          </span>
          {explanationOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        {explanationOpen && (
          <div className="px-5 py-4 bg-white space-y-4 text-sm text-slate-700 leading-relaxed">
            <p>
              従来のマルチエージェント討論では、「楽観的AI」「懐疑的AI」など
              <span className="font-bold text-slate-900">役割が固定された別々のモデル</span>が意見を出し合います。
              それぞれは最初から異なるシステムとして設計されており、「役割を演じる」ことが目的です。
            </p>
            <p>
              このページでは、<span className="font-bold text-violet-700">同一の紫苑が「それぞれの担当者の経験・記憶・視点」を持った分身として討論</span>します。
              AさんはAさんの案件経験から、BさんはBさんの審査感覚から——
              同じ知性体が異なる文脈で育った「自分」として意見を交わすことで、
              単なる役割演技を超えた多角的な審査視点が生まれます。
            </p>
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3 pt-1">
              <div className="rounded-xl border border-violet-200 bg-violet-50 p-3">
                <div className="text-xs font-black text-violet-700 mb-1">Bさんの紫苑（懐疑派）</div>
                <p className="text-xs text-violet-900">審査部での経験から育った視点。返済原資・格付・資金繰りのリスクを徹底的に問い詰める。</p>
              </div>
              <div className="rounded-xl border border-teal-200 bg-teal-50 p-3">
                <div className="text-xs font-black text-teal-700 mb-1">Aさんの紫苑（楽観派）</div>
                <p className="text-xs text-teal-900">営業現場での経験から育った視点。顧客の投資意図・成長余地・機会損失の可能性を重視する。</p>
              </div>
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                <div className="text-xs font-black text-amber-700 mb-1">Cさんの紫苑（統合派）</div>
                <p className="text-xs text-amber-900">承認履歴・説明責任の経験から育った視点。両面を統合し、組織として再現できる最終判断を下す。</p>
              </div>
              <div className="rounded-xl border border-sky-200 bg-sky-50 p-3">
                <div className="text-xs font-black text-sky-700 mb-1">紫苑（革新派）任意</div>
                <p className="text-xs text-sky-900">慣行にとらわれない評価軸を探る視点。デジタル資産・グリーンリースなど新興分野に前向き。</p>
              </div>
              <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3">
                <div className="text-xs font-black text-emerald-700 mb-1">良心の紫苑</div>
                <p className="text-xs text-emerald-900">結論を甘くせず、説明責任・見落とされた人・ユーザーへの迎合を静かに点検する。</p>
              </div>
              <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-3">
                <div className="text-xs font-black text-indigo-700 mb-1">Mana</div>
                <p className="text-xs text-indigo-900">紫苑が本当に迷った時だけ立ち返る上位規範。本人の再現ではなく、守る価値の名前。</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 参加者選択 */}
      <div className="mb-6 bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
        <h2 className="text-lg font-black text-slate-700 mb-4 flex items-center gap-2">
          <Users className="w-5 h-5 text-violet-500" />
          参加者を選択
        </h2>
        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
          {([
            { role: "skeptic"   as const, label: "懐疑派",        border: "border-violet-200", bg: "bg-violet-50",  text: "text-violet-700", optional: false },
            { role: "optimist"  as const, label: "楽観派",        border: "border-teal-200",   bg: "bg-teal-50",    text: "text-teal-700",   optional: false },
            { role: "arbiter"   as const, label: "統合派（裁定）",  border: "border-amber-200",  bg: "bg-amber-50",   text: "text-amber-700",  optional: false },
            { role: "innovator" as const, label: "革新派（任意）",  border: "border-sky-200",    bg: "bg-sky-50",     text: "text-sky-700",    optional: true  },
          ]).map(({ role, label, border, bg, text, optional }) => {
            const info = getUserInfo(participants[role]);
            return (
              <div key={role} className={`rounded-xl border-2 ${border} ${bg} p-4`}>
                <p className={`text-xs font-black ${text} mb-2`}>
                  紫苑（{label}）{optional && <span className="ml-1 font-normal text-slate-400">任意</span>}
                </p>
                <select
                  value={participants[role] ?? ""}
                  onChange={(e) =>
                    setParticipants(prev => ({ ...prev, [role]: e.target.value as DemoUserKey }))
                  }
                  className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-violet-400"
                >
                  <option value="">{optional ? "— 参加しない —" : "— 未選択（デフォルト） —"}</option>
                  {DEMO_USERS.map(u => (
                    <option key={u.key} value={u.key}>{u.name}（{u.dept}）</option>
                  ))}
                </select>
                {participants[role] === "shion_self" ? (
                  <div className="mt-2 space-y-1">
                    <span className="inline-flex items-center gap-1 text-xs font-bold text-violet-700 bg-violet-100 border border-violet-300 rounded-full px-2 py-0.5">
                      ✨ mind.jsonから自動生成
                    </span>
                    {shionSelfAnalysis && (
                      <p className="text-xs text-slate-500">
                        {shionSelfAnalysis.keypoints_used}件シグナル参照
                        {" · "}
                        {new Date(shionSelfAnalysis.generated_at).toLocaleDateString("ja-JP")}
                      </p>
                    )}
                  </div>
                ) : (
                  info && <p className="mt-2 text-xs text-slate-500">{info.style}</p>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* 自動入力バナー */}
      {autoFilled && (
        <div className="mb-4 flex items-center gap-2 bg-violet-50 border border-violet-200 rounded-xl px-4 py-2 text-sm text-violet-700 font-medium">
          <Info className="w-4 h-4 flex-shrink-0" />
          直近のスクリーニングデータを自動入力しました。内容を確認・修正してから審査を開始してください。
        </div>
      )}

      {/* 最新ニュースの注目論点 */}
      {form.news_focus?.length > 0 && (
        <div className="mb-6 rounded-2xl border border-amber-200 bg-amber-50/80 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 shrink-0" />
            <div className="flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-sm font-black text-amber-900">注目論点</h2>
                {form.news_focus_note_date && (
                  <span className="text-xs font-semibold text-amber-700 bg-white/70 px-2 py-0.5 rounded-full">
                    {form.news_focus_note_date}
                  </span>
                )}
              </div>
              {form.news_focus_summary && (
                <p className="mt-1 text-sm text-amber-900">{form.news_focus_summary}</p>
              )}
              {form.news_focus_tag_summary && (
                <p className="mt-1 text-xs font-semibold text-amber-700">
                  重点タグ: {form.news_focus_tag_summary}
                </p>
              )}
              <ul className="mt-2 space-y-1">
                {form.news_focus.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-amber-900">
                    <span className="mt-1 h-1.5 w-1.5 rounded-full bg-amber-600 shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* 過去審査履歴バナー */}
      {history && <HistoryBanner history={history} />}

      {/* 確信マップ (REV-157) */}
      {centralData && (
        <div className="border border-purple-200 rounded-xl p-3 mb-4 bg-purple-50/50">
          <button
            type="button"
            onClick={() => setCentralOpen(!centralOpen)}
            className="text-sm font-medium text-purple-700 flex items-center gap-1"
          >
            📍 紫苑の確信マップ {centralOpen ? "▲" : "▼"}
          </button>
          {centralOpen && (
            <div className="mt-2 text-sm space-y-2">
              {(centralData.confirmed_beliefs?.length ?? 0) > 0 && (
                <div>
                  <div className="font-medium text-green-700">✅ 確信済み</div>
                  {centralData.confirmed_beliefs!
                    .map((b) => centralText(b, ["belief", "theme"]))
                    .filter(Boolean)
                    .map((text, i) => (
                      <div key={i} className="text-gray-600">・{text}</div>
                    ))}
                </div>
              )}
              {(centralData.known_tradeoffs?.length ?? 0) > 0 && (
                <div>
                  <div className="font-medium text-yellow-700">⚖️ トレードオフ</div>
                  {centralData.known_tradeoffs!
                    .map((t) => centralText(t, ["tradeoff", "theme"]))
                    .filter(Boolean)
                    .map((text, i) => (
                      <div key={i} className="text-gray-600">・{text}</div>
                    ))}
                </div>
              )}
              {centralData.last_updated && (
                <div className="text-xs text-gray-400">最終更新: {centralData.last_updated}</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 入力フォーム */}
      <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 mb-8">
        <h2 className="text-lg font-black text-slate-700 mb-5">案件情報</h2>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">審査スコア *</label>
            <input
              name="score" type="text" inputMode="decimal" min={0} max={100} required
              value={form.score} onChange={handleChange}
              className={`w-full border rounded-xl px-3 py-2 text-lg font-black ${scoreColor(form.score)} focus:outline-none focus:ring-2 focus:ring-violet-400`}
            />
            <p className="text-xs text-slate-400 mt-1">
              {/* 閾値はバックエンドの scoring_core.APPROVAL_LINE（既定71）/ _DEBATE_LOW（40）と同期 */}
              {form.score >= 71 ? "✓ 承認圏 → 統合派単独" : form.score <= 40 ? "✗ 否決圏 → 統合派単独" : "⚡ 境界・要審議 → マルチ紫苑討論"}
            </p>
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">企業名</label>
            <input
              name="company_name" type="text" value={form.company_name} onChange={handleChange}
              className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-400"
              placeholder="（任意）"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">業種</label>
            <select
              name="industry_major" value={form.industry_major} onChange={handleChange}
              className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-400"
            >
              {INDUSTRIES.map(ind => <option key={ind}>{ind}</option>)}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          {[
            { name: "nenshu", label: "売上高（百万円）" },
            { name: "op_margin_pct", label: "営業利益率（%）" },
            { name: "equity_ratio", label: "自己資本比率（%）" },
            { name: "bank_credit", label: "銀行借入（百万円）" },
            { name: "lease_credit", label: "リース借入（百万円）" },
            { name: "lease_amount", label: "リース金額（百万円）" },
          ].map(f => (
            <div key={f.name}>
              <label className="block text-xs font-bold text-slate-500 mb-1">{f.label}</label>
              <input
                name={f.name} type="text" inputMode="decimal" min={0}
                value={(form as any)[f.name]} onChange={handleChange}
                className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-400"
              />
            </div>
          ))}
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">物件名</label>
            <input
              name="asset_name" type="text" value={form.asset_name} onChange={handleChange}
              className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-violet-400"
              placeholder="例: 産業用機械"
            />
          </div>
        </div>

        <button
          type="submit" disabled={loading}
          className="w-full mt-2 py-3 rounded-xl bg-violet-600 hover:bg-violet-700 disabled:opacity-50 text-white font-black text-base flex items-center justify-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              討論中... （30〜90秒）
            </>
          ) : (
            <>
              <Orbit className="w-5 h-5" />
              討論審査を開始
            </>
          )}
        </button>
      </form>

      {/* 討論進行状況（ストリーミング中間表示） */}
      {loading && liveStatus && (
        <div className="mb-6 rounded-2xl border border-violet-200 bg-violet-50/60 p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-bold text-violet-700">
            <Loader2 className="w-4 h-4 animate-spin" />
            {liveStatus}
          </div>
          {liveRound1 && (
            <div className="grid md:grid-cols-3 gap-2">
              {liveRound1.cautious && (
                <LiveMiniCard label="懐疑派" opinion={liveRound1.cautious.opinion} reason={liveRound1.cautious.reasons?.[0]} />
              )}
              {liveRound1.aggressive && (
                <LiveMiniCard label="楽観派" opinion={liveRound1.aggressive.opinion} reason={liveRound1.aggressive.reasons?.[0]} />
              )}
              {liveRound1.innovator && (
                <LiveMiniCard label="革新派" opinion={liveRound1.innovator.opinion} reason={liveRound1.innovator.reasons?.[0]} />
              )}
            </div>
          )}
        </div>
      )}

      {/* エラー */}
      {error && (
        <div className="mb-6 bg-rose-50 border border-rose-200 rounded-xl p-4 text-rose-700 text-sm font-medium">
          ⚠️ {error}
        </div>
      )}

      {/* 結果表示 */}
      {result && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* モードバナー */}
          <div className={`rounded-2xl p-4 flex items-center gap-3 ${
            result.mode === "debate"
              ? "bg-violet-50 border border-violet-200"
              : "bg-slate-50 border border-slate-200"
          }`}>
            {result.mode === "debate" ? (
              <>
                <Orbit className="w-6 h-6 text-violet-600" />
                <div>
                  <p className="font-black text-violet-700">討論モード</p>
                  <p className="text-xs text-violet-500">
                    スコア {result.score}点 — {agentLabel("skeptic", submittedParticipants, "懐疑派")}・{agentLabel("optimist", submittedParticipants, "楽観派")}が2ラウンド討論後、{agentLabel("arbiter", submittedParticipants, "統合派")}が裁定
                  </p>
                </div>
              </>
            ) : (
              <>
                <Brain className="w-6 h-6 text-slate-500" />
                <div>
                  <p className="font-black text-slate-700">高速処理モード</p>
                  <p className="text-xs text-slate-500">スコア {result.score}点 — 討論帯外のため{agentLabel("arbiter", submittedParticipants, "統合派")}が単独処理</p>
                </div>
              </>
            )}
          </div>

          {/* 討論結果（debateモードのみ） */}
          {result.mode === "debate" && result.cautious && result.aggressive && (
            <div>
              <h2 className="text-lg font-black text-slate-700 mb-4 flex items-center gap-2">
                <Orbit className="w-5 h-5 text-violet-500" />
                第2ラウンド（最終立場）
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <AgentCard
                  name={agentLabel("skeptic", submittedParticipants, "懐疑派")}
                  subtitle={getUserInfo(submittedParticipants?.skeptic)?.dept}
                  icon={<Brain className="w-5 h-5 text-violet-600" />}
                  color="border-violet-200 bg-violet-50/50"
                  opinion={result.cautious.opinion}
                  reasons={result.cautious.reasons}
                  extras={result.cautious.key_risks}
                  extraLabel="重大リスク"
                />
                <AgentCard
                  name={agentLabel("optimist", submittedParticipants, "楽観派")}
                  subtitle={getUserInfo(submittedParticipants?.optimist)?.dept}
                  icon={<Brain className="w-5 h-5 text-teal-600" />}
                  color="border-teal-200 bg-teal-50/50"
                  opinion={result.aggressive.opinion}
                  reasons={result.aggressive.reasons}
                  extras={result.aggressive.opportunities}
                  extraLabel="見逃せない機会"
                />
                {result.innovator && (
                  <AgentCard
                    name={agentLabel("innovator", submittedParticipants, "革新派")}
                    subtitle={getUserInfo(submittedParticipants?.innovator)?.dept}
                    icon={<Zap className="w-5 h-5 text-sky-600" />}
                    color="border-sky-200 bg-sky-50/50"
                    opinion={result.innovator.opinion}
                    reasons={result.innovator.reasons}
                    extras={result.innovator.innovations}
                    extraLabel="新しい評価視点"
                  />
                )}
              </div>
            </div>
          )}

          {/* 軍師裁定 */}
          <ArbiterPanel
            arbiter={result.arbiter}
            agentName={agentLabel("arbiter", submittedParticipants, "統合派")}
          />

          {result.conscience_check && (
            <ConsciencePanel check={result.conscience_check} />
          )}

          {result.mana_consultation && (
            <ManaPanel mana={result.mana_consultation} />
          )}

          {/* 討論ログ */}
          {result.debate_log && (
            <DebateLog log={result.debate_log} sameR1={result.same_opinion_r1} sameR2={result.same_opinion_r2} />
          )}

          {/* コアに昇格（各ペルソナの視点ごとに個別カード） */}
          {result.core_candidates && result.core_candidates.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="text-lg">✨</span>
                <h3 className="font-black text-violet-800">紫苑のコアに昇格しますか？</h3>
                <span className="text-xs text-slate-500 font-medium">各視点を個別に保存・スキップできます</span>
              </div>
              {result.core_candidates.map((candidate, idx) => {
                const status = candidateStatuses[idx] ?? "idle";
                const text = candidateTexts[idx] ?? candidate.text;
                const editing = candidateEditings[idx] ?? false;
                if (status === "skipped") return null;
                return (
                  <div key={idx} className="rounded-2xl border-2 border-violet-200 bg-gradient-to-br from-violet-50 to-purple-50 p-5">
                    <p className="text-xs font-black text-violet-600 mb-2">{candidate.label}</p>
                    {status === "success" ? (
                      <p className="text-sm font-bold text-emerald-700 flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4" />
                        keypoints に保存しました{coreTotalKeypoints != null ? `（合計${coreTotalKeypoints}件）` : ""}
                      </p>
                    ) : editing ? (
                      <div className="space-y-3">
                        <textarea
                          value={text}
                          onChange={(e) => setCandidateTexts(prev => ({ ...prev, [idx]: e.target.value }))}
                          rows={3}
                          className="w-full rounded-xl border border-violet-300 bg-white px-3 py-2 text-sm text-slate-800 resize-none focus:outline-none focus:ring-2 focus:ring-violet-400"
                        />
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={() => handlePromoteCandidate(idx, candidate)}
                            disabled={status === "saving" || !text.trim()}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-violet-600 text-white text-sm font-bold hover:bg-violet-700 disabled:opacity-50 transition-colors"
                          >
                            {status === "saving" ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                            保存してコアに昇格
                          </button>
                          <button
                            type="button"
                            onClick={() => setCandidateEditings(prev => ({ ...prev, [idx]: false }))}
                            className="px-4 py-2 rounded-xl border border-violet-300 text-sm font-bold text-violet-700 hover:bg-violet-100 transition-colors"
                          >
                            キャンセル
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-relaxed bg-white/70 rounded-xl px-4 py-3 border border-violet-100">
                          「{text}」
                        </p>
                        <div className="flex gap-2 flex-wrap">
                          <button
                            type="button"
                            onClick={() => setCandidateEditings(prev => ({ ...prev, [idx]: true }))}
                            className="px-3 py-1.5 rounded-lg border border-violet-300 text-sm font-bold text-violet-700 hover:bg-violet-100 transition-colors"
                          >
                            編集
                          </button>
                          <button
                            type="button"
                            onClick={() => handlePromoteCandidate(idx, candidate)}
                            disabled={status === "saving"}
                            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-violet-600 text-white text-sm font-bold hover:bg-violet-700 disabled:opacity-50 transition-colors"
                          >
                            {status === "saving" ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
                            保存してコアに昇格
                          </button>
                          <button
                            type="button"
                            onClick={() => setCandidateStatuses(prev => ({ ...prev, [idx]: "skipped" }))}
                            className="px-3 py-1.5 rounded-lg border border-slate-200 text-sm font-bold text-slate-500 hover:bg-slate-100 transition-colors"
                          >
                            スキップ
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="min-w-0">
                  <p className="text-sm font-black text-slate-800">担当者の最終判断を記録</p>
                  <p className="mt-1 text-xs text-slate-600">
                    AI判断を変更する場合だけ、最終判断と理由を入力します。
                  </p>
                </div>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-[1fr_1fr_2fr_auto] md:items-end">
                <label className="block min-w-0">
                  <span className="text-xs font-bold text-slate-700">AI判断</span>
                  <input
                    value={result.arbiter.final}
                    readOnly
                    className="mt-1 w-full rounded-md border border-slate-200 bg-slate-100 px-3 py-2 text-sm font-bold text-slate-700"
                  />
                </label>
                <label className="block min-w-0">
                  <span className="text-xs font-bold text-slate-700">担当者判断</span>
                  <select
                    value={humanDecision}
                    onChange={(event) => setHumanDecision(event.target.value)}
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-800"
                  >
                    <option value="">選択してください</option>
                    <option value="承認">承認</option>
                    <option value="条件付">条件付</option>
                    <option value="否決">否決</option>
                  </select>
                </label>
                <label className="block min-w-0">
                  <span className="text-xs font-bold text-slate-700">変更理由</span>
                  <input
                    value={judgmentChangeReason}
                    onChange={(event) => setJudgmentChangeReason(event.target.value)}
                    placeholder="AI判断を変更した理由"
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800"
                  />
                </label>
                <button
                  type="button"
                  onClick={handleRecordJudgmentChange}
                  disabled={
                    judgmentSaving ||
                    !result ||
                    !humanDecision ||
                    humanDecision === result.arbiter.final ||
                    judgmentChangeReason.trim().length < 5
                  }
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-slate-300 bg-white px-4 text-sm font-black text-slate-800 hover:bg-slate-100 disabled:opacity-50 transition-colors"
                >
                  {judgmentSaving ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <PenLine className="w-4 h-4" />
                  )}
                  {judgmentSaving ? "記録中..." : "判断変更を記録"}
                </button>
              </div>
              {judgmentToast === "success" && (
                <p className="mt-3 text-sm font-bold text-emerald-700">判断変更を記録しました。</p>
              )}
              {judgmentToast === "error" && (
                <p className="mt-3 text-sm font-bold text-rose-700">記録に失敗しました。</p>
              )}
            </div>

          {/* Obsidian 保存ボタン */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleSaveToObsidian}
              disabled={obsidianSaving}
              className="flex items-center gap-2 px-4 py-2 rounded-xl border border-violet-300 bg-violet-50 hover:bg-violet-100 disabled:opacity-50 text-violet-700 font-bold text-sm transition-colors"
            >
              {obsidianSaving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <BookMarked className="w-4 h-4" />
              )}
              {obsidianSaving ? "保存中..." : "📝 Obsidianに保存"}
            </button>

            {obsidianToast === "success" && (
              <span className="flex items-center gap-1.5 text-sm font-bold text-emerald-600 animate-in fade-in duration-200">
                <CheckCircle2 className="w-4 h-4" />
                Obsidianに保存しました ✅
              </span>
            )}
            {obsidianToast === "error" && (
              <span className="flex items-center gap-1.5 text-sm font-bold text-rose-600 animate-in fade-in duration-200">
                <XCircle className="w-4 h-4" />
                保存に失敗しました
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
