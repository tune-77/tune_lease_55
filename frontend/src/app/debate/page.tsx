"use client";

import React, { useState, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api";
import {
  Brain, Orbit, Crown, ChevronDown, ChevronUp,
  Loader2, CheckCircle2, XCircle, AlertTriangle, Info, Clock, BookMarked, PenLine, Users,
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
}

function getUserInfo(key: DemoUserKey) {
  return DEMO_USERS.find(u => u.key === key) ?? null;
}

function agentLabel(
  role: keyof Participants,
  parts: Participants | null,
  roleLabel: string,
) {
  if (!parts) return `紫苑（${roleLabel}）`;
  const info = getUserInfo(parts[role]);
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
interface DebateResult {
  score: number;
  mode: "solo" | "debate";
  cautious?: CautiousResult;
  aggressive?: AggressiveResult;
  arbiter: ArbiterResult;
  debate_log?: string;
  same_opinion_r1?: boolean;
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

function DebateLog({ log, sameR1 }: { log: string; sameR1?: boolean }) {
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
  });
  const [submittedParticipants, setSubmittedParticipants] = useState<Participants | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DebateResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [obsidianSaving, setObsidianSaving] = useState(false);
  const [obsidianToast, setObsidianToast] = useState<"success" | "error" | null>(null);
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
  const sessionIdRef = useRef<string>(
    typeof crypto !== "undefined" ? crypto.randomUUID() : Math.random().toString(36).slice(2)
  );
  const historyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
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
      const { data } = await apiClient.post("/api/multi-agent-screening", payload);
      setSubmittedParticipants(capturedParticipants);
      setResult(data);
      setHumanDecision(data.arbiter.final);
      setJudgmentChangeReason("");
      // 討論完了後に履歴を再取得
      if (form.company_name?.trim()) {
        apiClient.get(`/api/conversation-history?company_name=${encodeURIComponent(form.company_name.trim())}&limit=5`)
          .then(({ data: h }) => { if (h.count > 0) setHistory(h); })
          .catch(() => {});
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "エラーが発生しました");
    } finally {
      setLoading(false);
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
            <div className="grid md:grid-cols-3 gap-3 pt-1">
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
        <div className="grid md:grid-cols-3 gap-4">
          {([
            { role: "skeptic"  as const, label: "懐疑派",       border: "border-violet-200", bg: "bg-violet-50",  text: "text-violet-700" },
            { role: "optimist" as const, label: "楽観派",       border: "border-teal-200",   bg: "bg-teal-50",    text: "text-teal-700" },
            { role: "arbiter"  as const, label: "統合派（裁定）", border: "border-amber-200",  bg: "bg-amber-50",   text: "text-amber-700" },
          ]).map(({ role, label, border, bg, text }) => {
            const info = getUserInfo(participants[role]);
            return (
              <div key={role} className={`rounded-xl border-2 ${border} ${bg} p-4`}>
                <p className={`text-xs font-black ${text} mb-2`}>紫苑（{label}）</p>
                <select
                  value={participants[role]}
                  onChange={(e) =>
                    setParticipants(prev => ({ ...prev, [role]: e.target.value as DemoUserKey }))
                  }
                  className="w-full border border-slate-300 rounded-lg px-2 py-1.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-violet-400"
                >
                  <option value="">— 未選択（デフォルト） —</option>
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
              {form.score >= 60 ? "✓ 承認圏 → 軍師単独" : form.score <= 40 ? "✗ 否決圏 → 軍師単独" : "⚡ 境界域 → 討論モード"}
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
                  <p className="text-xs text-slate-500">スコア {result.score}点 — 境界外のため{agentLabel("arbiter", submittedParticipants, "統合派")}が単独処理</p>
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
                  subtitle={getUserInfo(submittedParticipants?.skeptic ?? "")?.dept}
                  icon={<Brain className="w-5 h-5 text-violet-600" />}
                  color="border-violet-200 bg-violet-50/50"
                  opinion={result.cautious.opinion}
                  reasons={result.cautious.reasons}
                  extras={result.cautious.key_risks}
                  extraLabel="重大リスク"
                />
                <AgentCard
                  name={agentLabel("optimist", submittedParticipants, "楽観派")}
                  subtitle={getUserInfo(submittedParticipants?.optimist ?? "")?.dept}
                  icon={<Brain className="w-5 h-5 text-teal-600" />}
                  color="border-teal-200 bg-teal-50/50"
                  opinion={result.aggressive.opinion}
                  reasons={result.aggressive.reasons}
                  extras={result.aggressive.opportunities}
                  extraLabel="見逃せない機会"
                />
              </div>
            </div>
          )}

          {/* 軍師裁定 */}
          <ArbiterPanel
            arbiter={result.arbiter}
            agentName={agentLabel("arbiter", submittedParticipants, "統合派")}
          />

          {/* 討論ログ */}
          {result.debate_log && (
            <DebateLog log={result.debate_log} sameR1={result.same_opinion_r1} />
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
