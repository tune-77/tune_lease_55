"use client";

import React, { useState } from "react";
import axios from "axios";
import { apiClient } from "@/lib/api";
import {
  Swords, Shield, Zap, Crown, ChevronDown, ChevronUp,
  Loader2, CheckCircle2, XCircle, AlertTriangle, Info,
} from "lucide-react";

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

const INDUSTRIES = [
  "製造業", "建設業", "卸売業", "小売業", "運輸業", "情報通信業",
  "不動産業", "医療・福祉", "サービス業", "飲食業", "農業・漁業",
  "金融・保険業", "教育・学習支援業", "宿泊業", "その他",
];

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
  name, icon, color, opinion, reasons, extras, extraLabel,
}: {
  name: string;
  icon: React.ReactNode;
  color: string;
  opinion: string;
  reasons: string[];
  extras: string[];
  extraLabel: string;
}) {
  return (
    <div className={`rounded-2xl border-2 ${color} p-5 flex flex-col gap-4`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-black text-lg">
          {icon}
          {name}
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

function ArbiterPanel({ arbiter }: { arbiter: ArbiterResult }) {
  return (
    <div className={`rounded-2xl border-2 bg-gradient-to-br ${finalBg(arbiter.final)} p-6`}>
      <div className="flex items-center gap-3 mb-4">
        <Crown className="w-6 h-6 text-yellow-500" />
        <h3 className="text-xl font-black text-slate-800">軍師・最終裁定</h3>
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

// ── メインページ ─────────────────────────────────────────────────────────────
export default function DebatePage() {
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
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DebateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: isNaN(Number(value)) || value === "" ? value : Number(value) }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await apiClient.post("/api/multi-agent-screening", form);
      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "エラーが発生しました");
    } finally {
      setLoading(false);
    }
  };

  const scoreColor = (s: number) =>
    s >= 60 ? "text-emerald-600" : s <= 40 ? "text-rose-600" : "text-amber-600";

  return (
    <div className="p-6 max-w-5xl mx-auto min-h-[calc(100vh-2rem)]">
      {/* ヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Swords className="w-8 h-8 text-violet-600" />
          マルチエージェント討論審査
        </h1>
        <p className="text-slate-500 font-medium mt-2">
          石橋（慎重派）vs 風林火山（積極派）の討論を軍師が裁定。スコア40〜60の境界案件で自動起動。
        </p>
      </div>

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
              <Swords className="w-5 h-5" />
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
                <Swords className="w-6 h-6 text-violet-600" />
                <div>
                  <p className="font-black text-violet-700">討論モード</p>
                  <p className="text-xs text-violet-500">スコア {result.score}点 — 石橋・風林火山が2ラウンド討論後、軍師が裁定</p>
                </div>
              </>
            ) : (
              <>
                <Zap className="w-6 h-6 text-slate-500" />
                <div>
                  <p className="font-black text-slate-700">高速処理モード</p>
                  <p className="text-xs text-slate-500">スコア {result.score}点 — 境界外のため軍師が単独処理</p>
                </div>
              </>
            )}
          </div>

          {/* 討論結果（debateモードのみ） */}
          {result.mode === "debate" && result.cautious && result.aggressive && (
            <div>
              <h2 className="text-lg font-black text-slate-700 mb-4 flex items-center gap-2">
                <Swords className="w-5 h-5 text-violet-500" />
                第2ラウンド（最終立場）
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <AgentCard
                  name="石橋（慎重派）"
                  icon={<Shield className="w-5 h-5 text-blue-600" />}
                  color="border-blue-200 bg-blue-50/50"
                  opinion={result.cautious.opinion}
                  reasons={result.cautious.reasons}
                  extras={result.cautious.key_risks}
                  extraLabel="重大リスク"
                />
                <AgentCard
                  name="風林火山（積極派）"
                  icon={<Zap className="w-5 h-5 text-orange-500" />}
                  color="border-orange-200 bg-orange-50/50"
                  opinion={result.aggressive.opinion}
                  reasons={result.aggressive.reasons}
                  extras={result.aggressive.opportunities}
                  extraLabel="見逃せない機会"
                />
              </div>
            </div>
          )}

          {/* 軍師裁定 */}
          <ArbiterPanel arbiter={result.arbiter} />

          {/* 討論ログ */}
          {result.debate_log && (
            <DebateLog log={result.debate_log} sameR1={result.same_opinion_r1} />
          )}
        </div>
      )}
    </div>
  );
}
