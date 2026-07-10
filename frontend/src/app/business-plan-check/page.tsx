"use client";

import React, { useState } from "react";
import { apiClient } from "@/lib/api";
import {
  FileCheck2, Loader2, CheckCircle2, AlertTriangle, XCircle, Info, Sparkles, HelpCircle,
} from "lucide-react";
import { INDUSTRIES } from "@/constants/industries";

// ── 型定義 ─────────────────────────────────────────────────────────────────
type CheckLevel = "ok" | "info" | "watch" | "warning";

interface PlanCheck {
  code: string;
  level: CheckLevel;
  title: string;
  message: string;
}

interface AiReview {
  verdict: string;
  comments: string[];
  questions: string[];
}

interface PlanCheckResult {
  verdict: string;
  summary_level: CheckLevel;
  checks: PlanCheck[];
  metrics: Record<string, number>;
  thresholds_note: string;
  ai_available: boolean;
  ai_review?: AiReview;
}

// ── スタイルヘルパー ─────────────────────────────────────────────────────────
const LEVEL_STYLES: Record<CheckLevel, { badge: string; icon: React.ReactNode }> = {
  ok: {
    badge: "bg-emerald-100 text-emerald-700 border border-emerald-200",
    icon: <CheckCircle2 className="w-4 h-4 text-emerald-600" />,
  },
  info: {
    badge: "bg-sky-100 text-sky-700 border border-sky-200",
    icon: <Info className="w-4 h-4 text-sky-600" />,
  },
  watch: {
    badge: "bg-amber-100 text-amber-700 border border-amber-200",
    icon: <AlertTriangle className="w-4 h-4 text-amber-600" />,
  },
  warning: {
    badge: "bg-rose-100 text-rose-700 border border-rose-200",
    icon: <XCircle className="w-4 h-4 text-rose-600" />,
  },
};

function summaryBg(level: CheckLevel) {
  if (level === "warning") return "from-rose-50 to-red-50 border-rose-200";
  if (level === "watch") return "from-amber-50 to-yellow-50 border-amber-200";
  return "from-emerald-50 to-teal-50 border-emerald-200";
}

// ── メインページ ─────────────────────────────────────────────────────────────
export default function BusinessPlanCheckPage() {
  const [form, setForm] = useState({
    company_name: "",
    industry_major: "製造業",
    nenshu: 0,
    op_margin_pct: 0,
    plan_nenshu: 0,
    plan_op_margin_pct: 0,
    lease_amount: 0,
    lease_months: 60,
    has_conservative_scenario: false,
    plan_basis: "",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PlanCheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value, type } = e.target;
    if (type === "checkbox") {
      setForm(prev => ({ ...prev, [name]: (e.target as HTMLInputElement).checked }));
      return;
    }
    setForm(prev => ({
      ...prev,
      [name]: isNaN(Number(value)) || value === "" ? value : Number(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await apiClient.post("/api/business-plan/validate", form);
      setResult(data);
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } }; message?: string })
          .response?.data?.detail;
      setError(detail || (err as { message?: string }).message || "エラーが発生しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto min-h-[calc(100vh-2rem)]">
      {/* ヘッダー */}
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <FileCheck2 className="w-8 h-8 text-teal-600" />
          事業計画チェック（簡易版）
        </h1>
        <p className="text-slate-500 font-medium mt-2">
          提出された事業計画（売上・利益計画）の妥当性を機械チェック＋AI講評で検証します。
          楽観シナリオのみの計画は評価を下げる方針（FAQ REV-040）に基づきます。
        </p>
      </div>

      {/* 入力フォーム */}
      <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 mb-8">
        <h2 className="text-lg font-black text-slate-700 mb-5">案件・計画情報</h2>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">企業名</label>
            <input
              name="company_name" type="text" value={form.company_name} onChange={handleChange}
              className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-400"
              placeholder="（任意）"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1">業種</label>
            <select
              name="industry_major" value={form.industry_major} onChange={handleChange}
              className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-400"
            >
              {INDUSTRIES.map(ind => <option key={ind}>{ind}</option>)}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          {[
            { name: "nenshu", label: "直近売上高（百万円）" },
            { name: "op_margin_pct", label: "直近営業利益率（%）" },
            { name: "plan_nenshu", label: "計画売上高（百万円）" },
            { name: "plan_op_margin_pct", label: "計画営業利益率（%）" },
            { name: "lease_amount", label: "リース金額（百万円）" },
            { name: "lease_months", label: "リース期間（回）" },
          ].map(f => (
            <div key={f.name}>
              <label className="block text-xs font-bold text-slate-500 mb-1">{f.label}</label>
              <input
                name={f.name} type="text" inputMode="decimal"
                value={String(form[f.name as keyof typeof form])} onChange={handleChange}
                className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-400"
              />
            </div>
          ))}
        </div>

        <div className="mb-4">
          <label className="block text-xs font-bold text-slate-500 mb-1">計画の根拠（担当者メモ・任意）</label>
          <textarea
            name="plan_basis" value={form.plan_basis} onChange={handleChange} rows={2}
            className="w-full border rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400 resize-none"
            placeholder="例: 大口受注の内示あり。新設備で生産能力1.5倍。"
          />
        </div>

        <label className="flex items-center gap-2 mb-5 text-sm font-medium text-slate-700 cursor-pointer">
          <input
            name="has_conservative_scenario" type="checkbox"
            checked={form.has_conservative_scenario} onChange={handleChange}
            className="w-4 h-4 rounded border-slate-300 text-teal-600 focus:ring-teal-400"
          />
          保守シナリオ（売上下振れ時の計画）が提示されている
        </label>

        <button
          type="submit" disabled={loading}
          className="w-full py-3 rounded-xl bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white font-black text-base flex items-center justify-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              検証中...
            </>
          ) : (
            <>
              <FileCheck2 className="w-5 h-5" />
              事業計画を検証
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
        <div className="space-y-5 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* 総合判定 */}
          <div className={`rounded-2xl border-2 bg-gradient-to-br ${summaryBg(result.summary_level)} p-6`}>
            <div className="flex items-center gap-3">
              {LEVEL_STYLES[result.summary_level].icon}
              <h3 className="text-xl font-black text-slate-800">総合判定</h3>
              <span className={`ml-auto inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-base font-black ${LEVEL_STYLES[result.summary_level].badge}`}>
                {result.verdict}
              </span>
            </div>
            <p className="mt-3 text-xs text-slate-500">{result.thresholds_note}</p>
          </div>

          {/* チェック項目 */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <h3 className="text-lg font-black text-slate-700 mb-4">チェック項目</h3>
            <ul className="space-y-3">
              {result.checks.map((c, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="mt-0.5 flex-shrink-0">{LEVEL_STYLES[c.level].icon}</span>
                  <div className="min-w-0">
                    <p className="text-sm font-bold text-slate-800">{c.title}</p>
                    <p className="text-sm text-slate-600 leading-relaxed">{c.message}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* AI講評 */}
          {result.ai_review && (
            <div className="rounded-2xl border border-indigo-200 bg-indigo-50 p-5 text-indigo-900">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-5 h-5 text-indigo-600" />
                <h3 className="font-black">AI講評</h3>
                {result.ai_review.verdict && (
                  <span className="rounded-full border border-indigo-300 bg-white/60 px-2 py-0.5 text-xs font-bold">
                    {result.ai_review.verdict}
                  </span>
                )}
              </div>
              {result.ai_review.comments.length > 0 && (
                <ul className="space-y-1 mb-3">
                  {result.ai_review.comments.map((c, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm">
                      <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-400" />
                      {c}
                    </li>
                  ))}
                </ul>
              )}
              {result.ai_review.questions.length > 0 && (
                <div className="bg-white/70 rounded-xl p-3">
                  <p className="text-xs font-black text-indigo-500 mb-2 flex items-center gap-1">
                    <HelpCircle className="w-3.5 h-3.5" />
                    顧客への確認質問
                  </p>
                  <ul className="space-y-1">
                    {result.ai_review.questions.map((q, i) => (
                      <li key={i} className="text-sm">
                        <span className="font-black text-indigo-500">{i + 1}.</span> {q}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {!result.ai_available && (
            <p className="text-xs text-slate-400">
              ※ AI講評は現在利用できないため、機械チェックのみの結果です。
            </p>
          )}
        </div>
      )}
    </div>
  );
}
