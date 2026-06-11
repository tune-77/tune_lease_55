"use client";

import React, { useState } from "react";
import { apiClient } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";
import {
  Zap, Calculator, TrendingUp, RefreshCw, ChevronDown,
  CheckCircle2, AlertTriangle
} from "lucide-react";

const ASSET_OPTIONS = [
  { id: "medical",      label: "医療機器" },
  { id: "it",           label: "IT機器・サーバー" },
  { id: "pc",           label: "PC・端末" },
  { id: "vehicle",      label: "車両・トラック" },
  { id: "machinery",    label: "工作機械" },
  { id: "construction", label: "建設機械" },
  { id: "solar",        label: "太陽光・省エネ設備" },
  { id: "other",        label: "その他" },
];

const GRADE_OPTIONS = [
  { id: "① 1-3先", label: "① 1-3先" },
  { id: "② 4-6先", label: "② 4-6先" },
  { id: "③ 要注意先", label: "③ 要注意先" },
  { id: "④ 無格付先", label: "④ 無格付先" },
];

const TERM_OPTIONS = [
  { months: 24, label: "2年" },
  { months: 36, label: "3年" },
  { months: 48, label: "4年" },
  { months: 60, label: "5年" },
  { months: 72, label: "6年" },
  { months: 84, label: "7年" },
];

type RateResult = {
  year_month: string;
  proposed_rate: number;
  breakdown: {
    base_rate: number;
    asset_spread: number;
    grade_spread: number;
    risk_adjustment: number;
  };
  monthly_payment: number;
  total_payment: number;
  total_interest: number;
  term_months: number;
  lease_amount: number;
  sensitivity: { score: number; rate: number; is_current: boolean }[];
};

const fmt = (n: number) => n.toLocaleString("ja-JP");

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { value: number; color: string }[]; label?: string }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg p-3 text-xs">
      <p className="font-black text-slate-700 mb-1">スコア {label}pt</p>
      <p className="font-bold" style={{ color: payload[0].color }}>提案金利: {payload[0].value.toFixed(2)}%</p>
    </div>
  );
};

export default function RateEnginePage() {
  const [score, setScore] = useState(70);
  const [termMonths, setTermMonths] = useState(60);
  const [assetId, setAssetId] = useState("other");
  const [grade, setGrade] = useState("② 4-6先");
  const [amount, setAmount] = useState("10000000");
  const [result, setResult] = useState<RateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.post<RateResult>("/api/rate-engine/propose", {
        score,
        term_months: termMonths,
        asset_id: assetId,
        grade,
        lease_amount: parseFloat(amount.replace(/,/g, "")) || 10000000,
      });
      setResult(res.data);
    } catch {
      setError("計算に失敗しました。APIサーバーを確認してください。");
    } finally {
      setLoading(false);
    }
  };

  const rateColor = (rate: number) => {
    if (rate <= 2.5) return "text-emerald-600";
    if (rate <= 3.5) return "text-amber-600";
    return "text-rose-600";
  };

  const bd = result?.breakdown;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Zap className="text-amber-500" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">動的金利提案エンジン</h1>
          <p className="text-sm text-slate-500">借手スコア・物件種別・期間から最適なリース金利を自動計算します。</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 入力パネル */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-5">
          <h2 className="text-sm font-black text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
            <Calculator size={14} /> 入力パラメータ
          </h2>

          {/* 借手スコア */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-xs font-black text-slate-500">借手スコア</label>
              <span className={`text-lg font-black ${score >= 70 ? "text-emerald-600" : score >= 50 ? "text-amber-600" : "text-rose-600"}`}>
                {score}pt
              </span>
            </div>
            <input
              type="range" min={0} max={100} step={1}
              value={score}
              onChange={e => setScore(Number(e.target.value))}
              className="w-full accent-indigo-600"
            />
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0</span><span>50</span><span>70</span><span>90</span><span>100</span>
            </div>
          </div>

          {/* 物件種別 */}
          <div>
            <label className="block text-xs font-black text-slate-500 mb-1.5">物件種別</label>
            <div className="relative">
              <select
                value={assetId}
                onChange={e => setAssetId(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-700 outline-none appearance-none pr-8"
              >
                {ASSET_OPTIONS.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
            </div>
          </div>

          {/* 格付 */}
          <div>
            <label className="block text-xs font-black text-slate-500 mb-1.5">格付</label>
            <div className="grid grid-cols-2 gap-1.5">
              {GRADE_OPTIONS.map(g => (
                <button
                  key={g.id}
                  onClick={() => setGrade(g.id)}
                  className={`py-2 px-2 rounded-xl text-xs font-bold border-2 transition-all text-left
                    ${grade === g.id ? "bg-indigo-50 border-indigo-500 text-indigo-700" : "bg-white border-slate-200 text-slate-500 hover:border-slate-300"}`}
                >
                  {g.label}
                </button>
              ))}
            </div>
          </div>

          {/* リース期間 */}
          <div>
            <label className="block text-xs font-black text-slate-500 mb-1.5">リース期間</label>
            <div className="grid grid-cols-3 gap-1.5">
              {TERM_OPTIONS.map(t => (
                <button
                  key={t.months}
                  onClick={() => setTermMonths(t.months)}
                  className={`py-2 rounded-xl text-xs font-bold border-2 transition-all
                    ${termMonths === t.months ? "bg-indigo-50 border-indigo-500 text-indigo-700" : "bg-white border-slate-200 text-slate-500 hover:border-slate-300"}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* リース金額 */}
          <div>
            <label className="block text-xs font-black text-slate-500 mb-1.5">リース金額（円）</label>
            <input
              type="text" inputMode="numeric"
              value={amount}
              onChange={e => setAmount(e.target.value.replace(/[^0-9]/g, ""))}
              placeholder="例: 10000000"
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-indigo-500/20"
            />
            {amount && (
              <p className="text-[10px] text-slate-400 mt-1">{fmt(Number(amount))}円</p>
            )}
          </div>

          <button
            onClick={calculate}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-amber-500 hover:bg-amber-400 text-white font-black text-sm transition-all disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <Zap size={16} />}
            金利を計算する
          </button>

          {error && <p className="text-xs text-rose-600 font-bold text-center">{error}</p>}
        </div>

        {/* 結果パネル */}
        <div className="lg:col-span-2 space-y-4">
          {!result && !loading && (
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-12 flex flex-col items-center justify-center text-slate-300 min-h-[300px]">
              <Zap size={48} className="mb-4" />
              <p className="font-bold text-sm">左のパラメータを入力して計算してください</p>
            </div>
          )}

          {result && (
            <>
              {/* メイン結果 */}
              <div className={`bg-white border rounded-2xl shadow-sm p-6 ${result.proposed_rate >= 4.0 ? "border-rose-200" : result.proposed_rate >= 3.0 ? "border-amber-200" : "border-emerald-200"}`}>
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-xs text-slate-500 font-bold mb-1">提案金利（{result.year_month}）</p>
                    <p className={`text-5xl font-black ${rateColor(result.proposed_rate)}`}>
                      {result.proposed_rate.toFixed(2)}
                      <span className="text-xl ml-1">%</span>
                    </p>
                  </div>
                  <div className="text-right">
                    {result.proposed_rate <= 3.0 ? (
                      <div className="flex items-center gap-1.5 text-emerald-600 font-black text-sm">
                        <CheckCircle2 size={16} /> 競争力あり
                      </div>
                    ) : result.proposed_rate <= 4.0 ? (
                      <div className="flex items-center gap-1.5 text-amber-600 font-black text-sm">
                        <AlertTriangle size={16} /> 標準レンジ
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 text-rose-600 font-black text-sm">
                        <AlertTriangle size={16} /> 高リスク案件
                      </div>
                    )}
                  </div>
                </div>

                {/* 内訳 */}
                <div className="grid grid-cols-4 gap-3 pt-4 border-t border-slate-100">
                  {[
                    { label: "基準金利", value: bd?.base_rate, color: "text-slate-700" },
                    { label: "物件スプレッド", value: bd?.asset_spread, color: "text-blue-600" },
                    { label: "格付スプレッド", value: bd?.grade_spread, color: "text-indigo-600" },
                    { label: "リスク補正", value: bd?.risk_adjustment, color: bd?.risk_adjustment !== undefined && bd.risk_adjustment < 0 ? "text-emerald-600" : "text-rose-600" },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="text-center">
                      <p className="text-[10px] text-slate-400 mb-0.5">{label}</p>
                      <p className={`text-sm font-black ${color}`}>
                        {value !== undefined ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}%` : "—"}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* 支払いシミュレーション */}
              <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
                <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                  <Calculator size={13} /> 支払いシミュレーション
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: "月次リース料", value: `¥${fmt(result.monthly_payment)}`, sub: `${result.term_months}ヶ月` },
                    { label: "総支払額", value: `¥${fmt(result.total_payment)}`, sub: "元利合計" },
                    { label: "リース料総額（金利分）", value: `¥${fmt(result.total_interest)}`, sub: `${((result.total_interest / result.lease_amount) * 100).toFixed(1)}%` },
                  ].map(({ label, value, sub }) => (
                    <div key={label} className="bg-slate-50 rounded-xl p-3 text-center">
                      <p className="text-[10px] text-slate-400 mb-1">{label}</p>
                      <p className="text-base font-black text-slate-800">{value}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* スコア感度分析グラフ */}
              {result.sensitivity.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
                  <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                    <TrendingUp size={13} /> スコア感度分析
                  </h3>
                  <p className="text-[11px] text-slate-400 mb-3">スコアが変わった場合の提案金利推移</p>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={result.sensitivity} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="score" tick={{ fontSize: 11, fill: "#94a3b8" }} unit="pt" />
                      <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "#94a3b8" }} unit="%" />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine x={score} stroke="#6366f1" strokeDasharray="4 4" label={{ value: "現在", fontSize: 10, fill: "#6366f1" }} />
                      <Line
                        type="monotone" dataKey="rate"
                        stroke="#f59e0b" strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-[10px] text-slate-400 mt-2 text-center">
                    スコアを10pt改善すると金利は約0.15〜0.20%低下します
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
