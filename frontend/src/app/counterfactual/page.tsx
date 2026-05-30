"use client";

import React, { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";
import {
  Lightbulb, RefreshCw, ChevronDown, Loader2, AlertCircle,
  TrendingUp, CheckCircle2, ArrowRight, BarChart2
} from "lucide-react";
import { apiClient } from "@/lib/api";

type CaseRow = {
  id: string;
  timestamp: string;
  company_name: string | null;
  score: number | null;
  final_status: string;
};

type Counterfactual = {
  param: string;
  label: string;
  current_display: string;
  required_display: string;
  change_pct: number | null;
  achieved_score: number;
  difficulty: "易" | "中" | "難";
  note: string;
};

type SensPoint = { pct_change?: number; op_margin?: number; eq_ratio?: number; score: number };

type CFResult = {
  case_id: string;
  current_score: number;
  target_score: number;
  gap: number;
  current_metrics: {
    op_margin: number;
    eq_ratio: number;
    nenshu: number;
    op_profit: number;
    net_assets: number;
    total_assets: number;
    grade: string;
  };
  counterfactuals: Counterfactual[];
  op_sensitivity: SensPoint[];
  eq_sensitivity: SensPoint[];
};

const DIFF_COLOR = { 易: "emerald", 中: "amber", 難: "rose" } as const;

const ChartTooltip = ({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow p-2.5 text-xs">
      <p className="text-slate-500 mb-1">{label}</p>
      <p className="font-black text-slate-800">スコア: {payload[0].value}pt</p>
    </div>
  );
};

export default function CounterfactualPage() {
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [loadingCases, setLoadingCases] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<CFResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [targetScore, setTargetScore] = useState(70);

  const fetchCases = useCallback(async () => {
    setLoadingCases(true);
    try {
      const res = await apiClient.get("/api/cases?limit=60&sort=desc");
      setCases(res.data);
    } catch {
      /* ignore */
    } finally {
      setLoadingCases(false);
    }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const analyze = async () => {
    if (!selectedId) return;
    setAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const res = await axios.post<CFResult>("/api/counterfactual/analyze", {
        case_id: selectedId,
        target_score: targetScore,
      });
      setResult(res.data);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg || "分析に失敗しました");
    } finally {
      setAnalyzing(false);
    }
  };

  const selectedCase = cases.find(c => c.id === selectedId);
  const scoreGap = result ? result.gap : 0;
  const alreadyApproved = result && result.current_score >= result.target_score;

  const scoreColor = (s: number) =>
    s >= 70 ? "text-emerald-600" : s >= 60 ? "text-amber-600" : "text-rose-600";

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Lightbulb className="text-yellow-500" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Counterfactual 分析</h1>
          <p className="text-sm text-slate-500">「あとどれだけ改善すれば承認された？」を自動計算します。</p>
        </div>
      </div>

      {/* 入力パネル */}
      <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          {/* 案件選択 */}
          <div className="md:col-span-2">
            <label className="block text-xs font-black text-slate-500 uppercase mb-1.5">分析する案件</label>
            <div className="relative">
              <select
                value={selectedId}
                onChange={e => { setSelectedId(e.target.value); setResult(null); setError(null); }}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-yellow-400/30 appearance-none pr-8"
              >
                <option value="">— 案件を選択 —</option>
                {cases.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.company_name || "（名称なし）"} {c.timestamp?.slice(0, 10)}
                    {c.score != null ? ` [${Math.round(c.score)}pt]` : ""} [{c.final_status}]
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
            </div>
            {selectedCase && (
              <p className="text-xs text-slate-400 mt-1.5">
                現在スコア: <span className={`font-black ${scoreColor(selectedCase.score ?? 0)}`}>{selectedCase.score != null ? Math.round(selectedCase.score) : "—"}pt</span>
                　ステータス: {selectedCase.final_status}
              </p>
            )}
          </div>

          {/* 目標スコア */}
          <div>
            <label className="block text-xs font-black text-slate-500 uppercase mb-1.5">目標スコア</label>
            <div className="flex gap-2">
              {[60, 70, 80].map(s => (
                <button
                  key={s}
                  onClick={() => setTargetScore(s)}
                  className={`flex-1 py-3 rounded-xl text-sm font-black border-2 transition-all
                    ${targetScore === s ? "bg-yellow-50 border-yellow-400 text-yellow-700" : "bg-white border-slate-200 text-slate-500 hover:border-slate-300"}`}
                >
                  {s}pt
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 mt-1">60=条件付き / 70=承認 / 80=優良</p>
          </div>
        </div>

        <button
          onClick={analyze}
          disabled={!selectedId || analyzing}
          className="w-full py-3 rounded-xl bg-yellow-500 hover:bg-yellow-400 text-white font-black text-sm transition-all disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {analyzing ? <><Loader2 size={16} className="animate-spin" />分析中...</> : <><Lightbulb size={16} />Counterfactual 分析を実行</>}
        </button>

        {error && (
          <div className="flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm">
            <AlertCircle size={16} className="shrink-0 mt-0.5" />
            {error}
          </div>
        )}
      </div>

      {/* 結果 */}
      {result && (
        <>
          {/* スコアサマリー */}
          <div className={`rounded-2xl border shadow-sm p-5 ${alreadyApproved ? "bg-emerald-50 border-emerald-200" : "bg-white border-slate-200"}`}>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">現在スコア</p>
                <p className={`text-4xl font-black ${scoreColor(result.current_score)}`}>{result.current_score}pt</p>
              </div>
              <ArrowRight className="text-slate-300" size={24} />
              <div className="text-center">
                <p className="text-xs text-slate-500 mb-1">目標スコア</p>
                <p className="text-4xl font-black text-emerald-600">{result.target_score}pt</p>
              </div>
              <div className="flex-1" />
              {alreadyApproved ? (
                <div className="flex items-center gap-2 text-emerald-600 font-black">
                  <CheckCircle2 size={20} /> 既に目標達成済み
                </div>
              ) : (
                <div className="text-right">
                  <p className="text-xs text-slate-500">不足</p>
                  <p className="text-3xl font-black text-rose-600">−{result.gap.toFixed(1)}pt</p>
                </div>
              )}
            </div>

            {/* 現在の主要指標 */}
            <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-3 md:grid-cols-6 gap-3 text-xs">
              {[
                { label: "営業利益率", value: `${result.current_metrics.op_margin.toFixed(1)}%` },
                { label: "自己資本比率", value: `${result.current_metrics.eq_ratio.toFixed(1)}%` },
                { label: "売上高", value: `${(result.current_metrics.nenshu / 1000).toFixed(1)}百万` },
                { label: "格付", value: result.current_metrics.grade.slice(0, 6) },
                { label: "純資産", value: `${(result.current_metrics.net_assets / 1000).toFixed(1)}百万` },
                { label: "総資産", value: `${(result.current_metrics.total_assets / 1000).toFixed(1)}百万` },
              ].map(({ label, value }) => (
                <div key={label} className="bg-slate-50 rounded-xl p-2 text-center">
                  <p className="text-slate-400 mb-0.5">{label}</p>
                  <p className="font-black text-slate-700">{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Counterfactual カード */}
          {!alreadyApproved && result.counterfactuals.length > 0 && (
            <div className="space-y-3">
              <h2 className="text-sm font-black text-slate-700 flex items-center gap-2">
                <TrendingUp size={16} className="text-yellow-500" />
                承認に必要な改善パターン（{result.counterfactuals.length}案）
              </h2>
              {result.counterfactuals.map((cf, i) => {
                const col = DIFF_COLOR[cf.difficulty];
                return (
                  <div key={i} className={`bg-white border border-${col}-200 rounded-2xl shadow-sm p-5`}>
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`text-xs px-2 py-0.5 rounded-full font-black bg-${col}-100 text-${col}-700`}>
                            {cf.difficulty}
                          </span>
                          <span className="font-black text-slate-800 text-sm">{cf.label}</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm mb-2">
                          <span className="text-slate-500">{cf.current_display}</span>
                          <ArrowRight size={14} className="text-slate-400" />
                          <span className={`font-black text-${col}-700`}>{cf.required_display}</span>
                          {cf.change_pct != null && (
                            <span className="text-xs text-slate-400">(+{cf.change_pct.toFixed(0)}%)</span>
                          )}
                        </div>
                        <p className="text-xs text-slate-500">{cf.note}</p>
                      </div>
                      <div className="text-right shrink-0">
                        <p className="text-xs text-slate-400 mb-0.5">達成スコア</p>
                        <p className="text-2xl font-black text-emerald-600">{cf.achieved_score}pt</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {!alreadyApproved && result.counterfactuals.length === 0 && (
            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 text-center text-slate-500">
              <p className="font-bold">改善シナリオの計算ができませんでした。</p>
              <p className="text-xs mt-1">財務データが不足している可能性があります。</p>
            </div>
          )}

          {/* 感度分析グラフ */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
              <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                <BarChart2 size={13} /> 営業利益率 vs スコア
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={result.op_sensitivity} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="op_margin" tick={{ fontSize: 10, fill: "#94a3b8" }} unit="%" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={targetScore} stroke="#f59e0b" strokeDasharray="4 4"
                    label={{ value: `目標${targetScore}pt`, fontSize: 10, fill: "#f59e0b" }} />
                  <ReferenceLine x={result.current_metrics.op_margin} stroke="#6366f1"
                    strokeDasharray="4 4" label={{ value: "現在", fontSize: 10, fill: "#6366f1" }} />
                  <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
              <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-1.5">
                <BarChart2 size={13} /> 自己資本比率 vs スコア
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={result.eq_sensitivity} margin={{ top: 5, right: 15, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="eq_ratio" tick={{ fontSize: 10, fill: "#94a3b8" }} unit="%" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={targetScore} stroke="#f59e0b" strokeDasharray="4 4"
                    label={{ value: `目標${targetScore}pt`, fontSize: 10, fill: "#f59e0b" }} />
                  <ReferenceLine x={result.current_metrics.eq_ratio} stroke="#6366f1"
                    strokeDasharray="4 4" label={{ value: "現在", fontSize: 10, fill: "#6366f1" }} />
                  <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-xs text-amber-800">
            <p className="font-semibold mb-1">💡 この分析について</p>
            <p>
              現在のスコアリングモデルと係数を使って「最小限の変更で目標スコアを達成する条件」を自動計算しています。
              実際の審査では定性評価・市場環境・担当者判断も加味されます。
              改善シナリオは顧客へのフィードバック・経営改善提案の参考としてお使いください。
            </p>
          </div>
        </>
      )}

      {!result && !analyzing && (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-12 flex flex-col items-center text-slate-300">
          <Lightbulb size={48} className="mb-4" />
          <p className="font-bold text-sm">案件を選択して分析を実行してください</p>
        </div>
      )}
    </div>
  );
}
