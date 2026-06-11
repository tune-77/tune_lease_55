"use client";

import React, { useCallback, useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine
} from "recharts";
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, TrendingDown, TrendingUp } from "lucide-react";

type MonthlyRow = {
  month: string;
  count: number;
  won: number;
  lost: number;
  win_rate: number | null;
  avg_score: number | null;
  avg_score_won: number | null;
  avg_score_lost: number | null;
};

type Summary = {
  total: number;
  won_count: number;
  lost_count: number;
  avg_score_won: number | null;
  avg_score_lost: number | null;
  separation: number | null;
  drift_alert: boolean;
};

type ScoreDist = { range: string; count: number };

type DriftData = {
  monthly: MonthlyRow[];
  summary: Summary;
  score_dist: ScoreDist[];
};

const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg p-3 text-xs space-y-1 min-w-[140px]">
      <p className="font-black text-slate-700 mb-2">{label}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex justify-between gap-4">
          <span style={{ color: p.color }} className="font-bold">{p.name}</span>
          <span className="font-black text-slate-700">{typeof p.value === "number" ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function DriftPage() {
  const [data, setData] = useState<DriftData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.get<DriftData>("/api/drift-stats");
      setData(res.data);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const sep = data?.summary.separation;
  const driftAlert = data?.summary.drift_alert;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Activity className="text-indigo-500" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">データドリフト監視</h1>
          <p className="text-sm text-slate-500">スコアリングモデルの予測精度・ドリフトを時系列で監視します。</p>
        </div>
        <button
          onClick={fetchData}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm text-slate-700"
        >
          <RefreshCw size={14} /> 更新
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <RefreshCw className="animate-spin text-slate-400" size={24} />
        </div>
      ) : !data ? (
        <div className="p-8 text-center text-red-600">データの取得に失敗しました</div>
      ) : (
        <>
          {/* ドリフトアラートバナー */}
          {driftAlert ? (
            <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-300 rounded-xl">
              <AlertTriangle className="text-red-500 shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-bold text-red-800">⚠️ ドリフト検知: スコア分離度が低下しています</p>
                <p className="text-sm text-red-700 mt-0.5">
                  成約・失注間のスコア差が <strong>{sep?.toFixed(1)}pt</strong> と閾値（5pt）を下回っています。
                  モデルの再学習または係数の見直しを検討してください。
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-3 p-4 bg-emerald-50 border border-emerald-200 rounded-xl">
              <CheckCircle2 className="text-emerald-500 shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-bold text-emerald-800">モデル正常稼働中</p>
                <p className="text-sm text-emerald-700 mt-0.5">
                  成約・失注スコア分離度: <strong>{sep?.toFixed(1)}pt</strong>（閾値 5pt以上）
                </p>
              </div>
            </div>
          )}

          {/* サマリーカード */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "総案件数", value: data.summary.total, icon: null, color: "text-slate-700" },
              { label: "成約平均スコア", value: data.summary.avg_score_won != null ? `${data.summary.avg_score_won}pt` : "—", icon: <TrendingUp size={14} className="text-emerald-500" />, color: "text-emerald-700" },
              { label: "失注平均スコア", value: data.summary.avg_score_lost != null ? `${data.summary.avg_score_lost}pt` : "—", icon: <TrendingDown size={14} className="text-rose-500" />, color: "text-rose-700" },
              { label: "スコア分離度", value: sep != null ? `${sep.toFixed(1)}pt` : "—", icon: <Activity size={14} className={driftAlert ? "text-red-500" : "text-indigo-500"} />, color: driftAlert ? "text-red-700" : "text-indigo-700" },
            ].map(({ label, value, icon, color }) => (
              <div key={label} className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">{icon}{label}</div>
                <p className={`text-2xl font-black ${color}`}>{value}</p>
              </div>
            ))}
          </div>

          {/* 月次スコア推移グラフ */}
          {data.monthly.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
              <h2 className="text-sm font-bold text-slate-700 mb-4">月次スコア推移（成約 vs 失注）</h2>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={data.monthly} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis domain={[40, 100]} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <ReferenceLine y={70} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "承認基準70", fontSize: 10, fill: "#94a3b8" }} />
                  <Line type="monotone" dataKey="avg_score_won" name="成約avg" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="avg_score_lost" name="失注avg" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} connectNulls />
                  <Line type="monotone" dataKey="avg_score" name="全体avg" stroke="#6366f1" strokeWidth={1.5} strokeDasharray="4 4" dot={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 月次成約率推移 */}
            {data.monthly.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <h2 className="text-sm font-bold text-slate-700 mb-4">月次成約率 (%)</h2>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={data.monthly.filter(m => m.win_rate != null)} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#94a3b8" }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#94a3b8" }} unit="%" />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" />
                    <Bar dataKey="win_rate" name="成約率" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* スコア分布 */}
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
              <h2 className="text-sm font-bold text-slate-700 mb-4">スコア分布</h2>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.score_dist} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="range" tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="件数" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500">
            <p className="font-semibold text-slate-600 mb-1">💡 ドリフト監視について</p>
            <p>
              成約・失注間のスコア分離度が5pt未満になると、モデルが成約と失注を区別できていない可能性があります。
              この場合は <strong>係数分析・更新</strong> ページで再学習を実施するか、係数を手動調整してください。
              月次で継続的に監視し、分離度を8pt以上に保つことを推奨します。
            </p>
          </div>
        </>
      )}
    </div>
  );
}
