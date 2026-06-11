"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import { BarChart2, RefreshCw, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type IndustryItem = {
  industry: string;
  won: number;
  lost: number;
  total: number;
  win_rate: number;
  diff: number;
};

type WinrateData = {
  items: IndustryItem[];
  overall_rate: number;
  total_won: number;
  total_lost: number;
};

const barColor = (rate: number) =>
  rate >= 60 ? "#16a34a" : rate >= 45 ? "#d97706" : "#dc2626";

export default function IndustryWinratePage() {
  const [data, setData] = useState<WinrateData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [minCases, setMinCases] = useState(10);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<WinrateData>("/api/cases/industry-winrate");
      setData(res.data);
    } catch (e) {
      setError("データの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const filtered = useMemo(
    () => (data?.items ?? []).filter((d) => d.total >= minCases),
    [data, minCases],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="animate-spin text-blue-500" size={32} />
        <span className="ml-3 text-slate-600">集計中...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 text-red-600">
        {error ?? "データがありません"}
        <button onClick={fetchData} className="ml-4 text-blue-600 underline">再取得</button>
      </div>
    );
  }

  const totalAll = data.total_won + data.total_lost;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* ヘッダー */}
      <div className="flex items-center gap-3">
        <BarChart2 className="text-blue-600" size={28} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">業種別成約率ダッシュボード</h1>
          <p className="text-sm text-slate-500">past_cases の実績データから集計。成約＋検収完了を「成約」としてカウント。</p>
        </div>
        <button
          onClick={fetchData}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm text-slate-700"
        >
          <RefreshCw size={14} /> 更新
        </button>
      </div>

      {/* サマリーカード */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "全体成約率", value: `${data.overall_rate.toFixed(1)}%`, icon: <TrendingUp size={20} className="text-blue-500" /> },
          { label: "累計成約", value: `${data.total_won.toLocaleString()} 件` , icon: <span className="text-green-600 font-bold text-lg">✓</span> },
          { label: "累計失注", value: `${data.total_lost.toLocaleString()} 件`, icon: <span className="text-red-500 font-bold text-lg">✗</span> },
        ].map(({ label, value, icon }) => (
          <div key={label} className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-3 shadow-sm">
            {icon}
            <div>
              <p className="text-xs text-slate-500">{label}</p>
              <p className="text-xl font-bold text-slate-800">{value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* フィルタ */}
      <div className="flex items-center gap-4 bg-slate-50 rounded-lg p-3">
        <label className="text-sm text-slate-600 whitespace-nowrap">最低件数フィルタ:</label>
        {[5, 10, 20, 50].map((n) => (
          <button
            key={n}
            onClick={() => setMinCases(n)}
            className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
              minCases === n
                ? "bg-blue-600 text-white"
                : "bg-white border border-slate-300 text-slate-600 hover:bg-blue-50"
            }`}
          >
            {n}件以上
          </button>
        ))}
        <span className="text-xs text-slate-400 ml-2">{filtered.length} 業種表示中</span>
      </div>

      {/* 棒グラフ */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <p className="text-sm font-semibold text-slate-700 mb-3">
          業種別成約率
          <span className="ml-2 text-xs font-normal text-slate-400">
            緑: 60%以上 ／ 黄: 45〜60% ／ 赤: 45%未満
          </span>
        </p>
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={filtered} margin={{ top: 20, right: 20, left: 0, bottom: 80 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="industry"
              tick={{ fontSize: 11 }}
              angle={-35}
              textAnchor="end"
              interval={0}
            />
            <YAxis domain={[0, 105]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0].payload as IndustryItem;
                return (
                  <div className="bg-white border border-slate-200 rounded-lg shadow-md px-3 py-2 text-xs">
                    <p className="font-semibold text-slate-700 mb-1">{row.industry}</p>
                    <p>成約率: <strong>{row.win_rate.toFixed(1)}%</strong></p>
                    <p className="text-slate-500">成約 {row.won}件 / 失注 {row.lost}件 / 計 {row.total}件</p>
                  </div>
                );
              }}
            />
            <ReferenceLine
              y={data.overall_rate}
              stroke="#6366f1"
              strokeDasharray="4 2"
              label={{ value: `全体平均 ${data.overall_rate.toFixed(1)}%`, position: "right", fontSize: 11, fill: "#6366f1" }}
            />
            <Bar dataKey="win_rate" radius={[4, 4, 0, 0]}>
              {filtered.map((entry) => (
                <Cell key={entry.industry} fill={barColor(entry.win_rate)} />
              ))}
              <LabelList
                dataKey="win_rate"
                position="top"
                formatter={(v: unknown) => `${v}%`}
                style={{ fontSize: 10 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* テーブル */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">業種別 詳細一覧</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-slate-600 text-xs">
                <th className="text-left px-4 py-2">業種</th>
                <th className="text-center px-3 py-2">成約率</th>
                <th className="text-center px-3 py-2">全体比</th>
                <th className="text-center px-3 py-2">成約</th>
                <th className="text-center px-3 py-2">失注</th>
                <th className="text-center px-3 py-2">合計</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const color = barColor(row.win_rate);
                const diffColor = row.diff >= 0 ? "#16a34a" : "#dc2626";
                return (
                  <tr key={row.industry} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                    <td className="px-4 py-2 font-medium text-slate-700">{row.industry}</td>
                    <td className="px-3 py-2 text-center">
                      <span
                        className="inline-block px-2 py-0.5 rounded-full text-xs font-bold"
                        style={{ background: `${color}22`, color, border: `1px solid ${color}` }}
                      >
                        {row.win_rate.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-2 text-center text-xs font-semibold" style={{ color: diffColor }}>
                      {row.diff >= 0 ? "+" : ""}{row.diff.toFixed(1)}pt
                    </td>
                    <td className="px-3 py-2 text-center text-green-700 font-medium">{row.won}</td>
                    <td className="px-3 py-2 text-center text-red-600 font-medium">{row.lost}</td>
                    <td className="px-3 py-2 text-center text-slate-600">{row.total}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="px-4 py-2 text-xs text-slate-400 border-t border-slate-100">
          ⚠️ 合計20件未満の業種は参考値です。営業判断の補助としてご活用ください。
        </p>
      </div>
    </div>
  );
}
