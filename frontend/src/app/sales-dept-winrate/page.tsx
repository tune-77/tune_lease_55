"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { Building2, RefreshCw, TrendingUp } from "lucide-react";
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

type DeptItem = {
  dept: string;
  won: number;
  lost: number;
  total: number;
  win_rate: number;
  avg_score: number;
  diff: number;
};

type WinrateData = {
  items: DeptItem[];
  overall_rate: number;
  total_won: number;
  total_lost: number;
};

const DEPT_NOTES: Record<string, string> = {
  "宇都宮営業部": "製造業・建設業案件が多い。スコア重視より関係構築型の成約が多い傾向。",
  "埼玉営業部": "情報通信・サービス業が中心。競合が多く成約率はやや低め。",
  "小山営業部": "農業・物流系が強い。設備投資ニーズが明確な案件が多い。",
  "足利営業部": "中小製造業が多い。初期相談から成約まで期間が長い傾向。",
};

const barColor = (rate: number) =>
  rate >= 60 ? "#16a34a" : rate >= 45 ? "#d97706" : "#dc2626";

export default function SalesDeptWinratePage() {
  const [data, setData] = useState<WinrateData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<WinrateData>("/api/cases/sales-dept-winrate");
      setData(res.data);
    } catch {
      setError("データの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const sorted = useMemo(
    () => [...(data?.items ?? [])].sort((a, b) => b.win_rate - a.win_rate),
    [data],
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

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* ヘッダー */}
      <div className="flex items-center gap-3">
        <Building2 className="text-blue-600" size={28} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">営業部別成約率ダッシュボード</h1>
          <p className="text-sm text-slate-500">past_cases の sales_dept カラムから集計。成約＋検収完了を「成約」としてカウント。</p>
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
          { label: "累計成約", value: `${data.total_won.toLocaleString()} 件`, icon: <span className="text-green-600 font-bold text-lg">✓</span> },
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

      {/* 棒グラフ */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <p className="text-sm font-semibold text-slate-700 mb-3">
          営業部別成約率
          <span className="ml-2 text-xs font-normal text-slate-400">緑: 60%以上 ／ 黄: 45〜60% ／ 赤: 45%未満</span>
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sorted} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="dept" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 105]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0].payload as DeptItem;
                return (
                  <div className="bg-white border border-slate-200 rounded-lg shadow-md px-3 py-2 text-xs">
                    <p className="font-semibold text-slate-700 mb-1">{row.dept}</p>
                    <p>成約率: <strong>{row.win_rate.toFixed(1)}%</strong></p>
                    <p>平均スコア: {row.avg_score}pt</p>
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
              {sorted.map((entry) => (
                <Cell key={entry.dept} fill={barColor(entry.win_rate)} />
              ))}
              <LabelList
                dataKey="win_rate"
                position="top"
                formatter={(v: unknown) => `${v}%`}
                style={{ fontSize: 11, fontWeight: 700 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 詳細テーブル */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">営業部別 詳細一覧</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 text-slate-600 text-xs">
              <th className="text-left px-4 py-2">営業部</th>
              <th className="text-center px-3 py-2">成約率</th>
              <th className="text-center px-3 py-2">全体比</th>
              <th className="text-center px-3 py-2">成約</th>
              <th className="text-center px-3 py-2">失注</th>
              <th className="text-center px-3 py-2">合計</th>
              <th className="text-center px-3 py-2">平均スコア</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row, i) => {
              const color = barColor(row.win_rate);
              const diffColor = row.diff >= 0 ? "#16a34a" : "#dc2626";
              return (
                <tr key={row.dept} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                  <td className="px-4 py-2 font-medium text-slate-700">{row.dept}</td>
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
                  <td className="px-3 py-2 text-center text-slate-500">{row.avg_score}pt</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 営業部特性メモ */}
      <div>
        <p className="text-sm font-semibold text-slate-700 mb-3">📝 営業部 特性メモ</p>
        <div className="grid grid-cols-2 gap-3">
          {sorted.map((row) => {
            const color = barColor(row.win_rate);
            const note = DEPT_NOTES[row.dept] ?? "特性データ未登録。";
            return (
              <div
                key={row.dept}
                className="bg-white rounded-xl border border-slate-200 shadow-sm p-4"
                style={{ borderLeft: `4px solid ${color}` }}
              >
                <p className="font-semibold text-slate-800 mb-1">{row.dept}</p>
                <p className="text-xs mb-2">
                  <span className="font-bold" style={{ color }}>{row.win_rate.toFixed(1)}%</span>
                  <span className="text-slate-400 ml-2">平均スコア {row.avg_score}pt</span>
                </p>
                <p className="text-xs text-slate-500">{note}</p>
              </div>
            );
          })}
        </div>
        <p className="text-xs text-slate-400 mt-2">⚠️ 特性メモはサンプルデータです。実態に合わせて随時更新してください。</p>
      </div>
    </div>
  );
}
