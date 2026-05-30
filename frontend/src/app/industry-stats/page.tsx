"use client";

import React, { useEffect, useState, useMemo } from 'react';
import { apiClient } from '../../lib/api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';
import { BarChart3, RefreshCw, TrendingUp, ArrowUpDown, AlertCircle, Loader2 } from 'lucide-react';

type IndustryRow = {
  industry: string;
  total: number;
  won: number;
  lost: number;
  contract_rate: number;
  avg_score: number | null;
};

type SortKey = 'total' | 'contract_rate' | 'avg_score' | 'won';
type SortDir = 'asc' | 'desc';

const rateColor = (rate: number) => {
  if (rate >= 55) return '#10b981';
  if (rate >= 45) return '#f59e0b';
  return '#f43f5e';
};

const rateLabel = (rate: number) => {
  if (rate >= 55) return { text: '高', cls: 'text-emerald-700 bg-emerald-50' };
  if (rate >= 45) return { text: '標準', cls: 'text-amber-700 bg-amber-50' };
  return { text: '低', cls: 'text-rose-700 bg-rose-50' };
};

function CustomTooltip({ active, payload }: { active?: boolean; payload?: { payload: IndustryRow }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-slate-200 rounded-xl shadow-lg p-3 text-xs max-w-[220px]">
      <p className="font-black text-slate-700 mb-1 text-[11px] leading-snug">{d.industry}</p>
      <p className="font-bold text-slate-600">成約率: <span className="text-base font-black" style={{ color: rateColor(d.contract_rate) }}>{d.contract_rate}%</span></p>
      <div className="mt-1 space-y-0.5">
        <p className="text-slate-500">総案件: {d.total}件 / 成約: {d.won}件 / 失注: {d.lost}件</p>
        {d.avg_score != null && <p className="text-slate-500">平均スコア: <span className="font-bold text-indigo-600">{d.avg_score}pt</span></p>}
      </div>
    </div>
  );
}

export default function IndustryStatsPage() {
  const [data, setData] = useState<IndustryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('total');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [chartKey, setChartKey] = useState<'contract_rate' | 'avg_score'>('contract_rate');

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<IndustryRow[]>('/api/industry/stats');
      setData(res.data);
    } catch {
      setError('業種データの取得に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      const av = a[sortKey] ?? -1;
      const bv = b[sortKey] ?? -1;
      return sortDir === 'desc' ? (bv as number) - (av as number) : (av as number) - (bv as number);
    });
  }, [data, sortKey, sortDir]);

  const chartData = useMemo(() => {
    return sorted.slice(0, 12).map(d => ({
      ...d,
      shortLabel: d.industry.replace(/^\d+[-\s]/, '').slice(0, 10),
    }));
  }, [sorted]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const avgRate = data.length ? data.reduce((s, d) => s + d.contract_rate, 0) / data.length : 0;

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <BarChart3 className="text-sky-500" size={26} />
          <div>
            <h1 className="text-2xl font-bold text-slate-800">業種別成約率分析</h1>
            <p className="text-sm text-slate-500">過去の審査案件から業種別の成約率・平均スコアを集計します。</p>
          </div>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-black text-sm disabled:opacity-50 transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          更新
        </button>
      </div>

      {/* サマリーカード */}
      {!loading && data.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: '対象業種数', value: `${data.length}業種`, cls: 'text-slate-800' },
            { label: '全体平均成約率', value: `${avgRate.toFixed(1)}%`, cls: avgRate >= 50 ? 'text-emerald-600' : 'text-amber-600' },
            { label: '最高成約率', value: `${Math.max(...data.map(d => d.contract_rate)).toFixed(1)}%`, cls: 'text-emerald-600' },
            { label: '最低成約率', value: `${Math.min(...data.map(d => d.contract_rate)).toFixed(1)}%`, cls: 'text-rose-600' },
          ].map(c => (
            <div key={c.label} className="bg-white border border-slate-200 rounded-2xl p-4 text-center shadow-sm">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">{c.label}</p>
              <p className={`text-xl font-black ${c.cls}`}>{c.value}</p>
            </div>
          ))}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center h-48 text-slate-400 gap-3">
          <Loader2 className="w-8 h-8 animate-spin" />
          <p className="font-bold">集計中...</p>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm font-bold">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          {error}
        </div>
      )}

      {!loading && data.length > 0 && (
        <>
          {/* グラフ */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-black text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4" /> 上位12業種グラフ
              </h2>
              <div className="flex gap-2">
                {(['contract_rate', 'avg_score'] as const).map(k => (
                  <button
                    key={k}
                    onClick={() => setChartKey(k)}
                    className={`px-3 py-1 rounded-lg text-xs font-black transition-all ${chartKey === k ? 'bg-sky-500 text-white' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}
                  >
                    {k === 'contract_rate' ? '成約率' : '平均スコア'}
                  </button>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{ top: 5, right: 10, bottom: 60, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="shortLabel"
                  tick={{ fontSize: 10, fill: '#64748b' }}
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: '#94a3b8' }}
                  unit={chartKey === 'contract_rate' ? '%' : 'pt'}
                  domain={chartKey === 'contract_rate' ? [0, 100] : ['auto', 'auto']}
                />
                <Tooltip content={<CustomTooltip />} />
                {chartKey === 'contract_rate' && (
                  <ReferenceLine y={avgRate} stroke="#6366f1" strokeDasharray="4 4"
                    label={{ value: `平均${avgRate.toFixed(0)}%`, fontSize: 10, fill: '#6366f1' }} />
                )}
                <Bar dataKey={chartKey} radius={[4, 4, 0, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={chartKey === 'contract_rate' ? rateColor(d.contract_rate) : '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* テーブル */}
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100">
              <h2 className="text-sm font-black text-slate-600 uppercase tracking-wider">全業種一覧</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50">
                    {[
                      { label: '業種', key: null, cls: 'text-left pl-5' },
                      { label: '総案件', key: 'total' as SortKey, cls: 'text-right' },
                      { label: '成約', key: 'won' as SortKey, cls: 'text-right' },
                      { label: '失注', key: null, cls: 'text-right' },
                      { label: '成約率', key: 'contract_rate' as SortKey, cls: 'text-right' },
                      { label: '平均スコア', key: 'avg_score' as SortKey, cls: 'text-right pr-5' },
                    ].map(col => (
                      <th
                        key={col.label}
                        className={`py-2.5 text-[11px] font-black text-slate-500 uppercase tracking-wider ${col.cls} ${col.key ? 'cursor-pointer hover:text-slate-700' : ''}`}
                        onClick={() => col.key && handleSort(col.key)}
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.label}
                          {col.key && <ArrowUpDown className="w-3 h-3 opacity-50" />}
                          {sortKey === col.key && (sortDir === 'desc' ? ' ↓' : ' ↑')}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {sorted.map((row, i) => {
                    const rl = rateLabel(row.contract_rate);
                    return (
                      <tr key={i} className="hover:bg-slate-50 transition-colors">
                        <td className="py-2.5 pl-5 font-bold text-slate-700 text-xs max-w-[180px] truncate">{row.industry}</td>
                        <td className="py-2.5 text-right font-bold text-slate-600 text-xs">{row.total}</td>
                        <td className="py-2.5 text-right font-bold text-emerald-600 text-xs">{row.won}</td>
                        <td className="py-2.5 text-right font-bold text-rose-500 text-xs">{row.lost}</td>
                        <td className="py-2.5 text-right pr-2">
                          <span className={`inline-flex items-center gap-1 text-xs font-black px-2 py-0.5 rounded-full ${rl.cls}`}>
                            {rl.text} {row.contract_rate}%
                          </span>
                        </td>
                        <td className="py-2.5 pr-5 text-right font-bold text-indigo-600 text-xs">
                          {row.avg_score != null ? `${row.avg_score}pt` : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="px-5 py-2 bg-slate-50 border-t border-slate-100 text-[10px] text-slate-400">
              ※ 3件以上の案件がある業種のみ表示。成約率 55%以上=高/45〜55%=標準/45%未満=低。
            </div>
          </div>
        </>
      )}
    </div>
  );
}
