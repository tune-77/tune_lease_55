import React, { useState, useEffect } from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, ReferenceLine,
  AreaChart, Area
} from 'recharts';

interface Props {
  companyName?: string;
  nenshu?: number;       // 千円単位
  opMarginPct?: number;  // 営業利益率(%)
  equityRatio?: number;  // 自己資本比率(%)
  scoreBorrower?: number;
  scoreBase?: number;
}

interface ForecastYear {
  year: string;
  revenue: number;
}

export default function RealGraphs({
  companyName = "",
  nenshu = 0,
  opMarginPct = 0,
  equityRatio = 0,
  scoreBorrower = 50,
  scoreBase = 50,
}: Props) {
  const [futureData, setFutureData] = useState<ForecastYear[]>([]);
  const [forecastLoading, setForecastLoading] = useState(false);

  // スコアからレーダーデータを生成
  const radarData = [
    { subject: 'P/L (収益性)', A: Math.min(100, Math.max(0, opMarginPct * 5 + 50)), fullMark: 100 },
    { subject: 'B/S (安全性)', A: Math.min(100, Math.max(0, equityRatio * 1.5 + 20)), fullMark: 100 },
    { subject: '信用 (借手)', A: Math.min(100, Math.max(0, scoreBorrower)), fullMark: 100 },
    { subject: '総合評価',    A: Math.min(100, Math.max(0, scoreBase)), fullMark: 100 },
    { subject: '財務安定性',  A: Math.min(100, Math.max(0, (equityRatio + opMarginPct * 2) / 2 + 20)), fullMark: 100 },
    { subject: '収益継続性',  A: Math.min(100, Math.max(0, opMarginPct * 3 + 40)), fullMark: 100 },
  ];

  // スコア要因分解（近似SHAP）
  const shapData = [
    { name: '営業利益率',     value: parseFloat(((opMarginPct - 5) * 0.8).toFixed(1)) },
    { name: '自己資本比率',   value: parseFloat(((equityRatio - 30) * 0.3).toFixed(1)) },
    { name: '借手スコア',     value: parseFloat(((scoreBorrower - 50) * 0.25).toFixed(1)) },
    { name: '業界調整',       value: parseFloat(((scoreBase - scoreBorrower) * 0.5).toFixed(1)) },
  ].sort((a, b) => a.value - b.value);

  // 5年後売上予測: GBMパスを年次に集約
  useEffect(() => {
    if (!nenshu && !companyName) return;
    setForecastLoading(true);
    fetch('/api/timesfm/financial_paths', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        company_name: companyName || '（未入力）',
        n_periods: 60,
      }),
    })
      .then(r => r.json())
      .then(d => {
        const median: number[] = d.gbm_median || [];
        if (!median.length) return;

        // GBMは千円単位で返ってくるが、revenues が nenshu ベース
        // median[0] が現在値、以降60期(月次)
        // 年次（12ヶ月ごと）に集約
        const currentYear = new Date().getFullYear();
        const yearly: ForecastYear[] = [];

        // 実績(現在)
        const base = median[0] || nenshu * 1000 || 10_000_000;
        yearly.push({ year: `${currentYear}(実)`, revenue: Math.round(base / 1000) });

        for (let y = 1; y <= 5; y++) {
          const idx = Math.min(y * 12, median.length - 1);
          yearly.push({
            year: `${currentYear + y}(予)`,
            revenue: Math.round((median[idx] || base) / 1000),
          });
        }
        setFutureData(yearly);
      })
      .catch(() => {})
      .finally(() => setForecastLoading(false));
  }, [companyName, nenshu]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-6 mb-8">

      {/* 1. 総合バランス レーダー */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-violet-500">🎯</span> 総合審査バランス (Radar)
        </h3>
        <div className="h-[200px] sm:h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 'bold' }} />
              <Radar name="対象企業" dataKey="A" stroke="#8b5cf6" strokeWidth={2} fill="#8b5cf6" fillOpacity={0.3} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}
                itemStyle={{ color: '#8b5cf6', fontWeight: 'bold' }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">
          営業利益率 {opMarginPct.toFixed(1)}% / 自己資本 {equityRatio.toFixed(1)}%
        </p>
      </div>

      {/* 2. スコア変動要因 (近似SHAP) */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-emerald-500">📈</span> スコア変動の要因 (近似SHAP)
        </h3>
        <div className="h-[200px] sm:h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 11, fill: '#475569', fontWeight: 600 }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: '#f8fafc' }}
                contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }}
                formatter={(val: unknown) => { const n = Number(val); return [n > 0 ? `+${n}点` : `${n}点`, '影響度']; }}
              />
              <ReferenceLine x={0} stroke="#cbd5e1" />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                {shapData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#f43f5e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">※ 右に伸びる要素が加点、左が減点要素</p>
      </div>

      {/* 3. 5年後売上予測シミュレーション */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 lg:col-span-2 xl:col-span-1">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-blue-500">🔮</span> 5年後売上予測シミュレーション
        </h3>
        <div className="h-[250px] w-full relative">
          {forecastLoading ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
              <div className="w-7 h-7 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-blue-500 font-bold animate-pulse">GBM パス計算中...</span>
            </div>
          ) : futureData.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-sm text-slate-400">スコア計算後に自動表示されます</span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={futureData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorRev" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }}
                  formatter={(val: unknown) => [`${Number(val).toLocaleString()} 千円`, '予測売上']}
                />
                <Area type="monotone" dataKey="revenue" name="予測値" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRev)" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">
          ※ GBM (幾何ブラウン運動) による売上期待値推移（千円）
        </p>
      </div>

    </div>
  );
}
