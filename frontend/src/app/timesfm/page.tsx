"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Activity, Clock, TrendingUp, Search, BarChart3, Calendar } from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, AreaChart, LineChart,
  Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine
} from 'recharts';

const TERM_COLS = ["r_2y","r_3y","r_4y","r_5y","r_6y","r_7y","r_8y","r_9y","r_over9y"];
const TERM_LABELS: Record<string, string> = {
  r_2y:"2年以内", r_3y:"3年以内", r_4y:"4年以内",
  r_5y:"5年以内", r_6y:"6年以内", r_7y:"7年以内",
  r_8y:"8年以内", r_9y:"9年以内", r_over9y:"9年超",
};
const TERM_COLORS = ["#6366f1","#8b5cf6","#ec4899","#f43f5e","#f97316","#eab308","#22c55e","#14b8a6","#3b82f6"];

const TAB_TOOLTIP = {
  contentStyle: { backgroundColor: '#1e293b', border: '1px solid #334155', color: '#f1f5f9', borderRadius: 8 },
  labelStyle: { color: '#94a3b8', fontWeight: 'bold' },
};

// ─── ファンチャート用データ構築 ─────────────────────────────────────────────

function buildFanChartData(
  monthsHist: string[], histVals: number[],
  monthsFore: string[], foreVals: number[],
  bandLow: number, bandHigh: number
) {
  const data: any[] = [];
  monthsHist.forEach((m, i) => {
    data.push({ month: m, actual: histVals[i], forecast: null, low: null, high: null });
  });
  const lastActual = histVals[histVals.length - 1];
  monthsFore.forEach((m, i) => {
    const v = foreVals[i];
    data.push({
      month: m,
      actual: i === 0 ? lastActual : null,
      forecast: v,
      low: bandLow,
      high: bandHigh,
      band: [bandLow, bandHigh],
    });
  });
  return data;
}

// ─── 業種トレンドチャート ────────────────────────────────────────────────────

function IndustryChart({ data }: { data: any }) {
  if (!data?.months_history) return null;
  const histLen = data.months_history.length;
  const foreLen = data.months_forecast.length;
  const chartData: any[] = [];

  for (let i = 0; i < histLen; i++) {
    chartData.push({ month: data.months_history[i], actual: data.avg_score_hist[i], forecast: null });
  }
  const lastActual = data.avg_score_hist[histLen - 1];
  const lastMonth = data.months_history[histLen - 1];
  for (let i = 0; i < foreLen; i++) {
    chartData.push({
      month: data.months_forecast[i],
      actual: i === 0 ? lastActual : null,
      forecast: data.avg_score_fore[i],
      low: data.avg_score_fore[i] - 3,
      high: data.avg_score_fore[i] + 3,
    });
  }

  const signal = data.risk_signal;
  const signalColor = signal === 'positive' ? '#22c55e' : signal === 'negative' ? '#ef4444' : '#f59e0b';
  const signalLabel = signal === 'positive' ? '上昇トレンド' : signal === 'negative' ? '下落トレンド' : '横ばい';

  return (
    <div>
      <div className="flex items-center gap-4 mb-4">
        <div className="px-4 py-2 rounded-xl font-black text-sm" style={{ background: signalColor + '22', color: signalColor, border: `1px solid ${signalColor}44` }}>
          {signalLabel}
        </div>
        <span className="text-slate-400 text-sm">予測手法: {data.method}</span>
      </div>
      <div style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="month" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 10 }} minTickGap={20} />
            <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
            <Tooltip {...TAB_TOOLTIP} />
            <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11 }} />
            <Area type="monotone" dataKey="band" fill="#3b82f6" fillOpacity={0.15} stroke="none" name="不確実性帯" />
            <Line type="monotone" dataKey="actual" stroke="#4ade80" strokeWidth={2} dot={false} name="実績スコア" connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#60a5fa" strokeWidth={2} strokeDasharray="5 5" dot={false} name="予測トレンド" connectNulls={false} />
            <ReferenceLine x={lastMonth} stroke="#475569" strokeDasharray="3 3" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 個別スコア予測チャート ─────────────────────────────────────────────────

function CompanyScoreChart({ data }: { data: any }) {
  if (!data?.score_history) return null;
  const histLen = data.score_history.length;
  const foreLen = data.score_forecast.length;
  const chartData: any[] = [];

  for (let i = 0; i < histLen; i++) {
    chartData.push({ idx: `T-${histLen - i}`, actual: data.score_history[i], forecast: null });
  }
  const lastActual = data.score_history[histLen - 1];
  for (let i = 0; i < foreLen; i++) {
    chartData.push({
      idx: `M+${i + 1}`,
      actual: i === 0 ? lastActual : null,
      forecast: data.score_forecast[i],
      low: data.band_low[i],
      high: data.band_high[i],
      band: [data.band_low[i], data.band_high[i]],
    });
  }
  const trend = data.trend;
  const trendColor = trend === 'up' ? '#22c55e' : trend === 'down' ? '#ef4444' : '#f59e0b';

  return (
    <div>
      <div className="flex gap-6 mb-4">
        {[
          { label: '最新スコア', val: data.score_history[histLen - 1]?.toFixed(1) },
          { label: '予測スコア', val: data.score_forecast[foreLen - 1]?.toFixed(1) },
          { label: 'トレンド', val: trend === 'up' ? '↑ 上昇' : trend === 'down' ? '↓ 下落' : '→ 横ばい', color: trendColor },
        ].map(({ label, val, color }) => (
          <div key={label} className="bg-slate-800 px-5 py-3 rounded-xl">
            <div className="text-xs text-slate-400 font-bold mb-1">{label}</div>
            <div className="text-2xl font-black" style={{ color: color || '#f1f5f9' }}>{val}</div>
          </div>
        ))}
      </div>
      <div style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="idx" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
            <Tooltip {...TAB_TOOLTIP} />
            <Area type="monotone" dataKey="band" fill="#8b5cf6" fillOpacity={0.15} stroke="none" name="不確実性帯" />
            <Line type="monotone" dataKey="actual" stroke="#4ade80" strokeWidth={2} dot={false} name="実績スコア" connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#a78bfa" strokeWidth={2} strokeDasharray="5 5" dot={false} name="予測スコア" connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 成約金利推移チャート ────────────────────────────────────────────────────

function RateForecastChart({ data }: { data: any }) {
  if (!data?.rate_history) return null;
  const chartData = buildFanChartData(
    data.rate_history.map((_: any, i: number) => `T-${data.rate_history.length - i}`),
    data.rate_history,
    data.horizon_forecast.map((_: any, i: number) => `M+${i + 1}`),
    data.horizon_forecast,
    data.band_low,
    data.band_high,
  );

  return (
    <div>
      <div className="flex gap-6 mb-4">
        {[
          { label: '将来中央金利', val: `${data.rate_forecast?.toFixed(2)}%` },
          { label: '予測下限', val: `${data.band_low?.toFixed(2)}%` },
          { label: '予測上限', val: `${data.band_high?.toFixed(2)}%` },
        ].map(({ label, val }) => (
          <div key={label} className="bg-slate-800 px-5 py-3 rounded-xl">
            <div className="text-xs text-slate-400 font-bold mb-1">{label}</div>
            <div className="text-2xl font-black text-emerald-400">{val}</div>
          </div>
        ))}
      </div>
      <div style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="month" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} />
            <Tooltip {...TAB_TOOLTIP} />
            <Area type="monotone" dataKey="band" fill="#10b981" fillOpacity={0.12} stroke="none" name="予測幅" />
            <Line type="monotone" dataKey="actual" stroke="#4ade80" strokeWidth={2} dot={false} name="実績金利" connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#34d399" strokeWidth={2} strokeDasharray="5 5" dot={false} name="予測金利" connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-500 mt-2">※ 成約案件の実績金利履歴から予測（手法: {data.method}）</p>
    </div>
  );
}

// ─── GBM vs TimesFM チャート ─────────────────────────────────────────────────

function CompareChart({ data }: { data: any }) {
  if (!data?.revenues) return null;
  const n = data.gbm_median?.length || 0;
  const chartData = data.revenues.map((v: number, i: number) => ({ idx: `T-${data.revenues.length - i}`, actual: v / 1e6 }));
  for (let i = 0; i < n; i++) {
    chartData.push({
      idx: `M+${i + 1}`,
      gbm: data.gbm_median[i] / 1e6,
      tfm: data.tfm_median?.[i] ? data.tfm_median[i] / 1e6 : null,
    });
  }

  return (
    <div>
      <div className="flex gap-3 mb-4 text-xs">
        <span className="flex items-center gap-1.5"><span className="w-4 h-0.5 bg-orange-400 inline-block" />GBM中央値</span>
        {data.timesfm_available && <span className="flex items-center gap-1.5"><span className="w-4 h-0.5 bg-fuchsia-400 inline-block border-dashed border-t-2 border-fuchsia-400" />TimesFM中央値</span>}
      </div>
      <div style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="idx" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} unit="M" />
            <Tooltip {...TAB_TOOLTIP} formatter={(v: any) => `${Number(v).toFixed(1)}M円`} />
            <Line type="monotone" dataKey="actual" stroke="#94a3b8" strokeWidth={2} dot={false} name="実績" connectNulls={false} />
            <Line type="monotone" dataKey="gbm" stroke="#fb923c" strokeWidth={2} strokeDasharray="5 5" dot={false} name="GBM予測" connectNulls={false} />
            {data.timesfm_available && (
              <Line type="monotone" dataKey="tfm" stroke="#e879f9" strokeWidth={2} strokeDasharray="5 5" dot={false} name="TimesFM予測" connectNulls={false} />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 基準金利予測チャート ────────────────────────────────────────────────────

function BaseRateChart({ data }: { data: any }) {
  if (!data?.rate_history) return null;
  const chartData = buildFanChartData(
    data.months_history, data.rate_history,
    data.months_forecast, data.horizon_forecast,
    data.band_low, data.band_high,
  );

  return (
    <div>
      <div className="flex gap-6 mb-4">
        {[
          { label: `現在 (${data.months_history[data.months_history.length - 1]})`, val: `${data.rate_history[data.rate_history.length - 1]?.toFixed(2)}%` },
          { label: `${data.horizon_forecast.length}ヶ月後 予測`, val: `${data.rate_forecast?.toFixed(2)}%` },
          { label: '予測変動幅', val: `±${((data.band_high - data.rate_forecast) || 0).toFixed(2)}%` },
        ].map(({ label, val }) => (
          <div key={label} className="bg-slate-800 px-5 py-3 rounded-xl">
            <div className="text-xs text-slate-400 font-bold mb-1">{label}</div>
            <div className="text-2xl font-black text-indigo-400">{val}</div>
          </div>
        ))}
      </div>
      <div style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="month" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 10 }} minTickGap={20} />
            <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} />
            <Tooltip {...TAB_TOOLTIP} formatter={(v: any) => `${Number(v).toFixed(2)}%`} />
            <Area type="monotone" dataKey="band" fill="#6366f1" fillOpacity={0.12} stroke="none" name="予測幅" />
            <Line type="monotone" dataKey="actual" stroke="#818cf8" strokeWidth={2} dot={false} name={`実績 (${data.term_label})`} connectNulls={false} />
            <Line type="monotone" dataKey="forecast" stroke="#a5b4fc" strokeWidth={2} strokeDasharray="5 5" dot={false} name="予測" connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-500 mt-2">※ 基準金利マスタの実績データから予測（手法: {data.method}）</p>
    </div>
  );
}

// ─── 全期間一括比較チャート ─────────────────────────────────────────────────

function BaseRateAllChart({ data }: { data: any }) {
  if (!data?.forecasts) return null;
  const forecasts = data.forecasts;
  const allMonths = new Set<string>();
  Object.values(forecasts).forEach((r: any) => {
    r.months_history?.forEach((m: string) => allMonths.add(m));
    r.months_forecast?.forEach((m: string) => allMonths.add(m + '_f'));
  });
  const histMonths = Object.values(forecasts)[0] ? (Object.values(forecasts)[0] as any).months_history : [];
  const foreMonths = Object.values(forecasts)[0] ? (Object.values(forecasts)[0] as any).months_forecast : [];

  const chartData: any[] = histMonths.map((m: string, i: number) => {
    const pt: any = { month: m };
    TERM_COLS.forEach(col => {
      const r: any = forecasts[col];
      if (r) pt[col] = r.rate_history[i] ?? null;
    });
    return pt;
  });
  foreMonths.forEach((m: string, i: number) => {
    const pt: any = { month: m + '▶' };
    TERM_COLS.forEach(col => {
      const r: any = forecasts[col];
      if (r) pt[col + '_f'] = r.horizon_forecast[i] ?? null;
    });
    chartData.push(pt);
  });

  return (
    <div style={{ height: 360 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis dataKey="month" stroke="#475569" tick={{ fill: '#94a3b8', fontSize: 9 }} minTickGap={25} />
          <YAxis stroke="#475569" tick={{ fill: '#94a3b8' }} />
          <Tooltip {...TAB_TOOLTIP} formatter={(v: any) => `${Number(v).toFixed(2)}%`} />
          <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 10 }} />
          {TERM_COLS.map((col, i) => (
            <React.Fragment key={col}>
              <Line type="monotone" dataKey={col} stroke={TERM_COLORS[i]} strokeWidth={1.5} dot={false} name={TERM_LABELS[col]} connectNulls={false} />
              <Line type="monotone" dataKey={col + '_f'} stroke={TERM_COLORS[i]} strokeWidth={1.5} strokeDasharray="4 3" dot={false} name={undefined} connectNulls={false} legendType="none" />
            </React.Fragment>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── メインページ ────────────────────────────────────────────────────────────

type TabId = 'industry' | 'company' | 'rate' | 'compare' | 'baserate';

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'industry',  label: '業種トレンド',    icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'company',   label: '個別スコア',      icon: <BarChart3 className="w-4 h-4" /> },
  { id: 'rate',      label: '成約金利推移',    icon: <Search className="w-4 h-4" /> },
  { id: 'compare',   label: 'GBM vs TimesFM',  icon: <Activity className="w-4 h-4" /> },
  { id: 'baserate',  label: '基準金利予測',    icon: <Calendar className="w-4 h-4" /> },
];

export default function TimesFMPage() {
  const [activeTab, setActiveTab] = useState<TabId>('industry');
  const [data, setData] = useState<any>(null);
  const [allData, setAllData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [target, setTarget] = useState('建設業');
  const [termCol, setTermCol] = useState('r_5y');
  const [horizon, setHorizon] = useState(6);

  useEffect(() => {
    triggerMebuki('guide', 'TimesFM時系列予測です！\n業種トレンドや基準金利の将来を先読みします！');
    fetchData('industry');
  }, []);

  const fetchData = async (tab: TabId, tgt?: string, col?: string, h?: number) => {
    setLoading(true);
    setData(null);
    const t = tgt ?? target;
    const c = col ?? termCol;
    const hz = h ?? horizon;
    try {
      if (tab === 'industry') {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/industry_trend`, { industry: t, horizon_months: 24 });
        setData(res.data);
      } else if (tab === 'company') {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/company_score`, { company_name: t, horizon_months: 12 });
        setData(res.data);
      } else if (tab === 'rate') {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/final_rate`, { industry: t, horizon_months: hz });
        setData(res.data);
      } else if (tab === 'compare') {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/financial_paths`, { company_name: t, n_periods: 12 });
        setData(res.data);
      } else if (tab === 'baserate') {
        const [single, all] = await Promise.all([
          axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/base_rate`, { term_col: c, horizon_months: hz }),
          axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/timesfm/base_rate_all`, { horizon_months: hz }),
        ]);
        setData(single.data);
        setAllData(all.data);
      }
    } catch (err: any) {
      setData({ error: err.response?.data?.detail || '取得に失敗しました' });
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (t: TabId) => {
    setActiveTab(t);
    setData(null);
    setAllData(null);
    const newTarget = (t === 'industry' || t === 'rate') ? '建設業' : '株式会社ABC';
    setTarget(newTarget);
    fetchData(t, newTarget);
  };

  const placeholderText = activeTab === 'company' || activeTab === 'compare'
    ? '企業名を入力...' : '業種を入力...';

  const showTargetInput = activeTab !== 'baserate';

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] bg-slate-900 rounded-3xl mt-4 mx-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-black text-white flex items-center gap-3">
          <Clock className="w-8 h-8 text-fuchsia-500" />
          Foundation Model 時系列洞察
        </h1>
        <p className="text-slate-400 font-bold mt-1">
          TimesFM（Google Research）による将来予測・基準金利シミュレーション。
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 mb-5 border-b border-slate-800 pb-4">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => handleTabChange(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-sm transition-all ${
              activeTab === tab.id
                ? 'bg-fuchsia-600 text-white shadow-lg shadow-fuchsia-900/30'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 mb-6">
        {showTargetInput && (
          <input
            type="text"
            className="px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl w-56 focus:ring-2 focus:ring-fuchsia-500 outline-none font-bold"
            value={target}
            onChange={e => setTarget(e.target.value)}
            placeholder={placeholderText}
          />
        )}
        {activeTab === 'baserate' && (
          <select
            className="px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl font-bold outline-none focus:ring-2 focus:ring-fuchsia-500"
            value={termCol}
            onChange={e => setTermCol(e.target.value)}
          >
            {TERM_COLS.map(col => (
              <option key={col} value={col}>{TERM_LABELS[col]}</option>
            ))}
          </select>
        )}
        {(activeTab === 'rate' || activeTab === 'baserate') && (
          <select
            className="px-4 py-2.5 bg-slate-800 border border-slate-700 text-white rounded-xl font-bold outline-none focus:ring-2 focus:ring-fuchsia-500"
            value={horizon}
            onChange={e => setHorizon(Number(e.target.value))}
          >
            {[3, 6, 9, 12].map(h => <option key={h} value={h}>{h}ヶ月</option>)}
          </select>
        )}
        <button
          onClick={() => fetchData(activeTab)}
          disabled={loading}
          className="px-6 py-2.5 bg-fuchsia-600 hover:bg-fuchsia-500 disabled:opacity-50 text-white font-black rounded-xl flex items-center gap-2 transition-colors"
        >
          {loading ? <Activity className="w-4 h-4 animate-spin" /> : '予測を実行'}
        </button>
      </div>

      {/* Chart area */}
      <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800/60 shadow-2xl">
        {loading ? (
          <div className="h-64 flex flex-col items-center justify-center text-slate-500">
            <Activity className="w-10 h-10 animate-spin mb-4 text-fuchsia-500" />
            <span className="text-sm uppercase tracking-widest">Computing...</span>
          </div>
        ) : data?.error ? (
          <div className="h-64 flex flex-col items-center justify-center text-rose-400 font-bold text-center">
            <span>{data.error}</span>
          </div>
        ) : !data ? (
          <div className="h-64 flex items-center justify-center text-slate-600">予測を実行してください</div>
        ) : activeTab === 'industry' ? (
          <IndustryChart data={data} />
        ) : activeTab === 'company' ? (
          <CompanyScoreChart data={data} />
        ) : activeTab === 'rate' ? (
          <RateForecastChart data={data} />
        ) : activeTab === 'compare' ? (
          <CompareChart data={data} />
        ) : activeTab === 'baserate' ? (
          <div className="space-y-8">
            <div>
              <h3 className="text-white font-black mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-indigo-400" />
                {TERM_LABELS[termCol]} の基準金利予測
              </h3>
              <BaseRateChart data={data} />
            </div>
            {allData && (
              <div>
                <h3 className="text-white font-black mb-2 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-indigo-400" />
                  全期間一括比較（実績 + 予測）
                </h3>
                <p className="text-slate-500 text-xs mb-4">実線=実績、点線=予測</p>
                <BaseRateAllChart data={allData} />
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}
