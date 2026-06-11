import React from 'react';
import { Database, ShieldCheck, Target, TrendingUp } from 'lucide-react';

type IndicatorData = {
  score_base?: number;
  user_op_margin?: number;
  bench_op_margin?: number;
  user_equity_ratio?: number;
  bench_equity_ratio?: number;
  estat_context?: {
    summary?: string;
    status?: string;
    score?: number;
    score_components?: {
      industry_gap_score?: number;
      lease_fit_score?: number;
      macro_cycle_score?: number;
    };
    recommendations?: string[];
    dimensions?: Array<{
      label?: string;
      status?: string;
      score?: number;
      summary?: string;
      comment?: string;
      metrics?: Record<string, number | string | null>;
    }>;
  };
};

interface IndicatorCardsProps {
  data: IndicatorData | null;
}

export default function IndicatorCards({ data }: IndicatorCardsProps) {
  if (!data) return null;

  const opMargin = data.user_op_margin ?? 0;
  const benchOpMargin = data.bench_op_margin ?? 0;
  const equityRatio = data.user_equity_ratio ?? 0;
  const benchEquityRatio = data.bench_equity_ratio ?? 0;
  const isApproved = (data.score_base ?? 0) >= 71;

  // 判定用のバッジカラー関数
  const isNegativeRisk = (label: string, userVal: number) => {
    return ['営業利益率', '自己資本比率', 'ROA', 'ROE', '経常利益率', '当期純利益率', '売上高総利益率'].includes(label) && userVal < 0;
  };

  const isFavorable = (label: string, userVal: number, benchVal: number, isInverted = false) => {
    if (isNegativeRisk(label, userVal)) return false;
    const diff = userVal - benchVal;
    return isInverted ? diff < 0 : diff > 0;
  };

  const getBadgeColor = (label: string, userVal: number, benchVal: number, isInverted = false) => {
    if (isNegativeRisk(label, userVal)) return 'bg-rose-50 text-rose-700 border-rose-200';
    const diff = userVal - benchVal;
    if (diff === 0) return 'bg-slate-100 text-slate-600 border-slate-200';
    
    // 良い場合 (isInverted = true のときはマイナスが良い)
    const isGood = isFavorable(label, userVal, benchVal, isInverted);
    return isGood ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
      
      {/* 営業利益率 */}
      <div className="bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-3xl p-6 shadow-sm hover:shadow-md transition">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
              <TrendingUp className="w-4 h-4" />
            </div>
            <span className="font-bold text-slate-700">営業利益率</span>
          </div>
          <span className={`text-[10px] font-bold px-2 py-1 rounded-md border ${getBadgeColor('営業利益率', opMargin, benchOpMargin)}`}>
            {isFavorable('営業利益率', opMargin, benchOpMargin) ? '優良' : '注意'}
          </span>
        </div>
        
        <div className="flex items-end gap-2 mb-2">
          <span className="text-4xl font-black text-slate-800">
            {opMargin.toFixed(1)}
          </span>
          <span className="text-lg font-bold text-slate-400 mb-1">%</span>
        </div>
        
        <div className="relative w-full h-2 bg-slate-200 rounded-full mt-4 overflow-hidden">
          <div 
            className="absolute top-0 left-0 h-full bg-blue-500 rounded-full"
            style={{ width: `${Math.min(100, Math.max(0, (opMargin + 20) * 2.5))}%` }} 
          />
          {/* ベンチマークライン */}
          <div 
            className="absolute top-0 w-1 h-full bg-slate-800"
            style={{ left: `${Math.min(100, Math.max(0, (benchOpMargin + 20) * 2.5))}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs font-semibold text-slate-500">
          <span>当社</span>
          <span>業界: {benchOpMargin.toFixed(1)}%</span>
        </div>
      </div>

      {/* 自己資本比率 */}
      <div className="bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-3xl p-6 shadow-sm hover:shadow-md transition">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-indigo-100 text-indigo-600 rounded-lg">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <span className="font-bold text-slate-700">自己資本比率</span>
          </div>
          <span className={`text-[10px] font-bold px-2 py-1 rounded-md border ${getBadgeColor('自己資本比率', equityRatio, benchEquityRatio)}`}>
            {isFavorable('自己資本比率', equityRatio, benchEquityRatio) ? '安定' : '過少資本'}
          </span>
        </div>
        
        <div className="flex items-end gap-2 mb-2">
          <span className="text-4xl font-black text-slate-800">
            {equityRatio.toFixed(1)}
          </span>
          <span className="text-lg font-bold text-slate-400 mb-1">%</span>
        </div>
        
        <div className="relative w-full h-2 bg-slate-200 rounded-full mt-4 overflow-hidden">
          <div 
            className="absolute top-0 left-0 h-full bg-indigo-500 rounded-full"
            style={{ width: `${Math.min(100, Math.max(0, equityRatio))}%` }} 
          />
          {/* ベンチマークライン */}
          <div 
            className="absolute top-0 w-1 h-full bg-slate-800"
            style={{ left: `${Math.min(100, Math.max(0, benchEquityRatio))}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs font-semibold text-slate-500">
          <span>当社</span>
          <span>業界: {benchEquityRatio.toFixed(1)}%</span>
        </div>
      </div>

      {/* ベーススコア */}
      <div className={`border-2 rounded-3xl p-6 shadow-lg transition transform hover:-translate-y-1 ${
        isApproved 
          ? 'bg-gradient-to-br from-emerald-500 to-teal-600 border-emerald-400 shadow-emerald-200' 
          : 'bg-gradient-to-br from-rose-500 to-rose-600 border-rose-400 shadow-rose-200'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-white/20 text-white rounded-lg">
              <Target className="w-4 h-4" />
            </div>
            <span className="font-bold text-white/90">ベーススコア</span>
          </div>
          <span className="text-[10px] font-bold px-2 py-1 rounded-md bg-white/20 text-white shadow-inner">
            補正前
          </span>
        </div>
        
        <div className="flex items-end gap-2 mb-1">
          <span className="text-5xl font-black text-white drop-shadow-md">
            {(data.score_base ?? 0).toFixed(1)}
          </span>
          <span className="text-lg font-bold text-white/70 mb-1">点</span>
        </div>
        <div className="text-xs font-medium text-white/80 mt-2">
          {isApproved ? '承認ラインをクリアしています🎉' : '承認ライン(71点)に達していません⚠️'}
        </div>
      </div>

      {/* e-Stat統合コンテキスト */}
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 border border-slate-700 rounded-3xl p-6 shadow-lg shadow-slate-200/20 text-white">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-white/10 text-cyan-200 rounded-lg">
              <Database className="w-4 h-4" />
            </div>
            <span className="font-bold text-white/90">e-Stat統合</span>
          </div>
          <span className={`text-[10px] font-bold px-2 py-1 rounded-md border ${
            data.estat_context?.status === 'green'
              ? 'bg-emerald-400/15 text-emerald-200 border-emerald-400/30'
              : data.estat_context?.status === 'red'
                ? 'bg-rose-400/15 text-rose-200 border-rose-400/30'
                : 'bg-amber-400/15 text-amber-200 border-amber-400/30'
          }`}>
            {data.estat_context?.status === 'green' ? '整合良好' : data.estat_context?.status === 'red' ? '要確認' : '参考'}
          </span>
        </div>

        <div className="mb-3">
          <div className="text-[10px] font-black uppercase tracking-widest text-white/50">総合（3項目合成点）</div>
          <div className="mt-1 text-sm font-bold text-white/90 leading-relaxed">
            {data.estat_context?.summary || '業種・リース・景気の参照情報です。'}
          </div>
          <div className="mt-1 text-[11px] text-white/60 leading-relaxed">
            100点満点で、同業平均との差・リース負担の重さ・景気の向きを合成した参考点です。
          </div>
        </div>

        <div className="flex items-end gap-2 mb-3">
          <span className="text-4xl font-black text-white drop-shadow-md">
            {(data.estat_context?.score ?? 50).toFixed(1)}
          </span>
          <span className="text-lg font-bold text-white/70 mb-1">点</span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {[
            ['同業差', data.estat_context?.score_components?.industry_gap_score],
            ['リース負担', data.estat_context?.score_components?.lease_fit_score],
            ['景気の向き', data.estat_context?.score_components?.macro_cycle_score],
          ].map(([label, value]) => (
            <div key={label as string} className="rounded-2xl bg-white/10 border border-white/10 px-3 py-2">
              <div className="text-[10px] font-black text-white/50">{label}</div>
              <div className="text-sm font-black text-white">{typeof value === 'number' ? `${Number(value).toFixed(1)}` : '—'}</div>
            </div>
          ))}
        </div>

        {!!data.estat_context?.recommendations?.length && (
          <div className="mt-3 rounded-2xl bg-white/10 border border-white/10 p-3">
            <div className="text-[10px] font-black uppercase tracking-widest text-white/50 mb-1">示唆</div>
            <div className="text-xs leading-relaxed text-white/85">
              {data.estat_context.recommendations[0]}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
