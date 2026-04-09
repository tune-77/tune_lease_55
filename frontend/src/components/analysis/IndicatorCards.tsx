import React from 'react';
import { Target, TrendingUp, AlertTriangle, ShieldCheck } from 'lucide-react';

interface IndicatorCardsProps {
  data: any;
}

export default function IndicatorCards({ data }: IndicatorCardsProps) {
  if (!data) return null;

  const isApproved = data.score_base >= 71;

  // 判定用のバッジカラー関数
  const getBadgeColor = (userVal: number, benchVal: number, isInverted = false) => {
    const diff = userVal - benchVal;
    if (diff === 0) return 'bg-slate-100 text-slate-600 border-slate-200';
    
    // 良い場合 (isInverted = true のときはマイナスが良い)
    const isGood = isInverted ? diff < 0 : diff > 0;
    return isGood ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      
      {/* 営業利益率 */}
      <div className="bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-3xl p-6 shadow-sm hover:shadow-md transition">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
              <TrendingUp className="w-4 h-4" />
            </div>
            <span className="font-bold text-slate-700">営業利益率</span>
          </div>
          <span className={`text-[10px] font-bold px-2 py-1 rounded-md border ${getBadgeColor(data.user_op_margin, data.bench_op_margin)}`}>
            {data.user_op_margin >= data.bench_op_margin ? '優良' : '注意'}
          </span>
        </div>
        
        <div className="flex items-end gap-2 mb-2">
          <span className="text-4xl font-black text-slate-800">
            {(data.user_op_margin ?? 0).toFixed(1)}
          </span>
          <span className="text-lg font-bold text-slate-400 mb-1">%</span>
        </div>
        
        <div className="relative w-full h-2 bg-slate-200 rounded-full mt-4 overflow-hidden">
          <div 
            className="absolute top-0 left-0 h-full bg-blue-500 rounded-full"
            style={{ width: `${Math.min(100, Math.max(0, (data.user_op_margin + 20) * 2.5))}%` }} 
          />
          {/* ベンチマークライン */}
          <div 
            className="absolute top-0 w-1 h-full bg-slate-800"
            style={{ left: `${Math.min(100, Math.max(0, (data.bench_op_margin + 20) * 2.5))}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs font-semibold text-slate-500">
          <span>当社</span>
          <span>業界: {(data.bench_op_margin ?? 0).toFixed(1)}%</span>
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
          <span className={`text-[10px] font-bold px-2 py-1 rounded-md border ${getBadgeColor(data.user_equity_ratio, data.bench_equity_ratio)}`}>
            {data.user_equity_ratio >= data.bench_equity_ratio ? '安定' : '過少資本'}
          </span>
        </div>
        
        <div className="flex items-end gap-2 mb-2">
          <span className="text-4xl font-black text-slate-800">
            {(data.user_equity_ratio ?? 0).toFixed(1)}
          </span>
          <span className="text-lg font-bold text-slate-400 mb-1">%</span>
        </div>
        
        <div className="relative w-full h-2 bg-slate-200 rounded-full mt-4 overflow-hidden">
          <div 
            className="absolute top-0 left-0 h-full bg-indigo-500 rounded-full"
            style={{ width: `${Math.min(100, Math.max(0, data.user_equity_ratio))}%` }} 
          />
          {/* ベンチマークライン */}
          <div 
            className="absolute top-0 w-1 h-full bg-slate-800"
            style={{ left: `${Math.min(100, Math.max(0, data.bench_equity_ratio))}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs font-semibold text-slate-500">
          <span>当社</span>
          <span>業界: {(data.bench_equity_ratio ?? 0).toFixed(1)}%</span>
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

    </div>
  );
}
