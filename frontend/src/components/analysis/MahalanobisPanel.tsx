"use client";

import React, { useState } from 'react';
import { BarChart2, ChevronDown, ChevronUp, Info, TrendingUp, TrendingDown } from 'lucide-react';

const FEAT_LABELS: Record<string, string> = {
  nenshu: '売上高',
  op_profit: '営業利益',
  ord_profit: '経常利益',
  net_income: '当期純利益',
  bank_credit: '銀行与信',
  dep_expense: '減価償却費',
  machines: '機械装置',
};

type AdviceItem = { feat: string; direction: string; delta: number };

type Props = {
  score: number;
  advice?: AdviceItem[] | null;
  compact?: boolean;
};

function getLevel(s: number): 'low' | 'mid' | 'high' {
  if (s >= 70) return 'high';
  if (s >= 40) return 'mid';
  return 'low';
}

const LEVEL_CONFIG = {
  high: {
    label: '優良',
    badge: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    bar: 'bg-emerald-400',
    wrap: 'bg-white border-slate-200',
    hdr: 'text-slate-700',
    note: '成約案件の財務プロファイルに近い構造です。',
  },
  mid: {
    label: '標準',
    badge: 'bg-blue-100 text-blue-700 border-blue-200',
    bar: 'bg-blue-400',
    wrap: 'bg-blue-50 border-blue-200',
    hdr: 'text-blue-800',
    note: '典型的な成約案件とは一部異なる財務構造があります。',
  },
  low: {
    label: '要確認',
    badge: 'bg-amber-100 text-amber-700 border-amber-200',
    bar: 'bg-amber-400',
    wrap: 'bg-amber-50 border-amber-200',
    hdr: 'text-amber-800',
    note: '過去の成約案件と財務プロファイルが大きく異なります。定性補完を推奨。',
  },
};

export default function MahalanobisPanel({ score, advice, compact = false }: Props) {
  const [expanded, setExpanded] = useState(!compact);
  const level = getLevel(score);
  const cfg = LEVEL_CONFIG[level];
  const pct = Math.min(100, Math.max(0, score));

  return (
    <div className={`rounded-xl border p-4 ${cfg.wrap}`}>
      {/* ヘッダー */}
      <div className="flex items-center gap-2 flex-wrap">
        <BarChart2 className="w-5 h-5 text-slate-500 flex-shrink-0" />
        <span className={`font-black text-sm ${cfg.hdr}`}>財務プロファイル類似度</span>
        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${cfg.badge}`}>
          {cfg.label} {score.toFixed(1)}
        </span>
        {compact && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="ml-auto text-slate-400 hover:text-slate-600"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        )}
      </div>

      {expanded && (
        <>
          {/* 説明 */}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-white/70 rounded-lg border border-slate-200">
            <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                <strong>財務プロファイル類似度とは：</strong>
                過去の成約案件1,170件の財務構造（売上・利益・設備・与信等）とマハラノビス距離で比較した類似スコアです。
                高いほど「典型的な成約企業」に近い財務プロファイルを持ちます。
              </p>
              <div className="mt-2 space-y-1">
                {[
                  { range: '70以上', label: '優良', color: 'text-emerald-700', bg: 'bg-emerald-50 border-emerald-200', note: '成約案件の財務構造に近い。審査上の追加確認は不要' },
                  { range: '40〜69', label: '標準', color: 'text-blue-700', bg: 'bg-blue-50 border-blue-200', note: '一部の財務指標が平均と乖離。改善余地あり' },
                  { range: '40未満', label: '要確認', color: 'text-amber-700', bg: 'bg-amber-50 border-amber-200', note: '財務プロファイルが典型案件と大きく異なる。定性面での補完確認を推奨' },
                ].map(row => (
                  <div key={row.range} className={`flex items-center gap-2 px-2 py-1 rounded border text-[10px] ${row.bg}`}>
                    <span className={`font-black w-12 flex-shrink-0 ${row.color}`}>{row.range}</span>
                    <span className={`font-black w-10 flex-shrink-0 ${row.color}`}>{row.label}</span>
                    <span className="text-slate-600">{row.note}</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 mt-1.5 font-bold">
                ※ 類似度は自動否決の根拠ではありません。財務構造の参考指標です。
              </p>
            </div>
          </div>

          {/* ゲージバー */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-bold">
              <span>0</span>
              <span className="text-amber-600">40 要確認</span>
              <span className="text-emerald-600">70 優良</span>
              <span>100</span>
            </div>
            <div className="relative h-3 bg-slate-100 rounded-full overflow-hidden">
              <div className="absolute inset-0 flex">
                <div className="w-[40%] bg-amber-50" />
                <div className="w-[30%] bg-blue-50" />
                <div className="w-[30%] bg-emerald-50" />
              </div>
              <div className="absolute top-0 bottom-0 left-[40%] w-px bg-amber-300" />
              <div className="absolute top-0 bottom-0 left-[70%] w-px bg-emerald-400" />
              <div
                className={`absolute top-0 left-0 h-full rounded-full transition-all duration-500 ${cfg.bar}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="text-right mt-1 text-xs font-black text-slate-500">{score.toFixed(1)} / 100</div>
          </div>

          {/* 現状コメント */}
          <p className={`mt-2 text-[11px] font-bold ${cfg.hdr}`}>{cfg.note}</p>

          {/* 改善アドバイス */}
          {advice && advice.length > 0 && (
            <div className="mt-3">
              <p className={`text-[10px] font-black uppercase tracking-widest mb-2 ${cfg.hdr}`}>
                類似度向上のヒント（上位3項目）
              </p>
              <div className="space-y-1.5">
                {advice.map((a, i) => {
                  const featLabel = FEAT_LABELS[a.feat] ?? a.feat;
                  const isUp = a.direction.includes('増');
                  return (
                    <div key={i} className="flex items-center gap-2 p-2 bg-white/80 rounded-lg border border-slate-200">
                      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${isUp ? 'bg-blue-100' : 'bg-orange-100'}`}>
                        {isUp
                          ? <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
                          : <TrendingDown className="w-3.5 h-3.5 text-orange-600" />
                        }
                      </div>
                      <div className="min-w-0">
                        <span className="text-xs font-black text-slate-700">{featLabel}</span>
                        <span className={`ml-1.5 text-[10px] font-bold ${isUp ? 'text-blue-600' : 'text-orange-600'}`}>
                          {a.direction}
                        </span>
                        {a.delta !== 0 && (
                          <span className="ml-1 text-[10px] text-slate-400">
                            ({isUp ? '+' : ''}{a.delta.toLocaleString()}千円)
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
