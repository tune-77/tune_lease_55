"use client";

import React, { useState } from 'react';
import { Activity, ChevronDown, ChevronUp, AlertTriangle, ShieldAlert, CheckCircle2, Info } from 'lucide-react';

type QRiskLevel = 'normal' | 'caution' | 'critical';

function getLevel(q: number): QRiskLevel {
  if (q >= 60) return 'critical';
  if (q >= 35) return 'caution';
  return 'normal';
}

const LEVEL_CONFIG = {
  normal: {
    label: '観測低',
    badge: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    bar: 'bg-emerald-400',
    wrap: 'bg-white border-slate-200',
    icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
    hdr: 'text-slate-700',
  },
  caution: {
    label: '探索対象',
    badge: 'bg-amber-100 text-amber-700 border-amber-200',
    bar: 'bg-amber-400',
    wrap: 'bg-amber-50 border-amber-200',
    icon: <AlertTriangle className="w-5 h-5 text-amber-500" />,
    hdr: 'text-amber-800',
  },
  critical: {
    label: '重点探索',
    badge: 'bg-rose-100 text-rose-700 border-rose-200',
    bar: 'bg-rose-500',
    wrap: 'bg-rose-50 border-rose-300',
    icon: <ShieldAlert className="w-5 h-5 text-rose-500" />,
    hdr: 'text-rose-800',
  },
};

// NEXT版: Q_risk は旧式の財務矛盾スコアではなく、成約外因子の探索軸として表示する。
const CAUTION_IMPACTS = [
  { label: '高スコア失注・低スコア成約の確認', detail: '信用スコアだけでは説明できない結果差を抽出し、価格、競合、営業導線、顧客事情を比較してください。' },
  { label: '条件提示後の離脱要因', detail: '金利、前受金、保証、期間短縮、競合条件への反応を確認し、成約を動かした非スコア因子をタグ化してください。' },
  { label: '物件・補助金・銀行支援の橋渡し', detail: '物件換金性、補助金の入金タイミング、銀行支援の具体性がスコア外の安心材料になっていないか確認してください。' },
];

const CRITICAL_IMPACTS = [
  ...CAUTION_IMPACTS,
  { label: '同スコア帯の結果分岐', detail: '同じスコア帯で成約・失注が割れる案件を並べ、営業部、業種細分、物件、提案順序、決裁者事情を比較してください。' },
  { label: '成約ストーリーの不足', detail: '稟議で通る説明、顧客の導入期限、銀行紹介の強さ、代替条件の提示有無を確認し、次回の提案条件へ戻してください。' },
];

type Props = {
  quantumRisk: number;
  creditQuantumStrongWarning?: boolean;
  compact?: boolean;
};

export default function QRiskPanel({ quantumRisk, creditQuantumStrongWarning = false, compact = false }: Props) {
  const [expanded, setExpanded] = useState(!compact);
  const level = getLevel(quantumRisk);
  const cfg = LEVEL_CONFIG[level];
  const pct = Math.min(100, Math.max(0, quantumRisk));
  const impacts = level === 'critical' ? CRITICAL_IMPACTS : CAUTION_IMPACTS;
  const showImpacts = level !== 'normal';

  return (
    <div className={`rounded-xl border p-4 ${cfg.wrap}`}>
      {/* ヘッダー */}
      <div className="flex items-center gap-2 flex-wrap">
        <Activity className="w-5 h-5 text-slate-500 flex-shrink-0" />
        <span className={`font-black text-sm ${cfg.hdr}`}>成約外因子探索（Q_risk）</span>
        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${cfg.badge}`}>
          {cfg.label} {quantumRisk.toFixed(1)}
        </span>
        {creditQuantumStrongWarning && (
          <span className="text-[10px] font-black px-2 py-0.5 rounded-full bg-rose-600 text-white">
            信用×成約外因子 分岐注意
          </span>
        )}
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
          {/* REV-113: Q_risk の説明 */}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-white/70 rounded-lg border border-slate-200">
            <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
            <p className="text-[11px] text-slate-600 leading-relaxed">
              <strong>Q_riskとは：</strong>既存スコアだけでは説明できない成約・失注の歪みを見つける探索シグナルです。
              今までの計算式や財務矛盾スコアには固定せず、高スコア失注、低スコア成約、同スコア帯の結果分岐から、価格・競合・銀行支援・補助金・物件換金性・営業導線などの非スコア因子を探します。
            </p>
          </div>

          {/* ゲージバー */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-bold">
              <span>0</span>
              <span className="text-emerald-600">35 探索対象</span>
              <span className="text-rose-600">60 重点探索</span>
              <span>100</span>
            </div>
            <div className="relative h-3 bg-slate-100 rounded-full overflow-hidden">
              {/* 段階的な背景 */}
              <div className="absolute inset-0 flex">
                <div className="w-[35%] bg-emerald-50" />
                <div className="w-[25%] bg-amber-50" />
                <div className="w-[40%] bg-rose-50" />
              </div>
              {/* 境界線 */}
              <div className="absolute top-0 bottom-0 left-[35%] w-px bg-amber-300" />
              <div className="absolute top-0 bottom-0 left-[60%] w-px bg-rose-400" />
              {/* バー */}
              <div
                className={`absolute top-0 left-0 h-full rounded-full transition-all duration-500 ${cfg.bar}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="text-right mt-1 text-xs font-black text-slate-500">{quantumRisk.toFixed(1)} / 100</div>
          </div>

          {/* REV-114: 審査影響（要注意・強警戒のみ） */}
          {showImpacts && (
            <div className="mt-3">
              <p className={`text-[10px] font-black uppercase tracking-widest mb-2 ${cfg.hdr}`}>
                {level === 'critical' ? '重点探索する非スコア因子' : '確認する非スコア因子'}
              </p>
              <div className="space-y-1.5">
                {impacts.map((imp, i) => (
                  <div key={i} className={`p-2.5 rounded-lg ${level === 'critical' ? 'bg-white border border-rose-200' : 'bg-white/80 border border-amber-200'}`}>
                    <p className={`text-[10px] font-black mb-0.5 ${level === 'critical' ? 'text-rose-700' : 'text-amber-700'}`}>{imp.label}</p>
                    <p className="text-xs text-slate-600 leading-relaxed">{imp.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {level === 'normal' && (
            <p className="mt-2 text-[11px] text-emerald-600 font-bold">
              ✓ 既存スコア外の成約分岐シグナルは低めです。ただし、Q_riskは減点ではなく探索対象の優先度です。
            </p>
          )}
        </>
      )}
    </div>
  );
}
