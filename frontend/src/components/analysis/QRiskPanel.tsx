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
    label: '通常',
    badge: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    bar: 'bg-emerald-400',
    wrap: 'bg-white border-slate-200',
    icon: <CheckCircle2 className="w-5 h-5 text-emerald-500" />,
    hdr: 'text-slate-700',
  },
  caution: {
    label: '要注意',
    badge: 'bg-amber-100 text-amber-700 border-amber-200',
    bar: 'bg-amber-400',
    wrap: 'bg-amber-50 border-amber-200',
    icon: <AlertTriangle className="w-5 h-5 text-amber-500" />,
    hdr: 'text-amber-800',
  },
  critical: {
    label: '強警戒',
    badge: 'bg-rose-100 text-rose-700 border-rose-200',
    bar: 'bg-rose-500',
    wrap: 'bg-rose-50 border-rose-300',
    icon: <ShieldAlert className="w-5 h-5 text-rose-500" />,
    hdr: 'text-rose-800',
  },
};

// REV-114: Q_risk が高い場合の審査影響
const CAUTION_IMPACTS = [
  { label: '財務整合性の再確認', detail: '3期分の財務諸表（PL・BS・CF計算書）の数値整合性を手動でクロスチェックしてください。' },
  { label: '減価償却・固定資産の矛盾確認', detail: '減価償却費と固定資産残高の推移に矛盾がないか確認。急激な減少・増加がある場合は詳細説明を要求。' },
  { label: '営業CFと利益の乖離確認', detail: '営業キャッシュフローと営業利益の乖離が大きい場合は、売掛金回収状況や棚卸評価の適正性を確認。' },
];

const CRITICAL_IMPACTS = [
  ...CAUTION_IMPACTS,
  { label: '信用リスク群との複合警戒', detail: '信用リスクスコアが高水準かつQ_riskも高水準です。財務データの信頼性に重大な疑義がある可能性があります。第三者機関（公認会計士等）の確認を推奨します。' },
  { label: '追加書類の強制要求', detail: '勘定科目内訳書・補助元帳・仕入先別明細などの補助資料提出を必須条件として設定してください。' },
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
        <span className={`font-black text-sm ${cfg.hdr}`}>量子干渉リスク（Q_risk）</span>
        <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${cfg.badge}`}>
          {cfg.label} {quantumRisk.toFixed(1)}
        </span>
        {creditQuantumStrongWarning && (
          <span className="text-[10px] font-black px-2 py-0.5 rounded-full bg-rose-600 text-white">
            信用×Q_risk 強警戒
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
          {/* REV-072/073/102: Q_risk の説明 + 解釈ガイド */}
          <div className="mt-3 flex items-start gap-2 p-2.5 bg-white/70 rounded-lg border border-slate-200">
            <Info className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                <strong>Q_riskとは：</strong>財務データの内部矛盾（利益率・減価償却・借入金・キャッシュフローの整合性）を量子干渉アルゴリズムで検出するリスクスコアです。
                値が高いほど財務データに矛盾・異常が多いことを示し、スコアリング精度に影響する場合があります。
              </p>
              {/* REV-072/073: 解釈ガイド */}
              <div className="mt-2 space-y-1">
                {[
                  { range: '0〜34', label: '通常', color: 'text-emerald-700', bg: 'bg-emerald-50 border-emerald-200', note: '財務整合性に問題なし。スコアリング精度は高い' },
                  { range: '35〜59', label: '要注意', color: 'text-amber-700', bg: 'bg-amber-50 border-amber-200', note: '軽微な矛盾あり。財務書類の精査を推奨' },
                  { range: '60以上', label: '強警戒', color: 'text-rose-700', bg: 'bg-rose-50 border-rose-200', note: '重大な矛盾の可能性。追加書類での確認が必要' },
                ].map(row => (
                  <div key={row.range} className={`flex items-center gap-2 px-2 py-1 rounded border text-[10px] ${row.bg}`}>
                    <span className={`font-black w-12 flex-shrink-0 ${row.color}`}>{row.range}</span>
                    <span className={`font-black w-10 flex-shrink-0 ${row.color}`}>{row.label}</span>
                    <span className="text-slate-600">{row.note}</span>
                  </div>
                ))}
              </div>
              {/* REV-102: 重要な補足 */}
              <p className="text-[10px] text-slate-500 mt-1.5 font-bold">
                ※ Q_riskは自動否決の指標ではなく、財務データの信頼性を示すフラグです。高値でも定性確認で補完できる場合があります。
              </p>
            </div>
          </div>

          {/* ゲージバー */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] text-slate-400 mb-1 font-bold">
              <span>0</span>
              <span className="text-emerald-600">35 要注意</span>
              <span className="text-rose-600">60 強警戒</span>
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
                {level === 'critical' ? '⚠ 審査上の必須確認事項' : '審査上の確認ポイント'}
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
              ✓ 財務データの内部矛盾は検出されませんでした。
            </p>
          )}
        </>
      )}
    </div>
  );
}
