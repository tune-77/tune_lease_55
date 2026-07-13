"use client";

import React, { useState } from 'react';
import { Activity, ChevronDown, ChevronUp, AlertTriangle, ShieldAlert, CheckCircle2, Info, Loader2 } from 'lucide-react';
import { apiClient } from '../../lib/api';

type QRiskLevel = 'normal' | 'caution' | 'critical';

type FinancialConsistencyRisk = {
  score?: number;
  level?: string;
  patterns?: string[];
  pattern_details?: Array<{
    code?: string;
    severity?: string;
    message?: string;
    values?: Record<string, unknown>;
  }>;
  role?: string;
};

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
  caseId?: string;
  score?: number | null;
  hantei?: string;
  context?: Record<string, unknown>;
  financialConsistencyRisk?: FinancialConsistencyRisk | null;
};

export default function QRiskPanel({
  quantumRisk,
  creditQuantumStrongWarning = false,
  compact = false,
  caseId = '',
  score = null,
  hantei = '',
  context = {},
  financialConsistencyRisk = null,
}: Props) {
  const [expanded, setExpanded] = useState(!compact);
  const [savingKey, setSavingKey] = useState('');
  const [savedKey, setSavedKey] = useState('');
  const [saveError, setSaveError] = useState('');
  const level = getLevel(quantumRisk);
  const cfg = LEVEL_CONFIG[level];
  const pct = Math.min(100, Math.max(0, quantumRisk));
  const impacts = level === 'critical' ? CRITICAL_IMPACTS : CAUTION_IMPACTS;
  const showImpacts = level !== 'normal';
  const oldRiskScore = Number(financialConsistencyRisk?.score ?? 0);
  const oldRiskPatterns = financialConsistencyRisk?.pattern_details ?? [];
  const oldRiskLevelLabel =
    financialConsistencyRisk?.level === 'high_risk' ? '強警戒' :
    financialConsistencyRisk?.level === 'caution' ? '要確認' :
    '低位';

  const saveInterpretation = async (kind: 'sales_competition' | 'credit_warning') => {
    const isSales = kind === 'sales_competition';
    const rating = isSales ? '信用ではなく成約/競合リスク' : '信用リスクとして警戒';
    const issue = isSales
      ? '業績や信用力そのものではなく、競合条件・金利・営業導線・銀行支援の弱さで成約しにくい案件として扱う。'
      : 'Q_riskを信用面の警戒信号として扱い、財務整合性・返済原資・資料確認を優先する。';
    const policy = isSales
      ? 'スコアが低くても直ちに信用否定せず、競合金利・銀行支援・提案速度・条件変更余地を確認する。'
      : '成約外因子だけで押さず、財務・物件・支援内容を確認条件に置く。';
    setSavingKey(kind);
    setSavedKey('');
    setSaveError('');
    try {
      await apiClient.post('/api/screening-loop-feedback', {
        surface: 'q_risk_panel',
        target: 'issue',
        rating,
        issue_text: issue,
        ringi_policy_text: policy,
        comment: `Q_risk ${quantumRisk.toFixed(1)}: ${rating}`,
        score: typeof score === 'number' ? score : null,
        hantei,
        context: {
          ...context,
          case_id: caseId,
          q_risk: quantumRisk,
          financial_consistency_score: oldRiskScore,
          financial_consistency_patterns: financialConsistencyRisk?.patterns ?? [],
          q_risk_feedback_kind: kind,
          judgment_asset_intent: 'separate_credit_risk_from_sales_competition_risk',
        },
      });
      setSavedKey(kind);
    } catch {
      setSaveError('保存できませんでした');
    } finally {
      setSavingKey('');
    }
  };

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

          {financialConsistencyRisk && (
            <div className="mt-3 rounded-lg border border-slate-200 bg-white/80 p-2.5">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-3.5 w-3.5 text-slate-400" />
                <p className="text-[11px] font-black text-slate-700">財務整合性チェック（旧Q_risk）</p>
                <span className={`ml-auto rounded-full border px-2 py-0.5 text-[10px] font-black ${
                  oldRiskScore >= 50
                    ? 'border-rose-200 bg-rose-50 text-rose-700'
                    : oldRiskScore >= 20
                      ? 'border-amber-200 bg-amber-50 text-amber-700'
                      : 'border-emerald-200 bg-emerald-50 text-emerald-700'
                }`}>
                  {oldRiskLevelLabel} {oldRiskScore.toFixed(0)}
                </span>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                入力された財務項目同士のつじつまを確認します。成約外因子ではなく、粗利率・債務/年商・設備と償却などの数字の整合性を見る補助指標です。
              </p>
              {oldRiskPatterns.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {oldRiskPatterns.slice(0, 3).map((item, index) => (
                    <li key={`${item.code || 'old-q'}-${index}`} className="rounded-md bg-slate-50 px-2 py-1 text-[11px] text-slate-600">
                      <span className="font-black text-slate-700">{item.code || 'FIN-CHECK'}:</span> {item.message || '財務整合性の確認が必要です'}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="mt-3 rounded-lg border border-slate-200 bg-white/80 p-2.5">
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">
              人間の判断として保存
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => saveInterpretation('sales_competition')}
                disabled={Boolean(savingKey)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[11px] font-black transition ${
                  savedKey === 'sales_competition'
                    ? 'border-blue-200 bg-blue-50 text-blue-800'
                    : 'border-slate-200 bg-white text-slate-700 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-800'
                } disabled:opacity-50`}
              >
                {savingKey === 'sales_competition' && <Loader2 className="h-3 w-3 animate-spin" />}
                {savedKey === 'sales_competition' ? '保存済' : '信用ではなく競合/成約リスク'}
              </button>
              <button
                type="button"
                onClick={() => saveInterpretation('credit_warning')}
                disabled={Boolean(savingKey)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[11px] font-black transition ${
                  savedKey === 'credit_warning'
                    ? 'border-rose-200 bg-rose-50 text-rose-800'
                    : 'border-slate-200 bg-white text-slate-700 hover:border-rose-200 hover:bg-rose-50 hover:text-rose-800'
                } disabled:opacity-50`}
              >
                {savingKey === 'credit_warning' && <Loader2 className="h-3 w-3 animate-spin" />}
                {savedKey === 'credit_warning' ? '保存済' : '信用リスクとして警戒'}
              </button>
            </div>
            {saveError && <p className="mt-2 text-[11px] font-bold text-rose-600">{saveError}</p>}
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
