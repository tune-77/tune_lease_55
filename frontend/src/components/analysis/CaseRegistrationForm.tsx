"use client";
import { useState } from "react";
import { apiClient } from "../../lib/api";
import { triggerMebuki } from "../layout/FloatingMebuki";
import { CheckCircle, XCircle, FileText, Activity, Save, Percent, Building2, TrendingDown } from "lucide-react";

export const conditionOptions = ["本件限度", "次回決算まで本件限度", "金融機関と協調", "独立・新設向け条件", "親会社等保証", "担保・保全あり", "その他"];

export const parseRateInput = (value: string, fallback = 0.0) => {
  const normalized = value.trim().replace(',', '.');
  if (!normalized) return fallback;
  const parsed = Number.parseFloat(normalized);
  return Number.isFinite(parsed) ? parsed : fallback;
};

type Props = {
  caseId: string;
  compact?: boolean;
  onRegistered?: (data: any) => void;
};

// register（PC）と lease-kun（スマホウィザード）の両方から使う、成約/失注の最終結果登録フォーム。
// case_id は呼び出し側（案件検索 or 審査直後の case_id）が渡す。
export default function CaseRegistrationForm({ caseId, compact = false, onRegistered }: Props) {
  const [status, setStatus] = useState<'成約' | '失注'>('成約');
  const [finalRate, setFinalRate] = useState('0.0');
  const [baseRate, setBaseRate] = useState('2.1');
  const [lostReason, setLostReason] = useState('');
  const [competitorName, setCompetitorName] = useState('');
  const [competitorRate, setCompetitorRate] = useState('0.0');
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [humanDiscomfort, setHumanDiscomfort] = useState('');
  const [butStillReason, setButStillReason] = useState('');
  const [approvalConditionMemo, setApprovalConditionMemo] = useState('');
  const [nonNegotiableCondition, setNonNegotiableCondition] = useState('');
  const [retrospectiveNote, setRetrospectiveNote] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const toggleCondition = (opt: string) => {
    setSelectedConditions(prev =>
      prev.includes(opt) ? prev.filter(c => c !== opt) : [...prev, opt]
    );
  };

  const handleRegister = async () => {
    if (!caseId) {
      triggerMebuki('challenge', '先に案件を選択してくださいね。');
      return;
    }
    setSubmitting(true);
    try {
      const res = await apiClient.post(`/api/cases/register`, {
        case_id: caseId,
        status: status,
        final_rate: parseRateInput(finalRate),
        base_rate_at_time: parseRateInput(baseRate, 2.1),
        lost_reason: lostReason,
        loan_conditions: selectedConditions,
        competitor_name: competitorName,
        competitor_rate: parseRateInput(competitorRate),
        note: note,
        human_discomfort: humanDiscomfort,
        but_still_reason: butStillReason,
        approval_condition_memo: approvalConditionMemo,
        non_negotiable_condition: nonNegotiableCondition,
        retrospective_note: retrospectiveNote
      });
      const promoted = res.data?.experience_promotion?.status === 'promoted';
      triggerMebuki(
        'approve',
        `${caseId} の結果を登録しました！${promoted ? '\n経験ケースにも自動昇格しました。' : ''}`
      );
      setFinalRate('0.0');
      setBaseRate('2.1');
      setLostReason('');
      setCompetitorName('');
      setCompetitorRate('0.0');
      setSelectedConditions([]);
      setNote('');
      setHumanDiscomfort('');
      setButStillReason('');
      setApprovalConditionMemo('');
      setNonNegotiableCondition('');
      setRetrospectiveNote('');
      onRegistered?.(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '登録に失敗しました。存在する案件か確認してください。');
    } finally {
      setSubmitting(false);
    }
  };

  const gridCols = compact ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2";
  const cardClass = compact
    ? "bg-white border border-slate-200 rounded-2xl shadow-sm p-4"
    : "bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8";
  const headingClass = compact
    ? "text-sm font-black text-slate-700 mb-4 flex items-center gap-2"
    : "text-lg font-black text-slate-700 mb-6 flex items-center gap-2";

  return (
    <div className={compact ? "space-y-4" : "space-y-6"}>
      <div className={cardClass}>
        <div className="flex gap-4">
          <button
            onClick={() => setStatus('成約')}
            className={`flex-1 p-4 rounded-2xl border-2 transition-all flex flex-col items-center gap-2 ${status === '成約' ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-white border-slate-100 text-slate-400'}`}
          >
            <CheckCircle className={`w-6 h-6 ${status === '成約' ? 'text-emerald-500' : 'text-slate-300'}`} />
            <span className="font-black text-sm">成約 (WIN)</span>
          </button>
          <button
            onClick={() => setStatus('失注')}
            className={`flex-1 p-4 rounded-2xl border-2 transition-all flex flex-col items-center gap-2 ${status === '失注' ? 'bg-rose-50 border-rose-500 text-rose-700' : 'bg-white border-slate-100 text-slate-400'}`}
          >
            <XCircle className={`w-6 h-6 ${status === '失注' ? 'text-rose-500' : 'text-slate-300'}`} />
            <span className="font-black text-sm">失注 (LOST)</span>
          </button>
        </div>
      </div>

      <div className={cardClass}>
        <h3 className={headingClass}>
          <Percent className="w-5 h-5 text-emerald-500" />
          金利・レート情報
        </h3>
        <div className={`grid ${gridCols} gap-4`}>
          <div>
            <label className="block text-xs font-black text-slate-400 uppercase mb-2">最終獲得レート (%)</label>
            <input
              type="text" inputMode="decimal" value={finalRate}
              onChange={(e) => setFinalRate(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-emerald-600 outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-black text-slate-400 uppercase mb-2">当時の基準金利 (%)</label>
            <input
              type="text" inputMode="decimal" value={baseRate}
              onChange={(e) => setBaseRate(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-slate-600 outline-none"
            />
          </div>
        </div>
      </div>

      <div className={cardClass}>
        <h3 className={headingClass}>
          <Building2 className="w-5 h-5 text-orange-500" />
          競合・失注分析
        </h3>
        <div className="space-y-4">
          <div className={`grid ${gridCols} gap-4`}>
            <div>
              <label className="block text-xs font-black text-slate-400 uppercase mb-2">競合他社名</label>
              <input
                type="text" value={competitorName}
                onChange={(e) => setCompetitorName(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-slate-700 outline-none"
                placeholder="〇〇銀行など"
              />
            </div>
            <div>
              <label className="block text-xs font-black text-slate-400 uppercase mb-2">他社提示レート (%)</label>
              <input
                type="text" inputMode="decimal" value={competitorRate}
                onChange={(e) => setCompetitorRate(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-orange-600 outline-none"
              />
            </div>
          </div>
          {status === '失注' && (
            <div className="animate-in slide-in-from-top-2 duration-300 space-y-3">
              <div>
                <label className="block text-xs font-black text-rose-500 uppercase mb-2 flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" /> 失注理由
                </label>
                <textarea
                  className="w-full bg-rose-50/30 border border-rose-100 p-4 rounded-xl text-sm font-bold text-rose-700 outline-none h-20"
                  value={lostReason}
                  onChange={(e) => setLostReason(e.target.value)}
                  placeholder="金利競合で敗退、あるいは条件不一致など..."
                />
              </div>
              {!lostReason && (
                <div className="flex items-start gap-2 rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-700 font-bold animate-in slide-in-from-top-2 duration-300">
                  <span>💡</span>
                  <span>失注理由を入力すると営業分析の精度が上がります（任意）</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className={cardClass}>
        <h3 className={headingClass}>
          <CheckCircle className="w-5 h-5 text-indigo-500" />
          成約/承認の付帯条件
        </h3>
        <div className="flex flex-wrap gap-2">
          {conditionOptions.map(opt => (
            <button
              key={opt}
              onClick={() => toggleCondition(opt)}
              className={`px-4 py-2 rounded-xl text-xs font-black border-2 transition-all ${selectedConditions.includes(opt) ? 'bg-indigo-600 border-indigo-600 text-white shadow-lg shadow-indigo-200' : 'bg-white border-slate-100 text-slate-400 hover:border-slate-300'}`}
            >
              {opt}
            </button>
          ))}
        </div>
      </div>

      <div className={cardClass}>
        <h3 className={compact ? "text-sm font-black text-slate-700 mb-2 flex items-center gap-2" : "text-lg font-black text-slate-700 mb-2 flex items-center gap-2"}>
          <FileText className="w-5 h-5 text-violet-500" />
          グレー判断メモ
        </h3>
        <p className="text-xs font-bold text-slate-400 mb-5">
          数字だけでは割り切れなかった判断を、紫苑が次回の稟議相談で優先参照します。
        </p>
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-black text-slate-400 uppercase mb-2">違和感・気になる点</label>
            <textarea
              className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl text-sm font-bold text-slate-700 outline-none min-h-[76px]"
              value={humanDiscomfort}
              onChange={(e) => setHumanDiscomfort(e.target.value)}
              placeholder="数字は悪くないが、資金使途の説明が弱い。代表者対応に少し違和感がある、など"
            />
          </div>
          <div>
            <label className="block text-xs font-black text-slate-400 uppercase mb-2">それでも通す/断る理由</label>
            <textarea
              className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl text-sm font-bold text-slate-700 outline-none min-h-[76px]"
              value={butStillReason}
              onChange={(e) => setButStillReason(e.target.value)}
              placeholder="メイン行支援が強い、既存取引で回収実績がある、逆にここは譲れない、など"
            />
          </div>
          <div className={`grid ${gridCols} gap-4`}>
            <div>
              <label className="block text-xs font-black text-slate-400 uppercase mb-2">通すなら条件</label>
              <textarea
                className="w-full bg-indigo-50/40 border border-indigo-100 p-4 rounded-xl text-sm font-bold text-indigo-800 outline-none min-h-[88px]"
                value={approvalConditionMemo}
                onChange={(e) => setApprovalConditionMemo(e.target.value)}
                placeholder="保証、資料徴求、限度、見積確認、金融機関連携など"
              />
            </div>
            <div>
              <label className="block text-xs font-black text-slate-400 uppercase mb-2">譲れない線</label>
              <textarea
                className="w-full bg-rose-50/40 border border-rose-100 p-4 rounded-xl text-sm font-bold text-rose-800 outline-none min-h-[88px]"
                value={nonNegotiableCondition}
                onChange={(e) => setNonNegotiableCondition(e.target.value)}
                placeholder="この資料が出ないなら否決、この金額以上は不可、など"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs font-black text-slate-400 uppercase mb-2">振り返り</label>
            <textarea
              className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl text-sm font-bold text-slate-700 outline-none min-h-[76px]"
              value={retrospectiveNote}
              onChange={(e) => setRetrospectiveNote(e.target.value)}
              placeholder="結果を見て次回に残したい教訓。見立てが当たった/外れた理由など"
            />
          </div>
        </div>
      </div>

      <div className={cardClass}>
        <h3 className={compact ? "text-sm font-black text-slate-700 mb-4 flex items-center gap-2" : "text-lg font-black text-slate-700 mb-4 flex items-center gap-2"}>
          <FileText className="w-5 h-5 text-slate-400" />
          備考・メモ
        </h3>
        <textarea
          className="w-full bg-slate-50 border border-slate-200 p-6 rounded-2xl text-sm text-slate-700 outline-none focus:ring-2 focus:ring-slate-500/10 min-h-[100px]"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="その他、特筆すべき事項があれば入力してください"
        />

        <div className={compact ? "mt-6 flex justify-end" : "mt-8 flex justify-end"}>
          <button
            onClick={handleRegister}
            disabled={submitting}
            className={`${compact ? 'py-4 px-8 text-base' : 'py-5 px-16 text-lg'} rounded-[2rem] shadow-2xl transition-all flex items-center gap-3 font-black ${status === '成約' ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-500/30' : 'bg-rose-600 hover:bg-rose-500 shadow-rose-500/30'} text-white group disabled:opacity-60`}
          >
            {submitting ? <Activity className="w-6 h-6 animate-spin" /> : <Save className="w-6 h-6 group-hover:scale-110 transition-transform" />}
            最終結果をデータベースへ書き込む
          </button>
        </div>
      </div>
    </div>
  );
}
