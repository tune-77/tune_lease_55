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
  onRegistered?: (data: CaseRegistrationResult) => void;
  onImpactCompleted?: (data: CaseRegistrationResult) => void;
};

export type CaseRegistrationStatus = '成約' | '失注';

export type CaseRegistrationResult = Record<string, unknown> & {
  registered_case_id: string;
  registered_status: CaseRegistrationStatus;
  final_rate: number;
  base_rate_at_time: number;
  competitor_rate: number;
  experience_promotion?: {
    status?: string;
    reason?: string;
  };
  prediction_error?: {
    status?: string;
    reason?: string;
  };
  followup_impact_pending?: boolean;
};

type ImpactLabel = 'decision_changed' | 'risk_prevented' | 'outcome_matched' | 'evidence_strengthened' | 'not_helpful';

type FollowupImpactSession = {
  followup_id: string;
  case_id: string;
  status: string;
  questions: Array<{ id: string; category: string; question: string }>;
  answers: Array<{ question_id: string; status: string; note?: string }>;
  impact_feedback?: Array<{ question_id: string; impact_label: ImpactLabel }>;
};

const impactOptions: Array<{ value: ImpactLabel; label: string; selectedClass: string }> = [
  { value: 'decision_changed', label: '判断・条件を変えた', selectedClass: 'border-indigo-400 bg-indigo-100 text-indigo-800' },
  { value: 'risk_prevented', label: '事故・見落とし防止', selectedClass: 'border-rose-400 bg-rose-100 text-rose-800' },
  { value: 'outcome_matched', label: '懸念が結果に表れた', selectedClass: 'border-amber-400 bg-amber-100 text-amber-800' },
  { value: 'evidence_strengthened', label: '根拠を補強', selectedClass: 'border-emerald-400 bg-emerald-100 text-emerald-800' },
  { value: 'not_helpful', label: '役立たなかった', selectedClass: 'border-slate-400 bg-slate-200 text-slate-700' },
];

// register（PC）と lease-kun（スマホウィザード）の両方から使う、成約/失注の最終結果登録フォーム。
// case_id は呼び出し側（案件検索 or 審査直後の case_id）が渡す。
export default function CaseRegistrationForm({ caseId, compact = false, onRegistered, onImpactCompleted }: Props) {
  const [status, setStatus] = useState<CaseRegistrationStatus>('成約');
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
  const [impactSession, setImpactSession] = useState<FollowupImpactSession | null>(null);
  const [lastRegistration, setLastRegistration] = useState<CaseRegistrationResult | null>(null);
  const [impactSaving, setImpactSaving] = useState('');
  const [impactError, setImpactError] = useState('');
  const [impactComplete, setImpactComplete] = useState(false);

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
    setImpactSession(null);
    setLastRegistration(null);
    setImpactError('');
    setImpactComplete(false);
    try {
      const parsedFinalRate = parseRateInput(finalRate);
      const parsedBaseRate = parseRateInput(baseRate, 2.1);
      const parsedCompetitorRate = parseRateInput(competitorRate);
      const res = await apiClient.post(`/api/cases/register`, {
        case_id: caseId,
        status: status,
        final_rate: parsedFinalRate,
        base_rate_at_time: parsedBaseRate,
        lost_reason: lostReason,
        loan_conditions: selectedConditions,
        competitor_name: competitorName,
        competitor_rate: parsedCompetitorRate,
        note: note,
        human_discomfort: humanDiscomfort,
        but_still_reason: butStillReason,
        approval_condition_memo: approvalConditionMemo,
        non_negotiable_condition: nonNegotiableCondition,
        retrospective_note: retrospectiveNote
      });
      const baseRegistrationResult: CaseRegistrationResult = {
        ...(res.data || {}),
        registered_case_id: String(res.data?.case_id || caseId),
        registered_status: status,
        final_rate: parsedFinalRate,
        base_rate_at_time: parsedBaseRate,
        competitor_rate: parsedCompetitorRate,
      };
      const responseImpactSessions = Array.isArray(res.data?.shion_followup?.impact_sessions)
        ? res.data.shion_followup.impact_sessions
        : [];
      let linkedSession: FollowupImpactSession | null = responseImpactSessions.find(
        (candidate: FollowupImpactSession) => candidate?.status?.startsWith('outcome_linked') && candidate?.answers?.length > 0,
      ) || null;
      const matches = Array.isArray(res.data?.shion_followup?.matches) ? res.data.shion_followup.matches : [];
      const linkedCaseIds = Array.from(new Set(
        matches
          .filter((match: Record<string, unknown>) => Number(match?.answered_count || 0) > 0)
          .map((match: Record<string, unknown>) => String(match?.case_id || '').trim())
          .filter(Boolean),
      ));
      for (const linkedCaseId of linkedSession ? [] : linkedCaseIds) {
        try {
          const followupResponse = await apiClient.get('/api/shion-followups', {
            params: { case_id: linkedCaseId, limit: 1 },
          });
          const candidate = Array.isArray(followupResponse.data?.followups) ? followupResponse.data.followups[0] : null;
          if (candidate?.status?.startsWith('outcome_linked') && Array.isArray(candidate?.answers) && candidate.answers.length > 0) {
            linkedSession = candidate as FollowupImpactSession;
            break;
          }
        } catch (followupError) {
          console.error('Follow-up impact prompt load failed', followupError);
        }
      }
      const answerIds = new Set((linkedSession?.answers || []).map((answer) => answer.question_id));
      const ratedIds = new Set((linkedSession?.impact_feedback || []).map((entry) => entry.question_id));
      const linkedQuestions = (linkedSession?.questions || []).filter((question) => answerIds.has(question.id));
      const alreadyComplete = linkedQuestions.length > 0 && linkedQuestions.every((question) => ratedIds.has(question.id));
      const registrationResult: CaseRegistrationResult = {
        ...baseRegistrationResult,
        followup_impact_pending: Boolean(linkedSession && !alreadyComplete),
      };
      setImpactSession(linkedSession);
      setImpactComplete(alreadyComplete);
      setLastRegistration(registrationResult);
      const promoted = registrationResult.experience_promotion?.status === 'promoted';
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
      onRegistered?.(registrationResult);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '登録に失敗しました。存在する案件か確認してください。');
    } finally {
      setSubmitting(false);
    }
  };

  const saveImpact = async (questionId: string, impactLabel: ImpactLabel) => {
    if (!impactSession || impactSaving) return;
    setImpactSaving(questionId);
    setImpactError('');
    try {
      const response = await apiClient.post(`/api/shion-followups/${impactSession.followup_id}/impact-feedback`, {
        entries: [{ question_id: questionId, impact_label: impactLabel, note: '' }],
      });
      const saved = Array.isArray(response.data?.impact_feedback) ? response.data.impact_feedback[0] : null;
      if (!saved) throw new Error('impact feedback response is empty');
      const updatedFeedback = [
        ...(impactSession.impact_feedback || []).filter((entry) => entry.question_id !== questionId),
        saved,
      ];
      const updatedSession = { ...impactSession, impact_feedback: updatedFeedback };
      setImpactSession(updatedSession);
      const answerIds = new Set(updatedSession.answers.map((answer) => answer.question_id));
      const ratingTargets = updatedSession.questions.filter((question) => answerIds.has(question.id));
      const ratedIds = new Set(updatedFeedback.map((entry) => entry.question_id));
      const completed = ratingTargets.length > 0 && ratingTargets.every((question) => ratedIds.has(question.id));
      if (completed) {
        setImpactComplete(true);
        const completedResult = lastRegistration ? { ...lastRegistration, followup_impact_pending: false } : null;
        if (completedResult) {
          setLastRegistration(completedResult);
          onImpactCompleted?.(completedResult);
        }
        triggerMebuki('approve', '追加確認の効果を保存しました。次の質問選びに使える証拠が増えました。');
      }
    } catch (requestError) {
      console.error('Follow-up impact feedback save failed', requestError);
      setImpactError('質問の効果を保存できませんでした。もう一度選んでください。');
    } finally {
      setImpactSaving('');
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

      {impactSession && (
        <div className={`${cardClass} border-violet-200 bg-violet-50/60`}>
          <h3 className={headingClass}>
            <Activity className="w-5 h-5 text-violet-600" />
            登録結果を見て、追加確認は何に役立ちましたか？
          </h3>
          <p className="mb-4 text-xs font-bold leading-5 text-violet-700">
            質問ごとの評価をその場で残します。スコアや判断資産は自動変更しません。
          </p>
          <div className="space-y-3">
            {impactSession.questions
              .filter((question) => impactSession.answers.some((answer) => answer.question_id === question.id))
              .map((question) => {
                const selected = impactSession.impact_feedback?.find((entry) => entry.question_id === question.id)?.impact_label;
                return (
                  <div key={question.id} className="rounded-2xl border border-violet-100 bg-white p-3">
                    <p className="text-xs font-black leading-5 text-slate-800">{question.category}: {question.question}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {impactOptions.map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          disabled={Boolean(impactSaving) || impactComplete}
                          onClick={() => saveImpact(question.id, option.value)}
                          className={`rounded-xl border px-3 py-2 text-[10px] font-black disabled:opacity-60 ${selected === option.value ? option.selectedClass : 'border-slate-200 bg-white text-slate-500'}`}
                        >
                          {impactSaving === question.id ? '保存中' : option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
          </div>
          {impactError && <p className="mt-3 text-xs font-black text-rose-700">{impactError}</p>}
          {impactComplete && (
            <p className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-center text-xs font-black text-emerald-700">
              すべての質問評価を保存しました。
            </p>
          )}
        </div>
      )}
    </div>
  );
}
