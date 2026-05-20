"use client";
import React, { useState, useEffect } from 'react';
import { apiClient } from '../../lib/api';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { CheckCircle, XCircle, FileText, Activity, Save, Search, User, Percent, Building2, ClipboardList, TrendingDown, Trash2 } from 'lucide-react';

const conditionOptions = ["本件限度", "次回決算まで本件限度", "金融機関と協調", "独立・新設向け条件", "親会社等保証", "担保・保全あり", "その他"];

export default function RegisterPage() {
  const [targetId, setTargetId] = useState('');
  const [status, setStatus] = useState<'成約'|'失注'>('成約');
  const [finalRate, setFinalRate] = useState(0.0);
  const [baseRate, setBaseRate] = useState(2.1);
  const [lostReason, setLostReason] = useState('');
  const [competitorName, setCompetitorName] = useState('');
  const [competitorRate, setCompetitorRate] = useState(0.0);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [pendingCases, setPendingCases] = useState<any[]>([]);
  const [selectedCase, setSelectedCase] = useState<any | null>(null);
  const [liveClosureProb, setLiveClosureProb] = useState<number | null>(null);
  const [progressStampingCaseId, setProgressStampingCaseId] = useState<string | null>(null);

  useEffect(() => {
    // Escaped string check: some environments use double backslash for display
    triggerMebuki('guide', '案件の最終的な結果を登録しましょう。\n全項目を入力することで、AIの学習精度が大幅に向上します！');
    fetchPendingCases();
  }, []);

  const fetchPendingCases = async () => {
    try {
      const res = await apiClient.get(`/api/cases/pending`);
      setPendingCases(res.data);
    } catch (err) {
      console.error("Failed to fetch pending cases", err);
    }
  };

  const deleteCase = async (caseId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`案件 ${caseId} を削除しますか？`)) return;
    try {
      await apiClient.delete(`/api/cases/${caseId}`);
      triggerMebuki('guide', '案件を削除しました。');
      fetchPendingCases();
    } catch (err) {
      triggerMebuki('reject', '削除に失敗しました。');
    }
  };

  const clearAllCases = async () => {
    if (!confirm('全ての未登録データを削除してもよろしいですか？')) return;
    try {
      await apiClient.delete(`/api/cases/operation/clear-all`);
      triggerMebuki('guide', '全ての未登録案件を削除しました。');
      fetchPendingCases();
    } catch (err) {
      triggerMebuki('reject', '一括削除に失敗しました。');
    }
  };

  const selectCase = (c: any) => {
    setTargetId(c.id);
    setSelectedCase(c);
    triggerMebuki('approve', `企業番号 #${c.company_no} を選択しました！`);
  };



  const stampProgress = async (eventType: 'estimate_sent' | 'customer_response', caseId?: string) => {
    const activeCaseId = caseId ?? targetId;
    if (!activeCaseId) {
      triggerMebuki('challenge', '先に案件を選択してください。');
      return;
    }
    setProgressStampingCaseId(activeCaseId);
    try {
      const res = await apiClient.post(`/api/cases/progress-stamp`, {
        case_id: activeCaseId,
        event_type: eventType,
      });
      const p = res?.data?.closure_probability_percent;
      if (typeof p === 'number') setLiveClosureProb(p);
      triggerMebuki('approve', `${eventType === 'estimate_sent' ? '見積提示' : '顧客反応'}を記録しました。`);
      fetchPendingCases();
    } catch (err) {
      triggerMebuki('reject', 'タイムスタンプ記録に失敗しました。');
    } finally {
      setProgressStampingCaseId(null);
    }
  };

  const toggleCondition = (opt: string) => {
    setSelectedConditions(prev => 
      prev.includes(opt) ? prev.filter(c => c !== opt) : [...prev, opt]
    );
  };

  const handleRegister = async () => {
    if(!targetId) {
        triggerMebuki('challenge', '企業名または案件IDを入力してくださいね。');
        return;
    }
    setSubmitting(true);
    try {
      await apiClient.post(`/api/cases/register`, {
        case_id: targetId,
        status: status,
        final_rate: finalRate,
        base_rate_at_time: baseRate,
        lost_reason: lostReason,
        loan_conditions: selectedConditions,
        competitor_name: competitorName,
        competitor_rate: competitorRate,
        note: note
      });
      triggerMebuki('approve', `${targetId} の結果を登録しました！ご協力ありがとうございます！`);
      setTargetId('');
      setNote('');
      setLostReason('');
      setCompetitorName('');
      setSelectedConditions([]);
      fetchPendingCases(); // リロード
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '登録に失敗しました。存在する案件か確認してください。');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-4xl font-black text-slate-800 flex items-center gap-4">
          <ClipboardList className="w-10 h-10 text-rose-500" />
          審査結果の最終登録
        </h1>
        <p className="text-slate-500 font-bold mt-2">成約・失注の情報を詳細に記録し、AIの「目利き」を強化します。</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl">
        {/* 基本情報とステータス */}
        <div className="space-y-6">
           <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-black text-slate-700 flex items-center gap-2">
                   <User className="w-5 h-5 text-indigo-500" />
                   1. 対象案件の特定
                </h3>
                {pendingCases.length > 0 && (
                  <button 
                    onClick={clearAllCases}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-600 rounded-xl text-[10px] font-black transition-all border border-rose-100"
                  >
                    <Trash2 className="w-3 h-3" />
                    全件削除
                  </button>
                )}
              </div>
              <div className="relative">
                  <Search className="absolute left-4 top-4 w-5 h-5 text-slate-400" />
                  <input 
                     type="text" 
                     className="w-full bg-slate-50 border border-slate-200 p-4 pl-12 rounded-2xl font-black text-slate-700 outline-none focus:ring-2 focus:ring-indigo-500/20 transition-all"
                     value={targetId}
                     onChange={(e) => setTargetId(e.target.value)}
                     placeholder="企業名 または 案件ID"
                  />
              </div>

              {pendingCases.length > 0 && (
                <div className="mt-6 overflow-hidden rounded-2xl border border-slate-200">
                  <div className="bg-slate-100 px-4 py-3 text-xs font-black text-slate-500">一覧で進捗更新（ボタンで即時記録）</div>
                  <div className="max-h-72 overflow-auto">
                    <table className="w-full text-xs">
                      <thead className="bg-white sticky top-0">
                        <tr className="text-slate-400">
                          <th className="text-left px-3 py-2">企業</th>
                          <th className="text-left px-3 py-2">案件ID</th>
                          <th className="text-left px-3 py-2">進捗操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pendingCases.map((c) => (
                          <tr key={`row-${c.id}`} className={`border-t border-slate-100 hover:bg-slate-50 ${selectedCase?.id === c.id ? 'bg-indigo-50/50' : ''}`}>
                            <td className="px-3 py-2 font-bold text-slate-700">#{c.company_no} {c.company_name}</td>
                            <td className="px-3 py-2 font-mono text-slate-500">{c.id}</td>
                            <td className="px-3 py-2">
                              <div className="flex flex-wrap gap-2">
                                <button onClick={() => { selectCase(c); stampProgress('estimate_sent', c.id); }} disabled={progressStampingCaseId === c.id} className="px-2.5 py-1 rounded-md bg-blue-50 text-blue-700 font-bold border border-blue-100 disabled:opacity-50">見積提示</button>
                                <button onClick={() => { selectCase(c); stampProgress('customer_response', c.id); }} disabled={progressStampingCaseId === c.id} className="px-2.5 py-1 rounded-md bg-violet-50 text-violet-700 font-bold border border-violet-100 disabled:opacity-50">顧客反応</button>
                                <button onClick={() => selectCase(c)} className="px-2.5 py-1 rounded-md bg-slate-50 text-slate-700 font-bold border border-slate-200">選択</button>
                                <button onClick={(e) => deleteCase(c.id, e)} className="px-2.5 py-1 rounded-md bg-rose-50 text-rose-700 font-bold border border-rose-100">削除</button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

                            {selectedCase && (
                <div className="mt-4 p-4 rounded-xl bg-indigo-50 border border-indigo-100 text-xs">
                  <p className="font-black text-indigo-700 mb-2">自動タイムスタンプ（編集不要）</p>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-slate-700">
                    <div>審査登録: <span className="font-bold">{selectedCase.registration_date || selectedCase.timestamp?.slice(0, 10) || '-'}</span></div>
                    <div>見積提示: <span className="font-bold">{selectedCase.estimate_sent_date || selectedCase.registration_date || selectedCase.timestamp?.slice(0, 10) || '-'}</span></div>
                    <div>確定時: <span className="font-bold">登録ボタン押下時に自動記録</span></div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button onClick={() => stampProgress('estimate_sent')} className="px-3 py-1.5 rounded-lg bg-blue-600 text-white font-bold">見積提示を今で記録</button>
                    <button onClick={() => stampProgress('customer_response')} className="px-3 py-1.5 rounded-lg bg-violet-600 text-white font-bold">顧客反応を今で記録</button>
                    {liveClosureProb !== null && (
                      <span className="px-3 py-1.5 rounded-lg bg-emerald-100 text-emerald-700 font-black">成約確率: {liveClosureProb.toFixed(1)}%</span>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-8 flex gap-4">
                 <button 
                    onClick={() => setStatus('成約')}
                    className={`flex-1 p-6 rounded-2xl border-2 transition-all flex flex-col items-center gap-2 ${status === '成約' ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-white border-slate-100 text-slate-400'}`}
                 >
                    <CheckCircle className={`w-8 h-8 ${status === '成約' ? 'text-emerald-500' : 'text-slate-300'}`} />
                    <span className="font-black">成約 (WIN)</span>
                 </button>
                 <button 
                    onClick={() => setStatus('失注')}
                    className={`flex-1 p-6 rounded-2xl border-2 transition-all flex flex-col items-center gap-2 ${status === '失注' ? 'bg-rose-50 border-rose-500 text-rose-700' : 'bg-white border-slate-100 text-slate-400'}`}
                 >
                    <XCircle className={`w-8 h-8 ${status === '失注' ? 'text-rose-500' : 'text-slate-300'}`} />
                    <span className="font-black">失注 (LOST)</span>
                 </button>
              </div>
           </div>

           <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
              <h3 className="text-lg font-black text-slate-700 mb-6 flex items-center gap-2">
                 <Percent className="w-5 h-5 text-emerald-500" />
                 2. 金利・レート情報
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                 <div>
                    <label className="block text-xs font-black text-slate-400 uppercase mb-2">最終獲得レート (%)</label>
                    <input 
                       type="text" inputMode="decimal" step="0.01" value={finalRate}
                       onChange={(e) => setFinalRate(parseFloat(e.target.value))}
                       className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-emerald-600 outline-none"
                    />
                 </div>
                 <div>
                    <label className="block text-xs font-black text-slate-400 uppercase mb-2">当時の基準金利 (%)</label>
                    <input 
                       type="text" inputMode="decimal" step="0.01" value={baseRate}
                       onChange={(e) => setBaseRate(parseFloat(e.target.value))}
                       className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-slate-600 outline-none"
                    />
                 </div>
              </div>
           </div>
        </div>

        {/* 詳細情報 */}
        <div className="space-y-6">
           <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
              <h3 className="text-lg font-black text-slate-700 mb-6 flex items-center gap-2">
                 <Building2 className="w-5 h-5 text-orange-500" />
                 3. 競合・失注分析
              </h3>
              <div className="space-y-4">
                 <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                           type="text" inputMode="decimal" step="0.01" value={competitorRate}
                           onChange={(e) => setCompetitorRate(parseFloat(e.target.value))}
                           className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-orange-600 outline-none"
                        />
                    </div>
                 </div>
                 {status === '失注' && (
                    <div className="animate-in slide-in-from-top-2 duration-300">
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
                 )}
              </div>
           </div>

           <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
              <h3 className="text-lg font-black text-slate-700 mb-6 flex items-center gap-2">
                 <CheckCircle className="w-5 h-5 text-indigo-500" />
                 4. 成約/承認の付帯条件
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
        </div>
      </div>

      <div className="mt-8 max-w-6xl">
         <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
            <h3 className="text-lg font-black text-slate-700 mb-4 flex items-center gap-2">
               <FileText className="w-5 h-5 text-slate-400" />
               備考・メモ
            </h3>
            <textarea 
               className="w-full bg-slate-50 border border-slate-200 p-6 rounded-2xl text-sm text-slate-700 outline-none focus:ring-2 focus:ring-slate-500/10 min-h-[100px]"
               value={note}
               onChange={(e) => setNote(e.target.value)}
               placeholder="その他、特筆すべき事項があれば入力してください"
            />
            
            <div className="mt-8 flex justify-end">
               <button 
                  onClick={handleRegister}
                  disabled={submitting}
                  className={`py-5 px-16 rounded-[2rem] shadow-2xl transition-all flex items-center gap-3 font-black text-lg ${status === '成約' ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-500/30' : 'bg-rose-600 hover:bg-rose-500 shadow-rose-500/30'} text-white group`}
               >
                  {submitting ? <Activity className="w-6 h-6 animate-spin" /> : <Save className="w-6 h-6 group-hover:scale-110 transition-transform" />}
                  最終結果をデータベースへ書き込む
               </button>
            </div>
         </div>
      </div>
    </div>
  );
}
