"use client";
import React, { useState, useEffect } from 'react';
import { apiClient } from '../../lib/api';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import CaseRegistrationForm from '../../components/analysis/CaseRegistrationForm';
import { Search, User, ClipboardList, Trash2, RefreshCw } from 'lucide-react';

export default function RegisterPage() {
  const [targetId, setTargetId] = useState('');
  const [pendingCases, setPendingCases] = useState<any[]>([]);
  const [selectedCase, setSelectedCase] = useState<any | null>(null);
  const [liveClosureProb, setLiveClosureProb] = useState<number | null>(null);
  const [progressStampingCaseId, setProgressStampingCaseId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Escaped string check: some environments use double backslash for display
    triggerMebuki('guide', '案件の最終的な結果を登録しましょう。\n全項目を入力することで、AIの学習精度が大幅に向上します！');
    fetchPendingCases();
  }, []);

  useEffect(() => {
    // 審査分析タブで新規に審査した案件が反映されるよう、このページに戻ってきたタイミングで再取得する
    const handleFocusOrVisible = () => {
      if (document.visibilityState === 'hidden') return;
      fetchPendingCases();
    };
    window.addEventListener('focus', handleFocusOrVisible);
    document.addEventListener('visibilitychange', handleFocusOrVisible);
    return () => {
      window.removeEventListener('focus', handleFocusOrVisible);
      document.removeEventListener('visibilitychange', handleFocusOrVisible);
    };
  }, []);

  const fetchPendingCases = async () => {
    setRefreshing(true);
    try {
      const res = await apiClient.get(`/api/cases/pending`);
      setPendingCases(res.data);
    } catch (err) {
      console.error("Failed to fetch pending cases", err);
      triggerMebuki('reject', '未登録案件一覧の取得に失敗しました。ネットワークまたはAPIの状態を確認してください。');
    } finally {
      setRefreshing(false);
    }
  };

  const scoreColor = (s: number | null | undefined) => {
    if (s === null || s === undefined || Number.isNaN(s)) return 'text-slate-400';
    if (s >= 70) return 'text-emerald-600';
    if (s >= 50) return 'text-amber-600';
    return 'text-rose-600';
  };

  const hanteiBadgeClass = (hantei: string | null | undefined) => {
    if (!hantei) return 'bg-slate-100 text-slate-400';
    if (hantei.includes('否決')) return 'bg-rose-100 text-rose-700';
    if (hantei.includes('条件付') || hantei.includes('要審議')) return 'bg-amber-100 text-amber-700';
    if (hantei.includes('承認')) return 'bg-emerald-100 text-emerald-700';
    return 'bg-amber-100 text-amber-700';
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

  const handleRegistered = () => {
    setTargetId('');
    setSelectedCase(null);
    fetchPendingCases(); // リロード
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

      <div className="max-w-6xl space-y-6">
           <div className="bg-white border border-slate-200 rounded-[2rem] shadow-xl p-8">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-black text-slate-700 flex items-center gap-2">
                   <User className="w-5 h-5 text-indigo-500" />
                   1. 対象案件の特定
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={fetchPendingCases}
                    disabled={refreshing}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 rounded-xl text-[10px] font-black transition-all border border-indigo-100 disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
                    更新
                  </button>
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
                          <th className="text-left px-3 py-2">分析結果</th>
                          <th className="text-left px-3 py-2">案件ID</th>
                          <th className="text-left px-3 py-2">進捗操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pendingCases.map((c) => (
                          <tr key={`row-${c.id}`} className={`border-t border-slate-100 hover:bg-slate-50 ${selectedCase?.id === c.id ? 'bg-indigo-50/50' : ''}`}>
                            <td className="px-3 py-2 font-bold text-slate-700">
                              <div>#{c.company_no || '-'} {c.company_name}</div>
                              {c._source && c._source !== 'past_cases' && (
                                <div className="mt-1 text-[10px] font-black text-teal-600">{c._source}</div>
                              )}
                            </td>
                            <td className="px-3 py-2">
                              <div className="flex items-center gap-2">
                                <span className={`font-black ${scoreColor(c.score)}`}>
                                  {c.score !== null && c.score !== undefined && c.score !== '' ? Math.round(Number(c.score)) : '—'}
                                </span>
                                {c.hantei && (
                                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-black ${hanteiBadgeClass(c.hantei)}`}>
                                    {c.hantei}
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-3 py-2 font-mono text-slate-500 break-all">{c.id}</td>
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
                  <p className="font-black text-indigo-700 mb-2">審査分析結果</p>
                  <div className="flex items-center gap-3 mb-4">
                    <span className={`text-2xl font-black ${scoreColor(selectedCase.score)}`}>
                      {selectedCase.score !== null && selectedCase.score !== undefined && selectedCase.score !== '' ? Math.round(Number(selectedCase.score)) : '—'}
                    </span>
                    <span className="text-slate-400 font-bold">点</span>
                    {selectedCase.hantei ? (
                      <span className={`px-3 py-1 rounded-full text-xs font-black ${hanteiBadgeClass(selectedCase.hantei)}`}>
                        {selectedCase.hantei}
                      </span>
                    ) : (
                      <span className="px-3 py-1 rounded-full text-xs font-black bg-slate-100 text-slate-400">判定なし</span>
                    )}
                  </div>
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
           </div>

           <CaseRegistrationForm caseId={targetId} onRegistered={handleRegistered} />
      </div>
    </div>
  );
}
