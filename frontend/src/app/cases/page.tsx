"use client";
import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../../lib/api';
import { Table2, RefreshCw, Trash2, CheckCircle, XCircle, Clock, ChevronLeft, ChevronRight } from 'lucide-react';

const LOSS_REASONS = ['設備見合わせ', '他社競合', '調達方法変更', 'その他'];
const VALID_STATUSES = ['成約', '失注', '未登録', 'スコアリングのみ', '検収', '検収完了'];

type Case = {
  id: string;
  timestamp: string;
  company_name: string;
  company_no: string;
  score: number | null;
  judgment: string | null;
  final_status: string;
  industry_sub: string;
};

type ResultForm = {
  final_status: string;
  competitor_rate: string;
  loss_reason: string;
  final_result_date: string;
};

const STATUS_STYLE: Record<string, string> = {
  '成約':       'bg-emerald-100 text-emerald-700',
  '失注':       'bg-rose-100 text-rose-700',
  '未登録':     'bg-slate-100 text-slate-500',
  '検収':       'bg-blue-100 text-blue-700',
  '検収完了':   'bg-blue-200 text-blue-800',
  'スコアリングのみ': 'bg-amber-100 text-amber-700',
};

export default function CasesPage() {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [selected, setSelected] = useState<Case | null>(null);
  const [form, setForm] = useState<ResultForm>({ final_status: '', competitor_rate: '', loss_reason: '', final_result_date: '' });
  const [submitting, setSubmitting] = useState(false);
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null);
  const LIMIT = 30;

  const fetchCases = useCallback(async (newOffset = 0) => {
    setLoading(true);
    setMsg(null);
    try {
      const res = await apiClient.get(`/api/cases?limit=${LIMIT}&offset=${newOffset}&sort=desc`);
      const data: Case[] = res.data;
      setCases(data);
      setOffset(newOffset);
      setHasMore(data.length === LIMIT);
    } catch {
      setMsg({ text: '案件の取得に失敗しました', ok: false });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchCases(0); }, [fetchCases]);

  const selectCase = (c: Case) => {
    setSelected(c);
    setForm({ final_status: c.final_status || '', competitor_rate: '', loss_reason: '', final_result_date: '' });
    setMsg(null);
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`案件 ${id} を削除しますか？`)) return;
    try {
      await apiClient.delete(`/api/cases/${id}`);
      if (selected?.id === id) setSelected(null);
      fetchCases(offset);
    } catch {
      setMsg({ text: '削除に失敗しました', ok: false });
    }
  };

  const handleSubmit = async () => {
    if (!selected) return;
    setSubmitting(true);
    setMsg(null);
    const payload: Record<string, unknown> = {};
    if (form.final_status) payload.final_status = form.final_status;
    if (form.competitor_rate) payload.competitor_rate = parseFloat(form.competitor_rate);
    if (form.loss_reason) payload.loss_reason = form.loss_reason;
    if (form.final_result_date) payload.final_result_date = form.final_result_date;
    try {
      await apiClient.patch(`/api/cases/${selected.id}/result`, payload);
      setMsg({ text: '✅ 更新しました', ok: true });
      fetchCases(offset);
    } catch {
      setMsg({ text: '❌ 更新に失敗しました', ok: false });
    } finally {
      setSubmitting(false);
    }
  };

  const scoreColor = (s: number | null) => {
    if (s === null) return 'text-slate-400';
    if (s >= 70) return 'text-emerald-600';
    if (s >= 50) return 'text-amber-600';
    return 'text-rose-600';
  };

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Table2 className="w-8 h-8 text-cyan-500" />
          過去案件一覧
        </h1>
        <button
          onClick={() => fetchCases(offset)}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold text-sm transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          更新
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* 案件テーブル */}
        <div className="xl:col-span-2">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr className="text-slate-500 text-xs font-black uppercase tracking-wider">
                    <th className="text-left px-4 py-3">会社</th>
                    <th className="text-left px-4 py-3">日付</th>
                    <th className="text-center px-4 py-3">スコア</th>
                    <th className="text-center px-4 py-3">ステータス</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {loading && cases.length === 0 && (
                    <tr><td colSpan={5} className="text-center py-12 text-slate-400 font-bold">読み込み中...</td></tr>
                  )}
                  {!loading && cases.length === 0 && (
                    <tr><td colSpan={5} className="text-center py-12 text-slate-400 font-bold">案件がありません</td></tr>
                  )}
                  {cases.map(c => (
                    <tr
                      key={c.id}
                      onClick={() => selectCase(c)}
                      className={`border-b border-slate-100 cursor-pointer transition-colors ${selected?.id === c.id ? 'bg-cyan-50' : 'hover:bg-slate-50'}`}
                    >
                      <td className="px-4 py-3">
                        <div className="font-bold text-slate-800 truncate max-w-[160px]">{c.company_name || '—'}</div>
                        <div className="text-xs text-slate-400 font-mono">{c.company_no || c.id.slice(0, 8)}</div>
                      </td>
                      <td className="px-4 py-3 text-slate-500 whitespace-nowrap">{c.timestamp?.slice(0, 10) || '—'}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`text-lg font-black ${scoreColor(c.score)}`}>
                          {c.score != null ? Math.round(c.score) : '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-black ${STATUS_STYLE[c.final_status] || 'bg-slate-100 text-slate-500'}`}>
                          {c.final_status || '未登録'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={(e) => handleDelete(c.id, e)}
                          className="p-1.5 rounded-lg text-slate-300 hover:text-rose-500 hover:bg-rose-50 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* ページネーション */}
            <div className="flex items-center justify-between px-4 py-3 border-t border-slate-100 bg-slate-50">
              <span className="text-xs text-slate-400 font-bold">{offset + 1}〜{offset + cases.length} 件</span>
              <div className="flex gap-2">
                <button
                  onClick={() => fetchCases(Math.max(0, offset - LIMIT))}
                  disabled={offset === 0 || loading}
                  className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-500 hover:bg-slate-100 disabled:opacity-40 transition-all"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  onClick={() => fetchCases(offset + LIMIT)}
                  disabled={!hasMore || loading}
                  className="p-1.5 rounded-lg bg-white border border-slate-200 text-slate-500 hover:bg-slate-100 disabled:opacity-40 transition-all"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 結果登録フォーム */}
        <div className="xl:col-span-1">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sticky top-4">
            {!selected ? (
              <div className="text-center py-12 text-slate-400">
                <Clock className="w-10 h-10 mx-auto mb-3 opacity-40" />
                <p className="font-bold text-sm">左の表から案件を選択してください</p>
              </div>
            ) : (
              <>
                <h2 className="text-base font-black text-slate-700 mb-1">結果登録</h2>
                <p className="text-xs text-slate-400 font-bold mb-5 truncate">
                  {selected.company_name || selected.id}
                </p>

                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-black text-slate-500 uppercase mb-1.5">ステータス</label>
                    <div className="grid grid-cols-2 gap-2">
                      {['成約', '失注'].map(s => (
                        <button
                          key={s}
                          onClick={() => setForm(f => ({ ...f, final_status: s }))}
                          className={`py-2 rounded-xl text-sm font-black border-2 transition-all flex items-center justify-center gap-1.5
                            ${form.final_status === s
                              ? s === '成約' ? 'bg-emerald-50 border-emerald-500 text-emerald-700' : 'bg-rose-50 border-rose-500 text-rose-700'
                              : 'bg-white border-slate-200 text-slate-400 hover:border-slate-300'}`}
                        >
                          {s === '成約' ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-black text-slate-500 uppercase mb-1.5">競合レート (%)</label>
                    <input
                      type="number" step="0.01"
                      value={form.competitor_rate}
                      onChange={e => setForm(f => ({ ...f, competitor_rate: e.target.value }))}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-cyan-500/20"
                      placeholder="例: 2.85"
                    />
                  </div>

                  {form.final_status === '失注' && (
                    <div>
                      <label className="block text-xs font-black text-rose-500 uppercase mb-1.5">失注理由</label>
                      <select
                        value={form.loss_reason}
                        onChange={e => setForm(f => ({ ...f, loss_reason: e.target.value }))}
                        className="w-full bg-rose-50 border border-rose-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-700 outline-none"
                      >
                        <option value="">— 選択 —</option>
                        {LOSS_REASONS.map(r => <option key={r} value={r}>{r}</option>)}
                      </select>
                    </div>
                  )}

                  <div>
                    <label className="block text-xs font-black text-slate-500 uppercase mb-1.5">確定日</label>
                    <input
                      type="date"
                      value={form.final_result_date}
                      onChange={e => setForm(f => ({ ...f, final_result_date: e.target.value }))}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-cyan-500/20"
                    />
                  </div>

                  <button
                    onClick={handleSubmit}
                    disabled={submitting || !form.final_status}
                    className="w-full py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-black text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {submitting ? '更新中...' : '結果を登録'}
                  </button>

                  {msg && (
                    <p className={`text-xs font-bold text-center ${msg.ok ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {msg.text}
                    </p>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
