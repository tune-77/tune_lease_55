"use client";
import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../../lib/api';
import { FileText, RefreshCw, ChevronDown, Loader2, AlertCircle } from 'lucide-react';

type CaseRow = {
  id: string;
  timestamp: string;
  company_name: string | null;
  company_no: string | null;
  score: number | null;
  final_status: string;
};

const STATUS_DOT: Record<string, string> = {
  '成約': 'bg-emerald-400',
  '失注': 'bg-rose-400',
  '未登録': 'bg-slate-300',
};

function MarkdownBlock({ md }: { md: string }) {
  const lines = md.split('\n');
  return (
    <div className="prose prose-sm max-w-none text-slate-700 leading-relaxed space-y-1">
      {lines.map((line, i) => {
        if (line.startsWith('## ')) return <h2 key={i} className="text-base font-black text-slate-800 mt-5 mb-2 border-b border-slate-200 pb-1">{line.slice(3)}</h2>;
        if (line.startsWith('# '))  return <h1 key={i} className="text-lg font-black text-slate-900 mt-4 mb-3">{line.slice(2)}</h1>;
        if (line.startsWith('### ')) return <h3 key={i} className="text-sm font-black text-slate-700 mt-3 mb-1">{line.slice(4)}</h3>;
        if (line.startsWith('- ') || line.startsWith('• '))
          return <li key={i} className="ml-4 text-sm list-disc">{line.slice(2)}</li>;
        if (line.startsWith('**') && line.endsWith('**'))
          return <p key={i} className="font-black text-slate-800 text-sm">{line.slice(2, -2)}</p>;
        if (line.trim() === '') return <div key={i} className="h-2" />;
        return <p key={i} className="text-sm">{line}</p>;
      })}
    </div>
  );
}

export default function ReportPage() {
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [loadingCases, setLoadingCases] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [generating, setGenerating] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchCases = useCallback(async () => {
    setLoadingCases(true);
    try {
      const res = await apiClient.get('/api/cases?limit=50&sort=desc');
      setCases(res.data);
    } catch {
      setError('案件一覧の取得に失敗しました');
    } finally {
      setLoadingCases(false);
    }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const generate = async () => {
    if (!selectedId) return;
    setGenerating(true);
    setReport(null);
    setError(null);
    try {
      const detail = await apiClient.get(`/api/cases/${selectedId}`);
      const caseData = detail.data;
      const result_data = caseData.result || {};
      const inputs = caseData.inputs || caseData;
      const res = await apiClient.post('/api/report/generate', { result_data, inputs });
      setReport(res.data.report_markdown || '（レポートが空です）');
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg || 'レポート生成に失敗しました');
    } finally {
      setGenerating(false);
    }
  };

  const selectedCase = cases.find(c => c.id === selectedId);

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <FileText className="w-8 h-8 text-indigo-500" />
          審査レポート
        </h1>
        <p className="text-slate-500 font-medium mt-1">過去案件を選択してAIレポートを生成します。</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* 左: 案件選択 */}
        <div className="xl:col-span-1 space-y-4">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-black text-slate-600 uppercase tracking-wider">案件を選択</h2>
              <button onClick={fetchCases} disabled={loadingCases}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all disabled:opacity-40">
                <RefreshCw className={`w-4 h-4 ${loadingCases ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <div className="relative">
              <select
                value={selectedId}
                onChange={e => { setSelectedId(e.target.value); setReport(null); setError(null); }}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-indigo-500/20 appearance-none pr-8"
              >
                <option value="">— 案件を選択 —</option>
                {cases.map(c => (
                  <option key={c.id} value={c.id}>
                    {c.company_name || '（名称なし）'} {c.timestamp?.slice(0, 10)} [{c.final_status}]
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
            </div>

            {selectedCase && (
              <div className="mt-3 p-3 bg-slate-50 rounded-xl text-xs space-y-1">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${STATUS_DOT[selectedCase.final_status] || 'bg-slate-300'}`} />
                  <span className="font-black text-slate-700">{selectedCase.final_status}</span>
                </div>
                <div className="text-slate-500">スコア: <span className="font-black text-slate-700">{selectedCase.score != null ? Math.round(selectedCase.score) : '—'}</span></div>
                <div className="text-slate-500 font-mono truncate">{selectedCase.id}</div>
              </div>
            )}

            <button
              onClick={generate}
              disabled={!selectedId || generating}
              className="mt-4 w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-black text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {generating
                ? <><Loader2 className="w-4 h-4 animate-spin" />生成中...</>
                : <><FileText className="w-4 h-4" />レポート生成</>}
            </button>
          </div>
        </div>

        {/* 右: レポート表示 */}
        <div className="xl:col-span-2">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 min-h-[400px]">
            {error && (
              <div className="flex items-start gap-3 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-sm font-bold">
                <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                {error}
              </div>
            )}

            {!report && !error && !generating && (
              <div className="flex flex-col items-center justify-center h-64 text-slate-300">
                <FileText className="w-16 h-16 mb-4" />
                <p className="font-bold text-sm">左で案件を選択して「レポート生成」を押してください</p>
              </div>
            )}

            {generating && (
              <div className="flex flex-col items-center justify-center h-64 text-slate-400 gap-3">
                <Loader2 className="w-10 h-10 animate-spin text-indigo-400" />
                <p className="font-bold text-sm">AIがレポートを作成しています...</p>
              </div>
            )}

            {report && !generating && (
              <>
                <div className="flex items-center justify-between mb-5 pb-4 border-b border-slate-100">
                  <h2 className="font-black text-slate-700">
                    {selectedCase?.company_name || '審査レポート'}
                  </h2>
                  <button
                    onClick={() => navigator.clipboard?.writeText(report)}
                    className="text-xs font-bold text-slate-400 hover:text-slate-600 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-all"
                  >
                    コピー
                  </button>
                </div>
                <MarkdownBlock md={report} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
