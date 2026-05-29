"use client";
import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../../lib/api';
import {
  FileText, RefreshCw, ChevronDown, Loader2, AlertCircle,
  Zap, ShieldAlert, CheckCircle2, XCircle, AlertTriangle,
  TrendingUp, TrendingDown, Minus, BarChart2
} from 'lucide-react';

type CaseRow = {
  id: string;
  timestamp: string;
  company_name: string | null;
  company_no: string | null;
  score: number | null;
  final_status: string;
};

type TotalScorerResult = {
  total_score?: number;
  grade?: string;
  grade_text?: string;
  grade_color?: string;
  obligor_score?: number;
  asset_score?: number;
};

type CaseDetail = {
  result?: {
    score?: number;
    hantei?: string;
    quantum_risk?: number | null;
    credit_quantum_strong_warning?: boolean;
    pd_percent?: number;
    total_scorer_result?: TotalScorerResult;
    user_op?: number;
    bench_op?: number;
    user_eq?: number;
    bench_eq?: number;
    user_dscr?: number;
    asset_name?: string;
    hints?: { subsidies?: string[]; risks?: string[]; mandatory?: string };
  };
  inputs?: Record<string, unknown>;
};

const STATUS_DOT: Record<string, string> = {
  '成約': 'bg-emerald-400',
  '失注': 'bg-rose-400',
  '未登録': 'bg-slate-300',
};

const GRADE_ACTIONS: Record<string, { icon: React.ReactNode; color: string; actions: string[] }> = {
  '承認': {
    icon: <CheckCircle2 className="w-4 h-4" />,
    color: 'emerald',
    actions: [
      '契約条件を最終確認し、正式承認書を発行する',
      '物件検収・引渡しスケジュールを調整する',
      '初回リース料の引落口座・日程を確認する',
      '顧客へ承認連絡と今後の流れを説明する',
    ],
  },
  '条件付き承認': {
    icon: <AlertTriangle className="w-4 h-4" />,
    color: 'amber',
    actions: [
      '保証人の追加 or 担保提供の可否を顧客に確認する',
      'リース金額の減額（当初申請の80〜90%水準）を検討する',
      '最新決算書・試算表の追加提出を要請する',
      '月次返済シミュレーションを提示し、DSCR余裕度を説明する',
      'リース期間を短縮して月額負担を上げる代わりにリスクを低減する',
    ],
  },
  '否決': {
    icon: <XCircle className="w-4 h-4" />,
    color: 'rose',
    actions: [
      '否決理由を整理し、顧客へ丁寧に説明する（具体的数値は伏せる）',
      '補助金活用・自己資本の積み増しなど改善策を提示する',
      '6〜12ヶ月後の再申請を視野に改善ロードマップを提案する',
      '競合他社への移行を防ぐため、別商品（割賦・レンタル）を提案する',
    ],
  },
};

function QRiskGauge({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value));
  const color = pct >= 60 ? '#ef4444' : pct >= 35 ? '#f59e0b' : '#22c55e';
  const label = pct >= 60 ? '高リスク' : pct >= 35 ? '要注意' : '正常';
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center text-xs">
        <span className="font-bold text-slate-600">Q_risk</span>
        <span className="font-black" style={{ color }}>{pct.toFixed(1)}</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <div className="flex justify-between text-[10px] text-slate-400">
        <span>0</span>
        <span className="font-bold" style={{ color }}>{label}</span>
        <span>100</span>
      </div>
    </div>
  );
}

function CompareRow({ label, user, bench, unit = '%' }: { label: string; user?: number; bench?: number; unit?: string }) {
  if (user == null || bench == null) return null;
  const diff = user - bench;
  const isGood = diff >= 0;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-100 last:border-0 text-xs">
      <span className="text-slate-600 font-medium">{label}</span>
      <div className="flex items-center gap-3">
        <span className="text-slate-400">目安 {bench.toFixed(1)}{unit}</span>
        <span className="font-black text-slate-800">{user.toFixed(1)}{unit}</span>
        <span className={`flex items-center gap-0.5 font-bold ${isGood ? 'text-emerald-500' : 'text-rose-500'}`}>
          {isGood ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {Math.abs(diff).toFixed(1)}
        </span>
      </div>
    </div>
  );
}

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
  const [caseDetail, setCaseDetail] = useState<CaseDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
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

  useEffect(() => {
    if (!selectedId) { setCaseDetail(null); return; }
    setLoadingDetail(true);
    setCaseDetail(null);
    apiClient.get(`/api/cases/${selectedId}`)
      .then(res => setCaseDetail(res.data))
      .catch(() => setCaseDetail(null))
      .finally(() => setLoadingDetail(false));
  }, [selectedId]);

  const generate = async () => {
    if (!selectedId) return;
    setGenerating(true);
    setReport(null);
    setError(null);
    try {
      const detail = caseDetail || (await apiClient.get(`/api/cases/${selectedId}`)).data;
      const result_data = detail.result || {};
      const inputs = detail.inputs || detail;
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
  const result = caseDetail?.result;
  const gradeText = result?.total_scorer_result?.grade_text;
  const gradeInfo = gradeText ? GRADE_ACTIONS[gradeText] ?? GRADE_ACTIONS['条件付き承認'] : null;
  const qRisk = result?.quantum_risk;
  const hasQRiskWarning = result?.credit_quantum_strong_warning || (qRisk != null && qRisk >= 60);

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
        {/* 左: 案件選択 + インサイトパネル */}
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

          {/* クイックインサイトパネル */}
          {loadingDetail && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 flex items-center justify-center h-32">
              <Loader2 className="w-5 h-5 animate-spin text-slate-300" />
            </div>
          )}

          {result && !loadingDetail && (
            <div className="space-y-3">
              {/* スコア・グレード */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
                <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                  <BarChart2 className="w-3.5 h-3.5" /> スコアサマリー
                </h3>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-3xl font-black text-slate-800">{result.score != null ? Math.round(result.score) : '—'}</p>
                    <p className="text-xs text-slate-400">総合スコア</p>
                  </div>
                  {gradeText && (
                    <div className={`px-3 py-1.5 rounded-xl text-sm font-black flex items-center gap-1.5
                      ${gradeText === '承認' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' :
                        gradeText === '否決' ? 'bg-rose-50 text-rose-700 border border-rose-200' :
                        'bg-amber-50 text-amber-700 border border-amber-200'}`}>
                      {gradeInfo?.icon}
                      {gradeText}
                    </div>
                  )}
                </div>
                {result.pd_percent != null && (
                  <div className="flex items-center gap-2 text-xs text-slate-500 pt-1 border-t border-slate-100">
                    <Minus className="w-3 h-3" /> PD率:
                    <span className={`font-black ${result.pd_percent > 30 ? 'text-rose-600' : result.pd_percent > 15 ? 'text-amber-600' : 'text-emerald-600'}`}>
                      {result.pd_percent.toFixed(1)}%
                    </span>
                  </div>
                )}
                {result.asset_name && (
                  <p className="text-xs text-slate-400">物件: {result.asset_name}</p>
                )}
              </div>

              {/* Q_risk パネル */}
              {qRisk != null && (
                <div className={`bg-white rounded-2xl border shadow-sm p-5 space-y-3 ${hasQRiskWarning ? 'border-red-200' : 'border-slate-200'}`}>
                  <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-yellow-500" /> 量子リスク (Q_risk)
                  </h3>
                  <QRiskGauge value={qRisk} />
                  {hasQRiskWarning && (
                    <div className="flex items-start gap-2 p-2.5 bg-red-50 border border-red-200 rounded-xl text-xs text-red-700">
                      <ShieldAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                      <span className="font-bold">財務指標の矛盾・異常が検出されました。決算書の精査を推奨します。</span>
                    </div>
                  )}
                  <p className="text-[10px] text-slate-400 leading-relaxed">
                    Q_riskは財務指標間の矛盾・異常パターンを量子干渉計算で検出したスコアです。
                    35以上で要注意、60以上で強警戒となります。
                  </p>
                </div>
              )}

              {/* 業種平均比較 */}
              {(result.user_op != null || result.user_eq != null) && (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
                  <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <TrendingUp className="w-3.5 h-3.5" /> 業種平均比較
                  </h3>
                  <CompareRow label="営業利益率" user={result.user_op} bench={result.bench_op} />
                  <CompareRow label="自己資本比率" user={result.user_eq} bench={result.bench_eq} />
                  {result.user_dscr != null && (
                    <div className="flex items-center justify-between py-1.5 border-b border-slate-100 last:border-0 text-xs">
                      <span className="text-slate-600 font-medium">DSCR</span>
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">目安 1.5倍</span>
                        <span className={`font-black ${result.user_dscr >= 1.5 ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {result.user_dscr.toFixed(2)}倍
                        </span>
                        {result.user_dscr >= 1.5 ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> : <AlertTriangle className="w-3 h-3 text-rose-500" />}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ヒント */}
              {result.hints && (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-2">
                  <h3 className="text-xs font-black text-slate-500 uppercase tracking-wider">AIヒント</h3>
                  {result.hints.mandatory && (
                    <div className="p-2.5 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800">
                      <span className="font-black">必須確認:</span> {result.hints.mandatory}
                    </div>
                  )}
                  {result.hints.subsidies && result.hints.subsidies.length > 0 && (
                    <div>
                      <p className="text-[10px] font-black text-slate-400 mb-1">💰 補助金候補</p>
                      {result.hints.subsidies.map((s, i) => (
                        <p key={i} className="text-xs text-slate-600">・{s}</p>
                      ))}
                    </div>
                  )}
                  {result.hints.risks && result.hints.risks.length > 0 && (
                    <div>
                      <p className="text-[10px] font-black text-slate-400 mb-1">⚠️ リスク要因</p>
                      {result.hints.risks.map((r, i) => (
                        <p key={i} className="text-xs text-slate-600">・{r}</p>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 右: 推奨アクション + レポート表示 */}
        <div className="xl:col-span-2 space-y-4">
          {/* 条件付き承認の推奨アクション (REV-019) */}
          {gradeInfo && result && !loadingDetail && (
            <div className={`rounded-2xl border shadow-sm p-5
              ${gradeText === '承認' ? 'bg-emerald-50 border-emerald-200' :
                gradeText === '否決' ? 'bg-rose-50 border-rose-200' :
                'bg-amber-50 border-amber-200'}`}>
              <div className="flex items-center gap-2 mb-3">
                <span className={`text-${gradeInfo.color}-600`}>{gradeInfo.icon}</span>
                <h2 className={`text-sm font-black text-${gradeInfo.color}-800`}>
                  {gradeText} — 推奨アクション
                </h2>
              </div>
              <ul className="space-y-2">
                {gradeInfo.actions.map((action, i) => (
                  <li key={i} className={`flex items-start gap-2 text-sm text-${gradeInfo.color}-800`}>
                    <span className={`text-${gradeInfo.color}-400 font-black shrink-0`}>{i + 1}.</span>
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* レポート表示エリア */}
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
