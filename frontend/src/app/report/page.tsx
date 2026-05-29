"use client";
import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../../lib/api';
import { FileText, RefreshCw, ChevronDown, Loader2, AlertCircle, AlertTriangle, TrendingDown, ShieldAlert, CheckCircle2, ClipboardList } from 'lucide-react';

type CaseRow = {
  id: string;
  timestamp: string;
  company_name: string | null;
  company_no: string | null;
  score: number | null;
  final_status: string;
};

type RiskFactor = {
  label: string;
  value: string;
  benchmark: string;
  severity: 'high' | 'medium';
};

const STATUS_DOT: Record<string, string> = {
  '成約': 'bg-emerald-400',
  '失注': 'bg-rose-400',
  '未登録': 'bg-slate-300',
};

function extractRiskFactors(inputs: Record<string, unknown>, result: Record<string, unknown>): RiskFactor[] {
  const risks: RiskFactor[] = [];

  const opMargin = typeof inputs.op_margin === 'number' ? inputs.op_margin : null;
  const eqRatio = typeof inputs.equity_ratio === 'number' ? inputs.equity_ratio
    : typeof inputs.eq_ratio === 'number' ? inputs.eq_ratio : null;
  const pdRaw = result.pd_percent ?? (result as Record<string, unknown>).pd;
  const pd = typeof pdRaw === 'number' ? pdRaw : null;
  const grade = typeof inputs.grade === 'number' ? inputs.grade : null;
  const debtRatio = typeof inputs.debt_ratio === 'number' ? inputs.debt_ratio : null;
  const currentRatio = typeof inputs.current_ratio === 'number' ? inputs.current_ratio : null;

  if (opMargin !== null && opMargin < 5) {
    risks.push({
      label: '営業利益率',
      value: `${opMargin.toFixed(1)}%`,
      benchmark: '目安 5%以上',
      severity: opMargin < 2 ? 'high' : 'medium',
    });
  }
  if (eqRatio !== null && eqRatio < 20) {
    risks.push({
      label: '自己資本比率',
      value: `${eqRatio.toFixed(1)}%`,
      benchmark: '目安 20%以上',
      severity: eqRatio < 10 ? 'high' : 'medium',
    });
  }
  if (pd !== null && pd > 3) {
    risks.push({
      label: 'デフォルト確率（PD）',
      value: `${pd.toFixed(2)}%`,
      benchmark: '目安 3%以下',
      severity: pd > 6 ? 'high' : 'medium',
    });
  }
  if (grade !== null && grade >= 7) {
    risks.push({
      label: '格付スコア',
      value: `${grade}点`,
      benchmark: '目安 6点以下',
      severity: grade >= 10 ? 'high' : 'medium',
    });
  }
  if (debtRatio !== null && debtRatio > 60) {
    risks.push({
      label: '負債比率',
      value: `${debtRatio.toFixed(1)}%`,
      benchmark: '目安 60%以下',
      severity: debtRatio > 80 ? 'high' : 'medium',
    });
  }
  if (currentRatio !== null && currentRatio < 100) {
    risks.push({
      label: '流動比率',
      value: `${currentRatio.toFixed(1)}%`,
      benchmark: '目安 100%以上',
      severity: currentRatio < 80 ? 'high' : 'medium',
    });
  }

  return risks.sort((a, b) => (a.severity === 'high' ? -1 : 1) - (b.severity === 'high' ? -1 : 1));
}

function ConditionalRiskPanel({ score, inputs, result }: {
  score: number;
  inputs: Record<string, unknown>;
  result: Record<string, unknown>;
}) {
  if (score < 60 || score >= 70) return null;

  const risks = extractRiskFactors(inputs, result);

  return (
    <div className="mb-5 p-4 bg-amber-50 border border-amber-300 rounded-xl">
      <div className="flex items-center gap-2 mb-3">
        <ShieldAlert className="w-5 h-5 text-amber-600 flex-shrink-0" />
        <span className="font-black text-amber-800 text-sm">条件付き承認 — 主要リスク要因</span>
        <span className="ml-auto text-xs font-bold text-amber-600 bg-amber-100 px-2 py-0.5 rounded-full">スコア {Math.round(score)}pt</span>
      </div>
      {risks.length === 0 ? (
        <p className="text-xs text-amber-700 font-bold">詳細な財務データが取得できませんでした。レポートを参照してください。</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {risks.map(r => (
            <div key={r.label} className={`flex items-start gap-2 p-2.5 rounded-lg border ${r.severity === 'high' ? 'bg-rose-50 border-rose-200' : 'bg-amber-100/60 border-amber-200'}`}>
              {r.severity === 'high'
                ? <AlertTriangle className="w-4 h-4 text-rose-500 flex-shrink-0 mt-0.5" />
                : <TrendingDown className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />}
              <div>
                <p className={`text-xs font-black ${r.severity === 'high' ? 'text-rose-700' : 'text-amber-800'}`}>{r.label}</p>
                <p className={`text-xs font-bold ${r.severity === 'high' ? 'text-rose-600' : 'text-amber-700'}`}>
                  {r.value} <span className="font-normal text-slate-500">（{r.benchmark}）</span>
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
      <p className="text-[10px] text-amber-600 mt-3 font-bold">
        ※ 上記リスク要因に対する改善条件（担保・保証人追加等）を付した上での承認を検討してください。
      </p>
    </div>
  );
}

type Action = { label: string; detail: string; priority: 'must' | 'should' };

function buildRecommendedActions(risks: RiskFactor[]): Action[] {
  const actions: Action[] = [];
  const labels = risks.map(r => r.label);

  if (labels.includes('営業利益率')) {
    actions.push({ label: '直近試算表・受注状況の追加提出を要求', detail: '利益率改善の見通しを確認する', priority: 'must' });
    actions.push({ label: '担保（動産・不動産）の設定を検討', detail: '利益率低水準のリスクを担保でカバー', priority: 'should' });
  }
  if (labels.includes('自己資本比率')) {
    actions.push({ label: '代表者連帯保証の取得', detail: '自己資本が薄い場合の信用補完', priority: 'must' });
    actions.push({ label: '追加担保（不動産・定期預金等）の検討', detail: '自己資本不足を担保で補填', priority: 'should' });
  }
  if (labels.includes('デフォルト確率（PD）')) {
    actions.push({ label: '信用保険（リース信用保険）の付保を検討', detail: 'PD高水準のリスクヘッジとして有効', priority: 'must' });
    actions.push({ label: 'リース期間を短縮して総エクスポージャーを圧縮', detail: '長期リースはリスクを拡大させる', priority: 'should' });
  }
  if (labels.includes('格付スコア')) {
    actions.push({ label: '代表者・第三者保証の強化', detail: '格付が低い場合の信用力補完', priority: 'must' });
    actions.push({ label: '物件担保（リース物件）の条件追加', detail: '中途解約時の残価リスク軽減', priority: 'should' });
  }
  if (labels.includes('流動比率')) {
    actions.push({ label: '運転資金・資金繰り表の提出を要求', detail: '短期支払能力の確認', priority: 'must' });
    actions.push({ label: 'リース料の分割実行・段階払いの検討', detail: '初期月次負担を軽減して流動性を確保', priority: 'should' });
  }
  if (labels.includes('負債比率')) {
    actions.push({ label: '既存借入の返済スケジュール確認', detail: '過多な負債が返済圧迫につながるリスクを評価', priority: 'must' });
  }

  // 共通
  actions.push({ label: '3期分の決算書（勘定科目内訳含む）の確認', detail: '財務トレンドの確認', priority: 'should' });
  actions.push({ label: '主要取引先・支払実績の確認', detail: '業容・信用状況の定性確認', priority: 'should' });

  // 優先度でソート：mustを先に
  return actions.sort((a, b) => (a.priority === 'must' ? -1 : 1) - (b.priority === 'must' ? -1 : 1));
}

function RecommendedActionsPanel({ score, inputs, result }: {
  score: number;
  inputs: Record<string, unknown>;
  result: Record<string, unknown>;
}) {
  if (score < 60 || score >= 70) return null;
  const risks = extractRiskFactors(inputs, result);
  const actions = buildRecommendedActions(risks);

  return (
    <div className="mb-5 p-4 bg-blue-50 border border-blue-200 rounded-xl">
      <div className="flex items-center gap-2 mb-3">
        <ClipboardList className="w-5 h-5 text-blue-600 flex-shrink-0" />
        <span className="font-black text-blue-800 text-sm">条件付き承認 — 推奨アクション</span>
      </div>
      <div className="space-y-2">
        {actions.map((a, i) => (
          <div key={i} className={`flex items-start gap-2.5 p-2.5 rounded-lg border ${a.priority === 'must' ? 'bg-white border-blue-300' : 'bg-blue-50/60 border-blue-100'}`}>
            {a.priority === 'must'
              ? <CheckCircle2 className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
              : <CheckCircle2 className="w-4 h-4 text-blue-300 flex-shrink-0 mt-0.5" />}
            <div>
              <p className={`text-xs font-black ${a.priority === 'must' ? 'text-blue-800' : 'text-blue-600'}`}>
                {a.priority === 'must' && <span className="mr-1 text-[9px] bg-blue-600 text-white rounded px-1 py-0.5">必須</span>}
                {a.label}
              </p>
              <p className="text-[11px] text-slate-500 mt-0.5">{a.detail}</p>
            </div>
          </div>
        ))}
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
  const [generating, setGenerating] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [caseDetail, setCaseDetail] = useState<{ inputs: Record<string, unknown>; result: Record<string, unknown> } | null>(null);

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

  const handleSelectCase = useCallback(async (id: string) => {
    setSelectedId(id);
    setReport(null);
    setError(null);
    setCaseDetail(null);
    if (!id) return;
    try {
      const detail = await apiClient.get(`/api/cases/${id}`);
      const d = detail.data;
      setCaseDetail({ inputs: d.inputs || {}, result: d.result || {} });
    } catch {
      // detail取得失敗は無視（レポート生成時に再取得する）
    }
  }, []);

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
      setCaseDetail({ inputs, result: result_data });
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
  const score = selectedCase?.score ?? null;
  const isConditional = score !== null && score >= 60 && score < 70;

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
                onChange={e => handleSelectCase(e.target.value)}
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
              <div className={`mt-3 p-3 rounded-xl text-xs space-y-1 ${isConditional ? 'bg-amber-50 border border-amber-200' : 'bg-slate-50'}`}>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${STATUS_DOT[selectedCase.final_status] || 'bg-slate-300'}`} />
                  <span className="font-black text-slate-700">{selectedCase.final_status}</span>
                  {isConditional && (
                    <span className="ml-auto text-[10px] font-black text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded-full">条件付き承認</span>
                  )}
                </div>
                <div className="text-slate-500">スコア: <span className={`font-black ${isConditional ? 'text-amber-700' : 'text-slate-700'}`}>{score != null ? Math.round(score) : '—'}</span></div>
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

                {/* REV-027: 条件付き承認リスクパネル */}
                {score !== null && caseDetail && (
                  <ConditionalRiskPanel
                    score={score}
                    inputs={caseDetail.inputs}
                    result={caseDetail.result}
                  />
                )}
                {/* REV-019: 推奨アクションパネル */}
                {score !== null && caseDetail && (
                  <RecommendedActionsPanel
                    score={score}
                    inputs={caseDetail.inputs}
                    result={caseDetail.result}
                  />
                )}

                <MarkdownBlock md={report} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
