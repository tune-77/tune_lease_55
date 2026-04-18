"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Calendar, Percent, Activity, Save, Download, AlertTriangle, CheckCircle, ChevronRight, Database, TrendingUp } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const TERM_COLS = ["r_2y","r_3y","r_4y","r_5y","r_6y","r_7y","r_8y","r_9y","r_over9y"] as const;
const TERM_LABELS: Record<string, string> = {
  r_2y: "2年以内", r_3y: "3年以内", r_4y: "4年以内",
  r_5y: "5年以内", r_6y: "6年以内", r_7y: "7年以内",
  r_8y: "8年以内", r_9y: "9年以内", r_over9y: "9年超",
};

type RateRow = {
  month: string; rate: number | null; note: string;
  r_2y: number|null; r_3y: number|null; r_4y: number|null;
  r_5y: number|null; r_6y: number|null; r_7y: number|null;
  r_8y: number|null; r_9y: number|null; r_over9y: number|null;
};

type FormState = { month: string; note: string } & Record<string, string>;

function makeDefaultForm(latest: RateRow | null, defaultMonth: string): FormState {
  const form: FormState = { month: defaultMonth, note: '' };
  for (const col of TERM_COLS) {
    form[col] = latest ? (latest[col] ?? '').toString() : '';
  }
  return form;
}

export default function InterestPage() {
  const [rates, setRates] = useState<RateRow[]>([]);
  const [current, setCurrent] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState<FormState>({ month: '', note: '', ...Object.fromEntries(TERM_COLS.map(c => [c, ''])) });
  const [submitting, setSubmitting] = useState(false);
  const [editRows, setEditRows] = useState<Record<string, Partial<RateRow>>>({});
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<'list' | 'seed' | 'chart'>('list');
  const [seeding, setSeeding] = useState(false);
  const [seedResult, setSeedResult] = useState<string | null>(null);
  const [toast, setToast] = useState<{msg: string; type: 'ok'|'err'} | null>(null);

  useEffect(() => {
    triggerMebuki('guide', '基準金利マスタ管理ですね！\n期間別9区分の金利を月次で管理します。');
    fetchAll();
  }, []);

  const showToast = (msg: string, type: 'ok'|'err' = 'ok') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const fetchAll = async () => {
    setLoading(true);
    try {
      const [ratesRes, currentRes] = await Promise.all([
        axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/interest`),
        axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/interest/current`),
      ]);
      const rateData: RateRow[] = ratesRes.data;
      const cur = currentRes.data;
      setRates(rateData);
      setCurrent(cur);
      // フォームのデフォルト月: 当月未登録→当月, 登録済→来月
      const defaultMonth = cur.current_rate_5y === null ? cur.current_month : cur.next_month;
      setForm(makeDefaultForm(cur.latest, defaultMonth));
    } catch (err) {
      triggerMebuki('reject', '金利データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const payload: any = { month: form.month, note: form.note };
      for (const col of TERM_COLS) {
        payload[col] = form[col] !== '' ? parseFloat(form[col]) : null;
      }
      await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/interest`, payload);
      triggerMebuki('approve', `${form.month} の基準金利を登録しました！`);
      showToast(`${form.month} を登録しました`);
      fetchAll();
    } catch {
      triggerMebuki('reject', '更新に失敗しました。');
      showToast('登録に失敗しました', 'err');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCellEdit = (month: string, col: string, val: string) => {
    setEditRows(prev => ({
      ...prev,
      [month]: { ...prev[month], [col]: val === '' ? null : parseFloat(val) }
    }));
  };

  const handleSaveGrid = async () => {
    const changed = Object.entries(editRows);
    if (!changed.length) { showToast('変更はありませんでした'); return; }
    setSaving(true);
    try {
      for (const [month, patch] of changed) {
        const original = rates.find(r => r.month === month);
        const payload: any = { month, note: original?.note ?? '' };
        for (const col of TERM_COLS) {
          payload[col] = (patch as any)[col] !== undefined ? (patch as any)[col] : (original as any)?.[col] ?? null;
        }
        await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/interest`, payload);
      }
      showToast(`${changed.length}件を保存しました`);
      setEditRows({});
      fetchAll();
    } catch {
      showToast('保存に失敗しました', 'err');
    } finally {
      setSaving(false);
    }
  };

  const handleSeed = async (overwrite: boolean) => {
    setSeeding(true);
    setSeedResult(null);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/interest/seed?overwrite=${overwrite}`);
      setSeedResult(`✅ ${res.data.inserted}件投入、${res.data.skipped}件スキップ`);
      fetchAll();
    } catch {
      setSeedResult('❌ 投入に失敗しました');
    } finally {
      setSeeding(false);
    }
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <Activity className="w-12 h-12 text-emerald-500 animate-spin" />
    </div>
  );

  const currentRate5y = current?.current_rate_5y;
  const nextRate5y = current?.next_rate_5y;

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Toast */}
      {toast && (
        <div className={`fixed top-6 right-6 z-50 px-6 py-3 rounded-2xl font-bold shadow-xl flex items-center gap-2 ${toast.type === 'ok' ? 'bg-emerald-600 text-white' : 'bg-rose-600 text-white'}`}>
          {toast.type === 'ok' ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Calendar className="w-8 h-8 text-emerald-500" />
          基準金利マスタ管理
        </h1>
        <p className="text-slate-500 font-bold mt-1">月次・リース期間別（9区分）の基準金利を管理します。</p>
      </div>

      {/* Current/Next month status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className={`p-5 rounded-2xl border flex items-center gap-4 ${currentRate5y === null ? 'bg-amber-50 border-amber-200' : 'bg-white border-slate-200'}`}>
          {currentRate5y === null
            ? <AlertTriangle className="w-8 h-8 text-amber-500 shrink-0" />
            : <CheckCircle className="w-8 h-8 text-emerald-500 shrink-0" />
          }
          <div>
            <div className="text-xs font-black text-slate-400 uppercase tracking-widest">当月 ({current?.current_month})</div>
            {currentRate5y === null
              ? <div className="text-sm font-bold text-amber-700 mt-0.5">⚠️ 未登録</div>
              : <div className="text-2xl font-black text-emerald-700">{currentRate5y.toFixed(2)}<span className="text-sm text-slate-400 ml-1">% (5年以内)</span></div>
            }
          </div>
        </div>
        <div className={`p-5 rounded-2xl border flex items-center gap-4 ${nextRate5y === null ? 'bg-slate-50 border-slate-200' : 'bg-white border-slate-200'}`}>
          <ChevronRight className="w-8 h-8 text-slate-400 shrink-0" />
          <div>
            <div className="text-xs font-black text-slate-400 uppercase tracking-widest">来月 ({current?.next_month})</div>
            {nextRate5y === null
              ? <div className="text-sm font-bold text-slate-500 mt-0.5">未登録</div>
              : <div className="text-2xl font-black text-slate-700">{nextRate5y.toFixed(2)}<span className="text-sm text-slate-400 ml-1">% (5年以内)</span></div>
            }
          </div>
        </div>
      </div>

      {/* Registration Form */}
      <div className="bg-white border border-slate-200 rounded-3xl shadow-sm overflow-hidden mb-6">
        <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
          <Percent className="w-5 h-5 text-emerald-500" />
          <span className="font-black text-slate-700 text-sm uppercase tracking-widest">月次金利更新</span>
          {current?.prev && <span className="ml-auto text-xs text-slate-400">前月値をデフォルト表示</span>}
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-black text-slate-600 mb-1.5">適用月 (YYYY-MM)</label>
              <input
                type="month"
                className="w-full bg-slate-50 border border-slate-200 p-3 rounded-xl font-bold text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500"
                value={form.month}
                onChange={e => setForm({ ...form, month: e.target.value })}
              />
            </div>
            <div>
              <label className="block text-xs font-black text-slate-600 mb-1.5">メモ（任意）</label>
              <input
                type="text"
                className="w-full bg-slate-50 border border-slate-200 p-3 rounded-xl text-sm text-slate-600 outline-none focus:ring-2 focus:ring-emerald-500"
                value={form.note}
                onChange={e => setForm({ ...form, note: e.target.value })}
                placeholder="例: 日銀利上げにより改定"
              />
            </div>
          </div>

          <div className="text-xs font-black text-slate-500 uppercase tracking-widest mb-3">リース期間別基準金利 (%)</div>
          <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-3 mb-5">
            {TERM_COLS.map(col => {
              const prev = current?.prev?.[col];
              return (
                <div key={col}>
                  <label className="block text-[10px] font-black text-slate-500 mb-1 text-center">{TERM_LABELS[col]}</label>
                  <input
                    type="number" step="0.01"
                    className="w-full bg-slate-50 border border-slate-200 p-2 rounded-lg font-mono text-sm text-center text-emerald-700 outline-none focus:ring-2 focus:ring-emerald-400"
                    value={form[col]}
                    onChange={e => setForm({ ...form, [col]: e.target.value })}
                    placeholder={prev !== undefined && prev !== null ? prev.toFixed(2) : '0.00'}
                  />
                  {prev !== null && prev !== undefined && (
                    <div className="text-[9px] text-center text-slate-400 mt-0.5">前月: {prev.toFixed(2)}</div>
                  )}
                </div>
              );
            })}
          </div>

          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-black py-3.5 rounded-2xl shadow-lg shadow-emerald-500/20 transition-all flex items-center justify-center gap-2"
          >
            {submitting ? <Activity className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            登録する
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-4">
        {[
          { id: 'list', label: '登録一覧・編集' },
          { id: 'seed', label: '初期データ一括投入' },
          { id: 'chart', label: '📈 金利推移グラフ' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-5 py-2.5 rounded-xl font-black text-sm transition-all ${activeTab === tab.id ? 'bg-emerald-600 text-white shadow-md' : 'bg-white border border-slate-200 text-slate-500 hover:text-slate-700'}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab: List & Grid Edit */}
      {activeTab === 'list' && (
        <div className="bg-white border border-slate-200 rounded-3xl shadow-sm overflow-hidden">
          <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
            <span className="font-black text-slate-700 text-sm uppercase tracking-widest">登録一覧（直近60件）</span>
            {Object.keys(editRows).length > 0 && (
              <button
                onClick={handleSaveGrid}
                disabled={saving}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-black px-5 py-2 rounded-xl text-sm flex items-center gap-2 transition-colors"
              >
                {saving ? <Activity className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                変更を保存 ({Object.keys(editRows).length}件)
              </button>
            )}
          </div>
          <div className="overflow-x-auto -mx-0">
            <table className="w-full text-left text-xs border-collapse min-w-[900px]">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  <th className="px-4 py-3 font-black text-slate-500 uppercase tracking-widest sticky left-0 bg-slate-50">適用月</th>
                  {TERM_COLS.map(col => (
                    <th key={col} className="px-3 py-3 font-black text-slate-500 uppercase tracking-widest text-center">{TERM_LABELS[col]}</th>
                  ))}
                  <th className="px-4 py-3 font-black text-slate-500 uppercase tracking-widest">メモ</th>
                </tr>
              </thead>
              <tbody>
                {rates.map((row, i) => {
                  const edits = editRows[row.month] || {};
                  const isEdited = !!editRows[row.month];
                  return (
                    <tr key={row.month} className={`border-b border-slate-50 transition-colors ${isEdited ? 'bg-emerald-50/60' : i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'} hover:bg-emerald-50/40`}>
                      <td className={`px-4 py-2 font-black text-slate-700 sticky left-0 ${isEdited ? 'bg-emerald-50/60' : i % 2 === 0 ? 'bg-white' : 'bg-slate-50/30'}`}>
                        {row.month}
                        {row.month === current?.current_month && (
                          <span className="ml-1 text-[8px] bg-emerald-500 text-white px-1.5 py-0.5 rounded-full uppercase font-black">当月</span>
                        )}
                      </td>
                      {TERM_COLS.map(col => {
                        const val = edits[col as keyof typeof edits] !== undefined ? edits[col as keyof typeof edits] : row[col as keyof RateRow];
                        return (
                          <td key={col} className="px-1 py-1 text-center">
                            <input
                              type="number" step="0.01"
                              className={`w-16 text-center font-mono rounded-lg border px-1 py-1 text-xs outline-none focus:ring-1 focus:ring-emerald-400 ${isEdited && edits[col as keyof typeof edits] !== undefined ? 'border-emerald-400 bg-emerald-50 text-emerald-700 font-black' : 'border-transparent bg-transparent text-slate-600 hover:border-slate-300'}`}
                              value={val !== null && val !== undefined ? Number(val).toFixed(2) : ''}
                              onChange={e => handleCellEdit(row.month, col, e.target.value)}
                            />
                          </td>
                        );
                      })}
                      <td className="px-4 py-2 text-slate-400 italic">{row.note || '-'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab: Chart */}
      {activeTab === 'chart' && (() => {
        const COLORS = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#ef4444','#ec4899','#06b6d4','#84cc16','#f97316'];
        // oldest first
        const chartData = [...rates].reverse().map(row => {
          const entry: Record<string, string | number | null> = { month: row.month };
          for (const col of TERM_COLS) entry[col] = row[col as keyof RateRow] as number | null;
          return entry;
        });
        const allVals = chartData.flatMap(d => TERM_COLS.map(c => d[c] as number | null).filter(v => v !== null)) as number[];
        const yMin = allVals.length ? Math.floor(Math.min(...allVals) * 10) / 10 : 0;
        const yMax = allVals.length ? Math.ceil(Math.max(...allVals) * 10) / 10 : 5;
        return (
          <div className="bg-white border border-slate-200 rounded-3xl shadow-sm overflow-hidden">
            <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-500" />
              <span className="font-black text-slate-700 text-sm uppercase tracking-widest">期間別基準金利 推移グラフ（{rates.length}ヶ月）</span>
            </div>
            <div className="p-6">
              {chartData.length === 0 ? (
                <div className="text-center text-slate-400 font-bold py-16">データがありません。先に初期データを投入してください。</div>
              ) : (
                <ResponsiveContainer width="100%" height={420}>
                  <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis
                      dataKey="month"
                      tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 700 }}
                      angle={-45}
                      textAnchor="end"
                      interval={Math.floor(chartData.length / 10)}
                    />
                    <YAxis
                      domain={[yMin, yMax]}
                      tickFormatter={v => `${v.toFixed(2)}%`}
                      tick={{ fontSize: 11, fill: '#64748b', fontWeight: 700 }}
                      width={60}
                    />
                    <Tooltip
                      formatter={(val: unknown, name: unknown) => [`${(val as number)?.toFixed(3)}%`, TERM_LABELS[name as string] ?? String(name)]}
                      labelStyle={{ fontWeight: 700, color: '#1e293b' }}
                      contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', fontSize: 12 }}
                    />
                    <Legend
                      formatter={(val: string) => TERM_LABELS[val] ?? val}
                      wrapperStyle={{ fontSize: 11, fontWeight: 700, paddingTop: 8 }}
                    />
                    {TERM_COLS.map((col, i) => (
                      <Line
                        key={col}
                        type="monotone"
                        dataKey={col}
                        stroke={COLORS[i]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        );
      })()}

      {/* Tab: Seed */}
      {activeTab === 'seed' && (
        <div className="bg-white border border-slate-200 rounded-3xl shadow-sm p-8 max-w-xl">
          <div className="flex items-center gap-3 mb-4">
            <Database className="w-6 h-6 text-emerald-500" />
            <h3 className="font-black text-slate-800 text-lg">初期データ一括投入（42件）</h3>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 mb-6 text-sm text-slate-600 leading-relaxed">
            基準金利テーブル（2022/11〜2026/4）を一括登録します。<br />
            「上書きなし」は既存月をスキップ、「上書きあり」は全件更新します。
          </div>
          <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200 mb-6">
            <span className="text-sm font-black text-slate-600">現在の登録件数</span>
            <span className="text-2xl font-black text-slate-800">{rates.length}<span className="text-sm text-slate-400 ml-1">件</span></span>
          </div>
          {seedResult && (
            <div className={`p-4 rounded-xl font-bold mb-4 text-sm ${seedResult.startsWith('✅') ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'}`}>
              {seedResult}
            </div>
          )}
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => handleSeed(false)}
              disabled={seeding}
              className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-black py-3.5 rounded-2xl flex items-center justify-center gap-2 transition-colors"
            >
              {seeding ? <Activity className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              上書きなし投入
            </button>
            <button
              onClick={() => handleSeed(true)}
              disabled={seeding}
              className="bg-white hover:bg-slate-50 disabled:opacity-50 border border-slate-300 text-slate-700 font-black py-3.5 rounded-2xl flex items-center justify-center gap-2 transition-colors"
            >
              {seeding ? <Activity className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              上書きあり（全件更新）
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
