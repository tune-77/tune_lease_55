"use client";

import React, { useState } from 'react';
import { FileText, ClipboardList, Copy, Printer, CheckCircle } from 'lucide-react';
import ReportGenerator from '@/components/analysis/ReportGenerator';

type Tab = 'ringi' | 'mitsumori';

const LEASE_PERIODS = [2, 3, 4, 5, 6, 7] as const;

type FormState = {
  applyDate: string;
  applicantName: string;
  counterparty: string;
  subject: string;
  leasePeriod: number;
  acquisitionValue: string;
  remarks: string;
};

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function RingiPage() {
  const [activeTab, setActiveTab] = useState<Tab>('ringi');
  const [form, setForm] = useState<FormState>({
    applyDate: todayStr(),
    applicantName: '',
    counterparty: '',
    subject: '',
    leasePeriod: 5,
    acquisitionValue: '',
    remarks: '',
  });
  const [preview, setPreview] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleGenerate = () => {
    const text = `リース見積依頼書

申請日：${form.applyDate}
申請者：${form.applicantName}

─────────────────────────────
取引先：${form.counterparty}
物　件：${form.subject}
取得価額：${form.acquisitionValue}百万円
希望期間：${form.leasePeriod}年
─────────────────────────────
備考：${form.remarks || '（なし）'}

上記物件についてリース見積をご依頼申し上げます。`;
    setPreview(text);
    setCopied(false);
  };

  const handleCopy = async () => {
    if (!preview) return;
    await navigator.clipboard.writeText(preview);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  const inputClass =
    'w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition';
  const labelClass = 'block text-xs font-bold text-slate-400 mb-1.5 uppercase tracking-wider';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 lg:p-10">
      <div className="max-w-4xl mx-auto">
        {/* ページヘッダー */}
        <div className="mb-8 flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <ClipboardList className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">業務書類</h1>
            <p className="text-sm text-slate-400 font-medium">稟議書・見積依頼書の生成</p>
          </div>
        </div>

        {/* タブ */}
        <div className="flex gap-1 mb-8 bg-slate-900 rounded-2xl p-1 border border-slate-800 w-fit">
          <button
            onClick={() => setActiveTab('ringi')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
              activeTab === 'ringi'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <FileText className="w-4 h-4" />
            稟議書
          </button>
          <button
            onClick={() => setActiveTab('mitsumori')}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
              activeTab === 'mitsumori'
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <ClipboardList className="w-4 h-4" />
            見積依頼書
          </button>
        </div>

        {/* タブ1: 稟議書 */}
        {activeTab === 'ringi' && (
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
            <div className="mb-4">
              <h2 className="text-lg font-black text-white mb-1">稟議書ジェネレーター</h2>
              <p className="text-sm text-slate-400">
                審査スコアリング結果をもとに稟議書を自動生成します。スコアリング実行後にご利用ください。
              </p>
            </div>
            <ReportGenerator apiResult={null} formData={null} />
          </div>
        )}

        {/* タブ2: 見積依頼書 */}
        {activeTab === 'mitsumori' && (
          <div className="space-y-6">
            {/* 入力フォーム */}
            <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
              <h2 className="text-lg font-black text-white mb-6">リース見積依頼書 入力フォーム</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className={labelClass}>申請日</label>
                  <input
                    type="date"
                    name="applyDate"
                    value={form.applyDate}
                    onChange={handleChange}
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>申請者名</label>
                  <input
                    type="text"
                    name="applicantName"
                    value={form.applicantName}
                    onChange={handleChange}
                    placeholder="例：山田 太郎"
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>取引先企業名</label>
                  <input
                    type="text"
                    name="counterparty"
                    value={form.counterparty}
                    onChange={handleChange}
                    placeholder="例：株式会社○○リース"
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>リース対象物件</label>
                  <input
                    type="text"
                    name="subject"
                    value={form.subject}
                    onChange={handleChange}
                    placeholder="例：医療用CT装置一式"
                    className={inputClass}
                  />
                </div>
                <div>
                  <label className={labelClass}>希望リース期間</label>
                  <select
                    name="leasePeriod"
                    value={form.leasePeriod}
                    onChange={handleChange}
                    className={inputClass}
                  >
                    {LEASE_PERIODS.map(y => (
                      <option key={y} value={y}>{y}年</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className={labelClass}>物件取得価額（百万円）</label>
                  <input
                    type="number"
                    name="acquisitionValue"
                    value={form.acquisitionValue}
                    onChange={handleChange}
                    placeholder="例：50"
                    min={0}
                    className={inputClass}
                  />
                </div>
                <div className="md:col-span-2">
                  <label className={labelClass}>備考（任意）</label>
                  <textarea
                    name="remarks"
                    value={form.remarks}
                    onChange={handleChange}
                    placeholder="特記事項があればご記入ください"
                    rows={3}
                    className={`${inputClass} resize-none`}
                  />
                </div>
              </div>
              <div className="mt-6">
                <button
                  onClick={handleGenerate}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all duration-200 shadow-lg shadow-blue-600/20 active:scale-95"
                >
                  <ClipboardList className="w-4 h-4" />
                  見積依頼書を生成
                </button>
              </div>
            </div>

            {/* プレビュー */}
            {preview && (
              <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-black text-white">プレビュー</h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleCopy}
                      className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all duration-200 border ${
                        copied
                          ? 'bg-emerald-600/20 border-emerald-500/40 text-emerald-400'
                          : 'bg-slate-800 border-slate-700 text-slate-300 hover:text-white hover:border-slate-600'
                      }`}
                    >
                      {copied ? (
                        <>
                          <CheckCircle className="w-4 h-4" />
                          コピー済み
                        </>
                      ) : (
                        <>
                          <Copy className="w-4 h-4" />
                          コピー
                        </>
                      )}
                    </button>
                    <button
                      onClick={handlePrint}
                      className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-600 transition-all duration-200"
                    >
                      <Printer className="w-4 h-4" />
                      印刷
                    </button>
                  </div>
                </div>
                <pre className="bg-slate-950 rounded-xl border border-slate-800 p-5 text-sm text-slate-200 font-mono whitespace-pre-wrap leading-relaxed">
                  {preview}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
