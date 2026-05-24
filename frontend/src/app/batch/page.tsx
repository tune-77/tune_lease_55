"use client";

import React, { useMemo, useState } from "react";
import { apiClient } from "../../lib/api";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Download,
  FileDown,
  FileSpreadsheet,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";

type BatchSummary = {
  total: number;
  good: number;
  border: number;
  rejected: number;
  errors: number;
  standard_scoring: number;
  saved_count: number;
  with_result: number;
  excluded_saved_count: number;
  failed_count?: number;
  backup_message?: string;
  training_message?: string;
};

type BatchResponse = {
  summary: BatchSummary;
  preview: Record<string, unknown>[];
  rows: Record<string, unknown>[];
  csv: string;
  batch_token?: string;
};

const RESULT_COLUMNS = [
  "取引先ID",
  "企業名",
  "判定",
  "総合スコア",
  "借手スコア",
  "物件スコア",
  "スコアリング",
  "bench_score",
  "ind_score",
  "信用リスク群判定",
  "エラー",
];

const MAX_CSV_BYTES = 5 * 1024 * 1024;

function bytesToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function valueText(value: unknown) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(1);
  return String(value);
}

function judgmentClass(judgment: unknown) {
  const text = String(judgment || "");
  if (text === "良決" || text === "承認圏内") return "bg-emerald-50 text-emerald-700 border-emerald-200";
  if (text === "ボーダー" || text === "要審議") return "bg-amber-50 text-amber-700 border-amber-200";
  if (text === "否決" || text === "エラー") return "bg-rose-50 text-rose-700 border-rose-200";
  if (text === "信用リスク群分離") return "bg-violet-50 text-violet-700 border-violet-200";
  return "bg-slate-50 text-slate-600 border-slate-200";
}

function StatCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "emerald" | "amber" | "rose" | "slate";
}) {
  const toneClass = {
    emerald: "text-emerald-700 bg-emerald-50 border-emerald-200",
    amber: "text-amber-700 bg-amber-50 border-amber-200",
    rose: "text-rose-700 bg-rose-50 border-rose-200",
    slate: "text-slate-700 bg-slate-50 border-slate-200",
  }[tone];
  return (
    <div className={`border rounded-lg p-4 ${toneClass}`}>
      <div className="text-xs font-black tracking-wider">{label}</div>
      <div className="mt-2 text-2xl font-black">{value}</div>
    </div>
  );
}

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [batchToken, setBatchToken] = useState<string | null>(null);

  const resultColumns = useMemo(() => {
    if (!result?.rows?.length) return RESULT_COLUMNS;
    const existing = new Set(Object.keys(result.rows[0]));
    return RESULT_COLUMNS.filter((col) => existing.has(col));
  }, [result]);

  const runBatch = async () => {
    if (!file) {
      setMessage({ ok: false, text: "CSVファイルを選択してください" });
      return;
    }
    if (file.size > MAX_CSV_BYTES) {
      setMessage({ ok: false, text: "CSVファイルが大きすぎます。上限は5MBです。" });
      return;
    }
    setRunning(true);
    setMessage(null);
    setResult(null);
    setBatchToken(null);
    try {
      const csvBase64 = bytesToBase64(await file.arrayBuffer());
      const res = await apiClient.post<BatchResponse>("/api/batch/score", {
        csv_base64: csvBase64,
      });
      setResult(res.data);
      setBatchToken(res.data.batch_token || null);
      setMessage({ ok: true, text: "一括審査が完了しました。結果を確認してから保存できます。" });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setMessage({ ok: false, text: detail || "バッチ審査に失敗しました" });
    } finally {
      setRunning(false);
    }
  };

  const saveResult = async () => {
    if (!batchToken || !result) return;
    if (!confirm("表示中の判定結果を過去案件DBへ保存します。実行してよろしいですか？")) return;
    setSaving(true);
    setMessage(null);
    try {
      const res = await apiClient.post<BatchResponse>("/api/batch/save", {
        batch_token: batchToken,
        confirmed: true,
      });
      setResult(res.data);
      const saved = res.data.summary.saved_count + res.data.summary.excluded_saved_count;
      const failed = res.data.summary.failed_count || 0;
      setMessage({
        ok: failed === 0,
        text: failed === 0
          ? `${saved}件を過去案件DBへ保存しました。`
          : `${saved}件を保存しましたが、${failed}件は保存できませんでした。`,
      });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setMessage({ ok: false, text: detail || "DB保存に失敗しました" });
    } finally {
      setSaving(false);
    }
  };

  const downloadResultCsv = () => {
    if (!result?.csv) return;
    const blob = new Blob([result.csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_shinsa_result.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
            <FileSpreadsheet className="w-8 h-8 text-yellow-500" />
            バッチ審査
          </h1>
          <p className="text-slate-500 font-medium mt-1">
            CSVをアップロードして複数案件を一括スコアリングします。
          </p>
        </div>
        <a
          href="/api/batch/template"
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-slate-200 bg-white text-slate-700 text-sm font-black hover:bg-slate-50"
        >
          <FileDown className="w-4 h-4" />
          テンプレートCSV
        </a>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-1 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
            <div className="text-xs font-black text-slate-500 tracking-wider mb-3">CSVアップロード</div>
            <label className="block border-2 border-dashed border-slate-300 rounded-lg p-5 text-center hover:border-yellow-400 hover:bg-yellow-50/40 transition-colors cursor-pointer">
              <Upload className="w-8 h-8 mx-auto text-slate-400 mb-3" />
              <div className="text-sm font-black text-slate-700">
                {file ? file.name : "CSVファイルを選択"}
              </div>
              <div className="text-xs text-slate-400 mt-1">UTF-8 / Shift-JIS 対応・上限5MB/1000件</div>
              <input
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(e) => {
                  setFile(e.target.files?.[0] || null);
                  setResult(null);
                  setMessage(null);
                  setBatchToken(null);
                }}
              />
            </label>

            <button
              onClick={runBatch}
              disabled={!file || running}
              className="mt-4 w-full inline-flex items-center justify-center gap-2 py-3 rounded-lg bg-yellow-500 hover:bg-yellow-400 text-slate-950 font-black text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              一括スコアリング実行
            </button>

            <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-500 leading-relaxed">
              判定結果を確認してから、結果表の上に表示される保存ボタンで過去案件DBへ登録します。
            </div>
          </div>

          {message && (
            <div className={`rounded-lg border p-4 text-sm font-bold flex items-start gap-3 ${
              message.ok ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-rose-50 border-rose-200 text-rose-700"
            }`}>
              {message.ok ? <CheckCircle2 className="w-5 h-5 shrink-0" /> : <AlertCircle className="w-5 h-5 shrink-0" />}
              <span>{message.text}</span>
            </div>
          )}

          {result?.summary.backup_message && (
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-4 text-sm font-bold text-blue-700">
              {result.summary.backup_message}
            </div>
          )}
          {result?.summary.training_message && (
            <div className="rounded-lg border border-violet-200 bg-violet-50 p-4 text-sm font-bold text-violet-700">
              {result.summary.training_message}
            </div>
          )}
        </div>

        <div className="xl:col-span-2 space-y-6">
          {result ? (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <StatCard label="総件数" value={`${result.summary.total}件`} tone="slate" />
                <StatCard label="良決" value={`${result.summary.good}件`} tone="emerald" />
                <StatCard label="ボーダー" value={`${result.summary.border}件`} tone="amber" />
                <StatCard label="否決/エラー" value={`${result.summary.rejected + result.summary.errors}件`} tone="rose" />
              </div>

              <div className="bg-white border border-slate-200 rounded-lg shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-200 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h2 className="text-sm font-black text-slate-700">判定結果</h2>
                    <p className="text-xs text-slate-500 mt-1">
                      標準モード {result.summary.standard_scoring}件 / 簡易モード {result.summary.total - result.summary.standard_scoring}件
                    </p>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <button
                      onClick={saveResult}
                      disabled={!batchToken || saving || result.summary.saved_count + result.summary.excluded_saved_count > 0}
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-black hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                      DB保存
                    </button>
                    <button
                      onClick={downloadResultCsv}
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-slate-900 text-white text-sm font-black hover:bg-slate-700"
                    >
                      <Download className="w-4 h-4" />
                      結果CSV
                    </button>
                  </div>
                </div>
                <div className="overflow-auto max-h-[560px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-slate-50 border-b border-slate-200">
                      <tr>
                        {resultColumns.map((col) => (
                          <th key={col} className="px-3 py-3 text-left text-xs font-black text-slate-500 whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.rows.map((row, index) => (
                        <tr key={index} className="hover:bg-slate-50">
                          {resultColumns.map((col) => (
                            <td key={col} className="px-3 py-3 text-slate-700 whitespace-nowrap">
                              {col === "判定" ? (
                                <span className={`inline-flex items-center px-2 py-1 rounded-md border text-xs font-black ${judgmentClass(row[col])}`}>
                                  {valueText(row[col])}
                                </span>
                              ) : col === "エラー" && row[col] ? (
                                <span className="inline-flex items-center gap-1 text-rose-600 font-bold">
                                  <XCircle className="w-4 h-4" />
                                  {valueText(row[col])}
                                </span>
                              ) : (
                                valueText(row[col])
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-10 min-h-80 flex flex-col items-center justify-center text-center">
              <FileSpreadsheet className="w-16 h-16 text-slate-300 mb-4" />
              <h2 className="text-lg font-black text-slate-700">CSVを選択して一括審査を実行</h2>
              <p className="text-sm text-slate-500 mt-2 max-w-lg">
                Streamlit版と同じバッチ審査ロジックをFastAPI経由で実行します。テンプレートの列名はそのまま使えます。
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
