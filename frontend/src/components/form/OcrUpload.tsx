'use client';
import React, { useRef, useState } from 'react';
import { apiClient } from '../../lib/api';

export type OcrResult = {
  nenshu: number | null;
  gross_profit: number | null;
  op_profit: number | null;
  ord_profit: number | null;
  net_income: number | null;
  net_assets: number | null;
  total_assets: number | null;
  depreciation: number | null;
  dep_expense: number | null;
  rent: number | null;
  rent_expense: number | null;
  machines: number | null;
  other_assets: number | null;
};

const FIELD_LABELS: Record<keyof OcrResult, string> = {
  nenshu: '売上高',
  gross_profit: '売上総利益',
  op_profit: '営業利益',
  ord_profit: '経常利益',
  net_income: '当期純利益',
  net_assets: '純資産',
  total_assets: '総資産',
  depreciation: '減価償却費（資産）',
  dep_expense: '減価償却費（経費）',
  rent: '賃借料（資産）',
  rent_expense: '賃借料（経費）',
  machines: '機械装置・運搬具',
  other_assets: 'その他固定資産',
};

type Props = {
  onApply: (data: OcrResult) => void;
};

export default function OcrUpload({ onApply }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<OcrResult | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setError(null);
    setPreview(null);
  };

  const handleRead = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setPreview(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await apiClient.post<OcrResult | { error: string }>('/api/ocr', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = res.data;
      if ('error' in data) {
        setError(data.error);
      } else {
        setPreview(data);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(`OCRエラー: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const handleApply = () => {
    if (preview) {
      onApply(preview);
      setPreview(null);
      setFile(null);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const previewEntries = preview
    ? (Object.entries(preview) as [keyof OcrResult, number | null][]).filter(
        ([, v]) => v !== null
      )
    : [];

  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">
        この画像はAI（Gemini）で処理されます。個人情報を含む場合はマスキングしてからアップロードしてください。
      </p>

      <div className="flex items-center gap-3 flex-wrap">
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
          onChange={handleFileChange}
          className="text-sm text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-100 file:text-blue-700 hover:file:bg-blue-200"
        />
        <button
          type="button"
          onClick={handleRead}
          disabled={!file || loading}
          className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              読み取り中…
            </>
          ) : (
            'OCR読み取り'
          )}
        </button>
      </div>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
      )}

      {preview && previewEntries.length > 0 && (
        <div className="bg-white border border-blue-200 rounded-xl p-4 space-y-3">
          <p className="text-sm font-semibold text-slate-700">
            以下の値でフォームに反映しますか？
          </p>
          <table className="w-full text-sm">
            <tbody>
              {previewEntries.map(([key, val]) => (
                <tr key={key} className="border-b border-slate-100 last:border-0">
                  <td className="py-1 pr-4 text-slate-600">{FIELD_LABELS[key]}</td>
                  <td className="py-1 text-right font-mono text-slate-800">
                    {val !== null ? val.toLocaleString() : '—'}
                  </td>
                  <td className="py-1 pl-2 text-slate-400 text-xs">百万円</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={handleApply}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-green-600 text-white hover:bg-green-700"
            >
              フォームに反映
            </button>
            <button
              type="button"
              onClick={() => setPreview(null)}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300"
            >
              キャンセル
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
