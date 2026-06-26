"use client";

import { useState, useRef } from "react";
import { Upload, FileImage, CheckCircle, AlertCircle, Loader2, ScanText } from "lucide-react";
import { apiClient } from "@/lib/api";
import { ScoringFormData } from "@/types";

type OcrResult = {
  nenshu?: number | null;
  gross_profit?: number | null;
  op_profit?: number | null;
  ord_profit?: number | null;
  net_income?: number | null;
  net_assets?: number | null;
  total_assets?: number | null;
  depreciation?: number | null;
  dep_expense?: number | null;
  rent?: number | null;
  rent_expense?: number | null;
  machines?: number | null;
  other_assets?: number | null;
  detected_fields?: string[];
  missing_fields?: string[];
  confidence?: number;
  error?: string;
};

const FIELD_LABELS: Record<string, string> = {
  nenshu: "売上高",
  gross_profit: "売上総利益",
  op_profit: "営業利益",
  ord_profit: "経常利益",
  net_income: "純利益",
  net_assets: "純資産",
  total_assets: "総資産",
  depreciation: "減価償却(BS)",
  dep_expense: "減価償却(PL)",
  rent: "賃借料(BS)",
  rent_expense: "賃借料(PL)",
  machines: "機械装置等",
  other_assets: "その他固定資産",
};

type Props = {
  onApply: (fields: Partial<ScoringFormData>) => void;
};

export default function OcrUpload({ onApply }: Props) {
  const [ocrResult, setOcrResult] = useState<OcrResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setFileName(file.name);
    setLoading(true);
    setOcrResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await apiClient.post("/api/ocr", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setOcrResult(res.data as OcrResult);
    } catch {
      setOcrResult({ error: "OCR処理に失敗しました。FastAPIサーバーが起動しているか確認してください。" });
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleApply = () => {
    if (!ocrResult) return;
    const numericFields = [
      "nenshu", "gross_profit", "op_profit", "ord_profit", "net_income",
      "net_assets", "total_assets", "depreciation", "dep_expense",
      "rent", "rent_expense", "machines", "other_assets",
    ] as (keyof ScoringFormData)[];
    const fields: Partial<ScoringFormData> = {};
    for (const field of numericFields) {
      const val = (ocrResult as Record<string, unknown>)[field as string];
      if (typeof val === "number") {
        (fields as Record<string, unknown>)[field as string] = val;
      }
    }
    onApply(fields);
  };

  const detectedCount = ocrResult?.detected_fields?.length ?? 0;
  const confidence = ocrResult?.confidence ?? 0;

  return (
    <div className="space-y-3">
      {/* アップロードエリア */}
      <div
        className="border-2 border-dashed border-indigo-200 bg-indigo-50/40 rounded-2xl p-6 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-colors"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        {loading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
            <span className="text-sm font-bold text-indigo-600">Gemini Vision で読み取り中...</span>
            <span className="text-xs text-slate-400">財務項目を自動抽出しています</span>
          </div>
        ) : fileName ? (
          <div className="flex flex-col items-center gap-2">
            <FileImage className="w-8 h-8 text-indigo-400" />
            <span className="text-sm font-bold text-slate-700 break-all">{fileName}</span>
            <span className="text-xs text-slate-400">別のファイルを選択するにはクリック</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="w-8 h-8 text-indigo-400" />
            <span className="text-sm font-bold text-indigo-700">決算書をドロップ または クリックして選択</span>
            <span className="text-xs text-slate-500">JPG / PNG / PDF（最大20MB）</span>
          </div>
        )}
      </div>

      {/* エラー */}
      {ocrResult?.error && (
        <div className="flex items-start gap-2 rounded-xl bg-rose-50 border border-rose-200 px-4 py-3">
          <AlertCircle className="w-4 h-4 text-rose-500 mt-0.5 flex-shrink-0" />
          <span className="text-sm font-bold text-rose-700">{ocrResult.error}</span>
        </div>
      )}

      {/* OCR結果 */}
      {ocrResult && !ocrResult.error && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <CheckCircle className="w-4 h-4 text-emerald-500 flex-shrink-0" />
            <span className="text-sm font-black text-slate-700">
              {detectedCount}項目を読み取り完了
            </span>
            <span className={`ml-auto text-[10px] font-black px-2 py-1 rounded-full border ${
              confidence >= 0.7
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : confidence >= 0.4
                ? "bg-amber-50 text-amber-700 border-amber-200"
                : "bg-rose-50 text-rose-700 border-rose-200"
            }`}>
              信頼度 {Math.round(confidence * 100)}%
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {Object.entries(FIELD_LABELS).map(([key, label]) => {
              const val = (ocrResult as Record<string, unknown>)[key];
              const hasValue = typeof val === "number";
              return (
                <div
                  key={key}
                  className={`rounded-xl border px-3 py-2 ${
                    hasValue
                      ? "border-emerald-200 bg-emerald-50"
                      : "border-slate-100 bg-slate-50 opacity-40"
                  }`}
                >
                  <div className="text-[10px] font-black text-slate-400">{label}</div>
                  <div className={`text-sm font-black ${hasValue ? "text-slate-800" : "text-slate-400"}`}>
                    {hasValue ? `${(val as number).toLocaleString("ja-JP")}百万円` : "-"}
                  </div>
                </div>
              );
            })}
          </div>

          {detectedCount > 0 && (
            <button
              type="button"
              onClick={handleApply}
              className="w-full py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-xl font-bold shadow-lg shadow-indigo-100 hover:shadow-xl hover:translate-y-[-1px] transition-all text-sm flex items-center justify-center gap-2"
            >
              <ScanText className="w-4 h-4" />
              財務フォームに反映する（{detectedCount}項目）
            </button>
          )}
        </div>
      )}
    </div>
  );
}
