'use client';
import React, { useState } from 'react';
import SliderInput from '../SliderInput';
import { ScoringFormData } from '../../types';
import OcrUpload, { OcrResult } from './OcrUpload';

interface FormFinancialProps {
  data: ScoringFormData;
  onChange: (name: string, value: number) => void;
}

export default function FormFinancial({ data, onChange }: FormFinancialProps) {
  const [showOcr, setShowOcr] = useState(false);

  const handleOcrApply = (ocr: OcrResult) => {
    (Object.entries(ocr) as [keyof OcrResult, number | null][]).forEach(([key, val]) => {
      if (val !== null && val !== undefined) {
        onChange(key, val);
      }
    });
    setShowOcr(false);
  };

  return (
    <div className="space-y-6">

      {/* OCR読み取り（折りたたみ）*/}
      <div className="bg-blue-50 p-4 rounded-2xl border border-blue-100">
        <button
          type="button"
          onClick={() => setShowOcr(!showOcr)}
          className="flex items-center gap-2 text-sm font-semibold text-blue-700 hover:text-blue-900"
        >
          <span>{showOcr ? '▲' : '▼'}</span>
          決算書画像からOCR読み取り（AI）
        </button>
        {showOcr && (
          <div className="mt-3">
            <OcrUpload onApply={handleOcrApply} />
          </div>
        )}
      </div>
      
      {/* P/L セクション */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-blue-500">📁</span> 損益計算書 (P/L)
        </h3>
        <SliderInput label="売上高" name="nenshu" value={data.nenshu} min={0} max={10000} step={1} onChange={onChange} quickValues={[{ label: "1億", value: 100 }, { label: "3億", value: 300 }, { label: "5億", value: 500 }, { label: "10億", value: 1000 }]} hint="直近決算期の年商（百万円）。スコアの基準値になります" />
        <SliderInput label="売上総利益 (粗利)" name="gross_profit" value={data.gross_profit} min={-5000} max={5000} step={1} onChange={onChange} quickValues={[{ label: "10", value: 10 }, { label: "30", value: 30 }, { label: "50", value: 50 }, { label: "100", value: 100 }]} hint="売上高 - 売上原価。マイナスは原価割れを示します" />
        <SliderInput label="営業利益" name="op_profit" value={data.op_profit} min={-1000} max={1000} step={1} onChange={onChange} quickValues={[{ label: "-10", value: -10 }, { label: "0", value: 0 }, { label: "10", value: 10 }, { label: "30", value: 30 }]} hint="本業の儲け。販管費控除後の利益" />
        <SliderInput label="経常利益" name="ord_profit" value={data.ord_profit} min={-1000} max={1000} step={1} onChange={onChange} quickValues={[{ label: "-10", value: -10 }, { label: "0", value: 0 }, { label: "10", value: 10 }, { label: "30", value: 30 }]} hint="特別損益・税金を除いた実力値。スコアリングで重視されます" />
        <SliderInput label="当期純利益" name="net_income" value={data.net_income} min={-1000} max={1000} step={1} onChange={onChange} quickValues={[{ label: "-10", value: -10 }, { label: "0", value: 0 }, { label: "5", value: 5 }, { label: "20", value: 20 }]} hint="税引後の最終利益。マイナスは赤字（内部留保が減少）" />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <h4 className="text-sm font-bold text-slate-600 mb-3 block">減価償却費</h4>
            <SliderInput label="減価償却費（資産）" name="depreciation" value={data.depreciation} min={0} max={1000} step={1} onChange={onChange} hint="B/S計上額（固定資産の期中償却額）" />
            <SliderInput label="減価償却費（経費）" name="dep_expense" value={data.dep_expense} min={0} max={1000} step={1} onChange={onChange} hint="P/L費用計上額（通常は資産計上額と同額）" />
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <h4 className="text-sm font-bold text-slate-600 mb-3 block">地代家賃</h4>
            <SliderInput label="賃借料（資産）" name="rent" value={data.rent} min={0} max={500} step={1} onChange={onChange} hint="B/S計上の使用権資産（IFRS16対応）" />
            <SliderInput label="賃借料（経費）" name="rent_expense" value={data.rent_expense} min={0} max={500} step={1} onChange={onChange} hint="P/L費用計上の賃借料" />
          </div>
        </div>
      </div>

      {/* B/S セクション */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-purple-500">📁</span> 貸借対照表 (B/S)
        </h3>
        <SliderInput label="総資産" name="total_assets" value={data.total_assets} min={1} max={10000} step={1} onChange={onChange} quickValues={[{ label: "1億", value: 100 }, { label: "3億", value: 300 }, { label: "5億", value: 500 }, { label: "10億", value: 1000 }]} hint="B/S合計値（百万円）。資産規模の基準になります" />
        <SliderInput label="純資産 (自己資本)" name="net_assets" value={data.net_assets} min={-5000} max={5000} step={1} onChange={onChange} quickValues={[{ label: "-10", value: -10 }, { label: "0", value: 0 }, { label: "50", value: 50 }, { label: "100", value: 100 }]} hint="自己資本。マイナス値は債務超過を示し、スコアに大きく影響します" />

        <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mt-4">
          <h4 className="text-sm font-bold text-slate-600 mb-3 block">資産の内訳</h4>
          <SliderInput label="機械装置・運搬具" name="machines" value={data.machines} min={0} max={5000} step={1} onChange={onChange} hint="有形固定資産のうち機械・車両の帳簿価額" />
          <SliderInput label="その他固定資産" name="other_assets" value={data.other_assets} min={0} max={5000} step={1} onChange={onChange} hint="建物・土地・その他固定資産の合計" />
        </div>
      </div>

    </div>
  );
}
