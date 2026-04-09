import React from 'react';
import SliderInput from '../SliderInput';
import { ScoringFormData } from '../../types';

interface FormFinancialProps {
  data: ScoringFormData;
  onChange: (name: string, value: number) => void;
}

export default function FormFinancial({ data, onChange }: FormFinancialProps) {
  return (
    <div className="space-y-6">
      
      {/* P/L セクション */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-blue-500">📁</span> 損益計算書 (P/L)
        </h3>
        <SliderInput label="売上高" name="nenshu" value={data.nenshu} min={0} max={10000000} step={1000} onChange={onChange} />
        <SliderInput label="売上総利益 (粗利)" name="gross_profit" value={data.gross_profit} min={-5000000} max={5000000} step={1000} onChange={onChange} />
        <SliderInput label="営業利益" name="op_profit" value={data.op_profit} min={-1000000} max={1000000} step={1000} onChange={onChange} />
        <SliderInput label="経常利益" name="ord_profit" value={data.ord_profit} min={-1000000} max={1000000} step={1000} onChange={onChange} />
        <SliderInput label="当期純利益" name="net_income" value={data.net_income} min={-1000000} max={1000000} step={1000} onChange={onChange} />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <h4 className="text-sm font-bold text-slate-600 mb-3 block">減価償却費</h4>
            <SliderInput label="減価償却費（資産）" name="depreciation" value={data.depreciation} min={0} max={1000000} step={1000} onChange={onChange} />
            <SliderInput label="減価償却費（経費）" name="dep_expense" value={data.dep_expense} min={0} max={1000000} step={1000} onChange={onChange} />
          </div>
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <h4 className="text-sm font-bold text-slate-600 mb-3 block">地代家賃</h4>
            <SliderInput label="賃借料（資産）" name="rent" value={data.rent} min={0} max={500000} step={1000} onChange={onChange} />
            <SliderInput label="賃借料（経費）" name="rent_expense" value={data.rent_expense} min={0} max={500000} step={1000} onChange={onChange} />
          </div>
        </div>
      </div>

      {/* B/S セクション */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-purple-500">📁</span> 貸借対照表 (B/S)
        </h3>
        <SliderInput label="総資産" name="total_assets" value={data.total_assets} min={1} max={10000000} step={1000} onChange={onChange} />
        <SliderInput label="純資産 (自己資本)" name="net_assets" value={data.net_assets} min={-5000000} max={5000000} step={1000} onChange={onChange} />
        
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mt-4">
          <h4 className="text-sm font-bold text-slate-600 mb-3 block">資産の内訳</h4>
          <SliderInput label="機械装置・運搬具" name="machines" value={data.machines} min={0} max={5000000} step={1000} onChange={onChange} />
          <SliderInput label="その他固定資産" name="other_assets" value={data.other_assets} min={0} max={5000000} step={1000} onChange={onChange} />
        </div>
      </div>

    </div>
  );
}
