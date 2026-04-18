import React, { useEffect, useState } from 'react';
import { ScoringFormData } from '../../types';
import { API_BASE } from '../../lib/api';

interface FormGeneralProps {
  data: ScoringFormData;
  onChange: (name: string, value: string | number | string[]) => void;
}

interface IndustryMasterEntry {
  mapping?: string;
  sub?: { [sub: string]: string };
  [key: string]: unknown;
}
interface IndustryMaster {
  [major: string]: IndustryMasterEntry | string[];
}

function extractSubs(entry: IndustryMasterEntry | string[] | undefined): string[] {
  if (!entry) return [];
  if (Array.isArray(entry)) return entry.filter(Boolean);
  if (entry.sub && typeof entry.sub === 'object') return Object.keys(entry.sub);
  return Object.keys(entry).filter(k => k !== 'mapping');
}

export default function FormGeneral({ data, onChange }: FormGeneralProps) {
  const [industryMaster, setIndustryMaster] = useState<IndustryMaster>({});
  const [majors, setMajors] = useState<string[]>([]);
  const [subs, setSubs] = useState<string[]>([]);

  // マスターデータの取得
  useEffect(() => {
    const fetchIndustries = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/master/industries`);
        if (res.ok) {
          const jsicData = await res.json();
          setIndustryMaster(jsicData);
          setMajors(Object.keys(jsicData));
          
          // 初期値の整合性チェック
          if (data.industry_major && jsicData[data.industry_major]) {
            setSubs(extractSubs(jsicData[data.industry_major]));
          }
        }
      } catch (err) {
        console.error("Failed to fetch industries:", err);
      }
    };
    fetchIndustries();
  }, []);

  // 大分類変更時の連動
  useEffect(() => {
    if (data.industry_major && industryMaster[data.industry_major]) {
      const newSubs = extractSubs(industryMaster[data.industry_major]);
      setSubs(newSubs);
      // もし現在の中分類が新しいリストになければ、最初の項目を選択
      if (!newSubs.includes(data.industry_sub)) {
        onChange('industry_sub', newSubs[0] || "");
      }
    }
  }, [data.industry_major, industryMaster]);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value, type } = e.target;
    onChange(name, type === 'number' ? Number(value) : value);
  };

  return (
    <div className="space-y-6">
      
      {/* 企業番号・企業名 - 最上段に表示 */}
      <div className="bg-blue-50 border-2 border-blue-200 p-5 rounded-2xl shadow-sm">
        <h3 className="text-lg font-bold text-blue-800 border-b border-blue-200 pb-3 mb-4 flex items-center gap-2">
          <span className="text-blue-500">🔢</span> 案件特定情報 <span className="text-xs font-normal text-blue-500 ml-1">（審査DBへの保存に使用）</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-1">
            <label className="text-xs font-bold text-blue-600 mb-1 block">企業番号 (必須6桁)</label>
            <input
              type="text"
              name="company_no"
              maxLength={6}
              value={data.company_no || ""}
              onChange={handleChange}
              placeholder="例）123456"
              className="w-full bg-white border-2 border-blue-300 rounded-xl p-3 text-slate-800 font-bold text-lg outline-none focus:ring-2 focus:ring-blue-500 placeholder-slate-300"
            />
          </div>
          <div className="md:col-span-2">
            <label className="text-xs font-bold text-blue-600 mb-1 block">企業名 (任意)</label>
            <input
              type="text"
              name="company_name"
              value={data.company_name || ""}
              onChange={handleChange}
              placeholder="例）株式会社〇〇商事"
              className="w-full bg-white border-2 border-blue-300 rounded-xl p-3 text-slate-800 font-bold text-lg outline-none focus:ring-2 focus:ring-blue-500 placeholder-slate-300"
            />
          </div>
        </div>
        <p className="text-xs text-blue-500 mt-2 font-bold">※ 企業番号は、審査結果登録画面での案件特定に優先して使用されます</p>
      </div>

      {/* 基本情報・属性 */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-amber-500">🏢</span> 基本属性・取引状況
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">大分類 (業種) 📌</label>
            <select name="industry_major" value={data.industry_major} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              {majors.map(m => <option key={m} value={m}>{m}</option>)}
              {majors.length === 0 && <option>{data.industry_major}</option>}
            </select>
          </div>
          
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">中分類 📌</label>
            <select name="industry_sub" value={data.industry_sub} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              {subs.map(s => <option key={s} value={s}>{s}</option>)}
              {subs.length === 0 && <option>{data.industry_sub}</option>}
            </select>
          </div>
        </div>

        <div className="mt-4">
          <label className="text-sm font-bold text-slate-600 block mb-1">詳細キーワード (任意)</label>
          <input
            type="text"
            name="industry_detail"
            value={data.industry_detail || ""}
            onChange={handleChange}
            placeholder="例）精密機械加工、土木工事、ソフトウェア開発..."
            className="w-full bg-slate-50 border border-slate-300 rounded-xl p-3 text-slate-700 outline-none focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-[10px] text-slate-400 mt-1">※ AIがより正確な業界分析を行うためのヒントになります</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">取引区分</label>
            <select name="main_bank" value={data.main_bank} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>メイン先</option><option>非メイン先</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">顧客区分</label>
            <select name="customer_type" value={data.customer_type} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>既存先</option><option>新規先</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">商談ソース</label>
            <select name="deal_source" value={data.deal_source} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>銀行紹介</option><option>その他</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">競合状況</label>
            <select name="competitor" value={data.competitor} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>競合なし</option><option>競合あり</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">競合社数</label>
            <select name="num_competitors" value={data.num_competitors} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>未入力</option>
              <option>0社（指名）</option>
              <option>1社</option>
              <option>2社</option>
              <option>3社以上</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">発生経緯</label>
            <select name="deal_occurrence" value={data.deal_occurrence} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>不明</option>
              <option>指名</option>
              <option>相見積もり</option>
            </select>
          </div>
        </div>
      </div>

      {/* 信用情報・契約状況 */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-emerald-500">🛡️</span> 信用・契約情報
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">格付</label>
            <select name="grade" value={data.grade} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>①1-3 (優良)</option>
              <option>②4-6 (標準)</option>
              <option>③要注意以下</option>
              <option>④無格付</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">契約種類</label>
            <select name="contract_type" value={data.contract_type} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option>一般</option><option>自動車</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">既存の契約数 (件)</label>
            <input type="number" name="contracts" value={data.contracts} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">銀行与信残高 (千円)</label>
            <input type="number" name="bank_credit" value={data.bank_credit} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>
          
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">リース与信残高 (千円)</label>
            <input type="number" name="lease_credit" value={data.lease_credit} onChange={handleChange} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>
        </div>

      </div>

    </div>
  );
}
