import React, { useEffect, useState } from 'react';
import { ScoringFormData } from '../../types';
import SliderInput from '../SliderInput';
import { API_BASE } from '../../lib/api';

interface FormQualitativeProps {
  data: ScoringFormData;
  onChange: (name: string, value: string | number | string[]) => void;
}

interface QualitativeItem {
  id: string;
  label: string;
  weight: number;
  options: [number, string][];
}

export default function FormQualitative({ data, onChange }: FormQualitativeProps) {
  const [qualItems, setQualItems] = useState<QualitativeItem[]>([]);
  const [assetItems, setAssetItems] = useState<string[]>([]);

  // マスターデータの取得
  useEffect(() => {
    const fetchMaster = async () => {
      try {
        const [qualRes, assetRes] = await Promise.all([
          fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/master/qualitative`),
          fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/master/assets`)
        ]);
        
        if (qualRes.ok) {
          const qualData = await qualRes.json();
          setQualItems(qualData.items || []);
        }
        
        if (assetRes.ok) {
          const assetData = await assetRes.json();
          // lease_assets.json の構造に合わせて抽出
          const items = assetData.items?.map((it: any) => it.name) || [];
          setAssetItems(items);
        }
      } catch (err) {
        console.error("Failed to fetch qual/asset master:", err);
      }
    };
    fetchMaster();
  }, []);
  
  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement | HTMLTextAreaElement>) => {
    onChange(e.target.name, e.target.value);
  };
  
  const handleNumber = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.name, parseInt(e.target.value, 10) || 0);
  };
  
  const handleSlider = (name: string, value: number) => {
    onChange(name, value);
  }

  return (
    <div className="space-y-6">
      
      {/* 物件情報 */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-indigo-500">🏢</span> リース物件情報・契約条件
        </h3>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
          <div className="space-y-1 lg:col-span-2">
            <label className="text-sm font-bold text-slate-600 block">対象物件</label>
            <select name="asset_name" value={data.asset_name} onChange={handleSelect as any} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option value="">（選択してください）</option>
              {assetItems.map(name => <option key={name} value={name}>{name}</option>)}
              {assetItems.length === 0 && (
                <>
                  <option>建設機械</option>
                  <option>IT・OA機器</option>
                  <option>医療機器</option>
                  <option>車両・運搬車</option>
                  <option>製造設備・工作機械</option>
                </>
              )}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">物件取得価額 (千円)</label>
            <input type="number" name="acquisition_cost" value={data.acquisition_cost} onChange={handleNumber} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">リース期間 (月)</label>
            <input type="number" name="lease_term" value={data.lease_term} onChange={handleNumber} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">検収年 (西暦)</label>
            <input type="number" name="acceptance_year" value={data.acceptance_year} onChange={handleNumber} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
          </div>
        </div>
      </div>

      {/* 定性評価（6大項目） */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-rose-500">📝</span> 審査定性評価スコアリング
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          {qualItems.map(item => (
            <div key={item.id} className="space-y-1">
              <label className="text-[13px] font-bold text-slate-600 block">
                {item.label} <span className="text-[10px] font-normal text-slate-400">(重み{item.weight}%)</span>
              </label>
              <select 
                name={`qual_corr_${item.id}`} 
                value={data[`qual_corr_${item.id}` as keyof ScoringFormData] as string || "未選択"} 
                onChange={handleSelect} 
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-sm outline-none focus:border-blue-500 focus:bg-white transition h-[42px]"
              >
                <option value="未選択">未選択</option>
                {item.options.map(([score, label]) => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </select>
            </div>
          ))}
          
          {qualItems.length === 0 && <p className="text-sm text-slate-400">定性評価項目を読込中...</p>}
        </div>

        <div className="mt-8 border-t border-slate-100 pt-6">
          <label className="text-sm font-bold text-slate-600 block mb-2">🎈 直感スコア (1:懸念あり 〜 5:確信あり)</label>
          <SliderInput label="" name="intuition" value={data.intuition} min={1} max={5} step={1} onChange={handleSlider} />
        </div>
      </div>

    </div>
  );
}
