"use client";

import React, { useState } from "react";
import { Factory, Search } from "lucide-react";

type AssetItem = {
  name: string;
  years: number;
  avgAmount: string;
  note?: string;
};

type IndustryEntry = {
  industry: string;
  code: string;
  icon: string;
  description: string;
  assets: AssetItem[];
};

const INDUSTRY_ASSETS: IndustryEntry[] = [
  {
    industry: "建設業",
    code: "D",
    icon: "🏗️",
    description: "重機・足場・作業車両など大型設備が中心。耐用年数4〜8年が多い。",
    assets: [
      { name: "油圧ショベル（ユンボ）", years: 6, avgAmount: "300〜800万円", note: "中古流通多く残価安定" },
      { name: "ホイールローダー", years: 6, avgAmount: "400〜1,000万円" },
      { name: "クローラークレーン", years: 6, avgAmount: "500万〜2億円" },
      { name: "フォークリフト（工事用）", years: 4, avgAmount: "100〜300万円" },
      { name: "仮設足場資材", years: 3, avgAmount: "50〜200万円", note: "消耗品扱いのため期間短め" },
      { name: "コンプレッサー", years: 6, avgAmount: "30〜100万円" },
    ],
  },
  {
    industry: "製造業",
    code: "E",
    icon: "⚙️",
    description: "CNC機械・プレス機など専用設備が多い。耐用年数8〜12年。残価は汎用性次第。",
    assets: [
      { name: "CNC旋盤", years: 10, avgAmount: "500万〜2,000万円", note: "汎用型は残価安定" },
      { name: "マシニングセンタ", years: 10, avgAmount: "800万〜3,000万円" },
      { name: "プレス機", years: 10, avgAmount: "200万〜5,000万円", note: "特注型は残価低め" },
      { name: "産業用ロボット", years: 8, avgAmount: "500万〜2,000万円", note: "補助金対象になりやすい" },
      { name: "3Dプリンター", years: 5, avgAmount: "50〜500万円", note: "技術革新が早く残価注意" },
      { name: "フォークリフト（工場内）", years: 4, avgAmount: "100〜300万円" },
    ],
  },
  {
    industry: "情報通信業",
    code: "G",
    icon: "💻",
    description: "サーバー・ネットワーク機器等。陳腐化が早くリース期間3〜4年が標準。",
    assets: [
      { name: "ラックサーバー", years: 4, avgAmount: "50〜500万円", note: "4〜5年で世代交代" },
      { name: "ネットワークスイッチ", years: 5, avgAmount: "20〜200万円" },
      { name: "ストレージ（NAS/SAN）", years: 5, avgAmount: "100〜2,000万円" },
      { name: "業務用PC（法人一括）", years: 4, avgAmount: "10〜50万円/台", note: "IT導入補助金対象" },
      { name: "複合機・プリンター", years: 5, avgAmount: "30〜150万円" },
    ],
  },
  {
    industry: "運輸・物流業",
    code: "H",
    icon: "🚛",
    description: "トラック・冷凍車など。走行距離・整備状況で残価大きく変動。",
    assets: [
      { name: "大型トラック（10t）", years: 5, avgAmount: "1,000〜2,000万円", note: "走行距離で残価変動" },
      { name: "中型トラック（4t）", years: 5, avgAmount: "400〜800万円" },
      { name: "冷凍冷蔵車", years: 5, avgAmount: "500万〜1,200万円", note: "冷凍機の耐用年数も考慮" },
      { name: "フォークリフト（倉庫）", years: 4, avgAmount: "100〜300万円" },
      { name: "自動倉庫システム", years: 10, avgAmount: "3,000万〜数億円", note: "大型案件・複数社審査" },
    ],
  },
  {
    industry: "農業・林業",
    code: "K",
    icon: "🌾",
    description: "農機具は農業次世代人材投資資金等の補助金と組み合わせやすい。",
    assets: [
      { name: "トラクター", years: 7, avgAmount: "200〜800万円", note: "農業補助金対象多い" },
      { name: "コンバイン", years: 7, avgAmount: "500万〜1,500万円" },
      { name: "田植機", years: 7, avgAmount: "100〜400万円" },
      { name: "農業用ドローン", years: 5, avgAmount: "100〜500万円", note: "新型は技術陳腐化注意" },
      { name: "ビニールハウス設備", years: 10, avgAmount: "200万〜1,000万円" },
    ],
  },
  {
    industry: "医療・福祉",
    code: "O-P",
    icon: "🏥",
    description: "医療機器は高額・専門性が高い。地方版補助金との組み合わせも多い。",
    assets: [
      { name: "CT・MRI装置", years: 6, avgAmount: "3,000万〜2億円", note: "大型案件・複数社審査が多い" },
      { name: "X線装置", years: 6, avgAmount: "500万〜3,000万円" },
      { name: "内視鏡システム", years: 6, avgAmount: "200万〜1,000万円" },
      { name: "介護用電動ベッド（一括）", years: 6, avgAmount: "20〜50万円/台" },
      { name: "透析装置", years: 5, avgAmount: "100〜300万円/台" },
    ],
  },
  {
    industry: "飲食・小売業",
    code: "I-J",
    icon: "🍳",
    description: "厨房機器・POSシステムが中心。耐用年数は比較的短い。",
    assets: [
      { name: "業務用厨房機器一式", years: 8, avgAmount: "200〜1,000万円" },
      { name: "冷蔵・冷凍ショーケース", years: 6, avgAmount: "50〜300万円" },
      { name: "POSレジシステム", years: 5, avgAmount: "20〜100万円", note: "IT導入補助金対象" },
      { name: "食洗機（業務用）", years: 6, avgAmount: "30〜200万円" },
    ],
  },
];

export default function IndustryAssetsPage() {
  const [search, setSearch] = useState("");
  const [selectedIndustry, setSelectedIndustry] = useState<string | null>(null);

  const filtered = INDUSTRY_ASSETS.filter((entry) => {
    const q = search.toLowerCase();
    const matchSearch = !q ||
      entry.industry.includes(q) ||
      entry.assets.some((a) => a.name.toLowerCase().includes(q));
    const matchIndustry = !selectedIndustry || entry.industry === selectedIndustry;
    return matchSearch && matchIndustry;
  });

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <Factory className="text-slate-600 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">業種別リース物件例</h1>
          <p className="text-sm text-slate-500">業種ごとの代表的なリース物件・取得価格目安・法定耐用年数の一覧。</p>
        </div>
      </div>

      {/* 検索・フィルタ */}
      <div className="flex flex-wrap gap-3 items-center bg-slate-50 rounded-xl p-3">
        <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-3 py-1.5 flex-1 min-w-48">
          <Search size={14} className="text-slate-400" />
          <input
            type="text"
            placeholder="設備名・業種で検索"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setSelectedIndustry(null); }}
            className="text-sm outline-none w-full text-slate-700 placeholder-slate-400"
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {INDUSTRY_ASSETS.map((entry) => (
            <button
              key={entry.industry}
              onClick={() => { setSelectedIndustry(selectedIndustry === entry.industry ? null : entry.industry); setSearch(""); }}
              className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                selectedIndustry === entry.industry
                  ? "bg-slate-700 text-white"
                  : "bg-white border border-slate-300 text-slate-600 hover:bg-slate-50"
              }`}
            >
              {entry.icon} {entry.industry}
            </button>
          ))}
        </div>
      </div>

      {/* カード一覧 */}
      <div className="space-y-4">
        {filtered.map((entry) => {
          const assetList = search
            ? entry.assets.filter((a) => a.name.toLowerCase().includes(search.toLowerCase()))
            : entry.assets;
          if (assetList.length === 0) return null;
          return (
            <div key={entry.industry} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="flex items-center gap-2 px-4 py-3 bg-slate-50 border-b border-slate-200">
                <span className="text-xl">{entry.icon}</span>
                <div>
                  <h2 className="font-semibold text-slate-700">{entry.industry}</h2>
                  <p className="text-xs text-slate-400">{entry.description}</p>
                </div>
                <span className="ml-auto text-xs text-slate-400">{assetList.length} 品目</span>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-slate-500 border-b border-slate-100 bg-slate-50">
                    <th className="text-left px-4 py-2">設備名</th>
                    <th className="text-center px-3 py-2 w-20">耐用年数</th>
                    <th className="text-left px-3 py-2">取得価格目安</th>
                    <th className="text-left px-3 py-2">備考</th>
                  </tr>
                </thead>
                <tbody>
                  {assetList.map((asset, i) => (
                    <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      <td className="px-4 py-2 font-medium text-slate-800">{asset.name}</td>
                      <td className="px-3 py-2 text-center">
                        <span className="inline-block px-2 py-0.5 rounded-full text-xs font-bold bg-blue-50 text-blue-700 border border-blue-200">
                          {asset.years}年
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-600">{asset.avgAmount}</td>
                      <td className="px-3 py-2 text-xs text-slate-400">{asset.note ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div className="text-center py-12 text-slate-400">該当する物件がありません</div>
        )}
      </div>
    </div>
  );
}
