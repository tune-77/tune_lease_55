import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, ReferenceLine,
  AreaChart, Area
} from 'recharts';

export default function RealGraphs() {
  
  // 1. レーダーチャート用データ (総合審査バランス)
  const radarData = [
    { subject: 'P/L (収益性)', A: 85, fullMark: 100 },
    { subject: 'B/S (安全性)', A: 65, fullMark: 100 },
    { subject: '定性 (将来性)', A: 90, fullMark: 100 },
    { subject: '物件 (汎用性)', A: 70, fullMark: 100 },
    { subject: '信用 (返済歴)', A: 80, fullMark: 100 },
    { subject: '保証 (担保)', A: 40, fullMark: 100 }
  ];

  // 2. SHAP的 要因分解バーデータ (限界スコア増減要因)
  const shapData = [
    { name: '営業利益(直近)', value: 15.2, fill: '#10b981' },
    { name: '設立・経営年数', value: 8.5, fill: '#10b981' },
    { name: '自己資本比率', value: 4.1, fill: '#10b981' },
    { name: '銀行与信依存度', value: -5.3, fill: '#f43f5e' },
    { name: '減価償却負担', value: -12.4, fill: '#f43f5e' },
  ].sort((a,b) => a.value - b.value); // バーの下から上へ

  // 3. 将来予測エリアデータ (TimesFM等)
  const futureData = [
    { year: '2023(実)', revenue: 4000, range: [4000, 4000] },
    { year: '2024(実)', revenue: 4200, range: [4200, 4200] },
    { year: '2025(予)', revenue: 4500, range: [4100, 4900] },
    { year: '2026(予)', revenue: 4800, range: [4200, 5500] },
    { year: '2027(予)', revenue: 5100, range: [4300, 6200] },
    { year: '2028(予)', revenue: 5300, range: [4200, 6800] },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mt-6 mb-8">
      
      {/* 1. 総合バランス レーダー */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-violet-500">🎯</span> 総合審査バランス (Radar)
        </h3>
        <div className="h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 'bold' }} />
              <Radar name="対象企業" dataKey="A" stroke="#8b5cf6" strokeWidth={2} fill="#8b5cf6" fillOpacity={0.3} />
              <Tooltip 
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }}
                itemStyle={{ color: '#8b5cf6', fontWeight: 'bold' }} 
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">※ 定性要因と収益性が牽引しています</p>
      </div>

      {/* 2. SHAPスコア貢献度 */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-emerald-500">📈</span> スコア変動の要因 (SHAP)
        </h3>
        <div className="h-[250px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis type="number" hide />
              <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 11, fill: '#475569', fontWeight: 600 }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{fill: '#f8fafc'}} 
                contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }} 
                formatter={(val: any) => [val > 0 ? `+${val}点` : `${val}点`, '影響度']} 
              />
              <ReferenceLine x={0} stroke="#cbd5e1" />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                {shapData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#f43f5e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">※ 右に伸びる要素が加点、左が減点要素</p>
      </div>

      {/* 3. 将来業績シミュレーション */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 lg:col-span-2 xl:col-span-1">
        <h3 className="text-sm font-black text-slate-800 mb-6 flex items-center gap-2">
          <span className="text-blue-500">🔮</span> 5年後売上予測シミュレーション
        </h3>
        <div className="h-[250px] w-full relative">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={futureData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorRev" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorRange" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#93c5fd" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#93c5fd" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
              <XAxis dataKey="year" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
              <Tooltip 
                contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }}
                formatter={(val: any, name: any) => {
                  if (name === '予測値') return [`${val} 千円`, name];
                  return [``, ''];
                }}
              />
              {/* 信頼区間 (Range) はAreaを使って描画する場合はデータ構造工夫が必要なため、今回は予測線のみを描画し影をつける */}
              <Area type="monotone" dataKey="revenue" name="予測値" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRev)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <p className="text-[11px] text-slate-500 mt-2 text-center">※ AI (TimesFM) がマクロ経済指標から導出した売上の期待値推移</p>
      </div>

    </div>
  );
}
