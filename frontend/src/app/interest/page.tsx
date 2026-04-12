"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Calendar, Percent, Plus, Table, Activity, History } from 'lucide-react';

export default function InterestPage() {
  const [rates, setRates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [newRate, setNewRate] = useState({ month: new Date().toISOString().slice(0, 7), rate: 2.1, note: '' });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    triggerMebuki('guide', '基準金利のマスタ管理ですね。\\n定期的に更新することで、最新の市場環境に合わせた審査が可能になります。');
    fetchRates();
  }, []);

  const fetchRates = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:8000/api/settings/interest");
      setRates(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '金利データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async () => {
    setSubmitting(true);
    try {
      await axios.post("http://localhost:8000/api/settings/interest", newRate);
      triggerMebuki('approve', `${newRate.month} 分の基準金利を更新しました！`);
      fetchRates();
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '更新に失敗しました。');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <Activity className="w-12 h-12 text-emerald-500 animate-spin" />
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
             <Calendar className="w-8 h-8 text-emerald-500" />
             基準金利マスタ管理
          </h1>
          <p className="text-slate-500 font-bold mt-2">月次で変動する基準金利（原価相当）を一括管理します。</p>
        </div>
        <div className="bg-emerald-50 border border-emerald-100 px-6 py-3 rounded-2xl shadow-sm text-center">
            <div className="text-[10px] font-black text-emerald-600 uppercase tracking-widest mb-1">Current Month Rate</div>
            <div className="text-2xl font-black text-emerald-700">{rates[0]?.rate?.toFixed(2)}%</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1">
          <div className="bg-white border border-slate-200 rounded-3xl shadow-xl overflow-hidden sticky top-8">
            <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
               <Plus className="w-5 h-5 text-emerald-500" />
               <span className="font-black text-slate-700 uppercase tracking-widest text-sm">金利を登録/更新</span>
            </div>
            <div className="p-8 space-y-6">
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">対象月 (YYYY-MM)</label>
                  <input 
                     type="month"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-slate-700 outline-none"
                     value={newRate.month}
                     onChange={(e) => setNewRate({...newRate, month: e.target.value})}
                  />
               </div>
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">基準金利 (%)</label>
                  <input 
                     type="number" step="0.01"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-emerald-600 outline-none"
                     value={newRate.rate}
                     onChange={(e) => setNewRate({...newRate, rate: parseFloat(e.target.value)})}
                  />
               </div>
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">メモ (任意)</label>
                  <input 
                     type="text"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl text-sm text-slate-600 outline-none"
                     value={newRate.note}
                     onChange={(e) => setNewRate({...newRate, note: e.target.value})}
                     placeholder="例: 市場金利上昇に伴う調整"
                  />
               </div>
               <button 
                   onClick={handleUpdate}
                   disabled={submitting}
                   className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-black py-4 rounded-2xl shadow-xl shadow-emerald-500/30 transition-all flex items-center justify-center gap-2"
                >
                   {submitting ? <Activity className="w-6 h-6 animate-spin" /> : <Percent className="w-6 h-6" />}
                   基準金利を登録
                </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
           <div className="bg-white border border-slate-200 rounded-3xl shadow-sm overflow-hidden">
              <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                 <div className="flex items-center gap-2">
                    <Table className="w-5 h-5 text-slate-400" />
                    <span className="font-black text-slate-700 uppercase tracking-widest text-sm">登録履歴一覧</span>
                 </div>
                 <History className="w-4 h-4 text-slate-300" />
              </div>
              <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
              <table className="w-full text-left border-collapse">
                 <thead>
                    <tr className="bg-slate-50/50 border-b border-slate-100">
                       <th className="p-4 text-xs font-black text-slate-400 uppercase tracking-widest pl-8">対象月</th>
                       <th className="p-4 text-xs font-black text-slate-400 uppercase tracking-widest">基準金利</th>
                       <th className="p-4 text-xs font-black text-slate-400 uppercase tracking-widest">備考・メモ</th>
                    </tr>
                 </thead>
                 <tbody>
                    {rates.map((rate, i) => (
                       <tr key={i} className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors">
                          <td className="p-4 text-sm font-black text-slate-700 pl-8">{rate.month}</td>
                          <td className="p-4 text-sm font-black text-emerald-600 font-mono">{rate.rate.toFixed(2)}%</td>
                          <td className="p-4 text-sm font-bold text-slate-400 italic">{rate.note || '-'}</td>
                       </tr>
                    ))}
                 </tbody>
              </table>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}