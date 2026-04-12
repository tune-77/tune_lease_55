"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Settings, ShieldCheck, Save, Activity, Sliders, Info } from 'lucide-react';

export default function RulesPage() {
  const [rules, setRules] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    triggerMebuki('guide', '審査ルールの設定画面です！\\nここでの変更はシステム全体の判定ロジックに影響するので、慎重にお願いしますね。');
    fetchRules();
  }, []);

  const fetchRules = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:8000/api/settings/rules");
      setRules(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', 'ルールの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await axios.post("http://localhost:8000/api/settings/rules", rules);
      triggerMebuki('approve', '審査ルールを更新しました！\\nこれで次からの審査に新しい基準が適用されます。');
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '保存に失敗しました。');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <Activity className="w-12 h-12 text-indigo-500 animate-spin" />
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Settings className="w-8 h-8 text-indigo-500" />
          審査ロジック・ルール設定
        </h1>
        <p className="text-slate-500 font-bold mt-2">承認ラインや金利スプレッドなど、ビジネスルールのコアパラメーターを管理します。</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl">
         <div className="bg-white border border-slate-200 rounded-3xl shadow-xl overflow-hidden">
            <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
               <Sliders className="w-5 h-5 text-indigo-500" />
               <span className="font-black text-slate-700 uppercase tracking-widest text-sm">判定閾値設定</span>
            </div>
            <div className="p-8 space-y-6">
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">自動承認しきい値 (%)</label>
                  <input 
                     type="number" step="1"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-emerald-600 outline-none"
                     value={Math.round(rules.thresholds.approval * 100)}
                     onChange={(e) => setRules({...rules, thresholds: {...rules.thresholds, approval: parseInt(e.target.value)/100}})}
                  />
                  <p className="text-[10px] text-slate-400 mt-2">この数値以上のスコアで「承認」判定となります。</p>
               </div>
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">要審議しきい値 (%)</label>
                  <input 
                     type="number" step="1"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-amber-600 outline-none"
                     value={Math.round(rules.thresholds.consultation * 100)}
                     onChange={(e) => setRules({...rules, thresholds: {...rules.thresholds, consultation: parseInt(e.target.value)/100}})}
                  />
               </div>
            </div>
         </div>

         <div className="bg-white border border-slate-200 rounded-3xl shadow-xl overflow-hidden">
            <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center gap-2">
               <Info className="w-5 h-5 text-emerald-500" />
               <span className="font-black text-slate-700 uppercase tracking-widest text-sm">収益性・スプレッド</span>
            </div>
            <div className="p-8 space-y-6">
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">目標利益スプレッド (%)</label>
                  <input 
                     type="number" step="0.1"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-indigo-600 outline-none"
                     value={rules.pricing.target_spread}
                     onChange={(e) => setRules({...rules, pricing: {...rules.pricing, target_spread: parseFloat(e.target.value)}})}
                  />
               </div>
               <div>
                  <label className="block text-sm font-black text-slate-700 mb-2">デフォルト成約率想定 (%)</label>
                  <input 
                     type="number" step="1"
                     className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-bold text-lg text-slate-600 outline-none"
                     value={Math.round(rules.pricing.expected_win_rate * 100)}
                     onChange={(e) => setRules({...rules, pricing: {...rules.pricing, expected_win_rate: parseInt(e.target.value)/100}})}
                  />
               </div>
            </div>
         </div>
      </div>

      <div className="mt-8 flex items-center gap-4">
         <button 
               onClick={handleSave}
               disabled={saving}
               className="bg-indigo-600 hover:bg-indigo-500 text-white font-black py-4 px-12 rounded-2xl shadow-xl shadow-indigo-500/30 transition-all flex items-center justify-center gap-2"
            >
               {saving ? <Activity className="w-6 h-6 animate-spin" /> : <Save className="w-6 h-6" />}
               ルールを反映・保存
         </button>
         <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
            <ShieldCheck className="w-4 h-4" />
            最終更新: {new Date().toLocaleDateString()}
         </div>
      </div>
    </div>
  );
}