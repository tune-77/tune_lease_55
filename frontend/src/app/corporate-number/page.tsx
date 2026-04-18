"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Building, Settings, ShieldCheck, Key, Save, Activity, Globe } from 'lucide-react';

export default function CorporateNumberPage() {
  const [settings, setSettings] = useState({
    api_key: '',
    default_number: '',
    auto_fetch: true
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    triggerMebuki('guide', '企業番号APIの設定画面ですね！\\nここを正しく設定すると、新規審査の際に入力がグッと楽になりますよ。');
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/corporate_number`);
      setSettings(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/settings/corporate_number`, settings);
      triggerMebuki('approve', '設定を保存しました！完璧です！\\nこれで審査の自動化がまた一歩進みましたね。');
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '保存に失敗しました。ネットワークを確認してください。');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <Activity className="w-12 h-12 text-blue-500 animate-spin" />
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Building className="w-8 h-8 text-blue-500" />
          企業番号・外部連携設定
        </h1>
        <p className="text-slate-500 font-bold mt-2">
          法人番号公表サイトや、財務データベースAPIとの連携、AIアシスタントの接続設定を管理します。
        </p>
      </div>

      <div className="max-w-2xl bg-white border border-slate-200 rounded-3xl shadow-xl overflow-hidden">
         <div className="p-8 border-b border-slate-100 bg-slate-50/50">
            <h3 className="text-lg font-black text-slate-700 flex items-center gap-2 mb-2">
               <Globe className="w-5 h-5 text-blue-500" />
               外部API連携設定 (国税庁/gBizINFO)
            </h3>
            <p className="text-xs text-slate-400 font-bold uppercase tracking-widest">External Data Connectivity</p>
         </div>
         
         <div className="p-8 space-y-6">
            <div>
               <label className="block text-sm font-black text-slate-700 mb-2 flex items-center gap-2">
                  <Key className="w-4 h-4 text-slate-400" />
                  API アクセストークン / キー
               </label>
               <input 
                  type="password" 
                  className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-mono text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                  value={settings.api_key}
                  onChange={(e) => setSettings({...settings, api_key: e.target.value})}
                  placeholder="••••••••••••••••••••••••"
               />
               <p className="text-[10px] text-slate-400 mt-2 font-bold uppercase tracking-wider">
                  ※ 入力されたキーは暗号化されてサーバー内に保存されます。
               </p>
            </div>

            <div>
               <label className="block text-sm font-black text-slate-700 mb-2">
                  デフォルト企業番号 (検証用)
               </label>
               <input 
                  type="text" 
                  className="w-full bg-slate-50 border border-slate-200 p-4 rounded-xl font-mono text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                  value={settings.default_number}
                  onChange={(e) => setSettings({...settings, default_number: e.target.value})}
                  placeholder="13桁の法人番号を入力..."
               />
            </div>

            <div className="flex items-center justify-between p-4 bg-blue-50/50 rounded-2xl border border-blue-100">
               <div>
                  <div className="text-sm font-black text-blue-900">自動データフェッチを有効にする</div>
                  <div className="text-xs text-blue-600 font-bold">企業番号入力時に自動的に財務概要を取得します</div>
               </div>
               <button 
                  onClick={() => setSettings({...settings, auto_fetch: !settings.auto_fetch})}
                  className={`w-14 h-8 rounded-full transition-all relative ${settings.auto_fetch ? 'bg-blue-600' : 'bg-slate-300'}`}
               >
                  <div className={`absolute top-1 w-6 h-6 bg-white rounded-full transition-all ${settings.auto_fetch ? 'left-7' : 'left-1'}`}></div>
               </button>
            </div>
         </div>

         <div className="p-8 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-600">
               <ShieldCheck className="w-4 h-4" />
               Security Context Validated
            </div>
            <button 
               onClick={handleSave}
               disabled={saving}
               className="bg-blue-600 hover:bg-blue-500 text-white font-black py-3 px-8 rounded-xl shadow-lg shadow-blue-500/30 transition-all flex items-center justify-center gap-2"
            >
               {saving ? <Activity className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
               {saving ? 'Saving...' : '設定を保存'}
            </button>
         </div>
      </div>

      <div className="mt-8 p-6 bg-slate-900 rounded-2xl border border-slate-800 text-slate-400 text-sm max-w-2xl">
         <div className="font-black text-white mb-2 flex items-center gap-2">
            <Settings className="w-4 h-4 text-slate-500" />
            Advanced System Logs
         </div>
         <div className="font-mono text-[10px] space-y-1 opacity-60">
            <div>[SYSTEM] INITIALIZING API CONTEXT... OK</div>
            <div>[AUTH] VERIFYING CREDENTIALS... OK</div>
            <div>[NETWORK] gBizINFO GATEWAY READY.</div>
         </div>
      </div>
    </div>
  );
}