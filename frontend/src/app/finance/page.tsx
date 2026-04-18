"use client";
import React, { useEffect } from 'react';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Activity, Layout } from 'lucide-react';

export default function FinancePage() {
  useEffect(() => {
    triggerMebuki('guide', '物件ファイナンス審査の画面ですね！\n現在AIモジュールへ接続準備中です。');
  }, []);

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Layout className="w-8 h-8 text-blue-600" />
          物件ファイナンス審査
        </h1>
        <p className="text-slate-500 font-bold mt-2">この機能は現在バックエンドAPIとの統合準備中です。</p>
      </div>

      <div className="bg-white border border-slate-200 p-12 rounded-2xl flex flex-col items-center justify-center text-center shadow-sm">
        <div className="w-20 h-20 bg-slate-50 border border-slate-100 rounded-full flex items-center justify-center mb-6">
          <Activity className="w-10 h-10 text-slate-400" />
        </div>
        <h3 className="text-xl font-black text-slate-700 mb-2">実装準備中 (Phase 15 Mega Migration)</h3>
        <p className="text-slate-500 max-w-md">
          物件ファイナンス審査 のバックエンド・ロジック（FastAPI経由）とフロントエンドUIの接続を進めています。近日中に稼働開始します。
        </p>
        <div className="mt-8">
          <button className="px-6 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold rounded-lg transition-colors cursor-not-allowed">
            初期化処理を実行する
          </button>
        </div>
      </div>
    </div>
  );
}