import os

routes = {
    'batch': 'バッチ審査',
    'finance': '物件ファイナンス審査',
    'agent': '汎用エージェントハブ',
    'register': '結果登録 (成約/失注)',
    'financial': '3期財務分析',
    'timesfm': 'TimesFM 時系列予測',
    'competitor': '競合関係グラフ',
    'similar': '案件類似ネットワーク',
    'visual': 'ビジュアルインサイト',
    'civilization': '文明年代記',
    'qualitative': '定性要因分析',
    'quantitative': '定量要因分析',
    'coef-analysis': '係数分析・更新 (β)',
    'coef-input': '係数入力（事前係数）',
    'coef-history': '係数変更履歴',
    'logs': 'アプリログ',
    'rules': '審査ルール設定',
    'interest': '基準金利マスタ',
    'corporate-number': '企業番号設定'
}

base_path = '/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/frontend/src/app'

template = """
"use client";
import React, { useEffect } from 'react';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Activity, Layout } from 'lucide-react';

export default function {component_name}Page() {{
  useEffect(() => {{
    triggerMebuki('guide', '{title}の画面ですね！\\n現在AIモジュールへ接続準備中です。');
  }}, []);

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Layout className="w-8 h-8 text-blue-600" />
          {title}
        </h1>
        <p className="text-slate-500 font-bold mt-2">この機能は現在バックエンドAPIとの統合準備中です。</p>
      </div>

      <div className="bg-white border border-slate-200 p-12 rounded-2xl flex flex-col items-center justify-center text-center shadow-sm">
        <div className="w-20 h-20 bg-slate-50 border border-slate-100 rounded-full flex items-center justify-center mb-6">
          <Activity className="w-10 h-10 text-slate-400" />
        </div>
        <h3 className="text-xl font-black text-slate-700 mb-2">実装準備中 (Phase 15 Mega Migration)</h3>
        <p className="text-slate-500 max-w-md">
          {title} のバックエンド・ロジック（FastAPI経由）とフロントエンドUIの接続を進めています。近日中に稼働開始します。
        </p>
        <div className="mt-8">
          <button className="px-6 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 font-bold rounded-lg transition-colors cursor-not-allowed">
            初期化処理を実行する
          </button>
        </div>
      </div>
    </div>
  );
}}
"""

for route, title in routes.items():
    dir_path = os.path.join(base_path, route)
    os.makedirs(dir_path, exist_ok=True)
    
    # Generate component name from route (e.g. batch -> Batch, coef-analysis -> CoefAnalysis)
    comp_name = "".join([part.capitalize() for part in route.split('-')])
    
    file_path = os.path.join(dir_path, "page.tsx")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(template.replace("{component_name}", comp_name).replace("{title}", title).strip())

print("All scaffolding pages generated successfully.")
