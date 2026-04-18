"use client";

import { useState } from "react";
import axios from "axios";
import { API_BASE } from "../lib/api";
import { Activity, ArrowRight, Building, Calculator, CheckCircle, PieChart, ShieldAlert, AlignLeft, Send } from "lucide-react";
import ScoreDAG from "../components/ScoreDAG";
import { ScoringFormData, defaultFormData } from "../types";
import FormGeneral from "../components/form/FormGeneral";
import FormFinancial from "../components/form/FormFinancial";
import FormQualitative from "../components/form/FormQualitative";

import IndicatorCards from "../components/analysis/IndicatorCards";
import RealGraphs from "../components/analysis/RealGraphs";
import AIAnalysis from "../components/analysis/AIAnalysis";
import AdvancedAnalysis from "../components/analysis/AdvancedAnalysis";
import GunshiAdvice from "../components/analysis/GunshiAdvice";
import ReportGenerator from "../components/analysis/ReportGenerator";
import { triggerMebuki } from "../components/layout/FloatingMebuki";

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [formData, setFormData] = useState<ScoringFormData>(defaultFormData);
  const [gunshiText, setGunshiText] = useState<string>("");
  
  // タブ管理
  const [activeTab, setActiveTab] = useState<"input" | "analysis">("input");

  // フィールドの変更ハンドラー
  const handleFieldChange = (name: string, value: string | number | string[]) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/api/score/full`, formData);
      setResult(res.data);
      // 審査完了後、自動的に分析タブへ遷移
      setActiveTab("analysis");

      // めぶきちゃんの表情をスコアに応じて切り替え
      const score = res.data.score_base;
      if (score >= 80) {
        triggerMebuki('approve', `スコア ${score.toFixed(1)} 点！\n素晴らしい内容です。\nこのまま稟議に掛けましょう！`);
      } else if (score >= 50) {
        triggerMebuki('challenge', `スコア ${score.toFixed(1)} 点。\n少し工夫が必要です。\n軍師のアドバイスを確認してください。`);
      } else {
        triggerMebuki('reject', `スコア ${score.toFixed(1)} 点。\nかなり厳しい状況です。\n抜本的な条件見直しが必要です！`);
      }

    } catch (error) {
      console.error("API Error", error);
      alert("審査エンジンの呼び出しに失敗しました。FastAPIサーバーが起動しているか確認してください。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-2rem)]">
      {/* タイトル領域 */}
      <div className="bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-sm p-4 sticky top-0 z-40 mb-6 flex justify-between items-center">
        <h2 className="text-xl font-black text-slate-800 flex items-center gap-2">
          <Calculator className="text-blue-500 w-6 h-6" />
          審査・分析ダッシュボード
        </h2>
        <button 
          onClick={() => setFormData(defaultFormData)}
          className="px-4 py-2 text-sm font-bold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors border border-slate-200 shadow-sm"
        >
          リセット
        </button>
      </div>

      <div className="px-4 md:px-6 lg:px-8 max-w-[1600px] mx-auto pb-20">
        <div className="flex flex-col xl:flex-row gap-6">
          
          {/* 左カラム: メイン操作エリア (入力・分析) */}
          <div className="w-full xl:w-2/3 flex flex-col">
            
            {/* タブナビゲーション */}
            <div className="flex bg-slate-200/50 p-1 rounded-xl mb-6 shadow-inner w-full sm:w-fit font-bold relative z-10">
              <button 
                onClick={() => setActiveTab("input")}
                className={`flex-1 sm:px-12 py-3 rounded-lg text-sm sm:text-base flex items-center justify-center gap-2 transition-all ${
                  activeTab === "input" ? "bg-white text-blue-600 shadow-md transform scale-100" : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                }`}
              >
                <AlignLeft className="w-5 h-5" />
                審査入力
              </button>
              <button 
                onClick={() => setActiveTab("analysis")}
                className={`flex-1 sm:px-12 py-3 rounded-lg text-sm sm:text-base flex items-center justify-center gap-2 transition-all ${
                  activeTab === "analysis" ? "bg-white text-indigo-600 shadow-md transform scale-100" : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                }`}
              >
                <PieChart className="w-5 h-5" />
                分析結果・レポート
              </button>
            </div>

            {/* コンテンツエリア */}
            {activeTab === "input" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 relative z-0">
                <FormGeneral data={formData} onChange={handleFieldChange} />
                <FormFinancial data={formData} onChange={handleFieldChange} />
                <FormQualitative data={formData} onChange={handleFieldChange} />

                <div className="sticky bottom-6 z-40 bg-white/90 backdrop-blur-md p-4 rounded-2xl shadow-xl border border-blue-100 flex items-center justify-between">
                  <div className="text-sm font-bold text-slate-500">
                    現在の入力状態で審査用API（フル機能版）を呼び出します
                  </div>
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={loading}
                    className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3.5 rounded-xl font-bold shadow-lg shadow-blue-200 hover:shadow-xl hover:translate-y-[-2px] transition-all disabled:opacity-50 text-lg"
                  >
                    {loading ? (
                      <>
                        <Activity className="w-5 h-5 animate-spin" />
                        審査中...
                      </>
                    ) : (
                      <>
                        <Calculator className="w-5 h-5" />
                        審査エンジンを実行
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {activeTab === "analysis" && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                {!result ? (
                  <div className="bg-white p-12 rounded-3xl shadow-sm border border-slate-200 text-center flex flex-col items-center justify-center min-h-[400px]">
                    <div className="w-20 h-20 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mb-6 shadow-inner border border-blue-100">
                      <PieChart className="w-10 h-10" />
                    </div>
                    <h3 className="text-2xl font-black text-slate-700 mb-3">まだ審査が実行されていません</h3>
                    <p className="text-slate-500 mb-8 font-medium">「審査入力」タブで各種数値を入力し、エンジンを実行してください。</p>
                    <button 
                      onClick={() => setActiveTab("input")}
                      className="px-8 py-3 bg-slate-800 text-white rounded-xl font-bold shadow-lg hover:bg-slate-700 transition"
                    >
                      入力画面へ戻る
                    </button>
                  </div>
                ) : (
                  <>
                    {/* DAGグラフ */}
                    <ScoreDAG data={result} />
                    
                    {/* 主要指標サマリ (カッコいいカード) */}
                    <IndicatorCards data={result} />

                    {/* 📊 新設: Recharts による本物のインタラクティブグラフ群 */}
                    <RealGraphs />

                    {/* AI分析テキスト (チャット風) */}
                    <AIAnalysis comparisonText={result.comparison} />
                    
                    {/* 今回追加した高度シミュレーションUI */}
                    <AdvancedAnalysis />

                    {/* 最終審査レポート */}
                    <ReportGenerator apiResult={result} formData={formData} gunshiText={gunshiText} />
                  </>
                )}
              </div>
            )}
          </div>

          {/* 右カラム: 審査軍師 (逆転プラン自動提案) */}
          <div className="w-full xl:w-1/3 mt-8 xl:mt-0 relative z-10">
            <GunshiAdvice 
              score={result?.score_base || 0} 
              pd_percent={result?.score_borrower || 0} 
              industry_major={result?.industry_major || formData.industry_major || ""}
              formData={formData}
              onChatLoaded={setGunshiText}
            />
          </div>
          
        </div>
      </div>
    </div>
  );
}
