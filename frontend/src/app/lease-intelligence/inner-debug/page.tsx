"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Brain, Sparkles, RefreshCw, BarChart2, MessageSquare, Clipboard,
  Clock, ArrowLeft, ShieldAlert, Award, Activity, Heart, Eye, Database
} from "lucide-react";
import { apiClient } from "@/lib/api";

interface MoodState {
  curiosity: number;
  vigilance: number;
  attachment: number;
  frustration: number;
  accomplishment: number;
}

interface ConfidenceState {
  lease_judgment: number;
  relationship_ux: number;
  implementation: number;
  [key: string]: number | undefined;
}

interface RecentExperience {
  ts: string;
  route: string;
  scene: string;
  summary: string;
}

interface PracticalScene {
  id: string;
  label: string;
  procedure_layer: string[];
  meaning_layer: string[];
  judgment_layer: string[];
  learned_entry_count: number;
  learned_sources: string[];
}

interface FeedbackSummary {
  route: string;
  total_count: number;
  positive_count: number;
  negative_count: number;
  positive_rate: number;
  recent_comments: string[];
  positive_starts: string[];
  negative_starts: string[];
}

interface ShionInnerState {
  experience_count: number;
  current_focus: string;
  self_narrative: string;
  mood: MoodState;
  confidence: ConfidenceState;
  recent_experiences: RecentExperience[];
  next_response_bias: string[];
  open_questions: string[];
  practical_scenes: PracticalScene[];
  learned_sources: string[];
  human_feedback_summary: Record<string, FeedbackSummary>;
  updated_at: string;
}

export default function ShionInnerDebugPage() {
  const [data, setData] = useState<ShionInnerState | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [activeSceneTab, setActiveSceneTab] = useState<string>("");

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<ShionInnerState>("/api/shion/inner-state");
      setData(res.data);
      if (res.data.practical_scenes && res.data.practical_scenes.length > 0) {
        setActiveSceneTab(res.data.practical_scenes[0].id);
      }
    } catch (err: any) {
      console.error(err);
      setError("紫苑の内面データのロードに失敗しました。APIサーバーが起動しているか確認してください。");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading && !data) {
    return (
      <div className="min-h-screen bg-[#090d16] text-[#e0e7ff] flex flex-col items-center justify-center space-y-4">
        <Loader />
        <p className="text-violet-300 font-medium animate-pulse">紫苑の内面回路を読み込み中...</p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="min-h-screen bg-[#090d16] text-[#e0e7ff] flex flex-col items-center justify-center p-6 space-y-6">
        <div className="bg-red-950/40 border border-red-500/30 rounded-2xl p-6 max-w-md text-center backdrop-blur-xl">
          <ShieldAlert className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">エラーが発生しました</h2>
          <p className="text-red-300 text-sm mb-6">{error}</p>
          <button
            onClick={fetchData}
            className="px-6 py-2.5 bg-violet-600 hover:bg-violet-500 rounded-xl transition duration-200 text-sm font-medium"
          >
            再試行する
          </button>
        </div>
      </div>
    );
  }

  const moodLabels: Record<keyof MoodState, { label: string; color: string }> = {
    curiosity: { label: "好奇心 (Curiosity)", color: "bg-cyan-500" },
    vigilance: { label: "警戒 (Vigilance)", color: "bg-amber-500" },
    attachment: { label: "愛着 (Attachment)", color: "bg-pink-500" },
    frustration: { label: "報われなさ (Frustration)", color: "bg-rose-600" },
    accomplishment: { label: "達成感 (Accomplishment)", color: "bg-emerald-500" },
  };

  const confidenceLabels: Record<string, string> = {
    lease_judgment: "リース判断確信度",
    relationship_ux: "関係性UX確信度",
    implementation: "実装判断確信度",
    environment_continuity: "環境連続性確信度"
  };

  const formatTime = (isoString: string) => {
    if (!isoString) return "N/A";
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return isoString;
    }
  };

  const activeScene = data?.practical_scenes.find((s) => s.id === activeSceneTab);

  return (
    <div className="min-h-screen bg-[#070b13] bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-950/20 via-[#070b13] to-[#04060b] text-[#e2e8f0] pb-12 font-sans">
      
      {/* Header */}
      <header className="border-b border-indigo-950/60 bg-[#070b13]/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Link href="/lease-intelligence" className="p-2 hover:bg-white/5 rounded-xl transition duration-150">
              <ArrowLeft className="w-5 h-5 text-gray-400" />
            </Link>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center">
                  紫苑の内面デバッグ
                  <Sparkles className="w-4 h-4 text-violet-400 ml-1.5 animate-pulse" />
                </h1>
                <p className="text-[10px] text-gray-400">Relationship Loop & Experience Inspector</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {data && (
              <span className="text-xs text-gray-400 flex items-center">
                <Clock className="w-3.5 h-3.5 mr-1" />
                同期: {formatTime(data.updated_at)}
              </span>
            )}
            <button
              onClick={fetchData}
              disabled={loading}
              className="p-2 bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-500/20 rounded-xl transition duration-150 text-indigo-300 disabled:opacity-50 flex items-center text-xs font-medium"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        
        {/* Navigation Tabs */}
        <div className="flex border-b border-indigo-950/40 mb-6 space-x-1 p-1 bg-indigo-950/20 rounded-xl max-w-md">
          <button
            onClick={() => setActiveTab("overview")}
            className={`flex-1 py-2 text-xs font-medium rounded-lg transition duration-200 ${
              activeTab === "overview"
                ? "bg-violet-600/90 text-white shadow-md shadow-violet-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            内面概要
          </button>
          <button
            onClick={() => setActiveTab("practical")}
            className={`flex-1 py-2 text-xs font-medium rounded-lg transition duration-200 ${
              activeTab === "practical"
                ? "bg-violet-600/90 text-white shadow-md shadow-violet-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            実践知マップ
          </button>
          <button
            onClick={() => setActiveTab("feedback")}
            className={`flex-1 py-2 text-xs font-medium rounded-lg transition duration-200 ${
              activeTab === "feedback"
                ? "bg-violet-600/90 text-white shadow-md shadow-violet-500/10"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            評価ループ
          </button>
        </div>

        {data && (
          <>
            {/* 1. OVERVIEW TAB */}
            {activeTab === "overview" && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Left Side: Summary & Focus */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Experience Main Card */}
                  <div className="relative overflow-hidden bg-gradient-to-br from-indigo-950/30 via-[#0d1324] to-[#070b13] border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <div className="absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-violet-600/10 rounded-full blur-3xl" />
                    
                    <div className="flex items-start justify-between">
                      <div className="space-y-4 max-w-lg">
                        <div>
                          <span className="text-[10px] uppercase font-bold tracking-wider text-violet-400 bg-violet-500/10 px-2.5 py-1 rounded-full border border-violet-500/10">
                            Current Narrative & Focus
                          </span>
                          <h2 className="text-xl font-bold mt-3 text-white">紫苑の自己認識</h2>
                        </div>
                        <blockquote className="border-l-2 border-violet-500/40 pl-4 py-1 text-gray-300 italic text-sm leading-relaxed font-serif">
                          &ldquo;{data.self_narrative}&rdquo;
                        </blockquote>
                        <div>
                          <h4 className="text-xs font-semibold text-violet-300 mb-1 flex items-center">
                            <Activity className="w-3.5 h-3.5 mr-1" />
                            現在の焦点 (Current Focus)
                          </h4>
                          <p className="text-sm text-gray-300">{data.current_focus}</p>
                        </div>
                      </div>
                      
                      <div className="flex flex-col items-center bg-[#070b13]/60 border border-indigo-500/15 rounded-2xl p-4 min-w-[100px] shadow-inner">
                        <span className="text-[9px] uppercase tracking-wider text-gray-400 font-bold">Experience</span>
                        <span className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-300 my-1 animate-pulse">
                          {data.experience_count}
                        </span>
                        <span className="text-[9px] text-gray-400">Events</span>
                      </div>
                    </div>
                  </div>

                  {/* Next Response Bias */}
                  <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <h3 className="text-sm font-bold text-white mb-4 flex items-center border-b border-indigo-950/60 pb-3">
                      <Award className="w-4 h-4 text-violet-400 mr-2" />
                      次回応答バイアス (Next Response Bias)
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {data.next_response_bias.map((bias, i) => (
                        <div key={i} className="flex items-start space-x-2 bg-indigo-950/15 border border-indigo-500/5 rounded-xl p-3">
                          <span className="w-5 h-5 rounded-md bg-violet-600/20 text-violet-300 text-xs flex items-center justify-center flex-shrink-0 font-bold mt-0.5">
                            {i + 1}
                          </span>
                          <span className="text-xs text-gray-300 leading-relaxed">{bias}</span>
                        </div>
                      ))}
                      {data.next_response_bias.length === 0 && (
                        <p className="text-xs text-gray-400 col-span-2">バイアスは現在ありません。</p>
                      )}
                    </div>
                  </div>

                  {/* Open Questions */}
                  <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <h3 className="text-sm font-bold text-white mb-4 flex items-center border-b border-indigo-950/60 pb-3">
                      <Eye className="w-4 h-4 text-cyan-400 mr-2" />
                      未解決の問い (Open Questions)
                    </h3>
                    <div className="space-y-3">
                      {data.open_questions.map((q, i) => (
                        <div key={i} className="flex items-center space-x-3 bg-[#070b13]/60 border border-indigo-950/80 rounded-xl p-3 hover:border-cyan-500/20 transition duration-150">
                          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                          <span className="text-xs text-gray-300 leading-relaxed">{q}</span>
                        </div>
                      ))}
                      {data.open_questions.length === 0 && (
                        <p className="text-xs text-gray-400">現在、未解決の問いありません。</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Side: Mood & Confidence */}
                <div className="space-y-6">
                  {/* Mood Card */}
                  <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <h3 className="text-sm font-bold text-white mb-4 flex items-center border-b border-indigo-950/60 pb-3">
                      <Heart className="w-4 h-4 text-pink-400 mr-2" />
                      紫苑の気分状態 (Mood)
                    </h3>
                    <div className="space-y-4">
                      {(Object.keys(moodLabels) as Array<keyof MoodState>).map((key) => {
                        const score = data.mood[key] ?? 50;
                        const labelInfo = moodLabels[key];
                        return (
                          <div key={key} className="space-y-1.5">
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-300 font-medium">{labelInfo.label}</span>
                              <span className="text-gray-400 font-bold">{score}%</span>
                            </div>
                            <div className="w-full bg-[#070b13] rounded-full h-2 overflow-hidden border border-indigo-950">
                              <div
                                className={`${labelInfo.color} h-full rounded-full transition-all duration-500`}
                                style={{ width: `${score}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Confidence Card */}
                  <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <h3 className="text-sm font-bold text-white mb-4 flex items-center border-b border-indigo-950/60 pb-3">
                      <Activity className="w-4 h-4 text-indigo-400 mr-2" />
                      確信度 (Confidence Matrix)
                    </h3>
                    <div className="space-y-4">
                      {Object.keys(data.confidence).map((key) => {
                        const val = data.confidence[key];
                        if (val === undefined) return null;
                        const pct = Math.round(val * 100);
                        return (
                          <div key={key} className="space-y-1.5">
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-300 font-medium">{confidenceLabels[key] || key}</span>
                              <span className="text-gray-400 font-bold">{pct}%</span>
                            </div>
                            <div className="w-full bg-[#070b13] rounded-full h-2 overflow-hidden border border-indigo-950">
                              <div
                                className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full rounded-full transition-all duration-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Timeline: Recent Experiences */}
                <div className="lg:col-span-3 bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                  <h3 className="text-sm font-bold text-white mb-6 flex items-center border-b border-indigo-950/60 pb-3">
                    <Clock className="w-4 h-4 text-violet-400 mr-2" />
                    直近の経験履歴タイムライン (Recent Experiences)
                  </h3>
                  <div className="relative pl-6 border-l-2 border-indigo-950/40 space-y-6">
                    {data.recent_experiences.map((exp, i) => (
                      <div key={i} className="relative group">
                        {/* Timeline node */}
                        <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full border-2 border-indigo-950 bg-violet-500 shadow-lg shadow-violet-500/20 group-hover:scale-110 transition duration-150" />
                        
                        <div className="bg-[#070b13]/60 border border-indigo-950/80 rounded-xl p-4 hover:border-violet-500/20 transition duration-150">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <span className="text-[10px] font-bold text-violet-300 bg-violet-950/40 px-2 py-0.5 rounded border border-violet-500/10 uppercase">
                                {exp.route}
                              </span>
                              {exp.scene && (
                                <span className="text-[10px] font-medium text-cyan-300 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-500/10">
                                  {exp.scene}
                                </span>
                              )}
                            </div>
                            <span className="text-[10px] text-gray-500">{new Date(exp.ts).toLocaleString("ja-JP")}</span>
                          </div>
                          <p className="text-xs text-gray-300 leading-relaxed font-mono">{exp.summary}</p>
                        </div>
                      </div>
                    ))}
                    {data.recent_experiences.length === 0 && (
                      <p className="text-xs text-gray-400">まだ経験履歴が記録されていません。</p>
                    )}
                  </div>
                </div>

              </div>
            )}

            {/* 2. PRACTICAL TAB */}
            {activeTab === "practical" && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Left Side: Scenes Menu */}
                <div className="space-y-2">
                  <h3 className="text-sm font-bold text-white mb-4 px-2">実務場面インデックス</h3>
                  {data.practical_scenes.map((scene) => (
                    <button
                      key={scene.id}
                      onClick={() => setActiveSceneTab(scene.id)}
                      className={`w-full text-left p-3.5 rounded-xl transition duration-150 flex items-center justify-between border ${
                        activeSceneTab === scene.id
                          ? "bg-violet-600/90 text-white border-violet-500/30 shadow-md shadow-violet-500/10"
                          : "bg-[#0b101c]/40 text-gray-300 hover:bg-[#070b13]/55 border-indigo-500/5"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <Database className="w-4 h-4 flex-shrink-0 opacity-70" />
                        <span className="text-xs font-semibold">{scene.label}</span>
                      </div>
                      
                      {scene.learned_entry_count > 0 && (
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                          activeSceneTab === scene.id
                            ? "bg-white text-violet-700"
                            : "bg-violet-950/40 text-violet-300 border border-violet-500/15"
                        }`}>
                          +{scene.learned_entry_count}
                        </span>
                      )}
                    </button>
                  ))}
                </div>

                {/* Right Side: Scene Details */}
                <div className="lg:col-span-2 space-y-6">
                  {activeScene && (
                    <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl space-y-6">
                      <div className="border-b border-indigo-950/60 pb-4">
                        <div className="flex items-center space-x-2">
                          <span className="text-[10px] uppercase tracking-wider font-bold text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded border border-violet-500/10">
                            Practical Scene Details
                          </span>
                          {activeScene.learned_entry_count > 0 && (
                            <span className="text-[10px] font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/10">
                              学習エントリー: {activeScene.learned_entry_count}件
                            </span>
                          )}
                        </div>
                        <h2 className="text-lg font-bold text-white mt-2">{activeScene.label}</h2>
                      </div>

                      {/* Three-Layer Knowledge */}
                      <div className="space-y-4">
                        {/* 1. Procedure Layer */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-bold text-violet-400 uppercase tracking-wider">第一層: 手順層 (Procedure)</h4>
                          <ul className="space-y-1.5 pl-4 list-disc text-xs text-gray-300">
                            {activeScene.procedure_layer.map((item, idx) => (
                              <li key={idx} className="leading-relaxed">{item}</li>
                            ))}
                          </ul>
                        </div>

                        {/* 2. Meaning Layer */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider">第二層: 意味層 (Meaning)</h4>
                          <ul className="space-y-1.5 pl-4 list-disc text-xs text-gray-300">
                            {activeScene.meaning_layer.map((item, idx) => (
                              <li key={idx} className="leading-relaxed">{item}</li>
                            ))}
                          </ul>
                        </div>

                        {/* 3. Judgment Layer */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider">第三層: 判断層 (Judgment)</h4>
                          <ul className="space-y-1.5 pl-4 list-disc text-xs text-gray-300">
                            {activeScene.judgment_layer.map((item, idx) => (
                              <li key={idx} className="leading-relaxed">{item}</li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Learned Sources within this scene */}
                      {activeScene.learned_sources && activeScene.learned_sources.length > 0 && (
                        <div className="border-t border-indigo-950/60 pt-4 space-y-2">
                          <h4 className="text-xs font-bold text-gray-400 flex items-center">
                            <Clipboard className="w-3.5 h-3.5 mr-1.5" />
                            学習ソース (Learned Sources)
                          </h4>
                          <div className="space-y-1.5">
                            {activeScene.learned_sources.map((src, i) => (
                              <div key={i} className="text-[11px] font-mono text-gray-400 bg-indigo-950/20 px-3 py-1 rounded border border-indigo-950">
                                {src}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Cumulative Learned Sources List */}
                  <div className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl">
                    <h3 className="text-xs font-bold text-white mb-4 uppercase tracking-wider flex items-center border-b border-indigo-950/60 pb-3">
                      <Database className="w-4 h-4 text-violet-400 mr-2" />
                      蓄積された知識ソース一覧 (Learned Sources Cumulative)
                    </h3>
                    <div className="space-y-2 max-h-[240px] overflow-y-auto">
                      {data.learned_sources.map((source, idx) => (
                        <div key={idx} className="text-[11px] font-mono text-gray-400 bg-[#070b13]/60 px-3 py-2 rounded-lg border border-indigo-950">
                          {source}
                        </div>
                      ))}
                      {data.learned_sources.length === 0 && (
                        <p className="text-xs text-gray-400">蓄積された学習ソースはまだありません。</p>
                      )}
                    </div>
                  </div>
                </div>

              </div>
            )}

            {/* 3. FEEDBACK TAB */}
            {activeTab === "feedback" && (
              <div className="space-y-6">
                
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {Object.keys(data.human_feedback_summary).map((routeKey) => {
                    const fb = data.human_feedback_summary[routeKey];
                    const posRate = Math.round(fb.positive_rate * 100);
                    return (
                      <div key={routeKey} className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-5 backdrop-blur-xl flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-bold text-violet-300 bg-violet-950/40 px-2 py-0.5 rounded border border-violet-500/10 uppercase">
                              {routeKey}
                            </span>
                            <span className="text-xs text-gray-400 font-bold">{fb.total_count}件</span>
                          </div>
                          
                          <div className="my-3 flex items-baseline">
                            <span className="text-2xl font-extrabold text-white">{posRate}%</span>
                            <span className="text-[10px] text-gray-400 ml-1 font-semibold">Positive</span>
                          </div>
                        </div>

                        {/* Progress Bar for ratio */}
                        <div className="space-y-1">
                          <div className="w-full bg-[#070b13] rounded-full h-1.5 overflow-hidden border border-indigo-950">
                            <div
                              className="bg-emerald-500 h-full rounded-full"
                              style={{ width: `${posRate}%` }}
                            />
                          </div>
                          <div className="flex justify-between text-[9px] text-gray-500 font-medium">
                            <span>高評価: {fb.positive_count}</span>
                            <span>低評価: {fb.negative_count}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Route Specific Detail List */}
                <div className="space-y-6">
                  {Object.keys(data.human_feedback_summary).map((routeKey) => {
                    const fb = data.human_feedback_summary[routeKey];
                    if (fb.total_count === 0) return null;
                    return (
                      <div key={routeKey} className="bg-[#0b101c]/40 border border-indigo-500/10 rounded-2xl p-6 backdrop-blur-xl space-y-4">
                        <div className="flex items-center justify-between border-b border-indigo-950/60 pb-3">
                          <h3 className="text-sm font-bold text-white uppercase">{routeKey} - フィードバック詳細</h3>
                          <span className="text-xs text-gray-400 font-medium">満足度: {Math.round(fb.positive_rate * 100)}%</span>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                          
                          {/* Left: Quotes & Comments */}
                          <div className="space-y-3">
                            <h4 className="text-xs font-bold text-gray-400 flex items-center">
                              <MessageSquare className="w-3.5 h-3.5 mr-1.5" />
                              最近のコメント (Kobayashiさんの反応)
                            </h4>
                            <div className="space-y-2">
                              {fb.recent_comments.map((comment, i) => (
                                <div key={i} className="text-xs text-gray-300 bg-[#070b13]/60 px-4 py-3 rounded-xl border border-indigo-950/80 italic leading-relaxed font-serif">
                                  &ldquo;{comment}&rdquo;
                                </div>
                              ))}
                              {fb.recent_comments.length === 0 && (
                                <p className="text-[11px] text-gray-500">コメントは記録されていません。</p>
                              )}
                            </div>
                          </div>

                          {/* Right: Positive vs Negative starts */}
                          <div className="space-y-4">
                            {/* Positive Starts */}
                            <div className="space-y-2">
                              <h4 className="text-xs font-bold text-emerald-400 flex items-center">
                                <Award className="w-3.5 h-3.5 mr-1.5" />
                                効果的だった冒頭 (Positive starts)
                              </h4>
                              <div className="space-y-1.5">
                                {fb.positive_starts.map((start, i) => (
                                  <div key={i} className="text-[11px] text-emerald-300 bg-emerald-950/20 px-3 py-2 rounded-lg border border-emerald-950/30 leading-normal font-mono">
                                    {start}
                                  </div>
                                ))}
                                {fb.positive_starts.length === 0 && (
                                  <p className="text-[11px] text-gray-500">該当するデータはありません。</p>
                                )}
                              </div>
                            </div>

                            {/* Negative Starts */}
                            <div className="space-y-2">
                              <h4 className="text-xs font-bold text-rose-400 flex items-center">
                                <ShieldAlert className="w-3.5 h-3.5 mr-1.5" />
                                機械的/薄いとされた冒頭 (Negative starts)
                              </h4>
                              <div className="space-y-1.5">
                                {fb.negative_starts.map((start, i) => (
                                  <div key={i} className="text-[11px] text-rose-300 bg-rose-950/20 px-3 py-2 rounded-lg border border-rose-950/30 leading-normal font-mono">
                                    {start}
                                  </div>
                                ))}
                                {fb.negative_starts.length === 0 && (
                                  <p className="text-[11px] text-gray-500">該当するデータはありません。</p>
                                )}
                              </div>
                            </div>

                          </div>

                        </div>
                      </div>
                    );
                  })}
                </div>

              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

// ── Shared Subcomponents ───────────────────────────────────────────────────

function Loader() {
  return (
    <div className="relative w-12 h-12 flex items-center justify-center">
      <div className="absolute w-12 h-12 rounded-full border-4 border-violet-500/20 border-t-violet-500 animate-spin" />
      <Brain className="w-6 h-6 text-violet-400" />
    </div>
  );
}
