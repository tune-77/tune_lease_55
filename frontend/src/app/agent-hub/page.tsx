"use client";

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { 
  Bot, 
  Terminal, 
  Search, 
  TrendingUp, 
  FileText, 
  Users, 
  Send, 
  AlertTriangle, 
  RefreshCcw, 
  BookOpen, 
  Zap, 
  Cpu, 
  Network,
  History,
  Play,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ChevronRight,
  MessageSquare
} from "lucide-react";
import { triggerMebuki } from "../../components/layout/FloatingMebuki";

interface Thought {
  ts: string;
  agent: string;
  thought: string;
  icon: string;
}

interface Agent {
  id: string;
  name: string;
  icon: React.ReactNode;
  description: string;
  category: string;
  color: string;
}

const AGENTS: Agent[] = [
  { id: "benchmark", name: "業界ベンチマーク自動取得", icon: <Search className="w-5 h-5" />, description: "AIが業種別の財務指標目安を推定します", category: "Analysis", color: "blue" },
  { id: "market", name: "金利・市況モニタリング", icon: <TrendingUp className="w-5 h-5" />, description: "最新の市場金利や経済動向をレポート", category: "Market", color: "indigo" },
  { id: "gunshi", name: "審査理由書自動生成", icon: <FileText className="w-5 h-5" />, description: "軍師モードによる高度な稟議書作成", category: "Report", color: "emerald" },
  { id: "team", name: "エージェントチーム議論", icon: <Users className="w-5 h-5" />, description: "多角的な専門家たちが是非を議論", category: "Debate", color: "violet" },
  { id: "slack", name: "Slack通知・営業フォロー", icon: <Send className="w-5 h-5" />, description: "AIが営業向けの追客文案を作成・送信", category: "Action", color: "orange" },
  { id: "anomaly", name: "異常検知（Z-Score）", icon: <AlertTriangle className="w-5 h-5" />, description: "統計的外れ値を持つ案件を自動検出", category: "Monitoring", color: "rose" },
  { id: "retrain", name: "モデル再学習トリガー", icon: <RefreshCcw className="w-5 h-5" />, description: "データ蓄積に応じた係数の自動最適化", category: "Admin", color: "amber" },
  { id: "novel", name: "文豪AI「波乱丸」", icon: <BookOpen className="w-5 h-5" />, description: "エージェントたちの日常を小説化", category: "Narrative", color: "pink" },
];

const INDUSTRIES = [
  "製造業", "建設業", "卸売業", "小売業", "運輸業", "情報通信業",
  "不動産業", "医療・福祉", "サービス業", "飲食業", "農業・漁業",
  "金融・保険業", "教育・学習支援業", "宿泊業", "その他"
];

const BENCHMARK_LABELS: Record<string, string> = {
  op_margin: "営業利益率",
  equity_ratio: "自己資本比率",
  roa: "ROA",
  current_ratio: "流動比率",
  dscr: "DSCR（債務返済余裕率）",
};

export default function AgentHubPage() {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [latestNovel, setLatestNovel] = useState<any>(null);
  const [loadingThoughts, setLoadingThoughts] = useState(true);
  const [benchmarkIndustry, setBenchmarkIndustry] = useState("製造業");

  const thoughtsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchThoughts();
    fetchLatestNovel();
    const interval = setInterval(fetchThoughts, 15000); // 15秒おきに更新
    return () => clearInterval(interval);
  }, []);

  const fetchThoughts = async () => {
    try {
      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/agent_hub/thoughts?limit=30`);
      setThoughts(res.data.thoughts || []);
    } catch (err) {
      console.error("Failed to fetch thoughts", err);
    } finally {
      setLoadingThoughts(false);
    }
  };

  const fetchLatestNovel = async () => {
    try {
      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/agent_hub/novel/latest`);
      setLatestNovel(res.data.novel);
    } catch (err) {
      console.error("Failed to fetch novel", err);
    }
  };

  const runAgent = async (agentId: string) => {
    setActiveAgent(agentId);
    setIsRunning(true);
    setResult(null);

    try {
      if (agentId === "novel") {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/agent_hub/novel/generate`);
        setResult(res.data);
        fetchLatestNovel();
        fetchThoughts();
      } else {
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/agent_hub/run_agent`, {
          agent_id: agentId,
          params: agentId === "benchmark" ? { industry: benchmarkIndustry } : {}
        });
        setResult(res.data.result);
        fetchThoughts();
      }
    } catch (err: any) {
      setResult({ error: err.response?.data?.detail || "実行に失敗しました" });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c] text-slate-200 p-8 pt-24 font-sans selection:bg-violet-500/30">
      {/* Background patterns */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-20">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-violet-600/20 blur-[120px] rounded-full" />
        <div className="absolute top-[20%] -right-[10%] w-[35%] h-[35%] bg-blue-600/20 blur-[100px] rounded-full" />
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-10" />
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-gradient-to-br from-violet-600 to-indigo-600 rounded-lg shadow-lg shadow-violet-900/20">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-4xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-slate-400">
                汎用エージェントハブ
              </h1>
            </div>
            <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
              自律型AIエージェント群が監視・分析・創作を行う、リースの統合コントロールセンター。
            </p>
          </div>
          
          <div className="flex items-center gap-4 px-4 py-2 bg-slate-900/50 backdrop-blur-md border border-slate-800/50 rounded-full">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-emerald-400 uppercase tracking-widest">System Active</span>
            </div>
            <div className="w-px h-4 bg-slate-800" />
            <div className="text-xs text-slate-500 tabular-nums">
              {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Agent Grid */}
          <div className="lg:col-span-8 space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {AGENTS.map((agent) => (
                <button
                  key={agent.id}
                  onClick={() => runAgent(agent.id)}
                  disabled={isRunning}
                  className={`group relative text-left p-6 rounded-2xl transition-all duration-300 border backdrop-blur-xl ${
                    activeAgent === agent.id 
                      ? 'bg-violet-600/10 border-violet-500/50 shadow-lg shadow-violet-900/20' 
                      : 'bg-slate-900/40 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className={`p-3 rounded-xl mb-4 transition-colors ${
                      activeAgent === agent.id ? 'bg-violet-600 text-white' : 'bg-slate-800 text-slate-400 group-hover:bg-slate-700 group-hover:text-white'
                    }`}>
                      {agent.icon}
                    </div>
                    <div className="px-2 py-1 bg-slate-800/50 border border-slate-700/50 rounded text-[10px] font-bold text-slate-500 tracking-wider uppercase">
                      {agent.category}
                    </div>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-1">{agent.name}</h3>
                  <p className="text-sm text-slate-400">{agent.description}</p>
                  
                  {activeAgent === agent.id && isRunning && (
                    <div className="absolute bottom-4 right-4">
                      <Loader2 className="w-5 h-5 text-violet-500 animate-spin" />
                    </div>
                  )}
                  
                  {activeAgent === agent.id && !isRunning && result && (
                    <div className="absolute bottom-4 right-4">
                      <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                    </div>
                  )}
                </button>
              ))}
            </div>

            {/* Benchmark: Industry Selector */}
            {activeAgent === "benchmark" && (
              <div className="flex items-center gap-4 p-5 bg-slate-900/60 border border-blue-500/20 rounded-2xl">
                <Search className="w-5 h-5 text-blue-400 shrink-0" />
                <div className="flex-1">
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">対象業種を選択</div>
                  <select
                    value={benchmarkIndustry}
                    onChange={e => setBenchmarkIndustry(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-white font-bold rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {INDUSTRIES.map(ind => (
                      <option key={ind} value={ind}>{ind}</option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={() => runAgent("benchmark")}
                  disabled={isRunning}
                  className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-black px-6 py-2.5 rounded-xl flex items-center gap-2 transition-colors"
                >
                  {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  取得
                </button>
              </div>
            )}

            {/* Execution Result Panel */}
            <div className={`rounded-2xl border transition-all duration-500 overflow-hidden ${
              activeAgent ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-4 scale-95 pointer-events-none'
            } ${
              result?.error ? 'bg-rose-950/20 border-rose-900/30' : 'bg-slate-900/60 border-slate-800'
            }`}>
              <div className="p-4 border-b border-white/5 bg-white/5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-bold uppercase tracking-widest text-slate-400">Agent Output</span>
                </div>
                {activeAgent && (
                  <div className="text-xs font-medium text-violet-400">
                    Agent: {AGENTS.find(a => a.id === activeAgent)?.name}
                  </div>
                )}
              </div>
              
              <div className="p-8">
                {isRunning ? (
                  <div className="flex flex-col items-center justify-center py-12 space-y-4">
                    <Loader2 className="w-10 h-10 text-violet-500 animate-spin" />
                    <p className="text-slate-400 animate-pulse tracking-widest uppercase text-sm">Processing Neural Context...</p>
                  </div>
                ) : result?.error ? (
                  <div className="flex items-start gap-4 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl">
                    <AlertCircle className="w-6 h-6 text-rose-500 flex-shrink-0" />
                    <div className="text-rose-200">{result.error}</div>
                  </div>
                ) : result ? (
                  <div className="space-y-6">
                    {/* Render different result types */}
                    {activeAgent === "benchmark" && (
                      <div>
                        <div className="text-slate-400 text-sm font-bold mb-4 flex items-center gap-2">
                          <Search className="w-4 h-4 text-blue-400" />
                          {benchmarkIndustry} の業界平均財務指標（AI推定）
                        </div>
                        {result.error ? (
                          <div className="text-rose-300 text-sm">{result.error}<br/><span className="text-slate-500 text-xs">{result.raw}</span></div>
                        ) : (
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {Object.entries(result).filter(([key]) => key !== "_source").map(([key, val]: [string, any]) => {
                              const label = BENCHMARK_LABELS[key] || key;
                              const unit = key === "dscr" ? "倍" : "%";
                              return (
                                <div key={key} className="p-5 bg-slate-800/60 border border-slate-700/50 rounded-xl">
                                  <div className="text-xs text-blue-300 font-bold mb-2">{label}</div>
                                  <div className="text-3xl font-black text-white">
                                    {typeof val === 'number' ? val.toFixed(1) : val}
                                    <span className="text-sm text-slate-400 ml-1">{unit}</span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        )}
                        <p className="text-[10px] text-slate-600 mt-4">
                          {result._source === "ai"
                            ? "※ AI（Gemini）による推定値です。実際の業界統計とは異なる場合があります。"
                            : "※ AIが一時利用不可のため、静的ベンチマークデータを表示しています。"}
                        </p>
                      </div>
                    )}
                    {activeAgent === "market" && (
                      <div className="text-slate-300 leading-relaxed text-lg whitespace-pre-wrap italic">
                        &quot;{result.content}&quot;
                      </div>
                    )}
                    {(activeAgent === "novel" || activeAgent === "gunshi") && (
                      <div className="space-y-4">
                        <h4 className="text-xl font-bold text-white border-b border-white/5 pb-2">{result.title}</h4>
                        <div className="text-slate-300 leading-relaxed max-h-[400px] overflow-y-auto pr-4 scrollbar-thin scrollbar-thumb-slate-700">
                          {result.body?.split('\n').map((line: string, i: number) => (
                            <p key={i} className="mb-4">{line}</p>
                          ))}
                        </div>
                      </div>
                    )}
                    {/* Fallback for others */}
                    {["anomaly", "retrain", "team", "slack"].includes(activeAgent!) && (
                      <pre className="p-4 bg-black/60 rounded-xl text-emerald-400 font-mono text-sm overflow-x-auto border border-emerald-900/30">
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-slate-600">
                    <Zap className="w-12 h-12 mb-3 opacity-20" />
                    <p>Select an agent to begin simulation</p>
                  </div>
                )}
              </div>
            </div>

            {/* Latest Novel Feature Section */}
            {latestNovel && (
              <div className="mt-12 group cursor-pointer overflow-hidden p-1 rounded-3xl bg-gradient-to-br from-pink-500/20 via-violet-500/20 to-transparent hover:from-pink-500/30 hover:via-violet-500/30 transition-all duration-700 hover:shadow-2xl hover:shadow-violet-500/10">
                <div className="bg-[#0f0f13] p-8 rounded-[22px] flex flex-col md:flex-row gap-8 items-center border border-white/5">
                  <div className="w-full md:w-32 h-44 bg-gradient-to-b from-slate-800 to-slate-950 rounded-xl shadow-2xl flex-shrink-0 flex flex-col items-center justify-center border border-white/10 relative overflow-hidden group-hover:scale-105 transition-transform duration-500">
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/old-map.png')] opacity-10" />
                    <div className="w-10 h-0.5 bg-pink-500/50 mb-6" />
                    <BookOpen className="w-10 h-10 text-pink-400/80 mb-2" />
                    <div className="text-[10px] font-bold text-slate-500 tracking-[0.2em] uppercase">VOL.{latestNovel.episode_no}</div>
                    <div className="absolute bottom-3 text-[8px] text-slate-600 font-serif">波乱丸 謹刊</div>
                  </div>
                  <div className="flex-1 space-y-4">
                    <div className="flex items-center gap-3">
                      <span className="px-2 py-0.5 bg-pink-500/10 text-pink-400 text-[10px] font-bold uppercase tracking-widest rounded border border-pink-500/20">Serial Narrative</span>
                      <span className="text-xs text-slate-500 font-mono tracking-tighter uppercase">{latestNovel.week_label}</span>
                    </div>
                    <h2 className="text-3xl font-serif text-white tracking-tight leading-tight group-hover:text-pink-100 transition-colors">
                      {latestNovel.title}
                    </h2>
                    <p className="text-slate-400 line-clamp-3 text-lg leading-relaxed font-serif italic opacity-80">
                      {latestNovel.body?.substring(0, 300)}...
                    </p>
                    <div className="flex items-center gap-2 text-pink-400 font-bold text-xs uppercase tracking-widest group-hover:translate-x-2 transition-transform">
                      Read Episode <ChevronRight className="w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Thoughts Feed */}
          <div className="lg:col-span-4 self-start sticky top-24">
            <div className="rounded-2xl bg-slate-900/40 border border-slate-800 backdrop-blur-xl overflow-hidden shadow-2xl">
              <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <MessageSquare className="w-5 h-5 text-violet-400" />
                    <span className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-500 rounded-full border-2 border-slate-900" />
                  </div>
                  <h2 className="text-sm font-bold uppercase tracking-widest">Agent Thoughts</h2>
                </div>
                <button 
                  onClick={fetchThoughts}
                  className="p-1.5 hover:bg-white/5 rounded-lg text-slate-500 hover:text-white transition-all transform hover:rotate-180 duration-500"
                >
                  <RefreshCcw className="w-4 h-4" />
                </button>
              </div>

              <div className="h-[calc(100vh-280px)] overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-800 hover:scrollbar-thumb-slate-700">
                {loadingThoughts ? (
                  <div className="flex justify-center py-12">
                    <Loader2 className="w-6 h-6 text-slate-700 animate-spin" />
                  </div>
                ) : thoughts.length === 0 ? (
                  <div className="text-center py-12 text-slate-600 text-sm italic">
                    <Terminal className="w-10 h-10 mx-auto mb-2 opacity-10" />
                    No internal logs available
                  </div>
                ) : (
                  thoughts.map((t, i) => (
                    <div 
                      key={i} 
                      className="group p-4 bg-slate-900/50 border border-slate-800/50 rounded-xl hover:border-slate-700 hover:bg-slate-800/50 transition-all duration-300"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-lg">{t.icon || "💭"}</span>
                        <span className="text-xs font-bold text-violet-400 uppercase tracking-wider">{t.agent}</span>
                        <span className="text-[10px] text-slate-600 tabular-nums ml-auto font-mono">
                          {new Date(t.ts).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-[13px] text-slate-400 leading-relaxed font-light group-hover:text-slate-200 transition-colors">
                        {t.thought}
                      </p>
                    </div>
                  ))
                )}
                <div ref={thoughtsEndRef} />
              </div>

              <div className="p-4 bg-slate-800/20 border-t border-slate-800 text-[10px] text-slate-600 font-mono flex justify-between items-center">
                <span>BUFFER: 0x{thoughts.length.toString(16).toUpperCase()}</span>
                <span>REALTIME TELEMETRY</span>
              </div>
            </div>
            
            {/* Quick Action */}
            <button 
              onClick={() => triggerMebuki("guide", "エージェントハブについて教えて")}
              className="mt-6 w-full py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 font-bold text-sm uppercase tracking-widest text-white shadow-lg shadow-violet-900/40 hover:shadow-violet-600/50 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-2"
            >
              <Bot className="w-5 h-5" />
              Ask Mebuki Assistant
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
