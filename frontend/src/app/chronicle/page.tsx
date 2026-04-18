"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import { 
  Milestone, 
  History, 
  Globe, 
  Wind, 
  Orbit, 
  Rewind, 
  Play, 
  AlertCircle, 
  CheckCircle, 
  Waves, 
  Layers, 
  Ghost,
  Sun,
  Telescope,
  MessageCircle,
  Database,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  Activity,
  Infinity as InfinityIcon,
  Loader2,
  Clock,
  Sparkles
} from "lucide-react";
import { triggerMebuki } from "../../components/layout/FloatingMebuki";

interface Snap {
  id: string;
  ts: string;
  comment: string;
  overrides: any;
}

interface SimRound {
  round_no: number;
  year: number;
  summary: string;
  events: any[];
  created_at: string;
}

interface ArchaiaLog {
  round_no: number;
  civ_name: string;
  event_type: string;
  bungo_style: string;
  narrative: string;
}

export default function ChroniclePage() {
  const [summary, setSummary] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [snapshots, setSnapshots] = useState<Snap[]>([]);
  const [simHistory, setSimHistory] = useState<SimRound[]>([]);
  const [archaiaLogs, setArchaiaLogs] = useState<ArchaiaLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("monitoring"); // monitoring, history, rollback, simulation
  const [simulating, setSimulating] = useState(false);

  // 安全な数値フォーマッター
  const pct = (v: any) => {
    const n = Number(v);
    if (!isFinite(n)) return "---";
    return (n * 100).toFixed(1);
  };
  const pctRaw = (v: any) => {
    const n = Number(v);
    if (!isFinite(n)) return 0;
    return n * 100;
  };

  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    setLoading(true);
    try {
      const safe = async (promise: Promise<any>, fallback: any) => {
        try { const res = await promise; return res; }
        catch { return { data: fallback }; }
      };

      const [sum, hist, snaps, sim, log] = await Promise.all([
        safe(axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/summary`), null),
        safe(axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/history`), { history: [] }),
        safe(axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/snapshots`), { snapshots: [] }),
        safe(axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/simulation/history`), { history: [] }),
        safe(axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/simulation/archaia_log`), { logs: [] }),
      ]);
      if (sum.data) setSummary(sum.data);
      setHistory(hist.data?.history || []);
      setSnapshots(snaps.data?.snapshots || []);
      setSimHistory(sim.data?.history || []);
      setArchaiaLogs(log.data?.logs || []);
    } catch (err) {
      console.error("Failed to fetch chronicle data", err);
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    setSimulating(true);
    try {
      await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/simulation/round`);
      await fetchAll();
    } catch (err: any) {
      alert("Simulation failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setSimulating(false);
    }
  };

  const handleRollback = async (snapId: string) => {
    if (!confirm(`Are you sure you want to rollback to ${snapId}?`)) return;
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/api/chronicle/rollback`, { snapshot_id: snapId });
      if (res.data.status === "success") {
        alert("Rollback Successful");
        fetchAll();
      } else {
        alert("Rollback Failed");
      }
    } catch (err) {
      alert("Error during rollback");
    }
  };

  if (loading && !summary) {
    return (
      <div className="min-h-screen bg-[#050508] flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-violet-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#05060a] text-slate-300 p-8 pt-24 font-sans overflow-x-hidden">
      {/* Space Particles / Background */}
      <div className="fixed inset-0 pointer-events-none opacity-40">
        <div className="absolute top-[10%] left-[20%] w-[30%] h-[30%] bg-violet-900/10 blur-[150px] rounded-full animate-pulse" />
        <div className="absolute bottom-[20%] right-[10%] w-[40%] h-[40%] bg-blue-900/10 blur-[180px] rounded-full" />
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-30" />
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <div className="mb-12 relative">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-tr from-indigo-600 to-violet-600 rounded-2xl shadow-xl shadow-indigo-900/30">
              <Globe className="w-7 h-7 text-white animate-[spin_10s_linear_infinite]" />
            </div>
            <div>
              <h1 className="text-5xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white via-indigo-200 to-slate-400">
                文明年代記
              </h1>
              <p className="text-slate-400 font-medium tracking-wide uppercase text-xs mt-1 font-mono">
                Chronicles of Governance & Evolutionary Simulation
              </p>
            </div>
          </div>
          
          <div className="h-px w-full bg-gradient-to-r from-indigo-500/50 via-transparent to-transparent mb-8" />
          
          {/* Navigation Tabs */}
          <div className="flex flex-wrap gap-2 mb-10">
            {["monitoring", "history", "rollback", "simulation"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2.5 rounded-full text-sm font-bold tracking-widest uppercase transition-all duration-300 border ${
                  activeTab === tab 
                    ? 'bg-white text-[#05060a] border-white shadow-lg shadow-white/10' 
                    : 'bg-white/5 text-slate-500 border-white/10 hover:bg-white/10 hover:text-white'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-12">
          {activeTab === "monitoring" && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Activity className="w-20 h-20" />
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-[0.2em] mb-4">全体ベースライン承認率</h3>
                {!summary ? (
                  <div className="animate-pulse h-16 bg-white/5 rounded-xl mb-2" />
                ) : (
                  <div className="text-5xl font-black text-white tabular-nums mb-2">
                    {pct(summary?.baseline_rate)}%
                  </div>
                )}
                <div className="text-xs text-slate-400">全スコア済み案件の基準値 / {summary?.total_cases || 0}件</div>
              </div>

              <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Clock className="w-20 h-20" />
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-[0.2em] mb-4">直近30件 承認率</h3>
                {!summary ? (
                  <div className="animate-pulse h-16 bg-white/5 rounded-xl mb-2" />
                ) : (
                  <div className="text-5xl font-black text-white tabular-nums mb-2 flex items-baseline gap-3">
                    {pct(summary?.recent_rate)}%
                    <span className={`text-lg ${pctRaw(summary?.recent_rate) - pctRaw(summary?.baseline_rate) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {pctRaw(summary?.recent_rate) - pctRaw(summary?.baseline_rate) >= 0 ? <TrendingUp className="inline w-5 h-5 mb-1" /> : <TrendingDown className="inline w-5 h-5 mb-1" />}
                      {Math.abs(pctRaw(summary?.recent_rate) - pctRaw(summary?.baseline_rate)).toFixed(1)}%
                    </span>
                  </div>
                )}
                <div className="text-xs text-slate-400">ベースラインからの現在の変位</div>
              </div>

              <div className={`backdrop-blur-xl border rounded-3xl p-8 relative overflow-hidden group transition-colors duration-500 ${
                summary ? (
                  (summary?.drift || 0) >= summary?.warn_threshold ? 'bg-rose-500/10 border-rose-500/30' : 'bg-emerald-500/10 border-emerald-500/30'
                ) : 'bg-white/5 border-white/10'
              }`}>
                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                  {summary && (summary?.drift || 0) >= summary?.warn_threshold ? <Ghost className="w-20 h-20" /> : <Sparkles className="w-20 h-20" />}
                </div>
                <h3 className="text-slate-500 text-xs font-bold uppercase tracking-[0.2em] mb-4">系統エントロピー乖離幅</h3>
                {!summary ? (
                  <div className="animate-pulse h-16 bg-white/5 rounded-xl mb-2" />
                ) : (
                  <div className={`text-5xl font-black tabular-nums mb-2 ${
                    (summary?.drift || 0) >= summary?.warn_threshold ? 'text-rose-400' : 'text-emerald-400'
                  }`}>
                    {pct(summary?.drift)}%
                  </div>
                )}
                <div className="text-xs font-bold tracking-widest uppercase">
                  {!summary ? 'Loading...' : (
                    (summary?.drift || 0) >= summary?.warn_threshold ? '⚠️ System Instability Detected' : '✅ System Equilibrium Maintained'
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === "history" && (
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
              <div className="p-6 bg-white/5 border-b border-white/5 flex items-center gap-3">
                <History className="w-5 h-5 text-indigo-400" />
                <h2 className="font-bold uppercase tracking-widest text-sm">係数変更履歴 タイムライン</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-[#0c0d12] text-xs uppercase tracking-widest text-slate-500">
                    <tr>
                      <th className="px-8 py-4 font-bold">Timestamp</th>
                      <th className="px-8 py-4 font-bold">Type</th>
                      <th className="px-8 py-4 font-bold">Comment</th>
                      <th className="px-8 py-4 font-bold">Changes</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {history.map((h, i) => (
                      <tr key={i} className="hover:bg-white/5 transition-colors group">
                        <td className="px-8 py-5 text-xs font-mono text-slate-400 tabular-nums">{h.timestamp}</td>
                        <td className="px-8 py-5">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-widest border ${
                            h.change_type === 'auto' ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                          }`}>
                            {h.change_type}
                          </span>
                        </td>
                        <td className="px-8 py-5 text-sm font-medium text-slate-200">{h.comment || "—"}</td>
                        <td className="px-8 py-5 text-xs text-slate-500 italic">
                          {Object.keys(h.changed_keys || {}).length} keys updated
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === "rollback" && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {snapshots.map((snap) => (
                <div key={snap.id} className="bg-white/5 border border-white/10 rounded-3xl p-6 hover:border-white/30 transition-all duration-300 group">
                  <div className="flex items-start justify-between mb-6">
                    <div className="p-3 bg-white/5 rounded-2xl group-hover:bg-white/10 transition-colors">
                      <Rewind className="w-6 h-6 text-indigo-400" />
                    </div>
                    <div className="text-[10px] font-mono text-slate-500 bg-white/5 px-2 py-1 rounded">
                      ID: {snap.id}
                    </div>
                  </div>
                  <div className="text-xs text-slate-500 font-mono mb-2">{snap.ts}</div>
                  <h3 className="font-bold text-lg text-white mb-4 line-clamp-2 min-h-[3.5rem]">
                    {snap.comment || "（コメントなし）"}
                  </h3>
                  <div className="flex items-center justify-between pt-4 border-t border-white/5 mt-auto">
                    <div className="text-xs font-bold text-indigo-400">
                      {Object.keys(snap.overrides || {}).length} Overrides
                    </div>
                    <button
                      onClick={() => handleRollback(snap.id)}
                      className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-black uppercase tracking-widest rounded-full transition-all active:scale-95"
                    >
                      Rollback
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "simulation" && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* Simulation Control */}
              <div className="lg:col-span-8 space-y-8">
                <div className="p-1 rounded-[40px] bg-gradient-to-br from-indigo-500/30 via-violet-500/30 to-transparent">
                  <div className="bg-[#0c0d12] p-10 rounded-[39px] text-center relative overflow-hidden">
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/dust.png')] opacity-10 pointer-events-none" />
                    <div className="relative z-10">
                      <div className="w-20 h-20 bg-white/5 rounded-full mx-auto flex items-center justify-center mb-6">
                        <Waves className="w-10 h-10 text-indigo-400 animate-pulse" />
                      </div>
                      <h2 className="text-4xl font-black text-white tracking-widest uppercase mb-4">
                        宇宙の鼓動を刻む
                      </h2>
                      <p className="text-slate-400 max-w-xl mx-auto mb-10 text-lg leading-relaxed font-serif italic">
                        100年を1単位とする宇宙文明シミュレーション。AIが意思決定を行い、
                        審査が文明の未来を定めます。
                      </p>
                      
                      <button
                        onClick={runSimulation}
                        disabled={simulating}
                        className="group relative px-12 py-5 bg-white text-[#05060a] font-black uppercase tracking-[0.3em] rounded-full overflow-hidden transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
                      >
                        {simulating ? (
                          <div className="flex items-center gap-3">
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Accelerating Time...
                          </div>
                        ) : (
                          <div className="flex items-center gap-3">
                            <Play className="w-5 h-5 fill-current" />
                            Next 100 Years
                          </div>
                        )}
                      </button>
                      
                      <div className="mt-12 grid grid-cols-3 gap-8 max-w-2xl mx-auto">
                        <div className="space-y-1">
                          <div className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">Current Era</div>
                          <div className="text-lg font-black text-white font-mono">G.{simHistory[0]?.round_no || 0}</div>
                        </div>
                        <div className="space-y-1 text-indigo-400">
                          <div className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">Archaia Year</div>
                          <div className="text-lg font-black font-mono tracking-tighter italic">A.{simHistory[0]?.year || 0}</div>
                        </div>
                        <div className="space-y-1">
                          <div className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">Total Energy</div>
                          <div className="text-lg font-black text-white font-mono">{(simHistory.length * 1.2).toFixed(1)} Pj</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Simulation Timeline */}
                <div className="space-y-6">
                  {simHistory.map((round, i) => (
                    <div key={i} className="flex gap-6 group">
                      <div className="w-24 shrink-0 pt-2 flex flex-col items-end gap-2">
                        <div className="text-xs font-black text-indigo-500 tracking-tighter">A.{round.year}</div>
                        <div className="h-full w-px bg-white/5 group-last:bg-transparent ml-auto mr-1.5" />
                      </div>
                      <div className="flex-1 pb-10">
                        <div className="bg-white/5 border border-white/5 rounded-3xl p-6 group-hover:bg-white/10 transition-colors">
                          <div className="flex items-center justify-between mb-4">
                            <div className="text-[10px] font-bold px-2 py-1 bg-white/5 rounded text-slate-500 uppercase">第{round.round_no}ラウンド</div>
                            <div className="text-[10px] text-slate-600 font-mono">{round.created_at}</div>
                          </div>
                          <h4 className="text-lg font-bold text-white mb-2">{round.summary}</h4>
                          <div className="flex flex-wrap gap-2">
                            {round.events.map((ev, ei) => (
                              <div key={ei} className="px-3 py-1 bg-white/5 border border-white/5 rounded-full text-xs text-slate-400">
                                <span className="font-bold text-indigo-400 mr-2">{ev.title || ev.event_type}</span>
                                {ev.civ}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Archaia Log Feed */}
              <div className="lg:col-span-4 self-start sticky top-24">
                <div className="bg-white/5 border border-white/10 rounded-3xl overflow-hidden backdrop-blur-xl">
                  <div className="p-6 bg-white/5 border-b border-white/5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Globe className="w-5 h-5 text-indigo-400" />
                      <h2 className="font-bold uppercase tracking-widest text-sm italic">Archaia Logs</h2>
                    </div>
                  </div>
                  <div className="max-h-[calc(100vh-300px)] overflow-y-auto p-6 space-y-8 scrollbar-thin scrollbar-thumb-white/5 hover:scrollbar-thumb-white/10">
                    {archaiaLogs.length === 0 ? (
                      <div className="text-center py-20 text-slate-600 text-sm italic font-serif">
                        Waiting for civilization events...
                      </div>
                    ) : (
                      archaiaLogs.map((log, i) => (
                        <div key={i} className="space-y-3 font-serif">
                          <div className="flex items-center gap-3">
                            <div className="w-6 h-6 border border-white/20 rounded-full flex items-center justify-center text-[10px] font-mono text-slate-500">{log.round_no}</div>
                            <span className="text-xs font-bold text-indigo-400 uppercase tracking-[0.2em]">{log.civ_name}</span>
                          </div>
                          <div className="text-[10px] text-slate-600 italic border-l border-indigo-500/20 pl-3">
                            Writing Style: {log.bungo_style}
                          </div>
                          <p className="text-sm text-slate-300 leading-relaxed italic opacity-80">
                            &quot;{log.narrative}&quot;
                          </p>
                        </div>
                      ))
                    )}
                  </div>
                  <div className="p-4 bg-white/5 border-t border-white/5 text-[10px] text-center text-slate-600 font-mono tracking-widest uppercase">
                    End of Archives
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
