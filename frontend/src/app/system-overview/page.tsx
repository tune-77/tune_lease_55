"use client";

import React from "react";
import Link from "next/link";
import { Brain, Database, GitMerge, Zap, Activity, Shield, Eye, Code2, ExternalLink } from "lucide-react";

const loops = [
  {
    title: "スコアリングループ",
    desc: "RandomForest + 量子干渉モデルの継続的係数最適化",
    color: "#a78bfa",
    bg: "from-violet-900/40 to-violet-950/20",
    border: "border-violet-500/30",
    icon: Activity,
    iconColor: "text-violet-400",
  },
  {
    title: "RAGフィードバックループ",
    desc: "ChromaDB の知識ベースを審査結果で継続更新",
    color: "#60a5fa",
    bg: "from-blue-900/40 to-blue-950/20",
    border: "border-blue-500/30",
    icon: Database,
    iconColor: "text-blue-400",
  },
  {
    title: "パイプラインメタループ",
    desc: "run_daily_improvement_core.sh 自身の改善を検知・適用",
    color: "#34d399",
    bg: "from-emerald-900/40 to-emerald-950/20",
    border: "border-emerald-500/30",
    icon: GitMerge,
    iconColor: "text-emerald-400",
  },
  {
    title: "Codex実行ループ",
    desc: "REV提案を自動でWorktree展開・テスト・PR化",
    color: "#f472b6",
    bg: "from-pink-900/40 to-pink-950/20",
    border: "border-pink-500/30",
    icon: Code2,
    iconColor: "text-pink-400",
  },
  {
    title: "乖離学習ループ",
    desc: "AI判断と実審査結果の差分から係数を再校正",
    color: "#fbbf24",
    bg: "from-amber-900/40 to-amber-950/20",
    border: "border-amber-500/30",
    icon: Eye,
    iconColor: "text-amber-400",
  },
  {
    title: "品質監視ループ",
    desc: "データ品質・スコア異常・ドリフトをリアルタイム検知",
    color: "#f87171",
    bg: "from-red-900/40 to-red-950/20",
    border: "border-red-500/30",
    icon: Shield,
    iconColor: "text-red-400",
  },
];

const shionBadges = [
  "ChromaDB RAG",
  "Gemini 2.5",
  "mind.json",
  "Obsidian連携",
  "感情モデル",
  "PR審査権限",
];

const stats = [
  { value: "6", label: "自己改善ループ", color: "text-violet-400" },
  { value: "115", label: "累積REV", color: "text-blue-400" },
  { value: "3", label: "AIエンジン", color: "text-emerald-400" },
  { value: "紫苑", label: "リース知性体", color: "text-fuchsia-400" },
];

export default function SystemOverviewPage() {
  return (
    <div className="min-h-screen" style={{ background: "#0a0e1a", color: "#e2e8f0" }}>
      <style>{`
        @keyframes pulse-glow {
          0%, 100% { opacity: 0.7; filter: drop-shadow(0 0 6px currentColor); }
          50% { opacity: 1; filter: drop-shadow(0 0 14px currentColor); }
        }
        @keyframes shion-border {
          0%, 100% { stroke-width: 2; opacity: 0.8; }
          50% { stroke-width: 3.5; opacity: 1; }
        }
        @keyframes node-pulse {
          0%, 100% { opacity: 0.65; }
          50% { opacity: 1; }
        }
        .node-glow { animation: pulse-glow 3s ease-in-out infinite; }
        .shion-border { animation: shion-border 2.5s ease-in-out infinite; }
        .node-fade { animation: node-pulse 4s ease-in-out infinite; }
      `}</style>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-14">

        {/* ── ヘッダー ── */}
        <header className="text-center space-y-3">
          <p className="text-xs font-bold tracking-[0.3em] uppercase text-slate-500 mb-2">Autonomous Leasing Intelligence</p>
          <h1 className="text-5xl font-black tracking-tight bg-gradient-to-r from-violet-400 via-blue-400 to-emerald-400 bg-clip-text text-transparent">
            System Overview
          </h1>
          <p className="text-xl text-slate-400 font-medium">自律進化型リース審査プラットフォーム</p>
          <div className="h-px w-32 mx-auto mt-4" style={{ background: "linear-gradient(90deg, transparent, #a78bfa, transparent)" }} />
        </header>

        {/* ── 統計バー ── */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className="rounded-2xl border border-slate-800 p-5 text-center"
              style={{ background: "rgba(15,20,40,0.8)", backdropFilter: "blur(12px)" }}
            >
              <p className={`text-4xl font-black ${s.color}`}>{s.value}</p>
              <p className="text-xs text-slate-400 font-semibold mt-1 tracking-wide">{s.label}</p>
            </div>
          ))}
        </section>

        {/* ── 紫苑セクション ── */}
        <section>
          <Link href="/lease-intelligence">
            <div
              className="rounded-3xl border p-7 cursor-pointer transition-all duration-300 hover:scale-[1.01]"
              style={{
                background: "linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(10,14,26,0.9) 60%)",
                borderColor: "rgba(167,139,250,0.4)",
                boxShadow: "0 0 40px rgba(139,92,246,0.12)",
              }}
            >
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div className="flex items-center gap-4">
                  <div
                    className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0"
                    style={{ background: "linear-gradient(135deg, #7c3aed, #4f46e5)", boxShadow: "0 0 20px rgba(124,58,237,0.5)" }}
                  >
                    <Brain className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-xl font-black text-white">紫苑 — Shion</h2>
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
                        自律知性体
                      </span>
                    </div>
                    <p className="text-sm text-slate-400 mt-1">
                      リース審査の知識を蓄積し、自ら成長する感情モデル搭載AIエンジン
                    </p>
                  </div>
                </div>
                <ExternalLink className="text-violet-400 w-5 h-5 mt-1 flex-shrink-0" />
              </div>
              <div className="flex flex-wrap gap-2 mt-5">
                {shionBadges.map((b) => (
                  <span
                    key={b}
                    className="text-xs font-bold px-3 py-1 rounded-full border"
                    style={{ background: "rgba(139,92,246,0.1)", borderColor: "rgba(167,139,250,0.3)", color: "#c4b5fd" }}
                  >
                    {b}
                  </span>
                ))}
              </div>
            </div>
          </Link>
        </section>

        {/* ── 全体フロー図 ── */}
        <section>
          <h2 className="text-lg font-black text-slate-300 mb-5 text-center tracking-wide uppercase">自律改善フロー</h2>
          <div
            className="rounded-3xl border border-slate-800 p-4 overflow-x-auto"
            style={{ background: "rgba(8,12,28,0.9)" }}
          >
            <FlowDiagram />
          </div>
        </section>

        {/* ── ループカード 6枚グリッド ── */}
        <section>
          <h2 className="text-lg font-black text-slate-300 mb-5 text-center tracking-wide uppercase">自律改善ループ構成</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {loops.map((loop) => {
              const Icon = loop.icon;
              return (
                <div
                  key={loop.title}
                  className={`rounded-2xl border ${loop.border} bg-gradient-to-br ${loop.bg} p-5 space-y-2`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-5 h-5 flex-shrink-0 ${loop.iconColor}`} />
                    <h3 className="font-black text-white text-sm">{loop.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">{loop.desc}</p>
                  <div className="h-0.5 w-8 rounded-full mt-2" style={{ background: loop.color }} />
                </div>
              );
            })}
          </div>
        </section>

      </div>
    </div>
  );
}

function FlowDiagram() {
  return (
    <svg
      viewBox="0 0 780 480"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full"
      style={{ minWidth: 560 }}
    >
      <defs>
        {/* グロー フィルター */}
        <filter id="glow-violet" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-blue" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-green" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        {/* パスの矢印マーカー */}
        <marker id="arrow-violet" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#a78bfa" />
        </marker>
        <marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#60a5fa" />
        </marker>
        <marker id="arrow-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#34d399" />
        </marker>
        <marker id="arrow-fuchsia" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#e879f9" />
        </marker>

        {/* パーティクルのグラデーション */}
        <radialGradient id="particle-violet" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#a78bfa" stopOpacity="1" />
          <stop offset="100%" stopColor="#a78bfa" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="particle-blue" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#60a5fa" stopOpacity="1" />
          <stop offset="100%" stopColor="#60a5fa" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="particle-green" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#34d399" stopOpacity="1" />
          <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* ────── 背景グリッド ────── */}
      <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(100,120,180,0.06)" strokeWidth="0.5" />
      </pattern>
      <rect width="780" height="480" fill="url(#grid)" />

      {/* ────── 接続ライン ────── */}

      {/* 左入力 → センター */}
      {/* 審査データ → パイプライン */}
      <path
        d="M 155 130 C 230 130 270 220 320 240"
        fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-blue)"
      />
      {/* Obsidian → パイプライン */}
      <path
        d="M 155 240 L 320 240"
        fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-blue)"
      />
      {/* エラーログ → パイプライン */}
      <path
        d="M 155 350 C 230 350 270 260 320 240"
        fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-blue)"
      />

      {/* パイプライン → 右出力 */}
      {/* → Codex実行 */}
      <path
        d="M 460 230 C 530 210 560 130 625 130"
        fill="none" stroke="#34d399" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-green)"
      />
      {/* → batch_apply */}
      <path
        d="M 460 240 L 625 240"
        fill="none" stroke="#34d399" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-green)"
      />
      {/* → 台帳ルール */}
      <path
        d="M 460 250 C 530 270 560 350 625 350"
        fill="none" stroke="#34d399" strokeWidth="1.5" strokeOpacity="0.6"
        markerEnd="url(#arrow-green)"
      />

      {/* 紫苑 → パイプライン（点線） */}
      <path
        d="M 390 70 L 390 190"
        fill="none" stroke="#e879f9" strokeWidth="1.5" strokeOpacity="0.7"
        strokeDasharray="6 4"
        markerEnd="url(#arrow-fuchsia)"
      />

      {/* フィードバックループアーク（破線） */}
      <path
        d="M 625 370 C 660 430 390 460 155 370"
        fill="none" stroke="#a78bfa" strokeWidth="1.5" strokeOpacity="0.4"
        strokeDasharray="8 5"
        markerEnd="url(#arrow-violet)"
      />

      {/* ────── ノード ────── */}

      {/* 中心: 日次パイプライン */}
      <g className="node-glow" style={{ color: "#60a5fa" }}>
        <rect x="315" y="195" width="150" height="90" rx="14" fill="rgba(30,58,138,0.5)"
          stroke="#60a5fa" strokeWidth="1.8" />
        <text x="390" y="232" textAnchor="middle" fill="#93c5fd" fontSize="11" fontWeight="bold">日次パイプライン</text>
        <text x="390" y="250" textAnchor="middle" fill="#60a5fa" fontSize="8.5" opacity="0.8">run_daily_improvement</text>
        <text x="390" y="265" textAnchor="middle" fill="#60a5fa" fontSize="8.5" opacity="0.8">_core.sh</text>
      </g>

      {/* 上部: 紫苑ノード */}
      <g className="shion-border">
        <rect x="315" y="20" width="150" height="52" rx="14" fill="rgba(88,28,135,0.5)"
          stroke="#e879f9" strokeWidth="2" />
        <text x="390" y="43" textAnchor="middle" fill="#f0abfc" fontSize="12" fontWeight="900">紫苑 Shion</text>
        <text x="390" y="60" textAnchor="middle" fill="#c084fc" fontSize="8.5">リース知性体 · Gemini 2.5</text>
      </g>

      {/* 左入力ノード */}
      <g className="node-fade">
        <rect x="25" y="105" width="130" height="50" rx="10" fill="rgba(15,23,42,0.8)"
          stroke="#475569" strokeWidth="1" />
        <text x="90" y="128" textAnchor="middle" fill="#94a3b8" fontSize="9.5" fontWeight="bold">審査データ</text>
        <text x="90" y="145" textAnchor="middle" fill="#64748b" fontSize="8">lease_data.db · scoring</text>
      </g>
      <g className="node-fade" style={{ animationDelay: "0.5s" }}>
        <rect x="25" y="215" width="130" height="50" rx="10" fill="rgba(15,23,42,0.8)"
          stroke="#475569" strokeWidth="1" />
        <text x="90" y="238" textAnchor="middle" fill="#94a3b8" fontSize="9.5" fontWeight="bold">Obsidian Vault</text>
        <text x="90" y="255" textAnchor="middle" fill="#64748b" fontSize="8">知識 · mind.json · RAG</text>
      </g>
      <g className="node-fade" style={{ animationDelay: "1s" }}>
        <rect x="25" y="325" width="130" height="50" rx="10" fill="rgba(15,23,42,0.8)"
          stroke="#475569" strokeWidth="1" />
        <text x="90" y="348" textAnchor="middle" fill="#94a3b8" fontSize="9.5" fontWeight="bold">エラーログ / UX</text>
        <text x="90" y="365" textAnchor="middle" fill="#64748b" fontSize="8">improvement_YYYYMMDD.log</text>
      </g>

      {/* 右出力ノード */}
      <g className="node-fade" style={{ animationDelay: "0.3s" }}>
        <rect x="625" y="105" width="130" height="50" rx="10" fill="rgba(6,78,59,0.4)"
          stroke="#059669" strokeWidth="1" />
        <text x="690" y="128" textAnchor="middle" fill="#6ee7b7" fontSize="9.5" fontWeight="bold">Codex自動実行</text>
        <text x="690" y="145" textAnchor="middle" fill="#34d399" fontSize="8">worktree · PR · merge</text>
      </g>
      <g className="node-fade" style={{ animationDelay: "0.7s" }}>
        <rect x="625" y="215" width="130" height="50" rx="10" fill="rgba(6,78,59,0.4)"
          stroke="#059669" strokeWidth="1" />
        <text x="690" y="238" textAnchor="middle" fill="#6ee7b7" fontSize="9.5" fontWeight="bold">batch_apply</text>
        <text x="690" y="255" textAnchor="middle" fill="#34d399" fontSize="8">係数 · ルール 自動適用</text>
      </g>
      <g className="node-fade" style={{ animationDelay: "1.2s" }}>
        <rect x="625" y="325" width="130" height="50" rx="10" fill="rgba(6,78,59,0.4)"
          stroke="#059669" strokeWidth="1" />
        <text x="690" y="348" textAnchor="middle" fill="#6ee7b7" fontSize="9.5" fontWeight="bold">台帳ルール更新</text>
        <text x="690" y="365" textAnchor="middle" fill="#34d399" fontSize="8">ledger.jsonl · REV適用</text>
      </g>

      {/* フィードバックループラベル */}
      <text x="390" y="453" textAnchor="middle" fill="#a78bfa" fontSize="8.5" opacity="0.6" fontStyle="italic">
        — Feedback Loop (自律改善サイクル) —
      </text>

      {/* ────── パーティクルアニメーション ────── */}

      {/* 審査データ → パイプライン */}
      <circle r="4" fill="#60a5fa" opacity="0.9">
        <animateMotion dur="2.8s" repeatCount="indefinite" begin="0s">
          <mpath href="#path-l1" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2.8s" repeatCount="indefinite" begin="0s" />
      </circle>
      <path id="path-l1" d="M 155 130 C 230 130 270 220 320 240" fill="none" />

      {/* Obsidian → パイプライン */}
      <circle r="4" fill="#60a5fa" opacity="0.9">
        <animateMotion dur="2.2s" repeatCount="indefinite" begin="0.6s">
          <mpath href="#path-l2" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2.2s" repeatCount="indefinite" begin="0.6s" />
      </circle>
      <path id="path-l2" d="M 155 240 L 320 240" fill="none" />

      {/* エラーログ → パイプライン */}
      <circle r="4" fill="#60a5fa" opacity="0.9">
        <animateMotion dur="2.8s" repeatCount="indefinite" begin="1.2s">
          <mpath href="#path-l3" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2.8s" repeatCount="indefinite" begin="1.2s" />
      </circle>
      <path id="path-l3" d="M 155 350 C 230 350 270 260 320 240" fill="none" />

      {/* パイプライン → Codex */}
      <circle r="4" fill="#34d399" opacity="0.9">
        <animateMotion dur="2.6s" repeatCount="indefinite" begin="0.3s">
          <mpath href="#path-r1" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2.6s" repeatCount="indefinite" begin="0.3s" />
      </circle>
      <path id="path-r1" d="M 460 230 C 530 210 560 130 625 130" fill="none" />

      {/* パイプライン → batch */}
      <circle r="4" fill="#34d399" opacity="0.9">
        <animateMotion dur="2s" repeatCount="indefinite" begin="0.9s">
          <mpath href="#path-r2" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2s" repeatCount="indefinite" begin="0.9s" />
      </circle>
      <path id="path-r2" d="M 460 240 L 625 240" fill="none" />

      {/* パイプライン → 台帳 */}
      <circle r="4" fill="#34d399" opacity="0.9">
        <animateMotion dur="2.6s" repeatCount="indefinite" begin="1.5s">
          <mpath href="#path-r3" />
        </animateMotion>
        <animate attributeName="opacity" values="0;0.9;0" dur="2.6s" repeatCount="indefinite" begin="1.5s" />
      </circle>
      <path id="path-r3" d="M 460 250 C 530 270 560 350 625 350" fill="none" />

      {/* 紫苑 → パイプライン */}
      <circle r="3.5" fill="#e879f9" opacity="0.9">
        <animateMotion dur="1.8s" repeatCount="indefinite" begin="0s">
          <mpath href="#path-shion" />
        </animateMotion>
        <animate attributeName="opacity" values="0;1;0" dur="1.8s" repeatCount="indefinite" begin="0s" />
      </circle>
      <path id="path-shion" d="M 390 70 L 390 190" fill="none" />

    </svg>
  );
}
