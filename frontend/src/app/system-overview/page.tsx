"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Brain, Database, GitMerge, Zap, Activity, Shield, Eye, Code2, ExternalLink, Clock, RefreshCw, FileText, HardDrive, Newspaper, Search, BarChart2 } from "lucide-react";

type PipelineEvent = {
  time: string;
  label: string;
  desc: string;
  script: string;
  color: string;
  bg: string;
  border: string;
  iconColor: string;
  icon: React.ElementType;
  badge?: string;
  badgeColor?: string;
  repeat?: string;
};

type SeciStep = {
  num: string;
  title: string;
  desc: string;
  color: string;
  bg: string;
  border: string;
};

const seciSteps: SeciStep[] = [
  {
    num: "①",
    title: "対応",
    desc: "審査チャット・Obsidianメモ",
    color: "#60a5fa",
    bg: "rgba(30,58,138,0.35)",
    border: "rgba(96,165,250,0.5)",
  },
  {
    num: "②",
    title: "学び抽出",
    desc: "step1 / 04:00自動実行",
    color: "#a78bfa",
    bg: "rgba(88,28,135,0.35)",
    border: "rgba(167,139,250,0.5)",
  },
  {
    num: "③",
    title: "人間承認",
    desc: "step2 / AIが根拠を提示",
    color: "#fbbf24",
    bg: "rgba(120,53,15,0.35)",
    border: "rgba(251,191,36,0.5)",
  },
  {
    num: "④",
    title: "自動適用",
    desc: "step3 / コード自動修正",
    color: "#34d399",
    bg: "rgba(6,78,59,0.35)",
    border: "rgba(52,211,153,0.5)",
  },
  {
    num: "⑤",
    title: "助言注入",
    desc: "次の審査に自動反映",
    color: "#f472b6",
    bg: "rgba(131,24,67,0.35)",
    border: "rgba(244,114,182,0.5)",
  },
];

const pipelineEvents: PipelineEvent[] = [
  {
    time: "03:00",
    label: "RAG日次見直し",
    desc: "ChromaDB インデックス再構築・検索精度テスト・メタデータ統計・改善候補TOP3",
    script: "morning_rag_review_v2.py / reindex_obsidian.py --full",
    color: "#818cf8",
    bg: "rgba(49,46,129,0.25)",
    border: "rgba(99,102,241,0.35)",
    iconColor: "#818cf8",
    icon: RefreshCw,
  },
  {
    time: "03:30",
    label: "AURION 深夜自律同期",
    desc: "Obsidian ノート同期・SQLite DB 監査・ナイトリーステータス記録・lease-wiki-vault 更新",
    script: "aurion_core_daily.py --mode midnight",
    color: "#a78bfa",
    bg: "rgba(88,28,135,0.25)",
    border: "rgba(139,92,246,0.35)",
    iconColor: "#a78bfa",
    icon: Brain,
    badge: "AURION",
    badgeColor: "rgba(139,92,246,0.2)",
  },
  {
    time: "04:00",
    label: "日次改善パイプライン",
    desc: "Obsidian改善インデックス抽出 → auto-improvement-pipeline → batch_apply → Codexキュー生成 → 再帰的自己改善レポート",
    script: "run_daily_improvement_pipeline.sh（core + post）",
    color: "#34d399",
    bg: "rgba(6,78,59,0.25)",
    border: "rgba(52,211,153,0.35)",
    iconColor: "#34d399",
    icon: GitMerge,
    badge: "MAIN",
    badgeColor: "rgba(52,211,153,0.2)",
  },
  {
    time: "04:00",
    label: "Obsidian バックアップ",
    desc: "Obsidian Vault を iCloud に 14世代保持でバックアップ",
    script: "run_obsidian_backup.sh --keep 14",
    color: "#60a5fa",
    bg: "rgba(30,58,138,0.2)",
    border: "rgba(96,165,250,0.3)",
    iconColor: "#60a5fa",
    icon: HardDrive,
  },
  {
    time: "04:30",
    label: "案件データバックアップ",
    desc: "lease_data.db を iCloud に 12世代保持で自動バックアップ",
    script: "backup_case_data.py --keep 12",
    color: "#38bdf8",
    bg: "rgba(12,74,110,0.2)",
    border: "rgba(56,189,248,0.3)",
    iconColor: "#38bdf8",
    icon: Database,
  },
  {
    time: "05:00",
    label: "週次システムヘルスチェック",
    desc: "全依存サービス疎通確認・バックアップ整合性・ログ容量・異常レポート生成",
    script: "check_system_health.py",
    color: "#fbbf24",
    bg: "rgba(120,53,15,0.2)",
    border: "rgba(251,191,36,0.3)",
    iconColor: "#fbbf24",
    icon: Shield,
    repeat: "毎週月曜",
  },
  {
    time: "06:00",
    label: "AURION 朝報告生成",
    desc: "前夜の処理サマリ・スコアリングドリフト・改善適用状況を lease-wiki-vault に朝報告として書き出し",
    script: "aurion_core_daily.py --mode morning-report",
    color: "#f472b6",
    bg: "rgba(131,24,67,0.2)",
    border: "rgba(244,114,182,0.3)",
    iconColor: "#f472b6",
    icon: FileText,
    badge: "AURION",
    badgeColor: "rgba(244,114,182,0.15)",
  },
  {
    time: "06:00",
    label: "業界ニュース収集",
    desc: "Google News RSS + METI/FSA/MLIT 公式フィード → Obsidian daily digest に書き出し（最大18件）",
    script: "run_lease_news_collection.sh --limit 18 --profile industry-watch",
    color: "#4ade80",
    bg: "rgba(20,83,45,0.2)",
    border: "rgba(74,222,128,0.3)",
    iconColor: "#4ade80",
    icon: Newspaper,
  },
  {
    time: "06:00",
    label: "業界ナレッジフィード",
    desc: "業種別最新知識をClaude APIで生成・Obsidian Vault にインジェクト",
    script: "daily_knowledge_feed.py",
    color: "#86efac",
    bg: "rgba(20,83,45,0.18)",
    border: "rgba(134,239,172,0.28)",
    iconColor: "#86efac",
    icon: Zap,
  },
  {
    time: "06:10",
    label: "リース審査自動調査",
    desc: "未解決審査ロジックを自動リサーチ → 知識ベースへ反映・Obsidian に保存",
    script: "auto_research_lease_judgment.py",
    color: "#22d3ee",
    bg: "rgba(8,51,68,0.22)",
    border: "rgba(34,211,238,0.3)",
    iconColor: "#22d3ee",
    icon: Search,
  },
  {
    time: "06:30",
    label: "月次プロンプトFBレポート",
    desc: "プロンプト品質フィードバック集計・改善提案を Obsidian に月次レポートとして出力",
    script: "run_monthly_prompt_feedback_report.py --obsidian",
    color: "#fb923c",
    bg: "rgba(124,45,18,0.2)",
    border: "rgba(251,146,60,0.3)",
    iconColor: "#fb923c",
    icon: BarChart2,
    repeat: "毎月1日",
  },
  {
    time: "22:00",
    label: "Obsidian 日次ログ書き出し",
    desc: "その日のパイプライン実行ログ・改善適用結果を Obsidian Vault に dispatch",
    script: "dispatch_log_to_obsidian.py",
    color: "#94a3b8",
    bg: "rgba(15,23,42,0.4)",
    border: "rgba(148,163,184,0.25)",
    iconColor: "#94a3b8",
    icon: FileText,
  },
];

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

        {/* ── 暗黙知サイクル（SECIモデル） ── */}
        <section>
          <div className="text-center mb-6">
            <p className="text-xs font-bold tracking-[0.25em] uppercase text-slate-500 mb-1">Knowledge Flywheel</p>
            <h2 className="text-2xl font-black text-white tracking-tight">使うほど、賢くなる。</h2>
            <p className="text-sm text-slate-400 mt-2">暗黙知サイクル — 審査現場の経験が自動でシステムに還流する</p>
          </div>
          <div
            className="rounded-3xl border border-slate-800 p-6"
            style={{ background: "rgba(8,12,28,0.9)" }}
          >
            <SeciCycleDiagram />
          </div>
        </section>

        {/* ── 自動改善パイプライン 24hタイムライン ── */}
        <section>
          <div className="text-center mb-6">
            <p className="text-xs font-bold tracking-[0.25em] uppercase text-slate-500 mb-1">Autonomous Pipeline</p>
            <h2 className="text-lg font-black text-slate-300 tracking-wide uppercase">自動改善パイプライン — 24h タイムライン</h2>
            <p className="text-xs text-slate-500 mt-2">launchd で毎日自律実行。人間の操作なしに知識取込・分析・自己改善が回る。</p>
          </div>

          <div className="relative">
            {/* 縦ライン */}
            <div
              className="absolute left-[72px] top-0 bottom-0 w-px"
              style={{ background: "linear-gradient(180deg, rgba(99,102,241,0.0) 0%, rgba(99,102,241,0.4) 8%, rgba(99,102,241,0.4) 92%, rgba(99,102,241,0.0) 100%)" }}
            />

            <div className="space-y-3">
              {pipelineEvents.map((ev, i) => {
                const Icon = ev.icon;
                return (
                  <div key={i} className="flex items-start gap-4 group">
                    {/* 時刻バッジ */}
                    <div className="flex-shrink-0 w-[64px] text-right">
                      <span
                        className="text-xs font-black tabular-nums"
                        style={{ color: ev.color }}
                      >
                        {ev.time}
                      </span>
                    </div>

                    {/* タイムラインドット */}
                    <div className="flex-shrink-0 relative z-10 mt-1">
                      <div
                        className="w-3 h-3 rounded-full border-2"
                        style={{
                          background: ev.bg,
                          borderColor: ev.color,
                          boxShadow: `0 0 8px ${ev.color}60`,
                        }}
                      />
                    </div>

                    {/* イベントカード */}
                    <div
                      className="flex-1 rounded-xl border p-3 mb-1 transition-all duration-200 group-hover:scale-[1.005]"
                      style={{
                        background: ev.bg,
                        borderColor: ev.border,
                      }}
                    >
                      <div className="flex items-start justify-between gap-2 flex-wrap">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: ev.iconColor }} />
                          <span className="text-xs font-black text-white">{ev.label}</span>
                          {ev.badge && (
                            <span
                              className="text-[9px] font-black px-1.5 py-0.5 rounded-full border"
                              style={{
                                background: ev.badgeColor ?? "rgba(99,102,241,0.2)",
                                borderColor: ev.color,
                                color: ev.color,
                              }}
                            >
                              {ev.badge}
                            </span>
                          )}
                          {ev.repeat && (
                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                              {ev.repeat}
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed mt-1.5">{ev.desc}</p>
                      <p
                        className="text-[10px] font-mono mt-1.5 opacity-50"
                        style={{ color: ev.iconColor }}
                      >
                        {ev.script}
                      </p>
                    </div>
                  </div>
                );
              })}

              {/* 常時稼働: Vault Watcher */}
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-[64px] text-right">
                  <span className="text-xs font-black" style={{ color: "#f59e0b" }}>常時</span>
                </div>
                <div className="flex-shrink-0 relative z-10 mt-1">
                  <div
                    className="w-3 h-3 rounded-full border-2"
                    style={{
                      background: "rgba(120,53,15,0.3)",
                      borderColor: "#f59e0b",
                      boxShadow: "0 0 8px #f59e0b60",
                      animation: "pulse-glow 2s ease-in-out infinite",
                    }}
                  />
                </div>
                <div
                  className="flex-1 rounded-xl border p-3"
                  style={{
                    background: "rgba(120,53,15,0.15)",
                    borderColor: "rgba(245,158,11,0.3)",
                  }}
                >
                  <div className="flex items-center gap-2">
                    <Eye className="w-3.5 h-3.5" style={{ color: "#f59e0b" }} />
                    <span className="text-xs font-black text-white">Vault Watcher</span>
                    <span className="text-[9px] font-black px-1.5 py-0.5 rounded-full border" style={{ background: "rgba(245,158,11,0.15)", borderColor: "#f59e0b", color: "#f59e0b" }}>
                      LIVE
                    </span>
                    <Clock className="w-3 h-3 ml-1" style={{ color: "#f59e0b", opacity: 0.6 }} />
                    <span className="text-[10px] text-slate-500">60秒間隔</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed mt-1.5">
                    Obsidian Vault の変更を60秒間隔で検知 → ChromaDB RAG インデックスをリアルタイム更新
                  </p>
                  <p className="text-[10px] font-mono mt-1.5 opacity-50" style={{ color: "#f59e0b" }}>
                    vault_watcher.py（KeepAlive: true）
                  </p>
                </div>
              </div>

            </div>
          </div>

          {/* 凡例 */}
          <div className="mt-6 flex flex-wrap gap-3 justify-center">
            {[
              { label: "深夜 (03:xx)", color: "#a78bfa" },
              { label: "早朝改善パイプライン (04:xx)", color: "#34d399" },
              { label: "朝 知識収集 (06:xx)", color: "#4ade80" },
              { label: "夜 ログ配信 (22:xx)", color: "#94a3b8" },
              { label: "常時監視", color: "#f59e0b" },
            ].map((leg) => (
              <div key={leg.label} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ background: leg.color }} />
                <span className="text-[10px] text-slate-500">{leg.label}</span>
              </div>
            ))}
          </div>
        </section>

      </div>
    </div>
  );
}

function SeciCycleDiagram() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % seciSteps.length);
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="space-y-6">
      <div className="overflow-x-auto pb-2">
        <div className="flex items-stretch min-w-[600px]">
          {seciSteps.map((step, i) => {
            const isActive = i === activeIndex;
            return (
              <React.Fragment key={i}>
                <div
                  className="flex-1 rounded-2xl border p-4 text-center flex flex-col items-center justify-center gap-1.5 transition-all duration-500"
                  style={{
                    background: isActive ? step.bg : "rgba(15,23,42,0.4)",
                    borderColor: isActive ? step.border : "rgba(51,65,85,0.4)",
                    boxShadow: isActive ? `0 0 20px ${step.color}35` : "none",
                    transform: isActive ? "scale(1.05)" : "scale(1)",
                    minHeight: "108px",
                  }}
                >
                  <span
                    className="text-2xl font-black transition-colors duration-500"
                    style={{ color: isActive ? step.color : "#334155" }}
                  >
                    {step.num}
                  </span>
                  <span className="text-xs font-black text-white leading-tight">{step.title}</span>
                  <span className="text-[10px] text-slate-400 leading-relaxed">{step.desc}</span>
                </div>
                {i < seciSteps.length - 1 && (
                  <div className="flex items-center flex-shrink-0 px-1.5">
                    <svg width="18" height="12" viewBox="0 0 18 12" fill="none">
                      <path
                        d="M0 6 L13 6 M8.5 2 L13 6 L8.5 10"
                        stroke={isActive ? step.color : "#334155"}
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div
          className="flex-1 h-px rounded"
          style={{ background: "linear-gradient(90deg, rgba(244,114,182,0.4), rgba(96,165,250,0.4))" }}
        />
        <div className="flex items-center gap-1.5">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M10 6A4 4 0 1 1 6 2" stroke="#60a5fa" strokeWidth="1.3" strokeLinecap="round" />
            <path d="M5.5 0 L8 2 L5.5 4" fill="none" stroke="#60a5fa" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-[10px] font-semibold text-slate-500">⑤ → ① に自動帰還・サイクル継続</span>
        </div>
        <div
          className="flex-1 h-px rounded"
          style={{ background: "linear-gradient(90deg, rgba(96,165,250,0.4), rgba(244,114,182,0.4))" }}
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { value: "137件", label: "改善適用件数", color: "#60a5fa", bg: "rgba(30,58,138,0.2)" },
          { value: "+8.2pt", label: "累計精度向上", color: "#34d399", bg: "rgba(6,78,59,0.2)" },
          { value: "78%", label: "自動適用率", color: "#a78bfa", bg: "rgba(88,28,135,0.2)" },
        ].map((stat) => (
          <div
            key={stat.label}
            className="rounded-xl border border-slate-700/40 p-4 text-center"
            style={{ background: stat.bg }}
          >
            <p className="text-2xl font-black" style={{ color: stat.color }}>{stat.value}</p>
            <p className="text-[11px] text-slate-400 font-semibold mt-1">{stat.label}</p>
          </div>
        ))}
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
