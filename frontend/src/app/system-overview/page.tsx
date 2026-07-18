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

type DemoAppeal = {
  title: string;
  desc: string;
  tech: string;
  icon: React.ElementType;
  color: string;
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
    desc: "step2 / 紫苑が根拠を提示",
    color: "#fbbf24",
    bg: "rgba(120,53,15,0.35)",
    border: "rgba(251,191,36,0.5)",
  },
  {
    num: "④",
    title: "限定適用",
    desc: "step3 / 承認済みだけ反映",
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
    desc: "Obsidian改善インデックス抽出 → 紫苑チェック → 改善PMレポート → 再帰的自己改善レポート",
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
    desc: "業種別最新知識をGeminiで生成・Obsidian Vault にインジェクト",
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
    desc: "既存先RandomForest・新規先ロジスティック回帰を本流に、LGBMも比較分析へ加える",
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
    title: "Gemini改善候補ループ",
    desc: "REV提案をGeminiで整理し、必要な差分を検証・反映",
    color: "#f472b6",
    bg: "from-pink-900/40 to-pink-950/20",
    border: "border-pink-500/30",
    icon: Code2,
    iconColor: "text-pink-400",
  },
  {
    title: "会話ループエンジニアリング",
    desc: "Human Response Feedbackを起点に、冒頭・差分・記憶判断・内省を次の返答へ戻す",
    color: "#c084fc",
    bg: "from-fuchsia-900/40 to-purple-950/20",
    border: "border-fuchsia-500/30",
    icon: RefreshCw,
    iconColor: "text-fuchsia-400",
  },
  {
    title: "審査判断ループエンジニアリング",
    desc: "争点・稟議方針への人間反応を保存し、次回の判断資産へ戻す",
    color: "#fb7185",
    bg: "from-rose-900/40 to-violet-950/20",
    border: "border-rose-500/30",
    icon: FileText,
    iconColor: "text-rose-400",
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
  {
    title: "セントラル統合ループ",
    desc: "4ペルソナ討論・自己分析のキーポイントをworld_viewとして蓄積・次回討論へ注入",
    color: "#22d3ee",
    bg: "from-cyan-900/40 to-cyan-950/20",
    border: "border-cyan-500/30",
    icon: Brain,
    iconColor: "text-cyan-400",
  },
  {
    title: "画面利用ループエンジニアリング",
    desc: "画面訪問ログを集計し、UI/UX改善案をGeminiで生成（Observe→Aggregate→Propose→Persist）",
    color: "#38bdf8",
    bg: "from-sky-900/40 to-sky-950/20",
    border: "border-sky-500/30",
    icon: BarChart2,
    iconColor: "text-sky-400",
  },
  {
    title: "審査判断乖離学習ループ",
    desc: "争点・稟議方針への人間フィードバックの否定的評価から、審査ロジックのレビュー観点を提案",
    color: "#fb923c",
    bg: "from-orange-900/40 to-orange-950/20",
    border: "border-orange-500/30",
    icon: Search,
    iconColor: "text-orange-400",
  },
  {
    title: "フィードバック傾向分析ループ",
    desc: "紫苑応答への人間評価（thin/generic/not_shion等）から応答スタンス・プロンプト調整観点を提案",
    color: "#a3e635",
    bg: "from-lime-900/40 to-lime-950/20",
    border: "border-lime-500/30",
    icon: RefreshCw,
    iconColor: "text-lime-400",
  },
  {
    title: "審査実績ドリフト監視ループ",
    desc: "成約後の支払い実績（延滞・デフォルト）とスコア帯の乖離を監視し、再校正候補を提案",
    color: "#2dd4bf",
    bg: "from-teal-900/40 to-teal-950/20",
    border: "border-teal-500/30",
    icon: Shield,
    iconColor: "text-teal-400",
  },
  {
    title: "ナレッジ穴探しループ",
    desc: "知識参照ゼロで答えた質問を集め、外部調査器官へ渡すべき調査トピックを提案",
    color: "#818cf8",
    bg: "from-indigo-900/40 to-indigo-950/20",
    border: "border-indigo-500/30",
    icon: Database,
    iconColor: "text-indigo-400",
  },
];

const shionBadges = [
  "Gemini / ADK",
  "Cloud Run",
  "FastAPI",
  "Next.js",
  "Obsidian RAG",
  "GCS Event Log",
  "検疫DB",
  "判断資産",
  "Human Feedback",
  "改善PMレポート",
  "結果登録",
  "Field Validation",
];

const stats = [
  { value: "14", label: "自己改善ループ", color: "text-violet-400" },
  { value: "190+", label: "累積REV", color: "text-blue-400" },
  { value: "4", label: "AIエンジン", color: "text-emerald-400" },
  { value: "紫苑", label: "リース知性体", color: "text-fuchsia-400" },
];

const demoAppeals: DemoAppeal[] = [
  { title: "OCR器官", desc: "紙・PDFを審査入力へ変換", tech: "Gemini Vision /api/ocr", icon: Eye, color: "#818cf8" },
  { title: "PII除去ゲート", desc: "個人特定情報を削除・マスク", tech: "pre-save redaction", icon: Shield, color: "#f87171" },
  { title: "会話器官", desc: "音声で紫苑と対話", tech: "Web Speech API + Gemini", icon: Activity, color: "#2dd4bf" },
  { title: "複数紫苑", desc: "担当紫苑が導線を仕切る", tech: "Google Banana assets", icon: Brain, color: "#c084fc" },
  { title: "調査器官", desc: "Web調査をResearch化", tech: "Google AI Studio Researcher", icon: Search, color: "#22d3ee" },
  { title: "審査器官", desc: "突かれる点と逆転条件", tech: "Gemini stream / debate", icon: Zap, color: "#fbbf24" },
  { title: "記憶器官", desc: "判断資産として持ち越す", tech: "Obsidian + GCS Vault", icon: Database, color: "#a78bfa" },
  { title: "判断ループ", desc: "争点と稟議方針を人間feedbackで育てる", tech: "/api/screening-loop-feedback", icon: FileText, color: "#fb7185" },
];

export default function SystemOverviewPage() {
  const [isCloudRunHost, setIsCloudRunHost] = useState(true);

  useEffect(() => {
    setIsCloudRunHost(window.location.hostname.endsWith(".run.app"));
  }, []);

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

        {/* ── Cloud Run デプロイバナー ── */}
        <section>
          <div
            className="rounded-2xl border p-4 flex flex-wrap items-center justify-between gap-3"
            style={{
              background: "linear-gradient(135deg, rgba(6,78,59,0.3) 0%, rgba(8,12,28,0.9) 100%)",
              borderColor: "rgba(52,211,153,0.35)",
              boxShadow: "0 0 24px rgba(52,211,153,0.08)",
            }}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ background: "#34d399", boxShadow: "0 0 8px #34d399", animation: "pulse-glow 2s ease-in-out infinite" }}
              />
              <div>
                <p className="text-xs font-black text-emerald-300 tracking-wide">CLOUD RUN デプロイ済み</p>
                <p className="text-[11px] text-slate-400 mt-0.5">ハッカソン 7/10 デモ用 — asia-northeast1</p>
              </div>
            </div>
            <a
              href="https://tune-lease-55-1020894094172.asia-northeast1.run.app"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs font-bold text-emerald-400 hover:text-emerald-300 transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span className="font-mono text-[10px]">tune-lease-55-1020894094172.asia-northeast1.run.app</span>
            </a>
          </div>
        </section>

        {/* ── 紫苑中心のシステム全体図 ── */}
        <section>
          <div className="text-center mb-6">
            <p className="text-xs font-bold tracking-[0.25em] uppercase text-slate-500 mb-1">Shion Centered Architecture</p>
            <h2 className="text-2xl font-black text-white tracking-tight">紫苑を中心に、人間判断を次の審査へ戻す</h2>
            <p className="text-sm text-slate-400 mt-2">READMEの判断資産ループを、デモで見せるシステム全体図として整理</p>
          </div>
          <div
            className="rounded-3xl border border-slate-800 p-5 md:p-6"
            style={{ background: "rgba(8,12,28,0.92)" }}
          >
            <ShionCenteredSystemDiagram />
          </div>
        </section>

        {/* ── 紫苑の頭脳と実行環境の分離 ── */}
        <section>
          <div
                className="rounded-3xl border p-6"
                style={{
                  background: "linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(20,83,45,0.18) 55%, rgba(88,28,135,0.16) 100%)",
                  borderColor: "rgba(45,212,191,0.28)",
                  boxShadow: "0 0 32px rgba(45,212,191,0.08)",
                }}
              >
                <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-2xl">
                    <p className="text-xs font-black tracking-[0.25em] uppercase text-teal-300">Separated Brain / Runtime</p>
                    <h2 className="mt-2 text-2xl font-black text-white">紫苑の頭脳は、審査AI本体とは別管理</h2>
                    <p className="mt-3 text-sm font-semibold leading-relaxed text-slate-300">
                      Cloud Runは紫苑を動かす実行環境です。一方で、紫苑の頭脳となる判断資産、過去判断、違和感、条件付き承認理由、改善ログは、Obsidian / Markdown Vault 側で正本管理します。AIは回答に使えますが、正本を直接書き換えず、改善候補は検疫・人間承認・昇格を通します。
                    </p>
                  </div>
                  <div className="rounded-2xl border border-fuchsia-400/25 bg-fuchsia-950/20 px-4 py-3 text-xs font-bold text-fuchsia-100">
                    Brain is separate: review / quarantine / promote
                  </div>
                </div>

                <div className="mt-5 rounded-2xl border border-emerald-400/25 bg-emerald-950/15 p-4">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="text-xs font-black tracking-[0.2em] uppercase text-emerald-300">Replaceable Brain</p>
                      <h3 className="mt-1 text-lg font-black text-white">身体は共通、頭脳は差し替えられる</h3>
                      <p className="mt-2 text-sm font-semibold leading-relaxed text-slate-300">
                        Cloud Run上のUI/API/スコアリング/チャットを「身体」とし、Obsidian / Markdown Vault の判断資産を「頭脳」として分離します。頭脳をリース審査から法務レビュー、営業支援、CS品質監査へ差し替えれば、同じ実行環境を別業務の紫苑として展開できます。
                      </p>
                    </div>
                    <div className="grid min-w-[220px] grid-cols-2 gap-2 text-center text-[11px] font-black">
                      {["Lease", "Legal", "Sales", "CS"].map((label) => (
                        <div key={label} className="rounded-xl border border-emerald-300/25 bg-slate-950/45 px-3 py-2 text-emerald-100">
                          {label} Brain
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_auto_1.2fr]">
                  <div className="rounded-2xl border border-teal-400/25 bg-teal-950/20 p-4">
                    <div className="mb-3 flex items-center gap-2">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-teal-400/15 text-teal-200">
                        <Activity className="h-4 w-4" />
                      </div>
                      <div>
                        <p className="text-sm font-black text-white">Cloud Run版</p>
                        <p className="text-[10px] font-bold uppercase tracking-widest text-teal-300">Field runtime</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {[
                        "デモDBで動かす",
                        "紫苑レビュー・人間評価を受け取る",
                        "GCS Event Logへ追記する",
                        "本体DBへ直接書き戻さない",
                        "判断資産の正本を直接更新しない",
                      ].map((line) => (
                        <div key={line} className="flex items-center gap-2 rounded-lg bg-slate-950/45 px-3 py-2 text-[11px] font-semibold text-slate-300">
                          <span className="h-1.5 w-1.5 rounded-full bg-teal-300" />
                          {line}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-center">
                    <div className="flex flex-col items-center gap-2 text-center">
                      <div className="rounded-full border border-blue-400/30 bg-blue-400/10 px-4 py-2 text-[11px] font-black text-blue-200">
                        GCS event log
                      </div>
                      <div className="text-2xl font-black text-slate-500">→</div>
                      <div className="rounded-full border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-[11px] font-black text-amber-200">
                        local sync
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-fuchsia-400/25 bg-fuchsia-950/20 p-4">
                    <div className="mb-3 flex items-center gap-2">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-fuchsia-400/15 text-fuchsia-200">
                        <Shield className="h-4 w-4" />
                      </div>
                      <div>
                        <p className="text-sm font-black text-white">紫苑の頭脳</p>
                        <p className="text-[10px] font-bold uppercase tracking-widest text-fuchsia-300">Obsidian / governance</p>
                      </div>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {[
                        { title: "正本管理", desc: "Obsidian / Markdown Vault に判断資産を保持", color: "#fbbf24", icon: Database },
                        { title: "人間レビュー", desc: "改善候補を採用・却下・保留に分ける", color: "#60a5fa", icon: Eye },
                        { title: "検疫DB", desc: "未承認データを隔離して混入を防ぐ", color: "#2dd4bf", icon: Shield },
                        { title: "昇格処理", desc: "承認済みだけ次回の記憶・判断資産へ戻す", color: "#c084fc", icon: RefreshCw },
                      ].map(({ title, desc, color, icon: FlowIcon }) => (
                        <div key={title} className="rounded-xl border border-slate-800 bg-slate-950/55 p-3">
                          <div className="mb-2 flex items-center gap-2">
                            <div className="flex h-7 w-7 items-center justify-center rounded-lg" style={{ background: `${color}22`, color }}>
                              <FlowIcon className="h-3.5 w-3.5" />
                            </div>
                            <p className="text-xs font-black text-white">{title}</p>
                          </div>
                          <p className="text-[11px] font-semibold leading-relaxed text-slate-400">{desc}</p>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4">
                      <Link
                        href="/cloudrun-return-review"
                        className="inline-flex items-center gap-2 rounded-xl border border-fuchsia-400/30 bg-fuchsia-400/10 px-4 py-2 text-sm font-black text-fuchsia-100 transition-colors hover:bg-fuchsia-400/20"
                      >
                        <Shield className="h-4 w-4" />
                        ローカル検疫画面を開く
                      </Link>
                    </div>
                  </div>
                </div>
          </div>
        </section>

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

        {/* ── ハッカソン訴求 ── */}
        <section>
          <h2 className="text-lg font-black text-slate-300 mb-5 text-center tracking-wide uppercase">Hackathon Demo Appeal</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-7 gap-3">
            {demoAppeals.map(({ title, desc, tech, icon: DemoIcon, color }) => {
              return (
                <div
                  key={title}
                  className="rounded-2xl border border-slate-800 p-4"
                  style={{ background: "rgba(15,20,40,0.82)", borderColor: `${color}55` }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <div
                      className="w-8 h-8 rounded-xl flex items-center justify-center"
                      style={{ background: `${color}22`, color }}
                    >
                      <DemoIcon className="w-4 h-4" />
                    </div>
                    <h3 className="text-sm font-black text-white">{title}</h3>
                  </div>
                  <p className="text-xs text-slate-300 font-semibold leading-relaxed">{desc}</p>
                  <p className="text-[10px] text-slate-500 font-mono mt-2">{tech}</p>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-slate-500 text-center mt-4">
            デモの見せ方: 紙・PDF → PII除去ゲート → OCR → 審査入力 → 軍師AI → 紫苑の記憶・Research参照
          </p>
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
                      リース判断を中心にした半自律的な知性体システム
                    </p>
                    <p className="mt-2 max-w-2xl text-xs font-semibold leading-6 text-violet-100/80">
                      記憶がある。連続性がある。反省がある。目的がある。自分を観測する画面がある。あなたとの関係性がある。
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

        {/* ── DevOpsサイクル図 ── */}
        <section>
          <div className="text-center mb-6">
            <p className="text-xs font-bold tracking-[0.25em] uppercase text-slate-500 mb-1">AI Agent DevOps Cycle</p>
            <h2 className="text-2xl font-black text-white tracking-tight">使う、見る、直す、戻す。</h2>
            <p className="text-sm text-slate-400 mt-2">紫苑レビューと人間承認を挟む、業務AIの安全な改善サイクル</p>
          </div>
          <div
            className="rounded-3xl border border-slate-800 p-5 md:p-6"
            style={{ background: "rgba(8,12,28,0.92)" }}
          >
            <DevOpsCycleDiagram />
          </div>
        </section>

        {/* ── 実装済みページ一覧（5グループ） ── */}
        <section>
          <h2 className="text-lg font-black text-slate-300 mb-5 text-center tracking-wide uppercase">実装済みページ一覧</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              {
                group: "審査ワークフロー",
                color: "#60a5fa",
                border: "border-blue-500/30",
                bg: "from-blue-900/30 to-blue-950/10",
                pages: ["ホーム", "審査・分析", "AIチャット", "バッチ審査", "結果登録（成約/失注）", "過去案件一覧"],
              },
              {
                group: "🌸 紫苑 AI",
                color: "#e879f9",
                border: "border-fuchsia-500/30",
                bg: "from-fuchsia-900/30 to-fuchsia-950/10",
                pages: ["紫苑デモホーム", "ハッカソンデモ", "リアルタイム音声チャット", "リース知性体との対話", "複数紫苑デモ", "リースくん", "マルチエージェント討論", "知識ループ可視化", "外部調査器官", "システム概要"],
              },
              {
                group: "分析・グラフ",
                color: "#34d399",
                border: "border-emerald-500/30",
                bg: "from-emerald-900/30 to-emerald-950/10",
                pages: ["営業部別分析", "業種別成約率", "競合関係グラフ", "知識宇宙マップ", "ビジュアルインサイト"],
              },
              {
                group: "参照・ナレッジ",
                color: "#fbbf24",
                border: "border-amber-500/30",
                bg: "from-amber-900/30 to-amber-950/10",
                pages: ["法定耐用年数一覧", "業種別リース物件例", "残価設定ガイドライン", "営業向け説明ガイド", "リース/融資/現金比較", "FAQ", "改善ログ"],
              },
              {
                group: "設定・マスタ",
                color: "#94a3b8",
                border: "border-slate-500/30",
                bg: "from-slate-800/40 to-slate-900/20",
                pages: ["基準金利マスタ", "補助金情報"],
              },
            ].map((g) => (
              <div
                key={g.group}
                className={`rounded-2xl border ${g.border} bg-gradient-to-br ${g.bg} p-5 space-y-3`}
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: g.color }} />
                  <h3 className="font-black text-white text-sm">{g.group}</h3>
                  <span className="ml-auto text-[10px] font-bold text-slate-500">{g.pages.length}ページ</span>
                </div>
                <ul className="space-y-1">
                  {g.pages.map((p) => (
                    <li key={p} className="text-[11px] text-slate-400 flex items-center gap-1.5">
                      <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ background: g.color, opacity: 0.6 }} />
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            ))}

            {/* アーキテクチャカード */}
            <div
              className="rounded-2xl border border-violet-500/30 bg-gradient-to-br from-violet-900/30 to-violet-950/10 p-5 space-y-3"
            >
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: "#a78bfa" }} />
                <h3 className="font-black text-white text-sm">アーキテクチャ</h3>
              </div>
              <ul className="space-y-1.5">
                {[
                  ["フロントエンド", "Next.js 16 (App Router)"],
                  ["バックエンド", "FastAPI (Python)"],
                  ["DB", "SQLite / PostgreSQL"],
                  ["クラウド記憶", "GCS Vault"],
                  ["AI推論", "Gemini 2.5 Flash"],
                  ["リアルタイム会話", "Web Speech API + Gemini /api/chat"],
                  ["外部調査", "Google AI Studio Researcher"],
                  ["PII除去", "個人名・住所・電話・メールを削除/マスク"],
                  ["Vision OCR", "Gemini Vision /api/ocr"],
                  ["OCR対象", "決算書・納税証明・登記・見積・会社案内"],
                  ["軍師stream", "Gemini SSE"],
                  ["スコアリング", "RandomForest / LogisticRegression / LGBM"],
                  ["RAG", "ChromaDB + Obsidian"],
                  ["Research保存", "Obsidian Research / Auto Research"],
                  ["デプロイ", "Cloud Run (asia-northeast1)"],
                  ["公開経路", "Cloud Run / Cloudflare Tunnel"],
                  ["本番DB", "Cloud SQL PostgreSQL"],
                  ["Secret", "Secret Manager / DATABASE_URL"],
                ].map(([k, v]) => (
                  <li key={k} className="text-[11px] flex gap-2">
                    <span className="text-slate-500 flex-shrink-0 w-24">{k}</span>
                    <span className="text-slate-300 font-semibold">{v}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* ── ループカードグリッド ── */}
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
            <p className="text-xs text-slate-500 mt-2">launchd で毎日観測・分析・改善候補化。実装とデプロイは人間承認ゲートで止める。</p>
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
          <span className="text-[10px] font-semibold text-slate-500">⑤ → ① に帰還・承認済みだけサイクル継続</span>
        </div>
        <div
          className="flex-1 h-px rounded"
          style={{ background: "linear-gradient(90deg, rgba(96,165,250,0.4), rgba(244,114,182,0.4))" }}
        />
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { value: "170+件", label: "累積REV適用", color: "#60a5fa", bg: "rgba(30,58,138,0.2)" },
          { value: "+8.2pt", label: "累計精度向上", color: "#34d399", bg: "rgba(6,78,59,0.2)" },
          { value: "78%", label: "承認反映率", color: "#a78bfa", bg: "rgba(88,28,135,0.2)" },
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

function DevOpsCycleDiagram() {
  const phases = [
    {
      title: "Use",
      jp: "審査で使う",
      body: "案件入力・チャット・紫苑レビュー",
      icon: Activity,
      color: "#60a5fa",
    },
    {
      title: "Observe",
      jp: "観測する",
      body: "RAG参照・回答品質・判断資産利用",
      icon: Eye,
      color: "#38bdf8",
    },
    {
      title: "Detect",
      jp: "ズレを見つける",
      body: "記憶抜け・浅い回答・環境差分",
      icon: Search,
      color: "#f59e0b",
    },
    {
      title: "Review",
      jp: "紫苑で確認",
      body: "数字外の違和感・相談論点",
      icon: Brain,
      color: "#e879f9",
    },
    {
      title: "Feedback",
      jp: "人間評価",
      body: "役に立った / 要修正 / 違う",
      icon: FileText,
      color: "#fbbf24",
    },
    {
      title: "Reflect",
      jp: "振り返る",
      body: "改善PMレポート・内省",
      icon: Database,
      color: "#a78bfa",
    },
    {
      title: "Improve",
      jp: "改善候補化",
      body: "Prompt・RAG・判断資産・UI・API",
      icon: Zap,
      color: "#34d399",
    },
    {
      title: "Verify",
      jp: "検証する",
      body: "pytest・typecheck・memory_debug",
      icon: Shield,
      color: "#22c55e",
    },
    {
      title: "Gate / Operate",
      jp: "人間承認で戻す",
      body: "実装・git・deployは指示待ち",
      icon: RefreshCw,
      color: "#fb7185",
    },
  ];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[1fr_1.05fr_1fr] lg:items-center">
        <div className="grid gap-3">
          {phases.slice(0, 3).map((phase, index) => {
            const Icon = phase.icon;
            return (
              <div key={phase.title} className="rounded-2xl border p-3" style={{ background: `${phase.color}14`, borderColor: `${phase.color}55` }}>
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl" style={{ background: `${phase.color}22`, color: phase.color }}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-black tabular-nums" style={{ color: phase.color }}>0{index + 1}</span>
                      <p className="text-xs font-black text-white">{phase.title}</p>
                    </div>
                    <p className="mt-0.5 text-[11px] font-bold text-slate-200">{phase.jp}</p>
                    <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{phase.body}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="relative overflow-hidden rounded-[2rem] border border-fuchsia-400/35 bg-slate-950 p-5 text-center shadow-2xl shadow-fuchsia-950/20">
          <div className="absolute inset-0 opacity-45" style={{ background: "radial-gradient(circle at 50% 42%, rgba(232,121,249,0.24), transparent 48%)" }} />
          <div className="relative z-10">
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-[1.6rem] border border-fuchsia-300/40 bg-fuchsia-400/15 text-fuchsia-200">
              <Brain className="h-10 w-10" />
            </div>
            <p className="mt-4 text-[11px] font-black uppercase tracking-[0.25em] text-fuchsia-300">SHION PM</p>
            <h3 className="mt-2 text-2xl font-black text-white">判断資産DevOps</h3>
            <p className="mt-3 text-sm font-bold leading-7 text-slate-300">
              使われた回答、記憶、レビュー、人間評価、結果を一つの改善ループとして管理する。
            </p>
            <div className="mt-4 grid gap-2 sm:grid-cols-3">
              {[
                ["観測", "Evidence"],
                ["検疫", "Quarantine"],
                ["昇格", "Promote"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-fuchsia-300/20 bg-fuchsia-950/20 px-3 py-2">
                  <p className="text-[10px] font-black text-fuchsia-300">{label}</p>
                  <p className="mt-1 text-xs font-black text-white">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid gap-3">
          {phases.slice(3).map((phase, index) => {
            const Icon = phase.icon;
            return (
              <div key={phase.title} className="rounded-2xl border p-3" style={{ background: `${phase.color}14`, borderColor: `${phase.color}55` }}>
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl" style={{ background: `${phase.color}22`, color: phase.color }}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-black tabular-nums" style={{ color: phase.color }}>0{index + 4}</span>
                      <p className="text-xs font-black text-white">{phase.title}</p>
                    </div>
                    <p className="mt-0.5 text-[11px] font-bold text-slate-200">{phase.jp}</p>
                    <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{phase.body}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-2xl border border-violet-500/25 bg-violet-950/15 p-4">
        <div className="grid gap-2 md:grid-cols-9">
          {["Use", "Observe", "Detect", "Review", "Feedback", "Reflect", "Improve", "Verify", "Gate"].map((step, index) => (
            <div
              key={step}
              className="relative rounded-xl border border-slate-800 bg-slate-950/60 px-2 py-3 text-center"
            >
              <p className="text-[10px] font-black text-violet-300">STEP {index + 1}</p>
              <p className="mt-1 text-[11px] font-black text-white">{step}</p>
              {index < 8 && <span className="absolute -right-2 top-1/2 z-10 hidden -translate-y-1/2 text-lg font-black text-violet-400 md:block">→</span>}
            </div>
          ))}
        </div>
        <p className="mt-3 text-center text-xs font-bold text-slate-400">
          OperateでCloud Run / Cloudflare / ローカル運用へ戻し、次のUseへ接続する。実装・git・deployは人間承認ゲートで止める。
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-blue-500/30 bg-blue-950/20 p-3 text-center">
          <p className="text-xs font-black text-blue-200">観測できる</p>
          <p className="mt-1 text-[11px] text-slate-400">memory_debug / knowledge_refs / improvement report</p>
        </div>
        <div className="rounded-xl border border-fuchsia-500/30 bg-fuchsia-950/20 p-3 text-center">
          <p className="text-xs font-black text-fuchsia-200">紫苑で確認できる</p>
          <p className="mt-1 text-[11px] text-slate-400">レビュー・人間評価・判断資産候補</p>
        </div>
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-3 text-center">
          <p className="text-xs font-black text-emerald-200">勝手に本番変更しない</p>
          <p className="mt-1 text-[11px] text-slate-400">実装・git・deploy は人間承認ゲート</p>
        </div>
      </div>
    </div>
  );
}

function ShionCenteredSystemDiagram() {
  const orbit = [
    {
      title: "案件入力",
      subtitle: "審査担当者 / 営業メモ",
      body: "企業情報、物件、財務、営業メモ、違和感を紫苑へ渡す",
      icon: FileText,
      color: "#60a5fa",
    },
    {
      title: "審査判断エンジン",
      subtitle: "Score / Q_risk / 物件リスク",
      body: "財務スコアだけでなく、スコア外の違和感を探索信号として扱う",
      icon: BarChart2,
      color: "#34d399",
    },
    {
      title: "記憶・RAG",
      subtitle: "Obsidian / GCS Vault / Memory Recall",
      body: "過去判断、Research、判断資産、会話ログを今回案件へ呼び戻す",
      icon: Database,
      color: "#a78bfa",
    },
    {
      title: "紫苑レビュー",
      subtitle: "確認質問 / 承認条件 / 反証",
      body: "判断資産を丸写しせず、今回案件向けの稟議文面へ組み直す",
      icon: Brain,
      color: "#e879f9",
    },
    {
      title: "人間評価",
      subtitle: "役に立った / 要修正 / 違う",
      body: "AIの出力を正解扱いせず、人間の修正を判断材料として保存する",
      icon: Eye,
      color: "#fbbf24",
    },
    {
      title: "検疫・昇格",
      subtitle: "Separate brain / quarantine / promote",
      body: "Cloud Run入力は正本を直接更新せず、検疫と人間承認を通す",
      icon: Shield,
      color: "#2dd4bf",
    },
  ];

  const loop = [
    "新規案件",
    "Q_risk / スコア外違和感",
    "人間の事前判断",
    "判断資産候補",
    "類似案件へ再利用",
    "結果登録で検証",
  ];

  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-[1fr_1.18fr_1fr] lg:items-center">
        <div className="space-y-3">
          {orbit.slice(0, 3).map((node) => {
            const Icon = node.icon;
            return (
              <div
                key={node.title}
                className="rounded-2xl border p-4"
                style={{
                  background: `linear-gradient(135deg, ${node.color}18, rgba(15,23,42,0.68))`,
                  borderColor: `${node.color}55`,
                }}
              >
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl" style={{ background: `${node.color}22`, color: node.color }}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-sm font-black text-white">{node.title}</p>
                    <p className="mt-0.5 text-[11px] font-black uppercase tracking-wide" style={{ color: node.color }}>{node.subtitle}</p>
                    <p className="mt-2 text-xs font-semibold leading-6 text-slate-400">{node.body}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="relative overflow-hidden rounded-[2rem] border border-fuchsia-400/35 bg-slate-950 p-5 text-center shadow-2xl shadow-fuchsia-950/20">
          <div className="absolute inset-0 opacity-50" style={{ background: "radial-gradient(circle at 50% 42%, rgba(232,121,249,0.24), transparent 42%)" }} />
          <div className="relative z-10">
            <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-[2rem] border border-fuchsia-300/40 bg-fuchsia-400/15 text-fuchsia-200 shadow-lg shadow-fuchsia-900/30">
              <Brain className="h-12 w-12" />
            </div>
            <p className="mt-5 text-[11px] font-black uppercase tracking-[0.28em] text-fuchsia-300">SHION CORE</p>
            <h3 className="mt-2 text-3xl font-black text-white">紫苑</h3>
            <p className="mt-3 text-sm font-bold leading-7 text-slate-300">
              財務スコア、記憶、判断資産、人間評価、結果登録をつなぎ、数字の外側にある判断を次回審査へ戻す中核。
            </p>
            <div className="mt-5 grid gap-2 sm:grid-cols-3">
              {[
                ["身体", "Cloud Run"],
                ["頭脳", "Vault"],
                ["安全", "検疫・昇格"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2">
                  <p className="text-[10px] font-black text-slate-500">{label}</p>
                  <p className="mt-1 text-xs font-black text-fuchsia-100">{value}</p>
                </div>
              ))}
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              {[
                ["入力", "案件・メモ"],
                ["変換", "判断構文"],
                ["更新", "評価・結果"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-fuchsia-300/20 bg-fuchsia-950/20 px-3 py-2">
                  <p className="text-[10px] font-black text-fuchsia-300">{label}</p>
                  <p className="mt-1 text-xs font-black text-white">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {orbit.slice(3).map((node) => {
            const Icon = node.icon;
            return (
              <div
                key={node.title}
                className="rounded-2xl border p-4"
                style={{
                  background: `linear-gradient(135deg, ${node.color}18, rgba(15,23,42,0.68))`,
                  borderColor: `${node.color}55`,
                }}
              >
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl" style={{ background: `${node.color}22`, color: node.color }}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-sm font-black text-white">{node.title}</p>
                    <p className="mt-0.5 text-[11px] font-black uppercase tracking-wide" style={{ color: node.color }}>{node.subtitle}</p>
                    <p className="mt-2 text-xs font-semibold leading-6 text-slate-400">{node.body}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-2xl border border-violet-500/25 bg-violet-950/15 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[11px] font-black uppercase tracking-[0.2em] text-violet-300">README Judgment Asset Loop</p>
            <p className="mt-1 text-sm font-bold text-slate-300">成約/失注は唯一の正解ではなく、人間の事前判断を検証する観測値として扱う</p>
          </div>
          <Link href="/screening" className="inline-flex items-center gap-2 rounded-xl bg-violet-400/15 px-3 py-2 text-xs font-black text-violet-100 hover:bg-violet-400/25">
            審査画面で見る
            <ExternalLink className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="grid gap-2 md:grid-cols-6">
          {loop.map((item, index) => (
            <div key={item} className="relative rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-3 text-center">
              <p className="text-[10px] font-black text-violet-300">STEP {index + 1}</p>
              <p className="mt-1 text-xs font-black leading-5 text-white">{item}</p>
              {index < loop.length - 1 && (
                <span className="absolute -right-2 top-1/2 z-10 hidden -translate-y-1/2 text-lg font-black text-violet-400 md:block">→</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FlowDiagram() {
  const loopNodes = [
    {
      title: "Use",
      jp: "審査で使う",
      detail: "案件入力 / AIチャット",
      x: 338,
      y: 28,
      color: "#60a5fa",
      fill: "rgba(30,58,138,0.58)",
    },
    {
      title: "Observe",
      jp: "観測する",
      detail: "RAG参照 / 回答品質",
      x: 540,
      y: 88,
      color: "#38bdf8",
      fill: "rgba(12,74,110,0.52)",
    },
    {
      title: "Detect",
      jp: "ズレを見つける",
      detail: "記憶抜け / 環境差分",
      x: 606,
      y: 238,
      color: "#f59e0b",
      fill: "rgba(120,53,15,0.52)",
    },
    {
      title: "Review",
      jp: "紫苑で確認",
      detail: "役に立った / 要修正",
      x: 520,
      y: 388,
      color: "#e879f9",
      fill: "rgba(88,28,135,0.56)",
    },
    {
      title: "Reflect",
      jp: "振り返る",
      detail: "改善PM / 内省",
      x: 308,
      y: 410,
      color: "#a78bfa",
      fill: "rgba(76,29,149,0.56)",
    },
    {
      title: "Improve",
      jp: "直す",
      detail: "Prompt / RAG / UI / API",
      x: 98,
      y: 350,
      color: "#34d399",
      fill: "rgba(6,78,59,0.52)",
    },
    {
      title: "Verify",
      jp: "検証する",
      detail: "pytest / typecheck",
      x: 36,
      y: 200,
      color: "#22c55e",
      fill: "rgba(20,83,45,0.5)",
    },
    {
      title: "Gate",
      jp: "人間が止める",
      detail: "実装 / git / deploy",
      x: 128,
      y: 72,
      color: "#fb7185",
      fill: "rgba(136,19,55,0.52)",
    },
  ];

  return (
    <svg
      viewBox="0 0 780 480"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full"
      style={{ minWidth: 560 }}
    >
      <defs>
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

        <marker id="arrow-violet" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#a78bfa" />
        </marker>
        <marker id="arrow-loop" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
          <path d="M0,0 L0,7 L9,3.5 z" fill="#a78bfa" />
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

      <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(100,120,180,0.06)" strokeWidth="0.5" />
      </pattern>
      <rect width="780" height="480" fill="url(#grid)" />

      <circle cx="390" cy="240" r="182" fill="none" stroke="rgba(167,139,250,0.25)" strokeWidth="12" />
      <circle cx="390" cy="240" r="182" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      <path
        id="loop-orbit"
        d="M 390 58 A 182 182 0 1 1 389 58"
        fill="none"
        stroke="#a78bfa"
        strokeWidth="2"
        strokeOpacity="0.72"
        strokeDasharray="10 7"
        markerEnd="url(#arrow-loop)"
      />

      <circle r="5" fill="url(#particle-violet)">
        <animateMotion dur="7s" repeatCount="indefinite" rotate="auto">
          <mpath href="#loop-orbit" />
        </animateMotion>
        <animate attributeName="opacity" values="0.1;1;0.1" dur="7s" repeatCount="indefinite" />
      </circle>
      <circle r="3.5" fill="#60a5fa" opacity="0.85">
        <animateMotion dur="7s" repeatCount="indefinite" begin="2.4s" rotate="auto">
          <mpath href="#loop-orbit" />
        </animateMotion>
        <animate attributeName="opacity" values="0.05;0.9;0.05" dur="7s" repeatCount="indefinite" begin="2.4s" />
      </circle>

      <g filter="url(#glow-violet)">
        <rect x="282" y="183" width="216" height="114" rx="20" fill="rgba(15,23,42,0.94)" stroke="#a78bfa" strokeWidth="1.8" />
        <text x="390" y="215" textAnchor="middle" fill="#f0abfc" fontSize="15" fontWeight="900">紫苑 改善PM</text>
        <text x="390" y="238" textAnchor="middle" fill="#c4b5fd" fontSize="10" fontWeight="700">判断資産 DevOps Loop</text>
        <text x="390" y="262" textAnchor="middle" fill="#94a3b8" fontSize="9">読む・報告・相談・開発依頼文</text>
        <text x="390" y="279" textAnchor="middle" fill="#fda4af" fontSize="9" fontWeight="700">実装 / git / deploy は人間承認</text>
      </g>

      {loopNodes.map((node, index) => (
        <g key={node.title} className="node-fade" style={{ animationDelay: `${index * 0.12}s` }}>
          <rect
            x={node.x}
            y={node.y}
            width="136"
            height="64"
            rx="14"
            fill={node.fill}
            stroke={node.color}
            strokeWidth="1.4"
          />
          <text x={node.x + 68} y={node.y + 21} textAnchor="middle" fill={node.color} fontSize="10.5" fontWeight="900">
            {node.title}
          </text>
          <text x={node.x + 68} y={node.y + 39} textAnchor="middle" fill="#f8fafc" fontSize="9.5" fontWeight="800">
            {node.jp}
          </text>
          <text x={node.x + 68} y={node.y + 54} textAnchor="middle" fill="#cbd5e1" fontSize="7.8">
            {node.detail}
          </text>
        </g>
      ))}

      <path
        d="M 463 163 C 520 148 555 171 575 216"
        fill="none"
        stroke="#38bdf8"
        strokeWidth="1.2"
        strokeOpacity="0.45"
        markerEnd="url(#arrow-blue)"
      />
      <text x="542" y="154" fill="#7dd3fc" fontSize="8" fontWeight="700">RAG / Memory Recall</text>

      <path
        d="M 494 288 C 540 314 546 350 506 386"
        fill="none"
        stroke="#e879f9"
        strokeWidth="1.2"
        strokeOpacity="0.5"
        markerEnd="url(#arrow-fuchsia)"
      />
      <text x="538" y="334" fill="#f0abfc" fontSize="8" fontWeight="700">紫苑レビュー</text>

      <path
        d="M 282 268 C 224 292 183 278 157 244"
        fill="none"
        stroke="#34d399"
        strokeWidth="1.2"
        strokeOpacity="0.5"
        markerEnd="url(#arrow-green)"
      />
      <text x="173" y="296" fill="#86efac" fontSize="8" fontWeight="700">pytest / typecheck</text>

      <path
        d="M 305 181 C 260 143 237 108 262 78"
        fill="none"
        stroke="#fb7185"
        strokeWidth="1.2"
        strokeOpacity="0.5"
        markerEnd="url(#arrow-violet)"
      />
      <text x="202" y="132" fill="#fda4af" fontSize="8" fontWeight="700">人間承認ゲート</text>

      <text x="390" y="464" textAnchor="middle" fill="#a78bfa" fontSize="9" opacity="0.72" fontStyle="italic">
        Use → Observe → Detect → Review → Reflect → Improve → Verify → Gate → Operate
      </text>

    </svg>
  );
}
