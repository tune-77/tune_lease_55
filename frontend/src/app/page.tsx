"use client";

import Link from "next/link";
import { useState } from "react";
import axios from "axios";
import { Activity, ArrowRight, Calculator, Eye, MessageSquare, Network, PieChart, AlignLeft, Share2, AlertTriangle, ListOrdered, BadgeInfo, ClipboardList, DollarSign, Database, CheckCircle2, FileText, Copy } from "lucide-react";
import ScoreDAG from "../components/ScoreDAG";
import { ScoringFormData, defaultFormData } from "../types";
import FormGeneral from "../components/form/FormGeneral";
import FormFinancial from "../components/form/FormFinancial";
import FormQualitative from "../components/form/FormQualitative";
import { toThousandYenPayload } from "../lib/scoringUnits";

import IndicatorCards from "../components/analysis/IndicatorCards";
import RealGraphs from "../components/analysis/RealGraphs";
import AIAnalysis from "../components/analysis/AIAnalysis";
import AdvancedAnalysis from "../components/analysis/AdvancedAnalysis";
import GunshiAdvice from "../components/analysis/GunshiAdvice";
import ReportGenerator from "../components/analysis/ReportGenerator";
import QRiskPanel from "../components/analysis/QRiskPanel";
import MahalanobisPanel from "../components/analysis/MahalanobisPanel";
import UMAPPanel from "../components/analysis/UMAPPanel";
import { triggerMebuki } from "../components/layout/FloatingMebuki";

function ConditionalApprovalActionsCard({ actions }: { actions?: Array<{ priority?: string; action?: string; reason?: string; category?: string }> }) {
  if (!actions?.length) return null;
  return (
    <section className="bg-amber-50 border border-amber-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <ClipboardList className="w-5 h-5 text-amber-700" />
        <div>
          <h3 className="text-sm font-black text-amber-900">条件付き承認アクション</h3>
          <p className="text-[11px] font-bold text-amber-700">稟議前に潰す条件を優先順で提示します。</p>
        </div>
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        {actions.map((item, index) => {
          const must = item.priority === "must";
          return (
            <div key={`${item.action}-${index}`} className={`rounded-xl border p-3 ${must ? "bg-white border-amber-300" : "bg-amber-100/60 border-amber-200"}`}>
              <div className="flex items-center gap-2">
                <CheckCircle2 className={`w-4 h-4 ${must ? "text-amber-700" : "text-amber-500"}`} />
                <span className={`text-[10px] font-black rounded-full px-2 py-0.5 ${must ? "bg-amber-700 text-white" : "bg-white text-amber-700 border border-amber-200"}`}>
                  {must ? "必須" : "推奨"}
                </span>
                {item.category && <span className="text-[10px] font-bold text-slate-500">{item.category}</span>}
              </div>
              <p className="mt-2 text-sm font-black text-slate-800">{item.action}</p>
              {item.reason && <p className="mt-1 text-xs font-bold leading-relaxed text-slate-600">{item.reason}</p>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function RateProposalCard({ proposal }: { proposal?: any }) {
  if (!proposal?.proposed_rate) return null;
  const breakdown = proposal.breakdown || {};
  const rows = [
    ["基準金利", breakdown.base_rate],
    ["物件スプレッド", breakdown.asset_spread],
    ["格付スプレッド", breakdown.grade_spread],
    ["リスク調整", breakdown.risk_adjustment],
  ];
  return (
    <section className="bg-emerald-50 border border-emerald-200 rounded-2xl p-4 shadow-sm">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-emerald-700" />
          <div>
            <h3 className="text-sm font-black text-emerald-900">動的金利提案</h3>
            <p className="text-[11px] font-bold text-emerald-700">{proposal.guidance || "審査結果欄の初期提示用です。"}</p>
          </div>
        </div>
        <div className="rounded-xl bg-white border border-emerald-200 px-4 py-2 text-right">
          <div className="text-[10px] font-black text-emerald-600">提案金利</div>
          <div className="text-3xl font-black text-emerald-800">{Number(proposal.proposed_rate).toFixed(2)}%</div>
        </div>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-5">
        {rows.map(([label, value]) => (
          <div key={label as string} className="rounded-xl bg-white border border-emerald-100 px-3 py-2">
            <div className="text-[10px] font-black text-slate-400">{label}</div>
            <div className="text-sm font-black text-slate-800">{typeof value === "number" ? `${value.toFixed(2)}%` : "-"}</div>
          </div>
        ))}
        <div className="rounded-xl bg-white border border-emerald-100 px-3 py-2">
          <div className="text-[10px] font-black text-slate-400">月額目安</div>
          <div className="text-sm font-black text-slate-800">{proposal.monthly_payment ? `${Number(proposal.monthly_payment).toLocaleString("ja-JP")}円` : "-"}</div>
        </div>
      </div>
    </section>
  );
}

function DataSourceSummaryCard({ summary }: { summary?: any }) {
  if (!summary) return null;
  const assetClarity = summary.asset_clarity;
  return (
    <section className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <Database className="w-5 h-5 text-slate-600" />
        <div>
          <h3 className="text-sm font-black text-slate-800">案件データの情報源</h3>
          <p className="text-[11px] font-bold text-slate-500">{summary.primary_source}</p>
        </div>
        <span className="ml-auto rounded-full bg-slate-100 px-2 py-1 text-[10px] font-black text-slate-600">
          入力 {summary.manual_input_count ?? 0}項目
        </span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-1">画面入力</div>
          <div className="flex flex-wrap gap-1.5">
            {(summary.manual_input_fields || []).map((field: string) => (
              <span key={field} className="rounded-md bg-slate-100 px-2 py-1 text-[11px] font-bold text-slate-600">{field}</span>
            ))}
          </div>
        </div>
        <div>
          <div className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-1">モデル/マスタ</div>
          <ul className="space-y-1">
            {(summary.model_sources || []).map((source: string) => (
              <li key={source} className="text-xs font-bold text-slate-600 flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
                {source}
              </li>
            ))}
          </ul>
        </div>
      </div>
      {assetClarity && (
        <div className={`mt-3 rounded-xl border px-3 py-2 ${
          assetClarity.status === "明確" ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"
        }`}>
          <div className="flex items-center justify-between gap-2">
            <span className={`text-xs font-black ${assetClarity.status === "明確" ? "text-emerald-800" : "text-amber-800"}`}>
              物件明確化: {assetClarity.status}
            </span>
            <span className={`text-[11px] font-black ${assetClarity.status === "明確" ? "text-emerald-700" : "text-amber-700"}`}>
              {assetClarity.filled_count}/{assetClarity.required_count}
            </span>
          </div>
          {!!assetClarity.warnings?.length && (
            <div className="mt-1 flex flex-wrap gap-1.5">
              {assetClarity.warnings.map((warning: string) => (
                <span key={warning} className="rounded-md bg-white px-2 py-1 text-[10px] font-bold text-amber-700">
                  {warning}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ScreeningContextNotesCard({ notes }: { notes?: any }) {
  if (!notes) return null;
  const commentary = Array.isArray(notes.commentary) ? notes.commentary : [];
  const risks = Array.isArray(notes.risk_reasons) ? notes.risk_reasons : [];
  const conditions = Array.isArray(notes.condition_rationale) ? notes.condition_rationale : [];
  const missing = Array.isArray(notes.missing_inputs) ? notes.missing_inputs : [];
  const reflected = Array.isArray(notes.reflected_inputs) ? notes.reflected_inputs : [];
  const score = typeof notes.reflection_score === "number" ? notes.reflection_score : 0;

  return (
    <section className="bg-indigo-50 border border-indigo-200 rounded-2xl p-4 shadow-sm">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-2">
          <FileText className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-700" />
          <div>
            <h3 className="text-sm font-black text-indigo-950">入力反映メモ</h3>
            <p className="mt-1 text-[11px] font-bold leading-relaxed text-indigo-700">
              {notes.summary || "入力情報を審査コメント・条件案・リスク理由へ反映しました。"}
            </p>
          </div>
        </div>
        <div className="rounded-xl border border-indigo-200 bg-white px-4 py-2 text-right">
          <div className="text-[10px] font-black text-indigo-500">反映度</div>
          <div className="text-2xl font-black text-indigo-800">{score}%</div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 xl:grid-cols-3">
        <div className="rounded-xl border border-indigo-100 bg-white p-3">
          <div className="mb-2 text-[11px] font-black uppercase tracking-widest text-indigo-500">審査コメント</div>
          <div className="space-y-2">
            {commentary.length === 0 ? (
              <p className="text-xs font-bold text-slate-400">反映できるコメントがありません。</p>
            ) : commentary.slice(0, 5).map((item: any, index: number) => (
              <div key={`${item.label}-${index}`} className="rounded-lg bg-slate-50 p-2">
                <div className="text-[11px] font-black text-slate-700">{item.label}</div>
                <p className="mt-1 text-xs font-bold leading-relaxed text-slate-600">{item.text}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-indigo-100 bg-white p-3">
          <div className="mb-2 text-[11px] font-black uppercase tracking-widest text-rose-500">リスク理由</div>
          <div className="space-y-2">
            {risks.length === 0 ? (
              <p className="text-xs font-bold text-slate-400">重大な追加リスク理由はありません。</p>
            ) : risks.slice(0, 5).map((item: any, index: number) => (
              <div key={`${item.title}-${index}`} className={`rounded-lg border p-2 ${
                item.level === "high" ? "border-rose-200 bg-rose-50" : "border-amber-200 bg-amber-50"
              }`}>
                <div className={`text-[11px] font-black ${item.level === "high" ? "text-rose-700" : "text-amber-700"}`}>{item.title}</div>
                <p className="mt-1 text-xs font-bold leading-relaxed text-slate-700">{item.reason}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-indigo-100 bg-white p-3">
          <div className="mb-2 text-[11px] font-black uppercase tracking-widest text-emerald-600">条件案への反映</div>
          <div className="space-y-2">
            {conditions.length === 0 ? (
              <p className="text-xs font-bold text-slate-400">条件案への追加反映はありません。</p>
            ) : conditions.slice(0, 5).map((item: any, index: number) => (
              <div key={`${item.condition}-${index}`} className="rounded-lg bg-emerald-50 p-2">
                <div className="text-[11px] font-black text-emerald-800">{item.condition}</div>
                {item.reason && <p className="mt-1 text-xs font-bold leading-relaxed text-slate-600">{item.reason}</p>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <div className="rounded-xl border border-indigo-100 bg-white px-3 py-2">
          <div className="mb-1 text-[10px] font-black text-slate-400">反映済み入力</div>
          <div className="flex flex-wrap gap-1.5">
            {reflected.slice(0, 14).map((field: string) => (
              <span key={field} className="rounded-md bg-indigo-100 px-2 py-1 text-[10px] font-black text-indigo-700">{field}</span>
            ))}
          </div>
        </div>
        <div className="rounded-xl border border-indigo-100 bg-white px-3 py-2">
          <div className="mb-1 text-[10px] font-black text-slate-400">不足している説明材料</div>
          {missing.length === 0 ? (
            <p className="text-xs font-bold text-emerald-700">主要な不足項目はありません。</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {missing.map((field: string) => (
                <span key={field} className="rounded-md bg-amber-100 px-2 py-1 text-[10px] font-black text-amber-700">{field}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function ApprovalCommentDraftCard({ draft }: { draft?: any }) {
  const [copied, setCopied] = useState(false);
  if (!draft?.full_text) return null;
  const sections = Array.isArray(draft.sections) ? draft.sections : [];

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(draft.full_text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-2">
          <FileText className="mt-0.5 h-5 w-5 flex-shrink-0 text-slate-700" />
          <div>
            <h3 className="text-sm font-black text-slate-900">稟議コメント案</h3>
            <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-500">
              入力反映メモを、稟議本文に貼り付けやすい形式へ整えました。
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2 text-xs font-black text-white shadow-sm hover:bg-slate-800 transition-colors"
        >
          <Copy className="h-4 w-4" />
          {copied ? "コピー済み" : "本文コピー"}
        </button>
      </div>

      <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-slate-900 px-2 py-1 text-[10px] font-black text-white">{draft.verdict || "未判定"}</span>
          {typeof draft.score === "number" && (
            <span className="rounded-full bg-white px-2 py-1 text-[10px] font-black text-slate-600 border border-slate-200">score {draft.score.toFixed(1)}</span>
          )}
          <span className="text-xs font-black text-slate-700">{draft.title}</span>
        </div>
        {draft.summary && <p className="mt-2 text-xs font-bold leading-relaxed text-slate-600">{draft.summary}</p>}
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        {sections.map((section: any, index: number) => (
          <div key={`${section.title}-${index}`} className="rounded-xl border border-slate-200 bg-white p-3">
            <div className="text-[11px] font-black text-slate-500">{section.title}</div>
            <div className="mt-2 whitespace-pre-wrap text-xs font-bold leading-relaxed text-slate-700">{section.body}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] font-bold leading-relaxed text-amber-800">
        {draft.copy_hint || "貼り付け後、正式資料名と個別事情を追記してください。"}
      </div>
    </section>
  );
}

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [result, setResult] = useState<any>(null);
  const [formData, setFormData] = useState<ScoringFormData>(defaultFormData);
  const [gunshiText, setGunshiText] = useState<string>("");

  // タブ管理
  const [activeTab, setActiveTab] = useState<"input" | "analysis">("input");
  const inputSections = [
    { id: "form-general", label: "1. 案件特定", hint: "企業番号・業種・取引区分" },
    { id: "form-financial", label: "2. 財務", hint: "P/L・B/S" },
    { id: "form-qualitative", label: "3. 定性・物件", hint: "物件条件・メモ・音声入力" },
  ];

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

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
      const res = await axios.post(`/api/score/full`, toThousandYenPayload(formData));
      setResult(res.data);
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
        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          <Link href="/lease-kun" className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">Primary</div>
              <div className="font-black text-slate-800 mt-1 flex items-center gap-2"><MessageSquare className="w-4 h-4 text-amber-500" />スマホUIで入力</div>
              <div className="text-xs text-slate-500 mt-1">入力が少ない運用はこちら。</div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400" />
          </Link>
          <Link href="/" className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">Core</div>
              <div className="font-black text-slate-800 mt-1 flex items-center gap-2"><Calculator className="w-4 h-4 text-blue-500" />審査・分析</div>
              <div className="text-xs text-slate-500 mt-1">入力と分析をまとめて見る入口。</div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400" />
          </Link>
          <Link href="/competitor" className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">Insight</div>
              <div className="font-black text-slate-800 mt-1 flex items-center gap-2"><Share2 className="w-4 h-4 text-orange-500" />競合関係グラフ</div>
              <div className="text-xs text-slate-500 mt-1">競合の勢力図を確認。</div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400" />
          </Link>
          <Link href="/similar" className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">Insight</div>
              <div className="font-black text-slate-800 mt-1 flex items-center gap-2"><Network className="w-4 h-4 text-teal-500" />案件類似ネットワーク</div>
              <div className="text-xs text-slate-500 mt-1">似た案件の関係を追う。</div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400" />
          </Link>
        </div>

        <div className="mb-6">
          <Link href="/visual" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 text-white font-bold shadow-sm hover:bg-slate-800 transition-colors">
            <Eye className="w-4 h-4" />
            ビジュアルインサイト
          </Link>
        </div>

        <div className="flex flex-col 2xl:flex-row gap-6">

          {/* 左カラム: メイン操作エリア (入力・分析) */}
          <div className="w-full 2xl:w-[58%] flex flex-col">
            
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
                <section className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                        <ListOrdered className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="text-sm font-black text-slate-800">入力の順番</h3>
                        <p className="mt-1 text-xs text-slate-500">上から順に埋めると、案件特定から分析まで迷いにくいです。</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
                      <BadgeInfo className="h-4 w-4 text-slate-400" />
                      必須は企業番号・売上高・総資産・取得価額です
                    </div>
                  </div>
                  <div className="mt-4 grid gap-2 md:grid-cols-3">
                    {inputSections.map((section) => (
                      <button
                        key={section.id}
                        type="button"
                        onClick={() => scrollToSection(section.id)}
                        className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-left hover:border-blue-200 hover:bg-blue-50 transition-colors"
                      >
                        <div className="text-sm font-black text-slate-800">{section.label}</div>
                        <div className="mt-1 text-xs text-slate-500">{section.hint}</div>
                      </button>
                    ))}
                  </div>
                </section>

                <section id="form-general" className="scroll-mt-28">
                  <FormGeneral data={formData} onChange={handleFieldChange} />
                </section>
                <section id="form-financial" className="scroll-mt-28">
                  <FormFinancial data={formData} onChange={handleFieldChange} />
                </section>
                <section id="form-qualitative" className="scroll-mt-28">
                  <FormQualitative data={formData} onChange={handleFieldChange} />
                </section>

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

                    {/* 財務分布マップ（UMAP + Isolation Forest）*/}
                    {result.umap_anomaly_score != null && result.umap_x != null && result.umap_y != null && (
                      <UMAPPanel
                        score={result.umap_anomaly_score}
                        umapX={result.umap_x}
                        umapY={result.umap_y}
                        similar={result.umap_similar}
                        compact={false}
                      />
                    )}

                    {/* 財務プロファイル類似度 */}
                    {result.mahalanobis_score != null && (
                      <MahalanobisPanel
                        score={result.mahalanobis_score}
                        advice={result.mahalanobis_advice}
                        compact={false}
                      />
                    )}

                    {/* REV-089/113/114: Q_risk パネル */}
                    {result.quantum_risk != null && (
                      <QRiskPanel
                        quantumRisk={result.quantum_risk}
                        creditQuantumStrongWarning={result.credit_quantum_strong_warning ?? false}
                        compact={false}
                      />
                    )}

                    {/* REV-004: デフォルト率モデル警告パネル */}
                    {result.default_warnings?.length > 0 && (
                      <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 shadow-sm">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0" />
                          <span className="font-black text-rose-800 text-sm">デフォルト率モデル 高リスク警告</span>
                          <span className="ml-auto text-[10px] font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-600 border border-rose-200">スコアに非影響</span>
                        </div>
                        <ul className="space-y-1">
                          {(result.default_warnings as string[]).map((w, i) => (
                            <li key={i} className="text-xs text-rose-700 font-medium flex items-start gap-1.5">
                              <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-rose-400 flex-shrink-0" />
                              {w}
                            </li>
                          ))}
                        </ul>
                        <p className="text-[10px] text-rose-400 mt-2">財務パターンを学習済みLightGBMモデルで判定。審査スコアとは独立した補助指標です。</p>
                      </div>
                    )}

                    <ConditionalApprovalActionsCard actions={result.conditional_approval_actions} />
                    <ScreeningContextNotesCard notes={result.screening_context_notes} />
                    <ApprovalCommentDraftCard draft={result.approval_comment_draft} />
                    <RateProposalCard proposal={result.rate_proposal} />
                    <DataSourceSummaryCard summary={result.data_source_summary} />

                    {/* 📊 新設: Recharts による本物のインタラクティブグラフ群 */}
                    <RealGraphs
                      companyName={formData.company_name || ""}
                      nenshu={formData.nenshu || 0}
                      opMarginPct={result?.user_op_margin || 0}
                      equityRatio={result?.user_equity_ratio || 0}
                      scoreBorrower={result?.score_borrower || 50}
                      scoreBase={result?.score_base || 50}
                    />

                    {/* AI分析テキスト (チャット風) */}
                    <AIAnalysis comparisonText={result.comparison} />
                    
                    {/* 今回追加した高度シミュレーションUI */}
                    <AdvancedAnalysis
                      industrySub={result?.industry_sub || ""}
                      companyName={formData.company_name || ""}
                      score={result?.score_base || 50}
                    />

                    {/* 最終審査レポート */}
                    <ReportGenerator apiResult={result} formData={formData} gunshiText={gunshiText} />
                  </>
                )}
              </div>
            )}
          </div>

          {/* 右カラム: 審査軍師 (逆転プラン自動提案) */}
          <div className="w-full 2xl:w-[42%] mt-8 2xl:mt-0 relative z-10">
            <GunshiAdvice
              score={result?.score_base || 0}
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
