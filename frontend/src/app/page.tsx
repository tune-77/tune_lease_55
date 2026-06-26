"use client";

import Link from "next/link";
import { useState } from "react";
import { apiClient } from "@/lib/api";
import { Activity, ArrowRight, Calculator, Eye, MessageSquare, Network, PieChart, AlignLeft, Share2, AlertTriangle, ListOrdered, BadgeInfo, DollarSign, Database, ChevronDown, ChartNoAxesCombined, FileOutput, SlidersHorizontal, ScanText, ShieldCheck, XCircle, Minus } from "lucide-react";
import ScoreDAG from "../components/ScoreDAG";
import { ScoringFormData, defaultFormData } from "../types";
import FormGeneral from "../components/form/FormGeneral";
import FormFinancial from "../components/form/FormFinancial";
import FormQualitative from "../components/form/FormQualitative";
import { toThousandYenPayload } from "../lib/scoringUnits";

import IndicatorCards from "../components/analysis/IndicatorCards";
import RealGraphs from "../components/analysis/RealGraphs";
import GunshiAdvice from "../components/analysis/GunshiAdvice";
import ReportGenerator from "../components/analysis/ReportGenerator";
import QRiskPanel from "../components/analysis/QRiskPanel";
import MahalanobisPanel from "../components/analysis/MahalanobisPanel";
import UMAPPanel from "../components/analysis/UMAPPanel";
import OcrUpload from "../components/analysis/OcrUpload";
import { triggerMebuki } from "../components/layout/FloatingMebuki";

const DATA_SOURCE_FIELD_LABELS: Record<string, string> = {
  company_no: "企業番号",
  company_name: "企業名",
  industry_major: "大分類業種",
  industry_sub: "小分類業種",
  grade: "社内格付",
  customer_type: "新規/既存",
  main_bank: "メイン先区分",
  competitor: "競合状況",
  deal_source: "商談ソース",
  sales_dept: "営業部",
  contract_type: "契約種類",
  nenshu: "売上高",
  op_profit: "営業利益",
  ord_profit: "経常利益",
  net_income: "純利益",
  net_assets: "純資産",
  total_assets: "総資産",
  bank_credit: "銀行与信",
  lease_credit: "リース与信",
  contracts: "契約件数",
  lease_term: "リース期間",
  acquisition_cost: "取得価格",
  asset_score: "物件スコア",
  selected_asset_id: "物件ID",
  asset_name: "対象物件名",
  asset_detail: "型式・仕様",
  asset_purpose: "導入目的",
  asset_location: "設置場所",
  asset_evidence_level: "確認資料",
  passion_text: "営業メモ",
  intuition: "直感スコア",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function AiHeroCard({ result }: { result?: Record<string, any> }) {
  if (!result) return null;
  const score: number = result.score_base ?? 0;
  const hantei: string = result.hantei ?? "";
  const isApproved = score >= 71;
  const isConditional = score >= 60 && score < 71;

  let gradientClass = "from-rose-500 to-rose-600";
  let shadowClass = "shadow-rose-200";
  let badge = "否決";
  let BadgeIcon = XCircle;
  if (isApproved) {
    gradientClass = "from-emerald-500 to-teal-600";
    shadowClass = "shadow-emerald-200";
    badge = "承認";
    BadgeIcon = ShieldCheck;
  } else if (isConditional) {
    gradientClass = "from-amber-500 to-orange-500";
    shadowClass = "shadow-amber-200";
    badge = "条件付き";
    BadgeIcon = Minus;
  }

  return (
    <div className={`bg-gradient-to-br ${gradientClass} rounded-3xl p-6 md:p-8 shadow-2xl ${shadowClass} text-white mb-6`}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="text-xs font-black uppercase tracking-widest text-white/60 mb-1">AI 審査判定</div>
          <div className="flex items-end gap-3">
            <span className="text-7xl md:text-8xl font-black drop-shadow-lg leading-none">
              {score.toFixed(1)}
            </span>
            <span className="text-2xl font-bold text-white/70 mb-2">点</span>
          </div>
          {hantei && (
            <p className="mt-2 text-sm font-bold text-white/80 leading-relaxed max-w-xl">
              {hantei}
            </p>
          )}
        </div>
        <div className="flex flex-col items-center gap-2">
          <div className="flex items-center gap-2 bg-white/20 rounded-2xl px-6 py-4 backdrop-blur-sm">
            <BadgeIcon className="w-7 h-7" />
            <span className="text-2xl font-black">{badge}</span>
          </div>
          <div className="text-[11px] font-bold text-white/60">
            承認ライン: 71点以上
          </div>
        </div>
      </div>
      {result.score_borrower != null && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
          {[
            { label: "借手スコア", value: result.score_borrower?.toFixed(1) },
            { label: "量子リスク", value: result.quantum_risk?.toFixed(1) },
            { label: "UMAP異常度", value: result.umap_anomaly_score?.toFixed(1) },
            { label: "マハラノビス", value: result.mahalanobis_score?.toFixed(1) },
          ].map(({ label, value }) =>
            value != null ? (
              <div key={label} className="rounded-xl bg-white/15 px-3 py-2 text-center backdrop-blur-sm">
                <div className="text-[10px] font-black text-white/60">{label}</div>
                <div className="text-lg font-black">{value}</div>
              </div>
            ) : null
          )}
        </div>
      )}
    </div>
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
  const manualFields = summary.manual_input_fields || [];
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
            {manualFields.map((field: string) => (
              <span key={field} className="rounded-md bg-slate-100 px-2 py-1 text-[11px] font-bold text-slate-600">
                {DATA_SOURCE_FIELD_LABELS[field] || field}
              </span>
            ))}
            {!manualFields.length && (
              <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] font-bold text-amber-700">
                画面入力の反映項目なし
              </span>
            )}
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
      const res = await apiClient.post(`/api/score/full`, toThousandYenPayload(formData));
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

        {/* AI審査判定ヒーローカード（結果あり時のみ最上部に表示） */}
        <AiHeroCard result={result} />

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
                数値分析
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

                {/* 決算書OCR読み取り */}
                <section className="bg-white border border-indigo-100 rounded-2xl p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                      <ScanText className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-black text-slate-800">決算書OCR読み取り</h3>
                      <p className="text-xs text-slate-500">画像・PDFから財務数値を自動入力（Gemini Vision）</p>
                    </div>
                  </div>
                  <OcrUpload
                    onApply={(fields) => {
                      Object.entries(fields).forEach(([k, v]) => handleFieldChange(k, v as string | number | string[]));
                    }}
                  />
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
                    {/* 初期表示は判断に必要な結論だけに絞る */}
                    <IndicatorCards data={result} />

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

                    <RateProposalCard proposal={result.rate_proposal} />

                    <div className="space-y-3">
                      <details className="group overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                        <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3">
                          <SlidersHorizontal className="h-4 w-4 text-indigo-600" />
                          <span className="text-sm font-black text-slate-800">スコア構成・補助指標</span>
                          <span className="ml-auto hidden text-[11px] font-bold text-slate-400 sm:inline">DAG / Q_risk / 類似度</span>
                          <ChevronDown className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
                        </summary>
                        <div className="space-y-4 border-t border-slate-100 p-4">
                          <ScoreDAG data={result} />

                          {result.quantum_risk != null && (
                            <QRiskPanel
                              quantumRisk={result.quantum_risk}
                              creditQuantumStrongWarning={result.credit_quantum_strong_warning ?? false}
                              compact={false}
                            />
                          )}

                          {result.umap_anomaly_score != null && result.umap_x != null && result.umap_y != null && (
                            <UMAPPanel
                              score={result.umap_anomaly_score}
                              umapX={result.umap_x}
                              umapY={result.umap_y}
                              similar={result.umap_similar}
                              compact={false}
                            />
                          )}

                          {result.mahalanobis_score != null && (
                            <MahalanobisPanel
                              score={result.mahalanobis_score}
                              advice={result.mahalanobis_advice}
                              compact={false}
                            />
                          )}
                        </div>
                      </details>

                      <details className="group overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                        <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3">
                          <ChartNoAxesCombined className="h-4 w-4 text-emerald-600" />
                          <span className="text-sm font-black text-slate-800">詳細グラフ・入力情報</span>
                          <span className="ml-auto hidden text-[11px] font-bold text-slate-400 sm:inline">財務比較 / 情報源</span>
                          <ChevronDown className="h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
                        </summary>
                        <div className="space-y-4 border-t border-slate-100 p-4">
                          <RealGraphs
                            companyName={formData.company_name || ""}
                            nenshu={formData.nenshu || 0}
                            opMarginPct={result?.user_op_margin || 0}
                            equityRatio={result?.user_equity_ratio || 0}
                            scoreBorrower={result?.score_borrower || 50}
                            scoreBase={result?.score_base || 50}
                          />
                          <DataSourceSummaryCard summary={result.data_source_summary} />
                        </div>
                      </details>

                      <details className="group overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
                        <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3">
                          <FileOutput className="h-4 w-4 text-slate-700" />
                          <span className="text-sm font-black text-slate-800">稟議書・レポート出力</span>
                          <ChevronDown className="ml-auto h-4 w-4 text-slate-400 transition-transform group-open:rotate-180" />
                        </summary>
                        <div className="border-t border-slate-100 p-4 [&>div]:mt-0">
                          <ReportGenerator apiResult={result} formData={formData} gunshiText={gunshiText} />
                        </div>
                      </details>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* 右カラム: 数値の再掲ではなく、戦略・質問・稟議表現を担当 */}
          <div className="w-full 2xl:w-[42%] mt-8 2xl:mt-0 relative z-10">
            <GunshiAdvice
              score={result?.score_base || 0}
              modelDecision={result?.hantei || ""}
              industry_major={result?.industry_major || formData.industry_major || ""}
              formData={formData}
              estatContext={result?.estat_context || null}
              onChatLoaded={setGunshiText}
            />
          </div>
          
        </div>
      </div>
    </div>
  );
}
