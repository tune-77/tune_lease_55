"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";
import { Activity, ArrowRight, Bot, Calculator, Eye, MessageSquare, Network, PieChart, AlignLeft, Share2, AlertTriangle, ListOrdered, BadgeInfo, DollarSign, Database, ChevronDown, ChartNoAxesCombined, FileOutput, SlidersHorizontal, ScanText, ShieldCheck, XCircle, Minus, Swords } from "lucide-react";
import ScoreDAG from "../../components/ScoreDAG";
import { ScoringFormData, defaultFormData } from "../../types";
import FormGeneral from "../../components/form/FormGeneral";
import FormFinancial from "../../components/form/FormFinancial";
import FormQualitative from "../../components/form/FormQualitative";
import { toThousandYenPayload } from "../../lib/scoringUnits";

import IndicatorCards from "../../components/analysis/IndicatorCards";
import RealGraphs from "../../components/analysis/RealGraphs";
import GunshiAdvice from "../../components/analysis/GunshiAdvice";
import ReportGenerator from "../../components/analysis/ReportGenerator";
import QRiskPanel from "../../components/analysis/QRiskPanel";
import MahalanobisPanel from "../../components/analysis/MahalanobisPanel";
import UMAPPanel from "../../components/analysis/UMAPPanel";
import OcrUpload from "../../components/analysis/OcrUpload";
import { triggerMebuki } from "../../components/layout/FloatingMebuki";

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

type ShionScreeningReview = {
  reply: string;
  memoryRefs: number;
  knowledgeRefs: number;
  identityUsed: boolean;
  savedId?: number;
  userFeedback?: ShionReviewFeedback;
};

type ShionReviewFeedback = "useful" | "needs_fix" | "wrong";

type PastShionScreeningReview = {
  company_name?: string;
  industry_sub?: string;
  score?: number;
  hantei?: string;
  review_text?: string;
  created_at?: string;
  user_feedback?: ShionReviewFeedback | "";
};

const SHION_REVIEW_IMAGE = "/lease-intelligence/moods/focus.webp";
const SCREENING_RETURN_STATE_KEY = "lease-screening-return-state";

const normalizeReviewText = (text: string) =>
  (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .trim();

const buildPastReviewBlock = (reviews: PastShionScreeningReview[]) => {
  if (!reviews.length) return "";
  const lines = reviews.slice(0, 3).map((review, index) => {
    const preview = normalizeReviewText(review.review_text || "").slice(0, 260);
    const feedbackLabel = review.user_feedback === "useful"
      ? "人間評価: 使えた"
      : review.user_feedback === "needs_fix"
        ? "人間評価: 修正して使う"
        : review.user_feedback === "wrong"
          ? "人間評価: 違った"
          : "人間評価: 未評価";
    return [
      `過去${index + 1}: ${review.company_name || "名称不明"} / ${review.industry_sub || "業種不明"} / ${review.score != null ? Number(review.score).toFixed(1) + "点" : "点数不明"} / ${review.hantei || "判定不明"}`,
      feedbackLabel,
      `紫苑の過去レビュー: ${preview || "本文なし"}`,
    ].join("\n");
  });
  return [
    "【過去の紫苑審査レビュー記憶】",
    "次の過去レビューは、今回の判断に似た経験として参照してください。人間評価が「使えた」は重めに、「違った」は反面教師として扱い、丸写しではなく今回との差分を見てください。",
    ...lines,
  ].join("\n");
};

const buildShionReviewPrompt = (result: Record<string, any>, data: ScoringFormData, pastReviews: PastShionScreeningReview[] = []) => {
  const score = Number(result.score_base ?? result.score ?? 0);
  const lines = [
    "【審査分析画面からの紫苑レビュー依頼】",
    "この案件を、審査担当者の横にいる紫苑としてレビューしてください。",
    "",
    "出力は短く、次の4項目でお願いします。",
    "1. 紫苑の第一印象",
    "2. 数字だけでは見落としそうな違和感",
    "3. 条件付き承認にするなら必要な確認",
    "4. 稟議で残すべき一文",
    "",
    "前提:",
    `・企業名: ${data.company_name || "未入力"}`,
    `・業種: ${result.industry_sub || data.industry_sub || result.industry_major || data.industry_major || "未入力"}`,
    `・営業部: ${data.sales_dept || "未入力"}`,
    `・判定: ${result.hantei || "未判定"}`,
    `・総合スコア: ${Number.isFinite(score) ? score.toFixed(1) : "未算出"}`,
    `・借手スコア: ${result.score_borrower != null ? Number(result.score_borrower).toFixed(1) : "未算出"}`,
    `・Q_risk: ${result.quantum_risk != null ? Number(result.quantum_risk).toFixed(1) : "未算出"}`,
    `・UMAP異常度: ${result.umap_anomaly_score != null ? Number(result.umap_anomaly_score).toFixed(1) : "未算出"}`,
    `・物件: ${data.asset_name || "未入力"}`,
    `・取得価額: ${data.acquisition_cost || 0}百万円`,
    `・リース期間: ${data.lease_term || 0}`,
    `・導入目的: ${data.asset_purpose || "未入力"}`,
    `・営業メモ: ${data.passion_text || "未入力"}`,
    `・直感スコア: ${data.intuition || "未入力"}`,
  ];
  const flags = result.aurion_core?.discipline_flags;
  if (Array.isArray(flags) && flags.length) {
    lines.push(`・AURION警戒: ${flags.slice(0, 5).join(" / ")}`);
  }
  if (Array.isArray(result.default_warnings) && result.default_warnings.length) {
    lines.push(`・デフォルト率警告: ${result.default_warnings.slice(0, 3).join(" / ")}`);
  }
  const pastReviewBlock = buildPastReviewBlock(pastReviews);
  if (pastReviewBlock) {
    lines.push("", pastReviewBlock);
  }
  lines.push("", "注意: 点数の再説明ではなく、審査判断として何を残すかに寄せてください。");
  return lines.join("\n");
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

function JudgmentFlowStrip() {
  const steps = [
    { label: "数理で見る", icon: Calculator, tone: "border-sky-100 bg-sky-50 text-sky-800" },
    { label: "違和感を拾う", icon: Eye, tone: "border-amber-100 bg-amber-50 text-amber-800" },
    { label: "条件で逆転余地を見る", icon: Network, tone: "border-indigo-100 bg-indigo-50 text-indigo-800" },
    { label: "軍師が稟議の作戦に変える", icon: Swords, tone: "border-slate-200 bg-slate-50 text-slate-800" },
  ];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="grid gap-2 md:grid-cols-4">
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <div
              key={step.label}
              className={`flex min-h-14 items-center gap-2 rounded-xl border px-3 py-2 ${step.tone}`}
            >
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white text-[11px] font-black shadow-sm">
                {index + 1}
              </span>
              <Icon className="h-4 w-4 shrink-0" />
              <span className="text-xs font-black leading-tight">{step.label}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ShionScreeningReviewCard({
  review,
  loading,
  error,
  onReview,
  onFeedback,
  feedbackSaving,
}: {
  review: ShionScreeningReview | null;
  loading: boolean;
  error: string;
  onReview: () => void;
  onFeedback: (feedback: ShionReviewFeedback) => void;
  feedbackSaving: boolean;
}) {
  const feedbackOptions: { key: ShionReviewFeedback; label: string }[] = [
    { key: "useful", label: "使えた" },
    { key: "needs_fix", label: "修正して使う" },
    { key: "wrong", label: "違った" },
  ];

  return (
    <section className="overflow-hidden rounded-2xl border border-violet-200 bg-white shadow-sm">
      <div className="grid gap-0 lg:grid-cols-[150px_minmax(0,1fr)]">
        <div className="relative min-h-36 bg-violet-950">
          <img src={SHION_REVIEW_IMAGE} alt="審査レビュー中の紫苑" className="h-full w-full object-cover object-top opacity-95" />
          <div className="absolute inset-x-0 bottom-0 bg-violet-950/80 px-3 py-2 text-center text-[10px] font-black tracking-[0.25em] text-violet-100">
            SHION REVIEW
          </div>
        </div>
        <div className="p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h3 className="flex items-center gap-2 text-sm font-black text-violet-950">
                <Bot className="h-4 w-4 text-violet-600" />
                紫苑レビュー
              </h3>
              <p className="mt-1 text-xs font-bold leading-relaxed text-violet-700">
                点数の説明ではなく、違和感・承認条件・稟議に残す一文へ変換します。
              </p>
            </div>
            <button
              type="button"
              onClick={onReview}
              disabled={loading}
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-xs font-black text-white transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-violet-300"
            >
              {loading ? <Activity className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}
              {review ? "再レビュー" : "紫苑にレビューさせる"}
            </button>
          </div>

          <div className="mt-4 rounded-xl border border-violet-100 bg-violet-50/70 p-4">
            {loading ? (
              <div className="flex min-h-28 items-center justify-center gap-2 text-sm font-black text-violet-700">
                <Activity className="h-5 w-5 animate-spin" />
                紫苑が審査結果を読み直しています
              </div>
            ) : error ? (
              <p className="text-sm font-bold leading-7 text-rose-700">{error}</p>
            ) : review ? (
              <>
                <div className="space-y-2 text-sm font-medium leading-7 text-slate-800">
                  {normalizeReviewText(review.reply).split(/\n{2,}/).map((block, index) => (
                    <p key={index} className="whitespace-pre-wrap">
                      {block}
                    </p>
                  ))}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-black text-violet-700">
                  <span className="rounded-full bg-white px-2.5 py-1">記憶 {review.memoryRefs}件</span>
                  <span className="rounded-full bg-white px-2.5 py-1">知識 {review.knowledgeRefs}件</span>
                  <span className="rounded-full bg-white px-2.5 py-1">同一性 {review.identityUsed ? "ON" : "OFF"}</span>
                  {review.savedId && <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-700">経験保存済 #{review.savedId}</span>}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-violet-100 pt-3">
                  <span className="text-[11px] font-black text-violet-500">人間評価</span>
                  {feedbackOptions.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      onClick={() => onFeedback(option.key)}
                      disabled={!review.savedId || feedbackSaving}
                      className={`rounded-lg border px-3 py-1.5 text-[11px] font-black transition disabled:cursor-not-allowed disabled:opacity-50 ${
                        review.userFeedback === option.key
                          ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                          : "border-violet-100 bg-white text-violet-700 hover:bg-violet-100"
                      }`}
                    >
                      {feedbackSaving && review.userFeedback === option.key ? "保存中" : option.label}
                    </button>
                  ))}
                  {!review.savedId && (
                    <span className="text-[11px] font-bold text-slate-400">経験保存後に評価できます</span>
                  )}
                </div>
              </>
            ) : (
              <p className="min-h-20 text-sm font-bold leading-7 text-violet-700">
                審査実行後に、紫苑がこの案件をレビューします。境界案件では、点数よりも「何を条件に残すか」を優先して見ます。
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildCurrentIssue(result: Record<string, any>, data: ScoringFormData) {
  const score = Number(result.score_base ?? result.score ?? 0);
  const isNewCustomer = String(data.customer_type || "").includes("新規");
  const hasNoLeaseHistory = Number(data.lease_credit || 0) <= 0 && Number(data.contracts || 0) <= 0;
  const hasCompetitor = data.competitor === "競合あり";
  const hasMainBank = data.main_bank === "メイン先";
  const aurionSeverity = String(result.aurion_core?.severity || "");
  const aurionFlags = Array.isArray(result.aurion_core?.discipline_flags)
    ? result.aurion_core.discipline_flags
    : [];

  if (score < 60) {
    if (isNewCustomer || hasNoLeaseHistory) {
      return "新規・実績薄めの案件を、保全条件と銀行支援で再設計できるか";
    }
    return "否決域のリスクを、条件変更で審議可能な形へ戻せるか";
  }

  if (score < 71) {
    if (hasCompetitor) {
      return "境界スコアで、競合条件に寄せすぎず承認条件を組めるか";
    }
    if (hasMainBank) {
      return "境界スコアだが、銀行支援と物件保全で条件付き承認に寄せられるか";
    }
    if (isNewCustomer) {
      return "新規先の不確実性を、確認条件でどこまで吸収できるか";
    }
    return "境界スコアを、追加確認と条件設定で承認側へ寄せられるか";
  }

  if (aurionFlags.includes("pricing_competition") || hasCompetitor) {
    return "承認域だが、競合条件に引っ張られず採算と稟議説明を守れるか";
  }
  if (["caution", "stop"].includes(aurionSeverity)) {
    return "点数は届くが、AURIONの違和感を稟議で説明できるか";
  }
  if (isNewCustomer || hasNoLeaseHistory) {
    return "承認域だが、新規先としての確認材料をどこまで揃えるか";
  }
  return "承認域の案件を、条件・採算・稟議説明まで崩さず通せるか";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CurrentIssueCard({ result, data }: { result: Record<string, any>; data: ScoringFormData }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-violet-50 text-violet-700">
          <BadgeInfo className="h-4 w-4" />
        </div>
        <div>
          <div className="text-[11px] font-black uppercase tracking-wider text-slate-400">今回の争点</div>
          <div className="mt-1 text-sm font-black leading-relaxed text-slate-900">
            {buildCurrentIssue(result, data)}
          </div>
        </div>
      </div>
    </section>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildRingiPolicy(result: Record<string, any>, data: ScoringFormData) {
  const score = Number(result.score_base ?? result.score ?? 0);
  const isNewCustomer = String(data.customer_type || "").includes("新規");
  const hasNoLeaseHistory = Number(data.lease_credit || 0) <= 0 && Number(data.contracts || 0) <= 0;
  const hasCompetitor = data.competitor === "競合あり";
  const hasMainBank = data.main_bank === "メイン先";
  const hasAsset = Boolean(data.asset_name || data.asset_purpose || data.asset_evidence_level);
  const aurionSeverity = String(result.aurion_core?.severity || "");

  if (score < 60) {
    if (hasMainBank || hasAsset) {
      return "稟議方針: 現状は否決域。銀行支援・物件保全・返済原資を追加確認し、条件再設計案として上申する。";
    }
    return "稟議方針: 現状条件では否決寄り。追加担保・保証・契約条件変更の余地を確認してから再審議する。";
  }

  if (score < 71) {
    const conditions = [
      hasAsset ? "物件保全" : "対象物件・用途確認",
      hasMainBank ? "銀行支援確認" : "返済原資確認",
      hasCompetitor ? "競合条件比較" : "",
    ].filter(Boolean);
    return `稟議方針: スコアは境界。${conditions.join("と")}を条件に、限定承認で組む。`;
  }

  if (hasCompetitor) {
    return "稟議方針: 承認域。競合条件との差分を整理し、採算を崩さない条件で上申する。";
  }
  if (["caution", "stop"].includes(aurionSeverity)) {
    return "稟議方針: 承認域だが、AURIONの警戒点を補足し、確認条件付きで上申する。";
  }
  if (isNewCustomer || hasNoLeaseHistory) {
    return "稟議方針: 承認域。新規先として取引背景・返済原資・物件保全を補足して上申する。";
  }
  return "稟議方針: 承認域。通常確認事項を押さえ、採算と取引継続性を根拠に上申する。";
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function RingiPolicyCard({ result, data }: { result: Record<string, any>; data: ScoringFormData }) {
  return (
    <section className="rounded-2xl border border-violet-200 bg-violet-50 px-4 py-3 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white text-violet-700 shadow-sm">
          <FileOutput className="h-4 w-4" />
        </div>
        <div>
          <div className="text-[11px] font-black uppercase tracking-wider text-violet-500">稟議に書くなら</div>
          <div className="mt-1 text-sm font-black leading-relaxed text-violet-950">
            {buildRingiPolicy(result, data)}
          </div>
        </div>
      </div>
    </section>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ScreeningLoopFeedbackPanel({ result, data }: { result: Record<string, any>; data: ScoringFormData }) {
  const [comment, setComment] = useState("");
  const [savingKey, setSavingKey] = useState("");
  const [savedKey, setSavedKey] = useState("");
  const [error, setError] = useState("");
  const issueText = buildCurrentIssue(result, data);
  const ringiPolicyText = buildRingiPolicy(result, data);

  const sendFeedback = async (target: "issue" | "ringi_policy", rating: string) => {
    const key = `${target}:${rating}`;
    setSavingKey(key);
    setSavedKey("");
    setError("");
    try {
      await apiClient.post("/api/screening-loop-feedback", {
        surface: "screening",
        target,
        rating,
        issue_text: issueText,
        ringi_policy_text: ringiPolicyText,
        comment,
        score: Number(result.score_base ?? result.score ?? 0),
        hantei: result.hantei ?? "",
        context: {
          customer_type: data.customer_type,
          main_bank: data.main_bank,
          competitor: data.competitor,
          has_lease_history: Number(data.lease_credit || 0) > 0 || Number(data.contracts || 0) > 0,
          has_asset_context: Boolean(data.asset_name || data.asset_purpose || data.asset_evidence_level),
        },
      });
      setSavedKey(key);
    } catch {
      setError("保存できませんでした");
    } finally {
      setSavingKey("");
    }
  };

  const buttonClass = "rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-[11px] font-black text-slate-700 shadow-sm transition hover:border-violet-200 hover:bg-violet-50 hover:text-violet-800 disabled:opacity-50";
  const activeClass = "border-emerald-200 bg-emerald-50 text-emerald-800";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-[11px] font-black uppercase tracking-wider text-slate-400">判断ループ</div>
          <div className="mt-1 text-sm font-black text-slate-900">紫苑の読みを、人間の判断で育てる</div>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-black text-slate-400">争点</span>
            {["合っている", "少し違う", "違う"].map((rating) => {
              const key = `issue:${rating}`;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => sendFeedback("issue", rating)}
                  disabled={Boolean(savingKey)}
                  className={`${buttonClass} ${savedKey === key ? activeClass : ""}`}
                >
                  {savingKey === key ? "保存中" : savedKey === key ? "保存済" : rating}
                </button>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-black text-slate-400">稟議</span>
            {["使える", "修正して使う", "使えない"].map((rating) => {
              const key = `ringi_policy:${rating}`;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => sendFeedback("ringi_policy", rating)}
                  disabled={Boolean(savingKey)}
                  className={`${buttonClass} ${savedKey === key ? activeClass : ""}`}
                >
                  {savingKey === key ? "保存中" : savedKey === key ? "保存済" : rating}
                </button>
              );
            })}
          </div>
        </div>
      </div>
      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <input
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          placeholder="人間メモ: 実際の争点・修正理由を一言"
          className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-700 outline-none focus:border-violet-300 focus:bg-white"
        />
        <div className="min-h-8 text-xs font-bold text-slate-400 sm:w-28 sm:py-2">
          {error || (savedKey ? "ループ保存済み" : "")}
        </div>
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

function AurionCoreCard({ core }: { core?: any }) {
  if (!core) return null;
  const severity = core.severity || "clear";
  const tone = core.emotion_synapse?.tone || "落ち着いた確認";
  const line = core.emotion_synapse?.shion_line || core.shion_ux_message || "";
  const flags = Array.isArray(core.discipline_flags) ? core.discipline_flags : [];
  const actions = Array.isArray(core.next_actions) ? core.next_actions : [];
  const styles: Record<string, string> = {
    clear: "border-emerald-200 bg-emerald-50 text-emerald-900",
    watch: "border-sky-200 bg-sky-50 text-sky-900",
    caution: "border-amber-200 bg-amber-50 text-amber-900",
    stop: "border-rose-200 bg-rose-50 text-rose-900",
  };
  return (
    <section className={`rounded-2xl border p-4 shadow-sm ${styles[severity] || styles.clear}`}>
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 flex-shrink-0" />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-black">AURION CORE 規律シナプス</h3>
              <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-black uppercase">
                {severity}
              </span>
              <span className="rounded-full bg-white/70 px-2 py-0.5 text-[10px] font-bold">
                {tone}
              </span>
            </div>
            <p className="mt-1 text-xs font-bold leading-relaxed">{core.shion_ux_message || line}</p>
            <p className="mt-1 text-[10px] font-bold opacity-70">
              Q_riskは自動減点ではなく、信用・価格・物件・銀行支援・営業プロセスを分けるための発見信号です。
            </p>
          </div>
        </div>
        {core.signals && (
          <div className="grid min-w-[220px] grid-cols-2 gap-2">
            {[
              ["Q_risk", core.signals.q_risk],
              ["警戒", core.emotion_synapse?.vigilance],
            ].map(([label, value]) => (
              <div key={String(label)} className="rounded-xl bg-white/70 px-3 py-2 text-center">
                <div className="text-[10px] font-black opacity-60">{String(label)}</div>
                <div className="text-lg font-black">{value != null ? Number(value).toFixed(1) : "-"}</div>
              </div>
            ))}
          </div>
        )}
      </div>
      {flags.length > 0 && (
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {flags.slice(0, 4).map((flag: any) => (
            <div key={flag.key || flag.title} className="rounded-xl border border-current/10 bg-white/70 px-3 py-2">
              <div className="text-xs font-black">{flag.title}</div>
              <div className="mt-1 text-[11px] font-medium leading-relaxed opacity-75">{flag.detail}</div>
            </div>
          ))}
        </div>
      )}
      {actions.length > 0 && (
        <div className="mt-3 rounded-xl bg-white/70 px-3 py-2">
          <div className="text-[10px] font-black uppercase opacity-60">Next Actions</div>
          <ul className="mt-1 space-y-1">
            {actions.slice(0, 4).map((action: string, index: number) => (
              <li key={index} className="flex gap-2 text-[11px] font-bold leading-relaxed">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-current opacity-50" />
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function BayesReverseStrategyCard({ strategy }: { strategy?: any }) {
  if (!strategy?.available) return null;
  const prior = Number(strategy.prior_percent ?? 0);
  const posterior = Number(strategy.posterior_percent ?? 0);
  const lift = Number(strategy.lift_percent ?? 0);
  const factors = Array.isArray(strategy.factors) ? strategy.factors : [];
  const moves = Array.isArray(strategy.moves) ? strategy.moves : [];
  const phrases = Array.isArray(strategy.phrases) ? strategy.phrases : [];
  const barWidth = Math.max(2, Math.min(100, posterior));
  const liftClass = lift >= 0 ? "text-emerald-700 bg-emerald-100 border-emerald-200" : "text-rose-700 bg-rose-100 border-rose-200";

  return (
    <section className="rounded-2xl border border-indigo-200 bg-indigo-50 p-4 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <Network className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-700" />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-black text-indigo-950">BN逆転・軍師提案</h3>
              <span className="rounded-full border border-indigo-300 bg-white px-2 py-0.5 text-[10px] font-black text-indigo-700">
                {strategy.stance || "Bayes"}
              </span>
              <span className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${liftClass}`}>
                {lift >= 0 ? "+" : ""}{lift.toFixed(1)}pt
              </span>
            </div>
            <p className="mt-1 text-xs font-bold leading-relaxed text-indigo-800">
              {strategy.headline || "証拠を足した時の承認余地をベイズ更新で見ます。"}
            </p>
          </div>
        </div>
        <div className="min-w-[240px] rounded-xl border border-indigo-100 bg-white px-4 py-3">
          <div className="flex items-end justify-between gap-3">
            <div>
              <div className="text-[10px] font-black text-slate-400">事前確率</div>
              <div className="text-lg font-black text-slate-700">{prior.toFixed(1)}%</div>
            </div>
            <ArrowRight className="mb-1 h-4 w-4 text-indigo-400" />
            <div className="text-right">
              <div className="text-[10px] font-black text-indigo-500">更新後</div>
              <div className="text-3xl font-black text-indigo-800">{posterior.toFixed(1)}%</div>
            </div>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
            <div className="h-full rounded-full bg-indigo-600" style={{ width: `${barWidth}%` }} />
          </div>
        </div>
      </div>

      {factors.length > 0 && (
        <div className="mt-3 grid gap-2 md:grid-cols-3">
          {factors.slice(1, 4).map((factor: any) => (
            <div key={factor.label} className="rounded-xl border border-indigo-100 bg-white px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-[11px] font-black text-slate-700">{factor.label}</div>
                <div className={`text-[11px] font-black ${Number(factor.delta_pct) >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                  {Number(factor.delta_pct) >= 0 ? "+" : ""}{Number(factor.delta_pct || 0).toFixed(1)}pt
                </div>
              </div>
              <div className="mt-1 text-[10px] font-medium leading-relaxed text-slate-500">{factor.detail}</div>
            </div>
          ))}
        </div>
      )}

      {moves.length > 0 && (
        <div className="mt-3 rounded-xl border border-indigo-100 bg-white px-3 py-2">
          <div className="text-[10px] font-black uppercase text-indigo-500">逆転の打ち手</div>
          <ul className="mt-1 space-y-1">
            {moves.slice(0, 4).map((move: string, index: number) => (
              <li key={index} className="flex gap-2 text-[11px] font-bold leading-relaxed text-slate-700">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-indigo-400" />
                <span>{move}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {phrases.length > 0 && (
        <div className="mt-3 rounded-xl bg-indigo-900 px-3 py-2 text-white">
          <div className="text-[10px] font-black uppercase text-indigo-200">軍師フレーズ</div>
          <div className="mt-1 text-xs font-bold leading-relaxed">{phrases[0]}</div>
        </div>
      )}
      {strategy.disclaimer && (
        <p className="mt-2 text-[10px] font-bold leading-relaxed text-indigo-500">{strategy.disclaimer}</p>
      )}
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
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [result, setResult] = useState<any>(null);
  const [formData, setFormData] = useState<ScoringFormData>(defaultFormData);
  const [gunshiText, setGunshiText] = useState<string>("");
  const [shionReview, setShionReview] = useState<ShionScreeningReview | null>(null);
  const [shionReviewLoading, setShionReviewLoading] = useState(false);
  const [shionReviewError, setShionReviewError] = useState("");
  const [shionFeedbackSaving, setShionFeedbackSaving] = useState(false);
  const shionReviewRequestSeq = useRef(0);

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

  useEffect(() => {
    const raw = window.localStorage.getItem(SCREENING_RETURN_STATE_KEY);
    if (!raw) return;
    try {
      const saved = JSON.parse(raw) as {
        formData?: ScoringFormData;
        result?: any;
        gunshiText?: string;
        shionReview?: ShionScreeningReview | null;
        activeTab?: "input" | "analysis";
      };
      if (saved.formData) setFormData(saved.formData);
      if (saved.result) setResult(saved.result);
      if (typeof saved.gunshiText === "string") setGunshiText(saved.gunshiText);
      if (saved.shionReview) setShionReview(saved.shionReview);
      setActiveTab(saved.activeTab || (saved.result ? "analysis" : "input"));
    } catch {
      window.localStorage.removeItem(SCREENING_RETURN_STATE_KEY);
    }
  }, []);

  // フィールドの変更ハンドラー
  const handleFieldChange = (name: string, value: string | number | string[]) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const buildShionReviewUserId = (targetResult: any, targetFormData: ScoringFormData) => {
    const rawId = String(targetResult?.case_id || targetFormData.company_no || targetFormData.company_name || "draft");
    const safeId = rawId.replace(/[^\w\-ぁ-んァ-ヶ一-龠ー]/g, "_").slice(0, 64);
    return `screening-shion-review:${safeId || "draft"}`;
  };

  const fetchPastShionReviews = async (targetResult: any, targetFormData: ScoringFormData) => {
    try {
      const industrySub = targetResult?.industry_sub || targetFormData.industry_sub || "";
      const res = await apiClient.get("/api/shion-screening-reviews", {
        params: {
          industry_sub: industrySub,
          limit: 3,
        },
      });
      return Array.isArray(res.data?.reviews) ? res.data.reviews as PastShionScreeningReview[] : [];
    } catch {
      return [];
    }
  };

  const saveShionScreeningReview = async (
    targetResult: any,
    targetFormData: ScoringFormData,
    promptText: string,
    review: ShionScreeningReview,
  ) => {
    const res = await apiClient.post("/api/shion-screening-reviews", {
      case_id: targetResult?.case_id || targetFormData.company_no || "",
      company_name: targetFormData.company_name || "",
      industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
      industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
      sales_dept: targetFormData.sales_dept || "",
      score: Number(targetResult?.score_base ?? targetResult?.score ?? 0),
      hantei: targetResult?.hantei || "",
      q_risk: targetResult?.quantum_risk ?? null,
      umap_anomaly_score: targetResult?.umap_anomaly_score ?? null,
      memory_refs: review.memoryRefs,
      knowledge_refs: review.knowledgeRefs,
      identity_used: review.identityUsed,
      review_text: review.reply,
      prompt_text: promptText,
      form_snapshot: targetFormData,
      result_snapshot: targetResult,
    });
    return Number(res.data?.review?.id || 0) || undefined;
  };

  const submitShionReviewFeedback = async (feedback: ShionReviewFeedback) => {
    if (!shionReview?.savedId || shionFeedbackSaving) return;
    const previous = shionReview.userFeedback;
    setShionReview((current) => current ? { ...current, userFeedback: feedback } : current);
    setShionFeedbackSaving(true);
    try {
      await apiClient.patch(`/api/shion-screening-reviews/${shionReview.savedId}/feedback`, {
        user_feedback: feedback,
      });
    } catch (error) {
      console.error("Shion review feedback save failed", error);
      setShionReview((current) => current ? { ...current, userFeedback: previous } : current);
      setShionReviewError("紫苑レビュー評価を保存できませんでした。");
    } finally {
      setShionFeedbackSaving(false);
    }
  };

  const requestShionReview = async (targetResult = result, targetFormData = formData) => {
    if (!targetResult) return;
    const seq = ++shionReviewRequestSeq.current;
    setShionReviewLoading(true);
    setShionReviewError("");
    try {
      const pastReviews = await fetchPastShionReviews(targetResult, targetFormData);
      if (seq !== shionReviewRequestSeq.current) return;
      const promptText = buildShionReviewPrompt(targetResult, targetFormData, pastReviews);
      const res = await apiClient.post("/api/chat", {
        message: promptText,
        user_id: buildShionReviewUserId(targetResult, targetFormData),
        response_mode: "shion",
        debug_memory: true,
      });
      if (seq !== shionReviewRequestSeq.current) return;
      const memoryDebug = res.data?.memory_debug || {};
      const memoryRecall = memoryDebug.memory_recall || {};
      const identityMemory = memoryDebug.identity_memory || {};
      const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0;
      const memoryRefs = Array.isArray(memoryRecall.refs) ? memoryRecall.refs.length : 0;
      const nextReview: ShionScreeningReview = {
        reply: String(res.data?.reply || "紫苑レビューが空でした。"),
        memoryRefs,
        knowledgeRefs,
        identityUsed: Boolean(identityMemory.used),
      };
      setShionReview(nextReview);
      saveShionScreeningReview(targetResult, targetFormData, promptText, nextReview)
        .then((savedId) => {
          if (!savedId || seq !== shionReviewRequestSeq.current) return;
          setShionReview((current) => current ? { ...current, savedId } : current);
        })
        .catch((error) => {
          console.warn("Shion screening review save failed", error);
        });
    } catch (error) {
      if (seq !== shionReviewRequestSeq.current) return;
      console.error("Shion review error", error);
      setShionReviewError("紫苑レビューを取得できませんでした。APIサーバーまたはAIチャットの状態を確認してください。");
    } finally {
      if (seq === shionReviewRequestSeq.current) {
        setShionReviewLoading(false);
      }
    }
  };

  const resetScreening = () => {
    shionReviewRequestSeq.current += 1;
    setFormData(defaultFormData);
    setResult(null);
    setGunshiText("");
    setShionReview(null);
    setShionReviewLoading(false);
    setShionReviewError("");
    setShionFeedbackSaving(false);
    setActiveTab("input");
    window.localStorage.removeItem(SCREENING_RETURN_STATE_KEY);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setShionReview(null);
    setShionReviewError("");
    try {
      const res = await apiClient.post(`/api/score/full`, toThousandYenPayload(formData));
      setResult(res.data);
      setActiveTab("analysis");
      void requestShionReview(res.data, formData);

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

  const handoffToShionChat = () => {
    if (!result) return;
    const chatContext = {
      score: result.score_base,
      hantei: result.hantei,
      score_borrower: result.score_borrower,
      company_name: formData.company_name,
      asset_name: formData.asset_name,
      asset_location: formData.asset_location,
      prefecture: result.prefecture || "",
      industry_sub: result.industry_sub || formData.industry_sub,
      industry_major: result.industry_major || formData.industry_major,
      sales_dept: formData.sales_dept,
      quantum_risk: result.quantum_risk,
      case_id: result.case_id,
    };
    window.localStorage.setItem(SCREENING_RETURN_STATE_KEY, JSON.stringify({
      formData,
      result,
      gunshiText,
      shionReview,
      activeTab: "analysis",
    }));
    window.localStorage.setItem("lease-gunshi-context", JSON.stringify(chatContext));
    router.push("/chat");
  };

  const handoffToShionDebate = () => {
    if (!result) return;
    const score = Number(result.score_base ?? result.score ?? 0);
    const debateContext = {
      score,
      hantei: result.hantei,
      company_name: formData.company_name,
      industry_major: result.industry_major || formData.industry_major,
      nenshu: formData.nenshu,
      op_margin_pct: result.user_op_margin ?? (formData.nenshu ? (formData.op_profit / formData.nenshu) * 100 : 0),
      equity_ratio: result.user_equity_ratio ?? (formData.total_assets ? (formData.net_assets / formData.total_assets) * 100 : 0),
      bank_credit: formData.bank_credit,
      lease_credit: formData.lease_credit,
      asset_name: formData.asset_name,
      lease_amount: formData.acquisition_cost,
      reason: "screening_handoff",
    };
    window.localStorage.setItem("lease-debate-context", JSON.stringify(debateContext));
    router.push("/debate");
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
          onClick={resetScreening}
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
          <Link href="/screening" className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
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
                    <JudgmentFlowStrip />
                    <CurrentIssueCard result={result} data={formData} />
                    <RingiPolicyCard result={result} data={formData} />
                    <ScreeningLoopFeedbackPanel result={result} data={formData} />
                    <IndicatorCards data={result} />
                    <ShionScreeningReviewCard
                      review={shionReview}
                      loading={shionReviewLoading}
                      error={shionReviewError}
                      onReview={() => requestShionReview()}
                      onFeedback={submitShionReviewFeedback}
                      feedbackSaving={shionFeedbackSaving}
                    />

                    <div className="rounded-2xl border border-violet-200 bg-violet-50 p-4 shadow-sm">
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                          <h3 className="flex items-center gap-2 text-sm font-black text-violet-900">
                            <MessageSquare className="h-4 w-4" />
                            この審査結果を紫苑へ渡す
                          </h3>
                          <p className="mt-1 text-xs font-bold leading-relaxed text-violet-700">
                            通常相談は紫苑チャットへ。要審議・境界案件は、懐疑派・楽観派・統合派のマルチ紫苑討論に回します。
                          </p>
                        </div>
                        <div className="flex flex-col gap-2 sm:flex-row">
                          <button
                            type="button"
                            onClick={handoffToShionChat}
                            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-xs font-black text-white transition hover:bg-violet-700"
                          >
                            紫苑へ相談
                            <ArrowRight className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={handoffToShionDebate}
                            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-black text-white transition hover:bg-slate-800"
                          >
                            マルチ紫苑で討論
                            <Swords className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>

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
                        <p className="text-[10px] text-rose-400 mt-2">財務パターンを学習済みMLモデルで判定。審査スコアとは独立した補助指標です。</p>
                      </div>
                    )}

                    <AurionCoreCard core={result.aurion_core} />

                    <BayesReverseStrategyCard strategy={result.bayes_reverse_strategy} />

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
