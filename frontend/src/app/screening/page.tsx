"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";
import { openKnowledgeSpaceFocus } from "@/lib/knowledgeSpaceRoute";
import { Activity, ArrowRight, Calculator, Eye, MessageSquare, Network, PieChart, AlignLeft, Share2, AlertTriangle, ListOrdered, BadgeInfo, DollarSign, Database, ChevronDown, ChartNoAxesCombined, SlidersHorizontal, ScanText, ShieldCheck, XCircle, Minus, Swords, Save, Trash2, Sparkles, Brain, Search, Copy } from "lucide-react";
import ScoreDAG from "../../components/ScoreDAG";
import { ScoringFormData, defaultFormData } from "../../types";
import FormGeneral from "../../components/form/FormGeneral";
import FormFinancial from "../../components/form/FormFinancial";
import FormQualitative from "../../components/form/FormQualitative";
import { toThousandYenPayload, fromThousandYenPayload } from "../../lib/scoringUnits";

import IndicatorCards from "../../components/analysis/IndicatorCards";
import { getScreeningScore, CurrentIssueCard, RingiPolicyCard, buildCurrentIssue, buildRingiPolicy } from "../../components/analysis/IssuePolicyCards";
import RealGraphs from "../../components/analysis/RealGraphs";
import GunshiAdvice from "../../components/analysis/GunshiAdvice";
import LeasePaymentSimulator from "../../components/analysis/LeasePaymentSimulator";
import QRiskPanel from "../../components/analysis/QRiskPanel";
import MahalanobisPanel from "../../components/analysis/MahalanobisPanel";
import UMAPPanel from "../../components/analysis/UMAPPanel";
import OcrUpload from "../../components/analysis/OcrUpload";
import ShionGrowthBadge from "../../components/analysis/ShionGrowthBadge";
import { ShionScreeningReviewCard } from "../../components/analysis/ShionReviewCard";
import { triggerMebuki } from "../../components/layout/FloatingMebuki";
import {
  isCanonicalJudgmentAsset,
  normalizeReviewText,
  validPastCompanyName,
  uniquePastCompanyHighlights,
  ensurePastCompanyMentionInReview,
  ensureCandidateJudgmentAssetMentionInReview,
  buildShionReviewPrompt,
  buildShionReviewFallback,
  parseExperienceSnapshot,
  normalizeExperienceCase,
  buildExperienceCaseQuery,
  hasExperienceSearchContext,
  buildShionReviewUserId,
  type ShionScreeningReview,
  type ShionReviewFeedback,
  type JudgmentAssetCandidateFeedback,
  type JudgmentAssetAdaptationMode,
  type JudgmentAssetCandidate,
  type PastCompanyHighlight,
  type PastShionScreeningReview,
  type DemoSimilarPastCase,
} from "../../lib/shionReview";

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

const JUDGMENT_ASSET_TYPE_TONES: Record<string, {
  label: string;
  card: string;
  badge: string;
  accent: string;
}> = {
  confirmation_question: {
    label: "確認質問",
    card: "border-sky-200 bg-sky-50/80",
    badge: "bg-sky-600 text-white",
    accent: "bg-sky-500",
  },
  condition_signal: {
    label: "条件兆候",
    card: "border-amber-200 bg-amber-50/80",
    badge: "bg-amber-600 text-white",
    accent: "bg-amber-500",
  },
  application_rule: {
    label: "適用ルール",
    card: "border-emerald-200 bg-emerald-50/80",
    badge: "bg-emerald-600 text-white",
    accent: "bg-emerald-500",
  },
  caution: {
    label: "反証",
    card: "border-rose-200 bg-rose-50/80",
    badge: "bg-rose-600 text-white",
    accent: "bg-rose-500",
  },
};

const JUDGMENT_ASSET_FEEDBACK_TONES: Record<JudgmentAssetCandidateFeedback | "none", {
  label: string;
  badge: string;
  ring: string;
}> = {
  useful: {
    label: "採用候補",
    badge: "bg-emerald-100 text-emerald-800",
    ring: "ring-2 ring-emerald-300",
  },
  neutral: {
    label: "修正中",
    badge: "bg-indigo-100 text-indigo-800",
    ring: "ring-2 ring-indigo-200",
  },
  rejected: {
    label: "見送り",
    badge: "bg-slate-200 text-slate-700",
    ring: "ring-1 ring-slate-200",
  },
  none: {
    label: "未評価",
    badge: "bg-white text-slate-600",
    ring: "ring-1 ring-white/70",
  },
};

const getJudgmentAssetTypeTone = (type: string) =>
  JUDGMENT_ASSET_TYPE_TONES[type] || {
    label: type || "候補",
    card: "border-violet-200 bg-violet-50/80",
    badge: "bg-violet-600 text-white",
    accent: "bg-violet-500",
  };

const SCREENING_RETURN_STATE_KEY = "lease-screening-return-state";
const SCREENING_DRAFT_VERSION = 1;
const SCREENING_DRAFT_SAVE_DELAY_MS = 300;

const SCREENING_COPY_EXCLUDED_FIELDS = new Set<keyof ScoringFormData>([
  "company_no",
  "company_name",
]);

const SCREENING_COPY_FIELD_ORDER: (keyof ScoringFormData)[] = [
  "industry_major",
  "industry_sub",
  "industry_detail",
  "grade",
  "trend_grade_t0",
  "trend_grade_t1",
  "trend_grade_t2",
  "customer_type",
  "main_bank",
  "competitor",
  "competitor_rate",
  "num_competitors",
  "deal_occurrence",
  "deal_source",
  "sales_dept",
  "nenshu",
  "gross_profit",
  "op_profit",
  "ord_profit",
  "net_income",
  "depreciation",
  "dep_expense",
  "rent",
  "rent_expense",
  "machines",
  "other_assets",
  "net_assets",
  "total_assets",
  "bank_credit",
  "lease_credit",
  "contracts",
  "contract_type",
  "lease_term",
  "acceptance_year",
  "acquisition_cost",
  "selected_asset_id",
  "asset_name",
  "asset_detail",
  "asset_purpose",
  "asset_location",
  "asset_evidence_level",
  "asset_score",
  "qual_corr_company_history",
  "qual_corr_customer_stability",
  "qual_corr_repayment_history",
  "qual_corr_business_future",
  "qual_corr_equipment_purpose",
  "qual_corr_main_bank",
  "passion_text",
  "strength_tags",
  "intuition",
];

const SCREENING_COPY_CONFIRM_FIELDS = new Set<keyof ScoringFormData>([
  "nenshu",
  "gross_profit",
  "op_profit",
  "ord_profit",
  "net_income",
  "net_assets",
  "total_assets",
  "bank_credit",
  "lease_credit",
  "contracts",
  "lease_term",
  "acquisition_cost",
  "asset_name",
  "competitor_rate",
]);

const SCREENING_FIELD_LABELS: Partial<Record<keyof ScoringFormData, string>> = {
  ...DATA_SOURCE_FIELD_LABELS,
  industry_detail: "詳細キーワード",
  trend_grade_t0: "今期格付",
  trend_grade_t1: "前期格付",
  trend_grade_t2: "前々期格付",
  num_competitors: "競合社数",
  deal_occurrence: "発生経緯",
  gross_profit: "売上総利益",
  depreciation: "減価償却費（資産）",
  dep_expense: "減価償却費（経費）",
  rent: "賃借料（資産）",
  rent_expense: "賃借料（経費）",
  machines: "機械装置・運搬具",
  other_assets: "その他固定資産",
  acceptance_year: "検収年",
  qual_corr_company_history: "業歴",
  qual_corr_customer_stability: "顧客基盤",
  qual_corr_repayment_history: "返済履歴",
  qual_corr_business_future: "事業将来性",
  qual_corr_equipment_purpose: "設備目的",
  qual_corr_main_bank: "メイン行姿勢",
  strength_tags: "強みタグ",
};

type ScreeningCopyDiff = {
  key: keyof ScoringFormData;
  label: string;
  before: string;
  after: string;
  needsConfirmation: boolean;
};

const formatScreeningCopyValue = (value: unknown): string => {
  if (Array.isArray(value)) return value.join(", ") || "未入力";
  if (value === undefined || value === null || value === "") return "未入力";
  return String(value);
};

const buildScreeningCopyDiffs = (
  current: ScoringFormData,
  snapshot: Partial<Record<keyof ScoringFormData, unknown>>,
): ScreeningCopyDiff[] => {
  return SCREENING_COPY_FIELD_ORDER.flatMap((key) => {
    if (SCREENING_COPY_EXCLUDED_FIELDS.has(key) || !(key in snapshot)) return [];
    const beforeRaw = current[key];
    const afterRaw = snapshot[key];
    const before = formatScreeningCopyValue(beforeRaw);
    const after = formatScreeningCopyValue(afterRaw);
    if (before === after) return [];
    return [{
      key,
      label: SCREENING_FIELD_LABELS[key] || key,
      before,
      after,
      needsConfirmation: SCREENING_COPY_CONFIRM_FIELDS.has(key),
    }];
  });
};

const getScreeningErrorMessage = (error: unknown) => {
  const err = error as {
    response?: { status?: number; data?: { detail?: unknown } };
    code?: string;
    message?: string;
  };
  const status = err.response?.status;
  const detail = err.response?.data?.detail;
  const detailText = typeof detail === "string" ? detail : "";

  if (!err.response || err.code === "ERR_NETWORK") {
    return "審査サーバーに一時的に接続できませんでした。数秒後にもう一度実行してください。";
  }
  if (status === 502 || status === 503 || status === 504 || status === 530) {
    return "審査サーバーが再起動または混雑中です。数秒後にもう一度実行してください。";
  }
  if (status && status >= 500) {
    return detailText
      ? `審査エンジンの実行に失敗しました。${detailText.slice(0, 180)}`
      : "審査エンジンの実行に失敗しました。入力内容を確認して、もう一度実行してください。";
  }
  return detailText || "審査を実行できませんでした。入力内容を確認してください。";
};

const getResultSnapshotScore = (snapshot?: Record<string, any>, fallback = 0) =>
  Number(snapshot?.score ?? snapshot?.score_base ?? fallback);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function AiHeroCard({
  result,
  data,
  onOpenKnowledge,
}: {
  result?: Record<string, any>;
  data?: Partial<ScoringFormData>;
  onOpenKnowledge?: () => void;
}) {
  if (!result) return null;
  const score = getScreeningScore(result);
  const hantei: string = result.hantei ?? "";
  const approvalLine: number = typeof result.approval_line === "number" ? result.approval_line : 71;
  const isApproved = score >= approvalLine;
  const isConditional = score >= 60 && score < approvalLine;

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
            承認ライン: {approvalLine}点以上
          </div>
          {onOpenKnowledge && (
            <button
              type="button"
              onClick={onOpenKnowledge}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl border border-white/25 bg-white/15 px-3 py-2 text-[11px] font-black text-white backdrop-blur-sm transition hover:bg-white/25"
            >
              <Network className="h-3.5 w-3.5" />
              関連知識を見る
            </button>
          )}
        </div>
      </div>
      {data && (
        <div className="mt-3 text-[11px] font-bold text-white/60">
          {[data.industry_sub || data.industry_major, data.asset_name, result.quantum_risk != null ? "Q_risk" : ""].filter(Boolean).join(" / ")}
        </div>
      )}
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


function JudgmentAssetCandidateCard({
  candidates,
  loading,
  feedbackSavingId,
  adaptationMode,
  onAdaptationModeChange,
  onFeedback,
  onCreate,
}: {
  candidates: JudgmentAssetCandidate[];
  loading: boolean;
  feedbackSavingId: string;
  adaptationMode: JudgmentAssetAdaptationMode;
  onAdaptationModeChange: (mode: JudgmentAssetAdaptationMode) => void;
  onFeedback: (candidate: JudgmentAssetCandidate, feedback: JudgmentAssetCandidateFeedback, editedClaim?: string) => void;
  onCreate: (claim: string, candidateType: string) => void;
}) {
  const [editingId, setEditingId] = useState("");
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [newClaim, setNewClaim] = useState("");
  const [newType, setNewType] = useState("confirmation_question");
  const labels: Record<JudgmentAssetCandidateFeedback, string> = {
    useful: "効いた",
    neutral: "微妙",
    rejected: "外した",
  };
  const adaptationLabels: Record<JudgmentAssetAdaptationMode, string> = {
    conservative: "保守的",
    standard: "標準",
    exploratory: "探索的",
    aggressive: "攻め",
  };
  const adaptationHints: Record<JudgmentAssetAdaptationMode, string> = {
    conservative: "教えた判断に近い範囲だけで使う",
    standard: "今回案件に合わせて少し変形する",
    exploratory: "関連する判断仮説を1つまで広げる",
    aggressive: "派生仮説も出すが断定しない",
  };

  return (
    <section className="rounded-2xl border border-amber-200 bg-amber-50/70 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-black text-amber-950">
            <Sparkles className="h-4 w-4 text-amber-600" />
            今回レビューに渡す判断資産
          </h3>
          <p className="mt-1 text-xs font-bold leading-relaxed text-amber-800">
            紫苑レビューへ最大3件だけ渡します。昇格済み判断資産と候補を、今回案件に合わせて使い分けます。
          </p>
          <p className="mt-1 text-[11px] font-bold leading-relaxed text-amber-700">
            役に立った/修正/違うの評価が、次の判断資産更新へ戻ります。
          </p>
        </div>
        {loading && <Activity className="h-4 w-4 animate-spin text-amber-700" />}
      </div>
      <div className="mt-3 rounded-xl border border-amber-100 bg-white p-3">
        <div className="mb-3">
          <p className="text-xs font-black text-amber-900">判断資産の発展度</p>
          <div className="mt-2 grid grid-cols-2 gap-1 md:grid-cols-4">
            {(Object.keys(adaptationLabels) as JudgmentAssetAdaptationMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => onAdaptationModeChange(mode)}
                className={`rounded-lg border px-2 py-2 text-left transition ${
                  adaptationMode === mode
                    ? "border-amber-400 bg-amber-100 text-amber-950"
                    : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                <span className="block text-[11px] font-black">{adaptationLabels[mode]}</span>
                <span className="mt-0.5 block text-[10px] font-bold leading-snug">{adaptationHints[mode]}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs font-black text-amber-900">新しい判断を追加</p>
          <select
            value={newType}
            onChange={(event) => setNewType(event.target.value)}
            className="rounded-lg border border-amber-100 bg-amber-50 px-2 py-1 text-xs font-bold text-amber-900 outline-none"
          >
            <option value="confirmation_question">確認質問</option>
            <option value="condition_signal">条件兆候</option>
            <option value="application_rule">適用ルール</option>
            <option value="caution">反証</option>
          </select>
        </div>
        <textarea
          value={newClaim}
          onChange={(event) => setNewClaim(event.target.value)}
          rows={3}
          placeholder="例: 運送業の増車なら、荷主契約の期間とドライバー確保をセットで確認する。"
          className="mt-2 w-full rounded-lg border border-amber-200 bg-amber-50/40 p-2 text-sm font-bold leading-6 text-slate-800 outline-none focus:border-amber-400 focus:bg-white"
        />
        <button
          type="button"
          onClick={() => {
            const value = newClaim.trim();
            if (!value) return;
            onCreate(value, newType);
            setNewClaim("");
          }}
          disabled={feedbackSavingId === "manual-create" || newClaim.trim().length < 8}
          className="mt-2 inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-black text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" />
          判断を追加
        </button>
      </div>
      <div className="mt-3 grid gap-2">
        {candidates.length ? candidates.map((candidate) => (
          <div
            key={candidate.id}
            className={`relative overflow-hidden rounded-xl border p-3 shadow-sm ${getJudgmentAssetTypeTone(candidate.candidate_type).card} ${
              JUDGMENT_ASSET_FEEDBACK_TONES[candidate.userFeedback || "none"].ring
            }`}
          >
            {(() => {
              const typeTone = getJudgmentAssetTypeTone(candidate.candidate_type);
              const feedbackTone = JUDGMENT_ASSET_FEEDBACK_TONES[candidate.userFeedback || "none"];
              const displayClaim = candidate.edited_claim || candidate.effective_claim || candidate.claim;
              const draft = drafts[candidate.id] ?? displayClaim;
              const isEditing = editingId === candidate.id;
              const isChanged = draft.trim() && draft.trim() !== displayClaim.trim();
              const isCanonical = isCanonicalJudgmentAsset(candidate);
              return (
                <>
            <div className={`absolute inset-y-0 left-0 w-1.5 ${typeTone.accent}`} />
            <div className="flex flex-wrap items-center gap-2 pl-1 text-[10px] font-black">
              <span className="rounded-full bg-slate-900 px-2 py-1 text-white">JA-{candidate.id.slice(0, 8)}</span>
              {isCanonical && <span className="rounded-full bg-emerald-600 px-2 py-1 text-white">正規判断資産</span>}
              {!isCanonical && <span className="rounded-full bg-amber-500 px-2 py-1 text-white">昇格候補</span>}
              <span className={`rounded-full px-2 py-1 ${typeTone.badge}`}>{typeTone.label}</span>
              {!isCanonical && <span className={`rounded-full px-2 py-1 ${feedbackTone.badge}`}>{feedbackTone.label}</span>}
              <span className="rounded-full bg-white/80 px-2 py-1 text-slate-700">{candidate.research_topic}</span>
              <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">use {candidate.use_count}</span>
              <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">useful {candidate.useful_count}</span>
              {candidate.edit_count ? <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700">修正 {candidate.edit_count}</span> : null}
            </div>
            {isEditing ? (
              <div className="mt-2 pl-1">
                <textarea
                  value={draft}
                  onChange={(event) => setDrafts((current) => ({ ...current, [candidate.id]: event.target.value }))}
                  rows={4}
                  className="w-full rounded-lg border border-amber-200 bg-amber-50/40 p-2 text-sm font-bold leading-6 text-slate-800 outline-none focus:border-amber-400 focus:bg-white"
                />
                <div className="mt-2 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => onFeedback(candidate, "neutral", draft.trim())}
                    disabled={feedbackSavingId === candidate.id || !isChanged}
                    className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-black text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Save className="h-3.5 w-3.5" />
                    修正を保存
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setDrafts((current) => ({ ...current, [candidate.id]: displayClaim }));
                      setEditingId("");
                    }}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-[11px] font-black text-slate-600 transition hover:bg-slate-50"
                  >
                    閉じる
                  </button>
                </div>
              </div>
            ) : (
              <p className="mt-2 pl-1 text-sm font-bold leading-6 text-slate-900">{displayClaim}</p>
            )}
            <p className="mt-2 break-all pl-1 text-[10px] font-bold text-slate-600">
              出典: {candidate.evidence_path || "manual"} / 元ID: {candidate.id}
            </p>
            <div className="mt-3 flex flex-wrap gap-2 pl-1">
              {isCanonical ? (
                <span className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-black text-emerald-700">
                  昇格済みのため昇格対象外
                </span>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => {
                      setEditingId(candidate.id);
                      setDrafts((current) => ({ ...current, [candidate.id]: current[candidate.id] ?? displayClaim }));
                    }}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-[11px] font-black text-slate-700 transition hover:bg-slate-50"
                  >
                    文面修正
                  </button>
                  {(Object.keys(labels) as JudgmentAssetCandidateFeedback[]).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onFeedback(candidate, key)}
                      disabled={feedbackSavingId === candidate.id}
                      className={`rounded-lg border px-3 py-1.5 text-[11px] font-black transition disabled:cursor-not-allowed disabled:opacity-50 ${
                        candidate.userFeedback === key
                          ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                          : "border-white/80 bg-white/80 text-slate-700 hover:bg-white"
                      }`}
                    >
                      {feedbackSavingId === candidate.id && candidate.userFeedback === key ? "保存中" : labels[key]}
                    </button>
                  ))}
                </>
              )}
            </div>
                </>
              );
            })()}
          </div>
        )) : (
          <p className="rounded-xl border border-amber-100 bg-white p-3 text-sm font-bold text-amber-800">
            この案件に合う候補はまだありません。紫苑レビューは通常の記憶・知識だけで生成します。
          </p>
        )}
      </div>
    </section>
  );
}

function InputJudgmentAssetPreview({
  candidates,
  loading,
  searched,
  onSearch,
}: {
  candidates: JudgmentAssetCandidate[];
  loading: boolean;
  searched: boolean;
  onSearch: () => void;
}) {
  return (
    <section className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-amber-50 text-amber-600">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-black text-slate-800">入力中の確認観点</h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-500">
              業種・物件・導入目的から、今回先に見ておく判断資産を表示します。スコアや承認判定は変えません。
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onSearch}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm font-black text-amber-800 transition hover:bg-amber-100 disabled:opacity-50"
        >
          {loading ? <Activity className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          観点を探す
        </button>
      </div>

      {candidates.length > 0 && (
        <div className="mt-4 grid gap-3 xl:grid-cols-3">
          {candidates.map((candidate) => {
            const typeTone = getJudgmentAssetTypeTone(candidate.candidate_type);
            const displayClaim = candidate.edited_claim || candidate.effective_claim || candidate.claim;
            const isCanonical = isCanonicalJudgmentAsset(candidate);
            return (
              <article
                key={candidate.id}
                className={`relative overflow-hidden rounded-xl border p-3 ${typeTone.card}`}
              >
                <div className={`absolute inset-y-0 left-0 w-1.5 ${typeTone.accent}`} />
                <div className="flex flex-wrap items-center gap-1.5 pl-1 text-[10px] font-black">
                  <span className="rounded-full bg-slate-900 px-2 py-1 text-white">JA-{candidate.id.slice(0, 8)}</span>
                  <span className={`rounded-full px-2 py-1 ${typeTone.badge}`}>{typeTone.label}</span>
                  <span className={`rounded-full px-2 py-1 ${isCanonical ? "bg-emerald-600 text-white" : "bg-amber-500 text-white"}`}>
                    {isCanonical ? "正規" : "候補"}
                  </span>
                </div>
                <p className="mt-2 line-clamp-4 pl-1 text-sm font-bold leading-6 text-slate-900">
                  {displayClaim}
                </p>
                <p className="mt-2 truncate pl-1 text-[10px] font-bold text-slate-600">
                  {candidate.research_topic || candidate.evidence_path || "screening"}
                </p>
              </article>
            );
          })}
        </div>
      )}

      {searched && !loading && candidates.length === 0 && (
        <div className="mt-4 rounded-xl border border-dashed border-amber-200 bg-amber-50/40 px-3 py-3 text-xs font-bold text-amber-800">
          この入力に合う判断資産はまだありません。業種・物件・導入目的を入れると候補が出やすくなります。
        </div>
      )}
    </section>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const formatExperienceValue = (value: unknown) => {
  if (value === null || value === undefined || value === "") return "未記録";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString("ja-JP") : "未記録";
  if (Array.isArray(value)) return value.length ? value.join(" / ") : "未記録";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const pickExperienceValue = (
  primary: Record<string, any> | undefined,
  fallback: Record<string, any>,
  keys: string[],
) => {
  for (const key of keys) {
    const value = primary?.[key] ?? fallback[key];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return "";
};

function ExperienceCaseDetailModal({
  item,
  data,
  onClose,
}: {
  item: DemoSimilarPastCase;
  data: ScoringFormData;
  onClose: () => void;
}) {
  const currentData = data as unknown as Record<string, any>;
  const formRows = [
    ["企業番号", pickExperienceValue(item.formSnapshot, currentData, ["company_no"])],
    ["営業部", pickExperienceValue(item.formSnapshot, currentData, ["sales_dept"])],
    ["取引区分", pickExperienceValue(item.formSnapshot, currentData, ["customer_type"])],
    ["メイン先", pickExperienceValue(item.formSnapshot, currentData, ["main_bank"])],
    ["競合", pickExperienceValue(item.formSnapshot, currentData, ["competitor"])],
    ["物件", pickExperienceValue(item.formSnapshot, currentData, ["asset_name", "asset_detail"])],
    ["取得価額", pickExperienceValue(item.formSnapshot, currentData, ["acquisition_cost"])],
    ["リース期間", pickExperienceValue(item.formSnapshot, currentData, ["lease_term", "lease_term_months"])],
  ];
  const resultRows = [
    ["総合スコア", getResultSnapshotScore(item.resultSnapshot, item.score)],
    ["判定", item.resultSnapshot?.hantei ?? item.decision],
    ["Q_risk", item.resultSnapshot?.quantum_risk],
    ["UMAP異常度", item.resultSnapshot?.umap_anomaly_score],
  ];
  const hasSnapshot = Boolean(item.formSnapshot || item.resultSnapshot);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-3 py-6 backdrop-blur-md" onClick={onClose}>
      <div
        className="relative max-h-[88vh] w-full max-w-5xl overflow-hidden rounded-xl border border-cyan-300/40 bg-slate-950 text-cyan-50 shadow-2xl shadow-cyan-950/60"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={`${item.companyName} の経験ケース詳細`}
      >
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.06)_1px,transparent_1px)] bg-[size:22px_22px]" />
        <div className="pointer-events-none absolute left-0 right-0 top-0 h-px bg-cyan-200/80 shadow-[0_0_20px_rgba(34,211,238,0.9)]" />

        <div className="relative flex items-start justify-between gap-4 border-b border-cyan-300/25 bg-slate-900/90 px-4 py-3">
          <div>
            <div className="text-[10px] font-black uppercase tracking-[0.24em] text-cyan-300">CASE MEMORY TRACE</div>
            <h3 className="mt-1 text-xl font-black text-white">{item.companyName}</h3>
            <p className="mt-1 text-xs font-bold text-cyan-100/80">{item.period || "時期未記録"} / {item.industry || "業種未記録"}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-cyan-300/30 bg-cyan-300/10 p-2 text-cyan-100 transition hover:bg-cyan-300/20"
            aria-label="閉じる"
          >
            <XCircle className="h-5 w-5" />
          </button>
        </div>

        <div className="relative max-h-[calc(88vh-76px)] overflow-y-auto p-4">
          <div className="grid gap-3 lg:grid-cols-[1fr_0.85fr]">
            <div className="rounded-lg border border-cyan-300/25 bg-slate-900/75 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-md border border-cyan-300/30 bg-cyan-300/10 px-2 py-1 text-[10px] font-black text-cyan-100">当時 {item.score.toFixed(1)}点</span>
                {!!item.similarityScore && (
                  <span className="rounded-md border border-emerald-300/30 bg-emerald-300/10 px-2 py-1 text-[10px] font-black text-emerald-100">
                    類似度 {Math.round(item.similarityScore)}
                  </span>
                )}
                <span className="rounded-md border border-violet-300/30 bg-violet-300/10 px-2 py-1 text-[10px] font-black text-violet-100">
                  {item.source || "experience"}
                </span>
              </div>
              {!!item.similarityReasons?.length && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {item.similarityReasons.map((reason) => (
                    <span key={reason} className="rounded-full border border-cyan-300/25 px-2 py-0.5 text-[10px] font-bold text-cyan-100/90">
                      {reason}
                    </span>
                  ))}
                </div>
              )}
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div>
                  <div className="text-[10px] font-black text-cyan-300/80">判断</div>
                  <div className="mt-1 text-sm font-black text-white">{item.decision || "未記録"}</div>
                  <div className="mt-1 text-xs font-bold leading-relaxed text-cyan-100/70">{item.outcome || "結果未記録"}</div>
                </div>
                <div>
                  <div className="text-[10px] font-black text-cyan-300/80">似ている点</div>
                  <div className="mt-1 text-xs font-bold leading-relaxed text-cyan-50">{item.similarity || "未記録"}</div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-cyan-300/25 bg-slate-900/75 p-3">
              <div className="text-[10px] font-black uppercase tracking-[0.18em] text-cyan-300">DATA LAYER</div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                {resultRows.map(([label, value]) => (
                  <div key={label} className="border border-cyan-300/15 bg-slate-950/70 px-2 py-2">
                    <div className="text-[10px] font-black text-cyan-300/70">{label}</div>
                    <div className="mt-1 break-words text-xs font-black text-white">{formatExperienceValue(value)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-3 grid gap-3 lg:grid-cols-3">
            <div className="rounded-lg border border-emerald-300/25 bg-emerald-950/30 p-3">
              <div className="text-[10px] font-black text-emerald-200">当時こうした</div>
              <div className="mt-2 text-xs font-bold leading-relaxed text-emerald-50">{item.actionTaken || "未記録"}</div>
            </div>
            <div className="rounded-lg border border-amber-300/25 bg-amber-950/25 p-3">
              <div className="text-[10px] font-black text-amber-200">得た教訓</div>
              <div className="mt-2 text-xs font-bold leading-relaxed text-amber-50">{item.lesson || "未記録"}</div>
            </div>
            <div className="rounded-lg border border-blue-300/25 bg-blue-950/25 p-3">
              <div className="text-[10px] font-black text-blue-200">今回との違い</div>
              <div className="mt-2 text-xs font-bold leading-relaxed text-blue-50">{item.difference || "未記録"}</div>
            </div>
          </div>

          <div className="mt-3 rounded-lg border border-cyan-300/25 bg-slate-900/75 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[10px] font-black uppercase tracking-[0.18em] text-cyan-300">INPUT SNAPSHOT</div>
                <div className="mt-1 text-[11px] font-bold text-cyan-100/60">
                  {hasSnapshot ? "保存時の主要入力を復元" : "詳細スナップショット未保存のため、現在入力と経験ケース概要から表示"}
                </div>
              </div>
              <Eye className="h-4 w-4 text-cyan-300" />
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {formRows.map(([label, value]) => (
                <div key={label} className="border border-cyan-300/15 bg-slate-950/60 px-2 py-2">
                  <div className="text-[10px] font-black text-cyan-300/70">{label}</div>
                  <div className="mt-1 break-words text-xs font-bold text-cyan-50">{formatExperienceValue(value)}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              <div className="border border-cyan-300/15 bg-slate-950/60 p-2">
                <div className="text-[10px] font-black text-cyan-300/70">導入目的</div>
                <div className="mt-1 text-xs font-bold leading-relaxed text-cyan-50">
                  {formatExperienceValue(pickExperienceValue(item.formSnapshot, currentData, ["asset_purpose"]))}
                </div>
              </div>
              <div className="border border-cyan-300/15 bg-slate-950/60 p-2">
                <div className="text-[10px] font-black text-cyan-300/70">営業メモ</div>
                <div className="mt-1 text-xs font-bold leading-relaxed text-cyan-50">
                  {formatExperienceValue(pickExperienceValue(item.formSnapshot, currentData, ["passion_text"]))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DemoSimilarPastCasesCard({
  data,
  experienceCases,
  onSaveExperience,
  saving,
}: {
  data: ScoringFormData;
  experienceCases: DemoSimilarPastCase[];
  onSaveExperience: () => void;
  saving: boolean;
}) {
  const [selectedCase, setSelectedCase] = useState<DemoSimilarPastCase | null>(null);
  if (!experienceCases.length) return null;

  return (
    <section className="rounded-2xl border border-sky-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm font-black text-sky-950">
            <Database className="h-4 w-4 text-sky-600" />
            保存済み経験ケース
          </div>
          <p className="mt-1 text-xs font-bold leading-relaxed text-sky-700">
            {data.industry_sub || "この案件"} と同じ論点で、後から再利用できる経験データを表示します。
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex w-fit rounded-full border border-sky-100 bg-sky-50 px-3 py-1 text-[10px] font-black text-sky-700">
            {experienceCases.length}件
          </span>
          <button
            type="button"
            onClick={onSaveExperience}
            disabled={saving}
            className="inline-flex items-center gap-1.5 rounded-lg border border-sky-200 bg-sky-50 px-3 py-1.5 text-[11px] font-black text-sky-800 transition hover:bg-sky-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" />
            {saving ? "保存中" : "今回を経験化"}
          </button>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        {experienceCases.map((item, index) => (
          <article
            key={`${item.id || item.demoCaseId || index}-${item.companyName}`}
            role="button"
            tabIndex={0}
            onClick={() => setSelectedCase(item)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setSelectedCase(item);
              }
            }}
            className="group cursor-pointer rounded-xl border border-slate-200 bg-slate-50 p-3 transition hover:border-cyan-300 hover:bg-cyan-50/40 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-cyan-300"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h4 className="text-sm font-black text-slate-900 group-hover:text-cyan-950">{item.companyName}</h4>
                <p className="mt-0.5 text-[11px] font-bold text-slate-500">
                  {item.period} / {item.industry}
                </p>
                {!!item.similarityReasons?.length && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {item.similarityReasons.slice(0, 4).map((reason) => (
                      <span key={reason} className="rounded-full border border-sky-100 bg-sky-50 px-2 py-0.5 text-[10px] font-black text-sky-700">
                        {reason}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="shrink-0 rounded-lg bg-white px-2.5 py-1 text-right shadow-sm">
                <div className="text-[10px] font-black text-slate-400">当時</div>
                <div className="text-xs font-black text-slate-800">{item.score.toFixed(1)}点</div>
                {!!item.similarityScore && (
                  <div className="mt-0.5 text-[10px] font-black text-sky-700">類似度 {Math.round(item.similarityScore)}</div>
                )}
              </div>
            </div>
            <div className="mt-2 flex items-center justify-end text-[10px] font-black text-cyan-700 opacity-80 transition group-hover:opacity-100">
              <Eye className="mr-1 h-3 w-3" />
              記録層を開く
            </div>

            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <div className="rounded-lg border border-white bg-white p-2">
                <div className="text-[10px] font-black text-slate-400">判断</div>
                <div className="mt-1 text-xs font-black text-slate-800">{item.decision}</div>
                <div className="mt-1 text-[11px] font-bold leading-relaxed text-slate-500">{item.outcome}</div>
              </div>
              <div className="rounded-lg border border-white bg-white p-2">
                <div className="text-[10px] font-black text-slate-400">似ている点</div>
                <div className="mt-1 text-[11px] font-bold leading-relaxed text-slate-600">{item.similarity}</div>
              </div>
            </div>

            <div className="mt-2 rounded-lg border border-emerald-100 bg-emerald-50 p-2">
              <div className="text-[10px] font-black text-emerald-700">当時こうした</div>
              <div className="mt-1 text-[11px] font-bold leading-relaxed text-emerald-950">{item.actionTaken}</div>
            </div>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <div className="rounded-lg border border-amber-100 bg-amber-50 p-2">
                <div className="text-[10px] font-black text-amber-700">教訓</div>
                <div className="mt-1 text-[11px] font-bold leading-relaxed text-amber-950">{item.lesson}</div>
              </div>
              <div className="rounded-lg border border-blue-100 bg-blue-50 p-2">
                <div className="text-[10px] font-black text-blue-700">今回との違い</div>
                <div className="mt-1 text-[11px] font-bold leading-relaxed text-blue-950">{item.difference}</div>
              </div>
            </div>
          </article>
        ))}
      </div>
      {selectedCase && (
        <ExperienceCaseDetailModal
          item={selectedCase}
          data={data}
          onClose={() => setSelectedCase(null)}
        />
      )}
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
        score: getScreeningScore(result),
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

function IndustryBankruptcyBenchCard({ bench }: { bench?: any }) {
  if (!bench) return null;
  const levelStyles: Record<string, string> = {
    "高": "border-rose-200 bg-rose-50 text-rose-900",
    "中高": "border-orange-200 bg-orange-50 text-orange-900",
    "中": "border-amber-200 bg-amber-50 text-amber-900",
    "低": "border-emerald-200 bg-emerald-50 text-emerald-900",
  };
  const style = levelStyles[bench.risk_level] || "border-slate-200 bg-slate-50 text-slate-900";
  const stars = "●".repeat(bench.risk_stars || 0) + "○".repeat(4 - (bench.risk_stars || 0));
  return (
    <section className={`rounded-2xl border p-4 shadow-sm ${style}`}>
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0" />
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-black">業種マクロリスク（参考情報）</h3>
              <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-black">
                {bench.matched_category}
              </span>
            </div>
            <p className="mt-1 text-xs font-bold leading-relaxed">
              リスクレベル: {bench.risk_level} {stars} ／ 全業種平均比: {bench.relative_risk}
            </p>
            <p className="mt-1 text-[10px] font-bold opacity-70">
              出典: TDB「倒産集計2025年度報」×中小企業庁データ。業種全体の母集団倒産率であり、個社の倒産確率（PD）ではありません。スコアには反映していません。
            </p>
            {bench.note && (
              <p className="mt-1 text-[10px] font-bold text-orange-600">⚠️ {bench.note}</p>
            )}
          </div>
        </div>
        <div className="grid min-w-[220px] grid-cols-2 gap-2">
          <div className="rounded-xl bg-white/70 px-3 py-2 text-center">
            <div className="text-[10px] font-black opacity-60">年間倒産率</div>
            <div className="text-lg font-black">{Number(bench.rate).toFixed(2)}%</div>
          </div>
          <div className="rounded-xl bg-white/70 px-3 py-2 text-center">
            <div className="text-[10px] font-black opacity-60">1万者あたり</div>
            <div className="text-lg font-black">{Number(bench.per_10k).toFixed(0)}件</div>
          </div>
        </div>
      </div>
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
  const [gameTheoryResult, setGameTheoryResult] = useState<any>(null);
  const [formData, setFormData] = useState<ScoringFormData>(defaultFormData);
  const [gunshiText, setGunshiText] = useState<string>("");
  const [shionReview, setShionReview] = useState<ShionScreeningReview | null>(null);
  const [shionReviewLoading, setShionReviewLoading] = useState(false);
  const [shionReviewError, setShionReviewError] = useState("");
  const [shionFeedbackSaving, setShionFeedbackSaving] = useState(false);
  const [shionPastCompanies, setShionPastCompanies] = useState<PastCompanyHighlight[]>([]);
  const [judgmentAssetCandidates, setJudgmentAssetCandidates] = useState<JudgmentAssetCandidate[]>([]);
  const [judgmentAssetCandidatesLoading, setJudgmentAssetCandidatesLoading] = useState(false);
  const [inputJudgmentAssetCandidates, setInputJudgmentAssetCandidates] = useState<JudgmentAssetCandidate[]>([]);
  const [inputJudgmentAssetLoading, setInputJudgmentAssetLoading] = useState(false);
  const [inputJudgmentAssetSearched, setInputJudgmentAssetSearched] = useState(false);
  const [judgmentAssetFeedbackSavingId, setJudgmentAssetFeedbackSavingId] = useState("");
  const [judgmentAssetAdaptationMode, setJudgmentAssetAdaptationMode] = useState<JudgmentAssetAdaptationMode>("standard");
  const [draftRestored, setDraftRestored] = useState(false);
  const [lastDraftSavedAt, setLastDraftSavedAt] = useState<Date | null>(null);
  const [currentExperienceCases, setCurrentExperienceCases] = useState<DemoSimilarPastCase[]>([]);
  const [experienceSaving, setExperienceSaving] = useState(false);
  const [inputAssistCases, setInputAssistCases] = useState<DemoSimilarPastCase[]>([]);
  const [inputAssistLoading, setInputAssistLoading] = useState(false);
  const [inputAssistSearched, setInputAssistSearched] = useState(false);
  const [inputAssistSelectedCase, setInputAssistSelectedCase] = useState<DemoSimilarPastCase | null>(null);
  const [inputAssistNotice, setInputAssistNotice] = useState("");
  const inputAssistSessionId = useRef(`screening-input-assist:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`);
  const inputStartedAt = useRef(Date.now());
  const inputStartedLogged = useRef(false);
  const lastCopiedAt = useRef<number | null>(null);
  const lastCopiedSnapshot = useRef<Partial<Record<keyof ScoringFormData, unknown>> | null>(null);
  const lastCopiedFields = useRef<(keyof ScoringFormData)[]>([]);
  const shionReviewRequestSeq = useRef(0);
  const suppressNextDraftSave = useRef(false);

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

  // リースくん（スマホUI）から case_id 付きで遷移してきた場合、保存済み入力を読み込んでこの画面で審査を実行する
  useEffect(() => {
    const caseIdParam = new URLSearchParams(window.location.search).get("case_id");
    if (!caseIdParam) return;
    setDraftRestored(true);
    (async () => {
      try {
        const res = await apiClient.get(`/api/cases/${caseIdParam}`);
        const inputs = (res.data?.inputs ?? {}) as Record<string, unknown>;
        const nextFormData = {
          ...defaultFormData,
          ...fromThousandYenPayload(inputs),
          strength_tags: Array.isArray(inputs.strength_tags) ? inputs.strength_tags : [],
        } as ScoringFormData;
        setFormData(nextFormData);
        setActiveTab("input");
        router.replace("/screening");
        void handleSubmit(nextFormData);
      } catch (error) {
        console.error("Failed to load case from case_id", error);
        alert("案件の読み込みに失敗しました。");
      }
    })();
  }, []);

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("case_id")) return;
    const raw = window.localStorage.getItem(SCREENING_RETURN_STATE_KEY);
    if (!raw) {
      setDraftRestored(true);
      return;
    }
    try {
      const saved = JSON.parse(raw) as {
        version?: number;
        formData?: ScoringFormData;
        result?: any;
        gunshiText?: string;
        shionReview?: ShionScreeningReview | null;
        shionPastCompanies?: PastCompanyHighlight[];
        judgmentAssetCandidates?: JudgmentAssetCandidate[];
        judgmentAssetAdaptationMode?: JudgmentAssetAdaptationMode;
        currentExperienceCases?: DemoSimilarPastCase[];
        activeTab?: "input" | "analysis";
        savedAt?: string;
      };
      if (saved.formData) setFormData(saved.formData);
      if (saved.result) setResult(saved.result);
      if (typeof saved.gunshiText === "string") setGunshiText(saved.gunshiText);
      if (saved.shionReview) setShionReview(saved.shionReview);
      if (Array.isArray(saved.shionPastCompanies) && saved.shionPastCompanies.length) {
        setShionPastCompanies(saved.shionPastCompanies);
      }
      if (Array.isArray(saved.judgmentAssetCandidates)) setJudgmentAssetCandidates(saved.judgmentAssetCandidates);
      if (saved.judgmentAssetAdaptationMode) setJudgmentAssetAdaptationMode(saved.judgmentAssetAdaptationMode);
      if (Array.isArray(saved.currentExperienceCases)) setCurrentExperienceCases(saved.currentExperienceCases);
      setActiveTab(saved.activeTab || (saved.result ? "analysis" : "input"));
      if (saved.savedAt) {
        const savedDate = new Date(saved.savedAt);
        if (!Number.isNaN(savedDate.getTime())) setLastDraftSavedAt(savedDate);
      }
    } catch {
      window.localStorage.removeItem(SCREENING_RETURN_STATE_KEY);
    } finally {
      setDraftRestored(true);
    }
  }, []);

  useEffect(() => {
    if (!draftRestored) return;
    if (suppressNextDraftSave.current) {
      suppressNextDraftSave.current = false;
      return;
    }
    const timer = window.setTimeout(() => {
      try {
        const savedAt = new Date();
        window.localStorage.setItem(SCREENING_RETURN_STATE_KEY, JSON.stringify({
          version: SCREENING_DRAFT_VERSION,
          formData,
          result,
          gunshiText,
          shionReview,
          shionPastCompanies,
          judgmentAssetCandidates,
          judgmentAssetAdaptationMode,
          currentExperienceCases,
          activeTab,
          savedAt: savedAt.toISOString(),
        }));
        setLastDraftSavedAt(savedAt);
      } catch (error) {
        console.warn("Screening draft save failed", error);
      }
    }, SCREENING_DRAFT_SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draftRestored, formData, result, gunshiText, shionReview, shionPastCompanies, judgmentAssetCandidates, judgmentAssetAdaptationMode, currentExperienceCases, activeTab]);



  const fetchExperienceCasesForContext = async (
    demoCaseId: string,
    targetFormData: Partial<ScoringFormData> = {},
    targetResult: any = null,
  ) => {
    if (!demoCaseId && !hasExperienceSearchContext(targetFormData, targetResult)) return [];
    try {
      const res = await apiClient.get("/api/screening-experience-cases", {
        params: buildExperienceCaseQuery(demoCaseId, targetFormData, targetResult),
      });
      const cases = Array.isArray(res.data?.cases)
        ? res.data.cases.map(normalizeExperienceCase)
        : [];
      setCurrentExperienceCases(cases);
      return cases;
    } catch {
      setCurrentExperienceCases([]);
      return [];
    }
  };

  // フィールドの変更ハンドラー
  const handleFieldChange = (name: string, value: string | number | string[]) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const recordInputAssistEvent = (
    action: "input_started" | "search_click" | "search_result" | "search_failed" | "candidate_selected" | "copied" | "score_submitted",
    extra: Record<string, unknown> = {},
  ) => {
    const payload = {
      action,
      surface: "screening",
      session_id: inputAssistSessionId.current,
      company_no: formData.company_no || "",
      company_name: formData.company_name || "",
      industry_major: formData.industry_major || "",
      industry_sub: formData.industry_sub || "",
      asset_name: formData.asset_name || "",
      elapsed_ms: Date.now() - inputStartedAt.current,
      ...extra,
    };
    void apiClient.post("/api/screening-input-assist-events", payload).catch((error) => {
      console.warn("Screening input assist event failed", error);
    });
  };

  useEffect(() => {
    if (!draftRestored || inputStartedLogged.current) return;
    inputStartedLogged.current = true;
    recordInputAssistEvent("input_started");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draftRestored]);

  const searchInputAssistCases = async () => {
    setInputAssistLoading(true);
    setInputAssistSearched(true);
    setInputAssistNotice("");
    setInputAssistSelectedCase(null);
    recordInputAssistEvent("search_click");
    try {
      const res = await apiClient.get("/api/screening-experience-cases", {
        params: buildExperienceCaseQuery("", formData, result),
      });
      const cases = Array.isArray(res.data?.cases)
        ? res.data.cases.map(normalizeExperienceCase).filter((item: DemoSimilarPastCase) => item.formSnapshot && Object.keys(item.formSnapshot).length > 0)
        : [];
      setInputAssistCases(cases);
      recordInputAssistEvent("search_result", {
        candidate_count: cases.length,
        metadata: {
          raw_count: String(res.data?.count ?? cases.length),
        },
      });
      if (!cases.length) {
        setInputAssistNotice("コピーできる過去案件が見つかりませんでした。審査後に経験ケース保存を増やすと候補が育ちます。");
      }
    } catch (error) {
      console.error("Input assist search failed", error);
      setInputAssistCases([]);
      setInputAssistNotice("過去案件の取得に失敗しました。API接続を確認してください。");
      recordInputAssistEvent("search_failed", {
        note: error instanceof Error ? error.message : "unknown error",
      });
    } finally {
      setInputAssistLoading(false);
    }
  };

  const applyInputAssistCase = (sourceCase: DemoSimilarPastCase) => {
    const snapshot = sourceCase.formSnapshot || {};
    const diffs = buildScreeningCopyDiffs(formData, snapshot);
    if (!diffs.length) {
      setInputAssistNotice("現在の入力とコピー元に差分がありません。");
      setInputAssistSelectedCase(null);
      return;
    }
    const copiedFields = diffs.map((diff) => diff.key);
    const confirmFields = diffs.filter((diff) => diff.needsConfirmation).map((diff) => diff.key);
    setFormData((prev) => {
      const next = { ...prev } as ScoringFormData;
      SCREENING_COPY_FIELD_ORDER.forEach((key) => {
        if (SCREENING_COPY_EXCLUDED_FIELDS.has(key) || !(key in snapshot)) return;
        (next as unknown as Record<string, unknown>)[key] = snapshot[key];
      });
      return next;
    });
    setResult(null);
    setGameTheoryResult(null);
    setGunshiText("");
    setShionReview(null);
    setShionPastCompanies([]);
    setJudgmentAssetCandidates([]);
    setCurrentExperienceCases([]);
    setInputAssistSelectedCase(null);
    setInputAssistNotice(`${sourceCase.companyName || "過去案件"}から${diffs.length}項目をコピーしました。企業番号・企業名は上書きしていません。`);
    lastCopiedAt.current = Date.now();
    lastCopiedSnapshot.current = snapshot;
    lastCopiedFields.current = copiedFields;
    recordInputAssistEvent("copied", {
      source_case_id: sourceCase.sourceCaseId || String(sourceCase.id || ""),
      source_company_name: sourceCase.companyName || "",
      diff_count: diffs.length,
      confirm_count: confirmFields.length,
      copied_field_count: copiedFields.length,
      copied_fields: copiedFields,
      confirm_fields: confirmFields,
      metadata: {
        source: sourceCase.source || "",
        score: String(sourceCase.score ?? ""),
        decision: sourceCase.decision || "",
        outcome: sourceCase.outcome || "",
      },
    });
  };

  const countChangedAfterInputAssistCopy = (targetFormData: ScoringFormData) => {
    const snapshot = lastCopiedSnapshot.current;
    if (!snapshot || !lastCopiedFields.current.length) return 0;
    return lastCopiedFields.current.filter((key) => (
      formatScreeningCopyValue(targetFormData[key]) !== formatScreeningCopyValue(snapshot[key])
    )).length;
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

  const fetchJudgmentAssetCandidatesForScreening = async (targetResult: any, targetFormData: ScoringFormData) => {
    setJudgmentAssetCandidatesLoading(true);
    try {
      const res = await apiClient.get("/api/judgment-asset-candidates/screening", {
        params: {
          industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
          industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
          asset_name: targetFormData.asset_name || "",
          asset_purpose: targetFormData.asset_purpose || "",
          hantei: targetResult?.hantei || "",
          score: getScreeningScore(targetResult),
          limit: 3,
        },
      });
      const candidates = Array.isArray(res.data?.candidates) ? res.data.candidates as JudgmentAssetCandidate[] : [];
      setJudgmentAssetCandidates(candidates);
      return candidates;
    } catch (error) {
      console.warn("Judgment asset candidates fetch failed", error);
      setJudgmentAssetCandidates([]);
      return [];
    } finally {
      setJudgmentAssetCandidatesLoading(false);
    }
  };

  const searchInputJudgmentAssets = async () => {
    setInputJudgmentAssetLoading(true);
    setInputJudgmentAssetSearched(true);
    try {
      const res = await apiClient.get("/api/judgment-asset-candidates/screening", {
        params: {
          industry_major: formData.industry_major || "",
          industry_sub: formData.industry_sub || "",
          asset_name: formData.asset_name || "",
          asset_purpose: formData.asset_purpose || "",
          hantei: "",
          score: "",
          limit: 3,
        },
      });
      const candidates = Array.isArray(res.data?.candidates) ? res.data.candidates as JudgmentAssetCandidate[] : [];
      setInputJudgmentAssetCandidates(candidates);
    } catch (error) {
      console.warn("Input judgment asset preview fetch failed", error);
      setInputJudgmentAssetCandidates([]);
    } finally {
      setInputJudgmentAssetLoading(false);
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
      score: getScreeningScore(targetResult),
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

  const submitJudgmentAssetCandidateFeedback = async (
    candidate: JudgmentAssetCandidate,
    feedback: JudgmentAssetCandidateFeedback,
    editedClaim?: string,
  ) => {
    if (!candidate.id || judgmentAssetFeedbackSavingId) return;
    const previous = judgmentAssetCandidates;
    const normalizedEditedClaim = String(editedClaim || "").trim();
    setJudgmentAssetCandidates((current) => current.map((item) => (
      item.id === candidate.id
        ? {
            ...item,
            userFeedback: feedback,
            ...(normalizedEditedClaim ? { edited_claim: normalizedEditedClaim, effective_claim: normalizedEditedClaim } : {}),
          }
        : item
    )));
    setJudgmentAssetFeedbackSavingId(candidate.id);
    try {
      const res = await apiClient.post(`/api/judgment-asset-candidates/${candidate.id}/feedback`, {
        feedback,
        case_id: result?.case_id || formData.company_no || formData.company_name || "",
        review_id: shionReview?.savedId || null,
        edited_claim: normalizedEditedClaim,
      });
      const updated = res.data?.candidate || {};
      setJudgmentAssetCandidates((current) => current.map((item) => (
        item.id === candidate.id
          ? {
              ...item,
              userFeedback: feedback,
              use_count: Number(updated.use_count ?? item.use_count ?? 0),
              useful_count: Number(updated.useful_count ?? item.useful_count ?? 0),
              rejected_count: Number(updated.rejected_count ?? item.rejected_count ?? 0),
              verified_status: String(updated.verified_status || item.verified_status || "unverified"),
              edited_claim: String(updated.edited_claim || item.edited_claim || ""),
              effective_claim: String(updated.edited_claim || item.effective_claim || item.claim || ""),
              edit_count: Number(updated.edit_count ?? item.edit_count ?? 0),
            }
          : item
      )));
    } catch (error) {
      console.error("Judgment asset candidate feedback save failed", error);
      setJudgmentAssetCandidates(previous);
      setShionReviewError("判断資産候補の評価を保存できませんでした。");
    } finally {
      setJudgmentAssetFeedbackSavingId("");
    }
  };

  const createManualJudgmentAssetCandidate = async (claim: string, candidateType: string) => {
    const normalizedClaim = claim.trim();
    if (!normalizedClaim || judgmentAssetFeedbackSavingId) return;
    setJudgmentAssetFeedbackSavingId("manual-create");
    try {
      const res = await apiClient.post("/api/judgment-asset-candidates/manual", {
        claim: normalizedClaim,
        candidate_type: candidateType,
        research_topic: "manual-screening",
        case_id: result?.case_id || formData.company_no || formData.company_name || "",
        review_id: shionReview?.savedId || null,
      });
      const candidate = res.data?.candidate as JudgmentAssetCandidate | undefined;
      if (candidate?.id) {
        setJudgmentAssetCandidates((current) => [
          {
            ...candidate,
            effective_claim: candidate.effective_claim || candidate.edited_claim || candidate.claim,
            edit_count: Number(candidate.edit_count ?? 1),
            use_count: Number(candidate.use_count ?? 0),
            useful_count: Number(candidate.useful_count ?? 0),
            rejected_count: Number(candidate.rejected_count ?? 0),
            verified_status: String(candidate.verified_status || "unverified"),
          },
          ...current.filter((item) => item.id !== candidate.id),
        ].slice(0, 5));
      }
    } catch (error) {
      console.error("Manual judgment asset candidate create failed", error);
      setShionReviewError("判断資産候補を追加できませんでした。");
    } finally {
      setJudgmentAssetFeedbackSavingId("");
    }
  };

  const requestShionReview = async (targetResult = result, targetFormData = formData) => {
    if (!targetResult) return;
    const seq = ++shionReviewRequestSeq.current;
    let promptText = "";
    let fallbackPastReviews: PastShionScreeningReview[] = [];
    let fallbackExperienceCases: DemoSimilarPastCase[] = [];
    let fallbackCandidates: JudgmentAssetCandidate[] = [];
    setShionReviewLoading(true);
    setShionReviewError("");
    try {
      const pastReviews = await fetchPastShionReviews(targetResult, targetFormData);
      const experienceCases = await fetchExperienceCasesForContext("", targetFormData, targetResult);
      const candidates = await fetchJudgmentAssetCandidatesForScreening(targetResult, targetFormData);
      fallbackPastReviews = pastReviews;
      fallbackExperienceCases = experienceCases as DemoSimilarPastCase[];
      fallbackCandidates = candidates;
      const pastCompanies = uniquePastCompanyHighlights(
        pastReviews,
        experienceCases as DemoSimilarPastCase[],
      );
      setShionPastCompanies(pastCompanies);
      if (seq !== shionReviewRequestSeq.current) return;
      promptText = buildShionReviewPrompt(
        targetResult,
        targetFormData,
        pastReviews,
        experienceCases,
        candidates,
        judgmentAssetAdaptationMode,
      );
      const res = await apiClient.post("/api/chat", {
        message: promptText,
        user_id: buildShionReviewUserId(targetResult, targetFormData),
        response_mode: "shion",
        debug_memory: true,
      }, {
        timeout: 18000,
      });
      if (seq !== shionReviewRequestSeq.current) return;
      const memoryDebug = res.data?.memory_debug || {};
      const memoryRecall = memoryDebug.memory_recall || {};
      const identityMemory = memoryDebug.identity_memory || {};
      const vertexSearch = memoryDebug.vertex_ai_search || res.data?.vertex_ai_search || {};
      const vertexAnswer = memoryDebug.vertex_answer_api || res.data?.vertex_answer_api || {};
      const parsedGroundingScore =
        typeof vertexAnswer.grounding_score === "number"
          ? vertexAnswer.grounding_score
          : vertexAnswer.grounding_score != null
            ? Number(vertexAnswer.grounding_score)
            : null;
      const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0;
      const memoryRefs = Array.isArray(memoryRecall.refs) ? memoryRecall.refs.length : 0;
      const nextReview: ShionScreeningReview = {
        reply: ensureCandidateJudgmentAssetMentionInReview(
          ensurePastCompanyMentionInReview(
            String(res.data?.reply || "紫苑レビューが空でした。"),
            pastCompanies,
          ),
          candidates,
        ),
        memoryRefs,
        knowledgeRefs,
        identityUsed: Boolean(identityMemory.used),
        vertexUsed: Boolean(vertexSearch.used),
        vertexStatus: String(vertexSearch.status || ""),
        vertexRefs: Array.isArray(vertexSearch.refs) ? vertexSearch.refs.map(String) : [],
        vertexAnswerUsed: Boolean(vertexAnswer.used),
        vertexAnswerStatus: String(vertexAnswer.status || ""),
        groundingScore: Number.isFinite(parsedGroundingScore) ? parsedGroundingScore : null,
        groundingScoreSource: String(vertexAnswer.grounding_score_source || ""),
        lowSupportClaimCount: Number(vertexAnswer.low_support_claim_count || 0),
        supportCount: Number(vertexAnswer.support_count || 0),
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
      const fallbackPastCompanies = uniquePastCompanyHighlights(fallbackPastReviews, fallbackExperienceCases);
      const fallbackReview: ShionScreeningReview = {
        reply: ensureCandidateJudgmentAssetMentionInReview(
          ensurePastCompanyMentionInReview(
            buildShionReviewFallback(
              targetResult,
              targetFormData,
              fallbackCandidates,
              fallbackExperienceCases,
              fallbackPastReviews,
            ),
            fallbackPastCompanies,
          ),
          fallbackCandidates,
        ),
        memoryRefs: fallbackPastReviews.length,
        knowledgeRefs: fallbackCandidates.length,
        identityUsed: false,
        vertexUsed: false,
        vertexStatus: "fallback",
        vertexRefs: [],
        vertexAnswerUsed: false,
        vertexAnswerStatus: "fallback",
        groundingScore: null,
        groundingScoreSource: "",
        lowSupportClaimCount: 0,
        supportCount: 0,
      };
      setShionReview(fallbackReview);
      setShionReviewError("");
      saveShionScreeningReview(
        targetResult,
        targetFormData,
        promptText || "fallback: local screening review from case fields and judgment assets",
        fallbackReview,
      )
        .then((savedId) => {
          if (!savedId || seq !== shionReviewRequestSeq.current) return;
          setShionReview((current) => current ? { ...current, savedId } : current);
        })
        .catch((saveError) => {
          console.warn("Fallback Shion screening review save failed", saveError);
        });
    } finally {
      if (seq === shionReviewRequestSeq.current) {
        setShionReviewLoading(false);
      }
    }
  };

  const resetScreening = () => {
    if (!window.confirm("審査分析の入力・分析結果・紫苑レビューをすべて消去します。よろしいですか？")) return;
    suppressNextDraftSave.current = true;
    shionReviewRequestSeq.current += 1;
    setFormData(defaultFormData);
    setResult(null);
    setGunshiText("");
    setShionReview(null);
    setShionReviewLoading(false);
    setShionReviewError("");
    setShionFeedbackSaving(false);
    setCurrentExperienceCases([]);
    setShionPastCompanies([]);
    setJudgmentAssetCandidates([]);
    setInputJudgmentAssetCandidates([]);
    setInputJudgmentAssetSearched(false);
    setJudgmentAssetFeedbackSavingId("");
    setActiveTab("input");
    window.localStorage.removeItem(SCREENING_RETURN_STATE_KEY);
    setLastDraftSavedAt(null);
  };

  const handleSubmit = async (targetFormData: ScoringFormData = formData) => {
    setLoading(true);
    setShionReview(null);
    setShionReviewError("");
    setShionPastCompanies([]);
    setJudgmentAssetCandidates([]);
    setInputJudgmentAssetCandidates([]);
    setInputJudgmentAssetSearched(false);
    setJudgmentAssetFeedbackSavingId("");
    const copiedBeforeSubmit = Boolean(lastCopiedAt.current);
    const elapsedFromCopyMs = lastCopiedAt.current ? Date.now() - lastCopiedAt.current : null;
    try {
      const res = await apiClient.post(`/api/score/full`, toThousandYenPayload(targetFormData));
      recordInputAssistEvent("score_submitted", {
        changed_after_copy_count: countChangedAfterInputAssistCopy(targetFormData),
        copied_field_count: lastCopiedFields.current.length,
        metadata: {
          copied_before_submit: String(copiedBeforeSubmit),
          elapsed_from_copy_ms: elapsedFromCopyMs === null ? "" : String(elapsedFromCopyMs),
        },
      });
      setResult(res.data);
      setActiveTab("analysis");
      // REV-224: 審査ゲーム理論分析（並列・非ブロッキング）
      apiClient.post("/api/game-theory/screening", {
        industry_major: targetFormData.industry_major ?? "",
        nenshu: Number(targetFormData.nenshu ?? 0),
        op_profit: Number(targetFormData.op_profit ?? 0),
        net_income: Number(targetFormData.net_income ?? 0),
        net_assets: Number(targetFormData.net_assets ?? 0),
        total_assets: Number(targetFormData.total_assets ?? 0),
      }).then((gtRes) => setGameTheoryResult(gtRes.data)).catch(() => {});
      void fetchExperienceCasesForContext("", targetFormData, res.data);
      void requestShionReview(res.data, targetFormData);

      // めぶきちゃんの表情を判定バッジ（AiHeroCard）と同じ基準で切り替える
      // approval_line は /api/score/full には現状含まれないため 71 にフォールバックする
      //（バッジ側も同じフォールバックなので文言と表示が食い違わない）
      const score = getScreeningScore(res.data);
      const approvalLine = typeof res.data.approval_line === "number" ? res.data.approval_line : 71;
      if (score >= approvalLine) {
        triggerMebuki('approve', `スコア ${score.toFixed(1)} 点！\n素晴らしい内容です。\nこのまま稟議に掛けましょう！`);
      } else if (score >= 60) {
        triggerMebuki('challenge', `スコア ${score.toFixed(1)} 点。\n条件付き承認圏です。\n軍師のアドバイスを確認してください。`);
      } else {
        triggerMebuki('reject', `スコア ${score.toFixed(1)} 点。\nかなり厳しい状況です。\n抜本的な条件見直しが必要です！`);
      }

    } catch (error) {
      console.error("API Error", error);
      alert(getScreeningErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const saveCurrentExperienceCase = async () => {
    if (!result || experienceSaving) return;
    const score = getScreeningScore(result);
    setExperienceSaving(true);
    try {
      await apiClient.post("/api/screening-experience-cases", {
        demo_case_id: "",
        source_case_id: result.case_id || formData.company_no || "",
        company_name: formData.company_name || "名称未設定",
        period: "今回の審査",
        industry_major: result.industry_major || formData.industry_major || "",
        industry_sub: result.industry_sub || formData.industry_sub || "",
        sales_dept: formData.sales_dept || "",
        score,
        decision: result.hantei || "",
        outcome: "審査経験として保存",
        similarity: buildCurrentIssue(result, formData),
        action_taken: buildRingiPolicy(result, formData),
        lesson: "今回の判断・条件・違和感を、次回の類似案件で再利用する。",
        difference: "実案件化する場合は、成約/失注/条件変更の最終結果でこの経験を更新する。",
        source: "screening_result",
        form_snapshot: formData,
        result_snapshot: result,
      });
      await fetchExperienceCasesForContext("", formData, result);
    } catch (error) {
      console.error("Screening experience save failed", error);
      alert("経験データを保存できませんでした。");
    } finally {
      setExperienceSaving(false);
    }
  };

  const handoffToShionChat = () => {
    if (!result) return;
    const chatContext = {
      score: getScreeningScore(result),
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
      version: SCREENING_DRAFT_VERSION,
      formData,
      result,
      gunshiText,
      shionReview,
      shionPastCompanies,
      judgmentAssetCandidates,
      judgmentAssetAdaptationMode,
      currentExperienceCases,
      activeTab: "analysis",
      savedAt: new Date().toISOString(),
    }));
    window.localStorage.setItem("lease-gunshi-context", JSON.stringify(chatContext));
    router.push("/chat");
  };

  const openScreeningKnowledgeSpace = () => {
    if (!result) return;
    const qRisk = result.quantum_risk != null ? Number(result.quantum_risk) : null;
    const terms = [
      formData.company_name,
      result.industry_sub || formData.industry_sub,
      result.industry_major || formData.industry_major,
      formData.asset_name,
      formData.asset_purpose,
      result.hantei,
      qRisk != null ? `Q_risk ${qRisk.toFixed(1)}` : "",
    ].filter(Boolean);
    openKnowledgeSpaceFocus(terms.join(" "), "screening_result");
  };

  const handoffToShionDebate = () => {
    if (!result) return;
    const score = getScreeningScore(result);
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
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1.5 rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1.5 text-xs font-black text-emerald-700">
            <Save className="h-3.5 w-3.5" />
            {lastDraftSavedAt ? `自動保存 ${lastDraftSavedAt.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" })}` : "自動保存"}
          </div>
          <button
            type="button"
            onClick={resetScreening}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-bold text-rose-700 bg-rose-50 hover:bg-rose-100 rounded-lg transition-colors border border-rose-100 shadow-sm"
          >
            <Trash2 className="h-4 w-4" />
            全消去
          </button>
        </div>
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
          <Link href="/help" className="bg-white border border-emerald-200 rounded-2xl p-4 shadow-sm hover:shadow-md transition-shadow flex items-center justify-between gap-3">
            <div>
              <div className="text-[10px] font-black uppercase tracking-widest text-emerald-500">Support</div>
              <div className="font-black text-slate-800 mt-1 flex items-center gap-2"><BadgeInfo className="w-4 h-4 text-emerald-500" />審査ヘルプ</div>
              <div className="text-xs text-slate-500 mt-1">FAQ・残価・営業説明をまとめて確認。</div>
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
        <AiHeroCard result={result} data={formData} onOpenKnowledge={openScreeningKnowledgeSpace} />

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

                <InputJudgmentAssetPreview
                  candidates={inputJudgmentAssetCandidates}
                  loading={inputJudgmentAssetLoading}
                  searched={inputJudgmentAssetSearched}
                  onSearch={searchInputJudgmentAssets}
                />

                <section className="bg-white border border-emerald-100 rounded-2xl p-4 shadow-sm">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
                        <Database className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="text-sm font-black text-slate-800">過去案件から作成</h3>
                        <p className="mt-1 text-xs text-slate-500">業種・物件・取引条件が近い保存済み経験ケースから、入力値を確認付きでコピーします。</p>
                        {inputAssistNotice && (
                          <p className="mt-2 rounded-lg bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-700">
                            {inputAssistNotice}
                          </p>
                        )}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={searchInputAssistCases}
                      disabled={inputAssistLoading}
                      className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-2.5 text-sm font-black text-white shadow-sm transition hover:bg-emerald-700 disabled:opacity-50"
                    >
                      {inputAssistLoading ? <Activity className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                      類似案件を探す
                    </button>
                    <Link
                      href="/improvement-log"
                      className="inline-flex items-center justify-center gap-2 rounded-xl border border-emerald-200 bg-white px-4 py-2.5 text-sm font-black text-emerald-700 shadow-sm transition hover:bg-emerald-50"
                    >
                      <ChartNoAxesCombined className="h-4 w-4" />
                      効果測定
                    </Link>
                  </div>

                  {inputAssistCases.length > 0 && (
                    <div className="mt-4 grid gap-3 xl:grid-cols-3">
                      {inputAssistCases.slice(0, 3).map((item) => {
                        const diffs = buildScreeningCopyDiffs(formData, item.formSnapshot || {});
                        const selected = inputAssistSelectedCase?.id === item.id;
                        return (
                          <button
                            key={`${item.id || item.sourceCaseId || item.companyName}-${item.period}`}
                            type="button"
                            onClick={() => {
                              setInputAssistSelectedCase(item);
                              recordInputAssistEvent("candidate_selected", {
                                source_case_id: item.sourceCaseId || String(item.id || ""),
                                source_company_name: item.companyName || "",
                                diff_count: diffs.length,
                                confirm_count: diffs.filter((diff) => diff.needsConfirmation).length,
                                metadata: {
                                  source: item.source || "",
                                  similarity_score: String(item.similarityScore ?? ""),
                                },
                              });
                            }}
                            className={`rounded-xl border p-3 text-left transition ${
                              selected ? "border-emerald-400 bg-emerald-50 ring-2 ring-emerald-100" : "border-slate-200 bg-slate-50 hover:border-emerald-200 hover:bg-emerald-50/60"
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0">
                                <div className="truncate text-sm font-black text-slate-800">{item.companyName || "名称未設定"}</div>
                                <div className="mt-1 truncate text-[11px] font-bold text-slate-500">
                                  {[item.industrySub || item.industry, item.salesDept].filter(Boolean).join(" / ") || "業種未設定"}
                                </div>
                              </div>
                              <span className="shrink-0 rounded-full bg-white px-2 py-1 text-[10px] font-black text-emerald-700">
                                {Math.round(item.similarityScore || 0)}
                              </span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1.5">
                              <span className="rounded-md bg-white px-2 py-1 text-[10px] font-black text-slate-600">{Number(item.score || 0).toFixed(1)}点</span>
                              {item.decision && <span className="rounded-md bg-white px-2 py-1 text-[10px] font-black text-slate-600">{item.decision}</span>}
                              {item.outcome && <span className="rounded-md bg-white px-2 py-1 text-[10px] font-black text-slate-600">{item.outcome}</span>}
                            </div>
                            <p className="mt-2 line-clamp-2 text-[11px] font-bold leading-relaxed text-slate-500">
                              {item.lesson || item.similarity || "保存済み入力をコピーできます。"}
                            </p>
                            <div className="mt-2 flex items-center justify-between text-[10px] font-black text-slate-400">
                              <span>{diffs.length}項目差分</span>
                              <span>{item.source || "experience"}</span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {inputAssistSelectedCase && (
                    <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                      {(() => {
                        const diffs = buildScreeningCopyDiffs(formData, inputAssistSelectedCase.formSnapshot || {});
                        const confirmCount = diffs.filter((item) => item.needsConfirmation).length;
                        return (
                          <>
                            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                              <div>
                                <h4 className="text-sm font-black text-emerald-900">
                                  {inputAssistSelectedCase.companyName || "過去案件"} からコピー
                                </h4>
                                <p className="mt-1 text-xs font-bold text-emerald-700">
                                  企業番号・企業名は上書きしません。財務・与信・物件系はコピー後に必ず確認してください。
                                </p>
                              </div>
                              <div className="flex shrink-0 gap-2">
                                <button
                                  type="button"
                                  onClick={() => setInputAssistSelectedCase(null)}
                                  className="rounded-xl border border-emerald-200 bg-white px-3 py-2 text-xs font-black text-slate-600 transition hover:bg-slate-50"
                                >
                                  閉じる
                                </button>
                                <button
                                  type="button"
                                  onClick={() => applyInputAssistCase(inputAssistSelectedCase)}
                                  disabled={!diffs.length}
                                  className="inline-flex items-center gap-2 rounded-xl bg-emerald-700 px-4 py-2 text-xs font-black text-white shadow-sm transition hover:bg-emerald-800 disabled:opacity-50"
                                >
                                  <Copy className="h-3.5 w-3.5" />
                                  コピーする
                                </button>
                              </div>
                            </div>
                            <div className="mt-3 flex flex-wrap gap-2 text-[10px] font-black">
                              <span className="rounded-full bg-white px-2 py-1 text-emerald-700">差分 {diffs.length}項目</span>
                              <span className="rounded-full bg-amber-100 px-2 py-1 text-amber-800">要確認 {confirmCount}項目</span>
                              <span className="rounded-full bg-white px-2 py-1 text-slate-600">会社情報は除外</span>
                            </div>
                            <div className="mt-3 max-h-64 overflow-auto rounded-lg border border-emerald-100 bg-white">
                              {diffs.length ? (
                                diffs.slice(0, 18).map((diff) => (
                                  <div key={diff.key} className="grid gap-2 border-b border-slate-100 px-3 py-2 text-xs last:border-b-0 md:grid-cols-[120px_1fr_1fr]">
                                    <div className="flex items-center gap-1.5 font-black text-slate-700">
                                      {diff.needsConfirmation && <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />}
                                      {diff.label}
                                    </div>
                                    <div className="min-w-0 truncate text-slate-400">{diff.before}</div>
                                    <div className="min-w-0 truncate font-bold text-emerald-800">{diff.after}</div>
                                  </div>
                                ))
                              ) : (
                                <div className="px-3 py-4 text-xs font-bold text-slate-500">現在の入力と差分がありません。</div>
                              )}
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  )}

                  {inputAssistSearched && !inputAssistLoading && !inputAssistCases.length && !inputAssistNotice && (
                    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-xs font-bold text-slate-500">
                      コピーできる過去案件はまだありません。
                    </div>
                  )}
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
                <LeasePaymentSimulator
                  source="screening"
                  initialPriceMillion={Number(formData.acquisition_cost) || 10}
                  initialYears={Number(formData.lease_term) || 5}
                  defaultOpen={false}
                  title="案件条件で取得方法を比較"
                  description="入力中の取得価格・リース期間を使い、支払方法別の実質負担をその場で確認します。"
                />
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
                    onClick={() => handleSubmit()}
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
                    <ShionGrowthBadge />
                    <JudgmentFlowStrip />
                    <ShionScreeningReviewCard
                      review={shionReview}
                      loading={shionReviewLoading}
                      error={shionReviewError}
                      onReview={() => requestShionReview()}
                      onFeedback={submitShionReviewFeedback}
                      feedbackSaving={shionFeedbackSaving}
                      pastCompanies={shionPastCompanies}
                      judgmentAssetCandidates={judgmentAssetCandidates}
                      result={result}
                    />
                    <JudgmentAssetCandidateCard
                      candidates={judgmentAssetCandidates}
                      loading={judgmentAssetCandidatesLoading}
                      feedbackSavingId={judgmentAssetFeedbackSavingId}
                      adaptationMode={judgmentAssetAdaptationMode}
                      onAdaptationModeChange={setJudgmentAssetAdaptationMode}
                      onFeedback={submitJudgmentAssetCandidateFeedback}
                      onCreate={createManualJudgmentAssetCandidate}
                    />
                    <CurrentIssueCard result={result} data={formData} />
                    <RingiPolicyCard result={result} data={formData} />
                    <DemoSimilarPastCasesCard
                      data={formData}
                      experienceCases={currentExperienceCases}
                      onSaveExperience={saveCurrentExperienceCase}
                      saving={experienceSaving}
                    />
                    <ScreeningLoopFeedbackPanel result={result} data={formData} />
                    <LeasePaymentSimulator
                      source="screening"
                      initialPriceMillion={Number(formData.acquisition_cost) || 10}
                      initialYears={Number(formData.lease_term) || 5}
                      defaultOpen={false}
                      title="条件変更時の支払負担を見る"
                      description="審査結果を見ながら、期間・金利・税率を動かして承認条件や稟議補足の材料にします。"
                    />
                    <IndicatorCards data={result} />
                    <IndustryBankruptcyBenchCard bench={result?.industry_bankruptcy_bench} />

                    {/* REV-224: 審査ゲーム理論パネル */}
                    {gameTheoryResult && (
                      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 shadow-sm">
                        <div className="flex items-center justify-between gap-2 mb-3">
                          <h3 className="flex items-center gap-2 text-sm font-black text-amber-900">
                            <ShieldCheck className="h-4 w-4" />
                            審査ゲーム理論分析
                          </h3>
                          <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-black ${
                            gameTheoryResult.risk_level === "高"
                              ? "bg-rose-100 text-rose-700"
                              : gameTheoryResult.risk_level === "中"
                              ? "bg-amber-100 text-amber-700"
                              : "bg-emerald-100 text-emerald-700"
                          }`}>
                            操作リスク: {gameTheoryResult.risk_level}
                          </span>
                        </div>

                        {/* 操作疑いスコアバー */}
                        <div className="mb-3">
                          <div className="flex items-center justify-between text-[11px] font-bold text-amber-800 mb-1">
                            <span>情報操作疑いスコア</span>
                            <span>{(gameTheoryResult.manipulation_suspicion_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 w-full overflow-hidden rounded-full bg-amber-100">
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${
                                gameTheoryResult.manipulation_suspicion_score >= 0.6 ? "bg-rose-500" :
                                gameTheoryResult.manipulation_suspicion_score >= 0.3 ? "bg-amber-400" : "bg-emerald-400"
                              }`}
                              style={{ width: `${gameTheoryResult.manipulation_suspicion_score * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* 支配戦略 */}
                        <div className="mb-3 rounded-xl bg-white border border-amber-100 px-3 py-2">
                          <div className="text-[10px] font-black uppercase tracking-wider text-amber-600 mb-1">支配戦略（ナッシュ均衡）</div>
                          <div className="text-xs font-bold text-slate-800">
                            {gameTheoryResult.dominant_strategy_analysis?.nash_equilibrium}
                          </div>
                          <div className="mt-1 text-[10px] text-slate-500 leading-relaxed">
                            正直申告の期待利得 {((gameTheoryResult.dominant_strategy_analysis?.honest_expected_payoff ?? 0) * 100).toFixed(0)}%
                            　vs　操作申告 {((gameTheoryResult.dominant_strategy_analysis?.manipulated_expected_payoff ?? 0) * 100).toFixed(0)}%
                          </div>
                        </div>

                        {/* フラグ一覧 */}
                        {gameTheoryResult.strategy_flags?.length > 0 && (
                          <div className="space-y-1.5">
                            <div className="text-[10px] font-black uppercase tracking-wider text-amber-700">
                              検出シグナル ({gameTheoryResult.flag_count}件)
                            </div>
                            {gameTheoryResult.strategy_flags.slice(0, 4).map((flag: any, i: number) => (
                              <div key={i} className="flex items-start gap-2 rounded-lg bg-white border border-amber-100 px-3 py-2">
                                <AlertTriangle className={`h-3 w-3 mt-0.5 shrink-0 ${
                                  (flag.suspicion ?? 0) >= 0.4 ? "text-rose-500" : "text-amber-400"
                                }`} />
                                <p className="text-[10px] leading-relaxed text-slate-700">{flag.message}</p>
                              </div>
                            ))}
                          </div>
                        )}
                        {(!gameTheoryResult.strategy_flags || gameTheoryResult.strategy_flags.length === 0) && (
                          <p className="text-[11px] text-emerald-700 font-bold">✓ 操作シグナルは検出されませんでした</p>
                        )}
                      </div>
                    )}

                    <div className="rounded-2xl border border-violet-200 bg-violet-50 p-4 shadow-sm">
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                          <h3 className="flex items-center gap-2 text-sm font-black text-violet-900">
                            <MessageSquare className="h-4 w-4" />
                            この審査結果を紫苑へ渡す
                          </h3>
                          <p className="mt-1 text-xs font-bold leading-relaxed text-violet-700">
                            通常相談は紫苑チャットへ。要審議・境界案件は、懐疑派・楽観派・統合派の討論に回します。
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
                            討論で確認
                            <Swords className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* REV-004: 高リスク財務パターン警告パネル */}
                    {result.default_warnings?.length > 0 && (
                      <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 shadow-sm">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0" />
                          <span className="font-black text-rose-800 text-sm">高リスク財務パターン警告</span>
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
                        <p className="text-[10px] text-rose-400 mt-2">高リスク格付先との財務類似度を見た補助指標です。実際のデフォルト確率（PD）ではありません。</p>
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
                              financialConsistencyRisk={result.financial_consistency_risk ?? null}
                              compact={false}
                              caseId={String(result.case_id || "")}
                              score={getScreeningScore(result)}
                              hantei={String(result.hantei || "")}
                              context={{
                                industry_major: formData.industry_major,
                                industry_sub: formData.industry_sub,
                                customer_type: formData.customer_type,
                                main_bank: formData.main_bank,
                                competitor: formData.competitor,
                                competitor_rate: formData.competitor_rate,
                                deal_source: formData.deal_source,
                                asset_name: formData.asset_name,
                                financial_consistency_score: result.financial_consistency_score ?? null,
                              }}
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

                          {Array.isArray(result.diagnostic_recommendations) && result.diagnostic_recommendations.length > 0 && (
                            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                              <div className="mb-2 flex items-center gap-2 text-sm font-black text-amber-900">
                                <BadgeInfo className="h-4 w-4" />
                                補助診断の推奨
                              </div>
                              <p className="mb-3 text-xs leading-relaxed text-amber-800">
                                UMAP / マハラノビスは常時使用ではありません。紫苑が必要性を示し、人間が実行判断します。結果は自動減点ではなく、確認論点・稟議補足に使います。
                              </p>
                              <div className="space-y-2">
                                {result.diagnostic_recommendations.map((rec: any, index: number) => (
                                  <div key={`${rec?.diagnostic || "diagnostic"}-${index}`} className="rounded-lg border border-amber-200 bg-white/70 p-3">
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="text-sm font-black text-slate-900">{rec?.label || rec?.diagnostic || "補助診断"}</div>
                                      <span className="rounded-full bg-amber-100 px-2 py-1 text-[10px] font-black text-amber-800">
                                        {rec?.status === "calculated" ? "算出済み" : "推奨"}
                                      </span>
                                    </div>
                                    {rec?.reason && <div className="mt-1 text-xs leading-relaxed text-slate-700">{rec.reason}</div>}
                                    {rec?.use_as && <div className="mt-1 text-[11px] leading-relaxed text-slate-500">{rec.use_as}</div>}
                                  </div>
                                ))}
                              </div>
                            </div>
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
                            scoreBase={getScreeningScore(result) || 50}
                          />
                          <DataSourceSummaryCard summary={result.data_source_summary} />
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
              score={getScreeningScore(result)}
              modelDecision={result?.hantei || ""}
              industry_major={result?.industry_major || formData.industry_major || ""}
              formData={formData}
              estatContext={result?.estat_context || null}
              onChatLoaded={setGunshiText}
              highlightCompanies={shionPastCompanies}
            />
          </div>
          
        </div>
      </div>
    </div>
  );
}
