"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";
import { Activity, ArrowRight, Bot, Calculator, Eye, MessageSquare, Network, PieChart, AlignLeft, Share2, AlertTriangle, ListOrdered, BadgeInfo, DollarSign, Database, ChevronDown, ChartNoAxesCombined, FileOutput, SlidersHorizontal, ScanText, ShieldCheck, XCircle, Minus, Swords, Save, Trash2 } from "lucide-react";
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

type DemoSimilarPastCase = {
  id?: number;
  demoCaseId?: string;
  sourceCaseId?: string;
  companyName: string;
  period: string;
  industry: string;
  industryMajor?: string;
  industrySub?: string;
  salesDept?: string;
  score: number;
  decision: string;
  outcome: string;
  similarity: string;
  actionTaken: string;
  lesson: string;
  difference: string;
  source?: string;
  similarityScore?: number;
  similarityReasons?: string[];
  formSnapshot?: Record<string, any>;
  resultSnapshot?: Record<string, any>;
};

const SHION_REVIEW_IMAGE = "/lease-intelligence/moods/focus.webp";
const SCREENING_RETURN_STATE_KEY = "lease-screening-return-state";
const SCREENING_DRAFT_VERSION = 1;
const SCREENING_DRAFT_SAVE_DELAY_MS = 300;

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

type DemoScreeningCase = {
  id: string;
  title: string;
  tone: string;
  summary: string;
  learningPoint: string;
  reviewFocus: string[];
  similarPastCases?: DemoSimilarPastCase[];
  data: Partial<ScoringFormData>;
};

const demoScreeningCases: DemoScreeningCase[] = [
  {
    id: "stable-manufacturing",
    title: "既存先・製造業",
    tone: "通しやすい案件",
    summary: "メイン先、黒字、自己資本あり。工作機械更新の標準的なリース案件。",
    learningPoint: "標準的に通る案件では、紫苑が何を安心材料として拾うかを見ます。",
    reviewFocus: ["返済原資と自己資本", "物件用途の明確さ", "稟議に残す承認理由"],
    similarPastCases: [
      {
        companyName: "柴犬精密工業",
        period: "2025年上期",
        industry: "金属製品製造業",
        score: 82.4,
        decision: "承認",
        outcome: "成約・延滞なし",
        similarity: "既存メイン先、工作機械更新、黒字基調、自己資本厚め",
        actionTaken: "受注増加の根拠資料と既存機の稼働状況を添付し、通常承認で稟議化。",
        lesson: "標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した。",
        difference: "過去事例は受注先が固定的。今回デモは受注増の説明を営業メモで補う必要がある。",
      },
      {
        companyName: "ビーグル加工",
        period: "2024年下期",
        industry: "金属加工業",
        score: 76.8,
        decision: "条件付き承認",
        outcome: "成約・初回検収完了",
        similarity: "加工設備の更改、既存取引あり、物件保全が見やすい",
        actionTaken: "見積・型式・設置場所の確認を条件に、設備更新目的を承認理由へ明記。",
        lesson: "物件が強い案件は、財務だけでなく回収可能性を押さえると通しやすい。",
        difference: "過去事例は非メイン先。今回デモはメイン先なので銀行接点を安心材料にできる。",
      },
    ],
    data: {
      company_no: "900101",
      company_name: "デモ精密工業",
      industry_major: "E 製造業",
      industry_sub: "24 金属製品製造業",
      industry_detail: "精密部品加工 工作機械更新",
      grade: "② 4-6先",
      customer_type: "既存先",
      main_bank: "メイン先",
      competitor: "競合なし",
      num_competitors: "0",
      deal_occurrence: "更改・増設",
      deal_source: "銀行紹介",
      sales_dept: "猫営業部",
      nenshu: 850,
      gross_profit: 210,
      op_profit: 48,
      ord_profit: 45,
      net_income: 28,
      depreciation: 35,
      dep_expense: 35,
      rent: 8,
      rent_expense: 8,
      machines: 180,
      other_assets: 120,
      net_assets: 260,
      total_assets: 720,
      bank_credit: 180,
      lease_credit: 45,
      contracts: 4,
      contract_type: "一般",
      lease_term: 60,
      acceptance_year: new Date().getFullYear(),
      acquisition_cost: 55,
      asset_name: "製造設備・工作機械",
      asset_detail: "マシニングセンタ更新 2台",
      asset_purpose: "既存受注の増加に伴う加工能力増強",
      asset_location: "本社工場 第2ライン",
      asset_evidence_level: "見積・型式確認済",
      asset_score: 78,
      qual_corr_company_history: "良好",
      qual_corr_customer_stability: "良好",
      qual_corr_repayment_history: "良好",
      qual_corr_business_future: "良好",
      qual_corr_equipment_purpose: "良好",
      qual_corr_main_bank: "良好",
      passion_text: "既存メイン先。受注増に伴う更新投資で、返済原資と物件用途の説明がしやすい。",
      intuition: 4,
    },
  },
  {
    id: "borderline-transport",
    title: "境界・運送業",
    tone: "条件付き承認向き",
    summary: "売上はあるが利益薄め。車両増車で、燃料費と運転手確保を確認したい案件。",
    learningPoint: "境界案件では、点数よりも条件付きで残す確認事項が主役です。",
    reviewFocus: ["燃料費・人件費の上昇耐性", "競合条件との差分", "追加確認すべき承認条件"],
    similarPastCases: [
      {
        companyName: "ハスキー運輸",
        period: "2025年夏",
        industry: "道路貨物運送業",
        score: 63.2,
        decision: "条件付き承認",
        outcome: "成約・採算は維持",
        similarity: "利益率薄め、増車、競合あり、非メイン先",
        actionTaken: "燃料サーチャージ契約、主要荷主との配送継続確認、競合金利との差分説明を条件化。",
        lesson: "運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する。",
        difference: "過去事例は既存荷主比率が高かった。今回デモは新規ルート分の採算確認が重い。",
      },
      {
        companyName: "ダックス物流",
        period: "2024年秋",
        industry: "一般貨物運送業",
        score: 58.7,
        decision: "見送り",
        outcome: "競合へ流出",
        similarity: "増車理由あり、競合金利あり、銀行支援が弱い",
        actionTaken: "燃料費上昇時の資金繰り表と運転手確保計画を依頼したが、資料不足で見送り。",
        lesson: "競合に急かされる案件ほど、資料不足のまま金利で追うと説明責任が残らない。",
        difference: "今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある。",
      },
    ],
    data: {
      company_no: "900202",
      company_name: "デモ北関東物流",
      industry_major: "H 運輸業・郵便業",
      industry_sub: "44 道路貨物運送業",
      industry_detail: "一般貨物 運送業 車両増車",
      grade: "② 4-6先",
      customer_type: "既存先",
      main_bank: "非メイン先",
      competitor: "競合あり",
      competitor_rate: 2.1,
      num_competitors: "2",
      deal_occurrence: "競合切替",
      deal_source: "その他",
      sales_dept: "鳥営業部",
      nenshu: 620,
      gross_profit: 92,
      op_profit: 9,
      ord_profit: 6,
      net_income: 3,
      depreciation: 18,
      dep_expense: 18,
      rent: 12,
      rent_expense: 12,
      machines: 95,
      other_assets: 70,
      net_assets: 42,
      total_assets: 430,
      bank_credit: 210,
      lease_credit: 68,
      contracts: 3,
      contract_type: "一般",
      lease_term: 60,
      acceptance_year: new Date().getFullYear(),
      acquisition_cost: 38,
      asset_name: "車両・運搬車",
      asset_detail: "大型トラック 2台",
      asset_purpose: "新規配送ルート対応の増車",
      asset_location: "栃木県小山市 営業所",
      asset_evidence_level: "見積あり",
      asset_score: 62,
      qual_corr_company_history: "普通",
      qual_corr_customer_stability: "普通",
      qual_corr_repayment_history: "良好",
      qual_corr_business_future: "普通",
      qual_corr_equipment_purpose: "良好",
      qual_corr_main_bank: "普通",
      passion_text: "増車理由はあるが、利益率が薄く燃料費・人件費上昇の影響確認が必要。競合条件との差分も確認したい。",
      intuition: 3,
    },
  },
  {
    id: "watch-service-new",
    title: "新規先・サービス業",
    tone: "慎重審査",
    summary: "新規先、薄い自己資本、出店設備。事業計画と撤退時物件価値を確認したい案件。",
    learningPoint: "厳しめの案件では、否決だけでなく何を確認すれば検討余地が残るかを見ます。",
    reviewFocus: ["新規先・赤字の重さ", "出店計画の根拠", "撤退時の物件価値"],
    similarPastCases: [
      {
        companyName: "プードルフード",
        period: "2025年春",
        industry: "飲食店",
        score: 46.5,
        decision: "条件再設計",
        outcome: "保証追加後に小口で成約",
        similarity: "新規先、出店設備、自己資本薄め、厨房機器",
        actionTaken: "初期投資を圧縮し、保証人追加・自己資金投入・厨房機器のみの小口化で再審議。",
        lesson: "飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的。",
        difference: "過去事例は既存店の売上実績があった。今回デモは新店舗計画の根拠確認がより重要。",
      },
      {
        companyName: "コーギーカフェ",
        period: "2024年冬",
        industry: "飲食サービス業",
        score: 39.8,
        decision: "否決",
        outcome: "自己資金不足で出店延期",
        similarity: "新規開拓、赤字、銀行支援弱め、内装設備比率が高い",
        actionTaken: "撤退時の物件処分価値が弱く、売上計画も未検証だったため否決。",
        lesson: "内装・造作比率が高い飲食案件は、設備の再販価値だけでは保全になりにくい。",
        difference: "今回デモは厨房機器も含むため、リース対象を再販可能な設備に絞れば再検討できる。",
      },
    ],
    data: {
      company_no: "900303",
      company_name: "デモフードサービス",
      industry_major: "M 宿泊業・飲食サービス業",
      industry_sub: "76 飲食店",
      industry_detail: "新店舗 厨房設備 出店",
      grade: "② 7-9先",
      customer_type: "新規先",
      main_bank: "非メイン先",
      competitor: "競合あり",
      competitor_rate: 2.8,
      num_competitors: "1",
      deal_occurrence: "新規開拓",
      deal_source: "その他",
      sales_dept: "犬営業部",
      nenshu: 180,
      gross_profit: 58,
      op_profit: -6,
      ord_profit: -8,
      net_income: -7,
      depreciation: 4,
      dep_expense: 4,
      rent: 18,
      rent_expense: 18,
      machines: 12,
      other_assets: 35,
      net_assets: 8,
      total_assets: 95,
      bank_credit: 65,
      lease_credit: 0,
      contracts: 0,
      contract_type: "一般",
      lease_term: 48,
      acceptance_year: new Date().getFullYear(),
      acquisition_cost: 24,
      asset_name: "飲食店設備",
      asset_detail: "厨房機器・内装設備一式",
      asset_purpose: "新店舗開業に伴う初期設備",
      asset_location: "埼玉県さいたま市 新店舗",
      asset_evidence_level: "見積あり",
      asset_score: 45,
      qual_corr_company_history: "懸念あり",
      qual_corr_customer_stability: "懸念あり",
      qual_corr_repayment_history: "未選択",
      qual_corr_business_future: "普通",
      qual_corr_equipment_purpose: "普通",
      qual_corr_main_bank: "懸念あり",
      passion_text: "新規先かつ赤字。出店計画の根拠、自己資金、撤退時の物件処分可能性を確認しないと通しにくい。",
      intuition: 2,
    },
  },
];

const findDemoScreeningCase = (data: Partial<ScoringFormData>) => {
  const companyNo = String(data.company_no || "");
  const companyName = String(data.company_name || "");
  return demoScreeningCases.find((demoCase) =>
    demoCase.data.company_no === companyNo ||
    (companyName && demoCase.data.company_name === companyName)
  ) || null;
};

const parseExperienceSnapshot = (value: unknown): Record<string, any> | undefined => {
  if (!value) return undefined;
  if (typeof value === "object" && !Array.isArray(value)) return value as Record<string, any>;
  if (typeof value !== "string") return undefined;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, any>
      : undefined;
  } catch {
    return undefined;
  }
};

const normalizeExperienceCase = (raw: any): DemoSimilarPastCase => ({
  id: Number(raw?.id || 0) || undefined,
  demoCaseId: String(raw?.demo_case_id || raw?.demoCaseId || ""),
  sourceCaseId: String(raw?.source_case_id || raw?.sourceCaseId || ""),
  companyName: String(raw?.company_name || raw?.companyName || "名称未設定"),
  period: String(raw?.period || ""),
  industry: String(raw?.industry_sub || raw?.industry || raw?.industry_major || ""),
  industryMajor: String(raw?.industry_major || raw?.industryMajor || ""),
  industrySub: String(raw?.industry_sub || raw?.industrySub || ""),
  salesDept: String(raw?.sales_dept || raw?.salesDept || ""),
  score: Number(raw?.score || 0),
  decision: String(raw?.decision || ""),
  outcome: String(raw?.outcome || ""),
  similarity: String(raw?.similarity || ""),
  actionTaken: String(raw?.action_taken || raw?.actionTaken || ""),
  lesson: String(raw?.lesson || ""),
  difference: String(raw?.difference || ""),
  source: String(raw?.source || ""),
  similarityScore: Number(raw?.similarity_score ?? raw?.similarityScore ?? 0),
  similarityReasons: Array.isArray(raw?.similarity_reasons)
    ? raw.similarity_reasons.map((reason: unknown) => String(reason)).filter(Boolean)
    : [],
  formSnapshot: parseExperienceSnapshot(raw?.form_snapshot ?? raw?.formSnapshot),
  resultSnapshot: parseExperienceSnapshot(raw?.result_snapshot ?? raw?.resultSnapshot),
});

const fallbackExperienceCasesForDemo = (demoCaseId: string) =>
  demoScreeningCases.find((demoCase) => demoCase.id === demoCaseId)?.similarPastCases || [];

const normalizeReviewText = (text: string) =>
  (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .trim();

const buildDemoSimilarPastCaseBlock = (cases: DemoSimilarPastCase[]) => {
  if (!cases.length) return "";
  return [
    "【保存済み経験ケース】",
    "次の事例は、DBに保存された類似経験ケースです。今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください。",
    ...cases.slice(0, 3).map((item, index) => [
      `事例${index + 1}: ${item.companyName} / ${item.period} / ${item.industry}`,
      item.similarityScore ? `類似度: ${Math.round(item.similarityScore)} / 理由: ${(item.similarityReasons || []).join("・") || "未計算"}` : "",
      `スコア・判断: ${item.score.toFixed(1)}点 / ${item.decision} / ${item.outcome}`,
      `似ている点: ${item.similarity}`,
      `当時の対応: ${item.actionTaken}`,
      `得た教訓: ${item.lesson}`,
      `今回との差分: ${item.difference}`,
    ].filter(Boolean).join("\n")),
  ].join("\n");
};

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

const buildShionReviewPrompt = (
  result: Record<string, any>,
  data: ScoringFormData,
  pastReviews: PastShionScreeningReview[] = [],
  experienceCases: DemoSimilarPastCase[] = [],
) => {
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
    `・Q_risk: ${result.quantum_risk != null ? `${Number(result.quantum_risk).toFixed(1)}（0-100スケール、35以上で要注意・60以上で強警戒）` : "未算出"}`,
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
    const flagTitles = flags
      .slice(0, 5)
      .map((f) => (typeof f === "string" ? f : (f as { title?: string })?.title ?? ""))
      .filter(Boolean);
    if (flagTitles.length) {
      lines.push(`・AURION警戒: ${flagTitles.join(" / ")}`);
    }
  }
  if (Array.isArray(result.default_warnings) && result.default_warnings.length) {
    lines.push(`・デフォルト率警告: ${result.default_warnings.slice(0, 3).join(" / ")}`);
  }
  const demoPastCaseBlock = buildDemoSimilarPastCaseBlock(experienceCases);
  if (demoPastCaseBlock) {
    lines.push("", demoPastCaseBlock);
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
    ["総合スコア", item.resultSnapshot?.score_base ?? item.resultSnapshot?.score ?? item.score],
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
  const demoCase = findDemoScreeningCase(data);
  const [selectedCase, setSelectedCase] = useState<DemoSimilarPastCase | null>(null);
  if (!demoCase && !experienceCases.length) return null;

  return (
    <section className="rounded-2xl border border-sky-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm font-black text-sky-950">
            <Database className="h-4 w-4 text-sky-600" />
            保存済み経験ケース
          </div>
          <p className="mt-1 text-xs font-bold leading-relaxed text-sky-700">
            {demoCase?.title || data.industry_sub || "この案件"} と同じ論点で、後から再利用できる経験データを表示します。
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
        {experienceCases.map((item) => (
          <article
            key={`${item.id || item.demoCaseId || demoCase?.id}-${item.companyName}`}
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
  const [draftRestored, setDraftRestored] = useState(false);
  const [lastDraftSavedAt, setLastDraftSavedAt] = useState<Date | null>(null);
  const [experienceCasesByDemo, setExperienceCasesByDemo] = useState<Record<string, DemoSimilarPastCase[]>>({});
  const [currentExperienceCases, setCurrentExperienceCases] = useState<DemoSimilarPastCase[]>([]);
  const [experienceSaving, setExperienceSaving] = useState(false);
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

  useEffect(() => {
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
        activeTab?: "input" | "analysis";
        savedAt?: string;
      };
      if (saved.formData) setFormData(saved.formData);
      if (saved.result) setResult(saved.result);
      if (typeof saved.gunshiText === "string") setGunshiText(saved.gunshiText);
      if (saved.shionReview) setShionReview(saved.shionReview);
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
          activeTab,
          savedAt: savedAt.toISOString(),
        }));
        setLastDraftSavedAt(savedAt);
      } catch (error) {
        console.warn("Screening draft save failed", error);
      }
    }, SCREENING_DRAFT_SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draftRestored, formData, result, gunshiText, shionReview, activeTab]);

  const buildExperienceCaseQuery = (
    demoCaseId: string,
    targetFormData: Partial<ScoringFormData>,
    targetResult: any = null,
  ) => ({
    demo_case_id: demoCaseId,
    industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
    industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
    company_name: targetFormData.company_name || "",
    asset_name: targetFormData.asset_name || targetFormData.asset_detail || "",
    customer_type: targetFormData.customer_type || "",
    main_bank: targetFormData.main_bank || "",
    competitor: targetFormData.competitor || "",
    outcome_status: targetResult?.final_status || targetResult?.result_status || targetResult?.hantei || "",
    score: targetResult?.score_base ?? targetResult?.score ?? "",
    limit: 8,
  });

  const hasExperienceSearchContext = (targetFormData: Partial<ScoringFormData>, targetResult: any = null) =>
    Boolean(
      targetFormData.industry_sub ||
      targetFormData.industry_major ||
      targetFormData.asset_name ||
      targetFormData.customer_type ||
      targetFormData.main_bank ||
      targetFormData.competitor ||
      targetResult?.hantei ||
      targetResult?.score_base ||
      targetResult?.score,
    );

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
      const nextCases = cases.length || !demoCaseId ? cases : fallbackExperienceCasesForDemo(demoCaseId);
      if (demoCaseId) {
        setExperienceCasesByDemo((prev) => ({ ...prev, [demoCaseId]: nextCases }));
      } else {
        setCurrentExperienceCases(nextCases);
      }
      return nextCases;
    } catch {
      const fallback = demoCaseId ? fallbackExperienceCasesForDemo(demoCaseId) : [];
      if (demoCaseId) {
        setExperienceCasesByDemo((prev) => ({ ...prev, [demoCaseId]: fallback }));
      } else {
        setCurrentExperienceCases([]);
      }
      return fallback;
    }
  };

  const fetchExperienceCasesForDemo = (demoCaseId: string) => {
    const demoCase = demoScreeningCases.find((item) => item.id === demoCaseId);
    if (!demoCase) return Promise.resolve([]);
    return fetchExperienceCasesForContext(demoCaseId, demoCase.data, null);
  };

  useEffect(() => {
    demoScreeningCases.forEach((demoCase) => {
      void fetchExperienceCasesForDemo(demoCase.id);
    });
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
      const demoCase = findDemoScreeningCase(targetFormData);
      const experienceCases = await fetchExperienceCasesForContext(demoCase?.id || "", targetFormData, targetResult);
      if (seq !== shionReviewRequestSeq.current) return;
      const promptText = buildShionReviewPrompt(targetResult, targetFormData, pastReviews, experienceCases);
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
    setActiveTab("input");
    window.localStorage.removeItem(SCREENING_RETURN_STATE_KEY);
    setLastDraftSavedAt(null);
  };

  const handleSubmit = async (targetFormData: ScoringFormData = formData) => {
    setLoading(true);
    setShionReview(null);
    setShionReviewError("");
    try {
      const res = await apiClient.post(`/api/score/full`, toThousandYenPayload(targetFormData));
      setResult(res.data);
      setActiveTab("analysis");
      void fetchExperienceCasesForContext(findDemoScreeningCase(targetFormData)?.id || "", targetFormData, res.data);
      void requestShionReview(res.data, targetFormData);

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
      alert(getScreeningErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const loadDemoCase = (demoCase: DemoScreeningCase, runImmediately = false) => {
    shionReviewRequestSeq.current += 1;
    const nextFormData = {
      ...defaultFormData,
      ...demoCase.data,
      strength_tags: demoCase.data.strength_tags || [],
    } as ScoringFormData;
    setFormData(nextFormData);
    setResult(null);
    setGunshiText("");
    setShionReview(null);
    setShionReviewError("");
    setShionFeedbackSaving(false);
    setCurrentExperienceCases([]);
    setActiveTab("input");
    if (runImmediately) {
      void handleSubmit(nextFormData);
    }
  };

  const saveCurrentExperienceCase = async () => {
    if (!result || experienceSaving) return;
    const demoCase = findDemoScreeningCase(formData);
    const score = Number(result.score_base ?? result.score ?? 0);
    setExperienceSaving(true);
    try {
      await apiClient.post("/api/screening-experience-cases", {
        demo_case_id: demoCase?.id || "",
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
      await fetchExperienceCasesForContext(demoCase?.id || "", formData, result);
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
      version: SCREENING_DRAFT_VERSION,
      formData,
      result,
      gunshiText,
      shionReview,
      activeTab: "analysis",
      savedAt: new Date().toISOString(),
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
                <section className="rounded-2xl border border-indigo-100 bg-white p-4 shadow-sm">
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <div className="text-[10px] font-black uppercase tracking-widest text-indigo-500">Sample Cases</div>
                      <h3 className="mt-1 text-base font-black text-slate-800">サンプル案件で動きを見る</h3>
                      <p className="mt-1 text-xs font-bold text-slate-500">
                        初めて使う人は、まず3件で「通る・境界・慎重」の違いを見てください。数字入力なしで審査結果と紫苑レビューまで進めます。
                      </p>
                    </div>
                    <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[11px] font-black text-slate-500">
                      入力例 + 見どころ付き
                    </div>
                  </div>
                  <div className="mt-4 grid gap-3 lg:grid-cols-3">
                    {demoScreeningCases.map((demoCase) => (
                      <div key={demoCase.id} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <h4 className="text-sm font-black text-slate-800">{demoCase.title}</h4>
                            <p className="mt-1 text-[11px] font-black text-indigo-600">{demoCase.tone}</p>
                          </div>
                          <span className="rounded-full bg-white px-2 py-1 text-[10px] font-black text-slate-500">
                            {demoCase.data.acquisition_cost}百万円
                          </span>
                        </div>
                        <p className="mt-2 min-h-10 text-xs font-bold leading-relaxed text-slate-500">{demoCase.summary}</p>
                        <div className="mt-3 rounded-lg border border-white bg-white p-2">
                          <p className="text-[11px] font-black text-slate-700">この案件の見どころ</p>
                          <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-500">{demoCase.learningPoint}</p>
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {demoCase.reviewFocus.map((focus) => (
                              <span key={focus} className="rounded-full bg-indigo-50 px-2 py-1 text-[10px] font-black text-indigo-700">
                                {focus}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div className="mt-2 rounded-lg border border-sky-100 bg-sky-50 p-2">
                          <p className="text-[11px] font-black text-sky-800">過去類似デモ</p>
                          <div className="mt-1 space-y-1">
                            {(experienceCasesByDemo[demoCase.id] || fallbackExperienceCasesForDemo(demoCase.id)).map((item) => (
                              <div key={item.companyName} className="text-[11px] font-bold leading-relaxed text-sky-700">
                                {item.companyName}: {item.decision} / {item.outcome}
                              </div>
                            ))}
                          </div>
                        </div>
                        <div className="mt-3 grid grid-cols-2 gap-2">
                          <button
                            type="button"
                            onClick={() => loadDemoCase(demoCase, false)}
                            className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-black text-slate-600 transition-colors hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                          >
                            読み込み
                          </button>
                          <button
                            type="button"
                            onClick={() => loadDemoCase(demoCase, true)}
                            disabled={loading}
                            className="rounded-lg bg-indigo-600 px-3 py-2 text-xs font-black text-white shadow-sm transition-colors hover:bg-indigo-700 disabled:bg-slate-300"
                          >
                            読み込んで審査
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

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
                    <JudgmentFlowStrip />
                    <CurrentIssueCard result={result} data={formData} />
                    <RingiPolicyCard result={result} data={formData} />
                    <DemoSimilarPastCasesCard
                      data={formData}
                      experienceCases={(() => {
                        const demoCase = findDemoScreeningCase(formData);
                        return demoCase
                          ? (experienceCasesByDemo[demoCase.id] || fallbackExperienceCasesForDemo(demoCase.id))
                          : currentExperienceCases;
                      })()}
                      onSaveExperience={saveCurrentExperienceCase}
                      saving={experienceSaving}
                    />
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
