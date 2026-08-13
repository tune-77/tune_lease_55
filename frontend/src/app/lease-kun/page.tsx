"use client";

import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Activity, ChevronDown, MessageSquare, CheckCircle2, Trash2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { toThousandYenPayload } from '../../lib/scoringUnits';
import { extractPrefectureFromText } from '@/lib/prefecture';
import { CurrentIssueCard, RingiPolicyCard } from '../../components/analysis/IssuePolicyCards';
import CaseRegistrationForm from '../../components/analysis/CaseRegistrationForm';
import { ShionScreeningReviewCard } from '../../components/analysis/ShionReviewCard';
import { useShionScreeningReview } from '../../lib/useShionScreeningReview';
import type { ShionReviewFeedback } from '../../lib/shionReview';
import { parseHumanNumberInput } from '@/lib/numberInput';

// --- 型定義 ---
type Message = {
  role: 'bot' | 'user' | 'humor';
  text: React.ReactNode;
};

type ScoreResult = {
  score: number;
  score_base?: number;
  hantei: string;
  score_borrower?: number;
  company_name?: string;
  asset_name?: string;
  asset_location?: string;
  industry_sub?: string;
  sales_dept?: string;
  quantum_risk?: number;
  case_id?: string;
};

type ConditionalApprovalAction = string | {
  action?: string;
  reason?: string;
};

type LeaseKunFullResult = ScoreResult & Record<string, unknown> & {
  conditional_approval_actions?: ConditionalApprovalAction[];
  umap_anomaly_score?: number;
};

type IndustryMasterEntry = {
  mapping?: string;
  sub?: { [sub: string]: string };
  [key: string]: unknown;
};
type IndustryMaster = { [major: string]: IndustryMasterEntry | string[] };

function extractSubs(entry: IndustryMasterEntry | string[] | undefined): string[] {
  if (!entry) return [];
  if (Array.isArray(entry)) return entry.filter(Boolean);
  if (entry.sub && typeof entry.sub === 'object') return Object.keys(entry.sub);
  return Object.keys(entry).filter(k => k !== 'mapping');
}

// --- 初期データ ---
const STEPS = [
  "企業・業種", "取引と競合", "リース物件", "損益計算", "資産情報",
  "経費・減価償却", "信用情報", "契約条件", "定性評価", "最終確認"
];

const PHASE_LABELS = {
  wizard: "入力",
  analysis: "分析結果",
  register: "結果登録",
  done: "完了",
} as const;

const DRAFT_STORAGE_KEY = 'lease-kun-draft-v1';

const INITIAL_FORM_DATA = {
  // Step 0
  company_no: '', company_name: '',
  industry_major: 'D 建設業', industry_sub: '06 総合工事業',
  // Step 1
  sales_dept: '未設定',
  main_bank: 'メイン先', competitor: '競合なし',
  num_competitors: '未入力', deal_source: 'その他', deal_occurrence: '不明',
  customer_type: '新規先',
  // Step 2
  asset_name: 'IT・OA機器',
  asset_location: '',
  // Step 3 (PL)
  nenshu: '', gross_profit: '', op_profit: '', ord_profit: '', net_income: '',
  // Step 4 (BS)
  total_assets: '', net_assets: '', machines: '', other_assets: '',
  // Step 5 (経費)
  depreciation: '', dep_expense: '', rent: '', rent_expense: '',
  // Step 6 (信用)
  grade: '②4-6 (標準)', contracts: '', bank_credit: '', lease_credit: '',
  // Step 7 (契約)
  contract_type: '一般',
  lease_term: 60, acceptance_year: new Date().getFullYear(), acquisition_cost: '',
  // Step 8 (定性)
  qual_corr_company_history: '未選択',
  qual_corr_customer_stability: '未選択',
  qual_corr_repayment_history: '未選択',
  qual_corr_business_future: '未選択',
  qual_corr_equipment_purpose: '未選択',
  qual_corr_main_bank: '未選択',
  passion_text: '',
  // Step 9
  intuition: 3
};

type LeaseKunFormData = typeof INITIAL_FORM_DATA;

type LeaseKunDraft = {
  version: 1;
  step: number;
  formData: LeaseKunFormData;
  updatedAt: string;
};

type QuickAmount = {
  label: string;
  value: string;
};

type FocusCheck = {
  title: string;
  reason: string;
  tone: 'risk' | 'condition' | 'sales';
};

const SALES_ASSET_AMOUNTS: QuickAmount[] = [
  { label: '50', value: '50' },
  { label: '100', value: '100' },
  { label: '300', value: '300' },
  { label: '1000', value: '1000' },
];

const PROFIT_AMOUNTS: QuickAmount[] = [
  { label: '-5', value: '-5' },
  { label: '0', value: '0' },
  { label: '5', value: '5' },
  { label: '10', value: '10' },
];

const SMALL_COST_AMOUNTS: QuickAmount[] = [
  { label: '0', value: '0' },
  { label: '1', value: '1' },
  { label: '3', value: '3' },
  { label: '5', value: '5' },
];

const CREDIT_AMOUNTS: QuickAmount[] = [
  { label: '0', value: '0' },
  { label: '10', value: '10' },
  { label: '30', value: '30' },
  { label: '100', value: '100' },
];

const ASSET_PRICE_AMOUNTS: QuickAmount[] = [
  { label: '1', value: '1' },
  { label: '3', value: '3' },
  { label: '5', value: '5' },
  { label: '10', value: '10' },
  { label: '30', value: '30' },
];

const TERM_AMOUNTS: QuickAmount[] = [
  { label: '36', value: '36' },
  { label: '48', value: '48' },
  { label: '60', value: '60' },
  { label: '72', value: '72' },
];

function parseMillionInput(value: string | number, fallback = 0): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback;
  const parsed = parseHumanNumberInput(value);
  return parsed ?? fallback;
}

function isPositiveMillionInput(value: string | number): boolean {
  return parseMillionInput(value, NaN) > 0;
}

function hasMeaningfulDraft(formData: LeaseKunFormData): boolean {
  return Boolean(
    formData.company_no.trim() ||
    formData.company_name.trim() ||
    formData.asset_location.trim() ||
    formData.nenshu.trim() ||
    formData.total_assets.trim() ||
    formData.acquisition_cost.trim() ||
    formData.passion_text.trim() ||
    formData.sales_dept !== INITIAL_FORM_DATA.sales_dept ||
    formData.industry_major !== INITIAL_FORM_DATA.industry_major ||
    formData.industry_sub !== INITIAL_FORM_DATA.industry_sub ||
    formData.asset_name !== INITIAL_FORM_DATA.asset_name
  );
}

function readLeaseKunDraft(): LeaseKunDraft | null {
  try {
    const raw = window.localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<LeaseKunDraft>;
    if (parsed.version !== 1 || typeof parsed.step !== 'number' || !parsed.formData) return null;
    const boundedStep = Math.min(Math.max(Math.floor(parsed.step), 0), STEPS.length - 1);
    return {
      version: 1,
      step: boundedStep,
      formData: { ...INITIAL_FORM_DATA, ...parsed.formData },
      updatedAt: typeof parsed.updatedAt === 'string' ? parsed.updatedAt : new Date().toISOString(),
    };
  } catch {
    return null;
  }
}

function clearLeaseKunDraft(): void {
  window.localStorage.removeItem(DRAFT_STORAGE_KEY);
}

function getScreeningScoreValue(result: LeaseKunFullResult): number {
  return Number(result.score_base ?? result.score ?? 0);
}

function buildFocusChecks(result: LeaseKunFullResult, data: LeaseKunFormData): FocusCheck[] {
  const checks: FocusCheck[] = [];
  const score = getScreeningScoreValue(result);
  const qRisk = typeof result.quantum_risk === 'number' ? result.quantum_risk : 0;
  const sales = parseMillionInput(data.nenshu);
  const assetPrice = parseMillionInput(data.acquisition_cost);
  const leaseCredit = parseMillionInput(data.lease_credit);
  const contracts = parseMillionInput(data.contracts);
  const isNewCustomer = data.customer_type.includes('新規') || (leaseCredit <= 0 && contracts <= 0);
  const hasCompetitor = data.competitor === '競合あり' || data.num_competitors !== '未入力' && data.num_competitors !== '0社';
  const assetToSalesRatio = sales > 0 ? assetPrice / sales : 0;

  if (qRisk >= 60) {
    checks.push({
      title: '入力値の桁と整合性',
      reason: 'Q_risk が高いため、売上・総資産・物件価格の桁違いを先に潰す。',
      tone: 'risk',
    });
  } else if (qRisk >= 35) {
    checks.push({
      title: '数字の違和感チェック',
      reason: 'Q_risk が出ているため、主要数値の前提を軽く確認する。',
      tone: 'risk',
    });
  }

  if (score < 60) {
    checks.push({
      title: '否決理由を条件で戻せるか',
      reason: 'スコアが低いため、保証・前受金・期間短縮で審議可能域に戻せるかを見る。',
      tone: 'condition',
    });
  } else if (score < 71) {
    checks.push({
      title: '境界スコアの承認条件',
      reason: '承認/否認の境目なので、追加確認と条件設定で説明できるかを見る。',
      tone: 'condition',
    });
  }

  if (isNewCustomer) {
    checks.push({
      title: '新規先としての支援材料',
      reason: 'リース実績が薄い可能性があるため、銀行支援・既存取引・回収原資を確認する。',
      tone: 'condition',
    });
  }

  if (hasCompetitor) {
    checks.push({
      title: '競合条件と採算下限',
      reason: '他社条件に寄せすぎず、採算を守れる下限と失注時の回収理由を決める。',
      tone: 'sales',
    });
  }

  if (assetToSalesRatio >= 0.25) {
    checks.push({
      title: '物件価格と売上規模のバランス',
      reason: '物件価格が売上に対して重いため、稼働目的・回収期間・支払原資を確認する。',
      tone: 'risk',
    });
  }

  checks.push({
    title: '物件の使い道と稼働開始',
    reason: '最後に、何に使い、いつ売上や効率に効く設備かを短く押さえる。',
    tone: 'sales',
  });

  const seen = new Set<string>();
  return checks.filter((check) => {
    if (seen.has(check.title)) return false;
    seen.add(check.title);
    return true;
  }).slice(0, 3);
}

function FocusCheckCard({ checks }: { checks: FocusCheck[] }) {
  const toneClass = {
    risk: 'border-rose-200 bg-rose-50 text-rose-800',
    condition: 'border-amber-200 bg-amber-50 text-amber-900',
    sales: 'border-sky-200 bg-sky-50 text-sky-900',
  };

  return (
    <section className="rounded-2xl border-2 border-[#1A1A2E] bg-white px-4 py-3 shadow-sm">
      <div className="text-[11px] font-black uppercase tracking-wider text-slate-400">今回まず見る3点</div>
      <div className="mt-2 space-y-2">
        {checks.map((check, index) => (
          <div key={check.title} className={`rounded-xl border px-3 py-2 ${toneClass[check.tone]}`}>
            <div className="flex items-center gap-2 text-xs font-black">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/80 text-[10px]">{index + 1}</span>
              <span>{check.title}</span>
            </div>
            <p className="mt-1 text-[11px] font-bold leading-relaxed opacity-90">{check.reason}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function AmountChips({
  amounts,
  activeValue,
  onSelect,
}: {
  amounts: QuickAmount[];
  activeValue: string | number;
  onSelect: (value: string) => void;
}) {
  const active = String(activeValue || '');
  return (
    <div className="mt-1.5 grid grid-cols-4 gap-1">
      {amounts.map((amount) => (
        <button
          key={`${amount.label}-${amount.value}`}
          type="button"
          onClick={() => onSelect(amount.value)}
          className={`h-7 rounded-lg border text-[11px] font-black transition-colors ${
            active === amount.value
              ? 'border-[#E8A838] bg-[#E8A838] text-white'
              : 'border-slate-200 bg-white text-slate-500'
          }`}
        >
          {amount.label}
        </button>
      ))}
    </div>
  );
}

// --- メインコンポーネント ---
export default function LeaseKunWizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<Message[]>([
    { role: 'bot', text: 'はじめまして！リースくんです 🎩 まず企業名と業種から教えてね！' }
  ]);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [draftRestored, setDraftRestored] = useState(false);
  const [draftSavedAt, setDraftSavedAt] = useState<string>('');
  const [doneMessage, setDoneMessage] = useState('結果登録まで完了しました！');
  const draftReadyRef = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 入力完了後は screening / register 画面へ離脱させず、この画面内で
  // 分析結果の確認 → 結果登録まで一気通貫で完結させる
  const [phase, setPhase] = useState<'wizard' | 'analysis' | 'register' | 'done'>('wizard');
  const [fullResult, setFullResult] = useState<LeaseKunFullResult | null>(null);
  const shionReview = useShionScreeningReview();
  const shionRequestedForCaseId = useRef<string | null>(null);

  const [industryMaster, setIndustryMaster] = useState<IndustryMaster>({});
  const [majors, setMajors] = useState<string[]>([]);
  const [subs, setSubs] = useState<string[]>([]);

  // --- フォームステート ---
  const [formData, setFormData] = useState<LeaseKunFormData>(INITIAL_FORM_DATA);

  useEffect(() => {
    const draft = readLeaseKunDraft();
    if (draft && hasMeaningfulDraft(draft.formData)) {
      setFormData(draft.formData);
      setStep(draft.step);
      setDraftRestored(true);
      setDraftSavedAt(draft.updatedAt);
      setHistory([
        { role: 'bot', text: `前回の下書きを復元しました。${STEPS[draft.step]} から続けられます。` }
      ]);
    }
    draftReadyRef.current = true;
  }, []);

  useEffect(() => {
    if (!draftReadyRef.current || phase !== 'wizard' || submitted) return;
    if (!hasMeaningfulDraft(formData)) {
      clearLeaseKunDraft();
      setDraftSavedAt('');
      return;
    }
    const updatedAt = new Date().toISOString();
    const draft: LeaseKunDraft = { version: 1, step, formData, updatedAt };
    window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(draft));
    setDraftSavedAt(updatedAt);
  }, [formData, step, phase, submitted]);

  // 業種マスター取得
  useEffect(() => {
    fetch('/api/master/industries')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data) return;
        setIndustryMaster(data);
        setMajors(Object.keys(data));
        if (INITIAL_FORM_DATA.industry_major && data[INITIAL_FORM_DATA.industry_major]) {
          setSubs(extractSubs(data[INITIAL_FORM_DATA.industry_major]));
        }
      })
      .catch(() => {});
  }, []);

  // 大分類変更時に中分類を連動
  useEffect(() => {
    if (!formData.industry_major || !industryMaster[formData.industry_major]) return;
    const newSubs = extractSubs(industryMaster[formData.industry_major]);
    setSubs(newSubs);
    if (newSubs.length > 0 && !newSubs.includes(formData.industry_sub)) {
      setFormData(prev => ({ ...prev, industry_sub: newSubs[0] }));
    }
  }, [formData.industry_major, formData.industry_sub, industryMaster]);

  // 自動スクロール
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, step]);

  // 分析結果フェーズに入ったら、screening画面と同じく紫苑レビューを自動生成する
  // （case_idごとに1回だけ。register⇄analysis往復での再生成は防ぐ）
  useEffect(() => {
    if (phase !== 'analysis' || !fullResult?.case_id) return;
    const caseId = String(fullResult.case_id);
    if (shionRequestedForCaseId.current === caseId) return;
    shionRequestedForCaseId.current = caseId;
    void shionReview.requestReview(fullResult, formData);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, fullResult]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    if (errors[name]) setErrors(prev => { const next = { ...prev }; delete next[name]; return next; });
  };

  const setQuickValue = (name: keyof LeaseKunFormData, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    if (errors[name]) setErrors(prev => { const next = { ...prev }; delete next[name]; return next; });
  };

  const handleNext = (e: React.FormEvent) => {
    e.preventDefault();
    if (step >= STEPS.length - 1) {
      submitScore();
      return;
    }

    const newErrors: Record<string, string> = {};
    let answerText = '';
    let nextBotText = '';

    switch (step) {
      case 0:
        if (!formData.sales_dept || formData.sales_dept === '未設定') {
          newErrors.sales_dept = '営業部を選択してください';
        }
        if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return; }
        answerText = `${formData.company_name || '（企業名未入力）'} / ${formData.industry_major} / ${formData.industry_sub} / ${formData.sales_dept}`;
        nextBotText = `次は取引状況について。当行メイン先？競合はいる？`;
        break;
      case 1:
        answerText = `${formData.main_bank} / ${formData.competitor} / 商談: ${formData.deal_source}`;
        nextBotText = `何をリースするのかな？（物件名）`;
        break;
      case 2:
        answerText = formData.asset_name || "その他";
        nextBotText = `損益計算書(P/L)の数値を入力してね！売上高は必須だよ。`;
        break;
      case 3:
        if (!isPositiveMillionInput(formData.nenshu)) {
          newErrors.nenshu = '売上高は必須です';
        }
        if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return; }
        answerText = `売上: ${formData.nenshu}百万円 / 営業利益: ${formData.op_profit || 0}百万円`;
        nextBotText = `貸借対照表(B/S)！総資産は必須。機械やその他の内訳もあれば。`;
        break;
      case 4:
        if (!isPositiveMillionInput(formData.total_assets)) {
          newErrors.total_assets = '総資産は必須です';
        }
        if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return; }
        answerText = `総資産: ${formData.total_assets}百万円 / 純資産: ${formData.net_assets || 0}百万円`;
        nextBotText = `減価償却や地代家賃などの経費項目はある？（なければ空欄かスキップでOK！）`;
        break;
      case 5:
        answerText = `償却: ${formData.depreciation || 0}百万円 / 家賃: ${formData.rent || 0}百万円`;
        nextBotText = `対象の格付や与信残高を教えてね。`;
        break;
      case 6:
        answerText = `格付: ${formData.grade} / 銀行与信: ${formData.bank_credit || 0}百万円`;
        nextBotText = `今回の契約期間や取得価格はどうなってる？`;
        break;
      case 7:
        if (!isPositiveMillionInput(formData.acquisition_cost)) {
          newErrors.acquisition_cost = '取得価格（百万円）は必須です';
        }
        if (Object.keys(newErrors).length > 0) { setErrors(newErrors); return; }
        answerText = `${formData.customer_type} / ${formData.lease_term}ヶ月 / ${formData.acquisition_cost}百万円`;
        nextBotText = `定性的な評価項目（6点）を教えて。難しければ「未選択」でも審査はできるよ。`;
        break;
      case 8:
        answerText = `定性評価 入力完了`;
        nextBotText = `最後！！直感スコア（1〜5）を教えて。これで審査を実行するよ。`;
        break;
    }

    setErrors({});

    const addedMessages: Message[] = [{ role: 'user', text: answerText }];

    if (Math.random() < 0.3) {
      const humors = [
        "業種によって「良い数字」の基準は変わります。比較する相手を間違えないように。",
        "数字の向こう側にある現場を、想像しながら読んでいます。",
        "審査は減点ゲームではなく、可能性を見つける作業です。"
      ];
      addedMessages.push({ role: 'humor', text: humors[Math.floor(Math.random() * humors.length)] });
    }
    addedMessages.push({ role: 'bot', text: nextBotText });

    setHistory(prev => [...prev, ...addedMessages]);
    setStep(s => s + 1);
  };

  const submitScore = async () => {
    setLoading(true);
    setHistory(prev => [...prev,
      { role: 'user', text: `直感: ${formData.intuition}点。これで審査よろしく！` },
      { role: 'bot', text: '了解！FastAPIのフル審査エンジンにデータを送っています... 🚀' }
    ]);

    try {
      const payload = toThousandYenPayload({
        company_no:                   formData.company_no,
        company_name:                 formData.company_name,
        asset_name:                   formData.asset_name,
        asset_location:               formData.asset_location,
        industry_major:               formData.industry_major,
        industry_sub:                 formData.industry_sub,
        sales_dept:                   formData.sales_dept,
        main_bank:                    formData.main_bank,
        competitor:                   formData.competitor,
        num_competitors:              formData.num_competitors,
        deal_source:                  formData.deal_source,
        deal_occurrence:              formData.deal_occurrence,
        customer_type:                formData.customer_type,
        contract_type:                formData.contract_type,
        grade:                        formData.grade,
        nenshu:                       parseMillionInput(formData.nenshu),
        gross_profit:                 parseMillionInput(formData.gross_profit),
        op_profit:                    parseMillionInput(formData.op_profit),
        ord_profit:                   parseMillionInput(formData.ord_profit),
        net_income:                   parseMillionInput(formData.net_income),
        total_assets:                 parseMillionInput(formData.total_assets),
        net_assets:                   parseMillionInput(formData.net_assets),
        machines:                     parseMillionInput(formData.machines),
        other_assets:                 parseMillionInput(formData.other_assets),
        depreciation:                 parseMillionInput(formData.depreciation),
        dep_expense:                  parseMillionInput(formData.dep_expense),
        rent:                         parseMillionInput(formData.rent),
        rent_expense:                 parseMillionInput(formData.rent_expense),
        bank_credit:                  parseMillionInput(formData.bank_credit),
        lease_credit:                 parseMillionInput(formData.lease_credit),
        contracts:                    parseMillionInput(formData.contracts),
        acquisition_cost:             parseMillionInput(formData.acquisition_cost),
        lease_term:                   parseMillionInput(formData.lease_term, 60),
        acceptance_year:              parseMillionInput(formData.acceptance_year, new Date().getFullYear()),
        qual_corr_company_history:    formData.qual_corr_company_history,
        qual_corr_customer_stability: formData.qual_corr_customer_stability,
        qual_corr_repayment_history:  formData.qual_corr_repayment_history,
        qual_corr_business_future:    formData.qual_corr_business_future,
        qual_corr_equipment_purpose:  formData.qual_corr_equipment_purpose,
        qual_corr_main_bank:          formData.qual_corr_main_bank,
        passion_text:                 formData.passion_text,
        intuition:                    Number(formData.intuition),
      });

      const res = await apiClient.post(`/api/score/full`, payload);
      setSubmitted(true);
      clearLeaseKunDraft();
      setDraftSavedAt('');
      const resultData = res.data as LeaseKunFullResult;
      setFullResult(resultData);

      const caseId = resultData.case_id;

      setHistory(prev => [...prev, {
        role: 'humor',
        text: (
          <span>
            <b>🎉 審査完了！</b><br/>
            総合スコア: {(resultData.score_base ?? resultData.score)?.toFixed(1)}点<br/>
            判定: {resultData.hantei}<br/>
            借手スコア: {resultData.score_borrower?.toFixed(1)}点<br/><br/>
            {caseId ? (
              <button
                onClick={() => setPhase('analysis')}
                className="flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-3 py-2 rounded-lg shadow transition-colors w-full justify-center"
              >
                📋 このまま分析結果を見る
              </button>
            ) : (
              <span className="text-rose-600 font-bold">
                ⚠️ 結果登録への保存に失敗した可能性があります。お手数ですが「📋 審査・分析」タブから内容を入力し直してください。
              </span>
            )}
            <br/><br/>
            <button
              onClick={() => handleGunshiConsult(resultData)}
              className="mt-2 flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold px-3 py-2 rounded-lg shadow transition-colors w-full justify-center"
            >
              <MessageSquare className="w-3.5 h-3.5" />
              軍師AIに相談する
            </button>
          </span>
        )
      }]);

      if (caseId) {
        setPhase('analysis');
      }
    } catch (e) {
      const err = e as { response?: { status?: number; data?: { detail?: string } }; message?: string };
      const status = err.response?.status;
      const detail = err.response?.data?.detail || err.message || '不明なエラー';
      const errorMsg = status ? `エラー ${status}: ${detail}` : `エラー: ${detail}`;
      setHistory(prev => [...prev, { role: 'humor', text: `送信失敗！${errorMsg}` }]);
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    if (step === 0) return;
    setStep(s => s - 1);
    setErrors({});
    setHistory(prev => {
      const nw = [...prev];
      nw.pop(); // bot message
      if (nw.length > 0 && nw[nw.length - 1].role === 'humor') nw.pop(); // humor message (if present)
      nw.pop(); // user message
      return nw;
    });
  };

  const resetWizard = () => {
    clearLeaseKunDraft();
    setDraftRestored(false);
    setDraftSavedAt('');
    setDoneMessage('結果登録まで完了しました！');
    setStep(0);
    setSubmitted(false);
    setErrors({});
    setPhase('wizard');
    setFullResult(null);
    setHistory([
      { role: 'bot', text: 'はじめまして！リースくんです 🎩 まず企業名と業種から教えてね！' }
    ]);
    setFormData(prev => ({
      ...prev,
      company_no: '', company_name: '',
      nenshu: '', gross_profit: '', op_profit: '', ord_profit: '', net_income: '',
      total_assets: '', net_assets: '', machines: '', other_assets: '',
      depreciation: '', dep_expense: '', rent: '', rent_expense: '',
      contracts: '', bank_credit: '', lease_credit: '',
      acquisition_cost: '',
      passion_text: '',
      asset_location: '',
      intuition: 3,
    }));
  };

  const discardDraft = () => {
    clearLeaseKunDraft();
    setDraftRestored(false);
    setDraftSavedAt('');
    setDoneMessage('結果登録まで完了しました！');
    setStep(0);
    setSubmitted(false);
    setErrors({});
    setPhase('wizard');
    setFullResult(null);
    setFormData(INITIAL_FORM_DATA);
    setHistory([
      { role: 'bot', text: '下書きを破棄しました。新しい審査を始めます。' }
    ]);
  };

  const deferResultRegistration = () => {
    setDoneMessage('審査結果を保存しました。成約/失注が分かったら、結果登録へ進めます。');
    setPhase('done');
  };

  const submitReviewFeedbackAndOpenRegistration = async (feedback: ShionReviewFeedback) => {
    const saved = await shionReview.submitFeedback(feedback);
    if (saved) {
      setPhase('register');
    }
  };

  const handleGunshiConsult = (result: ScoreResult) => {
    const assetLocation = formData.asset_location || result.asset_location || '';
    const context = {
      score: result.score_base ?? result.score,
      hantei: result.hantei,
      score_borrower: result.score_borrower,
      company_name: result.company_name || formData.company_name || '（未入力）',
      asset_name: result.asset_name || formData.asset_name,
      asset_location: assetLocation,
      prefecture: extractPrefectureFromText(assetLocation),
      industry_sub: result.industry_sub || formData.industry_sub,
      sales_dept: result.sales_dept || formData.sales_dept,
      quantum_risk: result.quantum_risk,
      case_id: result.case_id,
    };
    window.localStorage.setItem('lease-gunshi-context', JSON.stringify(context));
    router.push('/chat');
  };

  // 定性評価のオプション群
  const qualOpts = {
    qual_corr_company_history:    { label: "設立・経営年数",     opts: ["未選択","20年以上","10年〜20年","5年〜10年","3年〜5年","3年未満"] },
    qual_corr_customer_stability: { label: "顧客安定性",         opts: ["未選択","非常に安定（大口・長期）","安定（分散良好）","普通","やや不安定（集中あり）","不安定・依存大"] },
    qual_corr_repayment_history:  { label: "返済履歴",           opts: ["未選択","5年以上問題なし","3年以上問題なし","遅延少ない","遅延・リスケあり","問題あり・要確認"] },
    qual_corr_business_future:    { label: "事業将来性",         opts: ["未選択","有望（成長・ニーズ確実）","やや有望","普通","やや懸念","懸念（縮小・競争激化）"] },
    qual_corr_equipment_purpose:  { label: "設備目的",           opts: ["未選択","収益直結・受注必須","生産性向上・省力化","更新・維持・法定対応","やや不明確","不明確・要説明"] },
    qual_corr_main_bank:          { label: "メイン銀行関係",     opts: ["未選択","メイン先で取引良好・支援表明","メイン先","サブ扱い・取引あり","取引浅い・他社メイン","取引なし・不安"] },
  };

  const sel = "w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-sm font-bold text-[#1A1A2E] appearance-none outline-none focus:border-[#E8A838]";
  const inp = "w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-sm outline-none focus:border-[#E8A838]";
  const inpReq = "w-full bg-amber-50 border-2 border-[#E8A838] rounded-xl p-3 text-sm outline-none font-bold";
  const inpErr = "w-full bg-red-50 border-2 border-red-400 rounded-xl p-3 text-sm outline-none font-bold";
  const lbl = "text-[11px] font-black text-slate-500 mb-1 block";
  const errMsg = "text-red-500 text-xs mt-1";

  return (
    <div className="md:min-h-[calc(100vh-2rem)] flex items-center justify-center bg-slate-900 md:py-8 w-full">
      <div className="w-full h-[100dvh] md:max-w-[400px] md:h-[800px] bg-[#f4f1ec] md:rounded-[3rem] md:shadow-2xl overflow-hidden md:border-[12px] border-slate-800 relative flex flex-col">
        {/* スマホのノッチ（PC視聴時のみ） */}
        <div className="hidden md:flex absolute top-0 inset-x-0 mx-auto w-32 h-6 bg-slate-800 rounded-b-2xl z-20 justify-center items-center">
          <div className="w-12 h-1 bg-slate-900 rounded-full mt-1"></div>
        </div>

        {/* ヘッダー */}
        <div className="bg-gradient-to-r from-[#1A1A2E] to-[#2d2d4e] w-full pt-12 md:pt-10 pb-4 px-4 shadow flex justify-between items-center shrink-0 z-10 text-[#E8A838]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-[#E8A838] rounded-full flex justify-center items-center shadow-inner border-2 border-white/70 text-slate-950">
              <MessageSquare className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h3 className="font-black text-sm tracking-widest uppercase">Lease-Wizard</h3>
              <p className="text-[10px] opacity-80 mt-0.5">
                {phase === 'wizard' ? `Step ${step + 1} / ${STEPS.length}` : PHASE_LABELS[phase]}
              </p>
            </div>
          </div>
          <div className="text-[10px] font-bold bg-[#E8A838] text-slate-900 px-2 py-1 rounded-sm">
            {phase === 'wizard' ? STEPS[step] : PHASE_LABELS[phase]}
          </div>
        </div>

        {/* 進行バー */}
        <div className="h-1 bg-slate-200 shrink-0">
          <div
            className="h-full bg-gradient-to-r from-orange-400 to-[#E8A838] transition-all duration-300"
            style={{ width: phase === 'wizard' ? `${((step + 1) / STEPS.length) * 100}%` : '100%' }}
          />
        </div>

        {/* チャット履歴エリア */}
        <div ref={scrollRef} className="flex-1 w-full p-4 overflow-y-auto space-y-4 scrollbar-hide">
          {history.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'bot' && (
                <div className="bg-white border-2 border-[#1A1A2E] text-[#1A1A2E] rounded-2xl rounded-tl-none p-3 shadow-sm text-sm font-medium max-w-[85%] leading-relaxed">
                  {msg.text}
                </div>
              )}
              {msg.role === 'user' && (
                <div className="bg-[#1A1A2E] text-white rounded-2xl rounded-tr-none py-2 px-4 shadow-sm text-sm max-w-[85%] text-right leading-relaxed">
                  {msg.text}
                </div>
              )}
              {msg.role === 'humor' && (
                <div className="bg-[#FFF8E8] border border-[#E8A838] text-amber-900 rounded-xl py-3 px-4 shadow-sm text-sm w-full mx-2 my-2">
                  <div className="text-[10px] font-bold text-[#E8A838] mb-1">💬 リースくんのつぶやき</div>
                  <div>{msg.text}</div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border-2 border-[#1A1A2E] text-[#1A1A2E] rounded-2xl p-3 flex items-center gap-2">
                <Activity className="animate-spin w-4 h-4 text-orange-500" />
                <span className="text-sm">エンジン実行中...</span>
              </div>
            </div>
          )}
          <div className="h-2"></div>
        </div>

        {/* 下部フォームエリア */}
        {!loading && phase === 'wizard' && (
        <form onSubmit={handleNext} className="w-full bg-white border-t-2 border-[#1A1A2E] p-4 shrink-0 shadow-[0_-4px_15px_rgba(0,0,0,0.05)] rounded-t-2xl z-20">
          {(draftRestored || draftSavedAt) && (
            <div className="mb-3 flex items-center justify-between gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] font-bold text-amber-800">
              <span>{draftRestored ? '下書きを復元済み' : '下書き保存済み'}</span>
              <button
                type="button"
                onClick={discardDraft}
                className="flex h-7 w-7 items-center justify-center rounded-lg bg-white text-amber-700 shadow-sm"
                aria-label="下書きを破棄"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
          <div className="mb-4 space-y-3 max-h-[40vh] overflow-y-auto scrollbar-hide pb-2 px-1">

            {/* Step 0: 企業・業種 */}
            {step === 0 && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>企業番号</label>
                    <input type="text" name="company_no" value={formData.company_no} onChange={handleChange} placeholder="例: 123456" className={inp} />
                  </div>
                  <div>
                    <label className={lbl}>企業名</label>
                    <input type="text" name="company_name" value={formData.company_name} onChange={handleChange} placeholder="例: 株式会社○○" className={inp} />
                  </div>
                </div>
                <div className="relative">
                  <label className={lbl}>業種（大分類）</label>
                  <select name="industry_major" value={formData.industry_major} onChange={handleChange} className={sel}>
                    {majors.length > 0
                      ? majors.map(m => <option key={m} value={m}>{m}</option>)
                      : (
                        <>
                          <option>A 農業，林業</option>
                          <option>B 漁業</option>
                          <option>C 鉱業，採石業，砂利採取業</option>
                          <option>D 建設業</option>
                          <option>E 製造業</option>
                          <option>F 電気・ガス・熱供給・水道業</option>
                          <option>G 情報通信業</option>
                          <option>H 運輸業，郵便業</option>
                          <option>I 卸売業，小売業</option>
                          <option>J 金融業，保険業</option>
                          <option>K 不動産業，物品賃貸業</option>
                          <option>L 学術研究，専門・技術サービス業</option>
                          <option>M 宿泊業，飲食サービス業</option>
                          <option>N 生活関連サービス業，娯楽業</option>
                          <option>O 教育，学習支援業</option>
                          <option>P 医療，福祉</option>
                          <option>Q 複合サービス事業</option>
                          <option>R サービス業（他に分類されないもの）</option>
                        </>
                      )
                    }
                  </select>
                  <ChevronDown className="absolute right-3 top-8 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
                <div className="relative">
                  <label className={lbl}>業種（中分類）</label>
                  <select name="industry_sub" value={formData.industry_sub} onChange={handleChange} className={sel}>
                    {subs.length > 0
                      ? subs.map(s => <option key={s} value={s}>{s}</option>)
                      : <option value={formData.industry_sub}>{formData.industry_sub}</option>
                    }
                  </select>
                  <ChevronDown className="absolute right-3 top-8 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
                <div>
                  <label className={lbl}>営業部 <span className="text-red-500">※必須</span></label>
                  <select
                    name="sales_dept"
                    value={formData.sales_dept}
                    onChange={handleChange}
                    className={errors.sales_dept ? inpErr : sel}
                  >
                    <option>未設定</option>
                    <option>宇都宮営業部</option>
                    <option>小山営業部</option>
                    <option>足利営業部</option>
                    <option>埼玉営業部</option>
                  </select>
                  {errors.sales_dept && <p className={errMsg}>{errors.sales_dept}</p>}
                </div>
              </>
            )}

            {/* Step 1: 取引・競合 */}
            {step === 1 && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>取引区分</label>
                    <select name="main_bank" value={formData.main_bank} onChange={handleChange} className={sel}>
                      <option>メイン先</option><option>非メイン先</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>顧客区分</label>
                    <select name="customer_type" value={formData.customer_type} onChange={handleChange} className={sel}>
                      <option>既存先</option><option>新規先</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>競合状況</label>
                    <select name="competitor" value={formData.competitor} onChange={handleChange} className={sel}>
                      <option>競合なし</option><option>競合あり</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>競合社数</label>
                    <select name="num_competitors" value={formData.num_competitors} onChange={handleChange} className={sel}>
                      <option>未入力</option><option>0社</option><option>1社</option><option>2社</option><option>3社以上</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>商談ソース</label>
                    <select name="deal_source" value={formData.deal_source} onChange={handleChange} className={sel}>
                      <option>銀行紹介</option><option>その他</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>発生経緯</label>
                    <select name="deal_occurrence" value={formData.deal_occurrence} onChange={handleChange} className={sel}>
                      <option>不明</option><option>指名</option><option>相見積もり</option>
                    </select>
                  </div>
                </div>
              </>
            )}

            {/* Step 2: 物件 */}
            {step === 2 && (
              <div className="space-y-2">
                <label className={lbl}>物件選択</label>
                <select name="asset_name" value={formData.asset_name} onChange={handleChange} className={sel}>
                  <option>建設機械</option>
                  <option>IT・OA機器</option>
                  <option>医療機器</option>
                  <option>車両・運搬車</option>
                  <option>製造設備・工作機械</option>
                  <option>オフィス家具・内装</option>
                  <option>飲食店設備</option>
                  <option>太陽光・省エネ設備</option>
                  <option>その他・未選択</option>
                </select>
                <div>
                  <label className={lbl}>設置場所（任意）</label>
                  <input
                    type="text"
                    name="asset_location"
                    value={formData.asset_location}
                    onChange={handleChange}
                    placeholder="例: 大阪府大阪市"
                    className={inp}
                  />
                </div>
              </div>
            )}

            {/* Step 3: P/L */}
            {step === 3 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="text" inputMode="decimal" name="nenshu" value={formData.nenshu} step="0.1" onChange={handleChange} placeholder="売上高 (百万円) ※必須" className={errors.nenshu ? inpErr : inpReq} />
                  <AmountChips amounts={SALES_ASSET_AMOUNTS} activeValue={formData.nenshu} onSelect={(value) => setQuickValue('nenshu', value)} />
                  {errors.nenshu && <p className={errMsg}>{errors.nenshu}</p>}
                </div>
                <div>
                  <input type="text" inputMode="text" name="gross_profit" value={formData.gross_profit} step="0.1" onChange={handleChange} placeholder="売上総利益 (百万円) ※赤字は例: -5" className={inp} />
                  <AmountChips amounts={PROFIT_AMOUNTS} activeValue={formData.gross_profit} onSelect={(value) => setQuickValue('gross_profit', value)} />
                </div>
                <div>
                  <input type="text" inputMode="text" name="op_profit" value={formData.op_profit} step="0.1" onChange={handleChange} placeholder="営業利益 (百万円) ※赤字は例: -5" className={inp} />
                  <AmountChips amounts={PROFIT_AMOUNTS} activeValue={formData.op_profit} onSelect={(value) => setQuickValue('op_profit', value)} />
                </div>
                <div>
                  <input type="text" inputMode="text" name="ord_profit" value={formData.ord_profit} step="0.1" onChange={handleChange} placeholder="経常利益 (百万円) ※赤字は例: -5" className={inp} />
                  <AmountChips amounts={PROFIT_AMOUNTS} activeValue={formData.ord_profit} onSelect={(value) => setQuickValue('ord_profit', value)} />
                </div>
                <div>
                  <input type="text" inputMode="text" name="net_income" value={formData.net_income} step="0.1" onChange={handleChange} placeholder="当期純利益 (百万円) ※赤字は例: -5" className={inp} />
                  <AmountChips amounts={PROFIT_AMOUNTS} activeValue={formData.net_income} onSelect={(value) => setQuickValue('net_income', value)} />
                </div>
              </div>
            )}

            {/* Step 4: B/S */}
            {step === 4 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="text" inputMode="decimal" name="total_assets" value={formData.total_assets} step="0.1" onChange={handleChange} placeholder="総資産 (百万円) ※必須" className={errors.total_assets ? inpErr : inpReq} />
                  <AmountChips amounts={SALES_ASSET_AMOUNTS} activeValue={formData.total_assets} onSelect={(value) => setQuickValue('total_assets', value)} />
                  {errors.total_assets && <p className={errMsg}>{errors.total_assets}</p>}
                </div>
                <div className="col-span-2">
                  <input type="text" inputMode="text" name="net_assets" value={formData.net_assets} step="0.1" onChange={handleChange} placeholder="純資産/自己資本 (百万円) ※債務超過は例: -5" className={inp} />
                  <AmountChips amounts={PROFIT_AMOUNTS} activeValue={formData.net_assets} onSelect={(value) => setQuickValue('net_assets', value)} />
                </div>
                <div>
                  <input type="text" inputMode="decimal" name="machines" value={formData.machines} step="0.1" onChange={handleChange} placeholder="機械装置 (百万円)" className={inp} />
                  <AmountChips amounts={CREDIT_AMOUNTS} activeValue={formData.machines} onSelect={(value) => setQuickValue('machines', value)} />
                </div>
                <div>
                  <input type="text" inputMode="decimal" name="other_assets" value={formData.other_assets} step="0.1" onChange={handleChange} placeholder="その他資産 (百万円)" className={inp} />
                  <AmountChips amounts={CREDIT_AMOUNTS} activeValue={formData.other_assets} onSelect={(value) => setQuickValue('other_assets', value)} />
                </div>
              </div>
            )}

            {/* Step 5: 経費 */}
            {step === 5 && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <input type="text" inputMode="decimal" name="depreciation" value={formData.depreciation} step="0.1" onChange={handleChange} placeholder="減価償却(資産・百万円)" className={inp} />
                  <AmountChips amounts={SMALL_COST_AMOUNTS} activeValue={formData.depreciation} onSelect={(value) => setQuickValue('depreciation', value)} />
                </div>
                <div>
                  <input type="text" inputMode="decimal" name="dep_expense" value={formData.dep_expense} step="0.1" onChange={handleChange} placeholder="減価償却(経費・百万円)" className={inp} />
                  <AmountChips amounts={SMALL_COST_AMOUNTS} activeValue={formData.dep_expense} onSelect={(value) => setQuickValue('dep_expense', value)} />
                </div>
                <div>
                  <input type="text" inputMode="decimal" name="rent" value={formData.rent} step="0.1" onChange={handleChange} placeholder="賃借料(資産・百万円)" className={inp} />
                  <AmountChips amounts={SMALL_COST_AMOUNTS} activeValue={formData.rent} onSelect={(value) => setQuickValue('rent', value)} />
                </div>
                <div>
                  <input type="text" inputMode="decimal" name="rent_expense" value={formData.rent_expense} step="0.1" onChange={handleChange} placeholder="賃借料(経費・百万円)" className={inp} />
                  <AmountChips amounts={SMALL_COST_AMOUNTS} activeValue={formData.rent_expense} onSelect={(value) => setQuickValue('rent_expense', value)} />
                </div>
              </div>
            )}

            {/* Step 6: 信用 */}
            {step === 6 && (
              <div className="space-y-2">
                <select name="grade" value={formData.grade} onChange={handleChange} className={sel}>
                  <option>① 1-3先</option>
                  <option>② 4-6先</option>
                  <option>③ 要注意先</option>
                  <option>④ 無格付先</option>
                </select>
                <div className="grid grid-cols-1 gap-2">
                  <input type="text" inputMode="decimal" name="contracts" value={formData.contracts} onChange={handleChange} placeholder="契約件数" className={inp} />
                  <div>
                    <input type="text" inputMode="decimal" name="bank_credit" value={formData.bank_credit} step="0.1" onChange={handleChange} placeholder="銀行与信残(百万円)" className={inp} />
                    <AmountChips amounts={CREDIT_AMOUNTS} activeValue={formData.bank_credit} onSelect={(value) => setQuickValue('bank_credit', value)} />
                  </div>
                  <div>
                    <input type="text" inputMode="decimal" name="lease_credit" value={formData.lease_credit} step="0.1" onChange={handleChange} placeholder="リース与信残(百万円)" className={inp} />
                    <AmountChips amounts={CREDIT_AMOUNTS} activeValue={formData.lease_credit} onSelect={(value) => setQuickValue('lease_credit', value)} />
                  </div>
                </div>
              </div>
            )}

            {/* Step 7: 契約 */}
            {step === 7 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="text" inputMode="decimal" name="acquisition_cost" value={formData.acquisition_cost} step="0.1" onChange={handleChange} placeholder="取得価格 (百万円) ※必須" className={errors.acquisition_cost ? inpErr : inpReq} />
                  <AmountChips amounts={ASSET_PRICE_AMOUNTS} activeValue={formData.acquisition_cost} onSelect={(value) => setQuickValue('acquisition_cost', value)} />
                  {errors.acquisition_cost && <p className={errMsg}>{errors.acquisition_cost}</p>}
                </div>
                <div>
                  <label className={lbl}>契約種類</label>
                  <select name="contract_type" value={formData.contract_type} onChange={handleChange} className={sel}>
                    <option>一般</option><option>自動車</option>
                  </select>
                </div>
                <div>
                  <label className={lbl}>期間(月)</label>
                  <input type="text" inputMode="decimal" name="lease_term" value={formData.lease_term} onChange={handleChange} className={inp} />
                  <AmountChips amounts={TERM_AMOUNTS} activeValue={formData.lease_term} onSelect={(value) => setQuickValue('lease_term', value)} />
                </div>
                <div className="col-span-2">
                  <label className={lbl}>検収年(西暦)</label>
                  <input type="text" inputMode="decimal" name="acceptance_year" value={formData.acceptance_year} onChange={handleChange} className={inp} />
                </div>
              </div>
            )}

            {/* Step 8: 定性(6項目) + パッション */}
            {step === 8 && (
              <div className="space-y-2">
                {(Object.keys(qualOpts) as Array<keyof typeof qualOpts>).map(k => (
                  <div key={k} className="flex flex-col">
                    <label className="text-[10px] font-bold text-slate-400">{qualOpts[k].label}</label>
                    <select name={k} value={formData[k as keyof typeof formData] as string} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-md p-1.5 text-xs outline-none">
                      {qualOpts[k].opts.map(o => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
                <div>
                  <label className="text-[10px] font-bold text-slate-400">特記事項・アピールポイント（任意）</label>
                  <textarea name="passion_text" value={formData.passion_text} onChange={handleChange} rows={2} placeholder="担当者コメントがあれば..." className="w-full bg-slate-50 border border-slate-200 rounded-md p-1.5 text-xs outline-none resize-none" />
                </div>
              </div>
            )}

            {/* Step 9: 直感スコア */}
            {step === 9 && (
              <div className="flex flex-col items-center">
                <p className="text-xs text-slate-500 font-bold mb-3">担当者の直感スコア（1:懸念〜5:確信）</p>
                <div className="flex justify-center gap-2">
                  {[1,2,3,4,5].map(v => (
                    <button type="button" key={v} onClick={() => setFormData({...formData, intuition: v})}
                      className={`w-12 h-12 flex items-center justify-center font-black rounded-full border-2 transition-all ${formData.intuition === v ? 'bg-[#E8A838] border-[#E8A838] text-white scale-110' : 'border-slate-200 text-slate-400 bg-white hover:border-[#E8A838]'}`}>
                      {v}
                    </button>
                  ))}
                </div>
              </div>
            )}

          </div>

          <div className="flex gap-2 mt-2">
            <button
              type="button" onClick={goBack} disabled={step === 0}
              className="w-12 h-12 flex items-center justify-center rounded-xl bg-slate-100 text-slate-600 disabled:opacity-30">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <button
              type="submit"
              disabled={submitted && step >= STEPS.length - 1}
              className={`flex-1 h-12 flex items-center justify-center rounded-xl font-bold tracking-wide shadow-[0_4px_0_#0f0f1c] active:shadow-none active:translate-y-1 transition-all ${submitted && step >= STEPS.length - 1 ? 'bg-slate-400 text-white cursor-not-allowed shadow-none' : 'bg-[#1A1A2E] text-white'}`}>
              {submitted && step >= STEPS.length - 1 ? '送信済み ✓' : step >= STEPS.length - 1 ? '審査実行 🚀' : '次へ進む'}
            </button>
          </div>
        </form>
        )}

        {/* 分析結果フェーズ: screening 画面に離脱せず、その場で争点・稟議方針を確認 */}
        {!loading && phase === 'analysis' && fullResult && (
          <div className="w-full bg-white border-t-2 border-[#1A1A2E] p-4 shrink-0 shadow-[0_-4px_15px_rgba(0,0,0,0.05)] rounded-t-2xl z-20 max-h-[70vh] overflow-y-auto scrollbar-hide">
            <div className="space-y-3">
              <FocusCheckCard checks={buildFocusChecks(fullResult, formData)} />
              <CurrentIssueCard result={fullResult} data={formData} />
              <RingiPolicyCard result={fullResult} data={formData} />
              {Array.isArray(fullResult.conditional_approval_actions) && fullResult.conditional_approval_actions.length > 0 && (
                <section className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3">
                  <div className="text-[11px] font-black uppercase tracking-wider text-amber-600 mb-2">条件付き承認に向けた確認事項</div>
                  <ul className="space-y-1.5">
                    {fullResult.conditional_approval_actions.slice(0, 4).map((a, i) => (
                      <li key={i} className="text-xs font-bold text-amber-900 leading-relaxed">
                        ・{typeof a === 'string' ? a : a.action || a.reason || String(a)}
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {typeof fullResult.quantum_risk === 'number' && fullResult.quantum_risk >= 35 && (
                <section className={`rounded-2xl border px-4 py-3 ${fullResult.quantum_risk >= 60 ? 'border-rose-300 bg-rose-50' : 'border-amber-200 bg-amber-50'}`}>
                  <div className={`text-[11px] font-black uppercase tracking-wider ${fullResult.quantum_risk >= 60 ? 'text-rose-600' : 'text-amber-600'}`}>
                    Q_risk {fullResult.quantum_risk.toFixed(1)}
                  </div>
                  <div className="mt-1 text-xs font-bold text-slate-700">
                    {fullResult.quantum_risk >= 60 ? '強警戒：入力値の整合性を要確認' : '要注意：数値の矛盾がないか確認してください'}
                  </div>
                </section>
              )}
              <ShionScreeningReviewCard
                result={fullResult}
                review={shionReview.review}
                loading={shionReview.loading}
                error={shionReview.error}
                onReview={() => shionReview.requestReview(fullResult, formData)}
                onFeedback={submitReviewFeedbackAndOpenRegistration}
                feedbackSaving={shionReview.feedbackSaving}
                pastCompanies={shionReview.pastCompanies}
                judgmentAssetCandidates={shionReview.judgmentAssetCandidates}
              />
            </div>
            <div className="mt-4 space-y-2">
              <button
                type="button"
                onClick={deferResultRegistration}
                className="w-full h-12 flex items-center justify-center rounded-xl font-bold tracking-wide shadow-[0_4px_0_#0f0f1c] active:shadow-none active:translate-y-1 transition-all bg-[#1A1A2E] text-white"
              >
                後で結果登録する
              </button>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => router.push(`/screening?case_id=${encodeURIComponent(String(fullResult.case_id))}`)}
                  className="h-11 flex items-center justify-center rounded-xl font-bold text-[11px] bg-slate-100 text-slate-600"
                >
                  PCで詳細分析
                </button>
                <button
                  type="button"
                  onClick={() => setPhase('register')}
                  className="h-11 flex items-center justify-center rounded-xl font-bold text-[11px] bg-amber-100 text-amber-900"
                >
                  今わかるなら登録
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 結果登録フェーズ: 別画面（/register）へ移らず、そのまま成約/失注を確定 */}
        {!loading && phase === 'register' && fullResult?.case_id && (
          <div className="w-full bg-white border-t-2 border-[#1A1A2E] p-4 shrink-0 shadow-[0_-4px_15px_rgba(0,0,0,0.05)] rounded-t-2xl z-20 max-h-[70vh] overflow-y-auto scrollbar-hide">
            <button
              type="button"
              onClick={() => setPhase('analysis')}
              className="mb-3 flex items-center gap-1 text-xs font-bold text-slate-400"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> 分析結果に戻る
            </button>
            <CaseRegistrationForm
              caseId={String(fullResult.case_id)}
              compact
              onRegistered={() => {
                setDoneMessage('結果登録まで完了しました！');
                setPhase('done');
              }}
            />
          </div>
        )}

        {/* 完了フェーズ */}
        {!loading && phase === 'done' && (
          <div className="w-full bg-white border-t-2 border-[#1A1A2E] p-6 shrink-0 shadow-[0_-4px_15px_rgba(0,0,0,0.05)] rounded-t-2xl z-20 flex flex-col items-center text-center gap-3">
            <CheckCircle2 className="w-10 h-10 text-emerald-500" />
            <p className="text-sm font-black text-[#1A1A2E]">{doneMessage}</p>
            {fullResult?.case_id && (
              <button
                type="button"
                onClick={() => router.push(`/register?case_id=${encodeURIComponent(String(fullResult.case_id))}`)}
                className="w-full h-11 flex items-center justify-center rounded-xl font-bold bg-amber-100 text-amber-900"
              >
                結果登録画面でこの案件を開く
              </button>
            )}
            <button
              type="button"
              onClick={resetWizard}
              className="w-full h-12 flex items-center justify-center rounded-xl font-bold tracking-wide shadow-[0_4px_0_#0f0f1c] active:shadow-none active:translate-y-1 transition-all bg-[#1A1A2E] text-white"
            >
              もう一件、審査する
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
