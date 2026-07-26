"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Check,
  ClipboardList,
  Loader2,
  Save,
  Search,
  SkipForward,
} from "lucide-react";

type DrillItem = {
  case_id: string;
  status: string;
  case_theme: string;
  industry: string;
  company_stage: string;
  bank_is_main: string;
  bank_credit_balance_million_yen: string;
  lease_customer_status: string;
  existing_lease_credit_balance_million_yen: string;
  equity_ratio_percent: string;
  asset_type: string;
  amount_million_yen: string;
  lease_term_years: string;
  purpose: string;
  financial_snapshot: string;
  case_context: string;
  positive_points: string;
  risk_points: string;
  credit_score_20: string;
  repayment_source_score_20: string;
  asset_exit_score_20: string;
  plan_specificity_score_20: string;
  uncertainty_control_score_20: string;
  total_score_100: string;
  user_decision: string;
  heaviest_issue: string;
  additional_checks: string;
  ringi_sentence: string;
  score_decision_gap_note: string;
  ai_feedback_outcome: string;
  ai_feedback_note: string;
  okf_candidate_tags: string;
  presentation_use_ok: string;
  reviewer: string;
  judged_at: string;
};

type DrillResponse = {
  item: DrillItem | null;
  index: number;
  total: number;
  completed: number;
  pending: number;
  nextPendingIndex: number;
};

const SCORE_FIELDS = [
  ["credit_score_20", "信用力"],
  ["repayment_source_score_20", "返済原資"],
  ["asset_exit_score_20", "物件性・出口"],
  ["plan_specificity_score_20", "計画具体性"],
  ["uncertainty_control_score_20", "未確定リスク管理"],
] as const;

const DECISIONS = ["承認", "条件付き承認", "追加確認", "否認"];
const AI_OUTCOMES = [
  ["", "未評価"],
  ["helped", "使えた"],
  ["neutral", "観点のみ"],
  ["challenged", "要修正"],
  ["rejected", "外した"],
] as const;

const CONDITION_OPTIONS = [
  "本件限度",
  "次回決算まで本件限度",
  "金融機関と協調",
  "独立・新設向け条件",
  "親会社等保証",
  "担保・保全あり",
  "その他",
  "事業計画確認",
  "銀行との与信バランスを取る",
  "ピーク内与信",
  "利回り確保",
  "再リース確認",
] as const;

type ConditionOption = (typeof CONDITION_OPTIONS)[number];

const PRESETS = [
  {
    label: "前向き",
    decision: "承認",
    scores: [18, 18, 16, 16, 16],
    note: "大きな懸念なし",
  },
  {
    label: "条件付き",
    decision: "条件付き承認",
    scores: [15, 14, 13, 12, 10],
    note: "条件を置いて進める",
  },
  {
    label: "追加確認",
    decision: "追加確認",
    scores: [12, 10, 10, 8, 8],
    note: "不足情報を先に回収",
  },
  {
    label: "否認寄り",
    decision: "否認",
    scores: [6, 6, 8, 5, 4],
    note: "無理に通さない",
  },
] as const;

const emptyItem: DrillItem = {
  case_id: "",
  status: "",
  case_theme: "",
  industry: "",
  company_stage: "",
  bank_is_main: "",
  bank_credit_balance_million_yen: "",
  lease_customer_status: "",
  existing_lease_credit_balance_million_yen: "",
  equity_ratio_percent: "",
  asset_type: "",
  amount_million_yen: "",
  lease_term_years: "",
  purpose: "",
  financial_snapshot: "",
  case_context: "",
  positive_points: "",
  risk_points: "",
  credit_score_20: "",
  repayment_source_score_20: "",
  asset_exit_score_20: "",
  plan_specificity_score_20: "",
  uncertainty_control_score_20: "",
  total_score_100: "",
  user_decision: "",
  heaviest_issue: "",
  additional_checks: "",
  ringi_sentence: "",
  score_decision_gap_note: "",
  ai_feedback_outcome: "",
  ai_feedback_note: "",
  okf_candidate_tags: "",
  presentation_use_ok: "yes",
  reviewer: "",
  judged_at: "",
};

function clampScore(value: string) {
  if (value === "") return "";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "";
  return String(Math.max(0, Math.min(20, Math.round(numeric))));
}

function scoreDecision(total: number) {
  if (total >= 80) return "承認候補";
  if (total >= 65) return "条件付き承認候補";
  if (total >= 50) return "追加確認候補";
  return "否認候補";
}

const conditionCheckText = (condition: ConditionOption) => {
  switch (condition) {
    case "本件限度":
      return "本件限度での対応とし、追加与信は別途審査する";
    case "次回決算まで本件限度":
      return "次回決算確認までは本件限度とし、決算内容で継続可否を見直す";
    case "金融機関と協調":
      return "銀行融資50%・当社リース50%の協調リースとし、金融機関の支援姿勢を確認する";
    case "独立・新設向け条件":
      return "代表者保証、資金繰り、初回取引資料を確認する";
    case "親会社等保証":
      return "親会社等保証の有効性、保証意思、保証余力を確認する";
    case "担保・保全あり":
      return "担保・保全内容、換価性、実行可能性を確認する";
    case "その他":
      return "個別条件を手入力で補記する";
    case "事業計画確認":
      return "事業計画、売上根拠、稼働開始後の資金繰りを確認する";
    case "銀行との与信バランスを取る":
      return "銀行融資残高と当社リース与信のバランスを確認し、当社だけが過大与信にならないよう調整する";
    case "ピーク内与信":
      return "既存与信ピーク内での対応とし、総与信増加を抑える";
    case "利回り確保":
      return "リスクに見合う利回りを確保し、料率条件を承認条件に含める";
    case "再リース確認":
      return "満了後の再リース見込み、継続利用意思、物件耐用・保守状況を確認する";
  }
};

const conditionRingiText = (condition: ConditionOption) => {
  switch (condition) {
    case "本件限度":
      return "本件限度";
    case "次回決算まで本件限度":
      return "次回決算確認まで本件限度";
    case "金融機関と協調":
      return "銀行融資50%・当社リース50%の協調リース";
    case "独立・新設向け条件":
      return "独立・新設向けの初回取引条件";
    case "親会社等保証":
      return "親会社等保証";
    case "担保・保全あり":
      return "担保・保全設定";
    case "その他":
      return "個別条件";
    case "事業計画確認":
      return "事業計画確認";
    case "銀行との与信バランスを取る":
      return "銀行与信とのバランス調整";
    case "ピーク内与信":
      return "既存ピーク内与信";
    case "利回り確保":
      return "利回り確保";
    case "再リース確認":
      return "再リース見込み確認";
  }
};

function buildAdditionalChecks(item: DrillItem, conditions: ConditionOption[] = []) {
  const checks = [
    item.bank_is_main === "非メイン"
      ? `非メイン行として、銀行与信${item.bank_credit_balance_million_yen}百万円の支援姿勢と回収方針を確認`
      : `メイン行として、銀行与信${item.bank_credit_balance_million_yen}百万円の継続支援姿勢を確認`,
    item.lease_customer_status === "既存先"
      ? `既存リース与信${item.existing_lease_credit_balance_million_yen}百万円を含めた総与信と返済履歴を確認`
      : "新規先のため、代表者・取引実績・資金繰り資料で初回与信の妥当性を確認",
    item.risk_points || "未確定リスクの裏付け資料を確認",
    ...conditions.map(conditionCheckText),
  ];
  return checks.join("; ");
}

function buildRingiSentence(item: DrillItem, decision: string, total: number, conditions: ConditionOption[] = []) {
  const base = `${item.industry}向け${item.asset_type} ${item.amount_million_yen}百万円/${item.lease_term_years}年案件。`;
  const bank = `${item.bank_is_main}行の与信残高${item.bank_credit_balance_million_yen}百万円、リース${item.lease_customer_status}、自己資本比率${item.equity_ratio_percent}%を踏まえ、`;
  const issue = item.risk_points ? `主な論点は「${item.risk_points}」である。` : "主な論点は未確定リスクの確認である。";
  const conditionText = conditions.length
    ? `承認条件として${conditions.map(conditionRingiText).join("、")}を付す。`
    : "";
  return `${base}${bank}${total}点・${decision}として判断する。${issue}${conditionText}`;
}

function buildDecisionGapNote(item: DrillItem, decision: string, total: number) {
  const scoreSide = scoreDecision(total).replace("候補", "");
  if (decision === scoreSide) return "";
  return `点数上は${scoreSide}候補だが、${item.risk_points || item.case_context || "案件固有の未確定要素"}を重く見て${decision}とした。`;
}

function buildOneClickValues(item: DrillItem, preset: (typeof PRESETS)[number], conditions: ConditionOption[] = []) {
  const total = preset.scores.reduce((sum, score) => sum + score, 0);
  const next = { ...item };
  SCORE_FIELDS.forEach(([key], index) => {
    next[key] = String(preset.scores[index]);
  });
  next.total_score_100 = String(total);
  next.user_decision = preset.decision;
  next.heaviest_issue = item.risk_points || item.case_context || item.heaviest_issue || preset.note;
  next.additional_checks = buildAdditionalChecks(item, conditions);
  next.ringi_sentence = buildRingiSentence(item, preset.decision, total, conditions);
  next.score_decision_gap_note = buildDecisionGapNote(item, preset.decision, total);
  next.ai_feedback_outcome = item.ai_feedback_outcome || "neutral";
  next.ai_feedback_note = item.ai_feedback_note || "ボタン処理で初期判断。必要に応じて後で修正する。";
  next.presentation_use_ok = item.presentation_use_ok || "yes";
  return next;
}

function updateGeneratedTexts(item: DrillItem, conditions: ConditionOption[], total: number) {
  const decision = item.user_decision || scoreDecision(total).replace("候補", "");
  return {
    ...item,
    additional_checks: buildAdditionalChecks(item, conditions),
    ringi_sentence: buildRingiSentence(item, decision, total, conditions),
    score_decision_gap_note: buildDecisionGapNote(item, decision, total),
  };
}

export default function JudgmentDrillPage() {
  const [response, setResponse] = useState<DrillResponse | null>(null);
  const [form, setForm] = useState<DrillItem>(emptyItem);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [jump, setJump] = useState("");
  const [selectedConditions, setSelectedConditions] = useState<ConditionOption[]>([]);

  const loadCase = useCallback(async (index: number) => {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const res = await fetch(`/api/judgment-drill?index=${index}`, { cache: "no-store" });
      const data = (await res.json()) as DrillResponse & { error?: string };
      if (!res.ok) throw new Error(data.error || "読み込みに失敗しました");
      setResponse(data);
      setForm(data.item ?? emptyItem);
      setSelectedConditions([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "読み込みに失敗しました");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadCaseId = useCallback(async (caseId: string) => {
    if (!caseId.trim()) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const res = await fetch(`/api/judgment-drill?case_id=${encodeURIComponent(caseId.trim())}`, {
        cache: "no-store",
      });
      const data = (await res.json()) as DrillResponse & { error?: string };
      if (!res.ok) throw new Error(data.error || "読み込みに失敗しました");
      setResponse(data);
      setForm(data.item ?? emptyItem);
      setSelectedConditions([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "読み込みに失敗しました");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCase(0);
  }, [loadCase]);

  const scoreTotal = useMemo(
    () =>
      SCORE_FIELDS.reduce((sum, [key]) => {
        const value = Number(form[key]);
        return sum + (Number.isFinite(value) ? value : 0);
      }, 0),
    [form],
  );

  const completedRate = response?.total
    ? Math.round(((response.completed ?? 0) / response.total) * 100)
    : 0;

  const update = (key: keyof DrillItem, value: string) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const updateScore = (key: (typeof SCORE_FIELDS)[number][0], value: string) => {
    const nextValue = clampScore(value);
    setForm((current) => {
      const next = { ...current, [key]: nextValue };
      const total = SCORE_FIELDS.reduce((sum, [scoreKey]) => {
        const numeric = Number(next[scoreKey]);
        return sum + (Number.isFinite(numeric) ? numeric : 0);
      }, 0);
      return { ...next, total_score_100: String(total) };
    });
  };

  const updateDecision = (decision: string) => {
    setForm((current) => {
      const total = SCORE_FIELDS.reduce((sum, [key]) => {
        const value = Number(current[key]);
        return sum + (Number.isFinite(value) ? value : 0);
      }, 0);
      return updateGeneratedTexts({ ...current, user_decision: decision }, selectedConditions, total);
    });
  };

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    setForm((current) => {
      const next = { ...current };
      SCORE_FIELDS.forEach(([key], index) => {
        next[key] = String(preset.scores[index]);
      });
      const total = preset.scores.reduce((sum, score) => sum + score, 0);
      next.total_score_100 = String(total);
      next.user_decision = preset.decision;
      if (!next.heaviest_issue) next.heaviest_issue = current.risk_points;
      next.additional_checks = buildAdditionalChecks(current, selectedConditions);
      next.ringi_sentence = buildRingiSentence(current, preset.decision, total, selectedConditions);
      next.score_decision_gap_note = buildDecisionGapNote(current, preset.decision, total);
      return next;
    });
  };

  const toggleCondition = (condition: ConditionOption) => {
    setSelectedConditions((current) => {
      const nextConditions = current.includes(condition)
        ? current.filter((item) => item !== condition)
        : [...current, condition];
      setForm((currentForm) => {
        const total = SCORE_FIELDS.reduce((sum, [key]) => {
          const value = Number(currentForm[key]);
          return sum + (Number.isFinite(value) ? value : 0);
        }, 0);
        return updateGeneratedTexts(currentForm, nextConditions, total);
      });
      return nextConditions;
    });
  };

  const saveValues = async (values: DrillItem, moveNext = false) => {
    if (!values.case_id) return;
    setSaving(true);
    setError("");
    setNotice("");
    const total = SCORE_FIELDS.reduce((sum, [key]) => {
      const value = Number(values[key]);
      return sum + (Number.isFinite(value) ? value : 0);
    }, 0);
    try {
      const res = await fetch("/api/judgment-drill", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: values.case_id,
          values: {
            credit_score_20: values.credit_score_20,
            repayment_source_score_20: values.repayment_source_score_20,
            asset_exit_score_20: values.asset_exit_score_20,
            plan_specificity_score_20: values.plan_specificity_score_20,
            uncertainty_control_score_20: values.uncertainty_control_score_20,
            total_score_100: String(total),
            user_decision: values.user_decision,
            heaviest_issue: values.heaviest_issue,
            additional_checks: values.additional_checks,
            ringi_sentence: values.ringi_sentence,
            score_decision_gap_note: values.score_decision_gap_note,
            ai_feedback_outcome: values.ai_feedback_outcome,
            ai_feedback_note: values.ai_feedback_note,
            okf_candidate_tags: values.okf_candidate_tags,
            presentation_use_ok: values.presentation_use_ok,
            reviewer: values.reviewer,
          },
        }),
      });
      const data = (await res.json()) as DrillResponse & { error?: string };
      if (!res.ok) throw new Error(data.error || "保存に失敗しました");
      setResponse(data);
      setForm(data.item ?? emptyItem);
      setNotice("保存しました");
      if (moveNext && data.index + 1 < data.total) {
        await loadCase(data.index + 1);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存に失敗しました");
    } finally {
      setSaving(false);
    }
  };

  const save = async (moveNext = false) => saveValues(form, moveNext);

  const oneClickJudge = async (preset: (typeof PRESETS)[number]) => {
    const next = buildOneClickValues(form, preset, selectedConditions);
    setForm(next);
    await saveValues(next, true);
  };

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center bg-slate-900 text-white">
                <ClipboardList className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs font-black uppercase tracking-widest text-slate-500">
                  Human Judgment Asset
                </p>
                <h1 className="mt-1 text-xl font-black text-slate-950">
                  1000本判断ドリル入力シート
                </h1>
                <p className="mt-1 text-sm leading-6 text-slate-600">
                  CSVを直接触らず、1件ずつ点数・条件・稟議文・AIへのダメ出しを保存します。
                </p>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-3 lg:ml-auto lg:w-[520px]">
              <Metric label="完了" value={`${response?.completed ?? 0}`} />
              <Metric label="未了" value={`${response?.pending ?? 0}`} />
              <Metric label="進捗" value={`${completedRate}%`} />
            </div>
          </div>

          <div className="mt-4 h-2 overflow-hidden bg-slate-100">
            <div className="h-full bg-emerald-500" style={{ width: `${completedRate}%` }} />
          </div>
        </header>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <section className="space-y-5">
            <div className="border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center">
                <div>
                  <p className="text-xs font-black text-slate-500">
                    {response ? `${response.index + 1} / ${response.total}` : "-"}
                  </p>
                  <h2 className="mt-1 text-lg font-black text-slate-950">
                    {form.case_id || "読み込み中"}
                  </h2>
                </div>
                <div className="flex flex-wrap gap-2 md:ml-auto">
                  <button
                    type="button"
                    onClick={() => response && loadCase(response.index - 1)}
                    disabled={loading || !response || response.index <= 0}
                    className="inline-flex items-center gap-2 border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 disabled:opacity-40"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    前
                  </button>
                  <button
                    type="button"
                    onClick={() => response && loadCase(response.index + 1)}
                    disabled={loading || !response || response.index + 1 >= response.total}
                    className="inline-flex items-center gap-2 border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 disabled:opacity-40"
                  >
                    次
                    <ArrowRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => response && response.nextPendingIndex >= 0 && loadCase(response.nextPendingIndex)}
                    disabled={loading || !response || response.nextPendingIndex < 0}
                    className="inline-flex items-center gap-2 border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-bold text-emerald-800 disabled:opacity-40"
                  >
                    <SkipForward className="h-4 w-4" />
                    未了へ
                  </button>
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <input
                  value={jump}
                  onChange={(event) => setJump(event.target.value)}
                  placeholder="CASE-JD-20260725-0123"
                  className="min-w-0 flex-1 border border-slate-300 px-3 py-2 text-sm font-bold outline-none focus:border-slate-900"
                />
                <button
                  type="button"
                  onClick={() => loadCaseId(jump)}
                  className="inline-flex items-center gap-2 bg-slate-900 px-3 py-2 text-sm font-bold text-white"
                >
                  <Search className="h-4 w-4" />
                  検索
                </button>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700">
                <AlertCircle className="h-5 w-5" />
                {error}
              </div>
            )}

            {notice && (
              <div className="flex items-center gap-2 border border-emerald-200 bg-emerald-50 p-4 text-sm font-bold text-emerald-700">
                <Check className="h-5 w-5" />
                {notice}
              </div>
            )}

            <article className="border border-slate-200 bg-white p-5 shadow-sm">
              {loading ? (
                <div className="flex h-64 items-center justify-center text-sm font-bold text-slate-500">
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  読み込み中
                </div>
              ) : (
                <div className="space-y-5">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Fact label="テーマ" value={form.case_theme} />
                    <Fact label="業種" value={form.industry} />
                    <Fact label="企業状態" value={form.company_stage} />
                    <Fact label="自己資本比率" value={`${form.equity_ratio_percent}%`} />
                    <Fact label="銀行" value={form.bank_is_main} />
                    <Fact label="銀行与信" value={`${form.bank_credit_balance_million_yen}百万円`} />
                    <Fact label="リース取引" value={form.lease_customer_status} />
                    <Fact label="既存リース与信" value={`${form.existing_lease_credit_balance_million_yen}百万円`} />
                    <Fact label="物件" value={form.asset_type} />
                    <Fact label="申込額" value={`${form.amount_million_yen}百万円`} />
                    <Fact label="期間" value={`${form.lease_term_years}年`} />
                    <Fact label="目的" value={form.purpose} />
                  </div>

                  <div className="grid gap-3 lg:grid-cols-3">
                    <Narrative label="財務" value={form.financial_snapshot} />
                    <Narrative label="良い材料" value={form.positive_points} />
                    <Narrative label="懸念" value={form.risk_points} />
                  </div>

                  <Narrative label="案件文脈" value={form.case_context} />
                </div>
              )}
            </article>
          </section>

          <aside className="space-y-5">
            <section className="border border-indigo-200 bg-white p-5 shadow-sm">
              <div>
                <p className="text-xs font-black uppercase tracking-widest text-indigo-600">
                  Conditions
                </p>
                <h2 className="mt-1 text-lg font-black text-slate-950">成約/承認の付帯条件</h2>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  押すと追加確認と稟議文が変わります。協調は銀行融資50%・当社リース50%で生成します。
                </p>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {CONDITION_OPTIONS.map((condition) => {
                  const active = selectedConditions.includes(condition);
                  return (
                    <button
                      key={condition}
                      type="button"
                      onClick={() => toggleCondition(condition)}
                      disabled={loading || !form.case_id}
                      className={`border-2 px-3 py-2 text-xs font-black transition-all disabled:opacity-50 ${
                        active
                          ? "border-indigo-600 bg-indigo-600 text-white shadow-lg shadow-indigo-200"
                          : "border-slate-100 bg-white text-slate-400 hover:border-slate-300 hover:text-slate-700"
                      }`}
                    >
                      {condition}
                    </button>
                  );
                })}
              </div>
            </section>

            <section className="border border-emerald-200 bg-emerald-50 p-5 shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-black uppercase tracking-widest text-emerald-700">
                    One Click
                  </p>
                  <h2 className="mt-1 text-lg font-black text-slate-950">押すだけで保存して次へ</h2>
                  <p className="mt-1 text-xs leading-5 text-emerald-900">
                    採点・結論・主論点・追加確認・稟議文を自動で埋めます。違う時だけ下で修正してください。
                  </p>
                </div>
                {saving && <Loader2 className="h-5 w-5 animate-spin text-emerald-700" />}
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                {PRESETS.map((preset) => (
                  <button
                    key={`one-click-${preset.label}`}
                    type="button"
                    onClick={() => oneClickJudge(preset)}
                    disabled={saving || loading || !form.case_id}
                    className="flex min-h-20 flex-col items-start justify-center border border-emerald-300 bg-white px-3 py-3 text-left transition hover:border-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
                  >
                    <span className="text-sm font-black text-slate-950">{preset.label}</span>
                    <span className="mt-1 text-[11px] font-bold leading-4 text-slate-500">
                      {preset.decision} / {preset.scores.reduce((sum, score) => sum + score, 0)}点
                    </span>
                    <span className="mt-1 text-[11px] leading-4 text-emerald-800">{preset.note}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-black uppercase tracking-widest text-slate-500">
                    Scoring
                  </p>
                  <h2 className="mt-1 text-lg font-black text-slate-950">あなたの判断</h2>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-black text-slate-950">{scoreTotal}</p>
                  <p className="text-xs font-bold text-slate-500">{scoreDecision(scoreTotal)}</p>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => applyPreset(preset)}
                    className="border border-slate-300 bg-slate-50 px-3 py-2 text-sm font-black text-slate-800 hover:border-slate-900 hover:bg-white"
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              <div className="mt-4 space-y-3">
                {SCORE_FIELDS.map(([key, label]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-bold text-slate-700">{label}</span>
                      <input
                        type="number"
                        inputMode="numeric"
                        min={0}
                        max={20}
                        value={form[key]}
                        onChange={(event) => updateScore(key, event.target.value)}
                        className="w-16 border border-slate-300 px-2 py-1 text-right text-sm font-black outline-none focus:border-slate-900"
                      />
                    </div>
                    <div className="grid grid-cols-5 gap-1">
                      {[20, 15, 10, 5, 0].map((score) => (
                        <button
                          key={score}
                          type="button"
                          onClick={() => updateScore(key, String(score))}
                          className={`border px-2 py-1 text-xs font-black ${
                            form[key] === String(score)
                              ? "border-slate-900 bg-slate-900 text-white"
                              : "border-slate-200 bg-white text-slate-600"
                          }`}
                        >
                          {score}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-5 grid grid-cols-2 gap-2">
                {DECISIONS.map((decision) => (
                  <button
                    key={decision}
                    type="button"
                    onClick={() => updateDecision(decision)}
                    className={`border px-3 py-2 text-sm font-black ${
                      form.user_decision === decision
                        ? "border-slate-900 bg-slate-900 text-white"
                        : "border-slate-300 bg-white text-slate-700"
                    }`}
                  >
                    {decision}
                  </button>
                ))}
              </div>
            </section>

            <section className="border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => update("heaviest_issue", form.risk_points)}
                  className="border border-slate-300 bg-slate-50 px-2 py-2 text-xs font-black text-slate-700 hover:border-slate-900"
                >
                  懸念を主論点
                </button>
                <button
                  type="button"
                  onClick={() => update("additional_checks", buildAdditionalChecks(form, selectedConditions))}
                  className="border border-slate-300 bg-slate-50 px-2 py-2 text-xs font-black text-slate-700 hover:border-slate-900"
                >
                  確認事項作成
                </button>
                <button
                  type="button"
                  onClick={() => update("ringi_sentence", buildRingiSentence(form, form.user_decision || scoreDecision(scoreTotal).replace("候補", ""), scoreTotal, selectedConditions))}
                  className="border border-slate-300 bg-slate-50 px-2 py-2 text-xs font-black text-slate-700 hover:border-slate-900"
                >
                  稟議文作成
                </button>
              </div>
              <div className="space-y-3">
                <TextArea label="一番重い論点" value={form.heaviest_issue} onChange={(value) => update("heaviest_issue", value)} rows={3} />
                <TextArea label="追加確認" value={form.additional_checks} onChange={(value) => update("additional_checks", value)} rows={3} />
                <TextArea label="稟議に残す一文" value={form.ringi_sentence} onChange={(value) => update("ringi_sentence", value)} rows={3} />
                <TextArea label="点数と結論がズレた理由" value={form.score_decision_gap_note} onChange={(value) => update("score_decision_gap_note", value)} rows={2} />
              </div>
            </section>

            <section className="border border-slate-200 bg-white p-5 shadow-sm">
              <p className="text-sm font-black text-slate-950">AI案へのダメ出し</p>
              <div className="mt-3 grid grid-cols-2 gap-2">
                {AI_OUTCOMES.map(([value, label]) => (
                  <button
                    key={value || "none"}
                    type="button"
                    onClick={() => update("ai_feedback_outcome", value)}
                    className={`border px-3 py-2 text-sm font-black ${
                      form.ai_feedback_outcome === value
                        ? "border-indigo-700 bg-indigo-700 text-white"
                        : "border-slate-300 bg-white text-slate-700"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="mt-3 space-y-3">
                <TextArea label="ダメ出し・採用理由" value={form.ai_feedback_note} onChange={(value) => update("ai_feedback_note", value)} rows={3} />
                <Input label="OKF候補タグ" value={form.okf_candidate_tags} onChange={(value) => update("okf_candidate_tags", value)} />
                <Input label="レビュー者" value={form.reviewer} onChange={(value) => update("reviewer", value)} />
              </div>
            </section>

            <div className="sticky bottom-4 grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => save(false)}
                disabled={saving || loading}
                className="inline-flex items-center justify-center gap-2 bg-slate-900 px-4 py-3 text-sm font-black text-white disabled:opacity-50"
              >
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                保存
              </button>
              <button
                type="button"
                onClick={() => save(true)}
                disabled={saving || loading}
                className="inline-flex items-center justify-center gap-2 bg-emerald-600 px-4 py-3 text-sm font-black text-white disabled:opacity-50"
              >
                保存して次へ
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-slate-200 bg-slate-50 px-4 py-3">
      <p className="text-xs font-bold text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-black text-slate-950">{value}</p>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-slate-200 bg-slate-50 px-3 py-2">
      <p className="text-[11px] font-black text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-black text-slate-950">{value || "-"}</p>
    </div>
  );
}

function Narrative({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-slate-200 bg-white p-3">
      <p className="text-xs font-black text-slate-500">{label}</p>
      <p className="mt-1 text-sm leading-6 text-slate-800">{value || "-"}</p>
    </div>
  );
}

function TextArea({
  label,
  value,
  rows,
  onChange,
}: {
  label: string;
  value: string;
  rows: number;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs font-black text-slate-600">{label}</span>
      <textarea
        value={value}
        rows={rows}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 w-full resize-y border border-slate-300 px-3 py-2 text-sm leading-6 outline-none focus:border-slate-900"
      />
    </label>
  );
}

function Input({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs font-black text-slate-600">{label}</span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 w-full border border-slate-300 px-3 py-2 text-sm font-bold outline-none focus:border-slate-900"
      />
    </label>
  );
}
