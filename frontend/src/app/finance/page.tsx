"use client";

import React, { useMemo, useState } from "react";
import axios from "axios";
import {
  AlertCircle,
  Banknote,
  BookOpen,
  CheckCircle2,
  Factory,
  Loader2,
  Route,
  Save,
  Search,
  ShieldCheck,
  TrendingDown,
} from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type AssetType = "建機" | "工作機械" | "PC/IT" | "医療機器" | "ドローン" | "車両";
type FinancialScore = "High" | "Medium" | "Low";

type FinanceInput = {
  asset_name: string;
  asset_type: AssetType;
  term: number;
  down_payment: number;
  financial_score: FinancialScore;
  main_bank_support: boolean;
  bank_coordination: boolean;
  core_business: boolean;
  related_assets: boolean;
  annual_km: number;
  has_maintenance_lease: boolean;
  ai_residual_pct: string;
};

type CurvePoint = {
  month: number;
  asset_value: number;
  lease_balance: number;
};

type FinanceResult = {
  score: number;
  decision: string;
  icon: string;
  bep_month: number;
  bep_ratio: number;
  reasons: string[];
  deductions: string[];
  marketing_advice: string;
  bank_comparison: string;
  action_plan: string[];
  curve: CurvePoint[];
  asset_params: {
    depreciation_rate: number;
    priority: string;
    priority_score: number;
    info: string;
  };
  input: Record<string, unknown>;
};

type ObsidianHit = {
  path: string;
  snippet: string;
};

type ObsidianContext = {
  query?: string;
  generated_terms?: string[];
  hits: ObsidianHit[];
  evidence?: {
    used_market?: string[];
    residual_risk?: string[];
    approval_basis?: string[];
    cautions?: string[];
  };
  digest: {
    digest?: string;
    source_count?: string;
    links?: string;
  };
};

type SimilarNotesResult = {
  similar_notes: ObsidianHit[];
};

const ASSET_TYPES: AssetType[] = ["建機", "工作機械", "PC/IT", "医療機器", "ドローン", "車両"];

const FIN_LABELS: Record<FinancialScore, string> = {
  High: "優良",
  Medium: "標準",
  Low: "低評価",
};

const DECISION_STYLE: Record<string, string> = {
  承認: "border-emerald-500 bg-emerald-50 text-emerald-700",
  条件付き承認: "border-amber-500 bg-amber-50 text-amber-700",
  "要審議（上位承認）": "border-orange-500 bg-orange-50 text-orange-700",
  否決: "border-rose-500 bg-rose-50 text-rose-700",
};

const initialInput: FinanceInput = {
  asset_name: "",
  asset_type: "車両",
  term: 60,
  down_payment: 0.2,
  financial_score: "Medium",
  main_bank_support: false,
  bank_coordination: false,
  core_business: true,
  related_assets: false,
  annual_km: 15000,
  has_maintenance_lease: false,
  ai_residual_pct: "",
};

function stripBold(text: string) {
  return text.replace(/\*\*/g, "");
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function ToggleRow({
  label,
  caption,
  checked,
  onChange,
}: {
  label: string;
  caption: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 rounded-lg border border-slate-200 p-3 hover:bg-slate-50 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 accent-cyan-600"
      />
      <span>
        <span className="block text-sm font-black text-slate-700">{label}</span>
        <span className="block text-xs text-slate-500 mt-1">{caption}</span>
      </span>
    </label>
  );
}

export default function FinancePage() {
  const [form, setForm] = useState<FinanceInput>(initialInput);
  const [result, setResult] = useState<FinanceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [memoQuery, setMemoQuery] = useState("");
  const [obsidianContext, setObsidianContext] = useState<ObsidianContext | null>(null);
  const [obsidianLoading, setObsidianLoading] = useState(false);
  const [obsidianSaving, setObsidianSaving] = useState(false);
  const [obsidianMessage, setObsidianMessage] = useState<string | null>(null);
  const [similarNotes, setSimilarNotes] = useState<ObsidianHit[]>([]);
  const [similarLoading, setSimilarLoading] = useState(false);

  const decisionClass = useMemo(() => {
    if (!result) return "border-slate-300 bg-slate-50 text-slate-600";
    return DECISION_STYLE[result.decision] || "border-slate-300 bg-slate-50 text-slate-600";
  }, [result]);

  const activeInput = result?.input || form;
  const activeTerm = Number(activeInput.term || form.term);

  const update = <K extends keyof FinanceInput>(key: K, value: FinanceInput[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setResult(null);
    setError(null);
    setObsidianContext(null);
    setObsidianMessage(null);
    setSimilarNotes([]);
  };

  const fetchSimilarNotes = async (sourceInput: Record<string, unknown>, decision = "") => {
    setSimilarLoading(true);
    try {
      const res = await axios.post<SimilarNotesResult>("/api/asset-finance/similar-notes", {
        asset_type: sourceInput.asset_type || form.asset_type,
        asset_name: sourceInput.asset_name || form.asset_name,
        financial_score: sourceInput.financial_score || form.financial_score,
        decision,
        memo_query: memoQuery,
      });
      setSimilarNotes(res.data.similar_notes || []);
    } catch {
      setSimilarNotes([]);
    } finally {
      setSimilarLoading(false);
    }
  };

  const submit = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...form,
        ai_residual_pct: form.ai_residual_pct === "" ? null : Number(form.ai_residual_pct),
      };
      const res = await axios.post<FinanceResult>("/api/asset-finance/evaluate", payload);
      setResult(res.data);
      setObsidianMessage(null);
      fetchSimilarNotes(res.data.input || payload, res.data.decision);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || "物件ファイナンス審査に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  const searchObsidian = async () => {
    setObsidianLoading(true);
    setObsidianMessage(null);
    try {
      const res = await axios.post<ObsidianContext>("/api/asset-finance/obsidian-context", {
        asset_type: activeInput.asset_type || form.asset_type,
        asset_name: activeInput.asset_name || form.asset_name,
        financial_score: activeInput.financial_score || form.financial_score,
        decision: result?.decision || "",
        memo_query: memoQuery,
      });
      setObsidianContext(res.data);
      setObsidianMessage(res.data.hits.length ? null : "関連メモは見つかりませんでした。");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setObsidianMessage(detail || "Obsidian検索に失敗しました");
    } finally {
      setObsidianLoading(false);
    }
  };

  const saveToObsidian = async () => {
    if (!result) return;
    setObsidianSaving(true);
    setObsidianMessage(null);
    try {
      const relatedPaths = obsidianContext?.hits.map((hit) => hit.path) || [];
      await axios.post("/api/asset-finance/save-to-obsidian", {
        input: result.input || form,
        result,
        related_paths: relatedPaths,
      });
      setObsidianMessage("審査結果をObsidianへ保存しました。");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setObsidianMessage(detail || "Obsidian保存に失敗しました");
    } finally {
      setObsidianSaving(false);
    }
  };

  const EvidenceList = ({
    title,
    items,
    tone,
  }: {
    title: string;
    items?: string[];
    tone: "blue" | "amber" | "emerald" | "rose";
  }) => {
    if (!items?.length) return null;
    const tones = {
      blue: "border-blue-100 bg-blue-50 text-blue-800",
      amber: "border-amber-100 bg-amber-50 text-amber-800",
      emerald: "border-emerald-100 bg-emerald-50 text-emerald-800",
      rose: "border-rose-100 bg-rose-50 text-rose-800",
    };
    return (
      <div className={`rounded-lg border p-3 ${tones[tone]}`}>
        <div className="text-xs font-black mb-2">{title}</div>
        <div className="space-y-1">
          {items.slice(0, 4).map((item, i) => (
            <div key={`${title}-${i}`} className="text-xs font-bold leading-relaxed">・{item}</div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-6">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Factory className="w-8 h-8 text-cyan-600" />
          物件ファイナンス審査
        </h1>
        <p className="text-slate-500 font-medium mt-1">
          物件価値、リース残債、BEP、定性緩和因子から「なぜ通せるか」を確認します。
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-1 space-y-4">
          <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5 space-y-4">
            <div>
              <label className="block text-xs font-black text-slate-500 mb-2">物件名 任意</label>
              <input
                value={form.asset_name}
                onChange={(e) => update("asset_name", e.target.value)}
                placeholder="例: 4t冷凍車、マシニングセンタ"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-cyan-500/20"
              />
            </div>

            <div>
              <label className="block text-xs font-black text-slate-500 mb-2">物件種別</label>
              <select
                value={form.asset_type}
                onChange={(e) => update("asset_type", e.target.value as AssetType)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-cyan-500/20"
              >
                {ASSET_TYPES.map((asset) => (
                  <option key={asset} value={asset}>{asset}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-black text-slate-500 mb-2">リース期間</label>
                <input
                  type="number"
                  min={12}
                  max={84}
                  step={6}
                  value={form.term}
                  onChange={(e) => update("term", Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-slate-700 outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-black text-slate-500 mb-2">自己資金率</label>
                <select
                  value={form.down_payment}
                  onChange={(e) => update("down_payment", Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-700 outline-none"
                >
                  {[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5].map((v) => (
                    <option key={v} value={v}>{percent(v)}</option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label className="block text-xs font-black text-slate-500 mb-2">財務評価</label>
              <div className="grid grid-cols-3 gap-2">
                {(["High", "Medium", "Low"] as FinancialScore[]).map((score) => (
                  <button
                    key={score}
                    onClick={() => update("financial_score", score)}
                    className={`rounded-lg border px-3 py-2 text-sm font-black ${
                      form.financial_score === score
                        ? "border-cyan-500 bg-cyan-50 text-cyan-700"
                        : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {FIN_LABELS[score]}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="text-xs font-black text-slate-500">定性緩和因子</div>
              <ToggleRow
                label="メイン銀行の支援先"
                caption="+50点。推薦・協調の見込みがある案件。"
                checked={form.main_bank_support}
                onChange={(checked) => update("main_bank_support", checked)}
              />
              <ToggleRow
                label="銀行協調案件"
                caption="+20点。銀行との協調が可能な案件。"
                checked={form.bank_coordination}
                onChange={(checked) => update("bank_coordination", checked)}
              />
              <ToggleRow
                label="本業利用物件"
                caption="+20点。事業の根幹に関わり支払優先度が高い物件。"
                checked={form.core_business}
                onChange={(checked) => update("core_business", checked)}
              />
              <ToggleRow
                label="関係者資産による保全"
                caption="+15点。追加保全が見込める案件。"
                checked={form.related_assets}
                onChange={(checked) => update("related_assets", checked)}
              />
            </div>

            {form.asset_type === "車両" && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
                <div>
                  <label className="block text-xs font-black text-slate-500 mb-2">予想年間走行距離</label>
                  <input
                    type="number"
                    min={0}
                    max={100000}
                    step={1000}
                    value={form.annual_km}
                    onChange={(e) => update("annual_km", Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-slate-700 outline-none"
                  />
                  {form.annual_km >= 20000 && (
                    <div className="mt-2 text-xs font-bold text-amber-700">年2万km以上は過走行補正が入ります。</div>
                  )}
                </div>
                <ToggleRow
                  label="メンテナンスリース付帯"
                  caption="+10点。中古価値の維持を評価。"
                  checked={form.has_maintenance_lease}
                  onChange={(checked) => update("has_maintenance_lease", checked)}
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-black text-slate-500 mb-2">AI残価率 任意</label>
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={form.ai_residual_pct}
                onChange={(e) => update("ai_residual_pct", e.target.value)}
                placeholder="例: 35"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-slate-700 outline-none"
              />
            </div>

            <button
              onClick={submit}
              disabled={loading}
              className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-600 py-3 text-sm font-black text-white hover:bg-cyan-500 disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
              審査判定を実行
            </button>
          </div>

          {error && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700 flex gap-3">
              <AlertCircle className="w-5 h-5 shrink-0" />
              {error}
            </div>
          )}

          <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5 space-y-3">
            <h2 className="text-sm font-black text-slate-700 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-cyan-600" />
              Obsidian関連メモ
            </h2>
            <input
              value={memoQuery}
              onChange={(e) => setMemoQuery(e.target.value)}
              placeholder="任意: 承認条件、補助金、残価など"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-cyan-500/20"
            />
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={searchObsidian}
                disabled={obsidianLoading}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs font-black text-cyan-700 hover:bg-cyan-100 disabled:opacity-50"
              >
                {obsidianLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                メモ検索
              </button>
              <button
                onClick={saveToObsidian}
                disabled={!result || obsidianSaving}
                className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-black text-slate-700 hover:bg-slate-50 disabled:opacity-50"
              >
                {obsidianSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                結果保存
              </button>
            </div>
            {obsidianMessage && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-600">
                {obsidianMessage}
              </div>
            )}
            {obsidianContext?.generated_terms?.length ? (
              <div className="flex flex-wrap gap-1">
                {obsidianContext.generated_terms.slice(0, 12).map((term) => (
                  <span key={term} className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] font-bold text-slate-500">
                    {term}
                  </span>
                ))}
              </div>
            ) : null}
            {obsidianContext?.evidence && (
              <div className="space-y-2">
                <EvidenceList title="中古相場・再販観点" items={obsidianContext.evidence.used_market} tone="blue" />
                <EvidenceList title="残価・再販リスク" items={obsidianContext.evidence.residual_risk} tone="amber" />
                <EvidenceList title="稟議で使える根拠" items={obsidianContext.evidence.approval_basis} tone="emerald" />
                <EvidenceList title="注意すべき物件特性" items={obsidianContext.evidence.cautions} tone="rose" />
              </div>
            )}
            {obsidianContext?.hits.length ? (
              <div className="space-y-2">
                {obsidianContext.hits.slice(0, 5).map((hit) => (
                  <div key={hit.path} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                    <div className="text-xs font-black text-slate-700 truncate">{hit.path}</div>
                    <div className="mt-1 text-xs leading-relaxed text-slate-500 line-clamp-3">
                      {hit.snippet}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>

        <div className="xl:col-span-2 space-y-6">
          {result ? (
            <>
              <div className={`border-t-4 rounded-lg p-5 shadow-sm ${decisionClass}`}>
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="text-sm font-black opacity-80">判定</div>
                    <div className="mt-1 text-3xl font-black">{result.decision}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-black opacity-80">総合スコア</div>
                    <div className="mt-1 text-3xl font-black">{result.score}点</div>
                  </div>
                </div>
                <div className="mt-3 text-sm font-bold">
                  BEP {result.bep_month}ヶ月目 / {activeTerm}ヶ月（期間比 {Math.round(result.bep_ratio * 100)}%）
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-black text-slate-700 flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-cyan-600" />
                    物件時価 vs リース残債
                  </h2>
                  <div className="text-xs font-bold text-slate-500">
                    減価率 {Math.round(result.asset_params.depreciation_rate * 100)}% / 支払優先度 {result.asset_params.priority}
                  </div>
                </div>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.curve} margin={{ top: 10, right: 24, left: 0, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                      <YAxis tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} tick={{ fontSize: 12 }} />
                      <Tooltip
                        formatter={(value) => `${Math.round(Number(value || 0) * 100)}%`}
                        labelFormatter={(label) => `${label}ヶ月目`}
                      />
                      <Legend />
                      <ReferenceLine x={result.bep_month} stroke="#16a34a" strokeDasharray="4 4" label="BEP" />
                      <Line type="monotone" dataKey="asset_value" name="物件時価率" stroke="#2563eb" strokeWidth={3} dot={false} />
                      <Line type="monotone" dataKey="lease_balance" name="リース残債率" stroke="#dc2626" strokeWidth={3} strokeDasharray="6 4" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                  <h2 className="text-sm font-black text-slate-700 flex items-center gap-2 mb-4">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    承認根拠
                  </h2>
                  <div className="space-y-2">
                    {result.reasons.map((reason, i) => (
                      <div key={i} className="rounded-lg bg-emerald-50 border border-emerald-100 px-3 py-2 text-sm font-bold text-emerald-800">
                        {reason}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                  <h2 className="text-sm font-black text-slate-700 flex items-center gap-2 mb-4">
                    <AlertCircle className="w-4 h-4 text-amber-600" />
                    減点・リスク要因
                  </h2>
                  <div className="space-y-2">
                    {result.deductions.length ? result.deductions.map((risk, i) => (
                      <div key={i} className="rounded-lg bg-amber-50 border border-amber-100 px-3 py-2 text-sm font-bold text-amber-800">
                        {risk}
                      </div>
                    )) : (
                      <div className="rounded-lg bg-slate-50 border border-slate-100 px-3 py-2 text-sm font-bold text-slate-500">
                        主要な減点要因はありません。
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                  <h2 className="text-sm font-black text-slate-700 flex items-center gap-2 mb-3">
                    <Banknote className="w-4 h-4 text-blue-600" />
                    銀行システムとの差異
                  </h2>
                  <p className="text-sm leading-relaxed text-slate-700 font-medium whitespace-pre-line">{stripBold(result.bank_comparison)}</p>
                </div>
                <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                  <h2 className="text-sm font-black text-slate-700 flex items-center gap-2 mb-3">
                    <Route className="w-4 h-4 text-violet-600" />
                    ライフサイクル提案
                  </h2>
                  <p className="text-sm leading-relaxed text-slate-700 font-medium">{stripBold(result.marketing_advice)}</p>
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                <h2 className="text-sm font-black text-slate-700 mb-4">営業アクションプラン</h2>
                <div className="space-y-3">
                  {result.action_plan.map((plan, i) => (
                    <div key={i} className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="h-7 w-7 shrink-0 rounded-full bg-slate-900 text-white flex items-center justify-center text-xs font-black">
                        {i + 1}
                      </div>
                      <div className="text-sm font-bold leading-relaxed text-slate-700">{stripBold(plan)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-5">
                <div className="flex items-center justify-between gap-3 mb-4">
                  <h2 className="text-sm font-black text-slate-700">類似物件・過去案件メモ</h2>
                  <button
                    onClick={() => fetchSimilarNotes(result.input || form, result.decision)}
                    disabled={similarLoading}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-black text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                  >
                    {similarLoading ? "検索中..." : "再検索"}
                  </button>
                </div>
                {similarNotes.length ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {similarNotes.slice(0, 4).map((note) => (
                      <div key={note.path} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                        <div className="text-xs font-black text-slate-700 truncate">{note.path}</div>
                        <div className="mt-2 text-xs leading-relaxed text-slate-500 line-clamp-4">{note.snippet}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-slate-100 bg-slate-50 p-3 text-sm font-bold text-slate-500">
                    {similarLoading ? "類似メモを検索しています。" : "保存済みの類似物件メモはまだありません。"}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-10 min-h-96 flex flex-col items-center justify-center text-center">
              <Factory className="w-16 h-16 text-slate-300 mb-4" />
              <h2 className="text-lg font-black text-slate-700">条件を入力して物件保全性を判定</h2>
              <p className="text-sm text-slate-500 mt-2 max-w-xl">
                財務だけでは弱い案件でも、物件価値、BEP、支払優先度、銀行協調などの緩和因子を分解して確認できます。
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
