import { BadgeInfo, FileOutput } from "lucide-react";
import { ScoringFormData } from "../../types";

// screening（PC・ScoringFormData）と lease-kun（スマホウィザード・入力欄は文字列）の
// 両方から呼べるよう、数値項目は string も受け付ける緩めた型にしている。
export type IssuePolicyFormData = Partial<
  Pick<ScoringFormData, "customer_type" | "main_bank" | "competitor" | "asset_name" | "asset_purpose" | "asset_evidence_level">
> & {
  lease_credit?: ScoringFormData["lease_credit"] | string;
  contracts?: ScoringFormData["contracts"] | string;
};

export const getScreeningScore = (result?: Record<string, any> | null) =>
  Number(result?.score ?? result?.score_base ?? 0);

export function buildCurrentIssue(result: Record<string, any>, data: IssuePolicyFormData) {
  const score = getScreeningScore(result);
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
export function CurrentIssueCard({ result, data }: { result: Record<string, any>; data: IssuePolicyFormData }) {
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

export function buildRingiPolicy(result: Record<string, any>, data: IssuePolicyFormData) {
  const score = getScreeningScore(result);
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
export function RingiPolicyCard({ result, data }: { result: Record<string, any>; data: IssuePolicyFormData }) {
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
