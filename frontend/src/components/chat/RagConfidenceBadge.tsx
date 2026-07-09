export type RagConfidenceLevel = "high" | "medium" | "low";

export type RagKnowledgeRef = {
  doc_id?: string;
  obsidian_ref?: string;
  file_name?: string;
  rank_score?: number | null;
  confidence?: number;
  confidence_level?: RagConfidenceLevel;
};

const LEVEL_STYLES: Record<RagConfidenceLevel, { label: string; className: string }> = {
  high: { label: "🟢 高", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  medium: { label: "🟡 中", className: "bg-amber-50 text-amber-700 border-amber-200" },
  low: { label: "🔴 低", className: "bg-rose-50 text-rose-700 border-rose-200" },
};

type Props = {
  confidence?: number;
  level?: RagConfidenceLevel;
};

export default function RagConfidenceBadge({ confidence, level }: Props) {
  if (!level || !(level in LEVEL_STYLES)) return null;
  const style = LEVEL_STYLES[level];
  const percent = typeof confidence === "number" ? `信頼度 ${Math.round(confidence * 100)}%` : "信頼度";
  return (
    <span
      title={`${percent}（関連度・出典・新鮮度から算出。内容の正しさを保証するものではありません）`}
      className={`inline-flex shrink-0 items-center rounded border px-1 text-[10px] font-bold leading-4 ${style.className}`}
    >
      {style.label}
    </span>
  );
}
