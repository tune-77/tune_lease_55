/**
 * ui_labels.json から UI テキストを返すヘルパー。
 *
 * api/rule_engine の ui_text アプライヤーが ui_labels.json を書き換えると
 * Next.js dev サーバーの HMR が検出してコンポーネントを再レンダリングする。
 *
 * 使い方: getLabel("RISK_APPROVAL_LABEL")
 *         getLabel("RISK_APPROVAL_LABEL", "条件付き承認")  ← フォールバック指定
 */
import labelsData from "./ui_labels.json";

type LabelEntry = { key: string; value: string };

const _labelMap: Record<string, string> = Object.fromEntries(
  (labelsData as LabelEntry[]).map((e) => [e.key, e.value])
);

export function getLabel(key: string, fallback?: string): string {
  return _labelMap[key] ?? fallback ?? key;
}
