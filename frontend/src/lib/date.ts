// 日次表示キー用のローカル日付 (YYYY-MM-DD)。
// toISOString() は UTC 基準のため、日本時間 0:00〜8:59 は前日扱いになりズレる。
export const formatLocalDateKey = (date: Date = new Date()): string => {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
};
