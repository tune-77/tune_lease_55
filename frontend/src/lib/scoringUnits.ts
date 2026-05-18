export const monetaryScoringFields = [
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
  "acquisition_cost",
] as const;

type MonetaryScoringField = (typeof monetaryScoringFields)[number];

export function toThousandYenPayload<T extends object>(data: T): T {
  const payload: Record<string, unknown> = { ...data } as Record<string, unknown>;

  for (const field of monetaryScoringFields) {
    if (!(field in payload)) continue;
    const numericValue = Number(payload[field as MonetaryScoringField] || 0);
    payload[field] = numericValue * 1000;
  }

  return payload as T;
}
