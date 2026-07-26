import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

const EDITABLE_FIELDS = new Set([
  "credit_score_20",
  "repayment_source_score_20",
  "asset_exit_score_20",
  "plan_specificity_score_20",
  "uncertainty_control_score_20",
  "total_score_100",
  "user_decision",
  "heaviest_issue",
  "additional_checks",
  "ringi_sentence",
  "score_decision_gap_note",
  "ai_feedback_outcome",
  "ai_feedback_note",
  "okf_candidate_tags",
  "presentation_use_ok",
  "reviewer",
  "judged_at",
  "status",
]);

function isCompleted(row: Record<string, string>) {
  return Boolean(row.total_score_100 && row.user_decision && row.ringi_sentence);
}

function rowResponse(rows: Record<string, string>[], index: number) {
  const boundedIndex = Math.max(0, Math.min(index, rows.length - 1));
  const completed = rows.filter(isCompleted).length;
  const nextPendingIndex = rows.findIndex((row) => !isCompleted(row));
  return {
    item: rows[boundedIndex] ?? null,
    index: boundedIndex,
    total: rows.length,
    completed,
    pending: rows.length - completed,
    nextPendingIndex,
  };
}

async function fetchAllCases(): Promise<Record<string, string>[]> {
  const res = await fetch(`${FASTAPI_URL}/api/judgment-drill/cases?limit=1100&offset=0`);
  if (!res.ok) throw new Error(`FastAPI error: ${res.status}`);
  const data = (await res.json()) as { cases: Record<string, string>[] };
  return data.cases;
}

export async function GET(request: NextRequest) {
  try {
    const rows = await fetchAllCases();
    const search = request.nextUrl.searchParams;
    const caseId = search.get("case_id") || "";
    const indexParam = Number(search.get("index") || "0");
    const caseIndex = caseId ? rows.findIndex((row) => row.case_id === caseId) : -1;
    const index = caseIndex >= 0 ? caseIndex : Number.isFinite(indexParam) ? indexParam : 0;

    return NextResponse.json(rowResponse(rows, index));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to read judgment drill ledger" },
      { status: 500 },
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      case_id?: string;
      values?: Record<string, string | number | null | undefined>;
    };
    const caseId = String(body.case_id || "").trim();
    if (!caseId) {
      return NextResponse.json({ error: "case_id is required" }, { status: 400 });
    }

    // 現在の行を取得して派生フィールドを計算
    const currentRes = await fetch(
      `${FASTAPI_URL}/api/judgment-drill/cases/${encodeURIComponent(caseId)}`,
    );
    if (!currentRes.ok) {
      return NextResponse.json({ error: "case_id not found" }, { status: 404 });
    }
    const currentRow = (await currentRes.json()) as Record<string, string>;

    // 入力値をマージ
    const merged = { ...currentRow };
    const values = body.values ?? {};
    for (const [key, rawValue] of Object.entries(values)) {
      if (!EDITABLE_FIELDS.has(key)) continue;
      merged[key] = rawValue == null ? "" : String(rawValue).trim();
    }

    // judged_at・status を自動計算
    if (!merged.judged_at && isCompleted(merged)) {
      merged.judged_at = new Date().toISOString();
    }
    merged.status = isCompleted(merged) ? "judged_by_user" : "pending_user_judgment";

    // FastAPI PATCH に送る fields を組み立て
    const fields: Record<string, string> = {};
    for (const key of EDITABLE_FIELDS) {
      if (key in merged) fields[key] = merged[key];
    }

    const patchRes = await fetch(
      `${FASTAPI_URL}/api/judgment-drill/cases/${encodeURIComponent(caseId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields }),
      },
    );
    if (!patchRes.ok) {
      const err = await patchRes.text();
      throw new Error(`FastAPI PATCH error: ${patchRes.status} ${err}`);
    }

    // 全件取得して response を組み立て
    const rows = await fetchAllCases();
    const index = rows.findIndex((row) => row.case_id === caseId);

    return NextResponse.json(rowResponse(rows, Math.max(0, index)));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to save judgment drill ledger" },
      { status: 500 },
    );
  }
}
