import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

const LEDGER_RELATIVE_PATH = path.join(
  "data",
  "judgment_drills",
  "judgment_drill_1000_20260725.csv",
);

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
]);

function projectRoot() {
  let current = process.cwd();
  for (let depth = 0; depth < 8; depth += 1) {
    if (path.basename(current) === "tune_lease_55") return current;
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return path.resolve(process.cwd(), "..");
}

function ledgerPath() {
  return path.join(projectRoot(), LEDGER_RELATIVE_PATH);
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
      continue;
    }

    field += char;
  }

  if (field || row.length) {
    row.push(field);
    if (row.some((value) => value !== "")) rows.push(row);
  }

  return rows;
}

function escapeCsv(value: string) {
  if (!/[",\r\n]/.test(value)) return value;
  return `"${value.replaceAll('"', '""')}"`;
}

function stringifyCsv(headers: string[], rows: Record<string, string>[]) {
  const lines = [
    headers.map(escapeCsv).join(","),
    ...rows.map((row) => headers.map((header) => escapeCsv(row[header] ?? "")).join(",")),
  ];
  return `${lines.join("\n")}\n`;
}

async function readLedger() {
  const text = await fs.readFile(ledgerPath(), "utf-8");
  const parsed = parseCsv(text);
  const headers = parsed[0] ?? [];
  const rows = parsed.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
  return { headers, rows };
}

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

export async function GET(request: NextRequest) {
  try {
    const { rows } = await readLedger();
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

    const { headers, rows } = await readLedger();
    const index = rows.findIndex((row) => row.case_id === caseId);
    if (index < 0) {
      return NextResponse.json({ error: "case_id not found" }, { status: 404 });
    }

    const values = body.values ?? {};
    for (const [key, rawValue] of Object.entries(values)) {
      if (!EDITABLE_FIELDS.has(key)) continue;
      rows[index][key] = rawValue == null ? "" : String(rawValue).trim();
    }

    if (!rows[index].judged_at && isCompleted(rows[index])) {
      rows[index].judged_at = new Date().toISOString();
    }
    rows[index].status = isCompleted(rows[index])
      ? "judged_by_user"
      : "pending_user_judgment";

    await fs.writeFile(ledgerPath(), stringifyCsv(headers, rows), "utf-8");
    return NextResponse.json(rowResponse(rows, index));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to save judgment drill ledger" },
      { status: 500 },
    );
  }
}
