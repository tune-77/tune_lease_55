import { randomUUID } from "crypto";
import { mkdir, readFile, appendFile } from "fs/promises";
import path from "path";

export type AudienceResponse = {
  id: string;
  created_at: string;
  quiz_answer: string;
  quiz_correct: boolean;
  product_reaction: string;
  adoption_interest: string;
  poc_interest: string;
  macbook_vote: string;
  company?: string;
  contact?: string;
  insight_1001?: string;
};

export type AudienceSummary = {
  total: number;
  storage: "firestore" | "jsonl";
  quiz_correct: number;
  quiz_correct_rate: number;
  counts: {
    quiz_answer: Record<string, number>;
    product_reaction: Record<string, number>;
    adoption_interest: Record<string, number>;
    poc_interest: Record<string, number>;
    macbook_vote: Record<string, number>;
  };
  leads: number;
  trial_interest: number;
  recent_insights: Array<{
    created_at: string;
    text: string;
    company?: string;
    contact?: string;
  }>;
};

const COLLECTION = process.env.FIRESTORE_AUDIENCE_COLLECTION || "audienceResponses";
const LOCAL_FILE = path.resolve(process.cwd(), "..", "data", "audience_survey_responses.jsonl");

function cleanText(value: unknown, maxLength = 400) {
  return String(value ?? "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, maxLength);
}

export function normalizeAudienceResponse(body: Partial<AudienceResponse>): AudienceResponse {
  const quizAnswer = cleanText(body.quiz_answer, 80);
  return {
    id: cleanText(body.id, 80) || randomUUID(),
    created_at: body.created_at || new Date().toISOString(),
    quiz_answer: quizAnswer,
    quiz_correct: quizAnswer === "条件付きなら通せる",
    product_reaction: cleanText(body.product_reaction, 80),
    adoption_interest: cleanText(body.adoption_interest, 80),
    poc_interest: cleanText(body.poc_interest, 80),
    macbook_vote: cleanText(body.macbook_vote, 80),
    company: cleanText(body.company, 120),
    contact: cleanText(body.contact, 160),
    insight_1001: cleanText(body.insight_1001, 800),
  };
}

function projectId() {
  return (
    process.env.FIRESTORE_PROJECT_ID ||
    process.env.GOOGLE_CLOUD_PROJECT ||
    process.env.GCLOUD_PROJECT ||
    ""
  );
}

async function firestoreToken() {
  if (process.env.FIRESTORE_BEARER_TOKEN) return process.env.FIRESTORE_BEARER_TOKEN;

  const res = await fetch(
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
    { headers: { "Metadata-Flavor": "Google" } },
  );
  if (!res.ok) throw new Error(`metadata token error: ${res.status}`);
  const data = (await res.json()) as { access_token?: string };
  if (!data.access_token) throw new Error("metadata token missing access_token");
  return data.access_token;
}

function stringValue(value: string | undefined) {
  return { stringValue: value || "" };
}

function firestoreFields(response: AudienceResponse) {
  return {
    id: stringValue(response.id),
    created_at: { timestampValue: response.created_at },
    quiz_answer: stringValue(response.quiz_answer),
    quiz_correct: { booleanValue: response.quiz_correct },
    product_reaction: stringValue(response.product_reaction),
    adoption_interest: stringValue(response.adoption_interest),
    poc_interest: stringValue(response.poc_interest),
    macbook_vote: stringValue(response.macbook_vote),
    company: stringValue(response.company),
    contact: stringValue(response.contact),
    insight_1001: stringValue(response.insight_1001),
  };
}

function readFirestoreString(field: unknown) {
  const value = field as { stringValue?: string; timestampValue?: string } | undefined;
  return value?.stringValue ?? value?.timestampValue ?? "";
}

function readFirestoreBoolean(field: unknown) {
  const value = field as { booleanValue?: boolean } | undefined;
  return Boolean(value?.booleanValue);
}

function fromFirestoreDocument(doc: unknown): AudienceResponse | null {
  const fields = (doc as { fields?: Record<string, unknown> })?.fields;
  if (!fields) return null;
  return {
    id: readFirestoreString(fields.id),
    created_at: readFirestoreString(fields.created_at),
    quiz_answer: readFirestoreString(fields.quiz_answer),
    quiz_correct: readFirestoreBoolean(fields.quiz_correct),
    product_reaction: readFirestoreString(fields.product_reaction),
    adoption_interest: readFirestoreString(fields.adoption_interest),
    poc_interest: readFirestoreString(fields.poc_interest),
    macbook_vote: readFirestoreString(fields.macbook_vote),
    company: readFirestoreString(fields.company),
    contact: readFirestoreString(fields.contact),
    insight_1001: readFirestoreString(fields.insight_1001),
  };
}

async function saveToJsonl(response: AudienceResponse) {
  await mkdir(path.dirname(LOCAL_FILE), { recursive: true });
  await appendFile(LOCAL_FILE, `${JSON.stringify(response)}\n`, "utf8");
  return "jsonl" as const;
}

async function readFromJsonl() {
  try {
    const raw = await readFile(LOCAL_FILE, "utf8");
    return raw
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as AudienceResponse);
  } catch {
    return [];
  }
}

async function saveToFirestore(response: AudienceResponse) {
  const pid = projectId();
  if (!pid) throw new Error("Firestore project id is not configured");

  const token = await firestoreToken();
  const url = `https://firestore.googleapis.com/v1/projects/${encodeURIComponent(
    pid,
  )}/databases/(default)/documents:commit`;
  const name = `projects/${pid}/databases/(default)/documents/${COLLECTION}/${response.id}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      writes: [{ update: { name, fields: firestoreFields(response) } }],
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Firestore write error: ${res.status} ${text}`);
  }
  return "firestore" as const;
}

async function readFromFirestore() {
  const pid = projectId();
  if (!pid) throw new Error("Firestore project id is not configured");

  const token = await firestoreToken();
  const url = `https://firestore.googleapis.com/v1/projects/${encodeURIComponent(
    pid,
  )}/databases/(default)/documents/${COLLECTION}?pageSize=1000&orderBy=created_at%20desc`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Firestore read error: ${res.status} ${text}`);
  }
  const data = (await res.json()) as { documents?: unknown[] };
  return (data.documents || [])
    .map(fromFirestoreDocument)
    .filter((row): row is AudienceResponse => Boolean(row));
}

export async function saveAudienceResponse(response: AudienceResponse) {
  if (projectId()) {
    try {
      return await saveToFirestore(response);
    } catch (error) {
      if (process.env.FIRESTORE_STRICT === "1") throw error;
    }
  }
  return saveToJsonl(response);
}

export async function readAudienceResponses(): Promise<{
  storage: "firestore" | "jsonl";
  responses: AudienceResponse[];
}> {
  if (projectId()) {
    try {
      return { storage: "firestore", responses: await readFromFirestore() };
    } catch (error) {
      if (process.env.FIRESTORE_STRICT === "1") throw error;
    }
  }
  return { storage: "jsonl", responses: await readFromJsonl() };
}

function increment(map: Record<string, number>, key: string) {
  if (!key) return;
  map[key] = (map[key] || 0) + 1;
}

export function summarizeAudienceResponses(
  responses: AudienceResponse[],
  storage: AudienceSummary["storage"],
): AudienceSummary {
  const counts = {
    quiz_answer: {} as Record<string, number>,
    product_reaction: {} as Record<string, number>,
    adoption_interest: {} as Record<string, number>,
    poc_interest: {} as Record<string, number>,
    macbook_vote: {} as Record<string, number>,
  };
  for (const response of responses) {
    increment(counts.quiz_answer, response.quiz_answer);
    increment(counts.product_reaction, response.product_reaction);
    increment(counts.adoption_interest, response.adoption_interest);
    increment(counts.poc_interest, response.poc_interest);
    increment(counts.macbook_vote, response.macbook_vote);
  }

  const quizCorrect = responses.filter((row) => row.quiz_correct).length;
  const leads = responses.filter((row) => row.contact || row.company).length;
  const trialInterest = responses.filter((row) =>
    ["今すぐ使いたい", "検証してみたい", "自社データで試したい"].includes(
      row.adoption_interest,
    ),
  ).length;

  return {
    total: responses.length,
    storage,
    quiz_correct: quizCorrect,
    quiz_correct_rate: responses.length ? Math.round((quizCorrect / responses.length) * 100) : 0,
    counts,
    leads,
    trial_interest: trialInterest,
    recent_insights: responses
      .filter((row) => row.insight_1001)
      .slice(0, 12)
      .map((row) => ({
        created_at: row.created_at,
        text: row.insight_1001 || "",
        company: row.company,
        contact: row.contact,
      })),
  };
}
