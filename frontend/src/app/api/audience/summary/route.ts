import { NextResponse } from "next/server";
import { readAudienceResponses, summarizeAudienceResponses } from "@/lib/audienceSurvey";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { storage, responses } = await readAudienceResponses();
    return NextResponse.json(summarizeAudienceResponses(responses, storage));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to read audience summary" },
      { status: 500 },
    );
  }
}
