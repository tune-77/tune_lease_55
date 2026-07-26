import { NextRequest, NextResponse } from "next/server";
import { normalizeAudienceResponse, saveAudienceResponse } from "@/lib/audienceSurvey";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const response = normalizeAudienceResponse(body);
    if (!response.quiz_answer) {
      return NextResponse.json({ error: "quiz_answer is required" }, { status: 400 });
    }
    if (!response.product_reaction || !response.adoption_interest) {
      return NextResponse.json({ error: "required survey answers are missing" }, { status: 400 });
    }

    const storage = await saveAudienceResponse(response);
    return NextResponse.json({
      ok: true,
      storage,
      id: response.id,
      quiz_correct: response.quiz_correct,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "failed to save audience response" },
      { status: 500 },
    );
  }
}
