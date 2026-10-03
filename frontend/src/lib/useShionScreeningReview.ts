"use client";
import { useRef, useState } from "react";
import { apiClient } from "./api";
import {
  buildShionReviewChatBody,
  buildShionReviewFallbackRecord,
  buildShionReviewPrompt,
  getScreeningScore,
  SHION_REVIEW_HARD_TIMEOUT_MS,
  SHION_REVIEW_SOFT_TIMEOUT_MS,
  shionReviewFromChatResponse,
  waitForSoftTimeout,
  type JudgmentAssetCandidate,
  type ScreeningFormRecord,
  type ScreeningResultRecord,
  type ShionReviewFeedbackSample,
  type ShionReviewFeedback,
  type ShionScreeningReview,
} from "./shionReview";

// lease-kun（スマホウィザード）向けの紫苑レビュー。
// screening/page.tsx の requestShionReview と同じ生成ロジック（lib/shionReview.ts）を使うが、
// デモ案件キャッシュ・判断資産候補の手動編集UIなど screening 固有の状態は持たない簡易版。
export function useShionScreeningReview() {
  const [review, setReview] = useState<ShionScreeningReview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const [judgmentAssetCandidates, setJudgmentAssetCandidates] = useState<JudgmentAssetCandidate[]>([]);
  const requestSeq = useRef(0);

  // 直近レビューへの人間評価だけを読む。案件内容（社名・スコア・本文）は紫苑へ渡さない。
  const fetchRecentReviewFeedbacks = async () => {
    try {
      const res = await apiClient.get("/api/shion-screening-reviews", { params: { limit: 8 } });
      const reviews = Array.isArray(res.data?.reviews) ? res.data.reviews as ShionReviewFeedbackSample[] : [];
      return reviews.map((item) => item.user_feedback);
    } catch {
      return [];
    }
  };

  const fetchJudgmentAssetCandidates = async (targetResult: ScreeningResultRecord, targetFormData: ScreeningFormRecord) => {
    try {
      const res = await apiClient.get("/api/judgment-asset-candidates/screening", {
        params: {
          industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
          industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
          asset_name: targetFormData.asset_name || "",
          asset_purpose: targetFormData.asset_purpose || "",
          hantei: targetResult?.hantei || "",
          score: getScreeningScore(targetResult),
          limit: 3,
        },
      });
      const candidates = Array.isArray(res.data?.candidates) ? res.data.candidates as JudgmentAssetCandidate[] : [];
      setJudgmentAssetCandidates(candidates);
      return candidates;
    } catch (error) {
      console.warn("Judgment asset candidates fetch failed", error);
      setJudgmentAssetCandidates([]);
      return [] as JudgmentAssetCandidate[];
    }
  };

  const saveReview = async (
    targetResult: ScreeningResultRecord,
    targetFormData: ScreeningFormRecord,
    promptText: string,
    nextReview: ShionScreeningReview,
  ) => {
    const res = await apiClient.post("/api/shion-screening-reviews", {
      case_id: targetResult?.case_id || targetFormData.company_no || "",
      company_name: targetFormData.company_name || "",
      industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
      industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
      sales_dept: targetFormData.sales_dept || "",
      score: getScreeningScore(targetResult),
      hantei: targetResult?.hantei || "",
      q_risk: targetResult?.quantum_risk ?? null,
      umap_anomaly_score: targetResult?.umap_anomaly_score ?? null,
      memory_refs: nextReview.memoryRefs,
      knowledge_refs: nextReview.knowledgeRefs,
      identity_used: nextReview.identityUsed,
      review_text: nextReview.reply,
      prompt_text: promptText,
      form_snapshot: targetFormData,
      result_snapshot: targetResult,
    });
    return Number(res.data?.review?.id || 0) || undefined;
  };

  const requestReview = async (targetResult: ScreeningResultRecord, targetFormData: ScreeningFormRecord) => {
    if (!targetResult) return;
    const seq = ++requestSeq.current;
    let promptText = "";
    let fallbackCandidates: JudgmentAssetCandidate[] = [];
    setLoading(true);
    setError("");
    try {
      const recentFeedbacks = await fetchRecentReviewFeedbacks();
      const candidates = await fetchJudgmentAssetCandidates(targetResult, targetFormData);
      fallbackCandidates = candidates;
      if (seq !== requestSeq.current) return;
      promptText = buildShionReviewPrompt(targetResult, targetFormData, candidates, "standard", recentFeedbacks);
      const chatRequest = apiClient.post("/api/chat", buildShionReviewChatBody(targetResult, targetFormData, promptText), {
        timeout: SHION_REVIEW_HARD_TIMEOUT_MS,
      });
      const early = await waitForSoftTimeout(chatRequest, SHION_REVIEW_SOFT_TIMEOUT_MS);
      if (seq !== requestSeq.current) return;
      if (!early.done) {
        // 打ち切り時間を過ぎたら簡易生成を先に見せ、紫苑の本文が届いたら差し替える（簡易生成はまだ保存しない）
        setReview(buildShionReviewFallbackRecord(targetResult, targetFormData, candidates, true));
        setLoading(false);
      }
      const res = early.done ? early.value : await chatRequest;
      if (seq !== requestSeq.current) return;
      const nextReview = shionReviewFromChatResponse(res.data, candidates);
      setReview(nextReview);
      saveReview(targetResult, targetFormData, promptText, nextReview)
        .then((savedId) => {
          if (!savedId || seq !== requestSeq.current) return;
          setReview((current) => current ? { ...current, savedId } : current);
        })
        .catch((error) => {
          console.warn("Shion screening review save failed", error);
        });
    } catch (error) {
      if (seq !== requestSeq.current) return;
      console.error("Shion review error", error);
      const fallbackReview = buildShionReviewFallbackRecord(targetResult, targetFormData, fallbackCandidates);
      setReview(fallbackReview);
      setError("");
      saveReview(
        targetResult,
        targetFormData,
        promptText || "fallback: local screening review from case fields and judgment assets",
        fallbackReview,
      )
        .then((savedId) => {
          if (!savedId || seq !== requestSeq.current) return;
          setReview((current) => current ? { ...current, savedId } : current);
        })
        .catch((saveError) => {
          console.warn("Fallback Shion screening review save failed", saveError);
        });
    } finally {
      if (seq === requestSeq.current) {
        setLoading(false);
      }
    }
  };

  const submitFeedback = async (feedback: ShionReviewFeedback) => {
    if (!review?.savedId || feedbackSaving) return false;
    const previous = review.userFeedback;
    setReview((current) => current ? { ...current, userFeedback: feedback } : current);
    setFeedbackSaving(true);
    try {
      await apiClient.patch(`/api/shion-screening-reviews/${review.savedId}/feedback`, {
        user_feedback: feedback,
      });
      return true;
    } catch (error) {
      console.error("Shion review feedback save failed", error);
      setReview((current) => current ? { ...current, userFeedback: previous } : current);
      setError("紫苑レビュー評価を保存できませんでした。");
      return false;
    } finally {
      setFeedbackSaving(false);
    }
  };

  return {
    review,
    loading,
    error,
    feedbackSaving,
    judgmentAssetCandidates,
    requestReview,
    submitFeedback,
  };
}
