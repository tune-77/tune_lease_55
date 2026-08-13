"use client";
import { useRef, useState } from "react";
import { apiClient } from "./api";
import {
  buildExperienceCaseQuery,
  buildShionReviewFallback,
  buildShionReviewPrompt,
  buildShionReviewUserId,
  ensureCandidateJudgmentAssetMentionInReview,
  ensurePastCompanyMentionInReview,
  getScreeningScore,
  hasExperienceSearchContext,
  normalizeExperienceCase,
  uniquePastCompanyHighlights,
  type DemoSimilarPastCase,
  type JudgmentAssetCandidate,
  type PastCompanyHighlight,
  type PastShionScreeningReview,
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
  const [pastCompanies, setPastCompanies] = useState<PastCompanyHighlight[]>([]);
  const [judgmentAssetCandidates, setJudgmentAssetCandidates] = useState<JudgmentAssetCandidate[]>([]);
  const requestSeq = useRef(0);

  const fetchPastShionReviews = async (targetResult: any, targetFormData: Record<string, any>) => {
    try {
      const res = await apiClient.get("/api/shion-screening-reviews", {
        params: {
          industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
          limit: 3,
        },
      });
      return Array.isArray(res.data?.reviews) ? res.data.reviews as PastShionScreeningReview[] : [];
    } catch {
      return [];
    }
  };

  const fetchExperienceCases = async (targetResult: any, targetFormData: Record<string, any>) => {
    if (!hasExperienceSearchContext(targetFormData, targetResult)) return [] as DemoSimilarPastCase[];
    try {
      const res = await apiClient.get("/api/screening-experience-cases", {
        params: buildExperienceCaseQuery("", targetFormData, targetResult),
      });
      return Array.isArray(res.data?.cases) ? res.data.cases.map(normalizeExperienceCase) : [];
    } catch {
      return [] as DemoSimilarPastCase[];
    }
  };

  const fetchJudgmentAssetCandidates = async (targetResult: any, targetFormData: Record<string, any>) => {
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
    targetResult: any,
    targetFormData: Record<string, any>,
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

  const requestReview = async (targetResult: Record<string, any>, targetFormData: Record<string, any>) => {
    if (!targetResult) return;
    const seq = ++requestSeq.current;
    let promptText = "";
    let fallbackPastReviews: PastShionScreeningReview[] = [];
    let fallbackExperienceCases: DemoSimilarPastCase[] = [];
    let fallbackCandidates: JudgmentAssetCandidate[] = [];
    setLoading(true);
    setError("");
    try {
      const pastReviews = await fetchPastShionReviews(targetResult, targetFormData);
      const experienceCases = await fetchExperienceCases(targetResult, targetFormData);
      const candidates = await fetchJudgmentAssetCandidates(targetResult, targetFormData);
      fallbackPastReviews = pastReviews;
      fallbackExperienceCases = experienceCases;
      fallbackCandidates = candidates;
      const nextPastCompanies = uniquePastCompanyHighlights(pastReviews, experienceCases);
      setPastCompanies(nextPastCompanies);
      if (seq !== requestSeq.current) return;
      promptText = buildShionReviewPrompt(targetResult, targetFormData, pastReviews, experienceCases, candidates, "standard");
      const res = await apiClient.post("/api/chat", {
        message: promptText,
        user_id: buildShionReviewUserId(targetResult, targetFormData),
        response_mode: "shion",
        debug_memory: true,
      }, {
        timeout: 18000,
      });
      if (seq !== requestSeq.current) return;
      const memoryDebug = res.data?.memory_debug || {};
      const memoryRecall = memoryDebug.memory_recall || {};
      const identityMemory = memoryDebug.identity_memory || {};
      const vertexSearch = memoryDebug.vertex_ai_search || res.data?.vertex_ai_search || {};
      const vertexAnswer = memoryDebug.vertex_answer_api || res.data?.vertex_answer_api || {};
      const parsedGroundingScore =
        typeof vertexAnswer.grounding_score === "number"
          ? vertexAnswer.grounding_score
          : vertexAnswer.grounding_score != null
            ? Number(vertexAnswer.grounding_score)
            : null;
      const knowledgeRefs = Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0;
      const memoryRefs = Array.isArray(memoryRecall.refs) ? memoryRecall.refs.length : 0;
      const nextReview: ShionScreeningReview = {
        reply: ensureCandidateJudgmentAssetMentionInReview(
          ensurePastCompanyMentionInReview(
            String(res.data?.reply || "紫苑レビューが空でした。"),
            nextPastCompanies,
          ),
          candidates,
        ),
        memoryRefs,
        knowledgeRefs,
        identityUsed: Boolean(identityMemory.used),
        vertexUsed: Boolean(vertexSearch.used),
        vertexStatus: String(vertexSearch.status || ""),
        vertexRefs: Array.isArray(vertexSearch.refs) ? vertexSearch.refs.map(String) : [],
        vertexAnswerUsed: Boolean(vertexAnswer.used),
        vertexAnswerStatus: String(vertexAnswer.status || ""),
        groundingScore: Number.isFinite(parsedGroundingScore) ? parsedGroundingScore : null,
        groundingScoreSource: String(vertexAnswer.grounding_score_source || ""),
        lowSupportClaimCount: Number(vertexAnswer.low_support_claim_count || 0),
        supportCount: Number(vertexAnswer.support_count || 0),
      };
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
      const fallbackPastCompanies = uniquePastCompanyHighlights(fallbackPastReviews, fallbackExperienceCases);
      const fallbackReview: ShionScreeningReview = {
        reply: ensureCandidateJudgmentAssetMentionInReview(
          ensurePastCompanyMentionInReview(
            buildShionReviewFallback(
              targetResult,
              targetFormData,
              fallbackCandidates,
              fallbackExperienceCases,
              fallbackPastReviews,
            ),
            fallbackPastCompanies,
          ),
          fallbackCandidates,
        ),
        memoryRefs: fallbackPastReviews.length,
        knowledgeRefs: fallbackCandidates.length,
        identityUsed: false,
        vertexUsed: false,
        vertexStatus: "fallback",
        vertexRefs: [],
        vertexAnswerUsed: false,
        vertexAnswerStatus: "fallback",
        groundingScore: null,
        groundingScoreSource: "",
        lowSupportClaimCount: 0,
        supportCount: 0,
      };
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
    pastCompanies,
    judgmentAssetCandidates,
    requestReview,
    submitFeedback,
  };
}
