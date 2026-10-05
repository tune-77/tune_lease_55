// REV-470: 現場メモ（passion_text）から経営者の定性シグナルを推定する API の呼び出し。
// 結果は参考表示と紫苑の審査コメント用だけで、スコア・判定には使わない。
import { apiClient } from "@/lib/api";

export type InterviewSignalEvidence = { quote: string; cue: string };

export type InterviewSignal = {
  key: string;
  label: string;
  level: "弱" | "中" | "強";
  score: number;
  evidence: InterviewSignalEvidence[];
};

export type InterviewSignalsResult = {
  enabled: boolean;
  signals: InterviewSignal[];
  sentence_count?: number;
  excluded_sentence_count?: number;
  disclaimer?: string;
  prompt_block?: string;
};

export const fetchInterviewSignals = async (text: string): Promise<InterviewSignalsResult | null> => {
  if (!text.trim()) return null;
  try {
    const { data } = await apiClient.post<InterviewSignalsResult>("/api/screening/interview-signals", { text });
    return data;
  } catch (error) {
    console.warn("interview signals fetch failed", error);
    return null;
  }
};
