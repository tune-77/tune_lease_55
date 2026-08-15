// 紫苑レビュー（審査分析画面の AI レビュー）を screening（PC）と lease-kun（スマホ）の
// 両方から同一ロジックで呼べるようにするための、状態を持たない純粋関数・型の置き場。
// プロンプト文面や過去事例の扱いがここから外れると、画面間で紫苑の挙動が食い違うため、
// 生成ロジックは screening/page.tsx に重複定義せず必ずここを参照すること。

export type ShionReviewFeedback = "useful" | "needs_fix" | "wrong" | "specific" | "thin" | "discomfort_hit" | "over_inferred";
export type JudgmentAssetCandidateFeedback = "useful" | "neutral" | "rejected";
export type JudgmentAssetAdaptationMode = "conservative" | "standard" | "exploratory" | "aggressive";

export type ShionScreeningReview = {
  reply: string;
  memoryRefs: number;
  knowledgeRefs: number;
  identityUsed: boolean;
  vertexUsed?: boolean;
  vertexStatus?: string;
  vertexRefs?: string[];
  vertexAnswerUsed?: boolean;
  vertexAnswerStatus?: string;
  groundingScore?: number | null;
  groundingScoreSource?: string;
  lowSupportClaimCount?: number;
  supportCount?: number;
  savedId?: number;
  userFeedback?: ShionReviewFeedback;
};

export type JudgmentAssetCandidate = {
  id: string;
  candidate_type: string;
  research_topic: string;
  claim: string;
  effective_claim?: string;
  edited_claim?: string;
  edit_count?: number;
  evidence_path: string;
  promotion_status: string;
  source?: string;
  use_count: number;
  useful_count: number;
  rejected_count: number;
  verified_status: string;
  userFeedback?: JudgmentAssetCandidateFeedback;
};

export type PastCompanyHighlight = {
  name: string;
  label: "類似案件" | "過去レビュー" | "反面教師";
  experienceCase?: DemoSimilarPastCase;
  pastReview?: PastShionScreeningReview;
};

export type PastShionScreeningReview = {
  company_name?: string;
  industry_sub?: string;
  score?: number;
  hantei?: string;
  review_text?: string;
  created_at?: string;
  user_feedback?: ShionReviewFeedback | "";
};

export type DemoSimilarPastCase = {
  id?: number;
  demoCaseId?: string;
  sourceCaseId?: string;
  companyName: string;
  period: string;
  industry: string;
  industryMajor?: string;
  industrySub?: string;
  salesDept?: string;
  score: number;
  decision: string;
  outcome: string;
  similarity: string;
  actionTaken: string;
  lesson: string;
  difference: string;
  source?: string;
  similarityScore?: number;
  similarityReasons?: string[];
  formSnapshot?: Record<string, any>;
  resultSnapshot?: Record<string, any>;
};

export type ShionThoughtStep = {
  title: string;
  items: string[];
};

export const SHION_REVIEW_IMAGE = "/lease-intelligence/moods/focus.webp";

export const FEEDBACK_LABELS: Record<ShionReviewFeedback, string> = {
  useful: "使えた",
  needs_fix: "修正して使う",
  wrong: "違った",
  specific: "具体的だった",
  thin: "薄い",
  discomfort_hit: "違和感が当たった",
  over_inferred: "推測が強すぎた",
};

export const isCanonicalJudgmentAsset = (candidate: JudgmentAssetCandidate) => (
  candidate.source === "canonical_judgment_rules" ||
  candidate.promotion_status === "active" ||
  candidate.verified_status === "canonical" ||
  candidate.id.startsWith("cr-")
);

export const getScreeningScore = (result?: Record<string, any> | null) =>
  Number(result?.score ?? result?.score_base ?? 0);

export const normalizeReviewText = (text: string) =>
  (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .trim();

export const validPastCompanyName = (name: string) =>
  name.length >= 2 && name !== "名称不明" && name !== "名称未設定";

export const uniquePastCompanyHighlights = (
  reviews: PastShionScreeningReview[],
  experienceCases: DemoSimilarPastCase[],
): PastCompanyHighlight[] => {
  const byName = new Map<string, PastCompanyHighlight>();
  experienceCases.forEach((item) => {
    const name = String(item.companyName || "").trim();
    if (validPastCompanyName(name) && !byName.has(name)) {
      byName.set(name, { name, label: "類似案件", experienceCase: item });
    }
  });
  reviews.forEach((review) => {
    const name = String(review.company_name || "").trim();
    if (!validPastCompanyName(name)) return;
    const label: PastCompanyHighlight["label"] = ["wrong", "thin", "over_inferred"].includes(String(review.user_feedback)) ? "反面教師" : "過去レビュー";
    byName.set(name, { name, label, pastReview: review, experienceCase: byName.get(name)?.experienceCase });
  });
  return Array.from(byName.values());
};

export const judgmentAssetHighlightTerms = (candidates: JudgmentAssetCandidate[]) => {
  const terms = candidates
    .flatMap((candidate) => {
      const assetText = candidate.edited_claim || candidate.effective_claim || candidate.claim || "";
      return [
        assetText.trim(),
        candidate.claim?.trim() || "",
      ].filter((term) => term.length >= 12).map((term) => ({
        term,
        candidate,
        canonical: isCanonicalJudgmentAsset(candidate),
      }));
    })
    .sort((a, b) => b.term.length - a.term.length);
  const byTerm = new Map<string, (typeof terms)[number]>();
  for (const item of terms) {
    const existing = byTerm.get(item.term);
    if (!existing || (existing.canonical && !item.canonical)) {
      byTerm.set(item.term, item);
    }
  }
  return Array.from(byTerm.values());
};

export const ensurePastCompanyMentionInReview = (
  reviewText: string,
  pastCompanies: PastCompanyHighlight[],
) => {
  const company = pastCompanies.find((item) => validPastCompanyName(item.name));
  if (!company || reviewText.includes(company.name)) return reviewText;
  const detail = company.experienceCase;
  const comparison = detail
    ? `${company.name}（${detail.decision} / ${detail.outcome}）では「${detail.lesson}」が残っており、今回も似ている点より導入目的・返済原資・今回固有の差分を確認します。`
    : `${company.name}の過去レビューを参照し、今回案件との共通点と差分を確認します。`;
  const insertion = `過去取引事例: ${comparison}`;
  if (/2\.\s*数字だけでは見落としそうな違和感/.test(reviewText)) {
    return reviewText.replace(
      /(2\.\s*数字だけでは見落としそうな違和感[^\n]*\n)/,
      `$1${insertion}\n`,
    );
  }
  return `${reviewText.trim()}\n\n${insertion}`;
};

export const ensureCandidateJudgmentAssetMentionInReview = (
  reviewText: string,
  candidates: JudgmentAssetCandidate[],
) => {
  const candidate = candidates.find((item) => !isCanonicalJudgmentAsset(item));
  const claim = (candidate?.edited_claim || candidate?.effective_claim || candidate?.claim || "").trim();
  if (!candidate || !claim || reviewText.includes(claim)) return reviewText;
  const insertion = `昇格候補: ${claim}`;
  if (/3\.\s*条件付き承認にするなら必要な確認/.test(reviewText)) {
    return reviewText.replace(
      /(3\.\s*条件付き承認にするなら必要な確認[^\n]*\n)/,
      `$1${insertion}\n`,
    );
  }
  return `${reviewText.trim()}\n\n${insertion}`;
};

export const buildDemoSimilarPastCaseBlock = (cases: DemoSimilarPastCase[]) => {
  if (!cases.length) return "";
  return [
    "【保存済み経験ケース】",
    "次の事例は、DBに保存された類似経験ケースです。今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください。",
    ...cases.slice(0, 3).map((item, index) => [
      `事例${index + 1}: ${item.companyName} / ${item.period} / ${item.industry}`,
      item.similarityScore ? `類似度: ${Math.round(item.similarityScore)} / 理由: ${(item.similarityReasons || []).join("・") || "未計算"}` : "",
      `スコア・判断: ${item.score.toFixed(1)}点 / ${item.decision} / ${item.outcome}`,
      `似ている点: ${item.similarity}`,
      `当時の対応: ${item.actionTaken}`,
      `得た教訓: ${item.lesson}`,
      `今回との差分: ${item.difference}`,
    ].filter(Boolean).join("\n")),
  ].join("\n");
};

export const buildPastReviewBlock = (reviews: PastShionScreeningReview[]) => {
  if (!reviews.length) return "";
  const lines = reviews.slice(0, 3).map((review, index) => {
    const preview = normalizeReviewText(review.review_text || "").slice(0, 260);
    const feedbackLabel = review.user_feedback
      ? `人間評価: ${FEEDBACK_LABELS[review.user_feedback] || review.user_feedback}`
      : "人間評価: 未評価";
    return [
      `過去${index + 1}: ${review.company_name || "名称不明"} / ${review.industry_sub || "業種不明"} / ${review.score != null ? Number(review.score).toFixed(1) + "点" : "点数不明"} / ${review.hantei || "判定不明"}`,
      feedbackLabel,
      `紫苑の過去レビュー: ${preview || "本文なし"}`,
    ].join("\n");
  });
  return [
    "【過去の紫苑審査レビュー記憶】",
    "次の過去レビューは、今回の判断に似た経験として参照してください。人間評価が「使えた」は重めに、「違った」は反面教師として扱い、丸写しではなく今回との差分を見てください。",
    ...lines,
  ].join("\n");
};

export const buildVertexSearchHint = (result: Record<string, any>, data: Record<string, any>) => {
  const terms = [
    result.industry_sub || data.industry_sub || result.industry_major || data.industry_major,
    data.asset_name,
    data.asset_purpose,
    data.contract_type,
    data.customer_type,
    data.main_bank,
    data.deal_source,
  ]
    .map((value) => String(value || "").trim())
    .filter(Boolean);
  const memo = [data.passion_text, data.industry_detail, data.asset_detail].join(" ");
  if (/補助金|助成金|ものづくり|省力化/.test(memo)) terms.push("補助金", "リース料軽減", "公募要領", "対象経費");
  if (/再リース|延長|満了/.test(memo)) terms.push("再リース", "残価", "耐用年数", "中古流動性");
  if (/工作機械|機械|設備/.test(`${data.asset_name || ""} ${data.asset_detail || ""}`)) terms.push("工作機械", "設備稼働率", "保守", "更新投資");
  if (Number(result.quantum_risk) >= 35) terms.push("Q_risk", "違和感", "確認論点");
  return Array.from(new Set(terms)).slice(0, 14).join(" ");
};

export const buildShionReviewPrompt = (
  result: Record<string, any>,
  data: Record<string, any>,
  pastReviews: PastShionScreeningReview[] = [],
  experienceCases: DemoSimilarPastCase[] = [],
  judgmentAssetCandidates: JudgmentAssetCandidate[] = [],
  judgmentAssetAdaptationMode: JudgmentAssetAdaptationMode = "standard",
) => {
  const score = getScreeningScore(result);
  const baseScore = Number(result.score_base);
  const vertexSearchHint = buildVertexSearchHint(result, data);
  const lines = [
    "【審査分析画面からの紫苑レビュー依頼】",
    "この案件を、審査担当者の横にいる紫苑としてレビューしてください。",
    "",
    "【Vertex補助検索ヒント】",
    vertexSearchHint || "リース審査 判断資産 物件リスク 返済余力 承認条件",
    "",
    "出力は短く、次の4項目でお願いします。",
    "1. 紫苑の第一印象",
    "2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較）",
    "3. 条件付き承認にするなら必要な確認",
    "4. 稟議で残すべき一文",
    "",
    "専門家としての深掘りルール:",
    "・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください。",
    "・Q2では、提示された数字・Q_risk・定性項目・現場メモ・過去事例のうち、何が違和感の根拠になったかを具体的に結びつけてください。",
    "・過去事例を使う場合は、社名だけを飾りにせず、似ている点、違う点、今回の確認事項への変換を短く書いてください。",
    "・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください。",
    "・不確実な推測で採否を誘導しないでください。違和感は減点ではなく、人間が確認するための論点です。",
    "",
    "複数見立てサンプリング:",
    "・回答を書く前に、内部で5つの見立て候補を作り、それぞれに候補重みを置いてください。",
    "・候補重みはPDや信用スコアではなく、今回情報から見た検討優先度です。合計100%として扱ってください。",
    "・典型的で無難な見立てだけに寄せず、低確率でも当たると重要な見立てを1つ残してください。",
    "・最終出力では候補一覧を長く出さず、採用した上位見立て、低確率高影響の確認点、稟議に残す一文へ圧縮してください。",
    "",
    "前提:",
    `・企業名: ${data.company_name || "未入力"}`,
    `・業種: ${result.industry_sub || data.industry_sub || result.industry_major || data.industry_major || "未入力"}`,
    `・営業部: ${data.sales_dept || "未入力"}`,
    `・判定: ${result.hantei || "未判定"}`,
    `・総合スコア: ${Number.isFinite(score) ? score.toFixed(1) : "未算出"}`,
    ...(Number.isFinite(baseScore) && Math.abs(baseScore - score) >= 0.1
      ? [`・補正前スコア: ${baseScore.toFixed(1)}（表示・判断は総合スコアを優先）`]
      : []),
    `・借手スコア: ${result.score_borrower != null ? Number(result.score_borrower).toFixed(1) : "未算出"}`,
    `・Q_risk: ${result.quantum_risk != null ? `${Number(result.quantum_risk).toFixed(1)}（0-100スケール、35以上で要注意・60以上で強警戒）` : "未算出"}`,
    `・UMAP異常度: ${result.umap_anomaly_score != null ? Number(result.umap_anomaly_score).toFixed(1) : "未算出"}`,
    `・マハラノビス: ${result.mahalanobis_score != null ? Number(result.mahalanobis_score).toFixed(1) : "未算出"}`,
    `・物件: ${data.asset_name || "未入力"}`,
    `・取得価額: ${data.acquisition_cost || 0}百万円`,
    `・リース期間: ${data.lease_term || 0}`,
    `・導入目的: ${data.asset_purpose || "未入力"}`,
    `・営業メモ: ${data.passion_text || "未入力"}`,
    `・直感スコア: ${data.intuition || "未入力"}`,
  ];
  const flags = result.aurion_core?.discipline_flags;
  if (Array.isArray(flags) && flags.length) {
    const flagTitles = flags
      .slice(0, 5)
      .map((f) => (typeof f === "string" ? f : (f as { title?: string })?.title ?? ""))
      .filter(Boolean);
    if (flagTitles.length) {
      lines.push(`・AURION警戒: ${flagTitles.join(" / ")}`);
    }
  }
  if (Array.isArray(result.default_warnings) && result.default_warnings.length) {
    lines.push(`・高リスク財務パターン警告: ${result.default_warnings.slice(0, 3).join(" / ")}`);
  }
  if (Array.isArray(result.diagnostic_recommendations) && result.diagnostic_recommendations.length) {
    lines.push("・補助診断の扱い: UMAP/Mahalanobisは常時使用ではなく、必要時に人間が実行する補助診断。自動減点ではなく確認論点・稟議補足に使う。");
    for (const rec of result.diagnostic_recommendations.slice(0, 3)) {
      const label = String(rec?.label || rec?.diagnostic || "補助診断");
      const status = rec?.status === "calculated" ? "算出済み" : "推奨";
      const reason = String(rec?.reason || "");
      lines.push(`  - ${label}: ${status}${reason ? `（理由: ${reason}）` : ""}`);
    }
  }
  const demoPastCaseBlock = buildDemoSimilarPastCaseBlock(experienceCases);
  if (demoPastCaseBlock) {
    lines.push("", demoPastCaseBlock);
  }
  const pastReviewBlock = buildPastReviewBlock(pastReviews);
  if (pastReviewBlock) {
    lines.push("", pastReviewBlock);
  }
  if (judgmentAssetCandidates.length) {
    const hasCanonicalAssets = judgmentAssetCandidates.some((item) => (
      item.source === "canonical_judgment_rules" || item.promotion_status === "active" || item.verified_status === "canonical"
    ));
    const adaptationPolicies: Record<JudgmentAssetAdaptationMode, string> = {
      conservative: "発展度: 保守的。教えた判断を大きく変形せず、今回案件に明確に合う範囲だけで使ってください。新しい仮説は最小限にしてください。",
      standard: "発展度: 標準。教えた判断を今回案件に合わせて少し変形し、確認観点・承認条件・反証へ落としてください。",
      exploratory: "発展度: 探索的。教えた判断から関連する新しい確認観点や承認条件を1つまで提案してよいです。ただし判断仮説として扱ってください。",
      aggressive: "発展度: 攻め。教えた判断を起点に、人間がまだ明示していない派生仮説も提案してよいです。ただし必ず『判断仮説』として明記し、断定しないでください。",
    };
    lines.push(
      "",
      hasCanonicalAssets ? "【今回使う判断資産】" : "【今回試す判断資産候補】",
      hasCanonicalAssets
        ? "次の判断資産は、過去の会話・評価・結果から代表ルール化されたものです。丸写しせず、今回の業種・物件・導入目的・財務状態に合わせて応用生成してください。"
        : "次の候補はまだ昇格済みではありません。丸写しせず、今回の業種・物件・導入目的・財務状態に合わせて応用生成してください。",
      adaptationPolicies[judgmentAssetAdaptationMode],
      "使った判断資産は、回答末尾に「判断資産出典: 正規 JA-<ID短縮> / <research_topic>」または「判断資産出典: 候補 JA-<ID短縮> / <research_topic>」として明記してください。",
      "元判断と応用後の判断を混同しないでください。応用後の確認観点・承認条件・反証を本文に出し、出典は根拠トレースとして残してください。",
      ...judgmentAssetCandidates.slice(0, 3).map((item, index) => (
        [
          `${isCanonicalJudgmentAsset(item) ? "正規判断資産" : "昇格候補"}${index + 1}: JA-${item.id.slice(0, 8)} / ${item.candidate_type} / ${item.research_topic}`,
          `元判断: ${item.claim}`,
          `使う文面: ${item.edited_claim || item.effective_claim || item.claim}`,
          `出典: ${item.evidence_path || "manual"}`,
        ].join("\n")
      )),
    );
  }
  if (experienceCases.length || pastReviews.length) {
    lines.push(
      "",
      "過去会社引用ルール:",
      "・保存済み経験ケースまたは過去レビューがある場合、2番か3番の本文中で必ず過去会社名を1社以上明示してください。",
      "・会社名を出したうえで、似ている点、違う点、今回の追加確認にどう使うかを短く述べてください。",
      "・過去会社を丸写しせず、今回案件との差分を判断材料にしてください。",
    );
  }
  lines.push("", "注意: 点数の再説明ではなく、審査判断として何を残すかに寄せてください。");
  return lines.join("\n");
};

export const buildShionReviewFallback = (
  result: Record<string, any>,
  data: Record<string, any>,
  judgmentAssetCandidates: JudgmentAssetCandidate[] = [],
  experienceCases: DemoSimilarPastCase[] = [],
  pastReviews: PastShionScreeningReview[] = [],
) => {
  const score = getScreeningScore(result);
  const hantei = String(result.hantei || "未判定");
  const companyName = data.company_name || "この案件";
  const industry = String(result.industry_sub || data.industry_sub || result.industry_major || data.industry_major || "業種未入力");
  const assetName = data.asset_name || "対象物件";
  const purpose = data.asset_purpose || "導入目的未入力";
  const memo = data.passion_text || "営業メモ未入力";
  const qRisk = result.quantum_risk != null ? Number(result.quantum_risk) : null;
  const qRiskText = qRisk != null && Number.isFinite(qRisk)
    ? `Q_risk ${qRisk.toFixed(1)}`
    : "Q_risk 未算出";
  const candidateAsset = judgmentAssetCandidates.find((item) => !isCanonicalJudgmentAsset(item));
  const canonicalAsset = judgmentAssetCandidates.find((item) => isCanonicalJudgmentAsset(item));
  const primaryAsset = candidateAsset || canonicalAsset || judgmentAssetCandidates[0];
  const secondaryAsset = judgmentAssetCandidates.find((item) => item.id !== primaryAsset?.id);
  const pastCompany = uniquePastCompanyHighlights(pastReviews, experienceCases).at(0);
  const assetSources = judgmentAssetCandidates.slice(0, 3).map((item) => {
    const label = isCanonicalJudgmentAsset(item) ? "正規" : "候補";
    return `判断資産出典: ${label} JA-${item.id.slice(0, 8)} / ${item.research_topic || item.candidate_type || "screening"}`;
  });
  const primaryClaim = primaryAsset?.edited_claim || primaryAsset?.effective_claim || primaryAsset?.claim || "";
  const secondaryClaim = secondaryAsset?.edited_claim || secondaryAsset?.effective_claim || secondaryAsset?.claim || "";
  return [
    "1. 紫苑の第一印象",
    `${companyName}は${industry}の${assetName}案件です。総合スコアは${Number.isFinite(score) ? `${score.toFixed(1)}点` : "未算出"}、判定は${hantei}。点数だけで終わらせず、導入目的と返済原資の具体性を確認してから条件を組む案件です。`,
    "",
    "2. 数字だけでは見落としそうな違和感",
    `私なら、${qRiskText}と現場メモの具体性の差に注目します。営業メモは「${memo}」、導入目的は「${purpose}」。ここが抽象的なままだと、資金使途・稼働開始・売上寄与の説明が弱くなります。${pastCompany ? `過去の${pastCompany}と比べる場合も、社名の類似ではなく、似ている点・違う点・今回追加で聞くべきことに分けて確認します。` : "これは断定的な否認材料ではなく、確認論点として扱います。"}`,
    "",
    "3. 条件付き承認にするなら必要な確認",
    primaryClaim
      ? `判断資産を使うなら、まず「${primaryClaim}」を今回案件向けに確認質問へ落とします。${secondaryClaim ? `加えて「${secondaryClaim}」も条件文に使えるかを見ます。` : ""}`
      : "資金繰り表、稼働開始時期、既存債務、競合条件、物件の換価性を確認し、条件付き承認に足る説明を作ります。",
    "",
    "4. 稟議で残すべき一文",
    `本件は${assetName}導入による収益寄与と支払原資の具体性を確認し、未達時の代替返済原資または追加条件を明記したうえで判断する。`,
    ...(assetSources.length ? ["", ...assetSources] : []),
  ].join("\n");
};

export const buildShionThoughtProcessSteps = (
  result: Record<string, any>,
  judgmentAssetCandidates: JudgmentAssetCandidate[],
  pastCompanies: PastCompanyHighlight[],
  review: ShionScreeningReview | null,
): ShionThoughtStep[] => {
  const steps: ShionThoughtStep[] = [];
  if (!result) return steps;

  const numericItems: string[] = [];
  if (result.quantum_risk != null) {
    numericItems.push(`Q_risk ${Number(result.quantum_risk).toFixed(1)}（35以上で要注意・60以上で強警戒）`);
  }
  if (result.umap_anomaly_score != null) {
    numericItems.push(`UMAP異常度 ${Number(result.umap_anomaly_score).toFixed(1)}`);
  }
  if (result.mahalanobis_score != null) {
    numericItems.push(`マハラノビス距離 ${Number(result.mahalanobis_score).toFixed(1)}`);
  }
  if (numericItems.length) {
    steps.push({ title: "数値シグナルを確認", items: numericItems });
  }

  const flagItems: string[] = Array.isArray(result.aurion_core?.discipline_flags)
    ? result.aurion_core.discipline_flags
        .slice(0, 5)
        .map((f: any) => (typeof f === "string" ? f : f?.title ?? ""))
        .filter(Boolean)
    : [];
  if (flagItems.length) {
    steps.push({ title: "AURION警戒フラグを照合", items: flagItems });
  }

  const diagItems: string[] = Array.isArray(result.diagnostic_recommendations)
    ? result.diagnostic_recommendations.slice(0, 3).map((rec: any) => {
        const label = String(rec?.label || rec?.diagnostic || "補助診断");
        const status = rec?.status === "calculated" ? "算出済み" : "推奨";
        const reason = rec?.reason ? `（理由: ${rec.reason}）` : "";
        return `${label}: ${status}${reason}`;
      })
    : [];
  if (diagItems.length) {
    steps.push({ title: "補助診断を検討", items: diagItems });
  }

  const assetItems = judgmentAssetCandidates.slice(0, 3).map((item) => (
    `${isCanonicalJudgmentAsset(item) ? "正規判断資産" : "昇格候補"} JA-${item.id.slice(0, 8)} / ${item.research_topic || item.candidate_type || "screening"}`
  ));
  if (assetItems.length) {
    steps.push({ title: "参照した判断資産", items: assetItems });
  }

  const companyItems = pastCompanies.slice(0, 5).map((c) => `${c.name}（${c.label}）`);
  if (companyItems.length) {
    steps.push({ title: "参照した過去事例", items: companyItems });
  }

  if (review) {
    steps.push({
      title: "レビュー生成に使った参照数",
      items: [
        `記憶 ${review.memoryRefs}件 / 知識 ${review.knowledgeRefs}件`,
        `Vertex ${review.vertexUsed ? "使用" : review.vertexStatus || "未使用"}`,
      ],
    });
  }

  return steps;
};

export const parseExperienceSnapshot = (value: unknown): Record<string, any> | undefined => {
  if (!value) return undefined;
  if (typeof value === "object" && !Array.isArray(value)) return value as Record<string, any>;
  if (typeof value !== "string") return undefined;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, any>
      : undefined;
  } catch {
    return undefined;
  }
};

export const normalizeExperienceCase = (raw: any): DemoSimilarPastCase => ({
  id: Number(raw?.id || 0) || undefined,
  demoCaseId: String(raw?.demo_case_id || raw?.demoCaseId || ""),
  sourceCaseId: String(raw?.source_case_id || raw?.sourceCaseId || ""),
  companyName: String(raw?.company_name || raw?.companyName || "名称未設定"),
  period: String(raw?.period || ""),
  industry: String(raw?.industry_sub || raw?.industry || raw?.industry_major || ""),
  industryMajor: String(raw?.industry_major || raw?.industryMajor || ""),
  industrySub: String(raw?.industry_sub || raw?.industrySub || ""),
  salesDept: String(raw?.sales_dept || raw?.salesDept || ""),
  score: Number(raw?.score || 0),
  decision: String(raw?.decision || ""),
  outcome: String(raw?.outcome || ""),
  similarity: String(raw?.similarity || ""),
  actionTaken: String(raw?.action_taken || raw?.actionTaken || ""),
  lesson: String(raw?.lesson || ""),
  difference: String(raw?.difference || ""),
  source: String(raw?.source || ""),
  similarityScore: Number(raw?.similarity_score ?? raw?.similarityScore ?? 0),
  similarityReasons: Array.isArray(raw?.similarity_reasons)
    ? raw.similarity_reasons.map((reason: unknown) => String(reason)).filter(Boolean)
    : [],
  formSnapshot: parseExperienceSnapshot(raw?.form_snapshot ?? raw?.formSnapshot),
  resultSnapshot: parseExperienceSnapshot(raw?.result_snapshot ?? raw?.resultSnapshot),
});

export const buildExperienceCaseQuery = (
  demoCaseId: string,
  targetFormData: Record<string, any>,
  targetResult: any = null,
) => {
  const query: Record<string, string | number> = {
    demo_case_id: demoCaseId,
    industry_major: targetResult?.industry_major || targetFormData.industry_major || "",
    industry_sub: targetResult?.industry_sub || targetFormData.industry_sub || "",
    company_name: targetFormData.company_name || "",
    asset_name: targetFormData.asset_name || targetFormData.asset_detail || "",
    customer_type: targetFormData.customer_type || "",
    main_bank: targetFormData.main_bank || "",
    competitor: targetFormData.competitor || "",
    outcome_status: targetResult?.final_status || targetResult?.result_status || targetResult?.hantei || "",
    limit: 8,
  };
  // score は数値のときだけ送る。空文字を送ると FastAPI の Optional[float] が 422 を返す
  const scoreValue = targetResult?.score ?? targetResult?.score_base;
  if (typeof scoreValue === "number" && Number.isFinite(scoreValue)) {
    query.score = scoreValue;
  }
  return query;
};

export const hasExperienceSearchContext = (targetFormData: Record<string, any>, targetResult: any = null) =>
  Boolean(
    targetFormData.industry_sub ||
    targetFormData.industry_major ||
    targetFormData.asset_name ||
    targetFormData.customer_type ||
    targetFormData.main_bank ||
    targetFormData.competitor ||
    targetResult?.hantei ||
    targetResult?.score_base ||
    targetResult?.score,
  );

export const buildShionReviewUserId = (targetResult: any, targetFormData: Record<string, any>) => {
  const rawId = String(targetResult?.case_id || targetFormData.company_no || targetFormData.company_name || "draft");
  const safeId = rawId.replace(/[^\w\-ぁ-んァ-ヶ一-龠ー]/g, "_").slice(0, 64);
  return `screening-shion-review:${safeId || "draft"}`;
};
