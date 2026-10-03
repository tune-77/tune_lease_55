// 紫苑レビュー（審査分析画面の AI レビュー）を screening（PC）と lease-kun（スマホ）の
// 両方から同一ロジックで呼べるようにするための、状態を持たない純粋関数・型の置き場。
// プロンプト文面がここから外れると画面間で紫苑の挙動が食い違うため、
// 生成ロジックは screening/page.tsx に重複定義せず必ずここを参照すること。
//
// REV-358: 紫苑レビューには過去事歴（類似経験ケース・過去レビュー本文・過去会社名）を一切渡さない。
// 類似度計算が実質キーワード一致で無関係な会社を拾っていたため、レビューは今回案件の
// 数値・定性情報・判断資産だけで書く。過去案件は「過去案件から作成」入力アシストと
// 経験ケースパネル（screening/page.tsx）の側にのみ残す。

export type ShionReviewFeedback = "useful" | "needs_fix" | "wrong" | "specific" | "thin" | "discomfort_hit" | "over_inferred";
export type JudgmentAssetCandidateFeedback = "useful" | "neutral" | "rejected" | "not_applied";
export type JudgmentAssetAdaptationMode = "conservative" | "standard" | "exploratory" | "aggressive";

export type LooseRecord = Record<string, unknown>;
export type ScreeningFormRecord = {
  company_no?: unknown;
  company_name?: unknown;
  industry_major?: unknown;
  industry_sub?: unknown;
  industry_detail?: unknown;
  sales_dept?: unknown;
  asset_name?: unknown;
  asset_detail?: unknown;
  asset_purpose?: unknown;
  acquisition_cost?: unknown;
  lease_term?: unknown;
  contract_type?: unknown;
  customer_type?: unknown;
  main_bank?: unknown;
  qual_corr_main_bank?: unknown;
  bank_credit?: unknown;
  lease_credit?: unknown;
  deal_source?: unknown;
  passion_text?: unknown;
  intuition?: unknown;
  competitor?: unknown;
};
export type ScreeningResultRecord = LooseRecord & {
  score?: number;
  score_base?: number;
  hantei?: string;
  approval_line?: number;
  risk_review_required?: boolean;
  risk_review_reasons?: string[];
  score_borrower?: number;
  quantum_risk?: number;
  umap_anomaly_score?: number;
  mahalanobis_score?: number;
  aurion_core?: { severity?: string; discipline_flags?: unknown[] };
  diagnostic_recommendations?: LooseRecord[];
  q_risk_breakdown?: QRiskBreakdown;
};

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
  // 打ち切り時間を過ぎて簡易生成を先に見せている間 true。紫苑の本文が届いたら差し替える。
  lateReplyPending?: boolean;
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
  // "policy" は社内方針（PR #1225 の方針/知見の分類）。方針は変形させず、知見とは別に渡す。
  knowledge_kind?: string;
  userFeedback?: JudgmentAssetCandidateFeedback;
  lastFeedbackEventId?: string;
  user_feedback?: JudgmentAssetCandidateFeedback;
  last_feedback_event_id?: string;
};

// 紫苑レビューの人間評価だけを読むための最小形。
// REV-358 以降、過去レビューから紫苑へ渡してよいのは「書き方への評価ラベル」だけで、
// 社名・スコア・本文は渡さない。API レスポンスには他のフィールドも含まれるが、意図的に読まない。
export type ShionReviewFeedbackSample = {
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
  formSnapshot?: LooseRecord;
  resultSnapshot?: LooseRecord;
};

// Q_risk のルール別寄与内訳（API: /api/score/full の q_risk_breakdown）。
// 表示専用でスコアには影響しない。clipped=true のとき raw_total は 100 を超えており、
// weighted は表示 Q_risk へ按分済みの寄与点。
export type QRiskBreakdownItem = {
  code: string;
  label: string;
  contribution: number;
  detail?: string;
  share?: number;
  weighted?: number;
};

export type QRiskBreakdown = {
  total: number;
  raw_total: number;
  clipped: boolean;
  items: QRiskBreakdownItem[];
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

export const getScreeningScore = (result?: LooseRecord | null) =>
  Number(result?.score ?? result?.score_base ?? 0);

// プロンプトが LLM に渡す判断資産（知見）の件数。社内方針はこれとは別枠で全件渡す。
export const PROMPTED_JUDGMENT_ASSET_LIMIT = 3;

export const isPolicyJudgmentAsset = (candidate: JudgmentAssetCandidate) => candidate.knowledge_kind === "policy";

// プロンプトへ実際に渡す判断資産。方針（全件）と知見（先頭 PROMPTED_JUDGMENT_ASSET_LIMIT 件）。
export const promptedJudgmentAssets = (candidates: JudgmentAssetCandidate[] = []) => ({
  policies: candidates.filter(isPolicyJudgmentAsset),
  insights: candidates.filter((item) => !isPolicyJudgmentAsset(item)).slice(0, PROMPTED_JUDGMENT_ASSET_LIMIT),
});

export const formatJudgmentAssetCitation = (item: JudgmentAssetCandidate) => {
  const label = isPolicyJudgmentAsset(item) ? "方針" : isCanonicalJudgmentAsset(item) ? "正規" : "候補";
  return `判断資産出典: ${label} JA-${item.id.slice(0, 8)} / ${item.research_topic || item.candidate_type || "screening"}`;
};

const reviewBigrams = (text: string) => {
  const compact = String(text || "").replace(/[\s、。,.()（）「」『』:：/・*#>\-【】[\]0-9A-Za-z%]/g, "");
  const grams = new Set<string>();
  for (let i = 0; i < compact.length - 1; i += 1) grams.add(compact.slice(i, i + 2));
  return grams;
};

// 判断資産の文面の文字bigramのうち、本文に現れた割合。言い換えて使われても拾えるよう語ではなく bigram で見る。
// 2026-10-03 の実レビュー3件で、使った資産は 0.26〜0.41、使っていない資産は 0.16 だった。
export const JUDGMENT_ASSET_USE_THRESHOLD = 0.25;

export const judgmentAssetUsedInReview = (reviewText: string, item: JudgmentAssetCandidate) => {
  const claim = reviewBigrams(item.edited_claim || item.effective_claim || item.claim || "");
  if (!claim.size) return false;
  const body = reviewBigrams(String(reviewText || "").replace(/判断資産出典[^\n]*/g, ""));
  let shared = 0;
  claim.forEach((gram) => { if (body.has(gram)) shared += 1; });
  return shared / claim.size >= JUDGMENT_ASSET_USE_THRESHOLD;
};

// LLM が出典を書き忘れた判断資産のうち、本文で実際に使っているものだけ末尾に出典を補う。
//
// api/routers/feedback_loop.py の _record_judgment_asset_feedback_from_review は
// 本文中の「JA-cr-<rule_id先頭>」だけを手がかりに、レビュー評価を判断資産へ紐付ける。
// 出典が本文に無いと no_matching_refs で捨てられ field_validation が 0 のままになるため補う。
// ただし使っていない資産に出典を付けると、評価が使っていない資産に紐付いてしまうので補わない
// （2026-10-03 までは渡した3件すべてに無条件で付けていた）。渡していない資産の出典も作らない。
export const ensureJudgmentAssetCitations = (
  reviewText: string,
  judgmentAssetCandidates: JudgmentAssetCandidate[] = [],
) => {
  const text = reviewText || "";
  const { policies, insights } = promptedJudgmentAssets(judgmentAssetCandidates);
  const missing = [...policies, ...insights]
    .filter((item) => item.id && !text.includes(`JA-${item.id.slice(0, 8)}`) && !text.includes(item.id.replace(/^cr-/, "").slice(0, 8)))
    .filter((item) => judgmentAssetUsedInReview(text, item));
  if (!missing.length) return text;
  return [text.trimEnd(), "", ...missing.map(formatJudgmentAssetCitation)].join("\n");
};

export const normalizeReviewText = (text: string) =>
  (text || "")
    .replace(/\\r\\n/g, "\n")
    .replace(/\\n/g, "\n")
    .trim();

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

// 否定的な人間評価と、それを受けて次のレビューで直すべきことの対応表。
// 扱うのは「レビュー文の書き方」への指摘だけで、案件の中身には触れない。
const NEGATIVE_REVIEW_FEEDBACK_ACTIONS: Partial<Record<ShionReviewFeedback, string>> = {
  thin: "リスク項目の列挙で終わらせず、注目する1点に絞って根拠まで掘り下げること。",
  over_inferred: "根拠の薄い推測を断定で書かず、確認論点・仮説として置くこと。",
  wrong: "案件情報から確認できないことを事実として書かないこと。",
  needs_fix: "そのまま稟議に貼れる粒度まで具体化して書くこと。",
};

// 直近レビューへの人間評価から、次のレビューへの自己補正1ブロックを作る。
// 過去案件の社名・スコア・本文は渡さない（REV-358 の方針）。評価ラベルだけを使う。
export const buildReviewQualityFeedbackBlock = (
  feedbacks: (ShionReviewFeedback | "" | undefined)[],
) => {
  const counts = new Map<ShionReviewFeedback, number>();
  for (const feedback of feedbacks) {
    if (!feedback || !NEGATIVE_REVIEW_FEEDBACK_ACTIONS[feedback]) continue;
    counts.set(feedback, (counts.get(feedback) || 0) + 1);
  }
  if (!counts.size) return "";
  const ranked = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2);
  return [
    "【直近レビューへの人間評価】",
    "次は、今回案件とは無関係に、あなたの直近レビューの書き方に対して人間が付けた評価です。同じ指摘を繰り返さないでください。",
    ...ranked.map(([feedback, count]) => (
      `・「${FEEDBACK_LABELS[feedback]}」${count}件 → ${NEGATIVE_REVIEW_FEEDBACK_ACTIONS[feedback]}`
    )),
  ].join("\n");
};

export const buildVertexSearchHint = (result: ScreeningResultRecord, data: ScreeningFormRecord) => {
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

const formatMillionYen = (value: unknown) => {
  const amount = Number(value);
  return value === "" || value == null || !Number.isFinite(amount) ? "" : `${amount}百万円`;
};

// 金額は検索に要らないので帯だけにする（個人を特定しうる正確な値を検索・ログへ出さない）
const acquisitionCostBand = (value: unknown) => {
  const amount = Number(value);
  if (!Number.isFinite(amount) || amount <= 0) return "";
  if (amount < 10) return "取得価額1千万円未満";
  if (amount < 50) return "取得価額1千万〜5千万円";
  if (amount < 100) return "取得価額5千万〜1億円";
  return "取得価額1億円以上";
};

// 銀行与信とリース与信（今回分を含む）の大小。社内方針「メイン銀行の借入よりリースの借入を増やさない」の判定材料。
const leaseVsBankCreditText = (data: ScreeningFormRecord) => {
  const bank = Number(data.bank_credit);
  const lease = Number(data.lease_credit);
  const cost = Number(data.acquisition_cost);
  if (data.bank_credit === "" || data.bank_credit == null || !Number.isFinite(bank)) return "";
  const leaseAfter = (Number.isFinite(lease) ? lease : 0) + (Number.isFinite(cost) ? cost : 0);
  if (bank <= 0) return "銀行与信なし";
  return leaseAfter > bank ? "今回を含むリース与信が銀行与信を上回る" : "今回を含むリース与信は銀行与信以下";
};

// 紫苑レビューの検索・想起・方針照合に使う短い案件要約（/api/chat の retrieval_query）。
// 依頼文（約3,000字）をそのまま検索に使うと、埋め込みモデルが先頭の定型文しか読まず、
// どの案件でも同じ過去レビューが当たり、キーワード再ランクも1件数分かかっていた（2026-10-03）。
// 社名・営業メモ・正確な金額は入れない（検索ログ・Vertex へ出るため）。
export const buildShionReviewRetrievalQuery = (result: ScreeningResultRecord, data: ScreeningFormRecord) => {
  const qRiskLabels = (result.q_risk_breakdown as QRiskBreakdown | undefined)?.items?.slice(0, 3).map((item) => item.label) ?? [];
  const flags = Array.isArray(result.aurion_core?.discipline_flags)
    ? result.aurion_core.discipline_flags
        .slice(0, 3)
        .map((flag) => (typeof flag === "string" ? flag : (flag as { title?: string })?.title ?? ""))
    : [];
  const parts = [
    "リース審査",
    buildVertexSearchHint(result, data),
    String(data.qual_corr_main_bank || "").replace(/^未選択$/, ""),
    leaseVsBankCreditText(data),
    acquisitionCostBand(data.acquisition_cost),
    data.lease_term ? `リース期間${data.lease_term}ヶ月` : "",
    String(result.hantei || ""),
    Number(result.quantum_risk) >= 35 ? "Q_risk要注意" : "",
    ...qRiskLabels,
    ...flags,
  ];
  return Array.from(new Set(parts.map((part) => String(part || "").trim()).filter(Boolean))).join(" ").slice(0, 400);
};

// Q_risk の内訳を「営業赤字 29.5 / 売上規模対比の利益率異常 40.0」の形の1行へ整形する。
// 紫苑が「Q_risk のどの成分に反応したか」を根拠として書けるよう、プロンプトと思考プロセスの
// 両方から同じ文字列を使う。寄与の大きい順に最大3件まで。
export const formatQRiskBreakdown = (breakdown?: QRiskBreakdown | null, limit = 3) => {
  const items = breakdown?.items ?? [];
  if (!items.length) return "";
  return items
    .slice(0, limit)
    .map((item) => `${item.label} ${Number(item.weighted ?? item.contribution ?? 0).toFixed(1)}`)
    .join(" / ");
};

export const buildShionReviewPrompt = (
  result: ScreeningResultRecord,
  data: ScreeningFormRecord,
  judgmentAssetCandidates: JudgmentAssetCandidate[] = [],
  judgmentAssetAdaptationMode: JudgmentAssetAdaptationMode = "standard",
  recentReviewFeedbacks: (ShionReviewFeedback | "" | undefined)[] = [],
) => {
  const score = getScreeningScore(result);
  const baseScore = Number(result.score_base);
  const qRiskBreakdownText = formatQRiskBreakdown(result.q_risk_breakdown as QRiskBreakdown | undefined);
  const leaseVsBank = leaseVsBankCreditText(data);
  // 検索ヒントは依頼文に入れず、buildShionReviewRetrievalQuery で retrieval_query として別に渡す
  const lines = [
    "【審査分析画面からの紫苑レビュー依頼】",
    "この案件を、審査担当者の横にいる紫苑としてレビューしてください。",
    "",
    "出力は短くしてください。必ず書くのは次の2項目だけです。",
    "・違和感: 数字だけでは見落としそうな点。何を根拠にそう感じたかまで書く。",
    "・稟議に残す一文: そのまま稟議書に貼れる一文。",
    "",
    "そのうえで、この案件で書く価値がある項目だけを次から1〜2個選んで足してください。",
    "選択肢: 第一印象 / 条件付き承認にするなら必要な確認 / この見立てが外れるとしたら何か（反証） / 物件と保全の見方 / 今回は論点が薄いと判断した理由",
    "・用意された項目を全部書かないでください。選んだ項目名をそのまま見出しにし、番号や決まった順番に縛られなくてよいです。",
    "・毎回同じ組み合わせを選ばないでください。案件の性質から見て書く意味がある項目を選んでください。",
    "・埋めるための一般論を足さないでください。書くことがなければ項目は2つだけで構いません。",
    "",
    "専門家としての深掘りルール:",
    "・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください。",
    "・違和感の項目では、提示された数字・Q_risk・定性項目・現場メモのうち何が根拠になったかを具体的に結びつけてください。",
    "・Q_riskの内訳が提示されている場合は、合計値ではなく寄与の大きいルール名を根拠として挙げてください（例: 営業赤字が主因）。",
    "・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください。",
    "・不確実な推測で採否を誘導しないでください。違和感は減点ではなく、人間が確認するための論点です。",
    "・過去の類似案件や他社事例は渡していません。手元にない過去事例を推測で作って引用しないでください。",
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
    ...(qRiskBreakdownText
      ? [`・Q_riskの内訳（財務矛盾ルール別の寄与点）: ${qRiskBreakdownText}`]
      : []),
    `・UMAP異常度: ${result.umap_anomaly_score != null ? Number(result.umap_anomaly_score).toFixed(1) : "未算出"}`,
    `・マハラノビス: ${result.mahalanobis_score != null ? Number(result.mahalanobis_score).toFixed(1) : "未算出"}`,
    `・物件: ${data.asset_name || "未入力"}`,
    `・取得価額: ${data.acquisition_cost || 0}百万円`,
    `・リース期間: ${data.lease_term || 0}`,
    `・導入目的: ${data.asset_purpose || "未入力"}`,
    `・当行区分: ${data.main_bank || "未入力"}`,
    `・メイン銀行関係: ${data.qual_corr_main_bank && data.qual_corr_main_bank !== "未選択" ? data.qual_corr_main_bank : "未入力"}`,
    `・案件発生経路: ${data.deal_source || "未入力"}`,
    `・銀行与信残高: ${formatMillionYen(data.bank_credit) || "未入力"}`,
    `・リース与信残高（他社含む）: ${formatMillionYen(data.lease_credit) || "未入力"}`,
    ...(leaseVsBank ? [`・与信の大小: ${leaseVsBank}`] : []),
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
  const { policies, insights } = promptedJudgmentAssets(judgmentAssetCandidates);
  if (policies.length) {
    // 方針は知見と違い、変形・弱めずに使う（PR #1225 の【社内方針】と同じ扱い。サーバー側でも同じ節を末尾に置く）
    lines.push(
      "",
      "【社内方針（ユーザーが定めたルール）として登録された判断資産】",
      "次は社内方針です。丸写しせず変形・応用する判断資産とは違い、方針は文面を変えず、弱めずに当てはめてください。",
      "この案件に当てはまる方針があれば、レビューの冒頭で結論として述べてください（例:「社内方針では、〜とは取引しません」）。確認事項や例外は方針を示した後に補足として書きます。",
      "当てはまるか前提の情報だけでは決められない時は、何を確認すれば決まるかを書いてください。当てはまらない方針には触れないでください。",
      "使った方針は回答末尾に「判断資産出典: 方針 JA-<ID短縮> / <research_topic>」として明記してください。",
      ...policies.map((item) => `社内方針: JA-${item.id.slice(0, 8)} / ${item.research_topic}\n方針: ${item.claim}`),
    );
  }
  if (insights.length) {
    const hasCanonicalAssets = insights.some((item) => (
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
      ...insights.map((item, index) => (
        [
          `${isCanonicalJudgmentAsset(item) ? "正規判断資産" : "昇格候補"}${index + 1}: JA-${item.id.slice(0, 8)} / ${item.candidate_type} / ${item.research_topic}`,
          `元判断: ${item.claim}`,
          `使う文面: ${item.edited_claim || item.effective_claim || item.claim}`,
          `出典: ${item.evidence_path || "manual"}`,
        ].join("\n")
      )),
    );
  }
  const reviewQualityFeedbackBlock = buildReviewQualityFeedbackBlock(recentReviewFeedbacks);
  if (reviewQualityFeedbackBlock) {
    lines.push("", reviewQualityFeedbackBlock);
  }
  lines.push("", "注意: 点数の再説明ではなく、審査判断として何を残すかに寄せてください。");
  return lines.join("\n");
};

// LLM 応答が得られなかったときの簡易生成。定型文なので、カード側で「簡易生成」バッジを出して
// 紫苑が書いた本文と区別できるようにしている（ShionScreeningReviewCard の isFallback）。
export const buildShionReviewFallback = (
  result: ScreeningResultRecord,
  data: ScreeningFormRecord,
  judgmentAssetCandidates: JudgmentAssetCandidate[] = [],
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
  // 定型文は方針を判定できないので知見だけを使い、出典も本文で使った資産（primary/secondary）だけに付ける
  const { insights } = promptedJudgmentAssets(judgmentAssetCandidates);
  const candidateAsset = insights.find((item) => !isCanonicalJudgmentAsset(item));
  const canonicalAsset = insights.find((item) => isCanonicalJudgmentAsset(item));
  const primaryAsset = candidateAsset || canonicalAsset || insights[0];
  const secondaryAsset = insights.find((item) => item.id !== primaryAsset?.id);
  const assetSources = [primaryAsset, secondaryAsset]
    .filter((item): item is JudgmentAssetCandidate => Boolean(item))
    .map(formatJudgmentAssetCitation);
  const primaryClaim = primaryAsset?.edited_claim || primaryAsset?.effective_claim || primaryAsset?.claim || "";
  const secondaryClaim = secondaryAsset?.edited_claim || secondaryAsset?.effective_claim || secondaryAsset?.claim || "";
  return [
    "違和感",
    `${companyName}は${industry}の${assetName}案件、総合スコア${Number.isFinite(score) ? `${score.toFixed(1)}点` : "未算出"}で判定は${hantei}です。私なら、${qRiskText}と現場メモの具体性の差に注目します。営業メモは「${memo}」、導入目的は「${purpose}」。ここが抽象的なままだと、資金使途・稼働開始・売上寄与の説明が弱くなります。これは断定的な否認材料ではなく、確認論点として扱います。`,
    "",
    "条件付き承認にするなら必要な確認",
    primaryClaim
      ? `判断資産を使うなら、まず「${primaryClaim}」を今回案件向けに確認質問へ落とします。${secondaryClaim ? `加えて「${secondaryClaim}」も条件文に使えるかを見ます。` : ""}`
      : "資金繰り表、稼働開始時期、既存債務、競合条件、物件の換価性を確認し、条件付き承認に足る説明を作ります。",
    "",
    "稟議に残す一文",
    `本件は${assetName}導入による収益寄与と支払原資の具体性を確認し、未達時の代替返済原資または追加条件を明記したうえで判断する。`,
    ...(assetSources.length ? ["", ...assetSources] : []),
  ].join("\n");
};

export const buildShionThoughtProcessSteps = (
  result: ScreeningResultRecord,
  judgmentAssetCandidates: JudgmentAssetCandidate[],
  review: ShionScreeningReview | null,
): ShionThoughtStep[] => {
  const steps: ShionThoughtStep[] = [];
  if (!result) return steps;

  const numericItems: string[] = [];
  if (result.quantum_risk != null) {
    numericItems.push(`Q_risk ${Number(result.quantum_risk).toFixed(1)}（35以上で要注意・60以上で強警戒）`);
    const breakdownText = formatQRiskBreakdown(result.q_risk_breakdown as QRiskBreakdown | undefined);
    if (breakdownText) {
      numericItems.push(`Q_riskの内訳: ${breakdownText}`);
    }
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
        .map((flag) => (typeof flag === "string" ? flag : (flag as { title?: string })?.title ?? ""))
        .filter(Boolean)
    : [];
  if (flagItems.length) {
    steps.push({ title: "AURION警戒フラグを照合", items: flagItems });
  }

  const diagItems: string[] = Array.isArray(result.diagnostic_recommendations)
    ? result.diagnostic_recommendations.slice(0, 3).map((rec) => {
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

export const parseExperienceSnapshot = (value: unknown): LooseRecord | undefined => {
  if (!value) return undefined;
  if (typeof value === "object" && !Array.isArray(value)) return value as LooseRecord;
  if (typeof value !== "string") return undefined;
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as LooseRecord
      : undefined;
  } catch {
    return undefined;
  }
};

export const normalizeExperienceCase = (rawValue: unknown): DemoSimilarPastCase => {
  const raw = rawValue && typeof rawValue === "object" && !Array.isArray(rawValue)
    ? rawValue as LooseRecord
    : {};
  return ({
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
};

export const buildExperienceCaseQuery = (
  demoCaseId: string,
  targetFormData: ScreeningFormRecord,
  targetResult: LooseRecord | null = null,
) => {
  const query: Record<string, string | number> = {
    demo_case_id: demoCaseId,
    industry_major: String(targetResult?.industry_major || targetFormData.industry_major || ""),
    industry_sub: String(targetResult?.industry_sub || targetFormData.industry_sub || ""),
    company_name: String(targetFormData.company_name || ""),
    asset_name: String(targetFormData.asset_name || targetFormData.asset_detail || ""),
    customer_type: String(targetFormData.customer_type || ""),
    main_bank: String(targetFormData.main_bank || ""),
    competitor: String(targetFormData.competitor || ""),
    outcome_status: String(targetResult?.final_status || targetResult?.result_status || targetResult?.hantei || ""),
    limit: 8,
  };
  // score は数値のときだけ送る。空文字を送ると FastAPI の Optional[float] が 422 を返す
  const scoreValue = targetResult?.score ?? targetResult?.score_base;
  if (typeof scoreValue === "number" && Number.isFinite(scoreValue)) {
    query.score = scoreValue;
  }
  return query;
};

export const hasExperienceSearchContext = (targetFormData: ScreeningFormRecord, targetResult: LooseRecord | null = null) =>
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

export const buildShionReviewUserId = (targetResult: LooseRecord | null, targetFormData: ScreeningFormRecord) => {
  const rawId = String(targetResult?.case_id || targetFormData.company_no || targetFormData.company_name || "draft");
  const safeId = rawId.replace(/[^\w\-ぁ-んァ-ヶ一-龠ー]/g, "_").slice(0, 64);
  return `screening-shion-review:${safeId || "draft"}`;
};

// 画面の打ち切り。これを過ぎたら簡易生成を先に見せ、紫苑の本文が届いたら差し替える。
// 本文は SHION_REVIEW_HARD_TIMEOUT_MS まで待つ（以前は120秒で通信を切り、サーバーで完成した本文が画面に出なかった）。
export const SHION_REVIEW_SOFT_TIMEOUT_MS = 120000;
export const SHION_REVIEW_HARD_TIMEOUT_MS = 600000;

// /api/chat への紫苑レビュー依頼。caller でサーバーがレビューと判別し、依頼文を Vault・記録へ残さず、
// 検索・想起・方針照合には retrieval_query（社名・営業メモを含まない案件要約）を使う。
export const buildShionReviewChatBody = (
  targetResult: ScreeningResultRecord,
  targetFormData: ScreeningFormRecord,
  promptText: string,
) => ({
  message: promptText,
  user_id: buildShionReviewUserId(targetResult, targetFormData),
  response_mode: "shion" as const,
  debug_memory: true,
  caller: "screening_review",
  retrieval_query: buildShionReviewRetrievalQuery(targetResult, targetFormData),
});

// request が ms 以内に終われば {done: true, value}、間に合わなければ {done: false}。
// 間に合わなかった後の失敗は呼び出し側が request を await して受ける（ここでは握り潰さず、未処理にもしない）。
export const waitForSoftTimeout = <T>(request: Promise<T>, ms: number) =>
  new Promise<{ done: true; value: T } | { done: false }>((resolve, reject) => {
    const timer = setTimeout(() => resolve({ done: false }), ms);
    request.then(
      (value) => { clearTimeout(timer); resolve({ done: true, value }); },
      (error) => { clearTimeout(timer); reject(error); },
    );
  });

export const shionReviewFromChatResponse = (
  data: LooseRecord | undefined,
  judgmentAssetCandidates: JudgmentAssetCandidate[],
): ShionScreeningReview => {
  const payload = data || {};
  const memoryDebug = (payload.memory_debug || {}) as LooseRecord;
  const memoryRecall = (memoryDebug.memory_recall || {}) as LooseRecord;
  const identityMemory = (memoryDebug.identity_memory || {}) as LooseRecord;
  const vertexSearch = (memoryDebug.vertex_ai_search || payload.vertex_ai_search || {}) as LooseRecord;
  const vertexAnswer = (memoryDebug.vertex_answer_api || payload.vertex_answer_api || {}) as LooseRecord;
  const parsedGroundingScore = vertexAnswer.grounding_score != null ? Number(vertexAnswer.grounding_score) : null;
  return {
    reply: ensureJudgmentAssetCitations(String(payload.reply || "紫苑レビューが空でした。"), judgmentAssetCandidates),
    memoryRefs: Array.isArray(memoryRecall.refs) ? memoryRecall.refs.length : 0,
    knowledgeRefs: Array.isArray(memoryDebug.knowledge_refs) ? memoryDebug.knowledge_refs.length : 0,
    identityUsed: Boolean(identityMemory.used),
    vertexUsed: Boolean(vertexSearch.used),
    vertexStatus: String(vertexSearch.status || ""),
    vertexRefs: Array.isArray(vertexSearch.refs) ? vertexSearch.refs.map(String) : [],
    vertexAnswerUsed: Boolean(vertexAnswer.used),
    vertexAnswerStatus: String(vertexAnswer.status || ""),
    groundingScore: parsedGroundingScore != null && Number.isFinite(parsedGroundingScore) ? parsedGroundingScore : null,
    groundingScoreSource: String(vertexAnswer.grounding_score_source || ""),
    lowSupportClaimCount: Number(vertexAnswer.low_support_claim_count || 0),
    supportCount: Number(vertexAnswer.support_count || 0),
  };
};

export const buildShionReviewFallbackRecord = (
  targetResult: ScreeningResultRecord,
  targetFormData: ScreeningFormRecord,
  judgmentAssetCandidates: JudgmentAssetCandidate[],
  lateReplyPending = false,
): ShionScreeningReview => ({
  reply: buildShionReviewFallback(targetResult, targetFormData, judgmentAssetCandidates),
  memoryRefs: 0,
  knowledgeRefs: judgmentAssetCandidates.length,
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
  lateReplyPending,
});
