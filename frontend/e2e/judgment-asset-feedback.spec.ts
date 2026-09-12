import { expect, test, type Page, type Route } from "@playwright/test";
const candidate = {
  id: "cr-b259411afb954d6d", candidate_type: "application_rule", research_topic: "business_plan_specificity", claim: "受注根拠と返済原資を合わせて確認する。", evidence_path: "test-fixture", promotion_status: "active", source: "canonical_judgment_rules", use_count: 0, useful_count: 0, rejected_count: 0, verified_status: "unverified",
};
const installScreeningDraft = async (page: Page, savedId: number | null = 7) => {
  await page.addInitScript(({ draftCandidate, savedId }) => {
    window.sessionStorage.setItem("lease-screening-return-state", JSON.stringify({
      version: 1,
      result: { case_id: "case-e2e-1", score: 72, hantei: "条件付き承認" },
      shionReview: { reply: "今回案件では受注根拠の確認が重要です。", memoryRefs: 0, knowledgeRefs: 0, identityUsed: false, ...(savedId ? { savedId } : {}) },
      judgmentAssetCandidates: [draftCandidate], judgmentAssetAdaptationMode: "standard", currentExperienceCases: [],
      activeTab: "analysis",
      savedAt: new Date().toISOString(),
    }));
  }, { draftCandidate: candidate, savedId });
};
const fulfillGenericApi = async (route: Route) => route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
test("one-click feedback is idempotent and clears the diagnostic outbox", async ({ page }) => {
  await installScreeningDraft(page);
  let posts = 0;
  await page.route("**/api/**", fulfillGenericApi);
  await page.route("**/api/judgment-asset-candidates/*/feedback", async (route) => {
    posts += 1;
    const payload = route.request().postDataJSON();
    await new Promise((resolve) => setTimeout(resolve, 150));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        duplicate: false,
        candidate: { ...candidate, use_count: 1, useful_count: 1 },
        feedback_event: { event_id: payload.event_id, disposition: "helped" },
      }),
    });
  });
  await page.goto("/screening");
  const feedbackCard = page.locator("section").filter({ hasText: "今回レビューに渡す判断資産" }).last();
  const useful = feedbackCard.getByRole("button", { name: "効いた", exact: true });
  await useful.dblclick();
  await expect(useful).toHaveText("効いた");
  await expect.poll(() => posts).toBe(1);
  const outbox = await page.evaluate(() => window.sessionStorage.getItem("judgment-asset-feedback-diagnostic-outbox-v1"));
  expect(outbox).toBe("[]");
});
test("feedback stays disabled until the review has a stable saved id", async ({ page }) => {
  await installScreeningDraft(page, null);
  await page.route("**/api/**", fulfillGenericApi);
  await page.goto("/screening");
  const feedbackCard = page.locator("section").filter({ hasText: "今回レビューに渡す判断資産" }).last();
  await expect(feedbackCard).toContainText("レビュー保存後に評価できます");
  await expect(feedbackCard.getByRole("button", { name: "効いた", exact: true })).toBeDisabled();
});
test("failed save rolls back and explicit retry reuses the event without storing PII", async ({ page }) => {
  await installScreeningDraft(page);
  let posts = 0;
  await page.route("**/api/**", fulfillGenericApi);
  await page.route("**/api/judgment-asset-candidates/*/feedback", async (route) => {
    posts += 1;
    const payload = route.request().postDataJSON();
    if (posts === 1) {
      await route.fulfill({ status: 500, contentType: "application/json", body: '{"detail":"temporary"}' });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        duplicate: true,
        candidate: { ...candidate, use_count: 0 },
        feedback_event: { event_id: payload.event_id, disposition: "not_applied" },
      }),
    });
  });

  await page.goto("/screening");
  const feedbackCard = page.locator("section").filter({ hasText: "今回レビューに渡す判断資産" }).last();
  await feedbackCard.getByRole("button", { name: "今回は使わなかった", exact: true }).click();
  await expect(feedbackCard.getByRole("alert")).toContainText("保存できませんでした");

  const failedOutbox = await page.evaluate(() => window.sessionStorage.getItem("judgment-asset-feedback-diagnostic-outbox-v1"));
  expect(failedOutbox).toContain("network_or_server_error");
  expect(failedOutbox).not.toContain("case-e2e-1");
  expect(failedOutbox).not.toContain("受注根拠");

  await feedbackCard.getByRole("button", { name: "再試行" }).click();
  await expect(feedbackCard.getByRole("alert")).toHaveCount(0);
  await expect.poll(() => posts).toBe(2);
  const clearedOutbox = await page.evaluate(() => window.sessionStorage.getItem("judgment-asset-feedback-diagnostic-outbox-v1"));
  expect(clearedOutbox).toBe("[]");
});

test("conflict reloads the latest candidate and does not offer a blind retry", async ({ page }) => {
  await installScreeningDraft(page);
  let posts = 0;
  await page.route("**/api/**", fulfillGenericApi);
  await page.route("**/api/judgment-asset-candidates/screening*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        candidates: [{
          ...candidate,
          claim: "競合後の最新確認内容を確認する。",
          effective_claim: "競合後の最新確認内容を確認する。",
          user_feedback: "neutral",
          last_feedback_event_id: "33333333-3333-4333-8333-333333333333",
        }],
      }),
    });
  });
  await page.route("**/api/judgment-asset-candidates/*/feedback", async (route) => {
    posts += 1;
    await route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        detail: { message: "feedback changed elsewhere", current_event_id: "33333333-3333-4333-8333-333333333333" },
      }),
    });
  });

  await page.goto("/screening");
  const feedbackSection = page.locator("section").filter({ hasText: "今回レビューに渡す判断資産" }).last();
  await feedbackSection.getByRole("button", { name: "効いた", exact: true }).click();

  await expect(page.getByText(/最新内容を読み込みました/)).toBeVisible();
  const refreshedSection = page.locator("section").filter({ hasText: "今回レビューに渡す判断資産" }).last();
  await expect(refreshedSection).toContainText("競合後の最新確認内容を確認する。");
  await expect(refreshedSection.getByRole("button", { name: "再試行" })).toHaveCount(0);
  await expect.poll(() => posts).toBe(1);
});
