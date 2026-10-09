import { expect, test } from '@playwright/test';

test('審査時点の根拠・版・実効減点を表示し、再読込後も保持する', async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ json: {} }));
  await page.addInitScript(() => {
    if (window.sessionStorage.getItem('judgment-basis-fixture-installed')) return;
    window.sessionStorage.setItem('judgment-basis-fixture-installed', '1');
    window.sessionStorage.setItem('lease-screening-return-state', JSON.stringify({
      version: 1,
      result: {
        score: 80, score_borrower: 90, hantei: '要審議',
        judgment_reasons: [
          { asset_id: 'SCORING-EQUITY', asset_version: 1, title: '債務超過の補正', summary: '自己資本比率がマイナスの場合、比率の0.5倍を減点する（最大30点）。', applied_reason: '自己資本比率 -20.0%', effect: 'adjustment', score_delta: -10 },
          { asset_id: 'SCORING-REVIEW-EQUITY', asset_version: 1, title: '債務超過による審議', summary: '点数にかかわらず人間の審議へ送る。追加減点は行わない。', applied_reason: '債務超過（自己資本比率 -20.0%）', effect: 'review', score_delta: 0 },
        ],
      },
      activeTab: 'analysis', savedAt: new Date().toISOString(),
    }));
  });
  await page.goto('/screening');
  const basis = page.getByRole('region', { name: '判断根拠', exact: true });
  await expect(basis).toContainText('総合スコア 80.0点');
  await expect(basis).toContainText('借手スコア（補正前）：90.0点');
  await expect(basis).toContainText('SCORING-EQUITY / v1');
  await expect(basis).toContainText('-10.0点');
  await expect(basis).toContainText('要審議・追加減点なし');
  await expect(basis).toContainText('今回の適用理由：自己資本比率 -20.0%');
  await page.reload();
  await expect(basis).toContainText('SCORING-EQUITY / v1');
  await basis.scrollIntoViewIfNeeded();
  await basis.screenshot({ path: '/workspace/scoring-judgment-basis-desktop.png' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(basis).toBeVisible();
  await basis.screenshot({ path: '/workspace/scoring-judgment-basis-mobile.png' });
  expect(await basis.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
});

test('根拠のない旧結果には明示的な案内を表示する', async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ json: {} }));
  await page.addInitScript(() => {
    window.sessionStorage.setItem('lease-screening-return-state', JSON.stringify({
      version: 1, result: { score: 72, hantei: '承認圏内' },
      activeTab: 'analysis', savedAt: new Date().toISOString(),
    }));
  });
  await page.goto('/screening');
  await expect(page.getByRole('region', { name: '判断根拠', exact: true })).toContainText('この結果には判断根拠が記録されていません。');
});
