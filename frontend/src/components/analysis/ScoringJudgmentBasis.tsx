export type ScoringJudgmentReason = {
  asset_id: string;
  asset_version: number;
  title: string;
  summary: string;
  applied_reason: string;
  effect: 'adjustment' | 'warning' | 'review' | 'decision';
  score_delta: number;
};

type Props = {
  result: {
    score?: number;
    score_borrower?: number;
    judgment_reasons?: ScoringJudgmentReason[];
  };
};

export default function ScoringJudgmentBasis({ result }: Props) {
  const reasons = result.judgment_reasons ?? [];
  return (
    <section aria-label="判断根拠" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-bold text-slate-800">判断根拠</h3>
        {typeof result.score === 'number' && (
          <span className="text-sm font-bold text-slate-700">総合スコア {result.score.toFixed(1)}点</span>
        )}
      </div>
      {reasons.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">この結果には判断根拠が記録されていません。</p>
      ) : (
        <>
          <p className="mt-2 text-sm text-slate-500">
            モデル算出後の補正と判定ルールを表示しています。モデル内部の特徴量ごとの寄与は含みません。
            {typeof result.score_borrower === 'number' && ` 借手スコア（補正前）：${result.score_borrower.toFixed(1)}点。`}
          </p>
          <ul className="mt-4 divide-y divide-slate-100">
            {reasons.map((reason) => (
              <li key={reason.asset_id} className="py-3 first:pt-0 last:pb-0">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-bold text-slate-800">{reason.title}</span>
                  <span className={`text-xs font-bold ${reason.score_delta < 0 ? 'text-rose-700' : 'text-slate-600'}`}>
                    {reason.effect === 'adjustment'
                      ? `${reason.score_delta > 0 ? '+' : ''}${reason.score_delta.toFixed(1)}点`
                      : reason.effect === 'review' ? '要審議・追加減点なし'
                      : reason.effect === 'decision' ? '判定ルール・加減点なし' : '参考情報・加減点なし'}
                  </span>
                </div>
                <p className="mt-1 text-sm text-slate-600">{reason.summary}</p>
                <p className="mt-2 text-sm text-slate-800">今回の適用理由：{reason.applied_reason}</p>
                <p className="mt-1 text-xs text-slate-500">判断資産：{reason.asset_id} / v{reason.asset_version}</p>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-slate-500">加減点は0〜100点の範囲制限を反映した実際の差分です。説明と版は審査時点の記録です。</p>
        </>
      )}
    </section>
  );
}
