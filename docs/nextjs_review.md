# Next.js フロントエンド コードレビューレポート

**レビュー日**: 2026-05-21
**対象**: frontend/src/ 以下 全ファイル (49ファイル)
**スタック**: Next.js (App Router), React 19, TypeScript, Tailwind CSS v4, Recharts, D3

---

## エグゼクティブサマリー

全体的にUIの完成度は高く、スマホ対応・レイアウト構造・コンポーネント分割も概ねよく設計されている。一方で、**XSSリスクを持つ `dangerouslySetInnerHTML` の多用**が最も深刻な問題であり、AIの返答テキストを直接HTMLとして展開している。TypeScript型安全性については `any` 型が至る所に残存し、APIレスポンスの型定義が全体的に欠如している。パフォーマンス面では `RealGraphs` と `AdvancedAnalysis` コンポーネントがモックデータをハードコードしており実際の審査結果と連動していない。`agent/page.tsx` と `batch/page.tsx` に構文エラー（二重波括弧）が存在し、ビルドが通っているとしたら偶然に近い。

---

## 1. コード品質・構造

### 1-1. コンポーネント設計

**`src/app/page.tsx` (審査・分析ダッシュボード) — 240行、責務過多**

メインのダッシュボードページはフォーム入力・API呼び出し・結果表示・軍師チャット連携・タブ管理を1ファイルに収めており、単一責任原則を逸脱している。しかし子コンポーネントへの分割は適切に行われているため、実害は現在の行数規模では限定的。

**`src/app/lease-kun/page.tsx` — 549行、巨大ウィザード**

ステップ管理・フォームデータ・API呼び出し・チャット履歴管理を全て1コンポーネントに格納している。特に各ステップのフォームJSX（行308〜527）が1関数内にフラットに並んでおり、ステップ数増加時の保守が困難。ステップごとのフォームを独立コンポーネントに切り出すべき。

**`src/components/analysis/GunshiAdvice.tsx` — 412行、複合責務**

チャット履歴管理・類似案件取得・マークダウンレンダリング・UI描画を1コンポーネントが担っている。特にカスタムMarkdownパーサー（`renderMarkdown` 関数）を内部で持ち、後述のXSSリスクの根源になっている。

**`src/app/agent/page.tsx` および `src/app/batch/page.tsx` — 二重波括弧構文エラー**

両ファイルの関数定義に二重波括弧 `{{` が使われている（6行目 `export default function AgentPage() {{`、同様にBatchPage）。これはJSXでオブジェクトリテラルを開始してしまう構文エラーであり、通常はビルドエラーになるはず。

```tsx
// agent/page.tsx 6行目 — 誤り
export default function AgentPage() {{

// 正しくは
export default function AgentPage() {
```

**`src/components/analysis/AdvancedAnalysis.tsx` — モックデータのハードコード**

モンテカルロシミュレーション（行8〜13）は `setTimeout` で2.5秒待つだけで実際の計算を行わず、固定SVGパスを表示する（行62〜72）。連鎖倒産確率「14.2%」（行112）もハードコード。TimesFM予測も「改善傾向」という固定文字列（行132）を表示するだけ。実際のAPIと接続していないことを示す注記がUIに一切ない。

**`src/components/analysis/RealGraphs.tsx` — 全データがハードコード**

レーダーチャート・SHAPスコア・売上予測の3グラフすべてのデータが固定値（行11〜37）。コンポーネント名が「Real」Graphsであるにもかかわらず実際の審査結果データを受け取るpropsが存在しない。APIレスポンスの `data` を受け取るpropsを追加し、実際のスコア成分を反映すべき。

### 1-2. TypeScript型定義

**`any` 型の多用箇所一覧（抜粋）**

| ファイル | 行番号 | 問題 |
|---|---|---|
| `src/app/page.tsx` | 25 | `const [result, setResult] = useState<any>(null)` |
| `src/components/analysis/GunshiAdvice.tsx` | 9 | `formData: any` |
| `src/components/analysis/GunshiAdvice.tsx` | 70 | `const buildResponseMeta = (data: any)` |
| `src/components/analysis/ReportGenerator.tsx` | 9, 10 | `apiResult: any; formData: any` |
| `src/components/analysis/IndicatorCards.tsx` | 5 | `data: any` |
| `src/components/ScoreDAG.tsx` | 5 | `data: any` |
| `src/app/register/page.tsx` | 20, 21 | `pendingCases: any[]; selectedCase: any` |
| `src/app/agent-hub/page.tsx` | 74, 75 | `result: any; latestNovel: any` |
| `src/app/similar/page.tsx` | 184, 185 | `graphData: any; summary: any` |
| `src/app/competitor/page.tsx` | 11 | `data: any` |
| `src/app/financial/page.tsx` | 19 | `forecastData: any` |

**APIレスポンス型の未定義**

`/api/score/full` のレスポンス型が `any` のまま使用されている。`score_base`, `score_borrower`, `hantei`, `comparison` などのフィールドを持つ `ScoringResult` インターフェースをおよび各APIレスポンスの型を `src/types/index.ts` に追加すべき。

**`src/components/form/FormGeneral.tsx` 12〜16行目 — 型定義の曖昧さ**

```tsx
interface IndustryMasterEntry {
  mapping?: string;
  sub?: { [sub: string]: string };
  [key: string]: unknown;  // index signatureとoptional propertyの混在
}
interface IndustryMaster {
  [major: string]: IndustryMasterEntry | string[];  // 共用型でextractSubsが複雑化
}
```

`extractSubs` 関数（19〜24行目）がこの曖昧な型に起因して複雑になっている。

**`src/app/debate/page.tsx` 209行目 — `err: any` の型アサーション**

```tsx
} catch (err: any) {
  setError(err.response?.data?.detail || err.message || "エラーが発生しました");
```

`report/page.tsx` では適切に `unknown` を使って型ガードしているのに対し、`debate/page.tsx` では `any` を使用している（209行目）。統一すべき。

**`src/app/similar/page.tsx` 42行目 — `!` 非nullアサーション**

```tsx
data.find(d => d.label === beginForeLabel)!.forecast = lastHistVal;
```
`financial/page.tsx` 68行目のこのパターンは `find` が `undefined` を返した場合に実行時エラーになる。

### 1-3. 重複コード

**インライン定数の二重定義**

`INDUSTRIES` 配列が `src/app/debate/page.tsx` (37〜41行目) と `src/app/agent-hub/page.tsx` (56〜60行目) で全く同一のデータとして定義されている。共通定数ファイルに切り出すべき。

**`FormGeneral.tsx` の `useEffect` における依存配列漏れ**

```tsx
// FormGeneral.tsx 32〜51行目
useEffect(() => {
  const fetchIndustries = async () => { ... };
  fetchIndustries();
}, []);  // data.industry_major が依存配列に含まれていない

useEffect(() => {
  ...
  onChange('industry_sub', newSubs[0] || "");
  ...
}, [data.industry_major, industryMaster]);  // onChangeが依存配列に含まれていない
```

2つ目の `useEffect` は `onChange` が依存配列に含まれておらず、ESLint `react-hooks/exhaustive-deps` 違反。

**類似パターンのMarkdownパーサー**

`src/components/analysis/AIAnalysis.tsx`（12行目）、`src/app/report/page.tsx`（21〜38行目の `MarkdownBlock` コンポーネント）、`src/components/analysis/GunshiAdvice.tsx`（179〜186行目の `renderMarkdown` 関数）でそれぞれ独自のMarkdownパース処理を実装している。`react-markdown` ライブラリの導入か、共通Markdownレンダラーコンポーネントへの統一が必要。

---

## 2. バグ・動作リスク

### 2-1. XSS（クロスサイトスクリプティング）— 高リスク

**`src/components/analysis/GunshiAdvice.tsx` 343〜346行目**

```tsx
<div
  className="..."
  dangerouslySetInnerHTML={{ __html: renderMarkdown(chat.text) }}
/>
```

`chat.text` はバックエンドのAI（Gemini API）から受け取った文字列をそのままHTMLに変換して挿入している。`renderMarkdown` 関数（179〜186行目）はHTMLサニタイズを一切行わず、正規表現置換のみを実施。AIのレスポンスに `<script>` タグや `<img onerror=...>` などが含まれれば XSS が成立する。

**`src/components/analysis/ReportGenerator.tsx` 150〜168行目**

```tsx
return <span key={i} dangerouslySetInnerHTML={{ __html: line + '<br/>' }} />;
```

`gunshiText`（軍師AIの返答）と `report`（バックエンド生成レポート）の両方をサニタイズなしで `dangerouslySetInnerHTML` に渡している。特に153行目では `replace` を2回かけているが、最初の `replace` の結果が正しく機能していない（152行目の置換は `<h4>■ $1</h4>` というHTMLを生成し、153行目がそれを上書きしようとしているが、正規表現が再マッチしないため最初の置換が残る）。

**対処**: `DOMPurify` ライブラリを導入し、`dangerouslySetInnerHTML` に渡す前に必ずサニタイズすること。または `react-markdown` + `rehype-sanitize` に置き換えること。

### 2-2. APIエラーハンドリングの漏れ

**`src/app/register/page.tsx` 31〜36行目 — エラー時の UI フィードバックなし**

```tsx
const fetchPendingCases = async () => {
  try {
    const res = await apiClient.get(`/api/cases/pending`);
    setPendingCases(res.data);
  } catch (err) {
    console.error("Failed to fetch pending cases", err);
    // ユーザーへのフィードバックなし
  }
};
```

同様に `deleteCase`（40〜49行目）のエラー処理が `triggerMebuki('reject', ...)` だけで、削除失敗が実際に起きているかどうかユーザーが判断しにくい。

**`src/app/home/page.tsx` 48〜58行目 — フォールバックなし**

`/api/dashboard/stats` のエラー時に `loading` が `false` になるが、`stats` は `null` のまま。 `analysis` チェック（122行目）で「成約データが不足しています」というメッセージが表示されるが、実際にはAPIエラーである可能性をユーザーが識別できない。

**`src/app/competitor/page.tsx` 241行目 — ゼロ除算リスク**

```tsx
<div className="text-3xl font-black text-emerald-600">
  {Math.round((data?.summary?.total_won / data?.summary?.total_cases) * 100)}%
</div>
```

`total_cases` が `0` または `undefined` のとき `NaN%` または `Infinity%` が表示される。

### 2-3. null/undefined 参照リスク

**`src/app/financial/page.tsx` 68行目**

```tsx
data.find(d => d.label === beginForeLabel)!.forecast = lastHistVal;
```

`find` が `undefined` を返した場合（ラベルが一致しない場合）に `TypeError` が発生する。

**`src/app/lease-kun/page.tsx` 213〜221行目 — goBack の不安定な動作**

```tsx
const goBack = () => {
  if (step === 0) return;
  setStep(s => s - 1);
  setHistory(prev => {
    const nw = [...prev];
    nw.pop(); nw.pop();  // 常に末尾2件を削除
    return nw;
  });
};
```

`history` の末尾2件がユーザー入力とボットメッセージであることを前提としているが、`humor` メッセージが30%の確率で追加される（128〜134行目）ため、最大3件が追加される場合がある。「戻る」操作で意図しないメッセージが残る。

**`src/app/register/page.tsx` 253行目 — parseFloat が NaN になりうる**

```tsx
onChange={(e) => setFinalRate(parseFloat(e.target.value))}
```

ユーザーがフォームを空欄にした場合 `NaN` が state に格納され、API送信時に問題になる。

### 2-4. 入力バリデーション漏れ

**`src/app/register/page.tsx` 101〜130行目 — handleRegister**

`targetId` の存在チェックのみで、`finalRate` のNaN/負数チェック・`baseRate` の妥当性チェックが行われていない。APIに `NaN` や `Infinity` が送信される可能性がある。

**`src/app/debate/page.tsx` 196〜198行目**

```tsx
setForm(prev => ({ ...prev, [name]: isNaN(Number(value)) || value === "" ? value : Number(value) }));
```

`value === ""` の場合に文字列の空文字列が state に残り、後続の型変換で問題が起きうる。

---

## 3. UX・スマホ対応

### 3-1. レスポンシブデザインの問題

**`src/components/ScoreDAG.tsx` 31行目 — スマホで横スクロール必須**

```tsx
<div className="min-w-[700px] flex justify-between items-center px-4 relative">
```

`min-w-[700px]` が設定されており、スマホ（320〜390px幅）では横スクロールが強制される。28行目に「横スワイプで全体表示」というヒントテキストは追加されているが、UXとして不親切。フレックス縦配置への切り替えを検討すべき。

**`src/components/analysis/GunshiAdvice.tsx` 189行目 — 高さが固定**

```tsx
<div className="sticky top-24 h-[calc(100vh-8rem)] ...">
```

スマホではサイドバーが表示されないため、`top-24` のオフセットが不適切になる可能性がある。また `h-[calc(100vh-8rem)]` の固定高さは、スマホのアドレスバー表示/非表示（`dvh` vs `svh`）に対応していない。

**`src/app/similar/page.tsx` 322行目 — グラフエリアの高さ固定**

```tsx
<div className="... h-[78vh] min-h-[760px] ...">
```

`min-h-[760px]` があるため、スマホ（高さ700px程度）でビューポートからはみ出す。

### 3-2. スマホ入力の問題

**`src/app/register/page.tsx` 251〜256行目 — 数値入力の `NaN` 問題**

```tsx
<input 
  type="text" inputMode="decimal" step="0.01" value={finalRate}
  onChange={(e) => setFinalRate(parseFloat(e.target.value))}
```

`SliderInput.tsx` では適切にローカル文字列状態を管理しているが、`register/page.tsx` の数値入力では中間状態（`"-"` や空文字列）の管理がなく、入力途中で `NaN` が state に入る。`SliderInput.tsx` のパターンを使い回すべき。

**`src/app/debate/page.tsx` 236〜241行目 — `min` 属性のバグ**

```tsx
<input
  name="score" type="text" inputMode="decimal" min={0} max={100} required
  ...
```

`type="text"` に `min`/`max` 属性を指定しても HTML5 のバリデーションが効かない。`type="number"` か、手動バリデーションへ変更すべき。

### 3-3. ローディング・フィードバック

**`src/app/home/page.tsx` — API失敗時の区別ができない**

API失敗時も「成約データが不足しています」と表示されるため、ユーザーはネットワーク障害かデータ不足か判断できない。

**`src/app/agent-hub/page.tsx` 84〜86行目 — `isRunning` フラグが共有**

```tsx
const [isRunning, setIsRunning] = useState(false);
```

`isRunning` がグローバルで1つだけ存在するため、あるエージェントが実行中は他の全エージェントボタンが `disabled` になる。別のエージェントを並行して実行したい場合のUXが悪い。

---

## 4. パフォーマンス

### 4-1. 不要な再計算・再レンダリング

**`src/app/agent-hub/page.tsx` 84〜86行目 — setInterval でポーリング**

```tsx
const interval = setInterval(fetchThoughts, 15000); // 15秒おきに更新
```

コンポーネントがマウント中は常に15秒ごとにAPIコールを発火する。ユーザーがタブを離れた場合も継続し、サーバー負荷を増加させる。Page Visibility API を利用してバックグラウンドタブ時は停止すべき。

**`src/components/analysis/GunshiAdvice.tsx` 122〜132行目**

```tsx
useEffect(() => {
  ...
}, [score, pd_percent, industry_major, formData]);
```

`formData` オブジェクト全体が依存配列に含まれているため、フォームの任意のフィールドが変更されるたびにこのエフェクトが評価される（`fetchKey` 文字列による重複防止はあるが）。`formData.nenshu`, `formData.op_profit` など必要なフィールドのみを依存配列に並べるべき。

**`src/app/similar/page.tsx` 326〜335行目 — iframe srcDoc 再構築**

スライダー操作のたびに `D3_TEMPLATE` 文字列の置換が発生し、iframe がリロードされる。`viewKey` ステートは明示的に「再配置」ボタンだけでインクリメントされているが、スライダー変更時はプロパティ経由で D3 シミュレーションを更新する設計のほうが効率的。

### 4-2. 重いコンポーネント

**`src/app/similar/page.tsx` — D3_TEMPLATE 定数（7〜180行目）**

180行の HTML/JS 文字列がモジュールのトップレベルに定数として定義されている。JSバンドルに含まれるが、実際にはiframe内で使われるだけであり、動的インポートまたは別ファイルへの分離を検討すべき。

**`src/app/lease-kun/page.tsx` — インラインスタイル定数（233〜236行目）**

```tsx
const sel = "w-full bg-slate-50 border ...（省略）";
const inp = "...";
const inpReq = "...";
const lbl = "...";
```

毎レンダリング時に文字列変数が再作成される。`const` なので値は変わらないが、`useMemo` または モジュールレベルの定数として定義すべき。

---

## 5. セキュリティ

### 5-1. XSS リスク（高）

前述の通り以下の3箇所で `dangerouslySetInnerHTML` にサニタイズなしの外部コンテンツを渡している。

- `src/components/analysis/GunshiAdvice.tsx` 343行目 — Gemini API レスポンス直接展開
- `src/components/analysis/ReportGenerator.tsx` 154行目 — AI生成テキスト（gunshiText）直接展開
- `src/components/analysis/ReportGenerator.tsx` 165〜168行目 — バックエンド生成レポートテキスト直接展開

特に `GunshiAdvice.tsx` はユーザーが任意のテキストを入力して Gemini に送信し、その返答を DOM に注入する経路があるため、プロンプトインジェクションを経由した格納型 XSS の可能性がある。

**推奨対策**: `DOMPurify` (`npm install dompurify @types/dompurify`) を導入し、全ての `dangerouslySetInnerHTML` の使用前にサニタイズする。

```tsx
import DOMPurify from 'dompurify';
dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(renderMarkdown(chat.text)) }}
```

### 5-2. 外部ドメインへのリクエスト（情報漏洩リスク）

**`src/app/agent-hub/page.tsx` 142, 343行目 — 外部CDNから画像・スクリプトをロード**

```tsx
// 143行目
<div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]" />

// 343行目（latestNovelセクション）
bg-[url('https://www.transparenttextures.com/patterns/old-map.png')]
```

外部ドメイン `transparenttextures.com` へリクエストが飛ぶため、ページロードのたびにユーザーのIPアドレスが第三者に送信される。また、`src/app/lease-kun/page.tsx` 251行目でも外部CDN (`api.dicebear.com`) からアバター画像をリクエストしている。本番環境では静的アセットに変換すべき。

### 5-3. クライアント側のバリデーション漏れ

**`src/app/register/page.tsx` `handleRegister` — サーバー側への依存**

フォームの送信前バリデーションは `targetId` のnullチェックのみで、金利や日付の妥当性チェックをクライアント側で行っていない。バックエンドでのバリデーションに依存しているが、UX向上のためクライアント側でも基本チェックを実施すべき。

### 5-4. 機密情報露出リスク（低）

**`src/components/layout/Sidebar.tsx` 188行目**

```tsx
<p className="text-xs font-black text-slate-200 truncate">User</p>
<p className="text-[10px] font-bold text-slate-500">Premium Plan</p>
```

ユーザー名がハードコードされている。本番環境では認証コンテキストから動的に取得すべき。

---

## 6. 改善優先度ランキング

### 高優先度（すぐ対応すべき）

- [ ] **XSS修正**: `GunshiAdvice.tsx`・`ReportGenerator.tsx` の `dangerouslySetInnerHTML` に `DOMPurify.sanitize()` を適用する（または `react-markdown` + `rehype-sanitize` に移行）
- [ ] **構文エラー修正**: `src/app/agent/page.tsx` および `src/app/batch/page.tsx` の二重波括弧 `{{` を単一波括弧 `{` に修正する
- [ ] **ゼロ除算修正**: `src/app/competitor/page.tsx` 241行目の `total_won / total_cases` にガード条件を追加する（`total_cases > 0 ? ... : 0`）
- [ ] **非nullアサーション修正**: `src/app/financial/page.tsx` 68行目の `find(...)!` に `?.` または optional chain を適用する

### 中優先度（次スプリントで対応）

- [ ] **APIレスポンス型定義**: `ScoringResult`、`DashboardStats`、`CaseRow` などのインターフェースを `src/types/index.ts` に追加し、`any` を排除する
- [ ] **parseFloat/NaN ガード**: `register/page.tsx` の数値入力を `SliderInput.tsx` のパターン（ローカル文字列ステート）に統一する
- [ ] **外部CDN依存の排除**: `agent-hub/page.tsx` の `transparenttextures.com` および `lease-kun/page.tsx` の `dicebear.com` への依存を静的アセットに変換する
- [ ] **GunshiAdvice の useEffect 依存配列修正**: `formData` オブジェクト全体ではなく必要なフィールドのみを依存配列に列挙する（`formData.nenshu`, `formData.op_profit` など）
- [ ] **FormGeneral.tsx の useEffect 修正**: 2つ目の `useEffect` の依存配列に `onChange` を追加する（または `useCallback` でメモ化する）
- [ ] **INDUSTRIES 定数の統一**: `debate/page.tsx` と `agent-hub/page.tsx` の重複定義を共通定数ファイルに切り出す
- [ ] **RealGraphs を実データと接続**: propsで `apiResult` を受け取り、実際のスコア成分（`score_base` など）をグラフに反映する
- [ ] **AdvancedAnalysis のモック表示を明示**: UIにモックデータである旨を明示するか、実際のAPIに接続する

### 低優先度（余裕があれば）

- [ ] **ScoreDAG のモバイル対応改善**: `min-w-[700px]` を廃止し、スマホでは縦配置に切り替える
- [ ] **goBack 関数の修正** (`lease-kun/page.tsx`): humor メッセージを考慮した正確な履歴削除ロジックに修正する
- [ ] **GunshiAdvice の sticky 高さ問題**: `h-[calc(100vh-8rem)]` を `h-[calc(100dvh-8rem)]` などモバイルアドレスバー対応の単位に変更する
- [ ] **agent-hub のポーリング最適化**: Page Visibility API でバックグラウンドタブ時にインターバルを停止する
- [ ] **MarkdownBlock コンポーネントの統一**: AIAnalysis・report/page・GunshiAdvice の3箇所のMarkdownパーサーを1つの共通コンポーネントに統合する
- [ ] **Sidebar のハードコードユーザー名削除**: 認証コンテキストから動的に取得する実装に変更する

---

## 7. 総評

フロントエンド全体として、スマホUI（`lease-kun`）・マルチエージェント討論・D3.js ネットワーク可視化など機能的には非常に豊富であり、Tailwind を活用したUIの完成度も高い。コンポーネント分割の方向性も `form/`, `analysis/`, `layout/` などで適切に整理されている。

**最大の問題点はセキュリティ**：AIが生成したテキストをサニタイズなしで `dangerouslySetInnerHTML` に渡している箇所が3箇所存在し、特に GunshiAdvice コンポーネントはユーザー入力→AI→DOM という経路でXSSが成立しうる。これは最優先で対処すべき。

**次点はモックデータの残存**：`RealGraphs` および `AdvancedAnalysis` コンポーネントが実際のAPIレスポンスと連動せず固定値を表示しており、ユーザーに誤った信頼を与える可能性がある。これらが「デモ表示」であることを明示するか、実データと接続する必要がある。

**TypeScript型安全性**については `any` 型が広範に使用されており、APIレスポンスの型定義が欠如しているため、バックエンドのAPIスキーマ変更がフロントエンドに伝播せず静的解析で検出できない状態にある。`src/types/index.ts` に各APIエンドポイントのレスポンス型を追加することを推奨する。
