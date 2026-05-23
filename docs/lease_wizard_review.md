# Lease Wizard コードレビュー

**対象**: `frontend/src/app/lease-kun/page.tsx` および関連ファイル  
**日付**: 2026-05-21  
**レビュアー**: Claude (コードレビュー)

---

## エグゼクティブサマリー

Lease Wizard は10ステップのチャット型審査入力UIとして概ね機能しているが、**戻るボタンのhistory不整合バグ**・**asset_nameのAPI送信漏れ**・**未使用フィールドの混在**など、修正しないとデータの正確性に直結する問題が複数ある。UX面ではalert()によるバリデーションがモバイル体験を著しく損なっており、優先改善を推奨する。

---

## 1. バグ・動作リスク

### 🔴 [高] `goBack()` のhistory削除が不整合

```typescript
const goBack = () => {
  setStep(s => s - 1);
  setHistory(prev => {
    const nw = [...prev];
    nw.pop(); nw.pop(); // 常に2件削除
    return nw;
  });
};
```

`handleNext` は `Math.random() < 0.3` の確率でhumorメッセージを挿入するため、  
追加されるメッセージ数は **2件（通常）または3件（humor追加時）** と変動する。  
しかし `goBack` は常に2件しか削除しないため、humor挿入時に戻ると  
**チャット履歴にゴミメッセージが残る**。

**修正方針**: 各ステップ遷移で追加したメッセージ数をstateで記録するか、  
stepとhistoryを完全に分離してstepベースでhistoryを再構築する。

---

### 🔴 [高] `asset_name` がAPIペイロードに含まれない

Step 2で物件を選択させているにもかかわらず、`submitScore` の `payload` に  
`asset_name` が含まれていない。

```typescript
// submitScore内のpayload（asset_nameが無い）
const payload = toThousandYenPayload({
  company_no: formData.company_no,
  company_name: formData.company_name,
  // ... asset_name が存在しない！
  nenshu: Number(formData.nenshu || 0),
  ...
```

物件カテゴリはスコアリングの資産評価に影響するため、**送信漏れはスコアの精度に直結**する。  
バックエンドの `ScoringRequest` に `asset_name` フィールドがあるか確認の上、追加が必要。

---

### 🔴 [高] `deal_source2` フィールドが完全に未使用

`formData` に `deal_source2: 'その他'` が定義されているが、  
UIのどのステップにも入力欄がなく、APIペイロードにも含まれない。  
**無用なstateを汚染しており、開発者が混乱する原因になる**。削除推奨。

---

### 🟡 [中] `submitScore` 後のstep状態が9のまま残る

API送信完了（loading=false）後にフォームが再表示され、  
ユーザーが意図せず**審査を2回実行**できてしまう。  
送信完了後はフォームをdisabledにするか、完了状態を管理するstateが必要。

---

### 🟡 [中] `total_assets` の `|| 1` が暗黙の値替換

```typescript
total_assets: Number(formData.total_assets || 1),
```

`total_assets` が空文字や0の場合に **1（千円）に書き換えて送信される**。  
Step 4でバリデーション済みのはずだが、空文字はバリデーションをすり抜ける  
（`Number('') === 0` だが alert の条件は `<= 0`、`Number('' || 1) === 1`）。  
0除算回避の意図であれば、スコアリング側で処理すべきで、UI側で値を書き換えるのは危険。

---

### 🟡 [中] HTMLの `required` 属性が機能しない

```tsx
<input type="text" inputMode="decimal" name="nenshu" ... required />
```

`type="text"` では `required` はブラウザのネイティブバリデーションを使わない実装に依存する。  
実際は `handleNext` 内の `alert()` でカバーしているが、フォームの `required` 属性は  
**見た目の意味しかなく、実際の防御にはなっていない**。

---

### 🟡 [中] エラーハンドリングが不十分

```typescript
} catch (e) {
  setHistory(prev => [...prev, {
    role: 'humor',
    text: 'エラー発生！APIサーバーが立ち上がっているか確認してね。'
  }]);
}
```

- エラーの種類（ネットワーク切断 / HTTPエラー / バリデーションエラー）が区別されない
- HTTPステータスコードや詳細メッセージが表示されない
- `e` の型が `unknown` なのに `instanceof` チェックなし
- AxiosError の `response.data.detail` を表示できない

---

### 🟢 [低] progress barが100%に到達しない

```typescript
style={{ width: `${((step) / STEPS.length) * 100}%` }}
```

Step 9（最終）時: `9 / 10 * 100 = 90%`。**審査完了後も90%止まり**。  
`((step + 1) / STEPS.length)` に修正すべき。

---

### 🟢 [低] チャット履歴の数値表示に単位変換の言及がない

UIは百万円単位で入力・表示するが、API送信時に千円単位に変換される。  
ユーザーがAPIレスポンスの数値を見た際に混乱する可能性がある。

---

## 2. UX・入力体験

### 🔴 [高] `alert()` によるバリデーションはモバイルで最悪

```typescript
if (!formData.nenshu || Number(formData.nenshu) <= 0) return alert("売上高は必須です！");
```

iOSのSafariでは `alert()` はモーダルダイアログとして表示され、UXが著しく損なわれる。  
またアクセシビリティ的にも問題がある。  

**改善案**: `errorMsg` stateを追加し、フォーム内にインラインでエラーを表示する。

```tsx
{errorMsg && (
  <p className="text-red-500 text-xs font-bold px-1">{errorMsg}</p>
)}
```

---

### 🟡 [中] Step 8（定性6項目）でスクロール不可視問題

```tsx
className="mb-4 space-y-3 max-h-[40vh] overflow-y-auto scrollbar-hide pb-2 px-1"
```

`scrollbar-hide` でスクロールバーを非表示にしているため、  
**ユーザーがスクロール可能と気づかない**。特に定性6項目 + テキストエリアのStep 8は  
コンテンツが多く、スクロールが必要なのに気づかれないリスクが高い。  
スクロール可能な領域であることを示す視覚的ヒント（フェードアウトグラデーション等）を推奨。

---

### 🟡 [中] `customer_type` がStep 1とStep 7で二重管理

Step 1に `customer_type`（既存先/新規先）のセレクトがある。  
しかしStep 7の表示内容にも `formData.customer_type` を使っているが、  
Step 7には修正UIがない。Step 1で設定したものがそのまま使われる設計だが、  
ユーザーは「Step 1でそれを設定した」と忘れている可能性が高い。  
どちらか一方のステップに集約を推奨。

---

### 🟡 [中] Step 5（経費）の入力項目が不明瞭

```tsx
<input ... placeholder="減価償却(資産・百万円)" />
<input ... placeholder="減価償却(経費・百万円)" />
<input ... placeholder="賃借料(資産・百万円)" />
<input ... placeholder="賃借料(経費・百万円)" />
```

「資産」と「経費」の違いが一般ユーザーには直感的でない。  
ラベルなしでplaceholderのみの説明は、入力後に内容が見えなくなるため不親切。  
ラベルを必ず添えること。

---

### 🟡 [中] 業種（中分類）がフリーテキスト

```tsx
<input type="text" name="industry_sub" ... placeholder="例: 06 総合工事業" />
```

業種大分類はselectだが、中分類はフリーテキストになっている。  
スペルミスや形式不正によるスコアリングエラーのリスクがある。  
大分類に連動したselectか、最低限の形式バリデーションが必要。

---

### 🟢 [低] `type="text"` の `step="0.1"` 属性が無意味

```tsx
<input type="text" inputMode="decimal" name="nenshu" step="0.1" ... />
```

`step` 属性は `type="number"` に対してのみ有効。`type="text"` では無視される。  
（以前のコミットで `type="number"` から `type="text"` に変更した際の取り残し。）  
削除推奨。

---

### 🟢 [低] 完了後の誘導が弱い

審査完了後、「詳細は📋審査・分析タブから確認してね！」と表示されるが、  
**タブへのリンクや遷移ボタンが存在しない**。ユーザーが自分でナビゲートする必要がある。

---

## 3. コード品質

### 🟡 [中] 未使用 import が4個

```typescript
import { Send, ArrowRight, ArrowLeft, Bot, Activity, CheckCircle, ChevronDown } from 'lucide-react';
import { API_BASE } from '../../lib/api';
```

実際に使われているのは `ArrowLeft`, `Activity`, `ChevronDown` のみ。  
`Send`, `ArrowRight`, `Bot`, `CheckCircle`, `API_BASE` は未使用。  
バンドルサイズへの影響は小さいが、コードの可読性を下げる。

---

### 🟡 [中] `formData` に明示的な型定義がない

```typescript
const [formData, setFormData] = useState({ ... });
```

型推論のみに依存しており、`handleChange` で:
```typescript
setFormData({ ...formData, [e.target.name]: e.target.value });
```
`e.target.name` が `formData` の有効なキーであることが型レベルで保証されない。  
`type FormData = { company_no: string; ... }` を明示的に定義すべき。

---

### 🟡 [中] コンポーネントが450行超の単一ファイル

10ステップのフォームUI + チャット履歴 + API送信ロジックが全て `page.tsx` に集約されている。  
最低限以下の分離を推奨：

- `WizardStep0.tsx` 〜 `WizardStep9.tsx` （または `steps/` ディレクトリ）
- `useWizardForm.ts` （formData + handleChange + バリデーション）
- `useWizardChat.ts` （history + ステップ遷移ロジック）

---

### 🟢 [低] スタイル定数がコンポーネント関数内で定義されている

```typescript
const sel = "w-full bg-slate-50 ...";
const inp = "w-full bg-slate-50 ...";
```

再レンダリングのたびに再生成される。コンポーネント外に移動すべき。

---

### 🟢 [低] マジックナンバーのhardcode

```typescript
if (Math.random() < 0.3) { ... }  // humor表示確率
```

定数として `HUMOR_PROBABILITY = 0.3` 等にまとめておくと変更・テストが容易になる。

---

### 🟢 [低] `img` に `alt` 属性なし

```tsx
<img src="https://api.dicebear.com/7.x/bottts/svg?..." />
```

アクセシビリティ違反（WCAG 2.1 1.1.1）。`alt="リースくんアバター"` 等を追加。  
また外部CDNへの直接参照はCSP（Content Security Policy）の問題になりうる。

---

## 4. 通常入力欄との一貫性

### 🟡 [中] `grade`（格付）の選択肢がWizardと通常フォームで異なる可能性

Wizardのgarde選択肢:
```
①1-3 (優良) / ②4-6 (標準) / ③要注意以下 / ④無格付
```

通常のStreamlitフォームの格付選択肢と完全に一致しているか要確認。  
スコアリングエンジン側の `category_config.py` や `scoring_core.py` が  
想定するenum値と一致しないと、**定性スコアが0扱いになるリスク**がある。

---

### 🟡 [中] `deal_source` の選択肢が2択のみ（通常フォームと乖離の可能性）

```tsx
<option>銀行紹介</option><option>その他</option>
```

通常のStreamlitフォームでは `deal_source` の選択肢が増えている可能性がある。  
backend の `ScoringRequest` が受け付ける値のリストと照合が必要。

---

### 🟡 [中] `contract_type` の選択肢が2択（一般/自動車）のみ

通常フォームに「ファイナンスリース」「オペレーティングリース」等があれば乖離。

---

### 🟢 [低] `intuition`（直感スコア）が通常フォームに存在するか不明

Wizardには `intuition: 1〜5` が存在しApiに送信されているが、  
通常のStreamlitフォームに同フィールドがあるか確認が必要。  
スコアリングエンジンがこの値をどう使っているかも含め、  
`scoring_full.py` での `intuition` のマッピングを確認推奨。

---

## 5. 改善優先度ランキング

| 優先度 | 項目 | 影響 |
|--------|------|------|
| 🔴 高 | `asset_name` がAPIに送信されない | スコアリング精度 |
| 🔴 高 | `goBack()` のhistory削除が不整合（humor時に3件必要） | データ不整合・UXバグ |
| 🔴 高 | `deal_source2` 未使用フィールドの混在 | コード品質・混乱 |
| 🔴 高 | `alert()` → インラインバリデーションへ変更 | モバイルUX |
| 🔴 高 | `grade` 等の選択肢とバックエンドenumの照合 | スコアリング正確性 |
| 🟡 中 | submitScore後の再送信防止 | データ品質 |
| 🟡 中 | `total_assets || 1` の暗黙書き換え除去 | データ正確性 |
| 🟡 中 | エラーハンドリング強化（詳細表示） | デバッグ性・UX |
| 🟡 中 | `formData` の型定義を明示 | 型安全性 |
| 🟡 中 | `customer_type` を一箇所に集約 | UX明確性 |
| 🟡 中 | Step 8スクロール可視化 | モバイルUX |
| 🟡 中 | 業種中分類をselectまたはバリデーション付きに | 入力品質 |
| 🟡 中 | 未使用import削除（Send, ArrowRight, Bot, CheckCircle, API_BASE） | コード品質 |
| 🟢 低 | progress barを`(step+1)/STEPS.length`に修正 | UX |
| 🟢 低 | `step="0.1"` 属性を削除（type="text"では無効） | コード品質 |
| 🟢 低 | 完了後の審査詳細タブへのリンクボタン追加 | UX |
| 🟢 低 | スタイル定数をコンポーネント外に移動 | パフォーマンス |
| 🟢 低 | `img` に `alt` 属性追加 | アクセシビリティ |
| 🟢 低 | コンポーネントをstep別に分割 | 保守性 |

---

## 付録: 関連ファイル一覧

| ファイル | 役割 |
|----------|------|
| `frontend/src/app/lease-kun/page.tsx` | Wizardメインコンポーネント（要分割） |
| `frontend/src/lib/scoringUnits.ts` | 百万円→千円変換ユーティリティ |
| `frontend/src/lib/api.ts` | axiosクライアント（API_BASEが未使用） |
| `frontend/src/types/index.ts` | 共通型定義 |
| `api/schemas.py` | FastAPI リクエスト/レスポンス Pydantic モデル |
| `api/main.py` | `/api/score/full` エンドポイント |
| `api/scoring_full.py` | スコアリングエンジン呼び出し |
