# フロントエンド規約

## TypeScript / Next.js

- **strict mode** 厳守（`as any` は原則禁止、止む無き場合は `// eslint-disable-next-line` コメント付き）
- `any` 型が必要な場合は `Record<string, unknown>` または適切な型ガードに置換
- API 呼び出しは `src/lib/api.ts` の `apiClient`（axios インスタンス）を使用
- グラフは **Recharts** (`recharts`)、3D は **Three.js** (`three`)、アイコンは **lucide-react**
- スタイルは **Tailwind CSS**（カスタム CSS は原則不使用）
- Server Component / Client Component の使い分け：インタラクションがあれば `"use client"`

## コンポーネント設計

- ページ固有コンポーネントは `app/<page>/page.tsx` 内に定義
- 複数ページで再利用するものは `components/analysis/` 等に切り出す
- Props は明示的な型定義必須（`type Props = {...}` または inline）

## ⚠️ eslint --fix は絶対禁止

過去に `eslint --fix` が UI コンポーネントや API エンドポイントを削除する事故が発生している。

```bash
# ✅ チェックのみ
cd frontend && npm run lint

# ❌ 絶対禁止
cd frontend && npm run lint:fix
cd frontend && npx eslint --fix
```

lint エラーが出たとき：
1. エラーメッセージを読んで **手動で修正** する
2. `no-unused-vars` 警告でもコードが使われているなら削除しない
3. 意図的な未使用変数は `_` プレフィックスを付ける（例: `_unused`）
4. 抑制が必要なら `// eslint-disable-next-line rule-name` を使う

`next build` は lint を実行しない（`ignoreDuringBuilds: true`）。

## TypeScript チェック（PR 前に必ず実行）

```bash
cd frontend && npx tsc --noEmit
```
