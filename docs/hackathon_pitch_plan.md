# DevOps × AI Agent Hackathon 2026 — 紫苑 ピッチ & ループ実演設計

紫苑（リース審査AIエージェント）のハッカソン提出に向けた、**3分ピッチ構成**・
**自己改善ループのライブ実演設計**・**審査員Q&A想定**をまとめる。

> ポジショニング（1文）
> **紫苑は、Cloud Run 上で「自分のDevOpsループを安全に閉じる」自己改善型リース審査AIエージェント。**
> 主役は「自己改善ループ × 本番安全分離」。リース審査は題材であって主役ではない。

要件充足（`docs/agent_platform_adapter.md` / `api/shion_agent.py` 参照）:
- 実行プロダクト: **Cloud Run**（必須①クリア）
- AI技術: **Gemini API ＋ ADK**（必須②は1つで良いところ2つクリア）

---

## ② 自己改善ループの「ライブ実演」設計

### 使う実物（新規開発は最小限）
| 資産 | 役割 |
|---|---|
| `scripts/demo/run_demo_pipeline.sh` | 24hサイクルを**約3分に圧縮**して実演。チャットログ→課題抽出→候補ランク→auto-apply→サマリ |
| `scripts/demo/demo_chat_logs.json` | ループ入力（リアルな問い合わせ会話）。本PRで **quick_ui 種**を1件追加 |
| `scripts/demo/show_before_after.py` | Before/After 表示 |
| `/loop-proof`（`scripts/build_loop_proof.py` → `reports/loop_proof.html`） | 台帳から applied / PR紐づけを機械集計した「ループが閉じた証拠」1画面 |
| `/devops`, `/cloudrun-return-review` | 本番安全分離（検疫→人間承認→昇格）の可視化 |

### なぜ「今日いい候補が出るのを待つ」実演は成立しないか
本番の auto-fix キューは基本的に空（`Ranked queue: 0`）。理由は
`.agents/skills/auto-improvement-pipeline/scripts/auto_fix_policy.py` の
**安全ゲート**が、曖昧・大粒・高リスクな候補を意図的に自動修正対象外（needs_review）に落とすため。

これは弱点ではなく **設計どおりの安全機構**。`_DENY_KEYWORDS` は
`スコアリング / 係数 / 閾値 / モデル / db / schema / 認証 / セキュリティ / リース期間 …`
を含み、**審査ロジック・データ・インフラに AI が勝手に触れない**ようにしている。
自動修正が許可されるのは「対象ファイルが明確な小規模UI/文言修正（quick_ui）」だけ。

### 決定論化のしかけ（本PRで実装）
`scripts/demo/demo_chat_logs.json` に、**FAQページの文言の誤字を指摘する会話**を1件追加した。
これは `auto_fix_policy` 上で次のように**決定論的に quick_ui へ分類される**（`tests/test_demo_quick_ui_seed.py` で検証）:

- `_DENY_KEYWORDS` に非該当（「文言」「faq」「誤字」「表示」は `_ALLOWED_KEYWORDS`）
- `_TARGET_INFERENCE_RULES` により対象 = `frontend/src/app/faq/page.tsx`（単一・安全な拡張子・非危険パス）
- → `auto_fix_allowed: True` / `category: quick_ui`

対照として、既存の「ブルドーザーのリース期間」ケースは **DENY**（「リース期間」が deny 語）。
= **「直せるもの（UI文言）は自動、触ってはいけないもの（審査ロジック）は人間へ」** を同じ画面で見せられる。

### ライブ実演フロー（3分・圧縮）
1. `bash scripts/demo/run_demo_pipeline.sh` を実行
2. チャットログ → 課題抽出 → dedupe/rank → **policy通過（quick_ui 1件が Ranked queue に乗る）**
3. auto-fix が **実 diff / PR** を生成 → **demo環境**へ適用
4. `demo_ledger.jsonl` が `needs_review → applied` に遷移
5. `/loop-proof` の applied / pr_traced が +1 → **ループが閉じた瞬間を画面で提示**
6. 本番安全分離を見せる: 本番DBには書かない（`CLOUDRUN_DATA_MODE=demo`）→ 帰還データは
   検疫→人間承認→昇格（`/cloudrun-return-review`, `scripts/promote_cloudrun_return_data.py`）

### 事前チェックリスト（ピッチ前に必ず）
- [ ] `bash scripts/demo/run_demo_pipeline.sh` が `.venv` + LLMキーで**最後まで通る**（※要ローカル検証）
- [ ] 追加した FAQ 種が抽出され、`Ranked queue: 1`（quick_ui）になる
- [ ] auto-fix が**本物の diff / PR** を出す（`No auto-fix candidates` のままだと逆効果）
- [ ] `/loop-proof` が実行後に数字更新される
- [ ] **フォールバック**: LLM抽出が不安定な場合に備え、事前構造化した候補を直接注入する手順、
      または成功録画を1本用意（ライブ事故対策）

---

## ③ 3分ピッチ構成

| 時間 | 話す | 画面 | 効く審査軸 |
|---|---|---|---|
| 0:00–0:25 **課題** | 「プロトタイプは作れるが本番運用に届かない」 | タイトル＋一言 | 課題の明確さ |
| 0:25–1:05 **つくる** | 紫苑=ADKエージェント。案件に応じ**自分でツールを選び裏取り**して判定 | `/lease-intelligence` で `search_cases`→`get_score_detail` を自律呼び出しするログ | 必然性・独創性・実装 |
| 1:05–2:05 **まわす（主役）** | 圧縮デモで課題抽出→auto-fix→PR→適用を**ライブで1周** | `run_demo_pipeline.sh` → `/loop-proof` +1 | **DevOps・実装** |
| 2:05–2:40 **とどける** | 本番DBは触らない。検疫→人間承認→昇格 | `/devops` 分離図＋`/cloudrun-return-review` | **本番品質** |
| 2:40–3:00 **締め** | 「動くもの≠届くもの。自分で直し続け、安全に届くAI」 | loop-proof 全景 | 全軸 |

### 焦点維持のルール
- 25+ページの機能は見せない。**loop-proof / 自律ツール / 検疫昇格の3画面だけ**。
- 人格・感情トレンドは主軸にしない（触れるなら10秒の彩り）。DevOpsの本気度を薄めない。
- 他の応募が手薄な「まわす・とどける」に時間を集中（1:05以降で1分半）。

---

## 審査員Q&A 想定

**Q. それ、ただのスクリプト（cron）では？ エージェントの必然性は？**
A. 候補の**選定と修正内容の生成**は LLM（Gemini / ADK）が自律的に行う。cron はトリガー、
policy はガードレールに過ぎない。審査時も紫苑は案件に応じて `search_cases` /
`get_score_detail` 等のツールを**自分で選んで**裏取りしてから判定する（`api/shion_agent.py`）。

**Q. 本番で AI が暴走して壊さない？**
A. 二重の安全機構。(1) `auto_fix_policy` が `スコアリング/係数/閾値/db/認証/リース期間…`
を含む候補を自動修正対象外にし、**UI文言の微修正だけ**を自動化。(2) Cloud Run 上のデータは
本番DBに直接書き戻さず、**検疫DB→人間承認→昇格**を経る（`/cloudrun-return-review`）。
「直せるものだけ自動、触ってはいけないものは人間へ」を実演で示せる。

**Q. 自己改善ループは本当に回っているのか（applied は人手PRでは）？**
A. 正直に: 現時点で自律 auto-fix が回るのは quick_ui（UI文言）に限定され、審査ロジック等の
大きな改善は人間レビューを経る設計。**「全自動」ではなく「安全な範囲だけ自律、危険は人間ゲート」**。
これは誇張ではなく、むしろ本番運用に耐える設計判断として提示する。

**Q. 必須技術は満たしているか？**
A. Cloud Run（実行）＋ Gemini API ＋ ADK（AI技術）。②は1つで良いところ2つ。
Gemini Enterprise Agent Platform は移行アダプタとして設計済み（`docs/agent_platform_adapter.md`）。

**Q. スケールするのか / コストは？**
A. Cloud Run はアイドル時ゼロスケール。推論は Gemini Flash（最安級）。自律調査ツールは
ローカルDB読み取り中心で追加API課金なし。

---

## 正直な前提・残作業
- `run_demo_pipeline.sh` が**現在も完走するか未検証**（`.venv`・LLMキー必要。CI/本セッションから実行不可）。
  ピッチ成否はここの通し確認に懸かる。
- FAQ 種の **quick_ui 分類は CI で検証済み**（`tests/test_demo_quick_ui_seed.py`）だが、
  その手前の **LLM 抽出（STEP1）が種を quick_ui 候補として拾うか**は要ローカル検証。
- 紫苑の自律ツール呼び出し（別PR #613）は **ADK 導入環境での動作確認**が1回必要。
