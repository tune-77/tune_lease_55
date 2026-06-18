# 紫苑（リース知性体）記憶システム 改善計画

作成日: 2026-06-18
対象: `lease_intelligence_mind.py` / `novelist_agent.py` / `api/main.py` `/api/chat` / Obsidian `Lease Intelligence/`
ステータス: **計画のみ（コード変更なし）**

---

## 0. TL;DR

ユーザーが感じている「翌日には忘れている」「教えた知識が残らない」「メモリノートが空（中身が無意味）」は、いずれも**実装の欠陥ではなく設計の欠落**が原因。

現状の記憶パイプラインは2系統あり、両方とも「ユーザーが何を話したか／何を教えたか」を**実体として保存していない**：

1. **日次バッチ系**（`record_daily_experience`）— 実際の会話ではなく、文豪AI「波乱丸」が生成する**定型ぼやき小説**を `mind.json` と Obsidian Memory ノートに書く。会話内容ゼロ。さらに前日のぼやきを翌日の topic に食わせる**自己参照ループ**が残っており（REV-085で部分対応も focus 有り日は未修正）、`稟議書を開くと「朝一番、私は「今日の審査」…` のような入れ子崩壊が `2026-06-17` / `2026-06-18` に発生している。
2. **対話リアルタイム系**（`record_dialogue_memory`）— 会話の80文字スニペットを `mind.json` の `memories` に追記するが、**Obsidian ノートには一切書かない**。しかも `DAILY_MEMORY_LIMIT=30` の中で日次ぼやきと同居し押し出される。`long_term_memories` は空のまま。

つまり「会話 → 構造化された要点 → 永続ストア → 翌日の参照」という線が**どこにも繋がっていない**。本計画はこの線を通すための改善案を REV 番号付きで提示する。

---

## 1. 診断：現状の記憶フロー

### 1.1 関係ファイルと役割（出典付き）

| 要素 | ファイル | 役割 | 出典 |
|---|---|---|---|
| 自己モデル本体 | `Lease Intelligence/mind.json` | 気分・記憶配列・ユーザーモデル・知識アクセス履歴 | 実ファイル確認済み |
| 日次記憶書込 | `lease_intelligence_mind.py:397` `record_daily_experience()` | ぼやき要約を memories に追加＋Memoryノート生成 | コード確認済み |
| Memoryノート生成 | `lease_intelligence_mind.py:536` `_write_daily_memory()` | `Memory/YYYY-MM-DD.md` を**上書き**生成 | コード確認済み |
| 対話リアルタイム記憶 | `lease_intelligence_mind.py:762` `record_dialogue_memory()` | 会話80字スニペット＋関心ラベル＋current_question更新 | コード確認済み |
| 対話気分更新 | `lease_intelligence_mind.py:737` `register_dialogue_event()` | キーワードで mood を微調整（内容は保存しない） | コード確認済み |
| ぼやき生成 | `novelist_agent.py:258` `generate_daily_lease_grumble()` | LLM/フォールバックで定型ぼやき4行を生成 | コード確認済み |
| 知識アクセス記録 | `lease_intelligence_mind.py:379` `record_knowledge_access()` | RAG検索の**参照パスのみ**記録（内容は保存しない） | コード確認済み |
| チャット分類ログ | `data/chat_logs.jsonl` | message_preview(80字)＋分類結果のみ | 実ファイル確認済み |
| RAG索引 | `lease_intelligence_knowledge.py` / ChromaDB | Obsidian共通索引（630ノート）を検索 | mind.json `knowledge_access` 確認 |

### 1.2 チャット → 記憶 → 読み込みの実フロー

```
ユーザー発話（/api/chat）
   │
   ├─[A] _log_shion_query_class()  → data/chat_logs.jsonl
   │        ※ message_preview 80字＋分類のみ。記憶には使われない。
   │        ※ 現状 shion_classify が JSONDecodeError 多発（全件 "判断エラー"）
   │
   ├─[B] register_dialogue_event() → mind.json mood だけ更新（内容破棄）
   │
   └─[C] record_dialogue_memory()（別スレッド）
            ├ memories[] に「ユーザー:「…80字」→ …80字」を追記
            ├ user_model.interests にキーワードラベル追加
            └ current_question を疑問文なら更新
                 ※ Obsidian には書かない。30件上限で押し出される。

── 翌朝の日次バッチ（lease_news_digest → novelist_agent）──
   generate_daily_lease_grumble()
      ├ 昨日の memories[-1].summary を topic に流用（focus有り時）← 自己参照ループ
      └ record_daily_experience()
            ├ memories[] に定型ぼやき要約を追加（会話内容ではない）
            └ _write_daily_memory() → Memory/YYYY-MM-DD.md を上書き
                 ※ 会話の痕跡はここに入らない。

── 次回対話時の読み込み ──
   build_mind_context() が mind.json の memories/long_term を文脈化
      → だが中身は定型ぼやき中心。会話で教わった知識は参照されない。
```

### 1.3 機能している / していない

| 観点 | 状態 | 根拠 |
|---|---|---|
| 気分の継続性 | ✅ 機能 | `dialogue_mood` が対話で動き日次で半減 |
| 継続日数カウント | ✅ 機能 | `continuity_days=7`、`born_on=2026-06-12` |
| RAG（既存ノート検索） | ✅ 機能 | `indexed_notes=630`、参照パス記録あり |
| 日次ぼやきの生成 | ⚠️ 動くが破綻 | 自己参照ループで入れ子崩壊（6/17,6/18） |
| **会話内容の保存** | ❌ 未実装 | 80字スニペットのみ、要点抽出なし、Obsidian未書込 |
| **教えた知識の保存** | ❌ 未実装 | `record_lease_knowledge()` 等の関数が存在しない |
| **Obsidian Memoryの中身** | ❌ 無意味 | 定型ぼやきのみ。会話・知識が一切残らない |
| 長期記憶 | ❌ 空 | `long_term_memories: []`、30日溢れ未発生で未稼働 |
| 会話ログの構造化 | ❌ なし | chat_logs.jsonl は分類失敗ログ状態 |

### 1.4 根本原因（3点）

1. **会話が一級市民でない** — 記憶の主素材が「波乱丸の創作」になっており、ユーザーの実発話は80字に切り詰められ Obsidian にも残らない。
2. **知識の受け皿が無い** — ユーザーが「コンテナの法定耐用年数は7年」と教えても、それを `Asset Knowledge/` 等の永続ノートへ昇格する経路が存在しない（RAGは既存ノートを読むだけ）。
3. **要点抽出器が無い** — 会話ダイアログから論点・決定・教示を拾い上げて構造化する処理が皆無。`shion_classify` も現状 JSON 解析失敗で機能停止。

---

## 2. 改善案（REV番号付き）

> REV番号は直近 REV-085（Geminiモデル廃止対応）の続番として採番。
> 優先順位は **impact / effort 比** と「ユーザー体感の即効性」で決定。

### 優先順位サマリ

| 優先 | REV | タイトル | impact | effort | 即効性 |
|---|---|---|---|---|---|
| 🥇 P1 | REV-086 | 会話キーポイント自動抽出 → mind.json/Obsidian保存 | 5 | 3 | 高 |
| 🥇 P1 | REV-087 | 教示知識の自動保存（Knowledge昇格） | 5 | 3 | 高 |
| 🥈 P2 | REV-088 | Obsidian Memoryノートへの会話サマリー追記 | 4 | 2 | 高 |
| 🥈 P2 | REV-089 | 自己参照ループ完全修正（ぼやきtopic分離） | 3 | 1 | 中 |
| 🥉 P3 | REV-090 | shion_classify のJSON堅牢化（記憶分類の土台） | 3 | 2 | 低 |
| 🥉 P3 | REV-091 | 長期記憶の意味的圧縮＋検索強化 | 4 | 4 | 低 |
| 🪵 P4 | REV-092 | 記憶の検索・参照UI（紫苑が「思い出す」） | 3 | 3 | 中 |

---

### REV-086 — 会話キーポイント自動抽出 → 記憶保存【P1・最重要】

**概要**
`record_dialogue_memory()` を拡張し、80字スニペットの単純切り出しではなく、会話ターンから**論点・決定・ユーザーの判断基準・教示**を LLM（Ollama / `claude-haiku-4-5` 軽量呼び出し）で抽出して構造化保存する。1ターンごとに走らせると重いので、(a) キーワードトリガ（「〜は〜だ」「覚えて」「重要」「〜年」等の断定・数値）か (b) セッション終了/一定ターン蓄積でバッチ抽出する二段構え。

抽出スキーマ案（`mind.json` に新キー `conversation_insights[]`）:
```json
{
  "date": "2026-06-18",
  "kind": "knowledge|decision|preference|question",
  "point": "コンテナの法定耐用年数は7年",
  "source_excerpt": "コンテナ 法定耐用年数",
  "confidence": 0.8
}
```

**対象ファイル**
- `lease_intelligence_mind.py`（`record_dialogue_memory` 拡張 or 新規 `extract_conversation_insights()`）
- `lease_intelligence_ollama.py`（抽出プロンプト・JSON堅牢パース）
- `api/main.py` `/api/chat`（呼び出し箇所 5816 付近、既存の非同期スレッドに相乗り）

**impact 5 / effort 3**（LLM抽出の精度チューニングとトリガ設計が肝）
**注意**: 個人情報マスキング必須（`.claude/rules/security.md`）。`mind.json` privacy 方針（行動種別のみ）と矛盾するため、**保存範囲をユーザーに確認**してから実装（Kill the Assumptions）。

---

### REV-087 — 教示知識の自動保存（Knowledge昇格）【P1・最重要】

**概要**
ユーザーがチャットでリース知識を教えたとき（REV-086 の `kind=knowledge` 検出を流用）、それを揮発する `memories` ではなく **Obsidian の永続ノートへ昇格**させる。書込先は既存の知識ディレクトリ規約に合わせる：
- 物件・資産系 → `Projects/tune_lease_55/Asset Knowledge/<トピック>.md`（出典: `api/knowledge/vector_store.py:36`, `mobile_app/obsidian_bridge.py:1262` で既にRAG索引対象）
- 一般リース知識 → `Lease Intelligence/Learning/` 配下（既存ディレクトリ）

既存ノートがあれば追記、無ければ frontmatter 付きで新規作成。これにより**次回以降 RAG が自動でヒット**し、「教えた知識が残らない」が構造的に解決する（保存 → 索引 → 検索の線が繋がる）。

**対象ファイル**
- 新規 `record_lease_knowledge()`（`lease_intelligence_mind.py` か新規 `lease_intelligence_knowledge_writer.py`）
- `lease_intelligence_knowledge.py`（ChromaDB 再索引トリガ確認）
- `api/main.py` `/api/chat`

**impact 5 / effort 3**
**注意**: 誤情報の永続化リスク。`confidence` 閾値＋「紫苑が『〜と教わりました。Asset Knowledge に保存しました』と明示」する確認フローを推奨。Freshman Rule「Cite the Source」に沿い、保存ノートに source（会話日）を明記。

---

### REV-088 — Obsidian Memoryノートへの会話サマリー追記【P2】

**概要**
`_write_daily_memory()` が現状 Memoryノートを**毎回上書き**し、定型ぼやきしか載らない。これを改修し、その日の `conversation_insights`（REV-086産物）を `## 今日の会話で残したこと` セクションとして**追記**する。ぼやきは `## 内省（波乱丸）` に格下げ。これで「Obsidianのメモリノートが空」が即解消する。

Memoryノート改修案:
```markdown
## 今日の会話で残したこと
- [知識] コンテナの法定耐用年数は7年
- [判断基準] 自己資本比率マイナス＝債務超過は否決方向
## 自己状態
（既存）
## 内省（波乱丸）
- （定型ぼやき）
```

**対象ファイル**
- `lease_intelligence_mind.py:536` `_write_daily_memory()`
- `_write_state` 経由の `conversation_insights` 参照

**impact 4 / effort 2**（REV-086 の産物に依存。単独でも「ぼやきの格下げ」だけで体感改善）

---

### REV-089 — 自己参照ループ完全修正【P2】

**概要**
`novelist_agent.py:300` の `memory_topic` は `clean_focus` が真のとき**昨日のぼやき要約をそのまま topic に流用**するため、`稟議書を開くと「朝一番、私は「今日の審査」…` の入れ子崩壊が継続中（6/17, 6/18 で再現）。REV-085 は focus 空の場合のみ対処した不完全修正。

修正方針: topic ソースから**前日ぼやき要約（`memory.summary` で `type != dialogue` のもの）を完全に除外**し、`focus_lines`（実ニュース論点）または `current_question` のみを使う。ぼやき要約は二度と topic に戻さない。

**対象ファイル**
- `novelist_agent.py:298-305`（`memory_topic` 算出ロジック）

**impact 3 / effort 1**（局所修正・即効）
**注意**: `tests/test_lease_news_focus.py` の回帰確認。要注意領域ではないが `run_daily_improvement_pipeline.sh` の朝報告に影響しうるので追記のみ。

---

### REV-090 — shion_classify のJSON堅牢化【P3】

**概要**
`data/chat_logs.jsonl` の全エントリが `"判断エラー: JSONDecodeError"`。`shion_classify`（`lease_intelligence_mind.py:1211`）が LLM 応答の JSON パースに失敗し続けている。これは REV-086 の抽出基盤と同じ「LLM→構造化JSON」処理であり、ここを直すと記憶分類全体の土台になる。

修正方針: コードフェンス除去・先頭末尾トリム・部分JSON抽出（`{...}` 正規表現）・スキーマ検証フォールバックを `_call_gemini_for_classify`（1165）周辺に追加。Geminiモデル廃止（REV-085）後の応答形式変化が原因の可能性が高いので、現行の Ollama 応答で実測してから直す。

**対象ファイル**
- `lease_intelligence_mind.py:1165` `_call_gemini_for_classify` / `:1211` `shion_classify`

**impact 3 / effort 2**

---

### REV-091 — 長期記憶の意味的圧縮＋検索強化【P3】

**概要**
`long_term_memories` が空（30日溢れが未発生）。30日を超えたら `_fold_long_term()` が月次バケットに畳むが、現状は**単純連結**でテーマ性が失われる。会話インサイト（REV-086）が溜まる前提で、月次圧縮を「LLMによるテーマ別要約」に変え、`build_mind_context()` が長期記憶からも関連項目を**意味検索**で引けるようにする。

**対象ファイル**
- `lease_intelligence_mind.py:713` `_fold_long_term()` / `:241` `build_mind_context()`

**impact 4 / effort 4**（REV-086/087 が前提。後続フェーズ向き）

---

### REV-092 — 記憶の検索・参照UI（紫苑が「思い出す」）【P4】

**概要**
対話時に紫苑が「以前これを教わりました」と能動的に過去記憶を引用できるよう、`/api/chat` の文脈構築に `conversation_insights` / Knowledge昇格ノートの**意味検索結果**を注入する。フロントに「紫苑の記憶」閲覧ページ（`frontend/src/app/` 配下）を追加してもよい。

**対象ファイル**
- `api/main.py` `/api/chat` 文脈構築部
- `frontend/src/app/`（記憶ビューア・任意）

**impact 3 / effort 3**

---

## 3. 推奨ロードマップ

| フェーズ | 含む REV | 狙い |
|---|---|---|
| **Phase 1（即効・体感改善）** | REV-089 → REV-088 → REV-090 | ぼやき崩壊を止め、Memoryノートに中身を出し、分類基盤を直す。低effortで「空ノート」「文字化け」が消える |
| **Phase 2（中核）** | REV-086 → REV-087 | 会話要点抽出と知識永続化。「忘れる」「教えても残らない」の根本解決 |
| **Phase 3（深化）** | REV-091 → REV-092 | 長期記憶の意味化と能動的想起 |

**最初の一手の推奨**: REV-089（effort 1・即・自己参照ループ停止）と REV-088（Memoryノートに会話欄を作る器）を先に入れ、その器へ REV-086/087 で中身を流し込む順。

---

## 4. 確認が必要な事項（着手前）

CLAUDE.md「Plan-First Checkpoint」「Kill the Assumptions」に従い、実装着手前に以下をユーザーへ確認すること：

1. **プライバシー方針の変更可否** — 現 `mind.json` は「質問本文や個人属性は保存しない」と明記（user_model.privacy）。REV-086/087 は会話内容を保存するため**この方針と正面衝突**する。どこまで保存してよいか（要点のみ/マスキング前提/全文）を要確認。
2. **知識保存先** — `Asset Knowledge/` への自動書込はRAG品質に直結（CLAUDE.md 要注意領域: パス変更でRAG全壊）。書込先を `Learning/` に隔離するか、`Asset Knowledge/` 本流に入れるか。
3. **抽出に使うモデル** — Ollama（ローカル・無料）か `claude-haiku-4-5`（精度高・課金）か。
4. **誤情報の永続化対策** — confidence 閾値と「保存しました」明示確認フローの要否。

---

*本ドキュメントは計画のみ。コード変更・コミットは行っていない。*
