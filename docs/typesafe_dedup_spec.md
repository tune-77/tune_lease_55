# TypeSafe (Jev) 改善候補 重複判定 仕様書 — Codex 向け

作成: 2026-09-20 / 依頼元: Claude Code セッション
位置づけ: **Phase 1 = 隔離実装とオフライン実測のみ**。日次パイプラインへの結線は Phase 2 として別途承認を取る。

関連: `docs/typesafe_rag_pilot.md`（先行する RAG パイロット、参照実装）
      `docs/typesafe_eval_spec.md`（別テーマ。本仕様とは独立）

## 1. 目的

改善候補（REV 候補）の重複判定を、文字 bigram の Jaccard 類似度から Jev の意味判断へ置き換えた場合に
**日本語の言い換え重複をどれだけ拾えるか** を、既存の改善候補ログで実測する。

## 2. 非目標（Phase 1 で絶対に触らない）

- `scripts/extract_obsidian_improvements.py` 本体（**1 行も変更しない**）
- `run_daily_improvement_pipeline.sh` およびその呼び出しステップ
- 既存の `deduplicate_improvements()` の挙動
- 後段の AI 統合（`_load_consolidator()` / Gemini 経路、`extract_obsidian_improvements.py:741` 以降）
- `ledger.jsonl` / `api/rule_engine/ledger_rules.json` の書き込み
- `scoring_core.py` / `constants.py` / `api/` / `frontend/`

Phase 1 の成果物は **新規ファイルのみ**。既存ファイルへの差分をゼロに保つこと。

## 3. 現状分析（調査済み・再調査不要）

対象は `scripts/extract_obsidian_improvements.py` の `deduplicate_improvements()`（L565）。
重複判定は以下 4 段の逐次ヒューリスティックで、上から順に評価し最初に当たった時点で重複確定。

| 段 | 判定 | 位置 |
|---|---|---|
| 1 | タイトル完全一致 / 先頭 40 文字一致 | L595-598 |
| 2 | 一方が他方に含まれる（8 文字以上、`_SUBSET_MIN_LEN`） | L600-605 |
| 3 | テーマグループ一致（`_in_same_theme`） | L607-609 |
| 4 | 2-gram 文字 Jaccard ≥ `_JACCARD_THRESHOLD`（0.55） | L611-614 |

補助定義: `_jaccard_similarity()` L533-544 / `_THEME_GROUPS` L547-559 / `_JACCARD_THRESHOLD` L562。

**診断**: 段 4 は文字の重なりしか見ないため、語彙が異なる同一課題を取りこぼす。
例として「PR タイトルに REV 番号が入らず台帳が更新されない」と
「`cleanup_improvement_reviews.py --apply` が空振りする」は同一の根本課題だが共有 bigram がほぼ無い。
閾値を下げると今度は別課題を誤って併合するため、単一閾値では解けない。

段 3 の `_THEME_GROUPS` は 13 グループのキーワード表を手作業で維持しており、
この構造的限界に対する対症療法として増え続けている。

**仮説**: 段 1〜3 で確定しなかったペアのうち、Jaccard が中間帯にあるものを Jev の Noul で判定すれば、
語彙非依存の重複を追加検出できる。
**反証条件**: 中間帯で Jev が追加検出した重複が実測でほぼゼロ、または人手判定との不一致が過半なら仮説は棄却。

## 4. 設計方針

### 4.1 カスケード（既存ロジックを捨てない）

Jev を全ペアに投げると O(n²) でコールが膨らむ。既存 4 段を**前段フィルタとして残す**。

- 段 1〜3 に当たった → 従来どおり即「重複」。**Jev を呼ばない**
- Jaccard ≥ `_JACCARD_THRESHOLD`(0.55) → 従来どおり即「重複」。**Jev を呼ばない**
- Jaccard < `GRAY_LOW`(初期値 0.30) → 従来どおり即「別件」。**Jev を呼ばない**
- `GRAY_LOW` ≤ Jaccard < 0.55 の**灰色帯のみ** Jev に判定させる

これにより Jev が完全に落ちても、灰色帯を「別件」に倒せば現行と同一の出力に戻る（fail-open）。

`GRAY_LOW` の初期値 0.30 は仮置きであり普遍則ではない。§8 の実測で確定させること。

### 4.2 バッチ（1 リクエストに集約）

灰色帯のペアは互いに独立した判断なので、**全ペアを 1 回の POST に独立 question として詰める**。
`typesafe_rag_guard.py` の `build_passage_request()`（L128-145）と同じ構造を踏襲する。
ペアごとに 1 リクエストを投げる実装にしないこと。

### 4.3 参照実装

以下は `typesafe_rag_guard.py` に既に確立されているため、**再発明せず同じ型に合わせる**。

| 関心事 | 参照位置 |
|---|---|
| エンドポイント / 認証ヘッダ / タイムアウト | `_default_request()` L149-163 |
| Noul 応答の検証（型・範囲 0〜1・欠落） | `_noul()` L166-178 |
| 閾値を独立関数に切り出す形 | `route_passage()` L180-194 |
| `request_fn` 注入によるテスト用差し替え | `judge_passages()` L196-201 |
| 有効化ガード（env + APIキー必須） | `typesafe_rag_enabled()` L63-66 |

## 5. 環境

```bash
pip install typesafe-sdk    # 正規パッケージ。'typesafe' / 'typesafe-ai' は別物なので使わない
```

認証は既存パイロットと同じ経路を使う（新しい変数を増やさない）:

- `TYPESAFE_API_KEY`、または macOS では `TYPESAFE_API_KEYCHAIN_SERVICE`
- モデル既定値は `jev-latest`（`TYPESAFE_MODEL` で上書き可）

有効化フラグは本機能専用に分ける: `TYPESAFE_DEDUP_ENABLED`（既定 0 = 無効）。
RAG 側の `TYPESAFE_RAG_ENABLED` を流用しないこと。片方だけ止められる必要がある。

実装前に `https://docs.typesafe.ai/llms.txt` から Noul プリミティブと API のページを読むこと。

## 6. 成果物

```
typesafe_dedup_guard.py              # 新規モジュール（リポジトリ直下、rag_guard と同階層）
tests/test_typesafe_dedup_guard.py   # 新規テスト
experiments/typesafe_dedup/
├── README.md                        # 実測手順と結果
├── measure.py                       # 既存ログに対するオフライン実測
└── results/                         # 実測出力（コミット可、§10 の確認後）
```

## 7. Step 1 — `typesafe_dedup_guard.py`

公開する関数は最小限に保つ。

```python
GRAY_LOW = 0.30          # 灰色帯の下限（§8 で実測して確定）
GRAY_HIGH = 0.55         # extract_obsidian_improvements._JACCARD_THRESHOLD と一致させる
SAME_ISSUE_MIN = 0.60    # route_pair() の 2 分岐の既定。低レベル API として据え置き

def typesafe_dedup_enabled(environ=None) -> bool: ...
def select_gray_pairs(pairs, *, low=GRAY_LOW, high=GRAY_HIGH) -> list[tuple[int, int]]: ...
def build_pair_request(pairs, *, model=DEFAULT_MODEL) -> dict: ...
def route_pair(same_issue: float, *, threshold=SAME_ISSUE_MIN) -> str: ...
def judge_pairs(pairs, *, request_fn=None, model=None) -> tuple[list[dict], dict]: ...
```

運用ポリシー（§8 で確定）は 3 分岐で、`experiments/typesafe_dedup/measure.py` の
`classify_judged_pair()` が持つ。`route_pair()` は生の確率を単一閾値で二分する
低レベル API として残し、`REVIEW_LOW = 0.40` / `REVIEW_HIGH = 0.75` を運用側に置く。
推論を再実行せず境界だけ掃き直せる状態を保つための分離。

| `same_issue` | routing |
|---|---|
| ≥ 0.75 | `duplicate`（自動併合） |
| 0.40 〜 0.75 未満 | `needs_human_review` |
| < 0.40 | `distinct` |

質問は 1 ペアにつき Noul 1 問。`state` にはタイトルと理由のみを入れる。

```python
{
  "state": {
    "pairs": [
      {"a_title": "...", "a_reason": "...", "b_title": "...", "b_reason": "..."},
    ]
  },
  "model": "jev-latest",
  "questions": {
    "pair0_same_issue": {
      "type": "noul",
      "instructions": "`pairs[0].a_title` と `pairs[0].b_title` は同一の根本課題に対する改善案か？",
      "criteria": {
        "true":  "対象箇所と目的が同じで、片方を実装すればもう片方も解消される。",
        "false": "対象箇所または目的が異なり、両方を別々に実装する必要がある。"
      }
    }
  }
}
```

要件:

- `TYPESAFE_DEDUP_ENABLED` が真かつ API キーが解決できる時だけ外部通信する
- 応答が不正・タイムアウト・例外のいずれでも `TypeSafeDedupError` に正規化し、**呼び出し側が現行動作へ戻せる**こと
- 例外を握り潰して空リストを返さないこと（`load_all_cases` の事故と同型の失敗になる。`git log` の REV-000 参照）
- 文字列長は上限を設ける（title 200 / reason 600 程度）

### Phase 1 での結線範囲

`typesafe_dedup_guard.py` は**どこからも import されない**。`extract_obsidian_improvements.py` への
結線は Phase 2。Phase 1 で呼ぶのは `experiments/typesafe_dedup/measure.py` のみ。

## 8. Step 2 — オフライン実測

入力は既存の改善候補ログ（`~/Library/Logs/tunelease/improvement_YYYYMMDD.log`）。
`_parse_improvements()` 相当でタイトルと理由を取り出し、直近 30 日分を対象とする。

手順:

1. 全ペアの Jaccard を計算し、段 1〜3 で確定するペアを除外する
2. 残りを Jaccard 値で帯に分け、**灰色帯の幅を 0.20/0.25/0.30 の 3 通り**で試す
3. 灰色帯のペアを Jev に投げ、`same_issue` 確率を記録する
4. **人手で正解付けする**。灰色帯のペア数が 50 を超える場合は無作為 50 件に絞ってよい
5. `SAME_ISSUE_MIN` を 0.5 / 0.6 / 0.7 で振り、3 と 4 を突き合わせる

### TODO(human) — 誤判定の非対称性をどちらに倒すか

この判定には 2 種類の誤りがあり、コストが釣り合っていない。

- **誤併合**（別件を重複と判定）→ 改善候補が 1 件消える。気づきにくい
- **誤分割**（同一課題を別件と判定）→ 重複 REV が採番される。台帳に残るので後から気づける

`SAME_ISSUE_MIN` と灰色帯の扱いは、このトレードオフをどちらへ倒すかで決まる。

**回答（2026-09-20、運用者判断）**:

1. **誤併合をより強く避ける**。台帳に痕跡が残らず気づけないため
2. 灰色帯で Jev が使えなかった場合は「別件」へ倒す（§4.1 の fail-open と一致。
   `judge_pairs_if_enabled()` が実装済み）
3. **中間帯 0.40〜0.75 は自動処理せず人手レビューへ回す**

確定した 3 分岐（`measure.py` の `classify_judged_pair()`）:

| `same_issue` | routing |
|---|---|
| ≥ 0.75 | `duplicate`（自動併合） |
| 0.40 〜 0.75 未満 | `needs_human_review` |
| < 0.40 | `distinct` |

`typesafe_dedup_guard.route_pair()` の 2 分岐（`SAME_ISSUE_MIN` 単独）は
生の確率を扱う低レベル API として残す。運用ポリシーは `classify_judged_pair()`
側に置き、推論を再実行せず境界だけ掃き直せる状態を保つ。

（ここに方針を記入 → `SAME_ISSUE_MIN` の確定値と既定倒し方向を §7 に反映する）

## 9. 指標

README.md に以下を実測値で記載する。

1. 段 1〜3 で確定したペア数 / 全ペア数
2. 灰色帯に入ったペア数（帯幅 3 通りそれぞれ）＝ **Jev へのコール数**
3. 灰色帯で Jev が「重複」と判定した件数のうち、人手正解と一致した割合
4. 現行ロジック（Jaccard 0.55 のみ）が取りこぼしていた重複の実数
5. 1 リクエストあたりのレイテンシと usage トークン数

### この実測で測れないこと（README に明記すること）

- **全体の重複検出率**。人手正解を付けるのは灰色帯のみであり、段 1〜3 や Jaccard ≥ 0.55 で
  確定したペアの誤併合は検証対象外
- **日次パイプライン全体への影響**。Phase 1 は結線しないため、朝報告の内容は変化しない

## 10. データ取扱い

送信されるのは改善候補のタイトルと理由のみで、審査案件データ・財務数値・個人情報は含まない想定。

ただし改善候補は Obsidian の AI チャットログ由来であり、**実在の企業名や案件番号が混入しうる**。
Step 2 の実行前に、灰色帯へ送るペアの全文を目視し、混入があれば当該ペアを除外すること。
この確認を経ていない状態で外部送信しないこと。

`docs/typesafe_rag_pilot.md` の制約（機密審査案件では retention 条件の承認まで無効）は本機能にも適用する。

## 11. 受け入れ条件

- [x] `scripts/extract_obsidian_improvements.py` と `run_daily_improvement_pipeline.sh` に差分がない
- [x] `typesafe_dedup_guard.py` がどの既存モジュールからも import されていない（参照は `experiments/` と `tests/` のみ）
- [x] `TYPESAFE_DEDUP_ENABLED` が未設定の状態で外部通信が発生しない
- [x] テストが `request_fn` 注入のみで完結し、TypeSafe に接続しない（16 passed）
- [x] 不正応答・タイムアウトで例外が送出され、空リストへ退化しない
- [x] §8 の TODO(human) がユーザーによって埋められ、確定値が §7 に反映されている
- [x] Step 2 の目視確認（§10）を実施した旨が README.md に記載されている ← 2026-09-20 実施、帯 0.30-0.55 の 33 ペア全文を確認。機密混入 0 件、パース不良 5 ペアを除外
- [ ] README.md に §9 の 5 指標と、§3 の仮説に対する判定（支持 / 棄却）がある ← 指標 1・2 と仮説判定は記載済み。指標 3〜5 は `--send` 後
- [x] `cd frontend && npx tsc --noEmit` は不要（フロント変更なし）

## 12. 注意

- 結果がどうであれ、`extract_obsidian_improvements.py` への結線は **Phase 2 として別途承認**が必要
- `deduplicate_improvements()` の段 1〜3 を Jev で置き換えようとしないこと。安価で決定的な判定を
  有料の推論に置き換える理由がない
- `_THEME_GROUPS` は Phase 1 では削除しない。Jev が実測で上回った場合に Phase 2 で縮小を検討する
- 閾値は本リポジトリのデータで決める。ドキュメントの例値をそのまま採用しないこと
