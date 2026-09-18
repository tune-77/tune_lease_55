"""日次パイプラインスクリプトの「無音失敗」再発を防ぐラチェットガードテスト。

背景: scripts/sync_memory_from_daily.py が `## Promotable Items` 見出しの
書式ドリフトで3週間以上、自動昇格0件のまま無音停止していた
(PR #1060)。原因を横展開監査したところ、run_daily_improvement_core.sh /
post.sh が呼ぶスクリプトのうち63本が同型の構造
（main()にreturn文が無い、または常にreturn 0）を持っており、そのうち
32本を個別に精査してexit 1での異常検知を追加した（PR #1061〜#1064）。

この監査は一度きりでは再発を防げない。新しいスクリプトを
run_daily_improvement_core.sh / post.sh へ追加するたびに、同じ穴が
また空く。本テストはその穴を機械的に検知するラチェット:

- main()の戻り値が分岐していない（exit 1等の異常系が無い）
- かつ raise SystemExit / sys.exit によるファイル内の実質的な失敗
  シグナルも無い

スクリプトを新規追加/変更した際にこのテストへ引っかかったら、
「対象はあるのに0件」を検知する分岐を足すか、それが不要な設計判断
（0件が正常な業務状態、レポート専用でパイプラインを止めない、等）
であれば _ALLOWED_ALWAYS_ZERO へ理由付きで登録すること。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PIPELINE_SCRIPT_SOURCES = (
    "scripts/run_daily_improvement_core.sh",
    "scripts/run_daily_improvement_post.sh",
)

# 「常にexit 0（または失敗シグナル無し）」を意図的に許可するスクリプトと
# 理由。ここに載せる = 「対象0件とドリフトを区別できない／既に別経路で
# 失敗を報告済み／レポート専用でパイプラインを止めない」という設計判断
# を明文化すること。詳細は各PRの説明・コミットメッセージを参照。
_ALLOWED_ALWAYS_ZERO: dict[str, str] = {
    "scripts/analyze_shion_pm_quality.py": (
        "outcome同期は冪等設計。0件は自作スキーマ内の正当な業務状態で、"
        "ドリフトと区別できる信頼できる条件が無い。"
    ),
    "scripts/attach_shion_self_proposals_to_report.py": (
        "6種の異種JSONLソースに依存し、いずれも自然に長期間0件になりうる。"
        "全ソース横断で誤検知なく判定するには個別カウント計装が必要で"
        "過剰実装。"
    ),
    "scripts/auto_approve_safe_recipes.py": (
        "shion自身が書く安定スキーマ（明示キー）にのみ依存。"
        "0件承認は日常的に正常な状態。"
    ),
    "scripts/build_agent_action_ledger_report.py": (
        "log_action()がバリデーションして書く自己所有スキーマ。総件数0は"
        "紫苑が単に行動しなかった日として普通に起こる正常状態。"
    ),
    "scripts/build_experience_flywheel_report.py": (
        "4種の独立したJSONLフィードバックソースに依存。全ソース同時に"
        "長期間0件になるのは十分あり得る正常状態で、区別可能な検知条件を"
        "立てにくい。"
    ),
    "scripts/build_judgment_asset_ab_report.py": (
        "concept/domainのグルーピングが2件未満で候補ペア0件になるのは、"
        "判断資産の概念が多様であれば普通に起こる正常状態で、書式ドリフト"
        "と区別できない。"
    ),
    "scripts/build_judgment_asset_graph.py": (
        "canonical rulesのキー抽出ロジックはab_reportと同型。activeルール"
        "が少ない/0自体が正常な業務状態と区別できない。"
    ),
    "scripts/build_loop_proof.py": (
        "「数値が取れないソースはスキップして直近値を保つ（起動をブロック"
        "しない）」という明示的な設計意図がdocstringに書かれている。"
    ),
    "scripts/build_memory_engineering_report.py": (
        "10種の異種ソース（dual-keyフォールバック済みJSON中心）を集計する"
        "読み取り専用レポート。単一閾値での判定は誤検知リスクが高い。"
    ),
    "scripts/build_predictive_framework_report.py": (
        "既にtrustworthyフラグやdaily_review_focusで「予測が1件も記録され"
        "ていない」等をレポート本文に明示的に書き出す設計で、サイレントな"
        "0件化ではない。"
    ),
    "scripts/build_reflection_action_candidates.py": (
        "data/shion_reflection_delta.jsonは上書き保存のため鮮度の判定材料"
        "にならない。信頼できる区別には新規の状態管理が要り、新しい監視の"
        "仕組みは作らない方針に反する。"
    ),
    "scripts/build_shion_architecture_layer_audit.py": (
        "`_path_flags`はファイル存在チェックであり、リネーム時も"
        "`path_flags`/`risks`として可視化される設計で、無音の0件化では"
        "ない。"
    ),
    "scripts/build_shion_eval_candidates.py": (
        "月初のみ実行される設計。今月まだ繰り返しクエリ/低評価フィード"
        "バックが無ければ0件が普通。"
    ),
    "scripts/build_shion_growth_brief.py": (
        "5つの姉妹レポートを集計するだけで、`actions`は必ずフォールバック"
        "文言が入る設計（空になり得ない）。"
    ),
    "scripts/build_shion_memory_effect_report.py": (
        "data/shion_memory_index.json（build_shion_memory_index.pyが生成）"
        "の下流の読み取り専用レポート。抽出ロジックの実体は生成元側にあり、"
        "そちらは既にドリフト検知済み（PR #1063）。"
    ),
    "scripts/build_shion_memory_promotion_queue.py": (
        "llm_extract_candidates()はGemini呼び出し失敗をstderr警告するのみ"
        "（PR #1062）。ルールベース抽出と並走する比較用の補助機能のため、"
        "Gemini疎通監視自体は既存のapi-health-checkerに委ね、exit codeは"
        "変えない。"
    ),
    "scripts/build_shion_memory_sentinel_report.py": (
        "抽出ロジックの実体は`api/shion_memory_system_audit."
        "run_shion_memory_sentinel`にあり、本ファイルは薄いCLIラッパーの"
        "み。"
    ),
    "scripts/build_shion_practical_knowledge_map.py": (
        "data/shion_memory_index.json（既にドリフト検知済み）の下流かつ、"
        "実抽出ロジックは`api/shion_practical_knowledge.py`にある。"
    ),
    "scripts/build_shion_reflection_delta.py": (
        "`quality.status`（attention/pass）は内容の質を示す設計上の指標で、"
        "日によって「attention」が正常な結果になる（既存テストが前提）。"
        "見出し依存の構造的ドリフトはsync_memory_from_daily.pyが既に検知"
        "している。"
    ),
    "scripts/check_ledger_consistency.py": (
        "監視用途のため常に0で終了する明示的設計（コード内コメントで"
        "自覚済み）。乖離は出力で伝える。"
    ),
    "scripts/compact_append_logs.py": (
        "TARGETSはハードコードされた絶対パスのリスト。data/自体が無い"
        "開発環境での全件skipと、パスのドリフトを区別する手段が無い。"
    ),
    "scripts/detect_shion_memory_contradictions.py": (
        "レポート専用のため異常終了させない設計（コード内コメントで"
        "明記）。矛盾候補0件はそれ自体が良い結果。"
    ),
    "scripts/introspection.py": (
        "`status`はboredom/内省語彙不足など内容品質シグナルで日次変動が"
        "正常（既存テストが前提）。見出し依存の抽出は表示用メトリクスのみ"
        "で判定に使われていない。"
    ),
    "scripts/judgment_asset_growth_report.py": (
        "rule_id/outcome抽出のドリフトリスクはあるが、マッチしなかった行"
        "を明示的にトラッキングしておらず、抽出全滅を判別する仕組みが無い"
        "（追加のカウンタ新設が必要で機械的なセーフティネットの範囲を"
        "超える）。"
    ),
    "scripts/learn_from_case_differences.py": (
        "SQLiteスキーマドリフト時は捕捉されない例外で落ちる（実質的な"
        "失敗シグナルだが本ガードのAST走査では検出できない）。0件はDB上"
        "の正当な業務状態。"
    ),
    "scripts/mana_obsidian_curator.py": (
        "入力欠損はevaluate_monitor(None)等が明示的にFinding"
        "（\"monitor_report_missing\"等）としてレポート本文に記録する設計。"
        "read-only guardレポートは常に成功、リスクはstatusフィールドで"
        "伝える意図的設計。"
    ),
    "scripts/obsidian_curator_report.py": (
        "read-only診断レポート。materials_count=0は上流データ未生成の"
        "日常的な状態と区別できず、内部にheading/regex依存の抽出ロジック"
        "もない。"
    ),
    "scripts/promote_cloudrun_return_data.py": (
        "promote_approved_return_data()がFileNotFoundError/RuntimeError/"
        "ValueErrorを明示raiseしており、main()に到達する前に未捕捉例外と"
        "してexit非0になる（本ガードのAST走査では検出できない）。"
        "スキーマも自スクリプトが管理しておりドリフト経路がない。"
    ),
    "scripts/reconcile_pending_tasks.py": (
        "実体はlease_intelligence_pending.reconcile_pending()への薄い"
        "ラッパー。pending総数0は紫苑が調査約束をしていない日として普通"
        "に起こる正常状態。"
    ),
    "scripts/review_experience_replay_checklist.py": (
        "build_experience_replay_checklist_candidates.pyの出力を読むだけ"
        "の下流消費者で、上流側は既にドリフト検知済み（PR #1063）。"
        "active/unreviewed=0は普通に起こる正常状態。"
    ),
    "scripts/shion_llm_triage_proposal.py": (
        "LLM不可時は警告して正常終了する明示的設計（docstringに明記）。"
        "候補0件も日常的に起こる正常状態。"
    ),
    "scripts/update_shion_memory_freshness.py": (
        "docstringで非推奨と明記され、正式経路はapi/shion_memory_decay.py"
        "へ一本化済み。索引が読めない場合も夜間パイプライン全体を失敗扱い"
        "にしない意図的な設計。"
    ),
}


def _direct_returns(node: ast.AST) -> list[ast.Return]:
    out: list[ast.Return] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(child, ast.Return):
            out.append(child)
        out.extend(_direct_returns(child))
    return out


def _find_main_guard(tree: ast.Module) -> ast.If | None:
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                return node
    return None


def _has_real_failure_signal(tree: ast.Module, guard: ast.If | None) -> bool:
    """raise SystemExit(...) / sys.exit(...) がファイル内に実質的にあるか。

    `if __name__ == "__main__": sys.exit(main())` という定型ラッパー行は
    main()自身の戻り値を右から左へ流すだけなので、それ単体では実質的な
    シグナルとみなさない（guard節は除外して走査する）。
    """
    guard_nodes = set(ast.walk(guard)) if guard is not None else set()
    for node in ast.walk(tree):
        if node in guard_nodes:
            continue
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            fname = getattr(node.exc.func, "id", None) or getattr(node.exc.func, "attr", None)
            if fname == "SystemExit":
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "exit":
                return True
    return False


def _is_risky(py_path: Path) -> bool:
    """main()が分岐しない戻り値しか持たず、実質的な失敗シグナルも無いか。"""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    main_func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    if main_func is None:
        return False
    rets = _direct_returns(main_func)
    values = set()
    for r in rets:
        if r.value is None:
            values.add("None")
        elif isinstance(r.value, ast.Constant):
            values.add(repr(r.value.value))
        else:
            values.add("<expr>")
    no_branching = len(rets) == 0 or values == {"0"}
    if not no_branching:
        return False
    guard = _find_main_guard(tree)
    return not _has_real_failure_signal(tree, guard)


def _discover_pipeline_scripts() -> list[str]:
    combined = "\n".join(
        (_REPO_ROOT / src).read_text(encoding="utf-8") for src in _PIPELINE_SCRIPT_SOURCES
    )
    found: set[str] = set()
    for m in re.finditer(r"\$\{PROJECT_ROOT\}/([a-zA-Z0-9_./]+\.py)", combined):
        found.add(m.group(1))
    for m in re.finditer(r"(?<![\w/])([a-zA-Z0-9_./]*scripts/[a-zA-Z0-9_./]+\.py)", combined):
        found.add(m.group(1))
    return sorted(p for p in found if (_REPO_ROOT / p).exists())


def _scan_risky() -> set[str]:
    return {p for p in _discover_pipeline_scripts() if _is_risky(_REPO_ROOT / p)}


def test_no_new_silent_pipeline_script():
    """許可リストに無い日次パイプラインスクリプトが無音失敗の形を持たないこと。"""
    risky = _scan_risky()
    unknown = sorted(risky - set(_ALLOWED_ALWAYS_ZERO))
    assert unknown == [], (
        "日次パイプラインに、失敗を一切シグナルしないスクリプトが見つかりました"
        "（scripts/sync_memory_from_daily.pyと同型のバグパターン。"
        "詳細はPR #1060〜#1064参照）。対象はあるのに0件になるケースを検知する"
        "分岐を追加するか、それが不要な設計判断なら"
        "_ALLOWED_ALWAYS_ZEROへ理由付きで登録してください:\n  "
        + "\n  ".join(unknown)
    )


def test_allowlist_has_no_stale_entry():
    """許可リストのスクリプトが、実際にはもう無音失敗の形をしていないこと（ラチェットを締める）。"""
    risky = _scan_risky()
    stale = sorted(
        p for p in _ALLOWED_ALWAYS_ZERO if (_REPO_ROOT / p).exists() and p not in risky
    )
    assert stale == [], (
        "失敗シグナルが追加されたのに許可リストが古いままです。"
        "_ALLOWED_ALWAYS_ZEROからエントリを消してください:\n  "
        + "\n  ".join(stale)
    )
