"""reconcile_pending_tasks が日次パイプラインに配線され続けていることのデグレ防止テスト。

未完了調査タスクの陳腐化・履歴刈り込みは日次で回らないと件数が膨張するため、
run_daily_improvement_post.sh からの呼び出しとステップ記録が消えていないか検査する。
"""

from pathlib import Path


def test_reconcile_pending_wired_into_daily_post_pipeline():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    # スクリプトが呼ばれ、健全性監視に記録されている
    assert "scripts/reconcile_pending_tasks.py" in script
    assert 'log_step "reconcile_pending_tasks"' in script

    # 配布停止ゲート（Mana != allow）より前で保守が走ること
    reconcile_pos = script.index("scripts/reconcile_pending_tasks.py")
    gate_pos = script.index('if [ "${MANA_STATUS}" != "allow" ]')
    assert reconcile_pos < gate_pos


def test_reconcile_script_exists_and_runnable():
    path = Path("scripts/reconcile_pending_tasks.py")
    assert path.exists()
    body = path.read_text(encoding="utf-8")
    assert "reconcile_pending" in body
    assert "def main" in body
