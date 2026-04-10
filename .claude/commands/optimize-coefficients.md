# /optimize-coefficients — 係数自動最適化

## 使い方
```
/optimize-coefficients [--status] [--force] [--dry-run]
```
- 引数なし / `--status`: 学習メタ情報と現在の状況を表示
- `--force`: 件数条件を無視して強制的に最適化を実行
- `--dry-run`: 最適化計算のみ行い、係数は更新しない（AUC比較のみ）

## 処理手順

### ステータス確認（デフォルト / `--status`）

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from auto_optimizer import (
    load_training_meta, get_registered_count,
    MIN_START, RETRAIN_INTERVAL, AUC_MIN_IMPROVEMENT
)

meta  = load_training_meta()
count = get_registered_count()
last  = meta.get('last_trained_count', 0)
next_trigger = MIN_START if last == 0 else last + RETRAIN_INTERVAL
ready = count >= next_trigger

print('=== 係数最適化ステータス ===\n')
print(f'登録件数           : {count}件')
print(f'前回学習時件数     : {last}件')
print(f'次回トリガー件数   : {next_trigger}件')
print(f'前回学習日時       : {meta.get(\"last_trained_at\") or \"未実施\"}')
print(f'前回AUC            : {meta.get(\"last_auc\") or \"N/A\"}')
print(f'累計最適化回数     : {meta.get(\"total_runs\", 0)}回')
print()
print(f'最適化状態: {\"✅ 実行可能\" if ready else f\"⏳ あと{next_trigger - count}件で実行可能\"}')
print()
print(f'AUC改善閾値: {AUC_MIN_IMPROVEMENT:.0%}（これ未満の改善は採用しない）')
if not ready:
    print(f'\nヒント: /optimize-coefficients --force で強制実行できます')
"
```

### 最適化実行（`--force` または件数条件を満たした場合）

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from auto_optimizer import (
    load_training_meta, get_registered_count, should_retrain,
    run_optimization, save_training_meta,
    MIN_START, RETRAIN_INTERVAL
)

meta  = load_training_meta()
count = get_registered_count()
force = True  # --force の場合

if not force and not should_retrain(meta, count):
    print('⏳ 最適化条件未達（--force で強制実行）')
    exit(0)

print(f'🔄 係数最適化を開始します（登録件数: {count}件）...\n')
result = run_optimization(meta, dry_run=False)  # --dry-run の場合は dry_run=True

if result.get('improved'):
    print(f'✅ 係数を更新しました')
    print(f'   旧AUC: {result[\"old_auc\"]:.4f}  →  新AUC: {result[\"new_auc\"]:.4f}')
    print(f'   改善幅: +{result[\"new_auc\"] - result[\"old_auc\"]:.4f}')
    print(f'   更新モデル: {result.get(\"updated_models\", [])}')
else:
    print(f'ℹ️  係数は更新しませんでした（改善閾値未達）')
    print(f'   現AUC: {result[\"old_auc\"]:.4f}  候補AUC: {result[\"new_auc\"]:.4f}')
"
```

### ドライランモード（`--dry-run`）

上記と同じ処理だが `dry_run=True` で実行:
- 係数ファイルを更新しない
- AUC比較結果のみ表示

```
=== ドライラン結果 ===

現在係数 AUC: 0.7823
候補係数 AUC: 0.7951
改善幅      : +0.0128

✅ 本番適用すると改善見込み
実際に適用するには: /optimize-coefficients --force
```

### 最適化後の推奨アクション

```
✅ 係数更新完了後の推奨手順:
1. /validate-rules    — 更新係数の整合性チェック
2. /audit-scores      — スコアリング結果の異常検出
3. /build-check       — アプリ再起動確認
```

## 注意事項
- 初回最適化には最低 **50件** の成約/失注データが必要
- 以降は **20件ごと** に自動トリガー条件を満たす
- AUC が前回比 `-2%` 以上悪化する場合は係数を採用しない（安全弁）
- `--dry-run` でも XGBoost 学習は実行される（計算時間あり）
- 係数更新後は Streamlit を再起動しないと反映されない場合がある
