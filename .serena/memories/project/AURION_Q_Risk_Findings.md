# AURION Q_Risk 実装効果分析（2026-05-16）

## 現状
Q_Risk（量子リスク検知）は **設計通りに動作している** が、**スコアに影響していない**。

## 実装状況

### ✅ 完全実装
- **P2-001**: `aurion/q_risk.py` — 8種類の財務矛盾検知（FIN-CONTRADICT-001〜008）
  - 粗利率異常、売上ゼロ・費用正、営業利益>粗利、負債超過等
  - スコア計算: high×1=35点、high×2=70点
  - テスト全19件 PASS

- **P2-002**: `/predict` エンドポイント統合
  - レスポンス: `data["aurion"]["q_risk"]` に検知結果を追加
  - テスト8件 PASS（AC-501〜508）

- **P2-003**: HTML UI追加
  - Q_Risk パネル（最初は display:none、検知時に表示）
  - パターン詳細・severity 別カラーリング実装完了

### ⚠️ 制限事項
**Q_Risk は判定に影響しない（設計通り）**

`scoring_core.py:716-718` より：
```python
final_score = max(0, min(100, round(base_score + intuition_adj, 1)))
hantei = "承認圏内" if final_score >= APPROVAL_LINE else "要審議"
```

- `final_score` は借手スコア + 物件スコア + 直感補正のみで決定
- `quantum_risk_score` はこの計算には含まれない
- `quantum_risk` はレスポンスに含まれるが、スコアに反映されない

**警告トリガー（api.py:690-698）**：
```python
credit_quantum_strong_warning = (
    credit_risk_group["score"] >= 70.0 AND
    quantum_risk_score >= 60.0
)
```
- 信用リスク群が70以上 **かつ** Q_Risk≥60 で初めて警告メッセージ追加
- 単独では機能しない（信用リスク群との組み合わせのみ）

### テスト検証
- AC-504（test_P2-002.py:132-141）で明示的に確認：
  > "既存フィールド(score, judgment)が aurion 追加後も変化しない"
  
これはQ_Riskが参考値のみという設計の検証。

## 機能的評価

### 現在の役割
- 📊 情報提供：財務矛盾の可視化、監査ログ
- ⚠️ 補助警告：クレジットリスク≥70の案件に追加注記
- 🔍 リスク・サイニング：後段での詳細審査判断の材料

### 機能ギャップ
1. **スコア反映なし** — Q_Risk=70(high_risk) でも承認圏内スコアは変わらない
2. **単独警告なし** — Q_Risk だけでは警告されない（credit_risk≥70が必須）
3. **バックテスト未実施** — 実績データで有効性検証なし（cases_for_colab.json は quantum_risk を記録していない）

## 推奨改善
1. 【オプション A】スコア反映版：quantum_risk を base_score に -5〜-20 点の減点として組み込む
2. 【オプション B】警告強化版：Q_Risk≥50 単独で警告表示（信用リスク無関係に）
3. 【オプション C】現状維持：参考値として機能、別途バックテストで有効性確認

現在の設計は **保守的（参考値のみ）** で、スコアモデルの精度劣化を避ける判断と考えられる。