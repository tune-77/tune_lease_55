"""自律型改善リファクタリング・パイプラインの統合実行エンジン."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Step スクリプトの動的インポート
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from step1_extract_and_structure import extract_improvements_as_json
    from step2_validation_checker import validate_improvements_batch
    from step3_auto_apply import apply_improvements_pipeline
    from improvement_deduplicator import deduplicate_improvements
    from implementation_ranker import rank_improvements
    from obsidian_compliance_checker import check_obsidian_compliance
    from auto_fix_policy import evaluate_auto_fix_policy
except ImportError as e:
    print(f"❌ スクリプトインポートエラー: {e}")
    sys.exit(1)


class ImprovementPipeline:
    """
    3段階パイプラインを統合実行するエンジン。
    
    フロー:
    1. チャットログから改善点を抽出・構造化（JSON）
    2. 妥当性検証（RAG + 論理チェック）
    3. 承認済み改善を自動修正・デプロイ・Obsidian同期
    """
    
    def __init__(self, workspace_root: str | Path = None):
        self.workspace_root = Path(workspace_root) if workspace_root else self._detect_workspace_root()
        self.pipeline_log: list[dict[str, Any]] = []
    
    def run(
        self,
        chat_log: str,
        rag_context: str = "",
        dry_run: bool = False,
        source_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        パイプライン全体を実行する。

        Args:
            chat_log: チャット全体のテキスト
            rag_context: Obsidian RAGから取得したコンテキスト
            dry_run: True の場合、実際の修正は行わない（検証のみ）
            source_file: チャットログの元ファイルパス（Obsidian書き戻し用）

        Returns:
            パイプライン実行結果
        """

        print("🚀 自律型改善リファクタリング・パイプラインを開始します...\n")

        # ====================
        # Step 1: 抽出・構造化
        # ====================
        print("【Step 1】チャットログから改善点を抽出・構造化中...")

        try:
            improvements_json = extract_improvements_as_json(chat_log)
            improvements = json.loads(improvements_json)
            # 元ファイルパスを各改善案に注入（Obsidian書き戻しで使用）
            if source_file:
                for imp in improvements:
                    imp.setdefault("source_file", str(source_file))
            print(f"✅ {len(improvements)}件の改善点を抽出しました\n")
        except Exception as e:
            print(f"❌ Step 1 失敗: {e}")
            return {
                "status": "FAILED",
                "step": 1,
                "error": str(e),
            }
        
        if not improvements:
            print("⚠️ 改善点が見つかりませんでした。")
            return {
                "status": "NO_IMPROVEMENTS",
                "improvements": [],
                "validations": [],
                "applied": [],
            }

        original_improvements = improvements

        # ====================
        # Step 1.5: 統合・重複排除
        # ====================
        print("【Step 1.5】改善案を統合・重複排除中...")

        try:
            improvements, grouped_improvements = deduplicate_improvements(improvements)
            duplicate_count = sum(g.get("duplicate_count", 0) for g in grouped_improvements)
            print(
                f"✅ 統合完了: {len(original_improvements)}件 → "
                f"{len(improvements)}件（重複 {duplicate_count}件）\n"
            )
        except Exception as e:
            print(f"❌ Step 1.5 失敗: {e}")
            return {
                "status": "FAILED",
                "step": 1.5,
                "error": str(e),
                "improvements": original_improvements,
            }

        # ====================
        # Step 1.6: 実装可能性ランク付け
        # ====================
        print("【Step 1.6】実装可能性と改善順を判定中...")

        try:
            improvements, recommended_order = rank_improvements(improvements)
            print(f"✅ 優先順を作成しました: {len(recommended_order)}件\n")
            for item in recommended_order[:5]:
                print(
                    f"  {item['order']:02d}. [{item['id']}] {item['title']} "
                    f"({item['category']}, score={item['priority_score']})"
                )
            if len(recommended_order) > 5:
                print(f"  ... 他 {len(recommended_order) - 5}件")
            print()
        except Exception as e:
            print(f"❌ Step 1.6 失敗: {e}")
            return {
                "status": "FAILED",
                "step": 1.6,
                "error": str(e),
                "improvements": improvements,
                "grouped_improvements": grouped_improvements,
            }
        
        # ====================
        # Step 2: 妥当性検証
        # ====================
        print("【Step 2】改善点の妥当性を検証中...")
        
        try:
            validation_results = validate_improvements_batch(improvements, rag_context=rag_context)
            
            approved_count = sum(1 for v in validation_results if v["status"] == "APPROVED")
            rejected_count = len(validation_results) - approved_count
            
            print(f"✅ 検証完了: 承認 {approved_count}件, 拒否 {rejected_count}件\n")
            
            # 検証結果をログ
            for imp, val in zip(improvements, validation_results):
                print(f"  [{imp['id']}] {imp['title']}")
                print(f"    → {val['status']}")
                if val["status"] == "REJECTED" and val["critical_flaws"]:
                    for flaw in val["critical_flaws"][:2]:
                        print(f"       ⚠️ {flaw}")
        
        except Exception as e:
            print(f"❌ Step 2 失敗: {e}")
            return {
                "status": "FAILED",
                "step": 2,
                "error": str(e),
            }

        # ====================
        # Step 2.5: Obsidian連携整合性チェック
        # ====================
        print("\n【Step 2.5】Obsidian連携ルールを確認中...")

        try:
            obsidian_compliance = check_obsidian_compliance(improvements, self.workspace_root)
            violation_count = len(obsidian_compliance.get("violations", []))
            route_count = len(obsidian_compliance.get("route_sensitive_ids", []))
            print(
                f"✅ Obsidian整合性チェック完了: "
                f"status={obsidian_compliance.get('status')}, "
                f"violations={violation_count}, route_sensitive={route_count}\n"
            )
        except Exception as e:
            print(f"❌ Step 2.5 失敗: {e}")
            return {
                "status": "FAILED",
                "step": 2.5,
                "error": str(e),
                "improvements": improvements,
                "validations": validation_results,
                "grouped_improvements": grouped_improvements,
                "recommended_order": recommended_order,
            }
        
        # ====================
        # Step 3: 自動修正・デプロイ
        # ====================
        print("\n【Step 3】承認済み改善を自動修正・デプロイ中...")
        
        if dry_run:
            print("(DRY RUN モード: 実際の修正は行いません)")
        
        try:
            if dry_run:
                # Dry run: 承認された改善を小規模自動修正候補と要確認に分ける
                approved_improvements = [
                    imp for imp, val in zip(improvements, validation_results)
                    if val["status"] == "APPROVED"
                ]
                auto_fix_candidates: list[dict[str, Any]] = []
                policy_needs_review: list[dict[str, Any]] = []
                for imp in approved_improvements:
                    policy = evaluate_auto_fix_policy(imp, self.workspace_root)
                    item = {
                        "id": imp.get("id"),
                        "title": imp.get("title"),
                        "canonical_key": imp.get("canonical_key"),
                        "auto_fix_policy": policy,
                    }
                    if policy.get("auto_fix_allowed"):
                        auto_fix_candidates.append(item)
                    else:
                        policy_needs_review.append(item)
                
                result = {
                    "status": "DRY_RUN_COMPLETE",
                    "original_improvements": original_improvements,
                    "improvements": improvements,
                    "validations": validation_results,
                    "approved_improvements": approved_improvements,
                    "auto_fix_candidates": auto_fix_candidates,
                    "policy_needs_review": policy_needs_review,
                    "grouped_improvements": grouped_improvements,
                    "recommended_order": recommended_order,
                    "obsidian_compliance": obsidian_compliance,
                    "applied_count": 0,
                    "failed_count": 0,
                }
                
                print(
                    "✅ Dry run 完了: "
                    f"自動修正候補 {len(auto_fix_candidates)}件, "
                    f"ポリシー要確認 {len(policy_needs_review)}件"
                )
            
            else:
                # 本実行: 改善を適用
                result = apply_improvements_pipeline(
                    improvements,
                    validation_results,
                    self.workspace_root,
                )
                
                result["original_improvements"] = original_improvements
                result["improvements"] = improvements
                result["validations"] = validation_results
                result["grouped_improvements"] = grouped_improvements
                result["recommended_order"] = recommended_order
                result["obsidian_compliance"] = obsidian_compliance
                result["status"] = "COMPLETED" if result["applied_count"] > 0 else "NO_APPLIED"
                
                print(f"✅ 自動修正完了: {result['applied_count']}件適用, {result['failed_count']}件失敗")
                if result["commit_result"]["success"]:
                    print(f"✅ Git コミット成功: {result['commit_result']['commit_hash']}")
        
        except Exception as e:
            print(f"❌ Step 3 失敗: {e}")
            result = {
                "status": "FAILED",
                "step": 3,
                "error": str(e),
                "original_improvements": original_improvements,
                "improvements": improvements,
                "validations": validation_results,
                "grouped_improvements": grouped_improvements,
                "recommended_order": recommended_order,
                "obsidian_compliance": obsidian_compliance,
            }
        
        return result
    
    def _detect_workspace_root(self) -> Path:
        """スクリプト実行位置からプロジェクトルートを検出."""
        
        current = Path(__file__).parent
        
        while current != current.parent:
            # tune_lease_55フォルダを探す
            if (current / "tune_lease_55.py").exists() or (current / "CLAUDE.md").exists():
                return current
            if (current / "AGENTS.md").exists():
                return current
            current = current.parent
        
        # フォールバック
        return Path.cwd()


def run_pipeline_from_cli():
    """CLIからパイプラインを実行."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="自律型改善リファクタリング・パイプライン",
    )
    parser.add_argument(
        "chat_log_file",
        type=str,
        help="チャットログファイルのパス",
    )
    parser.add_argument(
        "--rag-context",
        type=str,
        default="",
        help="Obsidian RAGコンテキスト（オプション）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="検証のみ（実際の修正は行わない）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果を JSON ファイルに出力",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="プロジェクトルートパス",
    )
    
    args = parser.parse_args()
    
    # チャットログを読み込み
    chat_log_path = Path(args.chat_log_file)
    if not chat_log_path.exists():
        print(f"❌ ファイルが見つかりません: {chat_log_path}")
        sys.exit(1)
    
    try:
        chat_log = chat_log_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")
        sys.exit(1)
    
    # RAGコンテキストを読み込み（オプション）
    rag_context = args.rag_context
    if args.rag_context and args.rag_context.endswith(".txt"):
        try:
            rag_context = Path(args.rag_context).read_text(encoding="utf-8")
        except Exception:
            pass  # ファイルでない場合は文字列として使用
    
    # パイプライン実行
    pipeline = ImprovementPipeline(args.workspace)
    result = pipeline.run(chat_log, rag_context=rag_context, dry_run=args.dry_run)
    
    # 結果出力
    print("\n" + "=" * 50)
    print("パイプライン実行結果")
    print("=" * 50)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # ファイル出力
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 結果を保存しました: {output_path}")
    
    # 終了コード
    sys.exit(0 if result["status"] in ["COMPLETED", "DRY_RUN_COMPLETE"] else 1)


if __name__ == "__main__":
    run_pipeline_from_cli()
