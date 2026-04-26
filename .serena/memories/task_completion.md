# タスク完了時チェックリスト

1. `make lint` — pyflakes構文チェック
2. `make test` — pytestテスト実行
3. スコアリング変更時: `scoring-auditor` エージェント起動
4. rule_manager.py/coeff_definitions.py変更時: `rule-validator` エージェント起動
5. DBスキーマ変更時: `migration-validator` エージェント起動
6. data/以下ファイルをコミットしていないか確認
7. secrets.tomlをコミットしていないか確認
