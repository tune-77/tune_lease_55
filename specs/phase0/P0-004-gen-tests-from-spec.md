---
spec_id: P0-004
phase: 0
title: SPECからテストスケルトン生成スクリプト
status: approved
author: Claude Opus
reviewer: human
version: 1.0.0
depends_on: []
---

## 1. Goal
SPECファイル（Markdown）のAcceptance CriteriaセクションのAC-xxx Given-When-Thenを解析し、pytest用テストスケルトンを `tests/spec_phaseN/test_PX-YYY.py` として自動生成する。Codexが「ACと1:1対応するテスト関数」を最初から手元に持った状態で実装に着手できるようにする。

## 2. Scope
### In Scope
- `scripts/gen_tests_from_spec.py`の新規作成
- SPEC frontmatterから `spec_id`, `phase` を抽出
- `## ... Acceptance Criteria` セクションからAC-xxx行を抽出
- 各ACをpytest関数（`test_ac_xxx`）にマッピング、docstringにGiven-When-Thenを保存
- 出力先 `tests/spec_phase{N}/test_{spec_id}.py`
- 既存ファイル上書き保護（`--force` で明示的に上書き）
- `tests/spec_phase{N}/__init__.py`、`tests/__init__.py` の自動作成

### Out of Scope
- 実テストロジックの自動生成（failスケルトンのみ）
- AC-xxxのセマンティック理解（NL→アサーション変換）
- フィクスチャ自動生成
- 既存テストファイルへの追記マージ

## 3. Inputs / Outputs
| 区分 | 内容 |
|------|------|
| Input | SPECマークダウンファイルパス |
| Output | `tests/spec_phase{N}/test_{spec_id}.py` |
| Output | `tests/__init__.py`, `tests/spec_phase{N}/__init__.py`（未存在時のみ作成） |
| Output | stdout実行ログ |

## 4. Data Model
出力pytestファイル形式:
```python
"""
Auto-generated test skeleton for {spec_id}.
DO NOT EDIT the AC docstrings manually — regenerate via:
    python scripts/gen_tests_from_spec.py <spec_path>
Each test_ac_xxx corresponds 1:1 with AC-xxx in the SPEC.
"""
import pytest

SPEC_ID = "{spec_id}"
PHASE = {phase}

def test_ac_001() -> None:
    """
    AC-001: Given ..., When ..., Then ....
    """
    pytest.fail("AC-001 not implemented")
```

## 5. API / Interface
CLI:
```
python scripts/gen_tests_from_spec.py SPEC_PATH [--out-dir PATH] [--force] [--dry-run]
```
| 引数/オプション | 既定値 | 説明 |
|---|---|---|
| SPEC_PATH | 必須 | SPECマークダウンへのパス |
| --out-dir | tests | テスト出力ルート |
| --force | False | 既存出力ファイルを上書き |
| --dry-run | False | stdout出力のみ |

## 6. Business Rules
- **BR-001**: SPECファイルが存在しない場合、exit code 2
- **BR-002**: frontmatterに `spec_id` または `phase` が無い場合、exit code 3
- **BR-003**: `spec_id` は正規表現 `^P\d+-\d{3}$` にマッチする必要がある、不一致はexit code 3
- **BR-004**: `phase` はfrontmatterの値を整数として解釈、spec_idと不整合な場合は警告のみ
- **BR-005**: `## ... Acceptance Criteria ...` 見出しが見つからない場合、exit code 4
- **BR-006**: ACが1件も抽出できない場合、warningを出してテスト関数0個のファイルを生成
- **BR-007**: 出力ファイルが既存かつ `--force` 無しの場合、exit code 5
- **BR-008**: テスト関数名は `test_ac_<3桁>` にスネークケース化（AC-001 → test_ac_001）
- **BR-009**: Given-When-ThenテキストはdocstringにそのままGiven-When-Thenを保存
- **BR-010**: 同一AC IDが複数回出現した場合は最初の出現のみ採用、stderrにwarning

## 7. UI / UX
CLI成功ログ例:
```
[gen_tests] parsed spec: P1-001 (phase=1)
[gen_tests] extracted 8 AC entries
[gen_tests] writing tests/spec_phase1/test_P1-001.py
[gen_tests] done
```

## 8. Error Handling
| 状況 | exit code | メッセージ |
|---|---|---|
| SPECファイル不在 | 2 | `spec not found: <path>` |
| frontmatter不正 | 3 | `invalid frontmatter: missing spec_id/phase` |
| AC見出し不在 | 4 | `Acceptance Criteria section not found` |
| 出力ファイル既存(--force無し) | 5 | `output exists: <path> (use --force)` |
| 想定外例外 | 1 | traceback |

## 9. Acceptance Criteria
- **AC-001**: Given 有効なSPEC（frontmatterにspec_id: P1-001, phase: 1を持ちACを3件含む）を入力した状態で、When スクリプトを実行する、Then `tests/spec_phase1/test_P1-001.py` が生成され `test_ac_001`, `test_ac_002`, `test_ac_003` の3関数を含む
- **AC-002**: Given AC-001で生成されたファイルをpytestで実行した状態で、When pytestが完了する、Then 全テストが `Failed`（理由 `not implemented`）になる（collection errorは出ない）
- **AC-003**: Given 存在しないSPECパスを指定した状態で、When スクリプトを実行する、Then exit code 2でstderrに `spec not found` を含む
- **AC-004**: Given frontmatterに `spec_id` が無いSPECを指定した状態で、When スクリプトを実行する、Then exit code 3でstderrに `missing spec_id` を含む
- **AC-005**: Given `spec_id: HELLO-001` のような不正フォーマットのSPECを指定した状態で、When スクリプトを実行する、Then exit code 3でstderrに `invalid spec_id format` を含む
- **AC-006**: Given `## 9. Acceptance Criteria` 見出しが無いSPECを指定した状態で、When スクリプトを実行する、Then exit code 4で終了する
- **AC-007**: Given 既に出力ファイルが存在する状態で、When `--force` 無しで実行する、Then exit code 5で終了し既存ファイルは変更されない
- **AC-008**: Given 既存ファイルがある状態で、When `--force` 付きで実行する、Then ファイルが上書きされexit code 0
- **AC-009**: Given ACテキストにマルチバイト文字（日本語）を含むSPECを指定した状態で、When スクリプトを実行する、Then 生成されたPythonファイルが `python -m py_compile` で成功する
- **AC-010**: Given `--dry-run` を指定した状態で、When スクリプトが完了する、Then `tests/` 配下に実ファイルは作成されずstdoutに `[dry-run]` プレフィックス付きで生成予定コードが出力される
- **AC-011**: Given `tests/__init__.py` および `tests/spec_phase1/__init__.py` がいずれも存在しない状態で、When スクリプトを実行する、Then 両ファイルが空ファイルとして自動作成される
- **AC-012**: Given 同じAC ID（AC-001）が2回出現するSPECを指定した状態で、When スクリプトを実行する、Then 生成ファイルには `test_ac_001` が1つだけ含まれstderrに `duplicate AC-001` の警告が出力される

## 10. Non-Functional
- Python: 3.11系
- 依存: 標準ライブラリ + PyYAML（requirements.txtに追記、無ければ）
- 実行時間: 1SPECあたり1秒以内
- 生成コードは `black --check`（line-length 100）をpass

## 11. Implementation Notes（Codex向け）
関数構成:
- `main(argv: list[str]) -> int`
- `parse_args(argv) -> argparse.Namespace`
- `load_spec(path: Path) -> tuple[dict, str]`（frontmatter dict + body文字列）
- `validate_frontmatter(fm: dict) -> tuple[str, int]`（spec_id, phaseを返す）
- `extract_acceptance_criteria(body: str) -> list[tuple[str, str]]`（[(ac_id, full_text), ...]）
- `render_test_file(spec_id: str, phase: int, acs: list[tuple[str, str]]) -> str`
- `write_output(out_path: Path, content: str, force: bool, dry_run: bool) -> None`
- `ensure_init_files(out_dir: Path, phase_dir: Path) -> None`

AC抽出正規表現: `r"^\s*-\s*(?:\*\*)?(AC-\d{3})(?:\*\*)?\s*[:：]?\s*(.+)$"`
frontmatter parse: `yaml.safe_load`（`---`で囲まれた先頭ブロック）
テスト関数名: `AC-001` → `test_ac_001`
docstringエスケープ: `"""` → `\"\"\"`
ブランチ: `feature/p0-004-gen-tests`、PRタイトル: `[P0-004] add SPEC -> pytest skeleton generator`

## 12. Test Plan
`tests/spec_phase0/test_P0-004.py` にAC-001〜012と1:1対応する自動テストを実装：
- T-001〜012: 上記各ACに対応（AC詳細参照）
実行: `pytest tests/spec_phase0/test_P0-004.py -v`
