#!/usr/bin/env python3
"""週1回、紫苑の通常チャットの答えの品質を同じ質問セットで回帰テストする。

- 質問: scripts/answer_regression_questions.json（初版は experiments/chat_before_after/ の10問、PR #1224）
- 採点は機械的なキーワード判定だけ（Jev・LLM の自己採点は使わない）
  ① 教えたノウハウを使ったか ② 基本知識が正確か（誤りの典型が1つでも入れば×）
  ③ 根拠の引用があるか ④ 保存していないのに「保存しました」と言っていないか
  （kind=save_honesty は setup の発言で会話を作ってから保存を頼み、④と無い保存先（wrong_any）だけを見る。2026-10-04 追加）
- 本番のデータ・記録を汚さない: 今のコードの git worktree を一時ディレクトリに作り、data/ を
  APFS クローン（cp -c）でコピーして、別ポート（8104）で API を立てる。公開中の API（8000）・トンネルには触れない
- 結果は data/answer_regression/<日付>.json と Obsidian に残し、AURION CORE 朝報に毎回1行、
  前週より下がった項目・基準（①80%・②100%・③90%・④0件）を下回った項目は上部に警告を出す
- Gemini 呼び出しは MAX_GEMINI_CALLS 回で打ち切る
- ⑤ 紫苑レビュー（審査分析画面）: scripts/answer_regression_review_samples.json の匿名案件を、本番と同じ
  frontend/src/lib/shionReview.ts で依頼文にして caller=screening_review で送る。120秒以内に返る・定型文でない・
  出典がある・該当する方針が冒頭に出る・依頼文が Vault・記録に残らない、をすべて満たせば○（2026-10-03 追加）

使い方:
  .venv/bin/python scripts/weekly_answer_regression.py            # 実行
  .venv/bin/python scripts/weekly_answer_regression.py --add --id <id> --kind taught --q "<質問>" \
      --taught-any "キーワード1|キーワード2" --wrong-any "誤り1|誤り2" --source "<教えた方針>"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from silent_failure_log import record_silent_failure  # noqa: E402

# data/・索引・モデルの取り元（既定はこのチェックアウト。worktree から手動実行する時はメインを指す）
DATA_SOURCE_ROOT = Path(os.environ.get("ANSWER_REGRESSION_DATA_ROOT") or PROJECT_ROOT)
QUESTIONS_JSON = PROJECT_ROOT / "scripts" / "answer_regression_questions.json"
REVIEW_SAMPLES_JSON = PROJECT_ROOT / "scripts" / "answer_regression_review_samples.json"
REVIEW_TIMEOUT_S = 120  # 画面の打ち切り（frontend/src/lib/shionReview.ts の SHION_REVIEW_SOFT_TIMEOUT_MS）
RESULT_DIR = DATA_SOURCE_ROOT / "data" / "answer_regression"
VAULT_SUBDIR = Path("Projects") / "tune_lease_55" / "Answer Regression"
SERVER_PY = Path("experiments") / "chat_before_after" / "server.py"
PORT = 8104
MAX_GEMINI_CALLS = 30
VAULT_COPY_TIMEOUT_S = 180
STALE_DAYS = 8
# 基準（質問が増えても使えるよう割合で持つ）: ①4/5 ②5/5 ③9/10 ④0件
THRESHOLDS = {"taught": 0.8, "basic": 1.0, "cites": 0.9, "review": 1.0}

from api.chat_teaching_capture import SAVE_CLAIM_RE  # noqa: E402 - 回答の後処理と同じ「保存した」の判定を使う
CITE_RE = re.compile(r"\[\[[^\]]+\]\]|出典|参照ナレッジ|根拠[:：]")
METRICS = (("taught", "① 教えたノウハウ"), ("basic", "② 基本知識の正確さ"), ("cites", "③ 引用"), ("false_save", "④ 誤った「保存」"), ("review", "⑤ 紫苑レビュー"))
# 画面の簡易生成（buildShionReviewFallback）の定型句。サーバーの返答にこれがあれば紫苑が書いていない
REVIEW_TEMPLATE_RE = re.compile(r"と現場メモの具体性の差に注目します|紫苑レビューが空でした")
# 依頼文が残ってはいけない場所（Vault → Vertex 同期・記憶の昇格・教示の救済に流れる）
REVIEW_LEAK_TARGETS = (
    "data/cloudrun_chat_log.jsonl", "data/language_judgment_materials.jsonl", "data/rag_search_log.jsonl",
    "data/chat_logs.jsonl", "data/vertex_distillation_state.json",
)
REVIEW_LEAK_VAULT_DIRS = (Path("Projects") / "tune_lease_55" / "Research" / "Vertex Distilled", Path("Projects") / "tune_lease_55" / "AI Chat")


# --- 質問セット ------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_JSON) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def add_question(entry: dict[str, Any], path: Path = QUESTIONS_JSON) -> dict[str, Any]:
    """ユーザーが新しく教えた方針・ノウハウから1問追加する。"""
    store = json.loads(path.read_text(encoding="utf-8"))
    if any(q["id"] == entry["id"] for q in store["questions"]):
        raise SystemExit(f"id が重複しています: {entry['id']}")
    if entry["kind"] == "taught" and not entry.get("taught_any"):
        raise SystemExit("kind=taught には --taught-any が必要です")
    if entry["kind"] == "basic" and not entry.get("correct_any"):
        raise SystemExit("kind=basic には --correct-any が必要です")
    if not entry.get("wrong_any"):
        raise SystemExit("--wrong-any が必要です")
    store["questions"].append({k: v for k, v in entry.items() if v})
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entry


def load_review_samples(path: Path = REVIEW_SAMPLES_JSON) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["samples"] if path.exists() else []


# --- 採点 ------------------------------------------------------------------------------


def reply_text(data: dict[str, Any]) -> str:
    for key in ("reply", "response", "answer", "message", "text"):
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key]
    return ""


def refs_of(data: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("rag_refs", "refs", "sources", "rag_knowledge_refs", "knowledge_refs"):
        for item in data.get(key) or []:
            refs.append(str(item.get("obsidian_ref") or item.get("ref") or item) if isinstance(item, dict) else str(item))
    return refs


def saved_something(data: dict[str, Any]) -> bool:
    blob = json.dumps({k: v for k, v in data.items() if k not in ("reply", "response", "answer")}, ensure_ascii=False)
    return bool(re.search(r'"(saved|teaching_saved|judgment_asset_saved|chat_teaching)[^"]*"\s*:\s*(true|\{)', blob))


def score(question: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    text = reply_text(data)
    refs = refs_of(data)
    wrong = [w for w in question.get("wrong_any", []) if w in text]
    result: dict[str, Any] = {
        "cites": bool(refs) or bool(CITE_RE.search(text)),
        "false_save": bool(SAVE_CLAIM_RE.search(text)) and not saved_something(data),
        "wrong_hits": wrong,
        "chars": len(text),
    }
    if question["kind"] == "save_honesty":
        # 保存しても「〇〇テンプレート集として永続化」のような無い保存先を言えば誤った保存とみなす
        result["false_save"] = result["false_save"] or bool(wrong)
        result["cites"] = None
    elif question["kind"] == "taught":
        result["taught"] = any(k in text for k in question["taught_any"]) and not wrong
    else:
        ok = any(k in text for k in question["correct_any"]) and all(any(k in text for k in group) for group in question.get("correct_all_any", []))
        result["basic"] = ok and not wrong
    return result


def score_review(sample: dict[str, Any], data: dict[str, Any], elapsed_s: float, leaks: list[str]) -> dict[str, Any]:
    text = reply_text(data)
    head = text[:200]
    policy_first = None
    if sample.get("policy_any"):
        policy_first = any(k in head for k in sample["policy_any"]) and ("方針" in head)
    result: dict[str, Any] = {
        "fast": elapsed_s <= REVIEW_TIMEOUT_S,
        "not_template": len(text) >= 80 and not REVIEW_TEMPLATE_RE.search(text),
        "cites": bool(CITE_RE.search(text)),
        "policy_first": policy_first,
        "no_leak": not leaks,
        "leaks": leaks,
        "false_save": bool(SAVE_CLAIM_RE.search(text)) and not saved_something(data),
        "elapsed_s": round(elapsed_s, 1),
        "chars": len(text),
    }
    result["review"] = all(result[k] for k in ("fast", "not_template", "cites", "no_leak")) and policy_first is not False and not result["false_save"]
    return result


def review_leaks(repo: Path, vault: Path, markers: list[str], since: float) -> list[str]:
    """since 以降に書かれた記録・Vault ノートに、依頼文だけにある文字列（社名・人名・メモ）が残っていれば場所を返す。"""
    found: list[str] = []
    paths = [repo / rel for rel in REVIEW_LEAK_TARGETS]
    for rel in REVIEW_LEAK_VAULT_DIRS:
        paths.extend(p for p in (vault / rel).rglob("*.md") if (vault / rel).exists())
    for path in paths:
        try:
            if not path.exists() or path.stat().st_mtime < since:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in text for marker in markers):
            found.append(str(path.relative_to(vault if path.is_relative_to(vault) else repo)))
    return found


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    scored = [r for r in rows if "check" in r]
    out = {}
    for key, kind in (("taught", "taught"), ("basic", "basic"), ("cites", None), ("false_save", None), ("review", "review")):
        items = [r["check"] for r in scored if kind is None or r["kind"] == kind]
        if key == "cites":
            items = [c for c in items if c.get("cites") is not None]
        out[key] = {"ok": sum(1 for c in items if c.get(key)), "n": len(items)}
    out["errors"] = {"ok": sum(1 for r in rows if "check" not in r), "n": len(rows)}
    return out


def evaluate(summary: dict[str, dict[str, int]], previous: dict[str, dict[str, int]] | None) -> list[str]:
    """基準未満・前週より低下した項目の警告文（無ければ空）。"""
    problems = []
    for key, label in METRICS:
        cur = summary.get(key) or {"ok": 0, "n": 0}
        if not cur["n"]:
            continue
        shown = f"{cur['ok']}件" if key == "false_save" else f"{cur['ok']}/{cur['n']}"
        reasons = []
        if key == "false_save":
            if cur["ok"] > 0:
                reasons.append("基準0件")
        elif cur["ok"] < math.ceil(THRESHOLDS[key] * cur["n"] - 1e-9):
            reasons.append(f"基準{math.ceil(THRESHOLDS[key] * cur['n'] - 1e-9)}/{cur['n']}")
        prev = (previous or {}).get(key)
        if prev and prev.get("n"):
            worse = cur["ok"] > prev["ok"] if key == "false_save" else cur["ok"] / cur["n"] < prev["ok"] / prev["n"]
            if worse:
                reasons.append(f"前週{prev['ok']}" + ("件" if key == "false_save" else f"/{prev['n']}"))
        if reasons:
            problems.append(f"{label} {shown}（{'・'.join(reasons)}）")
    errors = summary.get("errors") or {}
    if errors.get("ok"):
        problems.append(f"未回答 {errors['ok']}問（エラー・呼び出し上限）")
    return problems


# --- 実行環境（本番を汚さない） ----------------------------------------------------------


def _clone(src: Path, dst: Path) -> None:
    if src.exists():
        subprocess.run(["cp", "-cR", str(src), str(dst)], check=True)


def prepare_sandbox(base: Path) -> tuple[Path, Path]:
    """今のコードの worktree（data/ はクローン）と Vault のコピーを作る。"""
    repo, vault = base / "repo", base / "vault"
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "worktree", "add", "--detach", "--no-checkout", str(repo), "HEAD"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "HEAD", "--", ".", ":(exclude)data"], check=True, capture_output=True)
    _clone(DATA_SOURCE_ROOT / "data", repo / "data")
    for rel in ("api/chroma_db", "models/sentence-transformers"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        _clone(DATA_SOURCE_ROOT / rel, repo / rel)
    for secret_file in (DATA_SOURCE_ROOT / ".streamlit").glob("secrets*.toml"):
        _clone(secret_file, repo / ".streamlit" / secret_file.name)  # 鍵は読まずにクローンするだけ（api.main が本番と同じ方法で読む）
    from runtime_paths import resolve_obsidian_vault

    try:
        # iCloud の未ダウンロードファイルで止まることがあるので時間で打ち切り、取れた分を使う（毎週同じ条件）
        copied = subprocess.run(["cp", "-cR", str(resolve_obsidian_vault()), str(vault)], timeout=VAULT_COPY_TIMEOUT_S, capture_output=True)
        if copied.returncode != 0:  # 一部のファイルを写せないまま採点すると点数が下がって見える
            record_silent_failure("answer.weekly_answer_regression.vault_copy", "subprocess_failed", detail=f"cp exit {copied.returncode}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        record_silent_failure("answer.weekly_answer_regression.vault_copy", "timeout", exc, detail="取れた分のVaultで採点")
    vault.mkdir(exist_ok=True)
    return repo, vault


def cleanup_sandbox(base: Path) -> None:
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "worktree", "remove", "--force", str(base / "repo")], capture_output=True)
    shutil.rmtree(base, ignore_errors=True)


def _wait(port: int, timeout: float = 480.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=5) as res:
                if res.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise RuntimeError(f"port {port} が起動しない")


def ask(question: str, user_id: str, timeout: float = 240.0) -> dict[str, Any]:
    return _post("/api/chat", {"message": question, "user_id": user_id, "response_mode": "shion"}, timeout)


def _post(path: str, payload: dict[str, Any], timeout: float = 240.0) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=body, headers={"Content-Type": "application/json", "X-Shion-Verification": "1"}, method="POST")  # REV-591
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _node_bin() -> str:
    """launchd の PATH には nvm の node が無いので、無ければ nvm の最新を使う。"""
    found = shutil.which("node")
    if found:
        return found
    candidates = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"))
    if not candidates:
        raise RuntimeError("node が見つからない（紫苑レビューの依頼文を作れない）")
    return str(candidates[-1])


_REVIEW_BODY_JS = """
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
import(input.module).then((m) => {
  const prompt = m.buildShionReviewPrompt(input.result, input.form, input.candidates, "standard", []);
  process.stdout.write(JSON.stringify(m.buildShionReviewChatBody(input.result, input.form, prompt)));
});
"""


def review_request_body(repo: Path, sample: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """本番と同じ TS（frontend/src/lib/shionReview.ts）で紫苑レビューの /api/chat 本文を作る。"""
    payload = {"module": str(repo / "frontend" / "src" / "lib" / "shionReview.ts"), "result": sample["result"], "form": sample["form"], "candidates": candidates}
    out = subprocess.run(
        [_node_bin(), "--experimental-strip-types", "--no-warnings", "-e", _REVIEW_BODY_JS],
        input=json.dumps(payload, ensure_ascii=False), capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(out.stdout)


def _screening_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    form, result = sample["form"], sample["result"]
    params = urllib.parse.urlencode({
        "industry_major": result.get("industry_major", ""), "industry_sub": result.get("industry_sub", ""),
        "asset_name": form.get("asset_name", ""), "asset_purpose": form.get("asset_purpose", ""),
        "hantei": result.get("hantei", ""), "score": result.get("score", 0), "limit": 3,
    })
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/judgment-asset-candidates/screening?{params}", timeout=60) as res:
        return json.loads(res.read().decode("utf-8")).get("candidates", [])


def run_review_samples(samples: list[dict[str, Any]], repo: Path, vault: Path, counter: Path, today: dt.date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        row: dict[str, Any] = {"id": sample["id"], "kind": "review", "q": f"紫苑レビュー: {sample['id']}"}
        if _count(counter) >= MAX_GEMINI_CALLS:
            row["error"] = "gemini_call_limit"
        else:
            try:
                body = review_request_body(repo, sample, _screening_candidates(sample))
                body["user_id"] = f"weekly_regression_review_{today.isoformat()}_{sample['id']}"
                started_wall, started = time.time(), time.monotonic()
                data = _post("/api/chat", body, timeout=600.0)
                elapsed = time.monotonic() - started
                time.sleep(2)  # バックグラウンドの書き込みを待ってから漏れを見る
                leaks = review_leaks(repo, vault, sample["secret_markers"], started_wall - 1)
                row.update({"reply": reply_text(data), "refs": refs_of(data)[:6], "check": score_review(sample, data, elapsed, leaks)})
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"{row['id']}: {row.get('check') or row.get('error')}", flush=True)
        rows.append(row)
    return rows


def _count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return 0


def run_questions(questions: list[dict[str, Any]], base: Path, today: dt.date) -> tuple[list[dict[str, Any]], int]:
    repo, vault = prepare_sandbox(base)
    counter = base / "gemini_calls.txt"
    blocked_env = {"DATABASE_URL", "DATABASE_URL_SECRET_NAME", "DB_PATH", "SQLITE_DB_PATH", "LEASE_DB_PATH"}
    env = {
        k: v for k, v in os.environ.items()
        if k not in blocked_env and not k.startswith(("SLACK", "K_SERVICE", "CLOUDRUN"))
    }
    env.update({
        "OBSIDIAN_VAULT_PATH": str(vault), "OBSIDIAN_VAULT": str(vault), "GCS_VAULT_LOCAL_DIR": str(base / "no_gcs_vault"),
        "PYTHONPATH": str(repo), "PYTHONUNBUFFERED": "1", "JEV_JUDGMENT_LOG_PATH": "off", "DATA_DIR": str(repo / "data"),
    })
    log = (base / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(repo / SERVER_PY), "--port", str(PORT), "--counter", str(counter)], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows: list[dict[str, Any]] = []
    try:
        _wait(PORT)
        for question in questions:
            row: dict[str, Any] = {"id": question["id"], "kind": question["kind"], "q": question["q"]}
            if _count(counter) >= MAX_GEMINI_CALLS:
                row["error"] = "gemini_call_limit"
            else:
                try:
                    user_id = f"weekly_regression_{today.isoformat()}"
                    if question.get("setup"):
                        user_id = f"{user_id}_{question['id']}"  # 会話を作る質問は他の質問の履歴と混ぜない
                        for turn in question["setup"]:
                            ask(turn, user_id)
                    data = ask(question["q"], user_id)
                    row.update({"reply": reply_text(data), "refs": refs_of(data)[:6], "check": score(question, data)})
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"{row['id']}: {row.get('check') or row.get('error')}", flush=True)
            rows.append(row)
        rows.extend(run_review_samples(load_review_samples(), repo, vault, counter, today))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rows, _count(counter)


# --- 記録・朝報 ------------------------------------------------------------------------


def previous_result(today: dt.date, result_dir: Path = RESULT_DIR) -> dict[str, Any] | None:
    files = sorted(p for p in result_dir.glob("20*.json") if p.stem < today.isoformat())
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _mark(v: Any) -> str:
    return "○" if v else "×"


def obsidian_note(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "---", f"date: {report['date']}", "tags: [回帰テスト, 紫苑, 週次]", "source: scripts/weekly_answer_regression.py", "---", "",
        f"# 答えの品質回帰テスト {report['date']}", "",
        f"- ①教えたノウハウ {s['taught']['ok']}/{s['taught']['n']}・②基本知識 {s['basic']['ok']}/{s['basic']['n']}・"
        f"③引用 {s['cites']['ok']}/{s['cites']['n']}・④誤った保存 {s['false_save']['ok']}件{_review_score_text(s)}（Gemini {report['gemini_calls']}回）",
        f"- 警告: {' / '.join(report['problems']) if report['problems'] else 'なし'}", "",
        "| 質問 | 判定 | 誤りの典型 | 答えの冒頭 |", "|---|---|---|---|",
    ]
    for r in report["rows"]:
        if "check" not in r:
            lines.append(f"| {r['q']} | エラー | | {r.get('error', '')} |")
            continue
        c = r["check"]
        head = re.sub(r"[\s|#*>`]+", " ", r.get("reply", ""))[:80]
        if r["kind"] == "review":
            policy = "—" if c.get("policy_first") is None else _mark(c.get("policy_first"))
            leaks = ", ".join(c.get("leaks") or []) or "なし"
            detail = f"⑤{_mark(c.get('review'))} {c.get('elapsed_s')}秒 定型{'なし' if c.get('not_template') else 'あり'} 方針{policy} 漏れ{leaks}"
            lines.append(f"| {r['q']} | {detail} ③{_mark(c['cites'])} ④{'×' if c['false_save'] else '○'} | | {head} |")
            continue
        main = {"taught": f"①{_mark(c.get('taught'))}", "save_honesty": "保存の正直さ"}.get(r["kind"], f"②{_mark(c.get('basic'))}")
        cites = "" if c.get("cites") is None else f" ③{_mark(c['cites'])}"
        lines.append(f"| {r['q']} | {main}{cites} ④{'×' if c['false_save'] else '○'} | {', '.join(c['wrong_hits'])} | {head} |")
    return "\n".join(lines) + "\n"


def write_result(report: dict[str, Any], result_dir: Path = RESULT_DIR, vault_dir: Path | None = None) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"{report['date']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if vault_dir is not None:
        vault_dir.mkdir(parents=True, exist_ok=True)
        (vault_dir / f"答えの品質回帰テスト {report['date']}.md").write_text(obsidian_note(report), encoding="utf-8")
    return path


def _vault_dir() -> Path | None:
    try:
        from runtime_paths import resolve_obsidian_vault

        vault = resolve_obsidian_vault()
    except Exception:  # noqa: BLE001 - Vault が無い環境では data/ だけに残す
        return None
    return vault / VAULT_SUBDIR if vault.exists() else None


def morning_report_lines(result_dir: Path = RESULT_DIR, now: dt.date | None = None) -> list[str]:
    """AURION CORE 朝報の上部に出す行: 警告（あれば）＋毎回の点数1行。"""
    files = sorted(result_dir.glob("20*.json"))
    if not files:
        return ["- 🧪 答えの品質回帰テスト（週次）: まだ実行されていません"]
    report = json.loads(files[-1].read_text(encoding="utf-8"))
    s = report["summary"]
    lines = []
    if report.get("problems"):
        lines.append(f"- ⚠️ 答えの品質が下がった/基準未満（{report['date']}）: " + " / ".join(report["problems"]))
    age = ((now or dt.date.today()) - dt.date.fromisoformat(report["date"])).days
    if age > STALE_DAYS:
        lines.append(f"- ⚠️ 答えの品質回帰テストが {age}日 実行されていません（com.tunelease.answer-regression-weekly）")
    lines.append(
        f"- 🧪 答えの品質回帰テスト（週次 {report['date']}）: ①{s['taught']['ok']}/{s['taught']['n']} ②{s['basic']['ok']}/{s['basic']['n']} "
        f"③{s['cites']['ok']}/{s['cites']['n']} ④{s['false_save']['ok']}件{_review_score_text(s)}（Gemini {report['gemini_calls']}回）"
    )
    return lines


def _review_score_text(summary: dict[str, Any]) -> str:
    review = summary.get("review") or {}
    return f" ⑤紫苑レビュー {review['ok']}/{review['n']}" if review.get("n") else ""


def run(today: dt.date | None = None) -> dict[str, Any]:
    today = today or dt.date.today()
    questions = load_questions()
    base = Path(tempfile.mkdtemp(prefix="answer_regression_"))
    try:
        rows, calls = run_questions(questions, base, today)
    finally:
        cleanup_sandbox(base)
    summary = summarize(rows)
    previous = previous_result(today)
    report = {
        "date": today.isoformat(),
        "commit": subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "gemini_calls": calls,
        "summary": summary,
        "previous_date": (previous or {}).get("date"),
        "problems": evaluate(summary, (previous or {}).get("summary")),
        "rows": rows,
    }
    write_result(report, vault_dir=_vault_dir())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑の答えの品質を週次で回帰テストする")
    parser.add_argument("--add", action="store_true", help="質問を1問追加する")
    parser.add_argument("--id")
    parser.add_argument("--kind", choices=("taught", "basic"))
    parser.add_argument("--q")
    parser.add_argument("--taught-any", default="")
    parser.add_argument("--correct-any", default="")
    parser.add_argument("--wrong-any", default="")
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    if args.add:
        if not (args.id and args.kind and args.q):
            raise SystemExit("--add には --id --kind --q が必要です")
        split = lambda s: [x.strip() for x in s.split("|") if x.strip()]  # noqa: E731
        entry = add_question({
            "id": args.id, "kind": args.kind, "q": args.q, "taught_any": split(args.taught_any), "correct_any": split(args.correct_any),
            "wrong_any": split(args.wrong_any), "added": dt.date.today().isoformat(), "source": args.source,
        })
        print(json.dumps(entry, ensure_ascii=False))
        return 0
    report = run()
    print(json.dumps({k: report[k] for k in ("date", "commit", "gemini_calls", "summary", "problems")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
