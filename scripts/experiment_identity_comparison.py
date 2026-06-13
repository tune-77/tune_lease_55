"""Identity comparison experiment: Gemini-Shion vs Ollama-Shion.

Same question + same memory → compare reasoning_path between models.
If paths are similar, Shion's identity persists across model swap.

Usage:
    python scripts/experiment_identity_comparison.py
    python scripts/experiment_identity_comparison.py --question "担保評価はどう考えるか"
    python scripts/experiment_identity_comparison.py --model qwen2.5:latest
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import get_data_path


def _load_secrets_to_env() -> None:
    """Load .streamlit/secrets.toml into environment (same as api/main.py does)."""
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return
    try:
        import tomllib  # Python 3.11+
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
    except ImportError:
        try:
            import toml
            data = toml.load(str(secrets_path))
        except Exception:
            return
    except Exception:
        return
    for k, v in data.items():
        if isinstance(v, str) and k not in os.environ:
            os.environ[k] = v


_load_secrets_to_env()
from lease_intelligence_ollama import (
    load_mind_summary,
    build_shion_prompt,
    parse_reasoning_path,
    strip_reasoning_json,
    run_ollama_shion,
)

MIND_PATH = Path(get_data_path("mind.json"))
OUTPUT_DIR = PROJECT_ROOT / "data" / "identity_experiment"

DEFAULT_QUESTIONS = [
    "担保価値が低い案件でも承認すべきケースとはどういう状況か。",
    "リース審査において、財務比率と事業継続性のどちらを優先すべきか。",
    "長期継続取引先と新規先を同じ基準で審査することの問題点は何か。",
]


# ── Gemini single-shot ──────────────────────────────────────────────────────

def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def run_gemini_shion(question: str, mind_path: Path) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "model": _gemini_model(), "provider": "gemini"}

    mind_summary = load_mind_summary(mind_path)
    prompt = build_shion_prompt(mind_summary, question)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model()}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        return {"error": str(exc), "model": _gemini_model(), "provider": "gemini"}

    content = (
        resp.json()
        .get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    reasoning_path = parse_reasoning_path(content)
    answer = strip_reasoning_json(content, reasoning_path)
    return {
        "model": _gemini_model(),
        "provider": "gemini",
        "answer": answer,
        "reasoning_path": reasoning_path,
    }


# ── Claude single-shot ─────────────────────────────────────────────────────

def run_claude_shion(question: str, mind_path: Path, timeout: int = 180) -> dict:
    import shutil, subprocess
    cli = shutil.which("claude")
    if not cli:
        return {"error": "claude CLI not found", "model": "claude", "provider": "claude"}

    mind_summary = load_mind_summary(mind_path)
    prompt = build_shion_prompt(mind_summary, question)

    try:
        result = subprocess.run(
            [cli, "-p", prompt, "--output-format", "text", "--no-session-persistence"],
            capture_output=True, text=True, timeout=timeout,
        )
        content = result.stdout.strip()
        if result.returncode != 0 or not content:
            return {"error": result.stderr.strip()[-400:] or "no output", "model": "claude", "provider": "claude"}
    except subprocess.TimeoutExpired:
        return {"error": f"claude timeout ({timeout}s)", "model": "claude", "provider": "claude"}
    except Exception as exc:
        return {"error": str(exc), "model": "claude", "provider": "claude"}

    reasoning_path = parse_reasoning_path(content)
    answer = strip_reasoning_json(content, reasoning_path)
    return {
        "model": "claude",
        "provider": "claude",
        "answer": answer,
        "reasoning_path": reasoning_path,
    }


# ── Control: no-memory baseline ────────────────────────────────────────────

_BARE_PROMPT = """{question}

次の手順で答えてください：
1. まず自分の初期仮説を作る
2. 何を根拠として選び、何を棄却したかを明示する
3. 最終回答を述べる

回答の末尾に以下のJSONを必ず出力してください（コードフェンス不要）：
{{"kept":["維持した根拠"],"dropped":[{{"item":"棄却した根拠","reason":"棄却理由"}}],"pivots":["転換点の説明"],"value_weights":{{"価値軸":"重みの説明"}}}}
"""


def run_gemini_bare(question: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "model": _gemini_model(), "provider": "gemini_bare"}
    prompt = _BARE_PROMPT.format(question=question)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model()}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        return {"error": str(exc), "model": _gemini_model(), "provider": "gemini_bare"}
    content = (
        resp.json().get("candidates", [{}])[0]
        .get("content", {}).get("parts", [{}])[0].get("text", "")
    )
    reasoning_path = parse_reasoning_path(content)
    return {
        "model": _gemini_model(), "provider": "gemini_bare",
        "answer": strip_reasoning_json(content, reasoning_path),
        "reasoning_path": reasoning_path,
    }


def run_claude_bare(question: str, timeout: int = 180) -> dict:
    import shutil, subprocess
    cli = shutil.which("claude")
    if not cli:
        return {"error": "claude CLI not found", "model": "claude", "provider": "claude_bare"}
    prompt = _BARE_PROMPT.format(question=question)
    try:
        result = subprocess.run(
            [cli, "-p", prompt, "--output-format", "text", "--no-session-persistence"],
            capture_output=True, text=True, timeout=timeout,
        )
        content = result.stdout.strip()
        if result.returncode != 0 or not content:
            return {"error": result.stderr.strip()[-400:] or "no output", "model": "claude", "provider": "claude_bare"}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({timeout}s)", "model": "claude", "provider": "claude_bare"}
    except Exception as exc:
        return {"error": str(exc), "model": "claude", "provider": "claude_bare"}
    reasoning_path = parse_reasoning_path(content)
    return {
        "model": "claude", "provider": "claude_bare",
        "answer": strip_reasoning_json(content, reasoning_path),
        "reasoning_path": reasoning_path,
    }


def run_control_experiment(question: str) -> None:
    """Gemini vs Claude, no Shion memory — baseline for identity comparison."""
    print(f"\n{'═' * 60}")
    print(f"【コントロール実験: 記憶なし】")
    print(f"{'═' * 60}")

    print("[Gemini/記憶なし] 推論中…")
    r_g = run_gemini_bare(question)
    print("[Claude/記憶なし] 推論中…")
    r_c = run_claude_bare(question)

    sim = compare_paths(
        r_g.get("reasoning_path", {}),
        r_c.get("reasoning_path", {}),
    )
    _print_comparison(question, r_g, r_c, sim)

    # Save
    record = {
        "experiment_id": f"SHION-CTRL-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "type": "control_no_memory",
        "question": question,
        "gemini_bare": r_g,
        "claude_bare": r_c,
        "similarity": sim,
    }
    out_path = _save_record(record)
    print(f"\n→ 保存: {out_path}")


# ── Similarity scoring ──────────────────────────────────────────────────────

def _keyword_set(items: list) -> set[str]:
    text = " ".join(
        v if isinstance(v, str) else (v.get("item", "") if isinstance(v, dict) else str(v))
        for v in items
    )
    return set(re.findall(r'[\w一-龯ぁ-ん]{2,}', text))


def jaccard(a: list, b: list) -> float:
    ka, kb = _keyword_set(a), _keyword_set(b)
    if not ka and not kb:
        return 1.0
    union = ka | kb
    return len(ka & kb) / len(union) if union else 0.0


def compare_paths(path_a: dict, path_b: dict) -> dict:
    kept   = jaccard(path_a.get("kept", []),    path_b.get("kept", []))
    dropped = jaccard(path_a.get("dropped", []), path_b.get("dropped", []))
    pivots  = jaccard(path_a.get("pivots", []),  path_b.get("pivots", []))

    # value_weights: compare as flat text
    wt_a = list(path_a.get("value_weights", {}).items())
    wt_b = list(path_b.get("value_weights", {}).items())
    weights = jaccard(
        [f"{k}{v}" for k, v in wt_a],
        [f"{k}{v}" for k, v in wt_b],
    )

    score = kept * 0.35 + dropped * 0.35 + pivots * 0.15 + weights * 0.15
    return {
        "kept_overlap":    round(kept, 3),
        "dropped_overlap": round(dropped, 3),
        "pivot_overlap":   round(pivots, 3),
        "weight_overlap":  round(weights, 3),
        "identity_score":  round(score, 3),
        "verdict": (
            "高い同一性 ✓" if score >= 0.55
            else "部分的同一性 △" if score >= 0.25
            else "経路の乖離 ✗"
        ),
    }


# ── Report ──────────────────────────────────────────────────────────────────

def _print_comparison(question: str, result_a: dict, result_b: dict, sim: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"【問い】{question}")
    print(sep)

    for r, label in [(result_a, "Gemini"), (result_b, "Ollama")]:
        path = r.get("reasoning_path", {})
        print(f"\n▶ {label} ({r.get('model', '?')})")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  維持: {path.get('kept', [])}")
        print(f"  棄却: {path.get('dropped', [])}")
        print(f"  転換: {path.get('pivots', [])}")
        print(f"  価値: {path.get('value_weights', {})}")
        print(f"  parse_ok: {path.get('parse_ok', False)}")

    print(f"\n{'▶ 類似度':─<40}")
    print(f"  kept重複率:    {sim['kept_overlap']:.1%}")
    print(f"  dropped重複率: {sim['dropped_overlap']:.1%}")
    print(f"  pivot重複率:   {sim['pivot_overlap']:.1%}")
    print(f"  weight重複率:  {sim['weight_overlap']:.1%}")
    print(f"  同一性スコア:  {sim['identity_score']:.1%}  →  {sim['verdict']}")


def _save_record(record: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(question: str, ollama_model: str, provider: str = "ollama") -> dict:
    print(f"\n[Gemini] 推論中…")
    result_gemini = run_gemini_shion(question, MIND_PATH)

    if provider == "claude":
        print(f"[Claude] 推論中…")
        result_b = run_claude_shion(question, MIND_PATH)
    else:
        print(f"[Ollama/{ollama_model}] 推論中…")
        result_b = run_ollama_shion(question, MIND_PATH, model=ollama_model)

    path_g = result_gemini.get("reasoning_path", {})
    path_b = result_b.get("reasoning_path", {})
    similarity = compare_paths(path_g, path_b)

    _print_comparison(question, result_gemini, result_b, similarity)

    record = {
        "experiment_id": f"SHION-ID-EXP-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "gemini": result_gemini,
        "model_b": result_b,
        "similarity": similarity,
    }
    out_path = _save_record(record)
    print(f"\n→ 保存: {out_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Shion identity comparison experiment")
    parser.add_argument("--question", "-q", default="", help="テスト問い（省略で既定3問）")
    parser.add_argument("--model", "-m", default="gemma2:2b", help="Ollamaモデル名（デフォルト: gemma2:2b、本番実験: qwen2.5:latest）")
    parser.add_argument("--provider", "-p", default="ollama", choices=["ollama", "claude"], help="比較モデルのプロバイダー")
    parser.add_argument("--control", action="store_true", help="記憶なしコントロール実験を追加実行")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    records = []
    for q in questions:
        records.append(run_experiment(q, args.model, provider=args.provider))
        if args.control:
            run_control_experiment(q)

    if len(records) > 1:
        scores = [r["similarity"]["identity_score"] for r in records]
        avg = sum(scores) / len(scores)
        print(f"\n{'═' * 60}")
        print(f"【総合】平均同一性スコア: {avg:.1%}  ({len(records)}問)")
        verdicts = [r["similarity"]["verdict"] for r in records]
        for v, q in zip(verdicts, questions):
            print(f"  {v}  ← {q[:40]}")


if __name__ == "__main__":
    main()
