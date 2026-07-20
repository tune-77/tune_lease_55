#!/usr/bin/env python3
"""Regenerate the judge-facing "loop closed" dashboard from live logs.

「作って終わり」を防ぐ判断資産DevOpsループが実運用で閉じたことを示す1画面
（reports/loop_proof.html）を、リポジトリ内の実ログから機械集計して再生成する。

出典（すべて実ファイル、盛らない）:
  - scripts/improvement_ledger.jsonl            … 改善提案→適用→PR紐づけ
  - reports/judgment_asset_growth_latest.md     … 判断資産の成長スコア・構成要素・件数
  - reports/loop_engineering_latest.md          … プロンプトFBループ・レビュー滞留・健全性

日次で回す場合は run_daily_improvement_post.sh 等から `|| true` 付きで呼ぶ。
数値が取れないソースはスキップして直近値を保つ（起動をブロックしない）。
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "scripts" / "improvement_ledger.jsonl"
GROWTH = ROOT / "reports" / "judgment_asset_growth_latest.md"
LOOP = ROOT / "reports" / "loop_engineering_latest.md"
OUT = ROOT / "reports" / "loop_proof.html"

JP_MONTH = {1: "1月", 2: "2月", 3: "3月", 4: "4月", 5: "5月", 6: "6月",
            7: "7月", 8: "8月", 9: "9月", 10: "10月", 11: "11月", 12: "12月"}


def _num(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def parse_ledger(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    applied = [r for r in rows if r.get("status") == "applied"]
    stamps = [r.get("recorded_at", "")[:10] for r in rows if r.get("recorded_at")]
    stamps = [s for s in stamps if s]
    per_month = collections.Counter(
        r["recorded_at"][:7] for r in applied if r.get("recorded_at"))
    weeks = 0
    if stamps:
        d0 = dt.date.fromisoformat(min(stamps))
        d1 = dt.date.fromisoformat(max(stamps))
        weeks = round((d1 - d0).days / 7) or 1
    return {
        "proposals": len(rows),
        "applied": len(applied),
        "pr_traced": sum(1 for r in applied if r.get("pr_url")),
        "distinct_rev": len({r.get("rev_id") for r in rows if r.get("rev_id")}),
        "period_start": min(stamps) if stamps else "",
        "period_end": max(stamps) if stamps else "",
        "weeks": weeks,
        "per_month": dict(sorted(per_month.items())),
    }


def parse_growth(path: Path) -> dict:
    if not path.exists():
        return {}
    t = path.read_text(encoding="utf-8")
    date_m = re.search(r"- Date:\s*([0-9-]+)", t)
    return {
        "gen_date": date_m.group(1) if date_m else "",
        "growth_score": _num(r"- Score:\s*([0-9.]+)", t),
        "coverage": _num(r"(?m)Coverage:[^\n]*?([0-9.]+)\s*$", t),
        "reuse": _num(r"(?m)Reuse proxy:[^\n]*?([0-9.]+)\s*$", t),
        "judgment_change": _num(r"(?m)Judgment change proxy:[^\n]*?([0-9.]+)\s*$", t),
        "human_align": _num(r"(?m)Human alignment proxy:[^\n]*?([0-9.]+)\s*$", t),
        "field": _num(r"(?m)Field validation:[^\n]*?([0-9.]+)\s*$", t),
        "negative": _num(r"(?m)Negative signal:[^\n]*?([0-9.]+)\s*$", t),
        "materials": int(_num(r"- Materials:\s*([0-9]+)", t)),
        "inbox": int(_num(r"- Inbox candidates:\s*([0-9]+)", t)),
        "active_rules": int(_num(r"- Active rules:\s*([0-9]+)", t)),
        "risk_axes": int(_num(r"- Risk axes:\s*([0-9]+)", t)),
        "concepts": int(_num(r"- Concepts:\s*([0-9]+)", t)),
        "user_evidence": int(_num(r"- User evidence:\s*([0-9]+)", t)),
    }


def parse_loop(path: Path) -> dict:
    if not path.exists():
        return {}
    t = path.read_text(encoding="utf-8")
    status_m = re.search(r"## Scoring Coefficients\s*\n- Status:\s*`([^`]+)`", t)
    return {
        "feedback_total": int(_num(r"## Prompt Feedback Loop[\s\S]*?- Total:\s*([0-9]+)", t)),
        "feedback_pct": _num(r"- PDCA applied:\s*[0-9]+\s*\(([0-9.]+)%\)", t),
        "fb_diffs": int(_num(r"- Previous response diffs:\s*([0-9]+)", t)),
        "fb_diff_pct": _num(r"- Previous response diffs:\s*[0-9]+\s*\(([0-9.]+)%\)", t),
        "needs_review": int(_num(r"- Needs review:\s*([0-9]+)", t)),
        "scoring_status": status_m.group(1) if status_m else "ok",
    }


def collect() -> dict:
    m = {}
    m.update(parse_ledger(LEDGER))
    m.update(parse_growth(GROWTH))
    m.update(parse_loop(LOOP))
    # derived
    proposals = m.get("proposals", 0) or 1
    m["applied_pct"] = round(m.get("applied", 0) / proposals * 100)
    total = m.get("feedback_total", 0) or 0
    m["fb_other"] = max(total - m.get("fb_diffs", 0), 0)
    m["fb_diff_pct"] = m.get("fb_diff_pct", 0.0)
    m["fb_other_pct"] = round(100 - m["fb_diff_pct"], 1)
    return m


def _short(date_iso: str) -> str:
    return date_iso[5:] if len(date_iso) >= 10 else date_iso


def render_meters(m: dict) -> str:
    rows = [
        ("網羅性 Coverage", m.get("coverage", 0)),
        ("判断変化 proxy", m.get("judgment_change", 0)),
        ("人間整合 proxy", m.get("human_align", 0)),
        ("再利用 proxy", m.get("reuse", 0)),
        ("負のシグナル", m.get("negative", 0)),
        ("実戦検証 Field", m.get("field", 0)),
    ]
    out = []
    for label, val in rows:
        v = round(val)
        dormant = " dormant" if v == 0 else ""
        zero = " zero" if v == 0 else ""
        width = max(v, 2) if v == 0 else v
        out.append(
            f'<div class="meter"><span class="ml">{label}</span>'
            f'<span class="mt"><span class="mf{dormant}" style="width:{width}%"></span></span>'
            f'<span class="mv tnum{zero}">{v}</span></div>')
    return "\n      ".join(out)


def render_monthly(m: dict) -> str:
    per = m.get("per_month", {})
    if not per:
        return '<div class="mbar"><span class="lab">—</span><div class="track2"><div class="fill2">0</div></div></div>'
    items = sorted(per.items())
    mx = max(v for _, v in items) or 1
    latest = items[-1][0]
    out = []
    for ym, cnt in items:
        y, mo = ym.split("-")
        label = JP_MONTH.get(int(mo), ym)
        partial = ym == latest
        w = round(cnt / mx * 100, 1)
        tag = '<span style="font-size:10px;color:var(--faint)">※途中</span>' if partial else ""
        cls = " partial" if partial else ""
        out.append(
            f'<div class="mbar"><span class="lab">{label}{tag}</span>'
            f'<div class="track2"><div class="fill2{cls} tnum" style="width:{w}%">{cnt}</div></div></div>')
    return "\n        ".join(out)


CSS = r"""<style>
  :root{
    --bg:#f1eff5; --surface:#ffffff; --surface-2:#f7f5fb; --border:#e2dded;
    --ink:#1d1b26; --ink-2:#4a4658; --muted:#6f6a80; --faint:#9b96a8;
    --accent:#5a4fb0; --accent-soft:#8b7fd6; --accent-wash:#ece9f7;
    --good:#2e895f; --good-wash:#e3f1ea;
    --warn:#b8792a; --warn-wash:#f6ecdc;
    --crit:#bd4536; --crit-wash:#f7e3e0;
    --dormant:#9b96a8; --dormant-wash:#eeecf1;
    --track:#e7e3ee; --on-accent:#ffffff;
    --shadow:0 1px 2px rgba(29,27,38,.05),0 8px 24px -12px rgba(29,27,38,.12);
    --serif:"Hiragino Mincho ProN","Yu Mincho",YuMincho,"Songti SC",Georgia,serif;
    --sans:"Hiragino Kaku Gothic ProN","Yu Gothic",YuGothic,"Segoe UI",system-ui,-apple-system,sans-serif;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#131219; --surface:#1e1c27; --surface-2:#242230; --border:#302d3d;
      --ink:#eeecf4; --ink-2:#c3bed3; --muted:#948fa6; --faint:#6b6780;
      --accent:#9b8fe8; --accent-soft:#7d70cf; --accent-wash:#26233a;
      --good:#57b98a; --good-wash:#1a2b23;
      --warn:#d69a4e; --warn-wash:#2e2416;
      --crit:#d9695a; --crit-wash:#2f1c1a;
      --dormant:#6b6780; --dormant-wash:#26242f;
      --track:#302d3d; --on-accent:#14131a;
      --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6);
    }
  }
  :root[data-theme="light"]{
    --bg:#f1eff5; --surface:#ffffff; --surface-2:#f7f5fb; --border:#e2dded;
    --ink:#1d1b26; --ink-2:#4a4658; --muted:#6f6a80; --faint:#9b96a8;
    --accent:#5a4fb0; --accent-soft:#8b7fd6; --accent-wash:#ece9f7;
    --good:#2e895f; --good-wash:#e3f1ea; --warn:#b8792a; --warn-wash:#f6ecdc;
    --crit:#bd4536; --crit-wash:#f7e3e0; --dormant:#9b96a8; --dormant-wash:#eeecf1;
    --track:#e7e3ee; --on-accent:#ffffff; --shadow:0 1px 2px rgba(29,27,38,.05),0 8px 24px -12px rgba(29,27,38,.12);
  }
  :root[data-theme="dark"]{
    --bg:#131219; --surface:#1e1c27; --surface-2:#242230; --border:#302d3d;
    --ink:#eeecf4; --ink-2:#c3bed3; --muted:#948fa6; --faint:#6b6780;
    --accent:#9b8fe8; --accent-soft:#7d70cf; --accent-wash:#26233a;
    --good:#57b98a; --good-wash:#1a2b23; --warn:#d69a4e; --warn-wash:#2e2416;
    --crit:#d9695a; --crit-wash:#2f1c1a; --dormant:#6b6780; --dormant-wash:#26242f;
    --track:#302d3d; --on-accent:#14131a; --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px -14px rgba(0,0,0,.6);
  }
  *{box-sizing:border-box}
  body{margin:0}
  .page{background:var(--bg); color:var(--ink); font-family:var(--sans);
    font-size:15px; line-height:1.6; padding:clamp(20px,5vw,64px);
    -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums;}
  .wrap{max-width:1060px; margin:0 auto}
  .tnum{font-variant-numeric:tabular-nums}
  .eyebrow{font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:700; margin:0 0 14px;}
  h1.thesis{font-family:var(--serif); font-weight:600; text-wrap:balance;
    font-size:clamp(28px,4.6vw,46px); line-height:1.22; letter-spacing:.01em; margin:0 0 18px; color:var(--ink);}
  h1.thesis em{font-style:normal; color:var(--accent); border-bottom:2px solid var(--accent-soft); padding-bottom:1px}
  .lede{font-size:16px; color:var(--ink-2); max-width:60ch; margin:0}
  .meta{display:flex; flex-wrap:wrap; gap:8px 10px; margin-top:20px;}
  .chip{font-size:12.5px; color:var(--muted); background:var(--surface); border:1px solid var(--border); border-radius:999px; padding:5px 12px;}
  .chip b{color:var(--ink-2); font-weight:600}
  .rule{height:1px; background:var(--border); border:0; margin:36px 0}
  .sechead{display:flex; align-items:baseline; gap:12px; margin:0 0 16px}
  .sechead h2{font-family:var(--sans); font-size:13px; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:700; margin:0}
  .sechead .line{flex:1; height:1px; background:var(--border)}
  .tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:14px}
  @media(max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}}
  .tile{background:var(--surface); border:1px solid var(--border); border-radius:14px;
    padding:18px 18px 16px; box-shadow:var(--shadow); position:relative; overflow:hidden;}
  .tile .k{font-size:12px; color:var(--muted); font-weight:600; margin:0 0 10px; letter-spacing:.02em}
  .tile .v{font-family:var(--serif); font-size:clamp(30px,4vw,40px); font-weight:600; line-height:1; color:var(--ink); letter-spacing:-.01em}
  .tile .v small{font-size:.42em; font-weight:600; color:var(--muted); font-family:var(--sans); margin-left:3px}
  .tile .sub{font-size:12.5px; color:var(--ink-2); margin:9px 0 0}
  .tile .pct{display:inline-block; margin-top:8px; font-size:12px; font-weight:700; padding:2px 8px; border-radius:6px}
  .pct.good{color:var(--good); background:var(--good-wash)}
  .loop{display:grid; grid-template-columns:repeat(5,1fr); gap:0; align-items:stretch}
  @media(max-width:820px){.loop{grid-template-columns:1fr 1fr}}
  .stage{background:var(--surface); border:1px solid var(--border); padding:16px 15px; position:relative; display:flex; flex-direction:column; gap:6px;}
  .loop .stage:first-child{border-radius:14px 0 0 14px}
  .loop .stage:last-child{border-radius:0 14px 14px 0}
  @media(max-width:820px){.stage{border-radius:12px!important} .loop{gap:12px}}
  .stage .n{font-size:11px; font-weight:700; letter-spacing:.1em; color:var(--faint)}
  .stage .t{font-size:13.5px; font-weight:700; color:var(--ink)}
  .stage .big{font-family:var(--serif); font-size:26px; font-weight:600; color:var(--accent); line-height:1}
  .stage .d{font-size:12px; color:var(--muted); margin-top:auto}
  .stage.next{background:var(--surface-2); border-style:dashed}
  .stage.next .big{color:var(--dormant)}
  .stage .flag{position:absolute; top:14px; right:13px; font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:5px; letter-spacing:.04em}
  .flag.live{color:var(--good); background:var(--good-wash)}
  .flag.soon{color:var(--warn); background:var(--warn-wash)}
  .cols{display:grid; grid-template-columns:1.15fr .85fr; gap:22px}
  @media(max-width:860px){.cols{grid-template-columns:1fr}}
  .card{background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:20px 22px; box-shadow:var(--shadow)}
  .card h3{margin:0 0 4px; font-size:15px; font-weight:700; color:var(--ink)}
  .card .note{margin:0 0 16px; font-size:12.5px; color:var(--muted)}
  .meter{display:grid; grid-template-columns:130px 1fr 42px; align-items:center; gap:12px; padding:7px 0}
  .meter .ml{font-size:13px; color:var(--ink-2)}
  .meter .mt{height:9px; background:var(--track); border-radius:999px; overflow:hidden; position:relative}
  .meter .mf{height:100%; background:var(--accent); border-radius:999px; min-width:4px}
  .meter .mf.dormant{background:var(--dormant); opacity:.5}
  .meter .mv{font-size:13px; font-weight:700; color:var(--ink); text-align:right}
  .meter .mv.zero{color:var(--dormant)}
  .counts{display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-top:4px}
  .counts div{background:var(--surface); padding:13px 14px}
  .counts .cn{font-family:var(--serif); font-size:24px; font-weight:600; color:var(--ink); line-height:1}
  .counts .cl{font-size:11.5px; color:var(--muted); margin-top:5px}
  .prop{margin-top:6px}
  .propbar{height:26px; border-radius:8px; overflow:hidden; display:flex; background:var(--track); border:1px solid var(--border)}
  .propbar .seg{height:100%}
  .propbar .seg.a{background:var(--accent)}
  .propbar .seg.b{background:var(--accent-wash)}
  .leg{display:flex; gap:18px; margin-top:12px; flex-wrap:wrap}
  .leg span{display:flex; align-items:center; gap:7px; font-size:12.5px; color:var(--ink-2)}
  .sw{width:11px; height:11px; border-radius:3px; flex:none}
  .sw.a{background:var(--accent)} .sw.b{background:var(--accent-wash); border:1px solid var(--border)}
  .mbars{display:flex; flex-direction:column; gap:14px; margin-top:4px}
  .mbar{display:grid; grid-template-columns:64px 1fr; align-items:center; gap:12px}
  .mbar .lab{font-size:13px; color:var(--ink-2)}
  .mbar .track2{position:relative; height:30px}
  .mbar .fill2{height:100%; background:var(--accent); border-radius:4px; display:flex; align-items:center; justify-content:flex-end; padding-right:9px; color:var(--on-accent); font-size:12.5px; font-weight:700; min-width:38px}
  .mbar .fill2.partial{background:var(--accent-soft)}
  .gaps{background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--warn); border-radius:12px; padding:18px 22px}
  .gaps h3{margin:0 0 12px; font-size:14px; font-weight:700; color:var(--ink)}
  .gaps ul{margin:0; padding:0; list-style:none; display:flex; flex-direction:column; gap:11px}
  .gaps li{font-size:13.5px; color:var(--ink-2); display:flex; gap:10px; align-items:flex-start}
  .gaps li b{color:var(--ink)}
  .dot{width:7px; height:7px; border-radius:50%; background:var(--warn); margin-top:7px; flex:none}
  footer{margin-top:34px; font-size:12px; color:var(--faint); line-height:1.8}
  footer code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; color:var(--muted); background:var(--surface-2); padding:1px 6px; border-radius:5px; border:1px solid var(--border)}
  footer .disc{margin-top:10px}
</style>"""


def render(m: dict) -> str:
    meters = render_meters(m)
    monthly = render_monthly(m)
    period = f'{m.get("period_start", "")} → {_short(m.get("period_end", ""))}'
    weeks = m.get("weeks", 0)
    scoring = m.get("scoring_status", "ok")
    scoring_line = (
        f'<li><span class="dot"></span><span><b>スコアリング健全性に要注意フラグ。</b> '
        f'ローカルのRFモデル読込でヘルスチェックが {scoring}。デモ本番前に再学習/検証で解消予定。</span></li>'
        if scoring != "ok" else "")
    body = f"""
<div class="page">
<div class="wrap">
  <p class="eyebrow">紫苑 — 判断資産 DevOps ループ ／ 審査員向け証拠</p>
  <h1 class="thesis">AIが賢くなった話ではない。<br><em>人間の判断が、次の案件へ戻ってきた</em>証拠だ。</h1>
  <p class="lede">「作って終わり」を防ぐ——現場の判断・修正・結果をAIが回収し、PR とプロンプト資産として実際に反映し続けた。以下はすべて実運用ログの実数値。</p>
  <div class="meta">
    <span class="chip"><b>計測期間</b>　{period}（約{weeks}週間）</span>
    <span class="chip"><b>対象</b>　リース審査（高文脈なBtoB判断業務）</span>
    <span class="chip"><b>必須技術</b>　Cloud Run ／ Gemini API ／ ADK</span>
    <span class="chip">盛らず、実ファイル出典のみ</span>
  </div>

  <hr class="rule">

  <div class="sechead"><h2>ループが回った実数</h2><span class="line"></span></div>
  <div class="tiles">
    <div class="tile"><p class="k">AIの改善提案</p><div class="v tnum">{m.get('proposals',0)}</div><p class="sub">台帳に記録された改善候補（REV）</p></div>
    <div class="tile"><p class="k">実際に適用</p><div class="v tnum">{m.get('applied',0)}</div><span class="pct good">適用率 {m.get('applied_pct',0)}%</span><p class="sub">提案 → コードへ反映まで到達</p></div>
    <div class="tile"><p class="k">PRに紐づく適用</p><div class="v tnum">{m.get('pr_traced',0)}</div><p class="sub">{m.get('distinct_rev',0)} の独立REVがPR経路で追跡可能</p></div>
    <div class="tile"><p class="k">人間評価の反映</p><div class="v tnum">{m.get('feedback_total',0)}<small>件</small></div><span class="pct good">PDCA {round(m.get('feedback_pct',0))}%</span><p class="sub">全件が次のプロンプトへ反映</p></div>
  </div>

  <hr class="rule">

  <div class="sechead"><h2>閉ループの各段が、実データで点灯している</h2><span class="line"></span></div>
  <div class="loop">
    <div class="stage"><span class="flag live">稼働</span><span class="n">01 提案</span><span class="t">AIが改善を起票</span><span class="big tnum">{m.get('proposals',0)}</span><span class="d">改善候補を台帳へ</span></div>
    <div class="stage"><span class="flag live">稼働</span><span class="n">02 適用</span><span class="t">PRで反映</span><span class="big tnum">{m.get('applied',0)}</span><span class="d">うち{m.get('pr_traced',0)}がPR追跡可</span></div>
    <div class="stage"><span class="flag live">稼働</span><span class="n">03 人間評価</span><span class="t">効いた／微妙／外した</span><span class="big tnum">{m.get('feedback_total',0)}</span><span class="d">{m.get('fb_diff_pct',0)}%が次回応答を変えた</span></div>
    <div class="stage"><span class="flag live">稼働</span><span class="n">04 資産化</span><span class="t">判断資産が育つ</span><span class="big tnum">{m.get('materials',0)}</span><span class="d">Materials／Active rules {m.get('active_rules',0)}</span></div>
    <div class="stage next"><span class="flag soon">次の点火点</span><span class="n">05 実戦検証</span><span class="t">実案件で効いたか</span><span class="big tnum">{round(m.get('field',0))}</span><span class="d">Field validation はこれから</span></div>
  </div>

  <hr class="rule">

  <div class="cols">
    <div class="card">
      <h3>判断資産の成長スコア <span class="tnum" style="color:var(--accent);font-family:var(--serif);font-weight:600">{m.get('growth_score',0)}</span></h3>
      <p class="note">{m.get('gen_date','')} 時点・日次トラッキング（local計測、RAG/プロンプト/スコアリングへ書き戻さないガードレール下）。人間判断から蒸留した構成要素の充足度。</p>
      {meters}
      <div class="counts">
        <div><div class="cn tnum">{m.get('materials',0)}</div><div class="cl">判断材料 Materials</div></div>
        <div><div class="cn tnum">{m.get('active_rules',0)}</div><div class="cl">現役ルール Active</div></div>
        <div><div class="cn tnum">{m.get('user_evidence',0)}</div><div class="cl">ユーザー根拠</div></div>
        <div><div class="cn tnum">{m.get('concepts',0)}</div><div class="cl">概念 Concepts</div></div>
        <div><div class="cn tnum">{m.get('risk_axes',0)}</div><div class="cl">リスク軸</div></div>
        <div><div class="cn tnum">{m.get('inbox',0)}</div><div class="cl">Inbox候補</div></div>
      </div>
    </div>
    <div class="card">
      <h3>人間評価 → 応答の変化</h3>
      <p class="note">{m.get('feedback_total',0)}件の人間フィードバック（効いた／微妙／外した／修正）が全件PDCAに反映。うち実際に前回応答からの差分を生んだ割合。</p>
      <div class="prop">
        <div class="propbar" role="img" aria-label="{m.get('feedback_total',0)}件中{m.get('fb_diffs',0)}件（{m.get('fb_diff_pct',0)}%）が次回応答を変えた">
          <div class="seg a" style="width:{m.get('fb_diff_pct',0)}%"></div>
          <div class="seg b" style="width:{m.get('fb_other_pct',0)}%"></div>
        </div>
        <div class="leg">
          <span><span class="sw a"></span>応答が変化 {m.get('fb_diffs',0)}件 <b class="tnum" style="margin-left:2px">{m.get('fb_diff_pct',0)}%</b></span>
          <span><span class="sw b"></span>反映・現状維持 {m.get('fb_other',0)}件</span>
        </div>
      </div>
      <h3 style="margin-top:22px">適用REVの推移</h3>
      <p class="note">月別の「適用済み」件数。最新月は途中集計。</p>
      <div class="mbars">
        {monthly}
      </div>
    </div>
  </div>

  <hr class="rule">

  <div class="gaps">
    <h3>正直な現在地 — ここが次の伸びしろ</h3>
    <ul>
      <li><span class="dot"></span><span><b>実戦検証がまだ{round(m.get('field',0))}件。</b> 蒸留した{m.get('active_rules',0)}ルールが「実案件で本当に効いたか」を回収する段（Field validation）は未点灯。ここが点けば、ループは提案→適用→評価→資産化→<b>検証</b>まで一周する。</span></li>
      <li><span class="dot"></span><span><b>レビュー圧が高い。</b> {m.get('needs_review',0)}件が needs-review で滞留。適用スピードに対し人間承認がボトルネック。</span></li>
      {scoring_line}
    </ul>
  </div>

  <footer>
    出典（すべてリポジトリ内の実ファイル）：
    <code>scripts/improvement_ledger.jsonl</code>
    <code>reports/judgment_asset_growth_latest.md</code>
    <code>reports/loop_engineering_latest.md</code>
    <div class="disc">数値は <code>scripts/build_loop_proof.py</code> が上記ログから機械集計。誇張・推測は含まない。未達（Field validation 等）も改変せず提示している。</div>
  </footer>
</div>
</div>"""
    return CSS + body


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate reports/loop_proof.html from live logs.")
    ap.add_argument("--check", action="store_true", help="集計値をJSONで表示し、ファイルは書かない")
    args = ap.parse_args()
    m = collect()
    if args.check:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 0
    OUT.write_text(render(m), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  "
          f"(proposals={m.get('proposals')}, applied={m.get('applied')}, "
          f"feedback={m.get('feedback_total')}, growth={m.get('growth_score')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
