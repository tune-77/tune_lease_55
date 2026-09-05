"""審査結果(result)から lieflat-charts 骨格の単体HTMLレポートを生成する。

紫苑レビュー・日次レポート／審査分析画面からのエクスポート機能向け
(shared-ai/skills/lieflat-charts 参照)。scoring_core.run_quick_scoring() が
返す result 辞書をそのまま受け取り、新規の集計・スコア計算は一切行わない。

チャート骨格は shared-ai/skills/lieflat-charts/templates/basics-gallery.html
(F1 Rung Bars / C7 Tick Gauge / C2 Paired Rungs / C1 Tick Rows) と
glance-gallery.html (Diverging Bar) の実装をそのまま流用し、データ部分だけ
このモジュールが差し込む。
"""
from __future__ import annotations

import json
from html import escape
from typing import Any

from constants import (
    APPROVAL_LINE,
    Q_RISK_ATTENTION_LINE,
    Q_RISK_STRONG_WARNING_LINE,
)

_HELPERS_JS = r"""
const INK='#1C1C1A',PAPER='#F0EFEB',MUTED='#8F8E88',GRID='#DEDDD6';
const NS='http://www.w3.org/2000/svg';
const el=(p,t,a)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);p.appendChild(n);return n};
const txt=(p,a,s)=>{const n=el(p,'text',a);n.textContent=s;return n};
const tip=(n,s)=>{const t=document.createElementNS(NS,'title');t.textContent=s;n.appendChild(t)};
const rnd=(i,k)=>Math.abs(((i*73856093)^(k*19349663))%1000)/1000;
const D2R=Math.PI/180;
const pol=(cx,cy,r,deg)=>[cx+r*Math.cos(deg*D2R),cy+r*Math.sin(deg*D2R)];
const obsReveal=(id,fn)=>{
  const n=document.getElementById(id);
  if(!n)return;
  const go=()=>{n.innerHTML='';fn(n)};
  const io=new IntersectionObserver(es=>{if(es[0].isIntersecting){go();io.disconnect()}},{threshold:.3});
  io.observe(n);
  n.style.cursor='pointer';
  n.addEventListener('click',go);
};
const eReveal=(id,opt)=>{
  const node=document.getElementById(id);
  if(!node)return;
  const go=()=>{const g=echarts.getInstanceByDom(node)||echarts.init(node);g.clear();g.setOption(opt)};
  const io=new IntersectionObserver(es=>{if(es[0].isIntersecting){go();io.disconnect()}},{threshold:.3});
  io.observe(node);
  node.style.cursor='pointer';
  node.addEventListener('click',go);
};
"""

_CSS = """
  :root{--bg:#F0EFEB;--ink:#1C1C1A;--muted:#8F8E88;--faint:#C6C5BF;--grid:#DEDDD6}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);font-family:'Inter',sans-serif;color:var(--ink);padding:40px;-webkit-font-smoothing:antialiased}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;max-width:1100px;margin:0 auto}
  .card{background:var(--bg);border-radius:24px;padding:28px 28px 20px}
  .card.wide{grid-column:1/-1}
  h2{font-weight:700;font-size:16.5px;letter-spacing:-.02em;margin-bottom:3px}
  .sub{font-size:11.5px;color:var(--muted);margin-bottom:14px}
  .src{font-size:9.5px;color:var(--faint);margin-top:10px;letter-spacing:.08em;font-weight:500}
  svg{width:100%;max-height:330px;display:block;margin:0 auto}
  .ch{height:320px}
  svg text{font-family:'Inter',sans-serif}
  .pop{transform-box:fill-box;transform-origin:center;animation:pop .5s cubic-bezier(.2,.7,.3,1.3) both}
  @keyframes pop{from{transform:scale(0)}to{transform:none}}
  .fade{animation:fade .9s ease both}
  @keyframes fade{from{opacity:0}}
  @media (prefers-reduced-motion: reduce){.pop,.fade{animation:none!important}}
  .pagehead{max-width:1100px;margin:0 auto 26px;padding:0 4px}
  .pagehead h1{font-size:22px;letter-spacing:-.02em;font-weight:800}
  .pagehead p{font-size:12px;color:#8F8E88;margin-top:5px;line-height:1.7}
  .empty{font-size:12px;color:#8F8E88;padding:24px 0}
"""


def _f(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_breakdown_card(result: dict) -> str:
    asset_score = _f(result.get("asset_score"))
    score_borrower = _f(result.get("score_borrower"))
    score_base = _f(result.get("score_base"))
    score = _f(result.get("score"))
    approval_line = _f(result.get("approval_line"), float(APPROVAL_LINE))
    if score is None or approval_line is None:
        return _empty_card("スコア内訳", "score / approval_line が取得できませんでした")

    rows = [
        ["物件スコア", round(asset_score or 0)],
        ["取引先スコア", round(score_borrower or 0)],
        ["モデル素点", round(score_base or 0)],
    ]
    below = [r[0] for r in rows if r[1] < approval_line]
    if below:
        headline = f"{'・'.join(below)}が承認ライン（{approval_line:.0f}点）を下回る"
    else:
        headline = f"総合{score:.0f}点、内訳はいずれも承認ライン（{approval_line:.0f}点）以上"

    return f"""
  <div class="card">
    <h2>{escape(headline)}</h2>
    <div class="sub">score={score:.0f}（承認ライン{approval_line:.0f}以上） · asset_score / score_borrower / score_base の内訳 · 単位=点</div>
    <svg id="rungs" viewBox="0 0 400 320" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="src">RUNG BARS · MONO-BASIC · scoring_core.run_quick_scoring()</div>
  </div>
  <script>
  (()=>{{
  const D={json.dumps(rows, ensure_ascii=False)};
  const LINE={approval_line};
  obsReveal('rungs',s=>{{
    const x0=i=>96+i*110,base=266,step=2.55,HW=20;
    D.forEach(([name,v],i)=>{{
      const x=x0(i);
      for(let k=0;k<v;k++){{
        const y=base-k*step,w=HW-1.5+rnd(k+1,i+2)*3;
        el(s,'line',{{x1:x-w,y1:y,x2:x+w,y2:y,stroke:INK,'stroke-width':1,
          opacity:.5+rnd(k+2,i+4)*.5,class:'fade',style:`animation-delay:${{i*.08+k*.006}}s`}});
        if(k%10===9)el(s,'circle',{{cx:x+HW+4.5,cy:y,r:.8,fill:'#C6C5BF',
          class:'fade',style:`animation-delay:${{i*.08+k*.006}}s`}});
      }}
      const topY=base-(v-1)*step;
      const num=txt(s,{{x,y:topY-10,'font-size':11,'font-weight':800,fill:INK,'text-anchor':'middle',
        class:'fade',style:`animation-delay:${{.4+i*.08}}s`}},v);
      tip(num,`${{name}} — ${{v}}点`);
      txt(s,{{x,y:base+18,'font-size':7.5,'font-weight':700,fill:MUTED,'text-anchor':'middle',
        'letter-spacing':'.02em',class:'fade',style:`animation-delay:${{i*.08}}s`}},name);
    }});
    const ly=base-LINE*step;
    el(s,'line',{{x1:36,y1:ly,x2:364,y2:ly,stroke:'#8F8E88','stroke-width':.8,'stroke-dasharray':'2 3',class:'fade',style:'animation-delay:.9s'}});
    txt(s,{{x:364,y:ly-4,'font-size':7.5,'font-weight':700,fill:'#6A6963','text-anchor':'end',
      'letter-spacing':'.04em',class:'fade',style:'animation-delay:.95s'}},`承認ライン ${{LINE}}点`);
    el(s,'line',{{x1:28,y1:base+4,x2:372,y2:base+4,stroke:GRID,'stroke-width':.8,class:'fade'}});
    txt(s,{{x:200,y:306,'font-size':7,'font-weight':600,fill:'#B0AFA9','text-anchor':'middle',
      'letter-spacing':'.12em',class:'fade',style:'animation-delay:1s'}},
      'ONE RUNG = 1 POINT · 総合scoreは単純平均ではなく調整後の値');
  }});
  }})();
  </script>
"""


def _q_risk_gauge_card(result: dict) -> str:
    quantum_risk = _f(result.get("quantum_risk"), None)
    if quantum_risk is None:
        return _empty_card("Q_riskゲージ", "quantum_risk が未計算のためスキップしました")

    val = max(0, min(100, round(quantum_risk)))
    if val >= Q_RISK_STRONG_WARNING_LINE:
        zone = "強警戒ゾーン"
    elif val >= Q_RISK_ATTENTION_LINE:
        zone = "要注意ゾーン"
    else:
        zone = "平常ゾーン"

    return f"""
  <div class="card">
    <h2>Q_riskは{escape(zone)}</h2>
    <div class="sub">quantum_risk={val} · 要注意ライン{Q_RISK_ATTENTION_LINE} / 強警戒ライン{Q_RISK_STRONG_WARNING_LINE}</div>
    <svg id="gauge" viewBox="0 0 400 320" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="src">TICK GAUGE · MONO-BASIC · scoring_core.quantum_risk</div>
  </div>
  <script>
  (()=>{{
  const VAL={val}, ZONE1={Q_RISK_ATTENTION_LINE}, ZONE2={Q_RISK_STRONG_WARNING_LINE};
  const ZONE_LABEL={json.dumps(zone, ensure_ascii=False)};
  obsReveal('gauge',s=>{{
    const cx=200,cy=190,R0=104,A0=-195,SW=210;
    for(let k=0;k<100;k++){{
      const a=A0+k/100*SW,inked=k<VAL;
      const len=inked?13+rnd(k+1,3)*6:5+rnd(k+1,7)*2.5;
      const [x1,y1]=pol(cx,cy,R0,a),[x2,y2]=pol(cx,cy,R0+len,a);
      el(s,'line',{{x1,y1,x2,y2,stroke:inked?INK:'#CFCEC7','stroke-width':inked?1:.6,
        class:'fade',style:`animation-delay:${{k*.012}}s`}});
    }}
    [[ZONE1,'要注意'],[ZONE2,'強警戒']].forEach(([m,label])=>{{
      const a=A0+m/100*SW,[dx,dy]=pol(cx,cy,R0-7,a),[tx2,ty2]=pol(cx,cy,R0-22,a);
      el(s,'circle',{{cx:dx,cy:dy,r:1.2,fill:'#6A6963',class:'fade',style:'animation-delay:.8s'}});
      txt(s,{{x:tx2,y:ty2+3,'font-size':7,'font-weight':700,fill:'#6A6963','text-anchor':'middle',
        'letter-spacing':'.04em',class:'fade',style:'animation-delay:.85s'}},label);
      txt(s,{{x:tx2,y:ty2+13,'font-size':6.5,'font-weight':600,fill:'#B0AFA9','text-anchor':'middle',
        class:'fade',style:'animation-delay:.85s'}},m);
    }});
    const aT=A0+VAL/100*SW,[ex,ey]=pol(cx,cy,R0+20,aT);
    el(s,'circle',{{cx:ex,cy:ey,r:2.4,fill:INK,class:'pop',style:'animation-delay:1.1s'}});
    const num=txt(s,{{x:cx,y:cy-4,'font-size':34,'font-weight':800,fill:INK,'text-anchor':'middle',
      class:'fade',style:'animation-delay:1s'}},VAL);
    tip(num,`quantum_risk=${{VAL}}`);
    txt(s,{{x:cx,y:cy+16,'font-size':8,'font-weight':600,fill:MUTED,'text-anchor':'middle',
      'letter-spacing':'.1em',class:'fade',style:'animation-delay:1.05s'}},'Q_RISK · '+ZONE_LABEL);
    txt(s,{{x:200,y:300,'font-size':7,'font-weight':600,fill:'#B0AFA9','text-anchor':'middle',
      'letter-spacing':'.12em',class:'fade',style:'animation-delay:1.2s'}},
      'ONE TICK = 1 PT · INKED = CURRENT VALUE');
  }});
  }})();
  </script>
"""


def _benchmark_card(result: dict) -> str:
    pairs = [
        ("営業利益率", result.get("bench_op_margin"), result.get("user_op_margin")),
        ("自己資本比率", result.get("bench_equity_ratio"), result.get("user_equity_ratio")),
        ("リース費用比率", result.get("bench_lease_cost_ratio"), result.get("user_lease_cost_ratio")),
    ]
    rows = []
    for name, bench, user in pairs:
        bench_f, user_f = _f(bench), _f(user)
        if bench_f is None or user_f is None:
            continue
        rows.append([name, max(round(bench_f), 0), max(round(user_f), 0), round(bench_f, 1), round(user_f, 1)])
    if not rows:
        return _empty_card("自社 vs 業界平均", "user_*/bench_* の財務指標が取得できませんでした")

    below = [r[0] for r in rows if r[2] < r[1]]
    headline = f"{'・'.join(below)}が業界平均を下回る" if below else "いずれの指標も業界平均以上"

    return f"""
  <div class="card">
    <h2>{escape(headline)}</h2>
    <div class="sub">淡色=業界平均(bench_*) · 濃色=自社(user_*) · 単位=% ・各指標は自スケール表示</div>
    <svg id="pairrungs" viewBox="0 0 400 320" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="src">PAIRED RUNGS · MONO-BASIC · user_op_margin / bench_op_margin ほか</div>
  </div>
  <script>
  (()=>{{
  const D={json.dumps(rows, ensure_ascii=False)};
  obsReveal('pairrungs',s=>{{
    const x0=i=>84+i*110,base=258,step=5.4,HW=12;
    D.forEach(([name,was,now,wasLabel,nowLabel],i)=>{{
      const xa=x0(i)-15,xb=x0(i)+15;
      for(let k=0;k<was;k++){{
        const y=base-k*step,w=HW-1.2+rnd(k+1,i+2)*2.4;
        el(s,'line',{{x1:xa-w,y1:y,x2:xa+w,y2:y,stroke:'#B0AFA9','stroke-width':1,
          opacity:.5+rnd(k+2,i+3)*.4,class:'fade',style:`animation-delay:${{i*.08+k*.01}}s`}});
      }}
      for(let k=0;k<now;k++){{
        const y=base-k*step,w=HW-1.2+rnd(k+1,i+7)*2.4;
        el(s,'line',{{x1:xb-w,y1:y,x2:xb+w,y2:y,stroke:INK,'stroke-width':1,
          opacity:.6+rnd(k+2,i+8)*.4,class:'fade',style:`animation-delay:${{.15+i*.08+k*.01}}s`}});
      }}
      const topB=base-Math.max(now-1,0)*step;
      const num=txt(s,{{x:xb,y:topB-9,'font-size':10.5,'font-weight':800,fill:INK,'text-anchor':'middle',
        class:'fade',style:`animation-delay:${{.5+i*.08}}s`}},nowLabel+'%');
      tip(num,`${{name}} 自社 — ${{nowLabel}}%（業界平均 ${{wasLabel}}%）`);
      txt(s,{{x:xa,y:base-Math.max(was-1,0)*step-9,'font-size':8.5,'font-weight':700,fill:'#B0AFA9','text-anchor':'middle',
        class:'fade',style:`animation-delay:${{.5+i*.08}}s`}},wasLabel+'%');
      txt(s,{{x:x0(i),y:base+18,'font-size':7.5,'font-weight':700,fill:MUTED,'text-anchor':'middle',
        'letter-spacing':'.02em',class:'fade',style:`animation-delay:${{i*.08}}s`}},name);
    }});
    el(s,'line',{{x1:30,y1:base+4,x2:370,y2:base+4,stroke:GRID,'stroke-width':.8,class:'fade'}});
    txt(s,{{x:200,y:306,'font-size':7,'font-weight':600,fill:'#B0AFA9','text-anchor':'middle',
      'letter-spacing':'.12em',class:'fade',style:'animation-delay:1s'}},
      'FAINT = 業界平均(bench_*) · INK = 自社(user_*)');
  }});
  }})();
  </script>
"""


def _q_risk_breakdown_card(result: dict) -> str:
    breakdown = result.get("q_risk_breakdown") or {}
    items = breakdown.get("items") if isinstance(breakdown, dict) else None
    if not items:
        return _empty_card("Q_risk内訳", "q_risk_breakdown.items が空です（財務矛盾ルール非発火、または未計算）")

    rows = []
    for item in items:
        code = item.get("code", "")
        label = item.get("label", "")
        contribution = _f(item.get("contribution"), 0.0) or 0.0
        rows.append([f"{code} {label}".strip(), round(contribution, 1)])
    rows.sort(key=lambda r: r[1], reverse=True)
    top_name = rows[0][0] if rows else ""

    return f"""
  <div class="card">
    <h2>Q_riskの主因は{escape(top_name)}</h2>
    <div class="sub">q_risk_breakdown.items（財務矛盾ルールのうち発火した項目）· 単位=点</div>
    <svg id="tickrows" viewBox="0 0 400 320" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="src">TICK ROWS · MONO-BASIC · quantum_analysis_module.compute_simple_q_risk()</div>
  </div>
  <script>
  (()=>{{
  const D={json.dumps(rows, ensure_ascii=False)};
  obsReveal('tickrows',s=>{{
    const y0=i=>60+i*60,X0=176,PX=8.6;
    D.forEach(([name,v],i)=>{{
      const y=y0(i);
      txt(s,{{x:166,y:y+3,'font-size':8,'font-weight':700,fill:'#6A6963','text-anchor':'end',
        'letter-spacing':'.01em',class:'fade',style:`animation-delay:${{i*.08}}s`}},name);
      el(s,'line',{{x1:X0,y1:y+9,x2:X0+20*PX,y2:y+9,stroke:GRID,'stroke-width':.6,
        class:'fade',style:`animation-delay:${{i*.08}}s`}});
      for(let k=0;k<Math.max(Math.round(v),0);k++){{
        const x=X0+k*PX+PX/2,h=9+rnd(k+1,i+2)*6;
        el(s,'line',{{x1:x,y1:y+9,x2:x,y2:y+9-h,stroke:INK,'stroke-width':.9,
          opacity:.55+rnd(k+3,i+5)*.45,class:'fade',style:`animation-delay:${{i*.08+k*.02}}s`}});
        if(k%5===4)el(s,'circle',{{cx:x,cy:y+13,r:.8,fill:'#C6C5BF',
          class:'fade',style:`animation-delay:${{i*.08+k*.02}}s`}});
      }}
      const lab=txt(s,{{x:X0+v*PX+10,y:y+4,'font-size':11,'font-weight':800,fill:INK,
        class:'fade',style:`animation-delay:${{.4+i*.08}}s`}},v);
      tip(lab,`${{name}} — 寄与+${{v}}点`);
    }});
    txt(s,{{x:200,y:308,'font-size':7,'font-weight':600,fill:'#B0AFA9','text-anchor':'middle',
      'letter-spacing':'.12em',class:'fade',style:'animation-delay:.9s'}},
      'ONE TICK = 1 PT');
  }});
  }})();
  </script>
"""


def _score_contributions_card(result: dict) -> str:
    contributions = result.get("score_contributions") or []
    top5 = sorted(contributions, key=lambda c: abs(_f(c.get("contribution"), 0.0) or 0.0), reverse=True)[:5]
    if not top5:
        return _empty_card("スコア寄与度トップ5", "score_contributions が空です")

    rows = [[c.get("label_ja", c.get("feature", "")), round(_f(c.get("contribution"), 0.0) or 0.0, 2)] for c in top5]
    top_positive = next((r[0] for r in rows if r[1] > 0), None)
    top_negative = next((r[0] for r in rows if r[1] < 0), None)
    if top_positive and top_negative:
        headline = f"{top_positive}がスコアを押し上げ、{top_negative}が押し下げる"
    elif top_positive:
        headline = f"{top_positive}が最大の加点要因"
    elif top_negative:
        headline = f"{top_negative}が最大の減点要因"
    else:
        headline = "上位寄与度は概ね中立"

    return f"""
  <div class="card wide">
    <h2>{escape(headline)}</h2>
    <div class="sub">score_contributions 上位5件（ロジット空間の寄与度、|寄与度|降順）· 正=加点要因 ・負=減点要因</div>
    <div class="ch" id="divbar"></div>
    <div class="src">DIVERGING BAR · MONO-GLANCE · scoring_core.compute_score_contributions()</div>
  </div>
  <script>
  (()=>{{
  const D={json.dumps(rows, ensure_ascii=False)};
  eReveal('divbar',{{
    animationDuration:900,animationEasing:'quarticOut',animationDelay:i=>i*80,
    tooltip:{{backgroundColor:INK,borderWidth:0,textStyle:{{color:'#F0EFEB',fontFamily:'Inter',fontSize:12}},padding:[10,14],
      formatter:p=>p.name+' — '+(p.value>0?'+':'')+p.value.toFixed(2)}},
    grid:{{left:210,right:60,top:8,bottom:8}},
    xAxis:{{type:'value',splitLine:{{lineStyle:{{color:'#DEDDD6'}}}},
      axisLine:{{show:false}},axisTick:{{show:false}},axisLabel:{{show:false}}}},
    yAxis:{{type:'category',data:D.map(d=>d[0]),inverse:true,
      axisLine:{{show:false}},axisTick:{{show:false}},
      axisLabel:{{color:'#6A6963',fontFamily:'Inter',fontSize:11,fontWeight:600}}}},
    series:[{{
      type:'bar',barWidth:20,
      data:D.map(([n,v])=>({{name:n,value:v,
        itemStyle:{{color:v>0?INK:'#B0AFA9',
          borderRadius:v>0?[0,9,9,0]:[9,0,0,9]}}}})),
      label:{{show:true,fontFamily:'Inter',fontSize:12,fontWeight:700,
        position:'outside',
        formatter:p=>(p.value>0?'+':'')+p.value.toFixed(2),color:INK}},
      markLine:{{symbol:'none',silent:true,label:{{show:false}},
        lineStyle:{{color:'#8F8E88',width:1.5}},
        data:[{{xAxis:0}}]}},
    }}],
  }});
  }})();
  </script>
"""


def _empty_card(title: str, message: str) -> str:
    return f"""
  <div class="card">
    <h2>{escape(title)}</h2>
    <div class="empty">{escape(message)}</div>
  </div>
"""


def render_screening_report_html(result: dict, case_label: str = "") -> str:
    """審査結果(result)から5枚構成のHTMLレポート文字列を生成する。

    result は scoring_core.run_quick_scoring() の返却値をそのまま渡す。
    未計算・欠損のフィールドがあるカードは、チャートの代わりに
    その旨のメッセージを表示する（新規の値の推定・補完は行わない）。
    """
    title = f"審査分析レポート{' — ' + case_label if case_label else ''}"
    cards = "".join([
        _score_breakdown_card(result),
        _q_risk_gauge_card(result),
        _benchmark_card(result),
        _q_risk_breakdown_card(result),
        _score_contributions_card(result),
    ])

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@6/dist/echarts.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="pagehead">
  <h1>{escape(title)}</h1>
  <p>lieflat-charts skill（shared-ai/skills/lieflat-charts）のテンプレートを使用して自動生成。数値は審査結果(result)からそのまま転記しており、このレポート生成処理では新規のスコア計算は行っていません。</p>
</div>
<script>
{_HELPERS_JS}
</script>
<div class="grid2">
{cards}
</div>
</body>
</html>
"""
