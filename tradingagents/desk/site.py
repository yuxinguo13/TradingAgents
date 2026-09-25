"""The report as a web site: one index, one page per name, shared style and script.

    python -m tradingagents.desk site [--date YYYY-MM-DD]

Reads what the report task already wrote under ``desk/report/``: the JSON
pack, the final markdown (``<date>-final.md``, or the pack's own ``<date>.md``
when no final exists), and the per-symbol pages. Writes
``desk/report/site/<date>/`` with ``index.html``, ``desk.css``, ``desk.js``
and ``pages/<SYMBOL>.html``, plus ``files.json`` listing the supporting files
for the Artifact publish. Every ticker on the index links to its page; each
page carries the interactive chart, Claude's write-up for that name when
there is one, the code's own reading, and the full analysis page.
"""

from __future__ import annotations

import html as H
import json
import logging
import re
from datetime import date
from pathlib import Path

from tradingagents.live import charting

from . import task_dir
from .market import SECTOR_ZH, Market

logger = logging.getLogger(__name__)

CSS = '\n:root{\n  --bg:#f4f5f7; --paper:#ffffff; --ink:#131a22; --ink-2:#4a5563; --ink-3:#7b8592; --rule:#dfe3e8; --rule-2:#eef0f3;\n  --accent:#1f4e79; --accent-soft:#e6eef7;\n  --up:#c8353b; --dn:#1e8a4c;\n  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;\n  --surface-1:#fcfcfb;\n  --sans:"Noto Sans SC",system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;\n  --serif:"Noto Serif SC","Songti SC","SimSun",Georgia,serif;\n  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;\n}\n@media (prefers-color-scheme: dark){\n  :root:not([data-theme="light"]){ color-scheme:dark;\n    --bg:#0f1318; --paper:#161b22; --ink:#e8ebef; --ink-2:#b4bcc6; --ink-3:#7f8994; --rule:#2a323c; --rule-2:#1f262e;\n    --accent:#8fb6dd; --accent-soft:#1b2a3a; --up:#e66767; --dn:#3ab36f; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --surface-1:#1a1a19; }\n}\n:root[data-theme="dark"]{ color-scheme:dark;\n  --bg:#0f1318; --paper:#161b22; --ink:#e8ebef; --ink-2:#b4bcc6; --ink-3:#7f8994; --rule:#2a323c; --rule-2:#1f262e;\n  --accent:#8fb6dd; --accent-soft:#1b2a3a; --up:#e66767; --dn:#3ab36f; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --surface-1:#1a1a19; }\n*{box-sizing:border-box}\nbody{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.75;-webkit-font-smoothing:antialiased}\n.wrap{max-width:820px;margin:0 auto;padding-block:0 64px;padding-inline:20px}\nheader.masthead{padding-block:40px 20px;border-bottom:2px solid var(--ink)}\n.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}\nh1{font-family:var(--serif);font-weight:700;font-size:clamp(28px,5vw,40px);line-height:1.2;margin:8px 0 10px;text-wrap:balance}\n.lede{color:var(--ink-2);font-size:15px;margin:0}\nnav.toc{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--rule);margin-inline:-20px;padding-inline:20px}\nnav.toc ul{list-style:none;margin:0;padding:0;display:flex;gap:4px;overflow-x:auto;scrollbar-width:none}\nnav.toc a{display:block;padding:10px 10px;font-size:13px;color:var(--ink-2);text-decoration:none;white-space:nowrap;border-bottom:2px solid transparent}\nnav.toc a:hover,nav.toc a:focus-visible{color:var(--accent);border-color:var(--accent);outline:none}\n.band{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-block:22px 8px}\n.tile{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;min-width:0}\n.tile .k{font-size:12px;color:var(--ink-3)}\n.tile .v{font-family:var(--mono);font-size:20px;font-weight:500;line-height:1.2;margin-top:2px;font-variant-numeric:tabular-nums}\n.tile .d{font-family:var(--mono);font-size:12px;margin-top:2px}\n.tile svg{display:block;width:100%;height:34px;margin-top:6px}\narticle h2{font-family:var(--serif);font-size:24px;font-weight:700;line-height:1.3;margin:44px 0 14px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance;scroll-margin-top:60px}\narticle h3{font-family:var(--serif);font-size:19px;font-weight:700;margin:32px 0 10px;text-wrap:balance}\narticle p{margin:0 0 14px;max-width:70ch}\narticle ul,article ol{padding-left:22px;margin:0 0 14px;max-width:72ch}\narticle li{margin-bottom:6px}\narticle li::marker{color:var(--ink-3)}\narticle a{color:var(--accent);text-decoration:underline;text-decoration-color:color-mix(in srgb,var(--accent) 35%,transparent);text-underline-offset:3px}\narticle a:hover{text-decoration-color:var(--accent)}\narticle hr{border:0;border-top:1px solid var(--rule);margin:36px 0}\narticle table{border-collapse:collapse;font-size:13.5px;margin:10px 0 22px;display:block;overflow-x:auto;white-space:nowrap;font-variant-numeric:tabular-nums;max-width:100%}\narticle thead th{text-align:left;font-weight:500;color:var(--ink-3);font-size:12px;letter-spacing:.04em;border-bottom:1px solid var(--ink);padding:6px 10px 8px}\narticle td{padding:7px 10px;border-bottom:1px solid var(--rule-2);vertical-align:top}\narticle td:first-child{font-family:var(--mono);font-weight:500}\narticle tbody tr:hover td{background:var(--accent-soft)}\n.up{color:var(--up)} .dn{color:var(--dn)}\n.spark{font-family:var(--mono);color:var(--accent);letter-spacing:-1px;font-size:13px}\nfigure.chart{margin:16px 0 20px;background:var(--surface-1);border:1px solid var(--rule);border-radius:6px;padding:12px 12px 8px;max-width:100%}\nfigure.chart .legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink-2);margin:0 0 6px 4px}\nfigure.chart .legend i{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}\nfigure.chart svg{display:block;width:100%;height:auto;font-family:var(--mono);font-size:11px}\nfigure.chart .box{position:relative}\nfigure.chart .tip{position:absolute;pointer-events:none;background:var(--paper);border:1px solid var(--rule);border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:12px;line-height:1.5;color:var(--ink);box-shadow:0 2px 8px rgba(0,0,0,.08);white-space:nowrap}\n.foot{margin-top:40px;padding-top:16px;border-top:2px solid var(--ink);font-size:13px;color:var(--ink-2)}\n@media (max-width:480px){ body{font-size:15px} article h2{font-size:21px} }\n@media (prefers-reduced-motion:reduce){ *{transition:none!important} }\n\n.crumb{font-size:13px;margin:18px 0 0}.crumb a{color:var(--accent);text-decoration:none}\n.badges{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 0}\n.badge{font-family:var(--mono);font-size:12px;padding:3px 9px;border:1px solid var(--rule);border-radius:999px;color:var(--ink-2);background:var(--paper)}\n.badge.score{border-color:var(--accent);color:var(--accent);font-weight:500}\n.mine{background:var(--accent-soft);border-left:3px solid var(--accent);padding:14px 18px;border-radius:0 6px 6px 0;margin:20px 0}\n.mine h3:first-child{margin-top:4px}\n.rule{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:12px 16px;margin:16px 0}\n.rule ul{margin:0}\narticle blockquote{margin:0 0 16px;padding:10px 16px;border-left:3px solid var(--rule);color:var(--ink-2)}\narticle pre{overflow-x:auto;font-family:var(--mono);font-size:12px;line-height:1.35;background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:12px}\n.sym{font-family:var(--mono);font-weight:500}\na.sym{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent) 40%,transparent)}\n'

JS = '(function(){\n  var D = JSON.parse(document.getElementById(\'chartdata\').textContent);\n  var css = getComputedStyle(document.documentElement);\n  function tok(n){ return css.getPropertyValue(n).trim(); }\n  var fmt = function(v,d){ if(v==null) return \'—\'; d=(d==null?2:d); return Number(v).toLocaleString(\'en-US\',{minimumFractionDigits:d,maximumFractionDigits:d}); };\n  var NS=\'http://www.w3.org/2000/svg\';\n  function el(n,a,t){ var e=document.createElementNS(NS,n); for(var k in a) e.setAttribute(k,a[k]); if(t!=null) e.textContent=t; return e; }\n\n  var band=document.getElementById(\'band\'); if(band){\n  var tiles=[[\'^GSPC\',\'标普500\',0],[\'^RUT\',\'罗素2000\',0],[\'^TNX\',\'10年期美债 %\',2],[\'^VIX\',\'VIX\',1],[\'CL=F\',\'原油 WTI\',1],[\'GC=F\',\'黄金\',0],[\'DX-Y.NYB\',\'美元指数\',1]];\n  tiles.forEach(function(t){\n    var s=D.macro[t[0]]; if(!s||!s.close.length) return;\n    var c=s.close, last=c[c.length-1], prev=c[Math.max(0,c.length-22)];\n    var isY=t[0]===\'^TNX\'; var chg=isY?(last-prev):(last/prev-1);\n    var div=document.createElement(\'div\'); div.className=\'tile\';\n    var dtxt=isY?((chg>=0?\'+\':\'\')+chg.toFixed(2)+\' pt / 月\'):((chg>=0?\'+\':\'\')+(chg*100).toFixed(1)+\'% / 月\');\n    div.innerHTML=\'<div class="k">\'+t[1]+\'</div><div class="v">\'+fmt(last,t[2])+\'</div><div class="d \'+(chg>=0?\'up\':\'dn\')+\'">\'+dtxt+\'</div>\';\n    var w=200,h=34,mn=Math.min.apply(null,c),mx=Math.max.apply(null,c),rg=(mx-mn)||1;\n    var pts=c.map(function(v,i){ return [ (i/(c.length-1))*w, h-2-((v-mn)/rg)*(h-4) ]; });\n    var svg=el(\'svg\',{viewBox:\'0 0 \'+w+\' \'+h,\'aria-hidden\':\'true\',preserveAspectRatio:\'none\'});\n    var d=\'M\'+pts.map(function(p){return p[0].toFixed(1)+\',\'+p[1].toFixed(1)}).join(\'L\');\n    svg.appendChild(el(\'path\',{d:d+\'L\'+w+\',\'+h+\'L0,\'+h+\'Z\',fill:tok(\'--accent\'),\'fill-opacity\':\'0.12\'}));\n    svg.appendChild(el(\'path\',{d:d,fill:\'none\',stroke:tok(\'--accent\'),\'stroke-width\':\'1.5\',\'vector-effect\':\'non-scaling-stroke\'}));\n    div.appendChild(svg); band.appendChild(div);\n  }); }\n\n  document.querySelectorAll(\'figure.chart\').forEach(function(fig){\n    var sym=fig.dataset.sym, s=D.stocks[sym]; if(!s) return;\n    var W=760,H=300,L=8,R=64,T=14,B=28, iw=W-L-R, ih=H-T-B;\n    var series=[s.close,s.sma20,s.sma50];\n    var all=[].concat.apply([],series).filter(function(v){return v!=null;}).concat([s.stop,s.target,s.entry].filter(function(v){return v!=null;}));\n    var mn=Math.min.apply(null,all),mx=Math.max.apply(null,all),pad=(mx-mn)*0.06; mn-=pad; mx+=pad;\n    var n=s.close.length;\n    var x=function(i){ return L+(i/(n-1))*iw; }, y=function(v){ return T+ih-((v-mn)/(mx-mn))*ih; };\n    var legend=document.createElement(\'div\'); legend.className=\'legend\';\n    legend.innerHTML=\'<span><i style="border-color:\'+tok(\'--s1\')+\'"></i>收盘 \'+fmt(s.close[n-1])+\'</span><span><i style="border-color:\'+tok(\'--s2\')+\'"></i>MA20 \'+fmt(s.sma20[n-1])+\'</span><span><i style="border-color:\'+tok(\'--s3\')+\'"></i>MA50 \'+fmt(s.sma50[n-1])+\'</span><span style="color:\'+tok(\'--ink-3\')+\'">\'+sym+\' · 近 \'+n+\' 个交易日 · 虚线：入场 / 止损 / 目标</span>\';\n    fig.appendChild(legend);\n    var box=document.createElement(\'div\'); box.className=\'box\'; fig.appendChild(box);\n    var svg=el(\'svg\',{viewBox:\'0 0 \'+W+\' \'+H,role:\'img\',\'aria-label\':sym+\' 收盘价与均线\'}); box.appendChild(svg);\n    var ticks=4; for(var k=0;k<=ticks;k++){ var v=mn+(mx-mn)*k/ticks; svg.appendChild(el(\'line\',{x1:L,x2:L+iw,y1:y(v),y2:y(v),stroke:tok(\'--rule-2\'),\'stroke-width\':\'1\'})); svg.appendChild(el(\'text\',{x:L+iw+6,y:y(v)+4,fill:tok(\'--ink-3\')},fmt(v))); }\n    [0,Math.floor(n/2),n-1].forEach(function(i,j){ svg.appendChild(el(\'text\',{x:x(i),y:H-8,fill:tok(\'--ink-3\'),\'text-anchor\':j===0?\'start\':(j===2?\'end\':\'middle\')},s.dates[i])); });\n    [[\'入场\',s.entry,tok(\'--ink-2\')],[\'止损\',s.stop,tok(\'--up\')],[\'目标\',s.target,tok(\'--dn\')]].forEach(function(lv){\n      if(lv[1]==null) return; var yy=y(lv[1]);\n      svg.appendChild(el(\'line\',{x1:L,x2:L+iw,y1:yy,y2:yy,stroke:lv[2],\'stroke-width\':\'1\',\'stroke-dasharray\':\'4 4\',\'stroke-opacity\':\'0.8\'}));\n      svg.appendChild(el(\'text\',{x:L+4,y:(lv[0]===\'止损\'?yy+12:yy-4),fill:lv[2],\'font-size\':\'11\'},lv[0]+\' \'+fmt(lv[1])));\n    });\n    var cols=[tok(\'--s1\'),tok(\'--s2\'),tok(\'--s3\')], widths=[\'2\',\'1.5\',\'1.5\'];\n    [2,1,0].forEach(function(si){\n      var d=\'\',started=false;\n      series[si].forEach(function(v,i){ if(v==null){started=false;return;} d+=(started?\'L\':\'M\')+x(i).toFixed(1)+\',\'+y(v).toFixed(1); started=true; });\n      if(si===0){ var area=d+\'L\'+x(n-1).toFixed(1)+\',\'+(T+ih)+\'L\'+x(0).toFixed(1)+\',\'+(T+ih)+\'Z\'; svg.appendChild(el(\'path\',{d:area,fill:cols[0],\'fill-opacity\':\'0.06\'})); }\n      svg.appendChild(el(\'path\',{d:d,fill:\'none\',stroke:cols[si],\'stroke-width\':widths[si],\'stroke-linejoin\':\'round\'}));\n    });\n    svg.appendChild(el(\'circle\',{cx:x(n-1),cy:y(s.close[n-1]),r:\'3.5\',fill:cols[0],stroke:tok(\'--surface-1\'),\'stroke-width\':\'2\'}));\n    var cross=el(\'line\',{x1:0,x2:0,y1:T,y2:T+ih,stroke:tok(\'--ink-3\'),\'stroke-width\':\'1\',\'stroke-dasharray\':\'2 3\',visibility:\'hidden\'}); svg.appendChild(cross);\n    var dot=el(\'circle\',{r:\'4\',fill:cols[0],stroke:tok(\'--surface-1\'),\'stroke-width\':\'2\',visibility:\'hidden\'}); svg.appendChild(dot);\n    var tip=document.createElement(\'div\'); tip.className=\'tip\'; tip.hidden=true; box.appendChild(tip);\n    function move(ev){\n      var r=svg.getBoundingClientRect(); var px=(ev.clientX-r.left)*(W/r.width); var i=Math.round(((px-L)/iw)*(n-1)); i=Math.max(0,Math.min(n-1,i));\n      cross.setAttribute(\'x1\',x(i)); cross.setAttribute(\'x2\',x(i)); cross.setAttribute(\'visibility\',\'visible\');\n      dot.setAttribute(\'cx\',x(i)); dot.setAttribute(\'cy\',y(s.close[i])); dot.setAttribute(\'visibility\',\'visible\');\n      tip.hidden=false; tip.innerHTML=s.dates[i]+\'<br>收盘 \'+fmt(s.close[i])+\'<br>MA20 \'+fmt(s.sma20[i])+\' · MA50 \'+fmt(s.sma50[i]);\n      var left=(x(i)/W)*r.width; tip.style.left=(left+(left>r.width*0.6?-tip.offsetWidth-12:12))+\'px\'; tip.style.top=Math.max(0,(y(s.close[i])/H)*r.height-40)+\'px\';\n    }\n    function leave(){ cross.setAttribute(\'visibility\',\'hidden\'); dot.setAttribute(\'visibility\',\'hidden\'); tip.hidden=true; }\n    svg.addEventListener(\'mousemove\',move); svg.addEventListener(\'touchmove\',function(e){ if(e.touches[0]) move(e.touches[0]); },{passive:true}); svg.addEventListener(\'mouseleave\',leave);\n  });\n})();\n'

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">'

SPARK = "▁▂▃▄▅▆▇█"


def _md(text: str) -> str:
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def _pct(v, d=1):
    return "—" if v is None or v != v else f'<span class="{"up" if v >= 0 else "dn"}">{v * 100:+.{d}f}%</span>'


def _num(v, d=2):
    return "—" if v is None or v != v else f"{v:,.{d}f}"


def my_scores(final_md: str) -> dict:
    """The ranking table Claude wrote: symbol → (score, entry, stop, target, r)."""
    out = {}
    for m in re.finditer(r"^\|\s*\d+\s*\|\s*([A-Z][A-Z.-]*)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|\s*([\d.,]+|[^|]*?)\s*\|\s*([\d.,]+|[^|]*?)\s*\|\s*([\d.,]+|[^|]*?)\s*\|\s*([\d.]+|[^|]*?)\s*\|", final_md, re.M):
        def f(x):
            try:
                return float(x.replace(",", ""))
            except ValueError:
                return None
        out[m.group(1)] = (int(m.group(2)), f(m.group(3)), f(m.group(4)), f(m.group(5)), f(m.group(6)))
    for m in re.finditer(r"^\s*[-*]\s+\*\*([A-Z][A-Z.-]*)[^*]*?(\d+)\s*分", final_md, re.M):
        out.setdefault(m.group(1), (int(m.group(2)), None, None, None, None))
    return out


def my_notes(final_md: str) -> dict:
    """Claude's write-up per symbol: a whole ### section, or the bullet that names it."""
    notes = {}
    for m in re.finditer(r"^### \d+\. ([A-Z][A-Z.-]*) .*?(?=^### |^## |\Z)", final_md, re.S | re.M):
        txt = re.sub(r"^```.*?```\n", "", m.group(0), flags=re.S | re.M)
        notes[m.group(1)] = re.sub(r"^### \d+\. [A-Z][A-Z.-]* (.*)$", r"### 我的判断：\1", txt, flags=re.M)
    for m in re.finditer(r"^\s*[-*]\s+\*\*([A-Z][A-Z.-]*(?:、[A-Z][A-Z.-]*)*)[^*]*\*\*.*$", final_md, re.M):
        for sym in m.group(1).split("、"):
            notes.setdefault(sym, "### 我的判断\n\n" + m.group(0).strip() + "\n")
    return notes


def _link_syms(frag: str, known: set) -> str:
    frag = re.sub(r"(<td>)([A-Z][A-Z.-]{0,5})(</td>)",
                  lambda m: f'{m.group(1)}<a class="sym" href="pages/{m.group(2)}.html">{m.group(2)}</a>{m.group(3)}' if m.group(2) in known else m.group(0), frag)
    frag = re.sub(r"<strong>([A-Z][A-Z.-]{0,5})</strong>",
                  lambda m: f'<strong><a class="sym" href="pages/{m.group(1)}.html">{m.group(1)}</a></strong>' if m.group(1) in known else m.group(0), frag)
    frag = re.sub(r"<strong>([A-Z][A-Z.-]*(?:、[A-Z][A-Z.-]*)+)</strong>",
                  lambda m: "<strong>" + "、".join(f'<a class="sym" href="pages/{x}.html">{x}</a>' if x in known else x for x in m.group(1).split("、")) + "</strong>", frag)
    frag = re.sub(r"<h3>(\d+\. )([A-Z][A-Z.-]*)( )",
                  lambda m: f'<h3>{m.group(1)}<a class="sym" href="pages/{m.group(2)}.html">{m.group(2)}</a>{m.group(3)}' if m.group(2) in known else m.group(0), frag)
    return frag


def build(when: str | date | None = None, *, market: Market | None = None, out_dir: Path | None = None,
          bars_n: int = 130) -> Path:
    rdir = task_dir("report")
    if when is None:
        finals = sorted(rdir.glob("*-final.md")) or sorted(p for p in rdir.glob("????-??-??.md"))
        if not finals:
            raise FileNotFoundError("no report under " + str(rdir))
        when = finals[-1].name[:10]
    when = str(when)[:10]
    pack = json.loads((rdir / f"{when}.json").read_text(encoding="utf-8"))
    final_path = rdir / f"{when}-final.md"
    final_md = (final_path if final_path.exists() else rdir / f"{when}.md").read_text(encoding="utf-8")
    data_date = pack.get("data_date", "")
    scored = {i["symbol"]: i for i in pack.get("scored", [])}
    mine, notes = my_scores(final_md), my_notes(final_md)
    m = market or Market(task="report")
    out = out_dir or (rdir / "site" / when)
    (out / "pages").mkdir(parents=True, exist_ok=True)

    # bars for every name
    bars = {}
    dd = date.fromisoformat(data_date) if data_date else date.today()
    for sym, i in scored.items():
        f = m.facts(sym, dd)
        if not f.bars.closes:
            continue
        c, d = f.bars.closes, f.bars.dates
        s20, s50 = charting.sma(c, 20), charting.sma(c, 50)
        lv = mine.get(sym)
        e, st, tg = (lv[1], lv[2], lv[3]) if lv and lv[1] else (i.get("entry"), i.get("stop"), i.get("target"))
        fix = lambda v: round(v, 2) if isinstance(v, (int, float)) and v == v else None
        bars[sym] = {"dates": d[-bars_n:], "close": [round(x, 2) for x in c[-bars_n:]],
                     "sma20": [round(x, 2) if x is not None else None for x in s20[-bars_n:]],
                     "sma50": [round(x, 2) if x is not None else None for x in s50[-bars_n:]],
                     "entry": fix(e), "stop": fix(st), "target": fix(tg)}
    macro = {}
    for sym in ("^GSPC", "^RUT", "^TNX", "^VIX", "CL=F", "GC=F", "DX-Y.NYB"):
        b = m.bars(sym, dd)
        if b.closes:
            macro[sym] = {"dates": b.dates[-63:], "close": [round(x, 3) for x in b.closes[-63:]]}
    known = set(bars)

    # the index
    body_md = re.sub(r"^# .*\n", "", final_md, count=1)
    title = re.search(r"^# (.*)$", final_md, re.M)
    title = title.group(1) if title else f"市场日报 · {when}"
    body = _md(body_md)
    body = re.sub(r"<pre><code>(.*?)</code></pre>", lambda mm: (lambda s: f'<figure class="chart" data-sym="{s.group(1)}"></figure>' if s and s.group(1) in known else mm.group(0))(re.search(r"^\s*([A-Z][A-Z.-]*) 近 \d+ 个交易日", H.unescape(mm.group(1)), re.M)), body, flags=re.S)
    body = re.sub(r"<code>([" + SPARK + r"]+)</code>", r'<span class="spark">\1</span>', body)
    for n, key in enumerate(["一、", "二、", "三、", "四、", "五、", "六、", "七、"], 1):
        body = body.replace(f"<h2>{key}", f'<h2 id="s{n}">{key}', 1)
    body = re.sub(r"(<td[^>]*>)(\+\d[\d,.]*%?)(</td>)", r'\1<span class="up">\2</span>\3', body)
    body = re.sub(r"(<td[^>]*>)(-\d[\d,.]*%?)(</td>)", r'\1<span class="dn">\2</span>\3', body)
    rows = ["<table><thead><tr><th>代码</th><th>板块</th><th>现价</th><th>日</th><th>月</th><th>三月</th><th>RSI</th><th>量比</th><th>距200日</th><th>参考分</th><th>我的分</th><th>页</th></tr></thead><tbody>"]
    for i in pack.get("scored", []):
        s = i["symbol"]
        if s not in known:
            continue
        my = mine.get(s)
        rows.append(f'<tr><td><a class="sym" href="pages/{s}.html">{s}</a></td><td>{SECTOR_ZH.get(i["sector"], i["sector"])}</td><td>{_num(i["price"])}</td><td>{_pct(i["change_pct"])}</td><td>{_pct(i["ret_1m"])}</td><td>{_pct(i["ret_3m"])}</td><td>{_num(i["rsi"], 0)}</td><td>{_num(i["vol_ratio"], 1)}</td><td>{_pct(i["ext_200"], 0)}</td><td>{i["score"]:+.0f}</td><td>{my[0] if my else "—"}</td><td><a href="pages/{s}.html">分析 →</a></td></tr>')
    rows.append("</tbody></table>")
    intro = "<p>点代码或「分析」进入每个名字的完整分析页：交互图、我的判断、代码的读数、基本面与财报、新闻、风险、研究链接。</p>"
    if re.search(r'<h2 id="s6">', body):
        body = re.sub(r'(<h2 id="s6">[^<]*</h2>\n)(?:<p>.*?</p>)?(?:<table>.*?</table>)?', lambda mm: mm.group(1) + intro + "\n".join(rows), body, count=1, flags=re.S)
    body = _link_syms(body, known)
    featured = [x for x in re.findall(r'data-sym="([^"]+)"', body) if x in bars]
    lede = "面向下一个交易日。分数是我打的，代码算的参考分只做起点；不看任何账户。红涨绿跌。"
    index = (f"<title>市场日报 · {when}</title>\n{FONTS}\n<link rel=\"stylesheet\" href=\"desk.css\">\n"
             + HEAD % {"eyebrow": f"Trading desk · 市场日报 · 数据截至 {data_date} 收盘", "title": H.escape(re.sub(r"^市场日报[：:·\s]*", "", title)), "lede": lede}
             + f"<article>\n{body}\n</article>\n"
             + "<footer class=\"foot\"><p>由交易台生成：代码出数据（K 线、指标、参考位、财报日期、新闻），Claude 做分析（搜索当天的宏观、政策、全球市场、个股财报与舆论），按操盘手册的维度打分并写理由。所有链接为写稿时可核实的来源。这不是投资建议。</p></footer>\n</div>\n"
             + f'<script id="chartdata" type="application/json">{json.dumps({"stocks": {s: bars[s] for s in featured}, "macro": macro})}</script>\n<script src="desk.js"></script>\n')
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / "desk.css").write_text(CSS, encoding="utf-8")
    (out / "desk.js").write_text(JS, encoding="utf-8")

    # one page per name
    files = ["desk.css", "desk.js"]
    for s, i in scored.items():
        src = rdir / when / f"{s}.md"
        if s not in known or not src.exists():
            continue
        md = src.read_text(encoding="utf-8")
        name = re.sub(r"^# [A-Z.-]+ · ", "", md.splitlines()[0]).strip()
        name = re.sub(r"\s*\(.*?\)", "", name)
        name = re.split(r" American Depositary| Depositary| - Class| Common Stock| Ordinary Shares", name)[0].strip()
        if len(name) > 48:
            name = name[:48].rsplit(" ", 1)[0] + "…"
        md = "\n".join(md.splitlines()[1:])
        md = re.sub(r"^\[← 回到.*?\]\(.*?\)\n", "", md, flags=re.M)
        pb = _md(md)
        pb = re.sub(r'<pre><code class="language-text">\s*' + re.escape(s) + r" · 近.*?</code></pre>", "", pb, count=1, flags=re.S)
        pb = re.sub(r"<code>([" + SPARK + r"]+)</code>", r'<span class="spark">\1</span>', pb)
        my = mine.get(s)
        note = notes.get(s)
        note_html = ('<div class="mine">' + _md(note) + "</div>") if note else \
            '<div class="mine"><h3>我的判断</h3><p>这只没有进入我今天重点分析的名字。下面是代码按同一把尺子给出的读数和详情，供参考；要我看它，说一声。</p></div>'
        rule = '<div class="rule"><strong>代码的读数（参考分 %+.0f）</strong><ul>%s%s</ul></div>' % (
            i["score"], "".join(f"<li>✓ {H.escape(r)}</li>" for r in i.get("reasons", [])),
            "".join(f"<li>⚠ {H.escape(c)}</li>" for c in i.get("cautions", [])))
        badges = [f'<span class="badge">{SECTOR_ZH.get(i["sector"], i["sector"])}</span>', f'<span class="badge">现价 {_num(i["price"])}</span>']
        if i.get("ret_3m") is not None and i["ret_3m"] == i["ret_3m"]:
            badges.append(f'<span class="badge">三月 {i["ret_3m"] * 100:+.0f}%</span>')
        badges += [f'<span class="badge">RSI {_num(i["rsi"], 0)}</span>', f'<span class="badge">参考分 {i["score"]:+.0f}</span>']
        if my:
            badges.append(f'<span class="badge score">我的分 {my[0]}</span>')
        if i.get("earnings_date"):
            badges.append(f'<span class="badge">财报 {i["earnings_date"][:10]}</span>')
        page = (f"<title>{s} 分析 · {when}</title>\n{FONTS}\n<link rel=\"stylesheet\" href=\"../desk.css\">\n"
                f'<div class="wrap">\n<p class="crumb"><a href="../index.html">← 回到市场日报 {when}</a></p>\n'
                f'<header class="masthead">\n  <div class="eyebrow">Trading desk · 个股分析 · 数据截至 {data_date} 收盘</div>\n'
                f'  <h1><span class="sym">{s}</span> · {H.escape(name)}</h1>\n  <div class="badges">{"".join(badges)}</div>\n</header>\n'
                f'<article>\n<figure class="chart" data-sym="{s}"></figure>\n{note_html}\n{rule}\n{pb}\n</article>\n'
                '<footer class="foot"><p>图、指标、参考位、财报、基本面、新闻由代码从公开数据算出；「我的判断」是 Claude 按操盘手册打的分和写的理由。这不是投资建议。</p></footer>\n</div>\n'
                f'<script id="chartdata" type="application/json">{json.dumps({"stocks": {s: bars[s]}, "macro": {}})}</script>\n<script src="../desk.js"></script>\n')
        (out / "pages" / f"{s}.html").write_text(page, encoding="utf-8")
        files.append(f"pages/{s}.html")
    (out / "files.json").write_text(json.dumps(files), encoding="utf-8")
    logger.info("site: %s (%d pages)", out, len(files) - 2)
    return out


HEAD = '<div class="wrap">\n<header class="masthead">\n  <div class="eyebrow">%(eyebrow)s</div>\n  <h1>%(title)s</h1>\n  <p class="lede">%(lede)s</p>\n</header>\n<nav class="toc" aria-label="目录"><ul>\n  <li><a href="#s1">一 宏观与利率</a></li><li><a href="#s2">二 政策与新闻</a></li><li><a href="#s3">三 板块</a></li>\n  <li><a href="#s4">四 值得关注</a></li><li><a href="#s5">五 回避</a></li><li><a href="#s7">七 今日判断</a></li>\n</ul></nav>\n<section class="band" id="band" aria-label="宏观看板"></section>\n'


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="tradingagents.desk site", description="the report as a web site")
    p.add_argument("--date", default=None)
    p.add_argument("--out", default=None, help="write the site here (the Artifact publish needs a folder under the repo)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = build(args.date, out_dir=Path(args.out) if args.out else None)
    print(out)
    print("publish: file_path=" + str(out / "index.html") + " root=" + str(out) + " files=" + str(out / "files.json"))
    return 0
