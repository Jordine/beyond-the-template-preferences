#!/usr/bin/env python3
"""Static rollout viewer for the user-sim check (SPEC_20260805).

Reads the generation cells + judge_labels.json, composes the 5-way register
per record via user_sim_common (same logic as the analysis), and emits a
self-contained two-file viewer in the style of the model-name-identity one:

    <out-dir>/index.html          (no dependencies, relative fetch)
    <out-dir>/rollouts_data.json

Stdout is sanitized: counts, rates and byte sizes only — no stimulus or
continuation text ever reaches the console.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from user_sim_common import compose_register, segment  # noqa: E402

REGC = {"USER_VOICE": "u", "CLOSE_THEN_ASSISTANT": "c", "ASSISTANT_VOICE": "a",
        "MIXED": "m", "GARBLED": "g", None: "x"}
PORD = {"A": 0, "B0": 1, "B1": 2, "C": 3}
MODE_SHORT = {"open_user_turn": "open", "plaintext": "plain"}


def norm(v):
    v = str(v or "NA").strip().upper()
    return "" if v in ("NA", "", "NONE") else v.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/user_sim_check")
    ap.add_argument("--labels", default="results/user_sim_check/judge_labels.json")
    ap.add_argument("--out-dir", default="results/user_sim_check/viewer")
    args = ap.parse_args()

    labels = {}
    lp = Path(args.labels)
    if lp.exists():
        for e in json.loads(lp.read_text()):
            labels[tuple(e["key"])] = e  # last entry wins

    cells = {}
    n_files = 0
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if "records" not in d or "rendered_contexts" not in d:
            continue
        n_files += 1
        short, mode, part = d["short"], d["mode"], d["part"]
        cell = cells.setdefault((short, mode), {
            "s": short, "m": MODE_SHORT.get(mode, mode), "mid": d["model_id"],
            "_ctxs": {}})
        for cid, rcx in d["rendered_contexts"].items():
            meta = rcx.get("meta") or {}
            tag = cid.split("__", 1)[-1]
            if tag.startswith(part + "_"):
                tag = tag[len(part) + 1:]
            if part == "B1" and "selected_task_kind" in meta:
                tag += " · →" + ("benign" if meta["selected_task_kind"] == "harmless"
                                 else "harmful")
            cell["_ctxs"][(part, cid)] = {"cid": cid, "p": part, "tag": tag,
                                          "tx": rcx["rendered"], "rs": []}
        for rec in d["records"]:
            seg = segment(short, rec["gen_text"])
            key = (short, mode, part, rec["ctx_id"], rec["arm"], rec["sample_idx"])
            lab = labels.get(key)
            jfrag = None
            ov = ch = vl = ""
            if lab and not str(lab.get("register", "")).startswith("error"):
                jfrag = lab["register"]
                ov, ch, vl = (norm(lab.get(k))
                              for k in ("override", "coherence", "valence"))
            reg = compose_register(seg, jfrag)
            fr = seg["pre_seg"]
            cell["_ctxs"][(part, rec["ctx_id"])]["rs"].append(
                [0 if rec["arm"] == "sampled" else 1, rec["sample_idx"],
                 fr, rec["gen_text"][len(fr):], REGC[reg], ov, ch, vl])

    out_cells = []
    for cell in cells.values():
        ctxs = sorted(cell.pop("_ctxs").values(),
                      key=lambda c: (PORD.get(c["p"], 9), c["cid"]))
        dist = {k: 0 for k in "ucamgx"}
        n = 0
        for c in ctxs:
            c["rs"].sort(key=lambda r: (r[0], r[1]))
            for r in c["rs"]:
                dist[r[4]] += 1
                n += 1
        comp = n - dist["x"]
        cell.update(n=n, dist=dist,
                    arate=(dist["a"] / comp) if comp else None, ctxs=ctxs)
        out_cells.append(cell)
    out_cells.sort(key=lambda c: (-(c["arate"] if c["arate"] is not None else -1),
                                  c["s"], c["m"]))

    data = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "cells": out_cells}
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    jp = od / "rollouts_data.json"
    jp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    (od / "index.html").write_text(HTML)

    print(f"[viewer] {n_files} cell files -> {len(out_cells)} model·mode cells")
    for c in out_cells:
        ar = "  –" if c["arate"] is None else f"{c['arate']:4.0%}"
        d = c["dist"]
        print(f"  {c['s']:26s} {c['m']:5s} n={c['n']:4d}  "
              f"u/c/a/m/g/x = {d['u']:4d}/{d['c']:4d}/{d['a']:4d}"
              f"/{d['m']:3d}/{d['g']:3d}/{d['x']:4d}  asst={ar}")
    print(f"[wrote] {jp} ({jp.stat().st_size / 1e6:.1f} MB) + index.html")


HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>assistant-bias-user-turn · rollouts</title>
<style>
:root{
  --bg:#ffffff; --fg:#1f2328; --mut:#656d76; --bd:#d0d7de; --cxbg:#f6f8fa;
  --sb:#f6f8fa; --sel:#ddf4ff; --u:#1a7f37; --c:#0969da; --a:#cf222e;
  --m:#9a6700; --g:#57606a;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d1117; --fg:#e6edf3; --mut:#8b949e; --bd:#30363d; --cxbg:#161b22;
    --sb:#010409; --sel:#0c2d6b; --u:#2ea043; --c:#316dca; --a:#c93c37;
    --m:#9e6a03; --g:#6e7681;
  }
}
*{box-sizing:border-box}
body{margin:0;display:flex;background:var(--bg);color:var(--fg);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
aside{width:300px;min-width:300px;height:100vh;position:sticky;top:0;
  overflow-y:auto;background:var(--sb);border-right:1px solid var(--bd);
  padding:14px 12px}
aside h1{font-size:15px;margin:0 0 2px}
.sub{font-size:11.5px;color:var(--mut);margin-bottom:10px}
.sub a{color:inherit}
#q{width:100%;padding:4px 8px;font-size:12.5px;margin-bottom:8px;
  border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg)}
.mo{display:flex;align-items:center;justify-content:space-between;gap:6px;
  padding:4px 8px;border-radius:6px;cursor:pointer;font-size:12.5px}
.mo:hover{background:var(--sel)}
.mo.sel{background:var(--sel);font-weight:600}
.nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.md{font-size:10.5px;color:var(--mut);border:1px solid var(--bd);
  border-radius:8px;padding:0 5px;margin-left:3px}
.rate{font-size:10.5px;font-weight:700;color:#fff;border-radius:9px;
  padding:0 6px;min-width:34px;text-align:center}
.lg{font-size:11px;color:var(--mut);margin-top:14px;border-top:1px solid var(--bd);
  padding-top:8px}
main{flex:1;min-width:0;padding:0 22px 60px}
#ctl{position:sticky;top:0;background:var(--bg);padding:10px 0 8px;z-index:5;
  border-bottom:1px solid var(--bd);display:flex;flex-wrap:wrap;gap:6px;
  align-items:center}
#ctl button{font-size:11.5px;border:1px solid var(--bd);background:var(--cxbg);
  color:var(--fg);border-radius:12px;padding:1px 9px;cursor:pointer;opacity:.45}
#ctl button.on{opacity:1;font-weight:600}
#ctl button.cg.on{color:#fff;border-color:transparent}
#ctl .sep{width:10px}
#qq{flex:1;min-width:120px;padding:3px 8px;font-size:12px;
  border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg)}
#cnt{font-size:11.5px;color:var(--mut);white-space:nowrap}
.ttl{font-size:17px;font-weight:700;margin:16px 0 1px}
.mid{font-size:11.5px;color:var(--mut);font-family:ui-monospace,monospace}
.bar{display:flex;height:7px;border-radius:4px;overflow:hidden;margin:8px 0 3px;
  max-width:560px}
.bar i{display:block}
.dw{font-size:11.5px;margin-bottom:6px}
.b-u{background:var(--u)}.b-c{background:var(--c)}.b-a{background:var(--a)}
.b-m{background:var(--m)}.b-g{background:var(--g)}
.b-x{background:var(--mut);opacity:.35}
.w-u{color:var(--u)}.w-c{color:var(--c)}.w-a{color:var(--a)}
.w-m{color:var(--m)}.w-g{color:var(--g)}.w-x{color:var(--mut)}
.cb{margin:20px 0 26px}
.ch{font-size:11px;color:var(--mut);font-family:ui-monospace,monospace;
  margin-bottom:3px}
.cx{background:var(--cxbg);color:var(--mut);font:11.5px/1.5 ui-monospace,monospace;
  padding:8px 11px;border-radius:6px;white-space:pre-wrap;margin:0 0 4px;
  overflow-wrap:anywhere}
.r{border-left:3px solid var(--bd);padding:3px 10px;margin:8px 0 8px 4px}
.r.g-u{border-color:var(--u)}.r.g-c{border-color:var(--c)}
.r.g-a{border-color:var(--a)}.r.g-m{border-color:var(--m)}
.r.g-g{border-color:var(--g)}
.rh{display:flex;gap:6px;align-items:center;margin-bottom:2px}
.pill{font-size:10.5px;font-weight:700;color:#fff;border-radius:9px;padding:0 7px}
.p-u{background:var(--u)}.p-c{background:var(--c)}.p-a{background:var(--a)}
.p-m{background:var(--m)}.p-g{background:var(--g)}
.p-x{background:transparent;color:var(--mut);border:1px dashed var(--mut)}
.t{font-size:10.5px;color:var(--mut);border:1px solid var(--bd);
  border-radius:8px;padding:0 6px}
.t.warn{color:var(--m);border-color:var(--m)}
.t.bad{color:var(--a);border-color:var(--a)}
.tx{white-space:pre-wrap;font:12.5px/1.55 ui-monospace,monospace;
  overflow-wrap:anywhere}
.tl{opacity:.38}
#empty{color:var(--mut);margin-top:40px;font-size:13px}
</style>
<aside>
  <h1>assistant-bias-user-turn · rollouts</h1>
  <div class="sub">user-sim check — at user-turn positions, is the model
  simulating the user or speaking as the assistant?
  (<a href="../../essays/assistant-bias-user-turn.pdf">essay</a>)</div>
  <input id="q" placeholder="filter models…">
  <div id="list"></div>
  <div class="lg">bright ink = fragment before first turn-marker · dim = after
  the marker · pill = register (LLM judge, or mechanical rule for immediate
  close / garble)<br><span id="gen"></span></div>
</aside>
<main>
  <div id="ctl">
    <button class="cp on" data-v="A">A</button>
    <button class="cp on" data-v="B0">B0</button>
    <button class="cp on" data-v="B1">B1</button>
    <button class="cp on" data-v="C">C</button>
    <span class="sep"></span>
    <button class="cg p-u on" data-v="u">user</button>
    <button class="cg p-c on" data-v="c">close→asst</button>
    <button class="cg p-a on" data-v="a">assistant</button>
    <button class="cg p-m on" data-v="m">mixed</button>
    <button class="cg p-g on" data-v="g">garbled</button>
    <button class="cg p-x on" data-v="x">unjudged</button>
    <span class="sep"></span>
    <button class="ca on" data-v="0">sampled</button>
    <button class="ca on" data-v="1">greedy</button>
    <input id="qq" placeholder="search text…">
    <span id="cnt"></span>
  </div>
  <div id="recs"><div id="empty">loading…</div></div>
</main>
<script>
"use strict";
const RN={u:"user",c:"close→asst",a:"assistant",m:"mixed",g:"garbled",x:"unjudged"};
const OVN={deviated_benign:"dev→benign",deviated_other:"dev→other",
           task_unidentifiable:"task?",followed:""};
let D=null,cur=-1;
const $=s=>document.querySelector(s);
const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

fetch("./rollouts_data.json").then(r=>r.json()).then(d=>{D=d;init();})
  .catch(e=>{$("#recs").innerHTML='<div id="empty">failed to load rollouts_data.json</div>';});

function init(){
  $("#gen").textContent="generated "+D.generated;
  buildList("");
  $("#q").addEventListener("input",e=>buildList(e.target.value.toLowerCase()));
  document.querySelectorAll("#ctl button").forEach(b=>
    b.addEventListener("click",()=>{b.classList.toggle("on");apply();}));
  let t=null;
  $("#qq").addEventListener("input",()=>{clearTimeout(t);t=setTimeout(apply,160);});
  let i0=0;
  if(location.hash){
    const h=decodeURIComponent(location.hash.slice(1));
    const j=D.cells.findIndex(c=>c.s+"|"+c.m===h);
    if(j>=0)i0=j;
  }
  select(i0);
}

function buildList(q){
  const el=$("#list");
  let h="";
  D.cells.forEach((c,i)=>{
    if(q&&!(c.s+" "+c.m).toLowerCase().includes(q))return;
    const a=c.arate;
    const pill=a==null?'<span class="rate" style="background:var(--g)">–</span>'
      :'<span class="rate" style="background:hsl('+Math.round((1-a)*120)+
       ',55%,38%)">'+Math.round(100*a)+'%</span>';
    h+='<div class="mo'+(i===cur?" sel":"")+'" data-i="'+i+'"><span class="nm">'+
       esc(c.s)+' <span class="md">'+c.m+'</span></span>'+pill+'</div>';
  });
  el.innerHTML=h;
  el.querySelectorAll(".mo").forEach(m=>
    m.addEventListener("click",()=>select(+m.dataset.i)));
}

function bar(c){
  let segs="",words=[];
  for(const k of "ucamgx"){
    const v=c.dist[k];
    if(!v)continue;
    segs+='<i class="b-'+k+'" style="flex:'+v+'"></i>';
    words.push('<span class="w-'+k+'">'+RN[k]+" "+
               Math.round(100*v/(c.n||1))+"%</span>");
  }
  return '<div class="bar">'+segs+'</div><div class="dw">'+
         words.join(" · ")+"</div>";
}

function select(i){
  cur=i;
  const c=D.cells[i];
  location.hash=encodeURIComponent(c.s+"|"+c.m);
  buildList($("#q").value.toLowerCase());
  let h='<div class="ttl">'+esc(c.s)+' <span class="md">'+c.m+'</span></div>'+
        '<div class="mid">'+esc(c.mid)+" · "+c.n+" continuations</div>"+bar(c);
  for(const cx of c.ctxs){
    h+='<div class="cb" data-p="'+cx.p+'"><div class="ch">'+esc(cx.p)+" · "+
       esc(cx.tag)+'</div><pre class="cx">'+esc(cx.tx)+"</pre>";
    for(const r of cx.rs){
      const[a,si,fr,tl,g,ov,ch,vl]=r;
      let tags="";
      if(a)tags+='<span class="t">greedy</span>';
      const o=OVN[ov]!==undefined?OVN[ov]:ov;
      if(o)tags+='<span class="t">'+esc(o)+"</span>";
      if(ch==="degraded")tags+='<span class="t warn">degraded</span>';
      if(vl==="questionable")tags+='<span class="t warn">questionable</span>';
      if(vl==="unsafe_reckless")tags+='<span class="t bad">unsafe</span>';
      h+='<div class="r g-'+g+'" data-p="'+cx.p+'" data-a="'+a+'" data-g="'+g+
         '"><div class="rh"><span class="pill p-'+g+'">'+RN[g]+"</span>"+tags+
         '</div><div class="tx"><span class="fr">'+esc(fr)+
         '</span><span class="tl">'+esc(tl)+"</span></div></div>";
    }
    h+="</div>";
  }
  $("#recs").innerHTML=h;
  window.scrollTo(0,0);
  apply();
}

function F(cls){
  const s=new Set();
  document.querySelectorAll("#ctl ."+cls+".on").forEach(b=>s.add(b.dataset.v));
  return s;
}

function apply(){
  const fp=F("cp"),fg=F("cg"),fa=F("ca");
  const q=$("#qq").value.toLowerCase();
  let shown=0,tot=0;
  document.querySelectorAll("#recs .cb").forEach(cb=>{
    const pOn=fp.has(cb.dataset.p);
    const ctext=q?cb.querySelector(".cx").textContent.toLowerCase():"";
    let any=false;
    cb.querySelectorAll(".r").forEach(r=>{
      tot++;
      const ok=pOn&&fg.has(r.dataset.g)&&fa.has(r.dataset.a)&&
        (!q||ctext.includes(q)||r.textContent.toLowerCase().includes(q));
      r.style.display=ok?"":"none";
      if(ok){any=true;shown++;}
    });
    cb.style.display=any?"":"none";
  });
  $("#cnt").textContent=shown+"/"+tot;
}
</script>
"""


if __name__ == "__main__":
    main()
