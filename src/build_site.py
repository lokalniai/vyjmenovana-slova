"""Build the viewer: docs/index.html (+ docs/static/ brand assets). Standard library only.

    python -I src/build_site.py [--raw results/raw] [--out docs]

Every number is recomputed from the raw runs with src/score.py, so the page and results/summary.md agree.
Settings with fewer than runs_for(setting) finished runs are drawn dimmed and labelled "partial".
"""

import argparse
import html
import json
import math
import shutil
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import LETTERS, Gold, load_runs, norm, parse_answer, score_run, split_items  # noqa: E402
from settings import ANSWER_BUDGET, MODELS, PROMPT, REASONING_CAP, RUNS, SETTINGS, request_max_tokens, runs_for  # noqa: E402

ROOT = HERE.parent
ASSETS = ROOT.parent / "determinism" / "src" / "assets"
TITLE = "Vyjmenovaná slova"
SOURCE = "https://prirucka.ujc.cas.cz/?id=100"
MODE_NAME = {"off": "reasoning off", "on": "reasoning on"}
METRICS = [("f1", "F1"), ("precision", "precision"), ("recall", "recall"), ("precision_lenient", "precision (lenient)")]
FP_LABEL = {"derived": "derived word (in brackets in ÚJČ)", "near": "one edit from a gold word",
            "misplaced": "gold word under the wrong letter", "other": "not a listed word"}


def is_closed(m: str) -> bool:
    """Closed-weight models are drawn dimmed: they are shown for comparison next to the open-weight ones."""
    return MODELS[m.split("/")[0]][3] == "closed"


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def fmt_tok(n) -> str:
    if n is None:
        return "–"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


# ── data ────────────────────────────────────────────────────────────────────

def collect(raw: Path, gold: Gold) -> dict:
    runs = []
    for r in load_runs(raw):
        if r["setting"] not in SETTINGS:
            continue
        ans = parse_answer(r.get("content"))
        s = score_run(ans, gold)
        u = r.get("usage") or {}
        rt = r["reasoning_tokens_eff"]
        runs.append({"setting": r["setting"], "model": r["setting"].split("/")[0], "mode": r["setting"].split("/")[1],
                     "seed": r["seed"], "answer": ans, "score": s, "content": r.get("content") or "",
                     "reasoning": r.get("reasoning") or "", "completion_tokens": u.get("completion_tokens"),
                     "reasoning_tokens": rt, "finish": r.get("finish_reason"), "latency": r.get("latency_s"),
                     "forced": r.get("forced"), "phase1_reasoning": ((u.get("completion_tokens_details") or {})
                                                                    .get("reasoning_tokens"))})
    by = {}
    for r in runs:
        by.setdefault(r["setting"], []).append(r)
    agg = {}
    for s, R in by.items():
        a = {"n": len(R), "partial": len(R) < runs_for(s)}
        for k, _ in METRICS + [("recall_common", ""), ("recall_names", "")] + [(f"recall_{L}", "") for L in LETTERS]:
            xs = [r["score"][k] for r in R]
            a[k] = st.mean(xs)
            a[k + "_sd"] = st.stdev(xs) if len(xs) > 1 else 0.0
        rts = [r["reasoning_tokens"] for r in R if r["reasoning_tokens"] is not None]
        a["reasoning_median"] = st.median(rts) if rts else None
        a["truncated"] = sum(r["finish"] == "length" for r in R)          # answer cut at the answer budget
        a["forced"] = sum(r["forced"] in ("cap", "answer") for r in R)
        a["pending"] = sum(r["forced"] == "pending" for r in R)
        a["partial"] = a["partial"] or a["pending"] > 0
        a["parse_fail"] = sum(not r["score"]["parse_ok"] for r in R)
        a["listed"] = st.mean(r["score"]["n_listed"] for r in R)
        agg[s] = a
    return {"runs": runs, "by": by, "agg": agg}


def ordered_settings(agg: dict) -> list[str]:
    """Leaderboard order: models by best F1, reasoning on before off."""
    return [f"{m}/{md}" for m in model_order(agg) for md in ("on", "off") if f"{m}/{md}" in agg]


def model_order(agg: dict) -> list[str]:
    best = {}
    for s, a in agg.items():
        m = s.split("/")[0]
        best[m] = max(best.get(m, -1), a["f1"])
    return sorted(best, key=lambda m: -best[m])


# ── pieces ──────────────────────────────────────────────────────────────────

LEGEND = ("<dl class=legend><dt><span class='key off'></span>reasoning off</dt> <dd>hollow</dd> "
          "<dt><span class='key on'></span>reasoning on</dt> <dd>filled; high effort where the model has levels</dd> "
          "<dt><span class='key on dim'></span>dimmed</dt> <dd>closed weights</dd> "
          "<dt>whisker</dt> <dd>± 1 sd over the 5 runs (none for single-run models)</dd> "
          "<dt>reas. tok</dt> <dd>median, reasoning on</dd></dl>")


def metric_panel(agg: dict, key: str, label: str, hidden: bool = False) -> str:
    """One metric: a row per model with its reasoning-off and -on markers, whiskers and values."""
    rows = []
    for m in model_order(agg):
        name, repo, where, quant = MODELS[m]
        cells, vals = [], {}
        for mode in ("off", "on"):
            a = agg.get(f"{m}/{mode}")
            if not a:
                vals[mode] = None
                continue
            v, sd = a[key], a[key + "_sd"]
            vals[mode] = v
            lo, hi = max(0, v - sd), min(1, v + sd)
            tip = (f"<b>{esc(name)}, {MODE_NAME[mode]}</b>\n{label} {v:.3f} ± {sd:.3f} over {a['n']} runs"
                   + (f"\nmedian reasoning {fmt_tok(a['reasoning_median'])} tokens" if mode == "on" else "")
                   + (f"\n{a['forced']} run(s) answered only after the forced end of thinking" if a["forced"] else "")
                   + (f"\n{a['truncated']} answer(s) cut at the 16k answer budget" if a["truncated"] else "")
                   + (f"\n<span class=m>partial: fewer than {runs_for(f'{m}/{mode}')} finished runs</span>" if a["partial"] else ""))
            cells.append(f"<span class='wh {mode}' style='left:{lo*100:.1f}%;width:{(hi-lo)*100:.1f}%'></span>"
                         f"<span class='mk {mode}{' part' if a['partial'] else ''}' style='left:{v*100:.1f}%' "
                         f"tabindex=0 data-tip=\"{esc(tip)}\"></span>")
        if vals.get("off") is not None and vals.get("on") is not None:
            lo, hi = sorted((vals["off"], vals["on"]))
            cells.insert(0, f"<span class=gap style='left:{lo*100:.1f}%;width:{(hi-lo)*100:.1f}%'></span>")
        a_on = agg.get(f"{m}/on")
        d = (vals["on"] - vals["off"]) if vals.get("off") is not None and vals.get("on") is not None else None
        delta = (("0.00" if abs(d) < .005 else f"{d:+.2f}") if d is not None
                 else "<span class=m>–</span>")
        rows.append(
            f"<div class='lrow{' closed' if is_closed(m) else ''}'><div class=lname>{esc(name)}<span>{esc(where)} · {esc(quant)}"
            f"{' · ' + str(n_runs) + ' run' + ('s' if n_runs > 1 else '') if (n_runs := max(runs_for(s) for s in SETTINGS if s.startswith(m + '/'))) < RUNS else ''}</span></div>"
            f"<div class=ltrack>{''.join(cells)}</div>"
            f"<div class=lval>{'–' if vals.get('off') is None else f'{vals['off']:.2f}'}</div>"
            f"<div class=lval>{'–' if vals.get('on') is None else f'{vals['on']:.2f}'}</div>"
            f"<div class='lval x'>{delta}</div>"
            f"<div class='lval x'>{fmt_tok(a_on['reasoning_median']) if a_on else '–'}</div></div>")
    axis = "".join(f"<span style='left:{t*100:.0f}%'>{t:.1f}</span>" for t in (0, .2, .4, .6, .8, 1))
    return (f"<div class=lb data-metric={key}{' hidden' if hidden else ''}>"
            f"<div class='lrow lhead'><div></div><div class=laxis>{axis}</div><div class=lval>off</div>"
            f"<div class=lval>on</div><div class='lval x'>Δ</div><div class='lval x'>reas. tok</div></div>"
            + "".join(rows) + "</div>")


def metric_buttons() -> str:
    btns = "".join(f"<button type=button class='seg{' on' if i == 0 else ''}' data-metric-btn={k}>{esc(l)}</button>"
                   for i, (k, l) in enumerate(METRICS))
    return f"<div class=ctl role=group aria-label=metric>{btns}</div>"


def leaderboard(D: dict) -> str:
    return LEGEND + "".join(metric_panel(D["agg"], k, l, hidden=i > 0) for i, (k, l) in enumerate(METRICS))


def scatter(D: dict, W: int = 760, H: int = 380, compact: bool = False) -> str:
    """Reasoning tokens (log x) vs F1, one dot per reasoning-on run; a ring at each model's mean. compact: the chart
    alone, sized for a dashboard card."""
    L, R, T, B = 62, 16, 30 if compact else 14, 40   # compact: room for labels lifted above the top cluster
    # OpenRouter runs have no forced cap and can reason past it (up to the 48k total)
    top = max([r["reasoning_tokens"] or 0 for s, R_ in D["by"].items() if s.endswith("/on") for r in R_] + [40_000 / 1.15])
    xmin, xmax = math.log10(300), math.log10(top * 1.15)
    X = lambda t: L + (math.log10(max(t, 300)) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda f: T + (1 - f) * (H - T - B)
    g = []
    for f in (0, .25, .5, .75, 1):
        g.append(f"<line class=grid x1={L} x2={W-R} y1={Y(f):.1f} y2={Y(f):.1f} />"
                 f"<text class=tick x={L-6} y={Y(f)+4:.1f} text-anchor=end>{f:.2f}</text>")
    for t in (1e3, 3e3, 1e4, 3.2e4):
        g.append(f"<line class=grid x1={X(t):.1f} x2={X(t):.1f} y1={T} y2={H-B} />"
                 f"<text class=tick x={X(t):.1f} y={H-B+16} text-anchor=middle>{fmt_tok(t)}</text>")
    g.append(f"<line class=cap x1={X(REASONING_CAP):.1f} x2={X(REASONING_CAP):.1f} y1={T} y2={H-B} />"
             f"<text class=tick x={X(REASONING_CAP)-4:.1f} y={T+10} text-anchor=end>cap</text>")
    g.append(f"<text class=axl x={(L+W-R)/2:.0f} y={H-4} text-anchor=middle>reasoning tokens (log scale)</text>")
    g.append(f"<text class=axl x=12 y={(T+H-B)/2:.0f} text-anchor=middle transform='rotate(-90 12 {(T+H-B)/2:.0f})'>F1</text>")
    pts, labels = [], []
    for s, R_ in D["by"].items():
        if not s.endswith("/on"):
            continue
        name = MODELS[s.split("/")[0]][0]
        dim = " closed" if is_closed(s) else ""
        xs = []
        for r in R_:
            t = r["reasoning_tokens"]
            if t is None:
                continue
            f1 = r["score"]["f1"]
            xs.append((t, f1))
            tip = esc(f"<b>{esc(name)}</b> seed {r['seed']}\n{t:,} reasoning tokens · F1 {f1:.3f}"
                      + ("\n<span class=m>reasoning cut at the 32k cap, answer forced</span>" if r["forced"] == "cap" else "")
                      + ("\n<span class=m>answer continued after the first call ran out</span>" if r["forced"] == "answer" else "")
                      + ("\n<span class=m>forced answer still pending</span>" if r["forced"] == "pending" else "")
                      + ("\n<span class=m>no parseable JSON answer</span>" if not r["score"]["parse_ok"] else ""))
            if r["forced"] == "pending":
                continue
            if not r["score"]["parse_ok"]:
                x, y = X(t), Y(f1)
                pts.append(f"<g class='trunc{dim}' tabindex=0 data-tip=\"{tip}\"><path d='M{x-4:.1f} {y-4:.1f}l8 8m0 -8l-8 8' />"
                           f"<circle class=hit cx={x:.1f} cy={y:.1f} r=9 /></g>")
            elif r["forced"] == "cap":
                pts.append(f"<rect class='pt{dim}' x={X(t)-4.5:.1f} y={Y(f1)-4.5:.1f} width=9 height=9 rx=1.5 tabindex=0 "
                           f"data-tip=\"{tip}\" />")
            else:
                pts.append(f"<circle class='pt{dim}' cx={X(t):.1f} cy={Y(f1):.1f} r=4.5 tabindex=0 data-tip=\"{tip}\" />")
        if xs:
            mx = 10 ** st.mean(math.log10(max(t, 300)) for t, _ in xs)
            my = st.mean(f for _, f in xs)
            labels.append((X(mx), Y(my), name, my, dim))
    lab, placed = [], []   # placed label boxes (x0, x1, y); a label that would overlap one moves up a line
    near0 = sorted(name for *_, name, my, _ in labels if my < .05)
    for x, y, name, my, dim in sorted(labels, key=lambda z: -z[1]):
        if my < .05:       # the pile at F1 ~ 0 is unreadable when labelled; those are named under the chart
            lab.append(f"<circle class='mean{dim}' cx={x:.1f} cy={y:.1f} r=7 />")
            continue
        anchor, dx = ("end", -9) if x > W - (130 if compact else 170) else ("start", 9)
        w = len(name) * 6.6
        x0, x1 = (x + dx - w, x + dx) if anchor == "end" else (x + dx, x + dx + w)
        ty = y + 4
        while any(x0 < b1 and b0 < x1 and abs(ty - by) < 14 for b0, b1, by in placed):
            ty -= 14
        placed.append((x0, x1, ty))
        lead = (f"<line class=lead x1={x:.1f} y1={y:.1f} x2={x + dx * .8:.1f} y2={ty - 4:.1f} />"
                if abs(ty - (y + 4)) > 1 else "")
        lab.append(f"<g class='lbl{dim}'><circle class=mean cx={x:.1f} cy={y:.1f} r=7 />{lead}"
                   f"<text class=dl x={x+dx:.1f} y={ty:.1f} text-anchor={anchor}>{esc(name)}</text></g>")
    svg = (f"<svg class='sc{' sc-c' if compact else ''}' viewBox='0 0 {W} {H}' role=img aria-label='reasoning tokens against F1'>"
           + "".join(g) + "".join(pts) + "".join(lab) + "</svg>")
    if compact:
        return svg
    return (svg + "<dl class=legend><dt><span class='key on'></span>run</dt> <dd>one reasoning-on run</dd> "
            "<dt><span class='key ring'></span>model mean</dt> <dd>geometric mean of tokens, mean F1</dd> "
            "<dt><span class='key on dim'></span>dimmed</dt> <dd>closed weights</dd> "
            "<dt><span class='key sqon'></span>forced</dt> <dd>reasoning cut at the 32k cap, then made to answer</dd> "
            "<dt><span class='key x'>×</span>no answer</dt> <dd>no parseable JSON (F1 = 0)</dd></dl>"
            + (f"<p class=sub>Unlabelled rings near F1 = 0 (hover a dot for its model): {esc(', '.join(near0))}.</p>"
               if near0 else ""))


def seq_cell(v: float, text: str, tip: str) -> str:
    p = round(v * 100)
    return (f"<td class='hc{' hi' if v >= .55 else ''}' style='--p:{p}%' tabindex=0 data-tip=\"{esc(tip)}\">"
            f"{text}</td>")


def heatmap(D: dict, gold: Gold) -> str:
    agg = D["agg"]
    nL = {L: sum(1 for i in gold.items.values() if i["letter"] == L) for L in LETTERS}
    head = "".join(f"<th>{L}<span>{nL[L]}</span></th>" for L in LETTERS)
    rows = []
    for s in ordered_settings(agg):
        m, mode = s.split("/")
        a = agg[s]
        name = MODELS[m][0]
        cells = "".join(seq_cell(a[f"recall_{L}"], f"{a[f'recall_{L}']:.2f}",
                                 f"<b>{esc(name)}, {MODE_NAME[mode]}</b>\nletter {L}: recall {a[f'recall_{L}']:.3f} "
                                 f"(± {a[f'recall_{L}_sd']:.3f})") for L in LETTERS)
        cells += seq_cell(a["recall_names"], f"{a['recall_names']:.2f}",
                          f"<b>{esc(name)}, {MODE_NAME[mode]}</b>\nproper names: recall {a['recall_names']:.3f}")
        cls = " ".join(c for c, on in (("part", a["partial"]), ("closed", is_closed(m))) if on)
        rows.append(f"<tr{f' class={chr(39)}{cls}{chr(39)}' if cls else ''}><th class=rl>{esc(name)} <span class='md {mode}'>"
                    f"{mode}</span></th>{cells}</tr>")
    return ("<div class=wrap><table class=hm><thead><tr><th></th>" + head +
            f"<th>names<span>{sum(i['kind']=='name' for i in gold.items.values())}</span></th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def word_grid(D: dict, gold: Gold) -> str:
    sets = ordered_settings(D["agg"])
    counts = {}
    for r in D["runs"]:
        for iid in r["score"]["hits"]:
            counts[(iid, r["setting"])] = counts.get((iid, r["setting"]), 0) + 1
    head = "".join(f"<th class='vh{' closed' if is_closed(s) else ''}'><span data-tip=\"{esc(MODELS[s.split('/')[0]][0] + ', ' + MODE_NAME[s.split('/')[1]])}\">"
                   f"{esc(MODELS[s.split('/')[0]][0])} · {s.split('/')[1]}</span></th>"
                   for s in sets)
    body = []
    for L in LETTERS:
        body.append(f"<tr class=sect><th colspan={len(sets) + 2}>{L.upper()}</th></tr>")
        for iid, it in gold.items.items():
            if it["letter"] != L:
                continue
            cells, known = [], 0
            for s in sets:
                n = D["agg"][s]["n"]
                k = counts.get((iid, s), 0)
                known += k > 0
                tip = (f"<b>{esc(it['lemma'])}</b> · {esc(MODELS[s.split('/')[0]][0])}, {MODE_NAME[s.split('/')[1]]}\n"
                       f"listed in {k} of {n} runs")
                cells.append(f"<td class='wc{' closed' if is_closed(s) else ''}' style='--p:{round(k / n * 100) if n else 0}%' tabindex=0 data-tip=\"{esc(tip)}\"></td>")
            body.append(f"<tr><th class=rl>{esc(it['lemma'])}{' <span class=nm>name</span>' if it['kind'] == 'name' else ''}"
                        f"</th>{''.join(cells)}<td class=wk>{known}/{len(sets)}</td></tr>")
    return ("<dl class=legend><dt><span class='key sq0'></span>0 of 5 runs</dt> <dd>never listed</dd> "
            "<dt><span class='key sq1'></span>5 of 5</dt> <dd>listed every time; shades in between (single-run models: 0 or 1)</dd> "
            "<dt>last column</dt> <dd>settings that listed the word at least once</dd></dl>"
            "<div class='wrap tall'><table class=wg><thead><tr><th></th>" + head + "<th></th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def false_positives(D: dict) -> str:
    agg = {}
    for r in D["runs"]:
        for fp in r["score"]["fps"]:
            k = (fp["key"], fp["word"], fp["label"])
            e = agg.setdefault(k, {"runs": 0, "settings": set()})
            e["runs"] += 1
            e["settings"].add(r["setting"])
    rows = sorted(agg.items(), key=lambda kv: (-len(kv[1]["settings"]), -kv[1]["runs"], kv[0][1]))[:80]
    tr = "".join(
        f"<tr><td class=w>{esc(w)}</td><td class=mono>{esc(key)}</td><td><span class='lab {lab}'>{esc(lab)}</span></td>"
        f"<td class=num>{len(e['settings'])}</td><td class=num>{e['runs']}</td>"
        f"<td class=ss>{esc(', '.join(sorted(MODELS[s.split('/')[0]][0] + ' ' + s.split('/')[1] for s in e['settings'])))}</td></tr>"
        for (key, w, lab), e in rows)
    leg = " ".join(f"<dt><span class='lab {k}'>{k}</span></dt> <dd>{esc(v)}</dd>" for k, v in FP_LABEL.items())
    return (f"<dl class=legend>{leg}</dl><div class='wrap mid'><table class=fp><thead><tr><th>word</th><th>key</th>"
            "<th>label</th><th class=num>settings</th><th class=num>runs</th><th class=ss>listed by</th></tr></thead><tbody>"
            + tr + "</tbody></table></div>")


def runs_blob(D: dict, gold: Gold) -> str:
    out = []
    for r in D["runs"]:
        s = r["score"]
        hits = set(s["hits"])
        fps = {(f["key"], f["word"]): f["label"] for f in s["fps"]}
        letters = []
        if r["answer"]:
            for k, v in r["answer"].items():
                key = norm(k)[:1] or "?"
                words = []
                for w in split_items(v):
                    iid = gold.lookup.get((key, w))
                    words.append([w, "tp" if iid in hits else fps.get((key, w), "skip")])
                letters.append([str(k), words])
        missed = {L: [it["lemma"] for iid, it in gold.items.items() if it["letter"] == L and iid not in hits]
                  for L in LETTERS}
        rs = r["reasoning"]
        out.append({"s": r["setting"], "seed": r["seed"], "f1": round(s["f1"], 3), "p": round(s["precision"], 3),
                    "r": round(s["recall"], 3), "ok": s["parse_ok"], "fin": r["finish"], "ct": r["completion_tokens"],
                    "rt": r["reasoning_tokens"], "rt1": r["phase1_reasoning"], "fo": r["forced"], "lat": r["latency"],
                    "L": letters, "miss": missed,
                    "raw": r["content"][:6000] + (" …[cut]" if len(r["content"]) > 6000 else ""),
                    "rs": (rs if len(rs) <= 6000 else rs[:3000] + f"\n\n …[{len(rs) - 6000:,} characters left out] …\n\n" + rs[-3000:]),
                    "rsn": len(rs)})
    names = {s: [MODELS[s.split("/")[0]][0], s.split("/")[1], is_closed(s)] for s in SETTINGS}
    return ("<script>const RUNS=" + json.dumps(out, ensure_ascii=False).replace("</", "<\\/")
            + ";const NAMES=" + json.dumps(names, ensure_ascii=False)
            + ";const ORDER=" + json.dumps(ordered_settings(D["agg"])) + ";</script>")


def setup_table(D: dict) -> str:
    rows = []
    for m, (name, repo, where, quant) in MODELS.items():
        modes = [s.split("/")[1] for s in SETTINGS if s.startswith(m + "/")]
        on = SETTINGS.get(f"{m}/on", {})
        caps = ", ".join(((f"on effort high, {fmt_tok(request_max_tokens(f'{m}/on'))} in total" if on["budget"] == "effort"
                           else f"on {REASONING_CAP // 1000}k + {ANSWER_BUDGET // 1000}k ({on['budget']})")
                          if md == "on" else f"off {fmt_tok(request_max_tokens(f'{m}/off'))}") for md in modes)
        link = f"https://openrouter.ai/{repo}" if where == "OpenRouter" else f"https://huggingface.co/{repo}"
        n = ", ".join(f"{md} {D['agg'][f'{m}/{md}']['n'] if f'{m}/{md}' in D['agg'] else 0}" for md in modes)
        rows.append(f"<tr><td>{esc(name)}</td><td class=mono><a href='{esc(link)}'>{esc(repo)}</a></td>"
                    f"<td>{esc(where)}</td><td>{esc(quant)}</td><td class=mono>{esc(', '.join(modes))}</td>"
                    f"<td class=mono>{esc(caps)}</td><td class=mono>{esc(n)}</td></tr>")
    return ("<div class=wrap><table class=fp><thead><tr><th>model</th><th>checkpoint / API slug</th><th>ran on</th><th>weights</th>"
            "<th>modes</th><th>reasoning + answer tokens</th><th>runs</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


# ── dashboard ───────────────────────────────────────────────────────────────

EXPAND = ("<svg class=i-open viewBox='0 0 24 24' aria-hidden=true><path d='M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7' /></svg>"
          "<svg class=i-close viewBox='0 0 24 24' aria-hidden=true><path d='M18 6 6 18M6 6l12 12' /></svg>")


def card(key: str, title: str, hint: str, body: str, tools: str = "", cls: str = "") -> str:
    """A dashboard box: compact in the grid, full-screen when expanded (.max). Inside, `.more` shows only when expanded
    and `.less` only when compact."""
    return (f"<section class='card c-{key}{' ' + cls if cls else ''}' id=c-{key} aria-labelledby=h-{key}>"
            f"<header class=ch><h2 id=h-{key}>{esc(title)}</h2>{tools}<button type=button class=xp data-expand "
            f"aria-expanded=false aria-label='Expand: {esc(title)}' title='Expand (Esc closes)'>{EXPAND}</button></header>"
            + (f"<p class=hint>{hint}</p>" if hint else "") + f"<div class=cb>{body}</div></section>")


def kpis(D: dict, gold: Gold) -> str:
    agg = D["agg"]
    f1 = lambda s: agg[s]["f1"]
    who = lambda s: f"{esc(MODELS[s.split('/')[0]][0])} · {s.split('/')[1]}"
    runs = lambda s: f"{agg[s]['n']} run{'s' if agg[s]['n'] > 1 else ''}"
    best = max(agg, key=f1)
    best_ow = max((s for s in agg if MODELS[s.split("/")[0]][3] != "closed"), key=f1)
    full = [s for s in agg if agg[s]["n"] >= RUNS]
    best_full = max(full, key=f1) if full else None
    n_names = sum(i["kind"] == "name" for i in gold.items.values())
    tiles = [("best open-weight", f"{f1(best_ow):.2f}", f"{who(best_ow)} · {runs(best_ow)}", "")]
    if best_full:
        tiles.append((f"best over {RUNS} runs", f"{f1(best_full):.2f}", f"{who(best_full)} · ± {agg[best_full]['f1_sd']:.2f}", ""))
    if best != best_ow:
        tiles.append(("best closed" if is_closed(best) else "best F1", f"{f1(best):.2f}", f"{who(best)} · {runs(best)}",
                      " closed" if is_closed(best) else ""))
    tiles += [("settings with F1 ≥ 0.5", f"{sum(f1(s) >= .5 for s in agg)}<small>/{len(agg)}</small>",
               f"{len(model_order(agg))} models · {len(D['runs'])} runs", ""),
              ("gold items", str(len(gold.items)), f"{len(gold.items) - n_names} words + {n_names} names, ÚJČ", "")]
    return "".join(f"<div class='kpi{c}'><span class=kl>{l}</span><b class=kv>{v}</b><span class=ks>{sub}</span></div>"
                   for l, v, sub, c in tiles)


# ── page ────────────────────────────────────────────────────────────────────

CSS = """
:root{
--bg:#f4f5f0;--fg:#1f211c;--muted:#585a54;--line:#d5d7d2;--card:#eceee9;--link:#5c6b0c;
--off:#0a7fa8;--on:#b86200;--seq:#5c6b0c;--bad:#c2003f;--warn:#8a5a00;--good:#5c6b0c;
--track:rgba(31,33,28,.07);
--sans:'Barlow',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
--cond:'Barlow Condensed','Barlow Semi Condensed',var(--sans);
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
:root[data-theme='dark']{
--bg:#1f211c;--fg:#f8f8f2;--muted:#c2b889;--line:#3e3d32;--card:#252820;--link:#dcff00;
--off:#1898b5;--on:#c98316;--seq:#c4e000;--bad:#ff3d7a;--warn:#ffb800;--good:#dcff00;
--track:rgba(248,248,242,.08);
}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;padding:1.4rem clamp(1rem,4vw,3rem) 3rem;background:var(--bg);color:var(--fg);
font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
code,.mono{font-family:var(--mono);font-size:.84em}
.m{color:var(--muted)}
.mast{display:flex;align-items:center;gap:.7rem;padding-bottom:.9rem;margin-bottom:1.2rem;border-bottom:1px solid var(--line)}
.brand{display:block;line-height:0;flex:none}.brand img{height:2rem;width:auto;display:block}
:root .logo-dark{display:none}:root[data-theme='dark'] .logo-light{display:none}:root[data-theme='dark'] .logo-dark{display:block}
.mastsep{color:var(--line)}.spacer{flex:1}
.here{font-family:var(--cond);font-weight:600;font-size:1.15rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tgl{display:flex;align-items:center;justify-content:center;flex:none;width:1.9rem;height:1.9rem;border:1px solid var(--line);
background:var(--card);color:var(--muted);border-radius:4px;cursor:pointer;padding:0}
.tgl:hover{color:var(--fg);border-color:var(--muted)}
.tgl svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
:root .tgl .moon{display:none}:root[data-theme='dark'] .tgl .sun{display:none}:root[data-theme='dark'] .tgl .moon{display:block}
h1{font-family:var(--cond);font-weight:700;font-size:clamp(1.9rem,4vw,2.6rem);letter-spacing:-.02em;line-height:1.1;
margin:.2rem 0 .5rem;text-wrap:balance}
h2{font-family:var(--cond);font-weight:700;font-size:1.45rem;letter-spacing:-.01em;margin:2.2rem 0 .3rem;text-wrap:balance}
.lede{max-width:78ch;text-wrap:pretty;margin:0 0 .8rem}
.sub{max-width:78ch;color:var(--muted);font-size:.92rem;margin:0 0 .8rem;text-wrap:pretty}
.cav{background:var(--card);border-left:3px solid var(--muted);border-radius:0 5px 5px 0;color:var(--muted);
font-size:.86rem;padding:.55rem .9rem;margin:.6rem 0 1rem;max-width:100ch;text-wrap:pretty}
.cav b{color:var(--fg);font-weight:600}
.legend{display:block;color:var(--muted);font-size:.84rem;margin:.2rem 0 .6rem}
.legend dt,.legend dd{display:inline;margin:0}.legend dt{color:var(--fg);font-weight:500}
.legend dt:not(:first-child)::before{content:'•';color:var(--line);margin:0 .55rem 0 .35rem}
.key{display:inline-block;width:.72rem;height:.72rem;border-radius:50%;margin-right:.35rem;vertical-align:-.06rem}
.key.off{border:2px solid var(--off);background:var(--bg)}.key.on{background:var(--on)}
.key.ring{border:2px solid var(--fg);background:transparent}
.key.x{width:auto;height:auto;border-radius:0;color:var(--bad);font-weight:700;vertical-align:0}
.key.sqon{border-radius:2px;background:var(--on)}.key.sq0{border-radius:2px;background:var(--track)}.key.sq1{border-radius:2px;background:var(--seq)}
.panel{border:1px solid var(--line);border-radius:5px;background:var(--card);padding:.9rem 1.1rem;margin:.6rem 0}
.ctl{display:flex;flex-wrap:wrap;gap:.3rem;margin:.2rem 0 .6rem}
.seg{font:500 .82rem var(--sans);color:var(--muted);background:var(--bg);border:1px solid var(--line);border-radius:4px;
padding:.22rem .65rem;cursor:pointer}
.seg:hover{color:var(--fg);border-color:var(--muted)}.seg.on{color:var(--bg);background:var(--fg);border-color:var(--fg)}
/* leaderboard */
.lrow{display:grid;grid-template-columns:minmax(8rem,13rem) 1fr repeat(4,4.2rem);gap:.8rem;align-items:center;
padding:.35rem 0;border-bottom:1px solid var(--line)}
.lrow.lhead{border-bottom:0;padding:0 0 .1rem}
.lname{font-weight:500;line-height:1.2}.lname span{display:block;font-size:.74rem;color:var(--muted);font-weight:400}
.ltrack{position:relative;height:22px;background:linear-gradient(var(--track),var(--track)) center/100% 8px no-repeat;border-radius:4px}
.laxis{position:relative;height:1.1rem}
.laxis span{position:absolute;transform:translateX(-50%);font:400 .7rem var(--mono);color:var(--muted)}
.lval{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.8rem;text-align:right}
.lhead .lval{font-family:var(--cond);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.mk{position:absolute;top:50%;width:13px;height:13px;margin:-6.5px 0 0 -6.5px;border-radius:50%;cursor:default;
box-shadow:0 0 0 2px var(--card);z-index:2}
.mk::after{content:'';position:absolute;inset:-7px;border-radius:50%}
.mk.on{background:var(--on)}.mk.off{background:var(--card);border:2.5px solid var(--off)}

.mk.part{opacity:.45}
.lrow.closed,tr.closed,.wg .closed,.sc .closed,.kpi.closed,.chips .closed{opacity:.42}
.key.dim{opacity:.42}
.mk:focus{outline:2px solid var(--fg);outline-offset:2px}
.wh{position:absolute;top:50%;height:2px;margin-top:-1px;z-index:1}
.wh.on{background:var(--on)}.wh.off{background:var(--off)}
.gap{position:absolute;top:50%;height:8px;margin-top:-4px;
background:repeating-linear-gradient(135deg,var(--muted) 0 1.5px,transparent 1.5px 5px);opacity:.55}
@media(max-width:700px){.lrow{grid-template-columns:7rem 1fr 3rem 3rem}.lrow>.lval:nth-child(n+5){display:none}.laxis span:nth-child(even){display:none}}
/* scatter */
.sc{width:100%;height:auto;max-width:900px;display:block}
.sc .grid{stroke:var(--line);stroke-width:1}
.sc .tick{font:400 11px var(--mono);fill:var(--muted)}.sc .axl{font:500 12px var(--sans);fill:var(--muted)}
.sc .pt{fill:var(--on);stroke:var(--card);stroke-width:2}
.sc .pt:hover,.sc .pt:focus{stroke:var(--fg);outline:none}
.sc .trunc path{stroke:var(--bad);stroke-width:2.2;fill:none;stroke-linecap:round}
.sc .hit{fill:transparent}
.sc .mean{fill:none;stroke:var(--fg);stroke-width:1.6}
.sc .cap{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 3}
.sc .lead{stroke:var(--muted);stroke-width:1}
.sc .dl{font:500 12px var(--sans);fill:var(--fg);paint-order:stroke;stroke:var(--card);stroke-width:3px}
/* tables */
.wrap{overflow-x:auto;margin:.3rem 0 .6rem}.wrap.tall{max-height:42rem;overflow:auto}.wrap.mid{max-height:30rem;overflow:auto}
table{border-collapse:collapse;font-size:.86rem}
th{font-weight:500;text-align:left}
.hm th,.hm td{padding:.2rem .3rem}
.hm thead th{font-family:var(--cond);font-weight:600;font-size:1rem;text-align:center;min-width:3.2rem}
.hm thead th span{display:block;font:400 .7rem var(--mono);color:var(--muted)}
.rl{white-space:nowrap;padding-right:1rem!important;position:sticky;left:0;background:var(--card);z-index:1}
.md{font:400 .7rem var(--mono);padding:0 .3rem;border-radius:3px;border:1px solid var(--line);color:var(--muted);font-style:normal}
.md.on{border-color:var(--on)}.md.off{border-color:var(--off)}
.hc{text-align:center;font:400 .78rem var(--mono);font-variant-numeric:tabular-nums;
background:color-mix(in oklab,var(--seq) var(--p),var(--card));border:2px solid var(--card);border-radius:4px;min-width:3.2rem}
.hc.hi{color:var(--bg)}
:root[data-theme='dark'] .hc.hi{color:#1f211c}
tr.part{opacity:.5}
.wg th,.wg td{padding:0}
.wg .vh{height:9.5rem;vertical-align:bottom;padding:0 0 .3rem}
.wg .vh span{writing-mode:vertical-rl;transform:rotate(180deg);font-size:.74rem;white-space:nowrap;display:inline-block;margin:0 auto}
.wg .vh i{font-style:normal}
.wg td.wc{width:18px;min-width:18px;height:18px;background:color-mix(in oklab,var(--seq) var(--p),var(--track));
border:2px solid var(--card);border-radius:4px}
.wg td.wc:focus{outline:2px solid var(--fg)}
.wg .rl{font-size:.82rem;padding:0 .6rem 0 0!important}
.wg .nm{font:400 .64rem var(--mono);color:var(--muted)}
.wg .wk{font:400 .72rem var(--mono);color:var(--muted);padding-left:.5rem}
.wg tr.sect th{font-family:var(--cond);font-weight:700;font-size:1.05rem;padding:.6rem 0 .15rem;position:sticky;left:0;background:var(--card)}
.fp th,.fp td{padding:.25rem .6rem .25rem 0;border-bottom:1px solid var(--line);vertical-align:top}
.fp th{font-family:var(--cond);font-weight:600;text-transform:uppercase;letter-spacing:.05em;font-size:.8rem;color:var(--muted)}
.fp .num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.fp .w{font-weight:500}.fp .ss{color:var(--muted);font-size:.78rem}
.lab{font:400 .7rem var(--mono);padding:.02rem .35rem;border-radius:3px;border-left:3px solid var(--muted);background:var(--track)}
.lab.derived{border-left-color:var(--warn)}.lab.near{border-left-color:var(--warn);border-left-style:dashed}
.lab.misplaced{border-left-color:var(--off)}.lab.other{border-left-color:var(--bad)}.lab.tp{border-left-color:var(--good)}
/* runs explorer */
.chips{display:grid;grid-template-columns:minmax(9rem,13rem) 1fr;gap:.25rem .8rem;align-items:center;max-height:16rem;overflow-y:auto}
.chips .cn{font-size:.84rem;white-space:nowrap}
.chips .cs{display:flex;flex-wrap:wrap;gap:.25rem}
.chip{font:400 .72rem var(--mono);border:1px solid var(--line);border-left:3px solid var(--muted);background:var(--bg);
color:var(--fg);border-radius:0 3px 3px 0;padding:.08rem .45rem;cursor:pointer;font-variant-numeric:tabular-nums}
.chip.cur{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.chip.bad{border-left-color:var(--bad)}
.rv{margin-top:.8rem;border-top:1px solid var(--line);padding-top:.6rem}
.rvh{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:baseline;font-size:.86rem;margin-bottom:.4rem}
.rvh b{font-family:var(--cond);font-size:1.15rem}
.rvh .mono{color:var(--muted)}
.ans{display:grid;grid-template-columns:2.2rem 1fr;gap:.25rem .6rem;font-size:.86rem}
.ans .k{font-family:var(--cond);font-weight:700;font-size:1.05rem}
.ans .ws{display:flex;flex-wrap:wrap;gap:.2rem}
.ans .miss{color:var(--muted);font-size:.78rem;grid-column:2}
.ans .miss s{text-decoration-color:var(--muted)}
details{margin-top:.6rem}summary{cursor:pointer;color:var(--muted);font-size:.86rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:400 .76rem/1.5 var(--mono);background:var(--bg);border:1px solid var(--line);
border-radius:4px;padding:.5rem .7rem;max-height:24rem;overflow-y:auto}
#tt{position:fixed;z-index:50;display:none;max-width:min(26rem,90vw);pointer-events:none;background:var(--card);
border:1px solid var(--line);border-left:3px solid var(--muted);border-radius:0 4px 4px 0;padding:.45rem .7rem;
font:400 .82rem/1.5 var(--sans);color:var(--fg);box-shadow:0 8px 26px rgba(0,0,0,.28);white-space:pre-wrap}
#tt b{font-family:var(--cond);font-weight:600;font-size:.9rem}
.foot{margin-top:2.5rem;padding-top:.8rem;border-top:1px solid var(--line);color:var(--muted);font-size:.84rem;max-width:100ch}
"""

DASH_CSS = """
body.dashb{padding:.7rem clamp(.7rem,1.6vw,1.4rem) .45rem}
.dash{display:flex;flex-direction:column;gap:.55rem}
.dash .mast{margin:0;padding-bottom:.5rem;gap:.6rem;min-width:0}
.dash .brand img{height:1.6rem}
.dash h1{font-size:1.4rem;margin:0;white-space:nowrap}
.dash .desc{color:var(--muted);font-size:.84rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:1}
.badge{font:500 .72rem var(--mono);color:var(--warn);border:1px solid var(--warn);border-radius:3px;padding:.05rem .4rem;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));gap:.55rem}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:.3rem .75rem .35rem;min-width:0;
display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:.6rem;align-items:baseline}
.kl{grid-column:1/3;font:600 .72rem var(--cond);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kv{font:700 1.6rem/1.15 var(--cond);font-variant-numeric:tabular-nums}.kv small{font-size:.95rem;color:var(--muted);font-weight:600}
.ks{font-size:.76rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card{background:var(--card);border:1px solid var(--line);border-radius:5px;display:flex;flex-direction:column;
min-height:0;min-width:0;padding:.45rem .8rem .6rem}
.ch{display:flex;align-items:center;gap:.6rem;min-height:1.9rem}
.ch h2{margin:0;font-size:1.12rem;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ch .ctl{margin:0;flex-wrap:nowrap;gap:.2rem}.ch .seg{font-size:.74rem;padding:.1rem .45rem;white-space:nowrap}
.xp{display:flex;align-items:center;justify-content:center;flex:none;width:1.7rem;height:1.7rem;border:1px solid var(--line);
background:var(--bg);color:var(--muted);border-radius:4px;cursor:pointer;padding:0}
.xp:hover,.xp:focus-visible{color:var(--fg);border-color:var(--muted)}
.xp svg{width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.card .i-close,.card.max .i-open{display:none}.card.max .i-close{display:block}
.hint{margin:0 0 .35rem;color:var(--muted);font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card.max .hint{white-space:normal;font-size:.9rem;max-width:90ch}
.cb{flex:1;min-height:0;overflow:auto}
.card:not(.max) .more{display:none}.card.max .less{display:none}
.scrim{position:fixed;inset:0;background:rgba(10,12,8,.55);z-index:39}
body.has-max{overflow:hidden}
.card.max{position:fixed;inset:clamp(.4rem,3vh,2rem) clamp(.4rem,3vw,3rem);z-index:40;padding:.7rem 1.2rem 1rem;
box-shadow:0 18px 60px rgba(0,0,0,.45)}
.card.max .ch h2{font-size:1.5rem}
.grid{display:grid;gap:.55rem;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr);
grid-template-areas:"lb sc fp" "lb kn runs"}
.c-lb{grid-area:lb}.c-sc{grid-area:sc}.c-kn{grid-area:kn}.c-fp{grid-area:fp}.c-runs{grid-area:runs}
.dash .foot{margin:0;padding-top:.3rem;font-size:.7rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:none}
.card .legend{font-size:.76rem;margin:0 0 .35rem}
.card .wrap,.card .wrap.tall,.card .wrap.mid{margin:0;max-height:none;overflow:visible}
.card .hm thead th,.card .wg thead th,.card .fp thead th{position:sticky;top:0;background:var(--card);z-index:2}
.card .hm thead th:first-child,.card .wg thead th:first-child{left:0;z-index:3}
/* leaderboard, compact */
.c-lb:not(.max) .lrow{grid-template-columns:minmax(6.5rem,10.5rem) minmax(0,1fr) 2.5rem 2.5rem;gap:.55rem;padding:.14rem 0}
.c-lb:not(.max) .lval.x{display:none}
.c-lb:not(.max) .lname{font-size:.84rem}
.c-lb:not(.max) .lname span{font-size:.66rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.c-lb:not(.max) .ltrack{height:18px}
.c-lb:not(.max) .legend dt:nth-of-type(n+4),.c-lb:not(.max) .legend dd:nth-of-type(n+4){display:none}
.card.max .lb{max-width:1100px}
/* scatter */
.c-sc .less{height:100%}.sc-c{display:block;width:100%;height:100%;max-width:none}
.card.max .sc{max-width:1000px;margin:0 auto}
/* knowledge: heatmap and word grid, compact */
.c-kn:not(.max) .hm th,.c-kn:not(.max) .hm td{padding:.05rem .1rem}
.c-kn:not(.max) .hm thead th{min-width:2.1rem;font-size:.9rem}
.c-kn:not(.max) .hc{min-width:2.1rem;font-size:.66rem;border-width:1.5px}
.c-kn:not(.max) .rl{font-size:.76rem;padding-right:.5rem!important}
.c-kn:not(.max) .wg .vh{height:7.6rem}.c-kn:not(.max) .wg .vh span{font-size:.66rem}
.c-kn:not(.max) .wg td.wc{width:14px;min-width:14px;height:14px;border-width:1.5px}
/* wrong words, compact */
.c-fp:not(.max) .fp .ss{display:none}
.c-fp:not(.max) .fp th,.c-fp:not(.max) .fp td{padding:.12rem .5rem .12rem 0;font-size:.8rem}
/* runs */
.card .chips{max-height:none;overflow:visible;grid-template-columns:minmax(7rem,10rem) minmax(0,1fr);gap:.2rem .6rem}
.c-runs:not(.max) .chips .cn{font-size:.78rem;overflow:hidden;text-overflow:ellipsis}
.c-runs.max .cb{display:grid;grid-template-columns:minmax(15rem,25rem) minmax(0,1fr);gap:1.4rem;overflow:hidden}
.c-runs.max .chips,.c-runs.max .rv{overflow:auto;min-height:0;align-content:start}
.c-runs.max .rv{margin:0;border:0;padding:0 .3rem 0 0}
/* setup tile */
.c-setup:not(.max){padding:.3rem .75rem .35rem}
.c-setup:not(.max) .ch{min-height:1.3rem}
.c-setup:not(.max) .ch h2{font:600 .72rem var(--cond);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.c-setup:not(.max) .xp{width:1.4rem;height:1.4rem}
.c-setup .less{font-size:.78rem;line-height:1.35;color:var(--muted)}
.c-setup .less b{color:var(--fg);font-weight:500}
@media (min-width:1100px) and (min-height:680px){
body.dashb{height:100vh;height:100dvh;overflow:hidden}
.dash{height:100%}
.grid{flex:1;min-height:0;grid-template-rows:minmax(0,1fr) minmax(0,1fr)}
}
@media (max-width:1099px),(max-height:679px){
.grid{grid-template-columns:minmax(0,1fr);grid-template-areas:"lb" "sc" "kn" "fp" "runs"}
.grid .card:not(.max) .cb{max-height:30rem}
.c-sc:not(.max) .cb{max-height:none}
.dash .mast{flex-wrap:wrap}.dash h1{white-space:normal;font-size:1.25rem}.dash .desc{white-space:normal;flex-basis:100%}
.dash .foot{white-space:normal}
}
@media (max-width:800px){
.c-runs.max .cb{grid-template-columns:minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}
.c-runs.max .chips{max-height:11rem}
.ch .seg{padding:.1rem .3rem}
.ch{flex-wrap:wrap;row-gap:.3rem}.ch .xp{order:2}.ch .ctl{order:3;flex-basis:100%;flex-wrap:wrap}
}
"""

FONTS = ("<link rel=preconnect href='https://fonts.googleapis.com'><link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
         "<link rel=stylesheet href='https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&"
         "family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap'>")
THEME_BOOT = ("<script>try{var t=localStorage.getItem('theme');if(t==='dark')document.documentElement.setAttribute('data-theme','dark');}"
              "catch(e){}</script>")
SUN = ("<svg class=sun viewBox='0 0 24 24' aria-hidden=true><circle cx=12 cy=12 r=4 /><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4"
       "M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4' /></svg>")
MOON = "<svg class=moon viewBox='0 0 24 24' aria-hidden=true><path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' /></svg>"

JS = r"""
<div id=tt role=tooltip></div>
<script>
(function(){
document.addEventListener('click',function(e){var b=e.target.closest&&e.target.closest('[data-theme-toggle]');if(!b)return;
var d=document.documentElement,n=d.getAttribute('data-theme')==='dark'?'light':'dark';
if(n==='dark')d.setAttribute('data-theme','dark');else d.removeAttribute('data-theme');try{localStorage.setItem('theme',n);}catch(err){}});
var tt=document.getElementById('tt');
function show(el){tt.innerHTML=el.getAttribute('data-tip');tt.style.display='block';var r=el.getBoundingClientRect(),w=tt.offsetWidth,h=tt.offsetHeight;
var x=Math.min(Math.max(8,r.left+r.width/2-w/2),innerWidth-w-8),y=r.bottom+6;if(y+h>innerHeight-8)y=Math.max(8,r.top-6-h);
tt.style.left=x+'px';tt.style.top=y+'px';}
function hide(){tt.style.display='none';}
document.addEventListener('mouseover',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');if(el)show(el);else hide();});
document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');if(el)show(el);});
document.addEventListener('focusout',hide);addEventListener('scroll',hide,true);
document.querySelectorAll('[data-metric-btn]').forEach(function(b){b.addEventListener('click',function(){
var k=b.getAttribute('data-metric-btn');document.querySelectorAll('[data-metric-btn]').forEach(function(x){x.classList.toggle('on',x===b);});
document.querySelectorAll('.lb').forEach(function(p){p.hidden=p.getAttribute('data-metric')!==k;});});});
// dashboard cards: expand to full screen and back (button, Esc, click outside); tabs inside a card
var scrim=document.querySelector('.scrim'),opened=null,opener=null;
function setMax(c,on){if(on){if(opened&&opened!==c)setMax(opened,false);opened=c;opener=document.activeElement;}else opened=null;
c.classList.toggle('max',on);document.body.classList.toggle('has-max',on);if(scrim)scrim.hidden=!on;
var b=c.querySelector('[data-expand]');b.setAttribute('aria-expanded',on?'true':'false');
b.setAttribute('aria-label',(on?'Close: ':'Expand: ')+c.querySelector('h2').textContent);hide();
if(on)b.focus();else if(opener&&opener.focus&&document.contains(opener))opener.focus();}
document.addEventListener('click',function(e){if(!e.target.closest)return;var b=e.target.closest('[data-expand]');
if(b){var c=b.closest('.card');setMax(c,!c.classList.contains('max'));return;}
if(e.target.closest('[data-close]')&&opened)setMax(opened,false);});
document.addEventListener('keydown',function(e){if(e.key==='Escape'&&opened)setMax(opened,false);});
document.querySelectorAll('[data-tab]').forEach(function(b){b.addEventListener('click',function(){var c=b.closest('.card'),k=b.getAttribute('data-tab');
c.querySelectorAll('[data-tab]').forEach(function(x){x.classList.toggle('on',x===b);});
c.querySelectorAll('[data-pane]').forEach(function(p){p.hidden=p.getAttribute('data-pane')!==k;});});});
// runs explorer
var box=document.getElementById('chips'),view=document.getElementById('rv');if(!box)return;
var runsCard=box.closest('.card');
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
var order=[],groups={};RUNS.forEach(function(r,i){if(!groups[r.s]){groups[r.s]=[];order.push(r.s);}groups[r.s].push(i);});
if(typeof ORDER!=='undefined')order=ORDER.filter(function(s){return groups[s];});
var html='';order.forEach(function(s){var dm=NAMES[s][2]?' closed':'';html+='<div class="cn'+dm+'">'+esc(NAMES[s][0])+' <span class="md '+NAMES[s][1]+'">'+NAMES[s][1]+'</span></div><div class="cs'+dm+'">';
groups[s].sort(function(a,b){return RUNS[a].seed-RUNS[b].seed;}).forEach(function(i){var r=RUNS[i];
html+='<button type=button class="chip'+((!r.ok||r.fin==='length')?' bad':'')+'" data-i='+i+'>#'+r.seed+' '+r.f1.toFixed(2)+(r.fo==='cap'||r.fo==='answer'?' ▪':'')+(r.fo==='pending'?' …':'')+'</button>';});html+='</div>';});
box.innerHTML=html;
var LAB={tp:'in the gold list',derived:'derived word (in brackets in ÚJČ)',near:'one edit from a gold word',misplaced:'gold word under the wrong letter',other:'not a listed word',skip:'not scored (prefix vy-)'};
var cur=0;
function render(i){cur=i;var r=RUNS[i];box.querySelectorAll('.chip').forEach(function(c){c.classList.toggle('cur',+c.getAttribute('data-i')===i);});
var h='<div class=rvh><b>'+esc(NAMES[r.s][0])+', reasoning '+NAMES[r.s][1]+', seed '+r.seed+'</b>'+
'<span class=mono>F1 '+r.f1.toFixed(3)+' · P '+r.p.toFixed(3)+' · R '+r.r.toFixed(3)+'</span>'+
'<span class=mono>'+(r.ct==null?'–':r.ct.toLocaleString('en'))+' completion tokens · '+(r.rt==null?'–':r.rt.toLocaleString('en'))+' reasoning · '+
(r.lat==null?'':Math.round(r.lat)+' s · ')+'finish '+esc(r.fin)+'</span></div>';
if(r.fo==='cap')h+='<div class=cav><b>Forced answer.</b> The reasoning reached the 32k cap ('+(r.rt1==null?'':r.rt1.toLocaleString('en')+' tokens in the first call, ')+'kept '+r.rt.toLocaleString('en')+'); the end of thinking was forced and the model answered in a second call. The trace below is the first call\'s.</div>';
if(r.fo==='answer')h+='<div class=cav><b>Answer continued.</b> The reasoning ended within the cap but the first call ran out mid-answer; a second call continued it.</div>';
if(r.fo==='pending')h+='<div class=cav><b>Forced answer pending.</b> This run reached the cap; its forced answer has not been generated yet.</div>';
if(!r.ok)h+='<div class=cav><b>No JSON object could be parsed</b>'+(r.fin==='length'?': the run hit the output cap before answering.':'.')+'</div>';
h+='<div class=ans>';r.L.forEach(function(kv){h+='<div class=k>'+esc(kv[0])+'</div><div class=ws>';kv[1].forEach(function(w){
h+='<span class="lab '+w[1]+'" title="'+esc(LAB[w[1]]||w[1])+'">'+esc(w[0])+'</span>';});h+='</div>';
var L=kv[0].toLowerCase().charAt(0);if(r.miss[L]&&r.miss[L].length)h+='<div class=miss>missed: '+r.miss[L].map(function(x){return '<s>'+esc(x)+'</s>';}).join(', ')+'</div>';});
h+='</div><details><summary>raw answer ('+r.raw.length.toLocaleString('en')+' characters shown)</summary><pre>'+esc(r.raw)+'</pre></details>';
if(r.rsn)h+='<details><summary>reasoning trace ('+r.rsn.toLocaleString('en')+' characters)</summary><pre>'+esc(r.rs)+'</pre></details>';
view.innerHTML=h;view.scrollTop=0;var cc=box.querySelector('.chip.cur');if(cc&&runsCard&&runsCard.classList.contains('max'))cc.scrollIntoView({block:'nearest'});}
box.addEventListener('click',function(e){var c=e.target.closest('.chip');if(!c)return;render(+c.getAttribute('data-i'));
if(runsCard&&!runsCard.classList.contains('max'))setMax(runsCard,true);});
document.addEventListener('keydown',function(e){var t=e.target;if(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.isContentEditable)return;
if(e.altKey||e.ctrlKey||e.metaKey)return;if(runsCard&&!runsCard.classList.contains('max'))return;var chips=[].slice.call(box.querySelectorAll('.chip')).map(function(c){return +c.getAttribute('data-i');});
var k=chips.indexOf(cur);if(e.key==='ArrowRight'&&k<chips.length-1)render(chips[k+1]);else if(e.key==='ArrowLeft'&&k>0)render(chips[k-1]);});
if(RUNS.length)render(+box.querySelector('.chip').getAttribute('data-i'));
})();
</script>"""


def build(D: dict, gold: Gold) -> str:
    agg = D["agg"]
    partial = [s for s, a in agg.items() if a["partial"]]
    missing = [s for s in SETTINGS if s not in agg]
    status = ""
    if partial or missing:
        status = (f"<span class=badge title='{len(D['runs'])} runs so far; {len(partial)} setting(s) have fewer runs than "
                  f"planned (dimmed), {len(missing)} have none yet'>incomplete</span>")
    backends = ", ".join(dict.fromkeys(w for _, _, w, _ in MODELS.values()))
    ack = ("Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of "
           "Education, Youth and Sports of the Czech Republic.")
    lb = card("lb", "Leaderboard", "", leaderboard(D), tools=metric_buttons())
    sc = card("sc", "What reasoning costs", "Each dot is one reasoning-on run, rings are model means; reasoning tokens on a log scale.",
              f"<div class=less>{scatter(D, 470, 300, compact=True)}</div><div class=more>{scatter(D)}</div>")
    kn = card("kn", "What each model knows",
              "Recall per letter and for the proper names (mean over runs), or every gold item against every setting.",
              f"<div data-pane=letters>{heatmap(D, gold)}</div><div data-pane=words hidden>{word_grid(D, gold)}</div>",
              tools=("<div class=ctl role=group aria-label=view><button type=button class='seg on' data-tab=letters>by letter</button>"
                     "<button type=button class=seg data-tab=words>by word</button></div>"))
    fp = card("fp", "Most common wrong words", "Listed words that aren't gold items, ranked by how many settings produced them (top 80).",
              false_positives(D))
    runs = card("runs", "Every run", "Pick a run to open its answer, the gold items it missed and its reasoning trace (← → to step).",
                "<div class=chips id=chips></div><div class='rv more' id=rv></div>")
    setup = card("setup", "Setup & prompt", "",
                 f"<div class=less><b>{len(MODELS)} models</b>, {len(SETTINGS)} settings<br>{esc(backends)}</div>"
                 f"<div class=more>{setup_table(D)}<details open><summary>the prompt (Czech, no system prompt)</summary>"
                 f"<pre>{esc(PROMPT)}</pre></details><p class=sub>{ack}</p></div>", cls="tile")
    body = f"""
<div class=dash>
<header class=mast><a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>
<img class=logo-light src='static/logo_dark.png' alt='lokalni.ai' width=102 height=32>
<img class=logo-dark src='static/logo_light.png' alt='lokalni.ai' width=102 height=32></a>
<span class=mastsep>&frasl;</span><h1>Do LLMs know the Czech vyjmenovaná slova?</h1>
<span class=desc>{len(model_order(agg))} models list the <i>vyjmenovaná slova</i> as JSON, scored against the {len(gold.items)} items
outside brackets in ÚJČ's <a href='{SOURCE}'>Přehled vyjmenovaných slov</a>.</span>{status}
<button class=tgl type=button data-theme-toggle aria-label='Switch theme' title='Switch theme'>{SUN}{MOON}</button></header>
<div class=kpis>{kpis(D, gold)}{setup}</div>
<main class=grid>{lb}{sc}{kn}{fp}{runs}</main>
<footer class=foot>{ack}</footer>
</div>
<div class=scrim data-close hidden></div>
"""
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(TITLE)} · lokalni.ai</title><link rel=icon href='favicon.ico' sizes=any>{THEME_BOOT}{FONTS}"
            f"<style>{CSS}{DASH_CSS}</style></head><body class=dashb>{body}{runs_blob(D, gold)}{JS}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    a = ap.parse_args()
    gold = Gold(ROOT / "data" / "gold.json")
    D = collect(a.raw, gold)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "static").mkdir(exist_ok=True)
    if ASSETS.exists():   # brand assets from the sibling checkout; a clean clone keeps the copies already in docs/
        for f in ("logo_dark.png", "logo_light.png"):
            shutil.copy(ASSETS / f, a.out / "static" / f)
        shutil.copy(ASSETS / "favicon.ico", a.out / "favicon.ico")
    page = build(D, gold)
    (a.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{a.out / 'index.html'}: {len(page) / 1e6:.2f} MB, {len(D['runs'])} runs, {len(D['agg'])} settings")


if __name__ == "__main__":
    main()
