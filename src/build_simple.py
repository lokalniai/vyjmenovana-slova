"""Build the one-chart summary: docs/summary.html, next to the full viewer (docs/index.html). Standard library only.

    python -I src/build_simple.py [--raw results/raw] [--out docs]

Run src/build_site.py too: this page shares its data, styles and brand assets. Every number, including the ones in
the prose, is computed from the raw runs, so the page can't drift from results/summary.md.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_site import CSS, FONTS, JS, LEGEND, MOON, SOURCE, SUN, THEME_BOOT, TITLE, collect, esc, external_section, fmt_tok, metric_panel, model_order  # noqa: E402
from score import Gold  # noqa: E402
from settings import MODELS, REASONING_CAP, RUNS, SETTINGS  # noqa: E402

ROOT = HERE.parent
EXTRA_CSS = """
.simple{max-width:1040px;margin-inline:auto}
.fp{width:100%}.fp td,.fp th{white-space:nowrap}.fp td:first-child{font-weight:500}
.fp th.num{text-align:right;font-family:var(--cond)}.nb{white-space:nowrap}
"""


def num(a: dict | None, key: str) -> str:
    return "–" if a is None else f"{a[key]:.2f}"


def table(agg: dict) -> str:
    rows = []
    for m in model_order(agg):
        off, on = agg.get(f"{m}/off"), agg.get(f"{m}/on")
        forced = "–" if on is None else f"{on['forced']}/{on['n']}"
        no_json = sum(a["parse_fail"] for a in (off, on) if a)
        n = sum(a["n"] for a in (off, on) if a)
        rows.append(f"<tr><td>{esc(MODELS[m][0])}</td>"
                    f"<td class=num>{num(off, 'f1')}</td><td class=num>{num(on, 'f1')}</td>"
                    f"<td class=num>{num(on, 'recall')}</td><td class=num>{num(on, 'precision')}</td>"
                    f"<td class=num>{num(on, 'precision_lenient')}</td>"
                    f"<td class=num>{fmt_tok(on['reasoning_median']) if on else '–'}</td>"
                    f"<td class=num>{forced}</td><td class=num>{no_json}/{n}</td></tr>")
    heads = ["model", "F1 off", "F1 on", "recall on", "precision on", "lenient on", "reasoning tokens", "forced answers", "no JSON"]
    th = "".join(f"<th{'' if i == 0 else ' class=num'}>{h}</th>" for i, h in enumerate(heads))
    return f"<div class=wrap><table class=fp><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build(D: dict, gold: Gold) -> str:
    agg = D["agg"]
    best = max(agg, key=lambda s: agg[s]["f1"])
    second = next((m for m in model_order(agg) if m != best.split("/")[0]), None)
    runner = max((s for s in agg if s.split("/")[0] == second), key=lambda s: agg[s]["f1"]) if second else None
    partial = [s for s, a in agg.items() if a["partial"]]
    missing = [s for s in SETTINGS if s not in agg]
    status = ""
    if partial or missing:
        status = (f"<div class=cav><b>Incomplete.</b> {len(D['runs'])} runs so far; {len(partial)} setting(s) have fewer "
                  f"than {RUNS} runs (dimmed){', ' + str(len(missing)) + ' have none yet' if missing else ''}.</div>")
    line = f"Among them, <b class=nb>{esc(MODELS[best.split('/')[0]][0])}</b> leads with F1 {agg[best]['f1']:.2f} with reasoning {best.split('/')[1]}"
    if runner:
        line += (f"; the next best model, <span class=nb>{esc(MODELS[second][0])}</span>, reaches {agg[runner]['f1']:.2f}")
    n_common = sum(i["kind"] == "common" for i in gold.items.values())
    body = f"""
<div class=simple>
<header class=mast><a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>
<img class=logo-light src='static/logo_dark.png' alt='lokalni.ai' width=102 height=32>
<img class=logo-dark src='static/logo_light.png' alt='lokalni.ai' width=102 height=32></a>
<span class=mastsep>&frasl;</span><span class=here>{esc(TITLE)}</span><span class=spacer></span>
<button class=tgl type=button data-theme-toggle aria-label='Switch theme' title='Switch theme'>{SUN}{MOON}</button></header>
<h1>Do LLMs know the Czech vyjmenovaná slova?</h1>
<p class=lede>{len(model_order(agg))} open-weight models listed the <i>vyjmenovaná slova</i> as JSON, {RUNS} times each, with
reasoning off and on. {line}. Lists are scored against the {len(gold.items)} items outside brackets in ÚJČ's
<a href='{SOURCE}'>Přehled vyjmenovaných slov</a> ({n_common} words and {len(gold.items) - n_common} proper names).</p>
{status}
<h2>F1 with reasoning off and on</h2>
<p class=sub>Mean over {RUNS} runs. Reasoning is capped at {REASONING_CAP // 1000}k tokens, then the model has to answer.</p>
<div class=panel>{LEGEND}{metric_panel(agg, "f1", "F1")}</div>
<h2>The numbers</h2>
<p class=sub>Precision counts ÚJČ's bracketed derived words as wrong; lenient precision leaves them out. An answer without
parseable JSON scores 0. Reasoning tokens are the median with reasoning on.</p>
<div class=panel>{table(agg)}</div>
{external_section(gold)}
<p class=sub>All metrics, recall by letter, every word and every run with its reasoning trace are in the
<a href='index.html'>full viewer</a>.</p>
<div class=foot>Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of
Education, Youth and Sports of the Czech Republic.</div>
</div>
"""
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(TITLE)}: summary · lokalni.ai</title><link rel=icon href='favicon.ico' sizes=any>{THEME_BOOT}{FONTS}"
            f"<style>{CSS}{EXTRA_CSS}</style></head><body>{body}{JS}</body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    a = ap.parse_args()
    gold = Gold(ROOT / "data" / "gold.json")
    a.out.mkdir(parents=True, exist_ok=True)
    page = build(collect(a.raw, gold), gold)
    (a.out / "summary.html").write_text(page, encoding="utf-8")
    print(f"{a.out / 'summary.html'}: {len(page) // 1024} kB")


if __name__ == "__main__":
    main()
