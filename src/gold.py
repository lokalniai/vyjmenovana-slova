"""Build data/gold.json from the frozen ÚJČ page (prirucka.ujc.cas.cz/?id=100, "Přehled vyjmenovaných slov").

Gold = the words outside brackets in each letter's row. Everything inside brackets (derived words such as
byt, nábytek, mlynář) is kept separately as `derived`, so the scorer can tell "derived word" from "wrong".
Items after the first semicolon are proper names (`names`). The "předpona vy-/vý-" item is not a word and is
left out. "X i Y" (myslit i myslet) is one item with two accepted spellings.

    python -I src/gold.py
"""

import html
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "source" / "ujc_ijp_id100.html"
OUT = ROOT / "data" / "gold.json"
LETTERS = "blmpsvz"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def page_text() -> str:
    s = SRC.read_text(encoding="utf-8")
    t = html.unescape(re.sub(r"<[^>]+>", " ", s))
    t = t.replace("⁠", "").replace("‑", "-").replace("\xa0", " ")
    return nfc(re.sub(r"[ \t]+", " ", t))


def letter_rows(t: str) -> dict[str, str]:
    start = t.index("\n B : ")              # the first "Přehled ..." is the page's table of contents
    end = t.index("Původ vyjmenovaných slov", start)
    block = t[start:end]
    rows = {}
    for line in block.splitlines():
        m = re.match(r"\s*([BFLMPSVZ]) : (.*)", line)
        if m:
            rows[m.group(1).lower()] = m.group(2).strip()
    return rows


def split_brackets(row: str) -> tuple[str, list[str]]:
    """Return (row with bracketed parts removed, list of bracketed contents), handling nested brackets."""
    outside, inside, depth, cur = [], [], 0, []
    for ch in row:
        if ch == "(":
            if depth == 0:
                cur = []
            else:
                cur.append(ch)
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inside.append("".join(cur))
            else:
                cur.append(ch)
        elif depth:
            cur.append(ch)
        else:
            outside.append(ch)
    return "".join(outside), inside


def clean_item(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" .,;")
    s = re.sub(r" (se|si)$", "", s)
    return s


def variants(item: str) -> list[str]:
    return [clean_item(v) for v in item.split(" i ")]


def parse_derived(contents: list[str]) -> list[str]:
    out = []
    for c in contents:
        c, _ = split_brackets(c)           # drop nested notes such as "(tj. lýkový)"
        for part in re.split(r"[,;]", c):
            part = part.replace("…", "").strip()
            if not part or part.startswith(("tj.", "zájmeno", "např.")):
                continue
            out += [v for v in variants(part) if v]
    return out


def main() -> None:
    rows = letter_rows(page_text())
    assert set(rows) == set(LETTERS) | {"f"}, rows.keys()
    gold = {}
    for L in LETTERS:
        outside, inside = split_brackets(rows[L])
        common, _, names = outside.partition(";")
        items = []
        for chunk, kind in ((common, "common"), (names, "name")):
            for part in re.split(r"[,;]", chunk):
                part = clean_item(part)
                if not part or part.startswith("předpona"):
                    continue
                items.append({"lemma": variants(part)[0], "variants": variants(part), "kind": kind})
        gold[L] = {"items": items, "derived": sorted(set(parse_derived(inside)))}
    n_common = sum(i["kind"] == "common" for L in gold for i in gold[L]["items"])
    n_name = sum(i["kind"] == "name" for L in gold for i in gold[L]["items"])
    OUT.write_text(json.dumps({"source": "https://prirucka.ujc.cas.cz/?id=100", "fetched": "2026-11-02",
                               "letters": gold}, ensure_ascii=False, indent=1), encoding="utf-8")
    for L in LETTERS:
        its = gold[L]["items"]
        print(f"{L}: {len(its):2d} items ({sum(i['kind']=='name' for i in its)} names), "
              f"{len(gold[L]['derived'])} derived | " + ", ".join("/".join(i["variants"]) for i in its))
    print(f"total: {n_common} common + {n_name} names = {n_common + n_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
