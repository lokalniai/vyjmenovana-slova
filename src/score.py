"""Score the raw runs against data/gold.json.

    python -I src/score.py [--raw results/raw] [--out results]

Per run: parse the first JSON object in the answer, normalize every listed word, and match it against the gold
items *of the key it was listed under*. A gold item with two spellings (myslit/myslet) counts once.
- TP: gold items listed. Recall = TP / gold items (also split into common words and proper names).
- Every other unique listed word is a false positive, labelled as `derived` (a bracketed ÚJČ word such as
  byt), `misplaced` (a gold word under the wrong letter), `near` (one edit from a gold word, e.g. a missing
  diacritic) or `other`. Strict precision = TP / (TP + FP). Lenient precision leaves `derived` out.
- The ÚJČ "předpona vy-/vý-" item is not scored either way.
- A run whose answer has no parseable JSON object scores 0.
"""

import argparse
import csv
import json
import re
import statistics as st
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import REASONING_CAP, SETTINGS  # noqa: E402

SLACK = 64   # e-INFRA reports 31,997 reasoning tokens for a run stopped at 32,000; the native budget adds end tokens


def reasoning_tokens(r: dict) -> int | None:
    return ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")


def needs_force(r: dict) -> str | None:
    """Why a reasoning-on run needs a forced answer: "cap" (reasoning reached the cap, or passed it where the budget
    was not enforced) or "answer" (reasoning ended in time but the answer was cut off); None if it is complete.
    OpenRouter runs (budget "effort") are never forced: a closed API can't continue cut reasoning, so they score as is."""
    if not r["setting"].endswith("/on") or SETTINGS[r["setting"]].get("budget") == "effort":
        return None
    if (reasoning_tokens(r) or 0) > REASONING_CAP + SLACK or (r["finish_reason"] == "length" and not r["content"].strip()):
        return "cap"
    if r["finish_reason"] == "length" and SETTINGS[r["setting"]].get("budget") == "forced":
        return "answer"
    return None
LETTERS = "blmpsvz"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)                     # "nazývat (se)", "výr (pták)"
    s = s.strip(" \t\n.,;:!?\"'„“”‚‘’*")         # hyphens stay: "vy-" is the prefix, "vy" the pronoun
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r" (se|si)$", "", s)
    return s.strip()


def split_items(v) -> list[str]:
    """A list item can hold several words: "myslit i myslet", "být, bydlit", "mýt/mýdlo"."""
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v if isinstance(v, list) else []:
        if isinstance(x, (list, dict)):
            out += split_items(x)
            continue
        for part in re.split(r"[,;/]| i (?=\S)", str(x)):
            p = norm(part)
            if p:
                out.append(p)
    return out


def parse_answer(text: str):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj:
            return obj
    return None


def edit1(a: str, b: str) -> bool:
    """Levenshtein distance <= 1, or equal once diacritics are stripped."""
    strip = lambda s: "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    if strip(a) == strip(b):
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


class Gold:
    def __init__(self, path: Path):
        g = json.loads(path.read_text(encoding="utf-8"))["letters"]
        self.items = {}      # id -> {letter, lemma, kind}
        self.lookup = {}     # (letter, normalized variant) -> id
        self.any_letter = {} # normalized variant -> letter (for misplaced)
        self.derived = {}    # normalized derived word -> letter
        for L in LETTERS:
            for it in g[L]["items"]:
                iid = f"{L}:{it['lemma']}"
                self.items[iid] = {"letter": L, "lemma": it["lemma"], "kind": it["kind"]}
                for v in it["variants"]:
                    self.lookup[(L, norm(v))] = iid
                    self.any_letter[norm(v)] = L
            for d in g[L]["derived"]:
                self.derived.setdefault(norm(d), L)
        self.variants = {v: iid for (L, v), iid in self.lookup.items()}

    def label(self, key: str, w: str) -> str:
        if w in self.derived:
            return "derived"
        if w in self.any_letter and self.any_letter[w] != key:
            return "misplaced"
        if any(edit1(w, v) for v in self.variants):
            return "near"
        return "other"


PREFIX = re.compile(r"^(předpona )?v[yý][-‑–]$")


def score_run(answer, gold: Gold) -> dict:
    n_gold = len(gold.items)
    n_common = sum(i["kind"] == "common" for i in gold.items.values())
    if answer is None:
        return {"parse_ok": False, "precision": 0.0, "precision_lenient": 0.0, "recall": 0.0, "f1": 0.0,
                "recall_common": 0.0, "recall_names": 0.0, "hits": [], "fps": [], "keys": [], "n_listed": 0,
                **{f"recall_{L}": 0.0 for L in LETTERS}, **{f"precision_{L}": 0.0 for L in LETTERS}}
    hits, fps, keys = set(), {}, []
    per_key_listed = {}
    for k, v in answer.items():
        key = norm(k)[:1] if norm(k) else "?"
        keys.append(norm(k))
        for w in split_items(v):
            if PREFIX.match(w):
                continue
            iid = gold.lookup.get((key, w))
            if iid:
                hits.add(iid)
                per_key_listed.setdefault(key, set()).add(iid)
            else:
                fps[(key, w)] = gold.label(key, w)
                per_key_listed.setdefault(key, set()).add(("fp", w))
    tp, fp = len(hits), len(fps)
    fp_derived = sum(v == "derived" for v in fps.values())
    prec = tp / (tp + fp) if tp + fp else 0.0
    prec_len = tp / (tp + fp - fp_derived) if tp + fp - fp_derived else 0.0
    rec = tp / len(gold.items)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"parse_ok": True, "precision": prec, "precision_lenient": prec_len, "recall": rec, "f1": f1,
           "recall_common": sum(gold.items[i]["kind"] == "common" for i in hits) / n_common,
           "recall_names": sum(gold.items[i]["kind"] == "name" for i in hits) / (n_gold - n_common),
           "n_listed": tp + fp, "tp": tp, "fp": fp,
           **{f"fp_{c}": sum(v == c for v in fps.values()) for c in ("derived", "misplaced", "near", "other")},
           "keys": sorted(set(keys)), "has_f": any(k.startswith("f") for k in keys),
           "letters_ok": set(k[:1] for k in keys) >= set(LETTERS),
           "hits": sorted(hits), "fps": [{"key": k, "word": w, "label": l} for (k, w), l in sorted(fps.items())]}
    for L in LETTERS:
        gL = [i for i, x in gold.items.items() if x["letter"] == L]
        hL = [i for i in hits if gold.items[i]["letter"] == L]
        listed = per_key_listed.get(L, set())
        out[f"recall_{L}"] = len(hL) / len(gL)
        out[f"precision_{L}"] = len(hL) / len(listed) if listed else 0.0
    return out


def usage_fields(r: dict) -> dict:
    u = r.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": r.get("reasoning_tokens_eff"), "reasoning_tokens_phase1": det.get("reasoning_tokens"),
            "forced": r.get("forced"), "finish_reason": r.get("finish_reason"),
            "latency_s": r.get("latency_s"), "reasoning_chars": len(r.get("reasoning") or "")}


def load_runs(raw: Path) -> list[dict]:
    """Phase-1 records with the current request settings, merged with their forced answers (<setting>.forced.jsonl).
    Adds `forced` ("cap"/"answer" when a forced answer replaced the content, "pending" when one is needed but missing)
    and `reasoning_tokens_eff` (reasoning tokens the answer was conditioned on: at most REASONING_CAP)."""
    runs, forced = {}, {}
    for p in sorted(raw.glob("*.forced.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            f = json.loads(line)
            if f.get("error") is None:
                forced[(f["setting"], f["seed"], f["phase1_time"])] = f
    for p in sorted(raw.glob("*.jsonl")):
        if p.name.endswith(".forced.jsonl"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cfg = SETTINGS.get(r["setting"])
            # skip failures and stale records made with request settings that have since been corrected
            if r.get("error") is None and cfg is not None and r.get("extra") == cfg["extra"]:
                runs[(r["setting"], r["seed"])] = r   # last successful record wins
    out = []
    for k in sorted(runs):
        r = dict(runs[k])
        rt = reasoning_tokens(r)
        r["reasoning_tokens_eff"] = rt if rt is not None else (0 if not r.get("reasoning") else None)
        r["forced"] = None
        case = needs_force(r)
        if case:
            f = forced.get((r["setting"], r["seed"], r["time"]))
            if f is None:
                r["forced"] = "pending"
            else:
                r.update(content=f["content"], finish_reason=f["finish_reason"], forced=case, forced_record=f)
                if case == "cap":
                    r["reasoning_tokens_eff"] = f["reasoning_tokens_kept"]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    gold = Gold(ROOT / "data" / "gold.json")
    rows, items = [], []
    for r in load_runs(a.raw):
        s = score_run(parse_answer(r.get("content")), gold)
        base = {"setting": r["setting"], "model": r["setting"].split("/")[0], "mode": r["setting"].split("/")[1],
                "seed": r["seed"], **usage_fields(r)}
        rows.append({**base, **{k: v for k, v in s.items() if k not in ("hits", "fps", "keys")}})
        items.append({**base, "hits": s["hits"], "fps": s["fps"], "keys": s["keys"]})
    if not rows:
        sys.exit(f"no runs in {a.raw}")
    a.out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(a.out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # summary
    lines = ["| setting | n | parse ok | F1 | precision | prec. lenient | recall | rec. common | rec. names | listed | reasoning tok (median) | forced answers | answer cut |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
    for s in dict.fromkeys(r["setting"] for r in rows):
        R = [r for r in rows if r["setting"] == s]
        rt = [r["reasoning_tokens"] for r in R if r["reasoning_tokens"] is not None]
        lines.append(f"| {s} | {len(R)} | {sum(r['parse_ok'] for r in R)} | {ms([r['f1'] for r in R])} | "
                     f"{ms([r['precision'] for r in R])} | {ms([r['precision_lenient'] for r in R])} | "
                     f"{ms([r['recall'] for r in R])} | {ms([r['recall_common'] for r in R])} | "
                     f"{ms([r['recall_names'] for r in R])} | {st.mean(r['n_listed'] for r in R):.0f} | "
                     f"{int(st.median(rt)) if rt else '–'} | {sum(r['forced'] in ('cap', 'answer') for r in R)}"
                     f"{' (+' + str(sum(r['forced'] == 'pending' for r in R)) + ' pending)' if any(r['forced'] == 'pending' for r in R) else ''} | "
                     f"{sum(r['finish_reason'] == 'length' for r in R)} |")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
