"""Second phase of the reasoning cap: make runs that ran out of budget answer.

    python -I src/force_answer.py --settings 'gpt-oss-120b/on' 'qwen3.8-*/on' 'gemma-4-31b/on'        # e-INFRA
    python -I src/force_answer.py --settings 'gemma-4-e4b/on' --base-url http://localhost:8000/v1 --raw DIR   # local

For every first-phase record of a reasoning-on setting (made with the current request settings):
- reasoning ended within REASONING_CAP and the answer finished          -> nothing to do
- reasoning reached or passed REASONING_CAP (case "cap")                 -> cut it to REASONING_CAP tokens, append the
  model's end-of-thinking marker and let the model answer
- reasoning ended within the cap but the answer was cut off (case "answer") -> continue the answer
The continuation is a raw /v1/completions call on the prompt rendered with the checkpoint's own chat template
(Hugging Face tokenizer files only), so it does not depend on how a server's template treats an assistant prefill.
The rendered prompt is checked against the first phase's prompt_tokens. Records go to <raw>/<setting>.forced.jsonl.
"""

import argparse
import asyncio
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import needs_force  # noqa: E402
from settings import ANSWER_BUDGET, EINFRA, LOCAL, MODELS, PROMPT, REASONING_CAP, SETTINGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = Path(os.environ.get("EINFRA_TOKEN_FILE", "/storage/plzen1/home/kasnezde/.config/einfra_token"))
DRY_RUN = False


def family(setting: str) -> str:
    m = setting.split("/")[0]
    if m.startswith("qwen"):
        return "qwen"
    if m.startswith("gemma"):
        return "gemma"
    if m.startswith("gpt-oss"):
        return "harmony"
    raise SystemExit(f"{setting}: no forced-answer format for this model family (Kimi/DeepSeek use the native budget)")


def repo(setting: str) -> str:
    m = setting.split("/")[0]
    return LOCAL[m]["repo"] if SETTINGS[setting]["backend"] == "local" else MODELS[m][1]


def render(tok, setting: str, reasoning: str, content: str) -> tuple[str, str]:
    """(prompt prefix as the server rendered it for phase 1, full continuation prompt)."""
    cfg = SETTINGS[setting]
    kw = dict(cfg["extra"].get("chat_template_kwargs", {}))
    if "reasoning_effort" in cfg["extra"]:
        kw["reasoning_effort"] = cfg["extra"]["reasoning_effort"]
    prefix = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                     add_generation_prompt=True, **kw)
    fam = family(setting)
    if fam == "qwen":          # generation prompt already ends with "<think>\n"
        assert prefix.endswith("<think>\n"), prefix[-60:]
        body = reasoning.strip("\n") + "\n</think>\n\n"
    elif fam == "gemma":       # the model opens its own thought channel after "<|turn>model\n"
        body = "<|channel>thought\n" + reasoning.strip("\n") + "\n<channel|>"
    else:                      # harmony: analysis channel, then the final channel
        body = "<|channel|>analysis<|message|>" + reasoning + "<|end|><|start|>assistant<|channel|>final<|message|>"
    return prefix, prefix + body + content


def needs(r: dict) -> str | None:
    return needs_force(r)


def phase1(raw: Path, setting: str) -> list[dict]:
    p = raw / (setting.replace("/", "__") + ".jsonl")
    if not p.exists():
        return []
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") is None and r.get("extra") == SETTINGS[setting]["extra"]:
            out[r["seed"]] = r
    return [out[k] for k in sorted(out)]


def done(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    return {(r["seed"], r["phase1_time"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None}


async def force(client, sem, lock, tok, setting, r, case, path):
    reasoning, content = r["reasoning"], r["content"]
    ids = tok.encode(reasoning, add_special_tokens=False)
    kept = min(len(ids), REASONING_CAP)
    if case == "cap":
        reasoning, content = tok.decode(ids[:REASONING_CAP]), ""
    prefix, full = render(tok, setting, reasoning, content)
    n_prefix = len(tok.encode(prefix, add_special_tokens=False))
    p1 = (r.get("usage") or {}).get("prompt_tokens")
    rec = {"setting": setting, "seed": r["seed"], "phase1_time": r["time"], "case": case,
           "phase1_reasoning_tokens": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "reasoning_tokens_kept": kept, "answer_prefix": content, "prefix_tokens": n_prefix, "phase1_prompt_tokens": p1}
    if DRY_RUN:
        print(f"[{setting} seed={r['seed']}] DRY case={case} kept={kept} prefix tokens {n_prefix} vs phase-1 prompt {p1}\n"
              f"   ...{full[len(prefix) - 80:len(prefix) + 60]!r} ... {full[-160:]!r}", flush=True)
        return
    if p1 is not None and abs(n_prefix - p1) > 2:
        rec["error"] = f"rendered prompt has {n_prefix} tokens, phase 1 had {p1}: template mismatch"
        print(f"[{setting} seed={r['seed']}] {rec['error']}", flush=True)
    else:
        budget = ANSWER_BUDGET - len(tok.encode(content, add_special_tokens=False))
        for attempt in range(60):
            status = None
            async with sem:
                t0 = time.time()
                try:
                    resp = await client.completions.create(model=SETTINGS[setting]["model"], prompt=full, max_tokens=budget,
                                                           seed=r["seed"], extra_body={"add_special_tokens": False})
                    ch = resp.choices[0]
                    rec.update(error=None, content=content + ch.text, generated=ch.text, finish_reason=ch.finish_reason,
                               usage=resp.usage.model_dump() if resp.usage else None, latency_s=round(time.time() - t0, 2))
                    break
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:600]
                    status = getattr(e, "status_code", None)
            if status != 429 and attempt >= 3:
                break
            await asyncio.sleep(120 if status == 429 else 30)
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    async with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{setting} seed={r['seed']}] case={case} kept={kept} prefix={n_prefix}/{p1} -> "
          f"{rec.get('finish_reason')} {len(rec.get('generated', ''))} chars"
          + (f" ERROR {rec['error'][:200]}" if rec.get("error") else ""), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", required=True)
    ap.add_argument("--raw", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="render and check prompts, call nothing")
    a = ap.parse_args()
    global DRY_RUN
    DRY_RUN = a.dry_run
    from transformers import AutoTokenizer

    names = [s for s in SETTINGS if s.endswith("/on") and SETTINGS[s]["backend"] != "openrouter"   # closed APIs: no raw completions
             and any(fnmatch.fnmatch(s, p) for p in a.settings)]
    key = "none" if a.base_url else TOKEN_FILE.read_text().strip()
    client = AsyncOpenAI(base_url=a.base_url or EINFRA, api_key=key, timeout=3 * 3600, max_retries=0)
    sem, lock, tasks = asyncio.Semaphore(a.parallel), asyncio.Lock(), []
    for s in names:
        todo = [(r, needs(r)) for r in phase1(a.raw, s)]
        todo = [(r, c) for r, c in todo if c]
        path = a.raw / (s.replace("/", "__") + ".forced.jsonl")
        have = done(path)
        todo = [(r, c) for r, c in todo if (r["seed"], r["time"]) not in have]
        print(f"{s}: {len(todo)} run(s) to force ({', '.join(f'seed {r['seed']}: {c}' for r, c in todo)})", flush=True)
        if not todo:
            continue
        tok = AutoTokenizer.from_pretrained(repo(s))
        tasks += [force(client, sem, lock, tok, s, r, c, path) for r, c in todo]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
