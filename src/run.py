"""Query the models. Each (setting, seed) is one request; finished ones are skipped on re-runs.

    python -I src/run.py --settings 'kimi-k3.1-pro/*' 'gpt-oss-120b/on'      # e-INFRA
    python -I src/run.py --settings 'gemma-4-e4b/*' --base-url http://localhost:8000/v1 --out DIR   # local vLLM
    python -I src/run.py --settings all --runs 1 --out results/pilot          # pilot: one call each

Local runs stream; e-INFRA runs do not (its proxy cut long streams after 2-3 min in the pilot). Records go to
<out>/<model>__<mode>.jsonl: content, reasoning text, usage (incl. reasoning_tokens), finish_reason, timing.
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
from settings import EINFRA, PROMPT, RUNS, SETTINGS, request_max_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = Path(os.environ.get("EINFRA_TOKEN_FILE", "/storage/plzen1/home/kasnezde/.config/einfra_token"))


def out_path(out: Path, setting: str) -> Path:
    return out / (setting.replace("/", "__") + ".jsonl")


def done_seeds(path: Path, extra: dict) -> set[int]:
    """Seeds already answered with the current request settings (a record made with other `extra` is stale)."""
    if not path.exists():
        return set()
    return {r["seed"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and r.get("extra") == extra}


def _extract(d) -> tuple[str | None, str | None]:
    """(reasoning, content) from a delta or a message; servers name the reasoning field differently."""
    r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
    if r is None and getattr(d, "model_extra", None):
        r = d.model_extra.get("reasoning_content") or d.model_extra.get("reasoning")
    return r, d.content


async def request(client, req, extra, stream):
    """One attempt. Returns (served_model, content, reasoning, usage, finish_reason, first_token_s)."""
    t0 = time.time()
    if not stream:
        resp = await client.chat.completions.create(**req, extra_body=extra or None)
        r, c = _extract(resp.choices[0].message)
        return (resp.model, c or "", r or "", resp.usage.model_dump() if resp.usage else None,
                resp.choices[0].finish_reason, None)
    content, reasoning, usage, finish, served, first = [], [], None, None, None, None
    s = await client.chat.completions.create(**req, extra_body=extra or None, stream=True,
                                             stream_options={"include_usage": True})
    async for chunk in s:
        served = served or chunk.model
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for ch in chunk.choices:
            r, c = _extract(ch.delta)
            if r:
                reasoning.append(r)
            if c:
                content.append(c)
            if (r or c) and first is None:
                first = time.time() - t0
            if ch.finish_reason:
                finish = ch.finish_reason
    return served, "".join(content), "".join(reasoning), usage, finish, first


async def one(client, sem, lock, setting, cfg, seed, path, max_tokens, stream, attempts=4, max_429=60):
    req = dict(model=cfg["model"], messages=[{"role": "user", "content": PROMPT}], max_tokens=max_tokens, seed=seed)
    rec = {"setting": setting, "model": cfg["model"], "seed": seed, "extra": cfg["extra"], "max_tokens": max_tokens,
           "stream": stream}
    errors, n_429 = [], 0
    while True:
        async with sem:
            t0 = time.time()
            try:
                served, content, reasoning, usage, finish, first = await request(client, req, cfg["extra"], stream)
                rec.update(error=None, served_model=served, content=content, reasoning=reasoning, usage=usage,
                           finish_reason=finish, latency_s=round(time.time() - t0, 2),
                           first_token_s=round(first, 2) if first else None)
                break
            except Exception as e:  # noqa: BLE001
                cause = e.__cause__ or e.__context__
                errors.append(f"{type(e).__name__}: {e}"[:600] + (f" <- {type(cause).__name__}: {cause}"[:300] if cause else "")
                              + f" after {time.time() - t0:.0f}s")
                throttled = getattr(e, "status_code", None) == 429
                n_429 += throttled
                if not throttled:
                    print(f"[{setting} seed={seed}] error {len(errors) - n_429}: {errors[-1][:300]}", flush=True)
        # 429 = all 4 slots busy (also with a request we gave up on, which keeps running server-side): just wait
        if throttled and n_429 < max_429:
            await asyncio.sleep(120)
            continue
        if not throttled and len(errors) - n_429 < attempts:
            await asyncio.sleep(60 * 2 ** (len(errors) - n_429))
            continue
        rec.update(error=errors[-1], content="", reasoning="", usage=None, finish_reason=None)
        break
    rec.update(attempt_errors=errors, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    async with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    u = rec.get("usage") or {}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(f"[{setting} seed={seed}] {rec['finish_reason']} {rec.get('latency_s')}s completion={u.get('completion_tokens')} "
          f"reasoning={rt} reasoning_chars={len(rec['reasoning'])} content_chars={len(rec['content'])}"
          + (f" FAILED {rec['error'][:200]}" if rec["error"] else ""), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["all"], help="names or globs from settings.SETTINGS")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--base-url", default=None, help="override (local vLLM); default e-INFRA")
    ap.add_argument("--parallel", type=int, default=3, help="e-INFRA allows 4 parallel requests per user")
    ap.add_argument("--stream", choices=["auto", "yes", "no"], default="auto",
                    help="auto: stream locally; not on e-INFRA, whose proxy dropped long streams after 2-3 min")
    ap.add_argument("--max-model-len", type=int, default=None, help="clamp max_tokens to fit (local)")
    a = ap.parse_args()

    pats = ["*"] if a.settings == ["all"] else a.settings
    names = [s for s in SETTINGS if any(fnmatch.fnmatch(s, p) for p in pats)]
    if not names:
        sys.exit(f"no settings match {pats}")
    a.out.mkdir(parents=True, exist_ok=True)
    key = "none" if a.base_url else TOKEN_FILE.read_text().strip()
    client = AsyncOpenAI(base_url=a.base_url or EINFRA, api_key=key, timeout=4 * 3600, max_retries=0)
    sem, lock, tasks = asyncio.Semaphore(a.parallel), asyncio.Lock(), []
    stream = a.stream == "yes" or (a.stream == "auto" and a.base_url is not None)
    for s in names:
        cfg = SETTINGS[s]
        mt = request_max_tokens(s)
        if a.max_model_len:
            mt = min(mt, a.max_model_len - 1024)
        path = out_path(a.out, s)
        have = done_seeds(path, cfg["extra"])
        tasks += [one(client, sem, lock, s, cfg, seed, path, mt, stream) for seed in range(a.runs) if seed not in have]
    print(f"{len(tasks)} requests over {len(names)} settings -> {a.out}", flush=True)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
