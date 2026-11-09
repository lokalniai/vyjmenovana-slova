"""Query the models. Each (setting, seed) is one request; finished ones are skipped on re-runs.

    python -I src/run.py --settings 'kimi-k3.1-pro/*' 'gpt-oss-120b/on'      # e-INFRA
    python -I src/run.py --settings 'gemma-4-e4b/*' --base-url http://localhost:8000/v1 --out DIR   # local vLLM
    python -I src/run.py --settings 'gpt-6.1-terra/*'                         # OpenRouter (1 run per setting)
    python -I src/run.py --settings all --runs 1 --out results/pilot          # pilot: one call each

Local and OpenRouter runs stream; e-INFRA runs do not (its proxy cut long streams after 2-3 min in the pilot). Records
go to <out>/<model>__<mode>.jsonl: content, reasoning text, usage (incl. reasoning_tokens), finish_reason, timing, and
for OpenRouter the provider that served the run and the usage cost.

Keys: e-INFRA from $EINFRA_TOKEN_FILE; OpenRouter from $OPENROUTER_API_KEY, else $OPENROUTER_KEY_FILE
(default ~/.config/openrouter_key on plzen1, one line, chmod 600).
"""

import argparse
import asyncio
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import EINFRA, OPENROUTER, PROMPT, SETTINGS, request_max_tokens, runs_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = Path(os.environ.get("EINFRA_TOKEN_FILE", "/storage/plzen1/home/kasnezde/.config/einfra_token"))
OPENROUTER_KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", "/storage/plzen1/home/kasnezde/.config/openrouter_key"))


class WebSearchUsed(Exception):
    """The answer came back with web-search citations."""


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or (OPENROUTER_KEY_FILE.read_text().strip() if OPENROUTER_KEY_FILE.exists() else "")
    if not key:
        sys.exit(f"no OpenRouter key: set OPENROUTER_API_KEY or put the key in {OPENROUTER_KEY_FILE} (chmod 600)")
    r = httpx.get(f"{OPENROUTER}/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code != 200:
        sys.exit(f"OpenRouter rejected the key ({r.status_code}): {r.text[:300]}")
    d = r.json().get("data") or {}
    print(f"OpenRouter key ok: usage ${d.get('usage')}, limit {d.get('limit')}, remaining {d.get('limit_remaining')}", flush=True)
    return key


# OpenRouter rejected the request before any model saw it (key, credit, rate limit): not an attempt, a re-submit fills it
NOT_ATTEMPTED = (401, 402, 429)


def out_path(out: Path, setting: str) -> Path:
    return out / (setting.replace("/", "__") + ".jsonl")


def done_seeds(path: Path, extra: dict, once: bool = False) -> set[int]:
    """Seeds already answered with the current request settings (a record made with other `extra` is stale).
    once: a failed attempt counts as done too (OpenRouter: one attempt per run, never redrawn), unless the request
    never reached a model (`not_attempted`)."""
    if not path.exists():
        return set()
    return {r["seed"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if (r.get("error") is None or (once and not r.get("not_attempted"))) and r.get("extra") == extra}


def _extract(d) -> tuple[str | None, str | None]:
    """(reasoning, content) from a delta or a message; servers name the reasoning field differently."""
    r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
    if r is None and getattr(d, "model_extra", None):
        r = d.model_extra.get("reasoning_content") or d.model_extra.get("reasoning")
    return r, d.content


def _x(obj) -> dict:
    return getattr(obj, "model_extra", None) or {}


def _annotations(d) -> list:
    """Message/delta annotations; OpenRouter attaches web-search results as type "url_citation"."""
    a = getattr(d, "annotations", None) or _x(d).get("annotations")
    return [x.model_dump() if hasattr(x, "model_dump") else x for x in a or []]


def _check_finish(choice):
    if choice.finish_reason == "error":   # OpenRouter reports a provider failure mid-response this way
        raise RuntimeError(f"finish_reason=error: {_x(choice).get('error') or _x(choice)}")


async def request(client, req, extra, stream):
    """One attempt. Returns (served_model, content, reasoning, usage, finish_reason, first_token_s, meta), where meta
    has OpenRouter's `provider` and any `annotations`."""
    t0 = time.time()
    if not stream:
        resp = await client.chat.completions.create(**req, extra_body=extra or None)
        ch = resp.choices[0]
        _check_finish(ch)
        r, c = _extract(ch.message)
        return (resp.model, c or "", r or "", resp.usage.model_dump() if resp.usage else None,
                ch.finish_reason, None, {"provider": _x(resp).get("provider"), "annotations": _annotations(ch.message)})
    content, reasoning, usage, finish, served, first = [], [], None, None, None, None
    provider, annotations = None, []
    s = await client.chat.completions.create(**req, extra_body=extra or None, stream=True,
                                             stream_options={"include_usage": True})
    async for chunk in s:
        served = served or chunk.model
        provider = provider or _x(chunk).get("provider")
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for ch in chunk.choices:
            _check_finish(ch)
            r, c = _extract(ch.delta)
            annotations += _annotations(ch.delta)
            if r:
                reasoning.append(r)
            if c:
                content.append(c)
            if (r or c) and first is None:
                first = time.time() - t0
            if ch.finish_reason:
                finish = ch.finish_reason
    return served, "".join(content), "".join(reasoning), usage, finish, first, {"provider": provider, "annotations": annotations}


async def one(client, sem, lock, setting, cfg, seed, path, max_tokens, stream, attempts=4, max_429=60):
    req = dict(model=cfg["model"], messages=[{"role": "user", "content": PROMPT}], max_tokens=max_tokens, seed=seed)
    rec = {"setting": setting, "model": cfg["model"], "seed": seed, "extra": cfg["extra"], "max_tokens": max_tokens,
           "stream": stream}
    openrouter = cfg["backend"] == "openrouter"
    errors, n_429 = [], 0
    while True:
        async with sem:
            t0 = time.time()
            try:
                served, content, reasoning, usage, finish, first, meta = await request(client, req, cfg["extra"], stream)
                if openrouter:
                    web = [a for a in meta["annotations"] if a.get("type") == "url_citation"]
                    if web:
                        raise WebSearchUsed(f"web search was used: {len(web)} url_citation annotation(s), e.g. "
                                    f"{(web[0].get('url_citation') or {}).get('url')}")
                    rec.update(provider=meta["provider"], annotations=meta["annotations"])
                rec.update(error=None, served_model=served, content=content, reasoning=reasoning, usage=usage,
                           finish_reason=finish, latency_s=round(time.time() - t0, 2),
                           first_token_s=round(first, 2) if first else None)
                break
            except Exception as e:  # noqa: BLE001
                cause = e.__cause__ or e.__context__
                errors.append(f"{type(e).__name__}: {e}"[:600] + (f" <- {type(cause).__name__}: {cause}"[:300] if cause else "")
                              + f" after {time.time() - t0:.0f}s")
                status = getattr(e, "status_code", None)
                throttled = status == 429
                n_429 += throttled
                if not throttled or openrouter:
                    print(f"[{setting} seed={seed}] error {len(errors) - n_429}{' (not retried)' if openrouter else ''}: "
                          f"{errors[-1][:300]}", flush=True)
        # OpenRouter: one attempt per run, whatever the error (also 429); the failure is recorded and never redrawn
        if openrouter:
            rec.update(error=errors[-1], content="", reasoning="", usage=None, finish_reason=None,
                       not_attempted=status in NOT_ATTEMPTED)
            break
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
          + (f" provider={rec.get('provider')} cost={u.get('cost')}" if openrouter else "")
          + (" WARNING: reasoning-off run reported reasoning" if setting.endswith("/off") and (rt or rec["reasoning"]) else "")
          + (f" FAILED {rec['error'][:200]}" if rec["error"] else ""), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["all"], help="names or globs from settings.SETTINGS")
    ap.add_argument("--runs", type=int, default=None, help="seeds 0..runs-1; default settings.runs_for (5, OpenRouter 1)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "raw")
    ap.add_argument("--base-url", default=None, help="override (local vLLM); default e-INFRA")
    ap.add_argument("--parallel", type=int, default=3, help="e-INFRA allows 4 parallel requests per user")
    ap.add_argument("--stream", choices=["auto", "yes", "no"], default="auto",
                    help="auto: stream locally and on OpenRouter; not on e-INFRA, whose proxy dropped long streams after 2-3 min")
    ap.add_argument("--max-model-len", type=int, default=None, help="clamp max_tokens to fit (local)")
    a = ap.parse_args()

    pats = ["*"] if a.settings == ["all"] else a.settings
    names = [s for s in SETTINGS if any(fnmatch.fnmatch(s, p) for p in pats)]
    if not names:
        sys.exit(f"no settings match {pats}")
    for s in names:
        if ":online" in SETTINGS[s]["model"]:
            sys.exit(f"{s}: ':online' slugs search the web; this benchmark runs without internet access")
    a.out.mkdir(parents=True, exist_ok=True)
    clients = {}

    def client_for(backend):
        """--base-url (local vLLM) overrides the backend for every selected setting."""
        b = "override" if a.base_url else backend
        if b not in clients:
            if b == "local":
                sys.exit("local settings need --base-url")
            url, key = {"override": (a.base_url, "none"), "einfra": (EINFRA, None),
                        "openrouter": (OPENROUTER, None)}[b]
            if b == "einfra":
                key = TOKEN_FILE.read_text().strip()
            elif b == "openrouter":
                key = openrouter_key()
            clients[b] = AsyncOpenAI(base_url=url, api_key=key, timeout=4 * 3600, max_retries=0)
        return clients[b]

    sem, lock, tasks = asyncio.Semaphore(a.parallel), asyncio.Lock(), []
    for s in names:
        cfg = SETTINGS[s]
        client = client_for(cfg["backend"])
        stream = a.stream == "yes" or (a.stream == "auto" and (a.base_url is not None or cfg["backend"] == "openrouter"))
        mt = request_max_tokens(s)
        if a.max_model_len:
            mt = min(mt, a.max_model_len - 1024)
        path = out_path(a.out, s)
        have = done_seeds(path, cfg["extra"], once=cfg["backend"] == "openrouter")
        tasks += [one(client, sem, lock, s, cfg, seed, path, mt, stream) for seed in range(a.runs or runs_for(s)) if seed not in have]
    print(f"{len(tasks)} requests over {len(names)} settings -> {a.out}", flush=True)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
