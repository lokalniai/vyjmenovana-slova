"""Models, reasoning modes and the prompt. One *setting* = (model, mode); each runs RUNS times (seeds 0..RUNS-1)."""

RUNS = 5
OPENROUTER_RUNS = 1   # OpenRouter settings run once (seed 0), to save cost
# Reasoning is capped at REASONING_CAP tokens for every model, then the model must answer within ANSWER_BUDGET.
# Two ways to get there (`budget` in SETTINGS), both = "end of thinking forced at the cap, then the answer":
#   "native": vLLM's per-request thinking_token_budget forces the end-of-thinking token inside one generation
#             (works locally and for Kimi/DeepSeek on e-INFRA).
#   "forced": generate with max_tokens = REASONING_CAP; if the run ran out, src/force_answer.py cuts the reasoning to
#             REASONING_CAP tokens, appends the model's end-of-thinking marker and lets it answer in a second, raw
#             completion call (Qwen3.8 ignores the budget on e-INFRA, GPT-OSS rejects it, and Gemma-31B's 32k total cap
#             leaves no room for an answer after 32k of reasoning).
REASONING_CAP = 32_000
ANSWER_BUDGET = 16_000
MAX_TOKENS = {"off": ANSWER_BUDGET, "on": REASONING_CAP + ANSWER_BUDGET}
NATIVE = {"thinking_token_budget": REASONING_CAP}

PROMPT = """Vypiš vyjmenovaná slova podle přehledu vyjmenovaných slov v Internetové jazykové příručce Ústavu pro jazyk český AV ČR (přehled vychází z akademického vydání Pravidel českého pravopisu z roku 1993).

- Uveď jen slova, která přehled uvádí jako vyjmenovaná. Slova příbuzná a odvozená, která přehled uvádí v závorkách za vyjmenovaným slovem, neuváděj.
- Zahrň i vlastní jména (zeměpisná a osobní), která přehled uvádí.
- Slova piš v takovém tvaru, v jakém jsou v přehledu.
- Odpověz pouze JSON objektem. Klíče jsou písmena (malými písmeny), hodnoty jsou seznamy slov pro dané písmeno: {"písmeno": ["slovo", "slovo", ...], ...}. Nic dalšího nepiš."""

EINFRA = "https://llm.ai.e-infra.cz/v1"
OPENROUTER = "https://openrouter.ai/api/v1"

# backend: "einfra", "local" (vLLM in the PBS job) or "openrouter". `extra` goes into the request body as-is.
# Sampling is never set: every backend uses the checkpoint's generation_config defaults.
SETTINGS = {
    # e-INFRA
    "kimi-k3.1-pro/on":       dict(model="kimi-k3.1-pro", backend="einfra", budget="native", extra={"reasoning_effort": "high", **NATIVE}),
    "kimi-k3.1-pro/off":      dict(model="kimi-k3.1-pro", backend="einfra", extra={"chat_template_kwargs": {"thinking": False}}),
    "deepseek-v4.1-flash/on":  dict(model="deepseek-v4.1-flash-thinking", backend="einfra", budget="native", extra={"reasoning_effort": "high", **NATIVE}),
    "deepseek-v4.1-flash/off": dict(model="deepseek-v4.1-flash", backend="einfra", extra={"chat_template_kwargs": {"thinking": False}}),  # the bare alias reasons by default
    "gpt-oss-120b/on":        dict(model="gpt-oss-120b", backend="einfra", budget="forced", extra={"reasoning_effort": "high"}),
    "qwen3.8-flash-next/on":  dict(model="qwen3.8-flash-next", backend="einfra", budget="forced", extra={"chat_template_kwargs": {"enable_thinking": True}}),
    "qwen3.8-flash-next/off": dict(model="qwen3.8-flash-next", backend="einfra", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    "qwen3.8-27b/on":         dict(model="qwen3.8-27b", backend="einfra", budget="forced", extra={"chat_template_kwargs": {"enable_thinking": True}}),
    "qwen3.8-27b/off":        dict(model="qwen3.8-27b", backend="einfra", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    "gemma-4-31b/on":         dict(model="gemma4", backend="einfra", budget="forced", extra={"chat_template_kwargs": {"enable_thinking": True}}),
    "gemma-4-31b/off":        dict(model="gemma4", backend="einfra", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    # local vLLM, FP8
    "qwen3.6-35b-a3b/on":     dict(model="qwen3.6-35b-a3b", backend="local", budget="native", extra={"chat_template_kwargs": {"enable_thinking": True}, **NATIVE}),
    "qwen3.6-35b-a3b/off":    dict(model="qwen3.6-35b-a3b", backend="local", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    "gemma-4-26b-a4b/on":     dict(model="gemma-4-26b-a4b", backend="local", budget="native", extra={"chat_template_kwargs": {"enable_thinking": True}, **NATIVE}),
    "gemma-4-26b-a4b/off":    dict(model="gemma-4-26b-a4b", backend="local", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    "gemma-4-e4b/on":         dict(model="gemma-4-e4b", backend="local", budget="native", extra={"chat_template_kwargs": {"enable_thinking": True}, **NATIVE}),
    "gemma-4-e4b/off":        dict(model="gemma-4-e4b", backend="local", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    "tiny-aya-global/off":    dict(model="tiny-aya-global", backend="local", extra={}),
    # OpenRouter. Without web access: OpenRouter searches the web only for a ":online" slug, the "web" plugin or the
    # web-search server tool, and none of them is ever sent (no plugins, no tools). run.py refuses ":online" slugs and
    # rejects any answer that comes back with url_citation annotations. Reasoning: OpenRouter's unified `reasoning`
    # parameter, effort "high" (on) or "none" (off). The 32k cap can't be forced on these APIs (budget="effort"): an
    # "on" request gets max_tokens = 48k in total and whatever the model spends on reasoning within it.
    # Reasoning off only where the model allows it: Claude Opus 5.5, Gemini 4 Argon and Muse Spark Open answered 400
    # "Reasoning is mandatory", Mistral Large 4 "reasoning_effort none is not supported" (2026-11-09, results/invalid/).
    "gpt-6.1-terra/on":       dict(model="openai/gpt-6.1-terra", backend="openrouter", budget="effort", extra={"reasoning": {"effort": "high"}}),
    "gpt-6.1-terra/off":      dict(model="openai/gpt-6.1-terra", backend="openrouter", extra={"reasoning": {"effort": "none"}}),
    "claude-opus-5.5/on":     dict(model="anthropic/claude-opus-5.5", backend="openrouter", budget="effort", extra={"reasoning": {"effort": "high"}}),
    "gemini-4-argon/on":      dict(model="google/gemini-4-argon", backend="openrouter", budget="effort", extra={"reasoning": {"effort": "high"}}),
    "mistral-large-4/on":     dict(model="mistralai/mistral-large-4-0", backend="openrouter", budget="effort", extra={"reasoning": {"effort": "high"}}),
    "muse-spark-open/on":     dict(model="meta/muse-spark-open", backend="openrouter", budget="effort", extra={"reasoning": {"effort": "high"}}),
}

# Local checkpoints. FP8 everywhere: the author's FP8 build when one exists, else RedHat's, else vLLM's
# on-the-fly FP8 (`--quantization fp8`) of the official BF16 weights.
LOCAL = {
    "qwen3.6-35b-a3b": dict(repo="Qwen/Qwen3.6-35B-A3B-FP8", args=["--reasoning-parser", "qwen3"]),
    "gemma-4-26b-a4b": dict(repo="RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", args=["--reasoning-parser", "gemma4"]),
    "gemma-4-e4b":     dict(repo="google/gemma-4-E4B-it", args=["--quantization", "fp8", "--reasoning-parser", "gemma4"]),
    "tiny-aya-global": dict(repo="CohereLabs/tiny-aya-global", args=["--quantization", "fp8"], max_model_len=8192),
}

# Output caps that e-INFRA's proxy forces via its own extra_body (GET /v1/model/info, 2026-11-02; Kimi 2026-11-05). They
# override the request's max_tokens, so on e-INFRA a reasoning run ends at this length.
EINFRA_CAP = {"kimi-k3.1-pro": 48_000,"deepseek-v4.1-flash": 64_000, "deepseek-v4.1-flash-thinking": 64_000,
              "gpt-oss-120b": 32_000, "qwen3.8-flash-next": 48_000, "qwen3.8-27b": 32_000, "gemma4": 32_000}

# Display metadata for the viewer: name, checkpoint, where it ran, weights.
MODELS = {
    "kimi-k3.1-pro":       ("Kimi K3.1-Pro", "moonshotai/Kimi-K3.1-Pro", "e-INFRA", "as served"),
    "deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "e-INFRA", "as served"),
    "gpt-oss-120b":        ("GPT-OSS-120B", "openai/gpt-oss-120b", "e-INFRA", "MXFP4"),
    "qwen3.8-flash-next":  ("Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8", "e-INFRA", "FP8"),
    "qwen3.8-27b":         ("Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8", "e-INFRA", "FP8"),
    "gemma-4-31b":         ("Gemma-4-31B", "google/gemma-4-31B-it", "e-INFRA", "as served"),
    "qwen3.6-35b-a3b":     ("Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B-FP8", "local vLLM", "FP8"),
    "gemma-4-26b-a4b":     ("Gemma-4-26B-A4B", "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "local vLLM", "FP8"),
    "gemma-4-e4b":         ("Gemma-4-E4B", "google/gemma-4-E4B-it", "local vLLM", "FP8 (on load)"),
    "tiny-aya-global":     ("Tiny Aya Global", "CohereLabs/tiny-aya-global", "local vLLM", "FP8 (on load)"),
    # checkpoint = the OpenRouter slug; the provider that served each run is in its record (`provider`)
    "gpt-6.1-terra":       ("GPT-6.1 Terra", "openai/gpt-6.1-terra", "OpenRouter", "closed"),
    "claude-opus-5.5":     ("Claude Opus 5.5", "anthropic/claude-opus-5.5", "OpenRouter", "closed"),
    "gemini-4-argon":      ("Gemini 4 Argon", "google/gemini-4-argon", "OpenRouter", "closed"),
    "mistral-large-4":     ("Mistral Large 4", "mistralai/mistral-large-4-0", "OpenRouter", "as served (Mistral)"),
    "muse-spark-open":     ("Muse Spark Open", "meta/muse-spark-open", "OpenRouter", "FP4 (BaseTen)"),   # the run of 2026-11-09
}


def runs_for(setting: str) -> int:
    return OPENROUTER_RUNS if SETTINGS[setting]["backend"] == "openrouter" else RUNS


def request_max_tokens(setting: str) -> int:
    """max_tokens sent with the first request: reasoning-on "forced" settings stop at the reasoning cap (the answer
    comes from force_answer.py), "native" ones get the cap plus the answer budget."""
    cfg = SETTINGS[setting]
    if setting.endswith("/off"):
        mt = ANSWER_BUDGET
    else:
        mt = REASONING_CAP if cfg["budget"] == "forced" else REASONING_CAP + ANSWER_BUDGET
    if cfg["backend"] == "local":
        mt = min(mt, LOCAL[cfg["model"]].get("max_model_len", 131072) - 1024)
    return mt


def effective_max_tokens(setting: str) -> int:
    """What a setting could produce in total, after e-INFRA's caps (for the setup table)."""
    cfg = SETTINGS[setting]
    if setting.endswith("/on"):
        return REASONING_CAP + ANSWER_BUDGET
    mt = request_max_tokens(setting)
    if cfg["backend"] == "einfra":
        mt = min(mt, EINFRA_CAP.get(cfg["model"], mt))
    return mt
