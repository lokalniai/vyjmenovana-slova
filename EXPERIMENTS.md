# Vyjmenovaná slova: do LLMs know the Czech "listed words"? (experiment plan, approved 2026-11-02)

Ask open-weight LLMs to list the Czech *vyjmenovaná slova* as JSON, with and without reasoning, 5 runs per
setting, and score the lists against the ÚJČ overview.

## Models and settings (`src/settings.py`)

| model | checkpoint | runs on | modes | reasoning cap mechanism |
|---|---|---|---|---|
| Kimi K3.1-Pro | `moonshotai/Kimi-K3.1-Pro` | e-INFRA `kimi-k3.1-pro` | off (`thinking: false`), on (`reasoning_effort=high`) | native |
| DeepSeek V4.1 Flash | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | e-INFRA `deepseek-v4.1-flash` (+ `thinking: false`) / `-thinking` | off, on | native |
| GPT-OSS-120B | `openai/gpt-oss-120b` | e-INFRA | on only (`high`); it cannot switch reasoning off | forced |
| Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next-FP8` | e-INFRA | off, on (`enable_thinking`) | forced |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B-FP8` | e-INFRA | off, on | forced |
| Gemma-4-31B | `google/gemma-4-31B-it` | e-INFRA `gemma4` | off, on (`enable_thinking`) | forced |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B-FP8` | local vLLM | off, on | native |
| Gemma-4-26B-A4B | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | local vLLM | off, on | native |
| Gemma-4-E4B | `google/gemma-4-E4B-it` + vLLM `--quantization fp8` | local vLLM | off, on | native |
| Tiny Aya Global | `CohereLabs/tiny-aya-global` + vLLM `--quantization fp8` | local vLLM | off only (no reasoning mode) | – (8k context) |

- e-INFRA is used wherever it serves the model. All local models run in FP8: the author's FP8 checkpoint
  where one exists, RedHat's for Gemma-26B, and vLLM's on-the-fly FP8 for E4B and Aya, which have no
  reputable FP8 release.
- Models that must reason (GPT-OSS) are left out of the reasoning-off comparison.
- Kimi K3.1-Pro replaced Kimi K3 on 2026-11-05, with the same request settings; K3's runs are in
  `results/superseded/kimi-k3/`. Always request `kimi-k3.1-pro` by its full name: e-INFRA's `/v1/model/info` also has a
  `kimi-k3` entry that routes to it, and the `kimi` alias does too.
- Sampling: each checkpoint's default generation config, seeds 0–4. No `response_format`. Reasoning off:
  `max_tokens` 16k.

### Reasoning cap: 32k reasoning tokens + forced answer, for every model

Reasoning is capped at **32,000 tokens**. At the cap the end of thinking is forced, and the model must answer
within **16,000 tokens** (`settings.REASONING_CAP`, `ANSWER_BUDGET`). There are two ways to do this:

- **native**: vLLM's per-request `thinking_token_budget` forces the end-of-thinking token inside the same
  generation (`max_tokens` = 48k). It works locally, where `--reasoning-parser` enables it, and for Kimi and
  DeepSeek on e-INFRA.
- **forced**: the first request stops at 32k (`max_tokens` = 32k). If it ran out,
  `src/force_answer.py` cuts the reasoning to 32k tokens (with the checkpoint's tokenizer), appends the
  model's end-of-thinking marker (`</think>`, Gemma's `<channel|>`, Harmony's `final` channel) and calls raw
  `/v1/completions` on the prompt, which is rendered with the checkpoint's own chat template and checked
  against the first call's `prompt_tokens`. If the reasoning ended within the cap but the answer was cut, the
  answer is continued instead. This is used where e-INFRA can't do it natively: it silently ignores the
  budget for both Qwen3.8 models, rejects it for GPT-OSS, and Gemma-31B's 32k total cap leaves no room for an
  answer.
- Why: e-INFRA's proxy forces its own `max_tokens` (32k–64k depending on the model, `settings.EINFRA_CAP`).
  In the pilot, GPT-OSS, Qwen3.8-Flash-Next and Qwen3.6 (locally, at 128k) reasoned until they hit the cap and
  never answered.
- Equivalence: with a fixed seed the reasoning does not depend on `max_tokens`, so cutting a longer run back
  to 32k is the same as running it with a 32k budget. Kimi, DeepSeek and Qwen3.6 reasoning-on were re-run
  with the native budget on all 5 seeds. Re-running only the runs that went over 32k would have biased the
  sample.

## Extra results for reference (`results/external/`)

Two answers retrieved by hand, from Claude Opus 5.5 (high reasoning) and from the unidentified model behind the
ChatGPT web interface, with "Nehledej nic na internetu." appended to the prompt. One run each and no usage data. They are
scored the same way (`score.load_external`) but shown separately and are not part of the comparison.

## Prompt (`settings.PROMPT`)

The prompt names the source (the ÚJČ Internetová jazyková příručka overview, based on PČP 1993). It asks for the
listed words only, without the related and derived words the overview gives in brackets, and for the proper
names the overview includes. It asks for a JSON object with lowercase letters as keys. It names neither the
letters nor any word.

## Gold (`src/gold.py` → `data/gold.json`)

The words **outside brackets** in each letter's row of https://prirucka.ujc.cas.cz/?id=100, fetched on 2026-11-02.
The result is in `data/gold.json`. The page itself isn't redistributed; see `data/source/README`.

- 99 items: 84 common words and 15 proper names (the part after the semicolon). "myslit i myslet" is one item
  with two accepted spellings. "předpona vy-/vý-" is left out and not scored either way.
- The bracketed words (about 275) are kept as `derived`, so a wrong word can be told apart from a derived
  one. Agreed trade-off: school-list head words that ÚJČ puts in brackets under *být* (**byt, bydlit, obyvatel,
  příbytek, nábytek, dobytek**) are not gold.

## Scoring (`src/score.py`)

- The answer is the first JSON object in it (code fences are stripped). If none can be parsed, the run
  scores 0. Words are normalized to NFC and lowercase, trailing *se/si* and parentheses are dropped, and
  items are split on `,` `;` `/` and " i ". Each word is matched against the gold items of **the key it is
  listed under**.
- Recall = gold items listed / 99, also split into common words, names and letters.
- Precision (strict) = TP / all unique listed words. **Lenient** precision leaves bracketed (derived) words
  out. F1 is computed from strict precision and recall.
- Each false positive gets a label: `derived`, `misplaced` (a gold word under the wrong letter), `near` (one
  edit or only diacritics away from a gold word), or `other`.
- Logged per run: completion and reasoning tokens (`usage.completion_tokens_details.reasoning_tokens`; for a
  forced answer, the tokens kept), whether the answer was forced, finish reason and latency.

## Running

- e-INFRA: `qsub jobs/einfra.pbs`. One CPU, no streaming (in the pilot the proxy cut long streams after
  2–3 min, and a cut request keeps its slot until it finishes server-side), 3 parallel requests (the limit
  is 4 per key), and `429` responses are retried every 2 min. Requests that finished are skipped on
  resubmit.
- Local: `qsub jobs/local.pbs`. One GPU with ≥ 80 GB and `compute_89` (native FP8), vLLM 0.31.0, weights
  downloaded to scratch, 5 parallel streams per model.
- Forced answers: `qsub jobs/force.pbs` for e-INFRA (after the first phase); the local job runs them right after each
  model's first phase, as a safety net.
- Score: `python -I src/score.py` → `results/scores.csv`, `results/summary.md`.
- Viewer: `python -I src/build_site.py` → `docs/index.html`, in the lokalni.ai companion-page style. One-chart summary: `python -I src/build_simple.py` → `docs/summary.html`.
