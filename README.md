# Vyjmenovaná slova: do LLMs know the Czech "listed words"?

Ten open-weight LLMs list the Czech *vyjmenovaná slova* as JSON, with reasoning off and on (at most 32k
reasoning tokens, then a forced answer), 5 runs each. Five more models (GPT-6.1 Terra, Claude Opus 5.5, Gemini 4 Argon,
Mistral Large 4, Muse Spark Open) ran once each through OpenRouter. The lists are scored against ÚJČ's overview of the
listed words.

- 💻 **[Viewer](https://lokalniai.github.io/vyjmenovana-slova/)** (standard library build: `python -I src/build_site.py`) and
  **[one-chart summary](https://lokalniai.github.io/vyjmenovana-slova/summary.html)** (`python -I src/build_simple.py`). It has the
  leaderboard, reasoning cost against F1, recall by letter, which settings know which words, the most common
  wrong words, and every run with its answer and reasoning trace.
- 📄 [Report](REPORT.md) · [experiment plan](EXPERIMENTS.md)

| model | F1, reasoning off | F1, reasoning on |
|---|---|---|
| Gemini 4 Argon ¹ (closed) | – | **0.93** |
| GPT-6.1 Terra ¹ (closed) | 0.80 | 0.84 |
| Muse Spark Open ¹ | – | 0.83 |
| Claude Opus 5.5 ¹ (closed) | – | 0.82 |
| DeepSeek V4.1 Flash | 0.40 | 0.76 |
| Mistral Large 4 ¹ | – | 0.36 |
| Kimi K3.1-Pro | 0.10 | 0.30 |
| GPT-OSS-120B | – | 0.19 |
| Qwen3.8-Flash-Next | 0.01 | 0.09 |
| Gemma-4-31B, Gemma-4-E4B, Gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.6-35B-A3B, Tiny Aya Global | ≤ 0.01 | ≤ 0.03 |

¹ Through OpenRouter, without web access, **one run per setting** (so no spread), the same prompt and scoring
(`jobs/openrouter.pbs`, 2026-11-09, $0.31 in total). Only GPT-6.1 Terra can switch reasoning off. Muse Spark Open was
served in FP4. These runs replace the two hand-retrieved answers (Claude Opus 5.5 and ChatGPT web) used before.

## Layout

| path | contents |
|---|---|
| `src/settings.py` | models, modes, prompt, reasoning cap |
| `src/gold.py` | ÚJČ page (not redistributed, see `data/source/README`) → `data/gold.json` (99 items outside brackets + the bracketed derived words) |
| `src/run.py` | queries e-INFRA, a local vLLM server or OpenRouter; resumable; one JSON line per run in `results/raw/` |
| `src/force_answer.py` | forced answers for runs that ran out of reasoning budget (`*.forced.jsonl`) |
| `src/score.py` | parsing, matching, metrics → `results/scores.csv`, `results/items.jsonl`, `results/summary.md` |
| `src/build_site.py`, `src/build_simple.py` | the viewer, the one-chart summary |
| `jobs/*.pbs` | MetaCentrum PBS jobs (e-INFRA, local vLLM, forced answers, OpenRouter) |
| `results/invalid/` | invalid runs (DeepSeek reasoning-off made without the `thinking: false` switch) |
| `results/superseded/` | valid runs no longer scored (Kimi K3, replaced by Kimi K3.1-Pro) |

Gold: [Internetová jazyková příručka, Vyjmenovaná slova](https://prirucka.ujc.cas.cz/?id=100), ÚJČ AV ČR,
fetched 2026-11-02.

Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of Education,
Youth and Sports of the Czech Republic.
