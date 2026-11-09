# Report: do LLMs know the Czech vyjmenovaná slova?

Ten open-weight models were asked to list the *vyjmenovaná slova* as JSON, 5 times each, with reasoning off
and on (reasoning capped at 32k tokens and then a forced answer). The lists are scored against the words outside
brackets in ÚJČ's *Přehled vyjmenovaných slov* (99 items). The runs were made on 2026-11-02, and Kimi K3.1-Pro's on
2026-11-05. Five more models (three of them closed) ran once each through OpenRouter on 2026-11-09; they have
[their own section](#models-via-openrouter-one-run-each), and Gemini 4 Argon beats every open-weight model there.
Plan and setup: [EXPERIMENTS.md](EXPERIMENTS.md). Every number below comes from `results/summary.md`.

## Results (ten open-weight models, 5 runs each)

F1 (mean ± sd over 5 runs). Recall is against all 99 items, precision is strict (bracketed derived words count
as wrong).

| model | runs on | F1 off | F1 on | recall on | precision on (lenient) | median reasoning tokens | forced answers |
|---|---|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash** | e-INFRA | 0.40 ± 0.17 | **0.76 ± 0.06** | 0.73 | 0.81 (0.92) | 32.0k | 0/5 (2 stopped by the native budget) |
| Kimi K3.1-Pro | e-INFRA | 0.10 ± 0.11 | 0.30 ± 0.04 | 0.26 | 0.38 (0.44) | 12.4k | 0/5 |
| GPT-OSS-120B | e-INFRA | – | 0.19 ± 0.04 | 0.14 | 0.38 (0.44) | 32.0k | 3/5 |
| Qwen3.8-Flash-Next | e-INFRA | 0.01 ± 0.02 | 0.09 ± 0.10 | 0.07 | 0.16 (0.19) | 32.0k | 5/5 |
| Gemma-4-31B | e-INFRA | 0.00 | 0.03 ± 0.02 | 0.02 | 0.07 | 10.3k | 0/5 |
| Gemma-4-E4B | local, FP8 | 0.00 | 0.02 ± 0.01 | 0.01 | 0.04 | 1.0k | 0/5 |
| Gemma-4-26B-A4B | local, FP8 | 0.00 | 0.01 ± 0.02 | 0.00 | 0.03 | 20.6k | 0/5 |
| Qwen3.8-27B | e-INFRA, FP8 | 0.01 | 0.00 | 0.00 | 0.00 | 32.0k | 5/5 |
| Qwen3.6-35B-A3B | local, FP8 | 0.00 | 0.00 | 0.00 | 0.00 | 13.3k | 0/5 |
| Tiny Aya Global | local, FP8 | 0.00 | – | – | – | – | – |

**Findings**

1. **Of these ten, only DeepSeek V4.1 Flash knows the list.** With reasoning on it recalls 82 % of the common words and has
   recall of at least 0.6 for every letter (b, l, m, p, s, v, z). Without reasoning it still gets F1 0.40. Every other
   model is at F1 ≤ 0.30, and six of the ten are near 0.
2. **Most models don't know what the task is.** Instead of the seven letters b, l, m, p, s, v and z, they
   invent keys (a, c, d, ch, j, k, …) and list arbitrary Czech words, often ones containing *y/ý* (*byrokrat*,
   *cyklus*) or not even that (*babička*, *bratr*, *pivo*). Only DeepSeek-on covered all seven letters in
   all 5 runs; Kimi K3.1-Pro-on left out *z* in 3 of 5. Lists that loop on one word (*babka, babka, …*) are common in the Qwen3.8 and
   Qwen3.6 models with reasoning off.
3. **Reasoning helps the models that know something** (DeepSeek +0.37, Kimi +0.20, Qwen3.8-Flash-Next +0.08) and
   does nothing for the rest. With the cap, the reasoning often doesn't converge: GPT-OSS, Qwen3.8-27B and
   Qwen3.8-Flash-Next spent the whole 32k on going round in circles ("the official list is… let me recall… I
   recall a typical list…") in 13 of 15 runs. Their scores come from forced answers. Without the forced answer,
   those runs would score 0.
4. **DeepSeek's errors are mostly the school list.** In 5 of 5 runs it lists *byt, dobytek, nábytek, obyvatel,
   příbytek*, and in 4 of 5 *bydlit*. These are head words in school lists but bracketed under *být* in ÚJČ. They
   account for most of the gap between its strict (0.81) and lenient (0.92) precision. What it misses in all 5
   runs are ÚJČ's rarer items: *mys, sumýš, syrý, vykat, cavyky, vyza*, and most proper names (*Zbyněk, Volyně,
   Kamýk, Chropyně, Bosyně, Výtoň*).
5. **The best-known words** across all 90 runs are *být, mýt, pýcha, sýr* (25–28 runs), then *myš, lyže, pytel, syn*
   (21–23 runs). None of these 90 runs listed *Zbyněk, Volyně, Chropyně, Bosyně, cavyky* or *vyza* (Gemini 4 Argon,
   below, lists all of them but *Bosyně*).
6. **Cost.** DeepSeek-on needs a median of 32k reasoning tokens (about 11 minutes per answer on e-INFRA) for its
   0.76. Kimi K3.1-Pro gets 0.30 from a median of 12.4k.

## Models via OpenRouter (one run each)

Three closed models and two open-weight models served by API ran through OpenRouter on 2026-11-09. They used the same
prompt and scoring, reasoning effort high, no web access and one attempt per run ([EXPERIMENTS.md](EXPERIMENTS.md)).
There was **one run per setting**, so there's no spread, and DeepSeek's ± 0.06 over 5 runs is a reminder of how far one
run can move. Only GPT-6.1 Terra accepts reasoning off; the other four reject it ("reasoning is mandatory", Mistral: not
supported). No answer came back with web citations. Total cost: $0.31.

| model | served by | F1 off | F1 on | recall on | rec. names | precision on (lenient) | reasoning tokens |
|---|---|---|---|---|---|---|---|
| **Gemini 4 Argon** | Google AI Studio | – | **0.93** | 0.94 | 12/15 | 0.92 (0.99) | 12.3k |
| GPT-6.1 Terra | OpenAI | 0.80 | 0.84 | 0.78 | 6/15 | 0.92 (1.00) | 5.9k |
| Muse Spark Open | BaseTen, FP4 | – | 0.83 | 0.77 | 6/15 | 0.90 (0.97) | 8.0k |
| Claude Opus 5.5 | Claude Platform on AWS | – | 0.82 | 0.75 | 3/15 | 0.91 (1.00) | 2.5k |
| Mistral Large 4 | Mistral | – | 0.36 | 0.27 | 0/15 | 0.52 (0.64) | 18.8k |

1. **Gemini 4 Argon is the best setting overall** (F1 0.93, against DeepSeek-on's 0.76). It misses only 6 of the 99
   items (*Bosyně, Vyškov, Výtoň, pyj, syrý, vykat*) and lists names and rare items that no open-weight run ever did:
   *Zbyněk, Volyně, Chropyně, Kamýk, cavyky, vyza, mys, sumýš*.
2. **GPT-6.1 Terra, Muse Spark Open and Claude Opus 5.5 sit at 0.82–0.84**, above DeepSeek's mean. Their lists are
   clean: almost every wrong word is a bracketed derived word, mostly the six school-list head words under *být*
   (lenient precision 0.97–1.00). What separates them from Gemini is recall of the rarer items and the names (3–6 of 15).
   Terra already reaches 0.80 with reasoning off.
3. **Mistral Large 4 doesn't know the list** (0.36). It adds a key *r* with *ryba, rychlý, rýč*, puts *kopyto, netopýr,
   pyl* and *Litomyšl* under the wrong letters, and lists no names, despite the longest reasoning of the five (18.8k).
4. **No reasoning cap was needed.** These APIs can't be cut off at 32k (they get 48k in total), but none came close: 2.5k
   (Claude) to 18.8k (Mistral).
5. Claude Opus 5.5's F1 (0.822) happens to equal the hand-retrieved answer used before.

## Caveats

- **The OpenRouter models ran once.** Their numbers are single draws and are not comparable in precision to the
  5-run means above. Their reasoning budget differs too: effort high within 48k output tokens, with no forced cut at
  32k (no run went past 18.8k). Claude's reasoning text is a summary, and GPT-6.1 Terra's is hidden; only the token
  counts are reported. Muse Spark Open was served in FP4 by BaseTen, which OpenRouter picked.

- **Gold = the ÚJČ overview, not a school list.** Six school head words (*byt, bydlit, obyvatel, příbytek,
  nábytek, dobytek*) are bracketed derived words of *být* in ÚJČ, so strict precision counts them as wrong.
  The prompt does say to use the ÚJČ overview and leave out bracketed words. Lenient precision is reported
  too. Recall also needs ÚJČ's rarer items and 15 proper names, which school lists mostly don't have, so
  0.73 is not "27 % of the school list missing".
- **One prompt, in Czech**, with the source named. That may make models try to recall "the exact ÚJČ page",
  which is what GPT-OSS keeps doing in its reasoning traces. A prompt asking for "the list from school" might
  score differently.
- **Reasoning cap.** 32k reasoning tokens + 16k for the answer, the same for every model. There are two
  mechanisms: the native `thinking_token_budget`, and for four e-INFRA models a second call that continues the
  cut reasoning. They are meant to be equivalent (in both, the end of thinking is forced at the cap), but they
  aren't bit-identical. DeepSeek and the local reasoning-on settings were re-run with the native budget on all
  5 seeds. The earlier uncapped runs are kept in `results/raw` but not scored. Their F1 was almost the same
  (DeepSeek 0.76 uncapped). Kimi K3.1-Pro ran with the budget from the start and never reached it (at most 17.6k).
- **Serving.** e-INFRA serves the models in its own precision: FP8 for both Qwen3.8 models, and unknown for
  Kimi, DeepSeek and Gemma-31B. The four local models ran in FP8, two of them (E4B, Aya) with vLLM's
  on-the-fly FP8 of the BF16 weights. Sampling uses each checkpoint's defaults. e-INFRA doesn't reproduce a
  seed across requests (Kimi K3 seed 0 reasoned for 4.8k tokens in one run and 32k in another).
- **Parse failures score 0** (15 of 90 runs). All of them are genuine: lists that loop until they're cut,
  `...` placeholders, a missing `]`, or no list at all (Kimi K3.1-Pro off, seed 4, offers "emotional support" and asks
  for confirmation). None were repaired.
- **One Kimi K3.1-Pro reasoning-off run may have reasoned.** Seed 2 reports 7,370 completion tokens and 0 reasoning
  tokens for an answer of only 721 characters. The missing tokens may be hidden reasoning, but this can't be confirmed
  from the response. It is the best reasoning-off run (F1 0.25) and is scored as it is; without it the mean F1 would be
  0.06 instead of 0.10.
- **5 runs per setting.** The sd is large for the middle models (Kimi off ± 0.11, Flash-Next on ± 0.10).

## Problems hit along the way (fixed)

- **DeepSeek's bare `deepseek-v4.1-flash` alias reasons by default.** Its first reasoning-off runs reasoned
  for 16k tokens and never answered. Fixed with `chat_template_kwargs: {thinking: false}`. The invalid runs are
  in `results/invalid/`.
- **e-INFRA's proxy cut long streams after 2–3 min**, and a cut request keeps its parallel slot (the limit is 4
  per key) until it finishes on the server. All e-INFRA calls are therefore non-streaming. The proxy also
  forces its own `max_tokens` (32k–64k depending on the model), which is why the uniform 32k cap was chosen.
- **Kimi K3.1-Pro is fine on e-INFRA.** Its odd reasoning-off lists (keys a, c, d, f … filled with random
  *y*-words) are real answers. A sanity check (17·23, the capital of Moravia) was answered correctly in every mode,
  the switches behave as they did for K3, and a test budget of 300 tokens was enforced exactly.
- **e-INFRA's `kimi-k3` name also routes to K3.1-Pro.** `/v1/model/info` has two `kimi-k3` entries, one of them pointing
  to `kimi-k3.1-pro`, so the settings use the full name, and every run's `served_model` is `kimi-k3.1-pro`. This
  model replaced Kimi K3 (F1 0.06 off, 0.21 on); those runs are in `results/superseded/kimi-k3/`.

## Reproducing

```bash
python -I src/gold.py                                   # data/gold.json from the ÚJČ page (see data/source/README)
qsub jobs/einfra.pbs                                    # e-INFRA first phase (all e-INFRA settings)
qsub jobs/local.pbs                                     # local models, FP8, vLLM 0.31.0, 1 GPU >= 80 GB
qsub -W depend=afterany:<einfra job> jobs/force.pbs    # forced answers for the "forced" settings
qsub jobs/openrouter.pbs                                # closed/hosted models via OpenRouter (key in ~/.config/openrouter_key)
python -I src/score.py                                  # results/scores.csv, results/items.jsonl, results/summary.md
python -I src/build_site.py                             # docs/index.html
python -I src/build_simple.py                           # docs/summary.html
```

Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of Education,
Youth and Sports of the Czech Republic.
