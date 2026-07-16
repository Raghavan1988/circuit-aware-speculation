# How It Works: Methods & Implementation

> **In one line:** We built a fast, exact way to run a big language model, and a careful way to measure it. Every number in our reports comes from a script reading saved data files — never typed by hand.

- **Date:** 2026-07-13
- **Engine:** exact greedy
- **Models tested:** Qwen and Llama pairs
- **Companion doc:** Progress Debrief
- **Live version:** https://claude.ai/code/artifact/d6fb20a7-e44b-47ef-bcf4-e1c83221ae76

---

## First, the big idea (plain English)

Big language models are slow because they write **one word at a time**. Each word needs a full pass through the model.

**Speculative decoding** is a trick to go faster. It uses two models:

- A **small, fast model** (the "draft") guesses the next few words.
- A **big, accurate model** (the "target") checks all those guesses **at once**, in a single pass.

If the small model guessed right, we keep those words for free — we got several words for the price of one big check. If it guessed wrong, we throw out the bad guesses and keep going. The key promise: **the final text is exactly what the big model would have written alone.** We never trade quality for speed.

Two words we use a lot:

- **Accept** — the big model agrees with a guess, so we keep it.
- **Draft length (L)** — how many words the small model guesses before we check. Guess too few and we waste the big model's parallel check. Guess too many and we waste the small model's time on words that get thrown out.

Our research asks: *can we pick a smart draft length on the fly, and does it help to swap in different small models for different topics?*

---

## A. The engine (the machine that does the work)

Our engine is **exact and lossless**. That means the words it produces are **identical** to what the big model would produce on its own. We treat the big model's top pick as the correct answer, always.

We keep two jobs separate on purpose (a design rule we call D021):

1. A **hookable path** — easy to inspect and measure. Used for all the science.
2. A **fast path** — used only when we time speed. It never touches the science path.

### The parts of the engine

| Part | What it does |
|---|---|
| **Target (big model, 7B)** | The judge. Runs one pass per round and checks all the guesses. Its top pick is the "correct" answer. (Qwen2.5-7B-Instruct, fixed version, run in `bf16`.) |
| **Draft (small model, 0.5B)** | The guesser. Proposes up to L words, one at a time. Shares the same vocabulary as the big model. |
| **Verify + commit** | The referee. Keeps the longest run of guesses that match the big model, then adds one guaranteed-correct word. Simple, plain code — tested on a normal CPU. |
| **Memory caches (KV)** | Both models remember what they've already read, so they don't redo work. After each round, the memory is trimmed back to exactly what was accepted. |
| **Signal recorder** | Notes, for every guess, how unsure the small model was (its "entropy" and "margin"). This is the data our smart controller later reads. |
| **Controller** | The decision-maker. Before each round it picks the draft length L from the menu `{0, 1, 2, 3, 4, 6, 8}`. (0 means "skip guessing this round.") |

### One round, step by step

1. **Pick a length.** The controller chooses L.
2. **Guess.** The small model proposes up to L words and records how confident it was.
3. **Check.** The big model does one pass over all the guesses.
4. **Commit.** Keep the matching guesses plus one correct word.
5. **Trim memory.** Roll both caches back to what was accepted.
6. **Stop?** Halt at the end-of-text signal or the length limit.
7. **Record.** Save a fresh confidence signal for the next round.
8. **Log.** Write down everything (the guesses, the big model's picks, timings) and repeat.

### The clever trick that saves us money

Here's the idea that makes the whole project cheap.

Every round, we don't just record *where* the guessing stopped. We record **whether each guess matched**, at *every* position — even past the stopping point. We call this the **match vector**.

Why it matters: with that full record, we can answer "what would have happened with a **different** draft length?" **without re-running any model.** One expensive run (guessing 8 words each time) lets us score *every* possible length setting for free, just by reading the saved file.

So all our later analysis is **replaying saved data**, not paying for new GPU runs.

### How we know it's correct

We run a test that checks the engine's output word-for-word against the big model alone. It must match every time.

- We can force full-precision math so both paths agree to about 1 part in 10 million.
- On the Llama model pair, this check **passed 118 out of 118 times.**
- The fast (timing-only) path must re-pass this same check before any of its numbers count.

> **Figure A — one round.** Solid arrows show data moving inside a round; the dashed arrow is a cheap hint passed to the controller for the next round. Fun fact: the small model's cost depends on its *number of layers*, not its size — which is why a 14×-smaller draft isn't 14× faster on our research setup.

---

## B. The test data (and how we collected it)

Because our engine is exact, whether a word gets accepted depends only on the two models and the text so far — nothing random. So the **variety of our test prompts is the single most important choice** (design rule D022).

Corpus version 2 covers **7 kinds of task** from **9 public sources**:

| Source | Task type | License | Prompts |
|---|---|---|---|
| HumanEval | code | MIT | 164 |
| MBPP (sanitized) | code | CC-BY-4.0 | 200 |
| GSM8K | math | MIT | 200 |
| MT-Bench | chat | Apache-2.0 | 80 |
| OASST1 | chat | Apache-2.0 | 150 |
| WMT14 de-en | translation | shared-task terms | 150 |
| Natural Questions Open | Q&A / retrieval | CC-BY-SA-3.0 | 200 |
| JSONSchemaBench | structured output | MIT | 150 |
| CNN / DailyMail | summarization | Apache-2.0 code; IDs only | 200 |
| **Total** | **7 task types** | | **1,494** |

Three ground rules:

- **How we split the data.** We divide prompts into a "practice" set and a "final exam" set **by whole prompt**, before doing any tuning. Then we freeze the split. We never split in the middle of a single prompt — that would let answers leak from practice into the exam.
- **Where each row came from.** Every item carries its license and a source ID. For sources with copyright concerns, we ship only the row ID plus our own model's output — not the original text.
- **Versioning.** Version 2 lives in its own folder, so all the older version-1 results stay valid and untouched.

> **Figure B — the pipeline.** 9 sources → clean up and tag licenses → remove duplicates → split by prompt (then freeze) → practice/exam sets → sealed data files. Those sealed files are the one and only input to every result. Nothing is recomputed by hand.

---

## C. How we answer each question

All three questions are answered **offline** — by replaying the sealed data. Thanks to the match-vector trick, one saved run per prompt tells us the answer for every draft length, every draft model, and the acceptance atlas. No new GPU runs needed.

### The score we use (and how sure we are)

We measure **efficiency**: useful words produced, divided by the work spent. We price a small-model guess at one-tenth the cost of a big-model check (a realistic serving assumption).

```
efficiency = (total words kept) / (total cost)

    cost of a round with length L = 1 + 0.1 × L
    words kept                    = matching guesses + 1 free correct word
```

To know how confident we are, we use a **prompt-grouped bootstrap**: we resample whole prompts 2,000 times and report the middle 95% range, plus the chance the true gain is zero or negative.

### The three questions

| Question | How we compute it |
|---|---|
| **RQ2 · draft length** | Replay each length policy on the saved match vectors. Compare our smart "entropy-stop" rule (tuned on practice, frozen for the exam) against the best fixed length. |
| **RQ3 · draft routing** | Score each candidate small model against the *same* big-model output, per task type. Use paired, same-size comparisons. The "oracle router" always picks the best model for each task. |
| **RQ1 · joint** | Combine the length controller with the routing oracle. Since routing turned out to add nothing, the joint answer just equals the length controller — RQ1 reduces to RQ2 for this model pair. |

> **Figure C — one round, then the payoff.** The top of the figure is a single round. The bottom band is the reward: because we saved the full match vector, one run answers **all** downstream questions cheaply — draft length, draft routing, and the acceptance atlas.

---

*A few naming notes: we say "representation" or "diagnostic signal" (not "mechanism") until deeper tests pass, and we refer to timeframes by date only. Length menu is `{0, 1, 2, 3, 4, 6, 8}`, where 0 means no guessing. These results come from sealed runs `sweep-2026-07-11T203836` (Qwen) and `sweep-llama-f8-2026-07-13` (Llama). Companion document: Progress Debrief.*
