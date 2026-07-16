# Acceptance-Aware Speculation: Results Summary

> **In one line:** We found a simple, free trick that makes a big AI model run faster — and we proved two fancier ideas don't help. Every number here comes from a script reading saved data, never typed by hand.

- **Prepared 2026-07-13** (replaces the 2026-07-12 version)
- **Big model:** Qwen2.5-7B-Instruct · **Small models:** Qwen2.5 0.5B and 1.5B · **Ran on:** Modal, A100

---

## TL;DR — the answers

- ✅ **Guessing a smart number of words works.** A simple, free rule beats the best fixed number by **+11.2%** and cuts wasted guessing by **62%**.
- ❌ **Specialist small models don't help.** A general-purpose model of the same size ties or beats them — even on the specialist's own topic.
- ➡️ **So the "combine both" question shrinks** down to just the first idea (smart word count).
- 📊 **The supporting findings hold up.** Some kinds of words are much easier to guess than others (we mapped this — the "atlas"); cheap signals predict success just as well as digging into the model's internals; and our research setup makes the small model look slower than it really is, so we hold off on any raw speed claim.
- 🗂️ **We built a better test set.** Version 2: 1,494 prompts, 7 task types, 9 datasets, all license-checked. A repeat of the specialist test on it is running now.

---

## First, the idea in plain English

Big AI models are slow because they write **one word at a time**. Each word takes a full pass through the model.

**Speculative decoding** speeds this up with two models working together:

- A **small, fast model** guesses the next few words.
- A **big, accurate model** checks all the guesses **at once**, in one pass.

Right guesses are kept for free. Wrong guesses are thrown away. The catch: the final text is **exactly** what the big model would have written alone. We never trade quality for speed.

Our study asked three questions:

- Can we pick a smart number of words to guess each time, instead of a fixed number?
- Does swapping in a "specialist" small model for each topic (code, math, etc.) help?
- If we combine both ideas, do we win even more?

---

## 1. The three questions and where they landed

| # | Question | Answer | Why |
|---|---|---|---|
| **RQ2** | Does picking a smart word count beat the best fixed count? | **Yes** | +11.2% better, 62% less waste; holds on fresh data (+11.3 → +11.2) |
| **RQ3** | Do specialist small models help? | **No** | Specialists tie or lose to a same-size generalist — even on their own topic; the perfect router would just pick the generalist everywhere |
| **RQ1** | Does combining both beat the best fixed setup? | **Shrinks to RQ2** | Specialists add nothing (RQ3), and a true speed number is blocked by our research setup |

---

## 2. What we built

- **An exact fast-decoding engine** — proven to produce the *same* words as the big model alone (passed 73 of 73 checks in full precision).
- **A sealed test set** — 644 prompts (code, math, chat, summaries), run under 8 settings → about **302,000 labeled words** saved as locked, checksummed data.
- **A replay tool** — because we saved the big model's choice at every spot, we can test any word-count rule *without re-running a model*.
- **A model-comparison tool** — every candidate small model is scored on the *same* big-model output, so comparisons are fair.
- **A bigger test set (version 2)** — 1,494 prompts, 7 task types, 9 public datasets, each with license info.
- **A strict rule** — every number is produced by a script from locked data, and we write down failures too.

---

## 3. Main results

### 3.1 A smart word count beats the best fixed count *(answers RQ2)*

We replayed different word-count rules on the saved data. We tuned our rule on a "practice" set, froze it, then tested it on a fresh "exam" set. We measure **efficiency** (useful words per unit of work), not raw clock time.

| Rule (fresh exam set) | Words/round | Wasted per kept word | Efficiency |
|---|---|---|---|
| Best fixed count (8) | 4.32 | 1.08 | 2.40 |
| **Our smart rule** | 3.85 | **0.41** | **2.67 (+11.2%)** |
| Perfect hindsight (ceiling) | 4.32 | 0.02 | 3.23 |

**The win:** our rule beats the best fixed count by **+11.2%**. We're confident: the likely range is **+10.3% to +12.1%** across 322 exam prompts, and the chance of no real gain is **below 1 in 2,000**. It also:

- cuts wasted guesses by about **62%** while keeping ~89% of the payoff;
- captures about **one third** of the best-possible improvement;
- keeps working on fresh data without re-tuning;
- wins on **every** topic.

**Why it works — it's the timing, not the learning:**

- Simpler "try and learn" methods (bandits) just settle on the best fixed count and **can't beat it**. That's expected — they only learn one best setting.
- Our rule wins because it reads how **unsure** the small model is *right now*, before each round. Confidence swings a lot: when the small model is sure, its guesses are accepted 97% of the time; when unsure, only 16%.
- Bonus: a simple 4-option menu keeps **98%** of the value, so it's easy to ship.

### 3.2 Specialist small models don't beat a generalist *(answers RQ3)*

We scored five small models across four topics, each on the same big-model output.

| Small model | chat | code | math | summaries |
|---|---|---|---|---|
| general 0.5B | 0.738 | 0.898 | 0.910 | 0.672 |
| Coder 0.5B | 0.675 | 0.899 | 0.877 | 0.603 |
| **general 1.5B** | **0.789** | **0.917** | **0.928** | **0.736** |
| Math 1.5B | 0.662 | 0.862 | 0.919 | 0.554 |
| Coder 1.5B | 0.753 | 0.917 | 0.917 | 0.695 |

**What we found:**

- The code specialists basically **tie** the same-size generalist on code.
- The math specialist is actually **worse** than the generalist — *even on math*.
- The best possible router would just pick the general 1.5B everywhere. So routing among specialists buys **almost nothing**.

**Why:** acceptance is about how well the small model matches the *big model's own style*. A same-family general model matches a general big model better than a narrow expert does. **Bigger helps; specializing doesn't.** A repeat on the bigger version-2 set is running now.

### 3.3 The "atlas": some words are far easier to guess *(headline positive finding)*

How often a guess is accepted ranges from **0.52 to 0.88**, depending on the *kind* of word. We measured this over 302,464 words.

- **Easy:** structural stuff like code symbols, operators, numbers (~0.85–0.88).
- **Hard:** meaning-heavy or surprising words like names and reasoning turns (~0.58–0.67).

This is the *reason* the smart word-count rule works — and it doesn't depend on any timing.

### 3.4 Cheap signals are enough; digging deeper doesn't help

- Free signals the small model already gives off predict success well (score ~0.87).
- Probing the model's inner workings peaks lower (0.803) and adds **at most +0.006** on top.

So for this model pair, the useful information is already in the free signals. (A clean "no extra benefit" result.)

### 3.5 Our setup makes the small model look too slow *(a systems finding)*

Running the small model one word at a time costs **~24 ms per word** — nearly as much as one big-model check (**~30 ms**). That's not because it's doing lots of math; it's overhead from launching the work.

- Standard quick fixes don't solve it; the real fix is deeper engineering we're deliberately saving for later.
- **The upshot:** on this setup, the speedup looks like only ~5%. But the same data says it would be **25% to 46%** on a proper serving setup.

That's why we report **efficiency**, not raw clock time, and say so plainly.

### 3.6 A better test set (version 2)

Our first set (4 topics) didn't stretch across the full range of easy and hard words. So we built version 2: **1,494 prompts across 7 task types** (code, math, chat, translation, Q&A, structured output, summaries), each license-checked. The old set and its results stay untouched.

---

## 4. What it all means

The story fits together — the positives and the negatives back each other up:

1. **The win:** easy and hard words follow a clear pattern (the atlas), and a free rule that reads a cheap signal turns that pattern into a real efficiency gain.
2. **Two dead ends:** digging into the model's internals doesn't beat free signals, and specialist models don't beat a same-size generalist. Both save other researchers wasted effort and point to the real levers: model **size**, matching the **big model's style**, and adapting the **word count**.
3. **Honest about speed:** efficiency gains are real today; raw clock-time gains wait on better engineering, and we've measured exactly how much is waiting (**25–46%**).

---

## 5. Next steps

1. **Report the version-2 repeat** of the specialist test (running now; done within hours).
2. **Decide on the full version-2 run** (~2.3× the cost of the first, roughly **$60–100** of GPU time) — the main experiment left. *First:* lock dataset versions and apply the recorded split-labeling fix.
3. **Finish the controller comparison** with the cheap replay tool (CPU-only).
4. **Package the shareable dataset** — fix the split column, strip copyrighted text down to row IDs, write the dataset card.
5. **Write the paper** around the atlas, the two dead ends, the word-count win, and the slowness finding.
6. **Saved for later:** the deeper engineering that unlocks the real speed claim, plus a smarter way to make specialist small models.

---

## 6. Risks

- **Version 2 might change version-1 conclusions.** *Plan:* same method on both sets; the repeat is already running; we'll report any differences honestly.
- **The raw speed claim stays blocked** until the deeper engineering exists. We won't state clock-time numbers from this setup.
- **Some version-2 loaders aren't version-locked yet** — must be locked before those results count.
- **A known split-labeling bug** must be fixed before any new run. (Current results are unaffected — they recover the split at analysis time.)
- **The Llama repeat is blocked** — we still don't have access to the small Llama model. Findings are currently one model family (Qwen).
- **The field moves fast** — related papers appear monthly; our staged-release plan reduces the risk of being scooped.
- **Budget:** the only real spend left is the version-2 run; everything else is cheap CPU work.

---

## 7. Replication status

Second model family (Llama 3.1 8B with 3.2 1B): the wiring is done and tested, but blocked on access to the small model. The same-family repeat on version 2 is in progress.

---

## Appendix: key numbers

| Item | Value |
|---|---|
| Correctness check (full precision) | 73/73 word-for-word identical |
| Test set v1 / v2 | 644 prompts, 4 topics / 1,494 prompts, 7 topics |
| Labeled words (v1 run) | ~302,000 |
| Atlas cells / acceptance range | 38 / 0.52 to 0.88 |
| Cheap-signal predictor score | ~0.84 (full); 0.870 (subset) |
| Best inner-workings probe | 0.803 (layer 18); adds at most +0.006 |
| Smart rule vs best fixed (fresh data) | +11.2%, range +10.3 to +12.1; waste 0.41 vs 1.08 per word |
| Try-and-learn methods | settle at best fixed (2.40); can't beat the smart rule (2.67) |
| Share of best-possible gain captured | ~33% |
| Specialist vs generalist | ties on code; worse on math (its own topic) |
| Small-model cost on this setup | ~24 ms/word (overhead-bound); big-model check ~30 ms |
| Speedup: this setup / proper serving | ~5% / 25 to 46% |
