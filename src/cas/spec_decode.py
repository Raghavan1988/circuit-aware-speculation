"""Exact greedy speculative-decoding engine (issue I02).

Design (see docs/DECISIONS.md D014 and docs/EXPERIMENT_CONTRACT.md):

  * Greedy, exact-match verification -> output is token-identical to target-only
    greedy decoding (lossless). Proven by test_equivalence_gpu.py.
  * Runtime-selectable action per round: L in {0,1,2,3,4,6,8}; L=0 is `skip`.
  * Persistent KV caches for both models, rolled back after each round with a
    bounded O(1) "catch-up" re-feed (never a full-prefix re-prefill, so the
    drafting-cost metric is unbiased -- D014 rationale).
  * eager attention -> deterministic and hookable for later activation capture.

Cache invariant (S = committed token ids so far):
  at each round start, the target and draft caches each cover some prefix of S;
  the "gap" (S beyond a model's cached length) is re-fed in one forward. After a
  round the gap is length 1 (the bonus token), except after consecutive skips
  where the *draft* gap may grow -- still bounded and cheap.

The pure accept/commit math lives in cas.commit (unit-tested separately). This
module wires it to real forward passes, signals, timing, and trace records.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import torch

from .commit import verify_and_commit
from .config import EngineConfig
from .models import LoadedPair
from .signals import entropy_from_logits, greedy_token, top1_margin_from_logits
from .timing import Stopwatch
from .trace.records import RequestSummary, RoundTrace

# A policy maps (round context) -> action length L. Fixed-length and adaptive
# controllers both implement this signature; the engine is agnostic.
ActionPolicy = Callable[["RoundContext"], int]


@dataclass
class RoundContext:
    """Read-only view a policy may use to choose L (kept minimal for I02)."""

    round_id: int
    generated_so_far: int
    last_round: RoundTrace | None


def fixed_length_policy(length: int) -> ActionPolicy:
    """Constant action L for every round (the fixed baselines of I07)."""
    return lambda _ctx: length


@dataclass
class GenerationResult:
    output_ids: list[int]          # generated tokens (excludes the prompt)
    rounds: list[RoundTrace]
    summary: RequestSummary


def _forward(model, input_ids: torch.Tensor, cache, past_len: int):
    """One forward pass with an explicit past length.

    Passes position_ids and a full-length attention mask so behavior is
    identical across transformers versions regardless of automatic
    cache_position inference.

    Returns (logits[1, seq, vocab], updated_cache, new_past_len).
    """
    seq = input_ids.shape[1]
    device = input_ids.device
    position_ids = torch.arange(past_len, past_len + seq, device=device).unsqueeze(0)
    attn = torch.ones((1, past_len + seq), dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=position_ids,
        attention_mask=attn,
        use_cache=True,
    )
    return out.logits, out.past_key_values, past_len + seq


class SpeculativeDecoder:
    """Runs exact greedy speculative decoding with per-round action control."""

    def __init__(self, pair: LoadedPair, cfg: EngineConfig):
        self.pair = pair
        self.cfg = cfg
        self.device = pair.device

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        policy: ActionPolicy,
        request_id: str,
        run_id: str = "adhoc",
        policy_name: str = "unnamed",
        max_new_tokens: int | None = None,
        record_signals: bool = True,
    ) -> GenerationResult:
        max_new = max_new_tokens or self.cfg.max_new_tokens
        eos_id = self.pair.tokenizer.eos_token_id
        sw = Stopwatch(self.device)

        from transformers import DynamicCache

        target_cache = DynamicCache()
        draft_cache = DynamicCache()
        t_len = 0  # tokens the target cache covers
        d_len = 0  # tokens the draft cache covers

        S: list[int] = list(prompt_ids)  # committed sequence (prompt + generated)
        prompt_n = len(prompt_ids)

        # ---- prefill: process the prompt on both models -------------------
        ids = torch.tensor([prompt_ids], device=self.device)
        with sw.measure("prefill"):
            t_logits, target_cache, t_len = _forward(self.pair.target, ids, target_cache, 0)
            _, draft_cache, d_len = _forward(self.pair.draft, ids, draft_cache, 0)
        # first committed token = target greedy argmax after the prompt
        first_tok = int(greedy_token(t_logits[0, -1]))
        S.append(first_tok)

        rounds: list[RoundTrace] = []
        generated: list[int] = [first_tok]
        ttft_ns = sw.acc.components_ns.get("prefill", 0)
        termination = "max_new_tokens"
        last_round: RoundTrace | None = None

        if first_tok == eos_id:
            termination = "eos"

        round_id = 0
        while len(generated) < max_new and termination != "eos":
            ctx = RoundContext(round_id, len(generated), last_round)
            L = int(policy(ctx))
            if L not in self.cfg.action_lengths:
                raise ValueError(f"policy returned invalid action {L}")

            cache_before = t_len
            proposals: list[int] = []
            entropies: list[float] = []
            margins: list[float] = []

            # ---- draft phase: bring draft to end of S, then propose L ------
            # `skip` (L == 0) runs NO draft forward at all -- the draft cache is
            # left behind and re-synced (via the bounded gap) the next time we
            # draft, so skip is charged zero drafting cost (D014).
            with sw.measure("draft"):
                if L > 0:
                    gap = S[d_len:]  # always non-empty: draft never caches the anchor
                    d_ids = torch.tensor([gap], device=self.device)
                    d_logits, draft_cache, d_len = _forward(
                        self.pair.draft, d_ids, draft_cache, d_len
                    )
                    cur_logits = d_logits[0, -1]
                    for _ in range(L):
                        if record_signals:
                            entropies.append(float(entropy_from_logits(cur_logits)))
                            margins.append(float(top1_margin_from_logits(cur_logits)))
                        tok = int(greedy_token(cur_logits))
                        proposals.append(tok)
                        step_ids = torch.tensor([[tok]], device=self.device)
                        d_logits, draft_cache, d_len = _forward(
                            self.pair.draft, step_ids, draft_cache, d_len
                        )
                        cur_logits = d_logits[0, -1]

            # ---- verify phase: target over (gap_t + proposals) in one pass --
            with sw.measure("verify"):
                gap_t = S[t_len:]
                verify_ids = torch.tensor([gap_t + proposals], device=self.device)
                v_logits, target_cache, t_len = _forward(
                    self.pair.target, verify_ids, target_cache, t_len
                )
                # last L+1 positions predict t_1..t_{L+1}
                tail = v_logits[0, -(L + 1):, :]
                target_argmax = [int(x) for x in greedy_token(tail)]

            # ---- commit (pure) + roll caches back --------------------------
            with sw.measure("controller"):
                res = verify_and_commit(proposals, target_argmax)

            keep = len(S) - 1 + res.accepted  # committed positions that stay cached
            # S currently ends at the previous bonus/first token (index len(S)-1).
            # After verify, target cache covers S[:len(S)] + proposals; keep the
            # prefix through the accepted drafts, then append the emitted tokens.
            target_cache.crop(keep + 1)
            t_len = keep + 1
            draft_cache.crop(min(d_len, keep + 1))
            d_len = min(d_len, keep + 1)

            for tok in res.emitted_ids:
                S.append(tok)
                generated.append(tok)
                if len(generated) >= max_new:
                    break
            if eos_id is not None and eos_id in res.emitted_ids:
                termination = "eos"

            rt = RoundTrace(
                request_id=request_id,
                round_id=round_id,
                start_output_pos=len(generated) - len(res.emitted_ids),
                requested_action=L,
                realized_draft_len=L,
                proposed_token_ids=tuple(proposals),
                accepted_prefix_len=res.accepted,
                first_rejection_pos=res.first_rejection,
                emitted_token_ids=res.emitted_ids,
                draft_entropy=tuple(entropies),
                draft_top1_margin=tuple(margins),
                latency_ns=dict(sw.acc.components_ns),
                cache_len_before=cache_before,
                cache_len_after=t_len,
            )
            rounds.append(rt)
            last_round = rt
            round_id += 1

        generated = generated[:max_new]
        out_hash = hashlib.sha256(
            ",".join(map(str, generated)).encode()
        ).hexdigest()
        total_drafted = sum(r.realized_draft_len for r in rounds)
        total_accepted = sum(r.accepted_prefix_len for r in rounds)
        summary = RequestSummary(
            run_id=run_id,
            request_id=request_id,
            dataset="",
            domain="",
            split="",
            prompt_hash="",
            policy_name=policy_name,
            prompt_tokens=prompt_n,
            output_tokens=len(generated),
            termination_reason=termination,
            output_token_hash=out_hash,
            total_drafted=total_drafted,
            total_accepted=total_accepted,
            total_rejected=total_drafted - total_accepted,
            n_rounds=len(rounds),
            prefill_ns=sw.acc.components_ns.get("prefill", 0),
            decode_ns=sw.acc.total_ns() - sw.acc.components_ns.get("prefill", 0),
            ttft_ns=ttft_ns,
            end_to_end_ns=sw.acc.total_ns(),
            peak_mem_bytes=(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
        )
        return GenerationResult(output_ids=generated, rounds=rounds, summary=summary)

    @torch.no_grad()
    def greedy_reference(
        self, prompt_ids: list[int], max_new_tokens: int | None = None
    ) -> list[int]:
        """Target-only greedy decoding: the ground truth for the equivalence
        check (I03). Deliberately simple (re-uses HF generate semantics via a
        manual argmax loop) so it shares no code path with the speculative loop.
        """
        max_new = max_new_tokens or self.cfg.max_new_tokens
        eos_id = self.pair.tokenizer.eos_token_id
        from transformers import DynamicCache

        cache = DynamicCache()
        ids = torch.tensor([prompt_ids], device=self.device)
        logits, cache, past = _forward(self.pair.target, ids, cache, 0)
        out: list[int] = []
        for _ in range(max_new):
            tok = int(greedy_token(logits[0, -1]))
            out.append(tok)
            if tok == eos_id:
                break
            step = torch.tensor([[tok]], device=self.device)
            logits, cache, past = _forward(self.pair.target, step, cache, past)
        return out
