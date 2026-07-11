"""I03 (GPU): output-equivalence and cache-rollback tests requiring real models.

Skipped automatically when CUDA or the models are unavailable, so the pure-logic
suite still runs anywhere. On Modal these are the load-bearing correctness gate:
speculative decoding must be token-identical to target-only greedy (D014:
bit-exact with logged floating-point-tie exceptions).

Run: modal run modal_app.py::run_tests
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA; GPU equivalence tests run on Modal", allow_module_level=True)

from cas.config import ACTION_LENGTHS, EngineConfig
from cas.models import load_pair
from cas.spec_decode import SpeculativeDecoder, fixed_length_policy

PROMPTS = [
    "def quicksort(arr):",
    "The capital of France is",
    "Solve for x: 2x + 3 = 11.",
    "Write a haiku about autumn.",
]


@pytest.fixture(scope="module")
def decoder():
    cfg = EngineConfig(max_new_tokens=96)
    pair = load_pair(cfg)
    return SpeculativeDecoder(pair, cfg), pair


@pytest.mark.parametrize("length", [L for L in ACTION_LENGTHS if L > 0])
def test_bit_identical_to_greedy(decoder, length):
    """Every fixed draft length must reproduce target-only greedy exactly.

    Divergences are allowed only as rare floating-point argmax ties; this test
    asserts full identity and any failure is investigated per D014 (expected
    rate <0.1%, characterized separately by test_fp_divergence_rate).
    """
    dec, pair = decoder
    for prompt in PROMPTS:
        ids = pair.tokenizer(prompt)["input_ids"]
        ref = dec.greedy_reference(ids)
        res = dec.generate(ids, fixed_length_policy(length), request_id="eq",
                           policy_name=f"fixed_{length}")
        assert res.output_ids == ref, (
            f"L={length} diverged on {prompt!r} at first mismatch "
            f"{_first_diff(res.output_ids, ref)}"
        )


def test_skip_action_equivalent(decoder):
    """The skip action (L=0) is pure autoregressive; must equal greedy."""
    dec, pair = decoder
    ids = pair.tokenizer(PROMPTS[0])["input_ids"]
    ref = dec.greedy_reference(ids)
    res = dec.generate(ids, fixed_length_policy(0), request_id="skip",
                       policy_name="skip_only")
    assert res.output_ids == ref


@pytest.mark.parametrize("threshold", [0.0, 1e9])
def test_stop_rule_equivalent(decoder, threshold):
    """The D017 stop-rule seam must never change output tokens: an
    always-stop rule (threshold 0 stops at every positive-entropy token,
    exercising realized-length-0 rounds) and a never-stop rule must both be
    token-identical to greedy."""
    from cas.policies import EntropyStopRule

    dec, pair = decoder
    ids = pair.tokenizer(PROMPTS[0])["input_ids"]
    ref = dec.greedy_reference(ids)
    res = dec.generate(ids, fixed_length_policy(4), request_id="stoprule",
                       policy_name=f"fixed_4+stop_{threshold}",
                       stop_rule=EntropyStopRule(threshold))
    assert res.output_ids == ref
    if threshold == 1e9:  # never stops: realized == requested every round
        assert all(r.realized_draft_len == r.requested_action for r in res.rounds)


def test_eos_mid_round_equivalent(decoder):
    """Regression (review 2026-07-11, critical): when eos lands inside the
    accepted draft prefix, the engine must stop committing at eos exactly like
    greedy — a chat-templated prompt reaches eos well within max_new."""
    dec, pair = decoder
    msgs = [{"role": "user", "content": "Reply with exactly the word: hi"}]
    ids = pair.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    ref = dec.greedy_reference(ids)
    for L in (4, 8):
        res = dec.generate(ids, fixed_length_policy(L), request_id="eos",
                           policy_name=f"fixed_{L}")
        assert res.output_ids == ref, f"eos-path divergence at L={L}"
    eos = pair.tokenizer.eos_token_id
    if eos in ref:  # expected for a chat prompt; guarded for robustness
        assert res.output_ids[-1] == eos
        assert res.summary.termination_reason == "eos"


def test_d018_fields_recorded(decoder):
    """Rounds must carry target argmax ids, per-position matches consistent
    with the accepted prefix, and the frontier entropy/margin byproducts."""
    dec, pair = decoder
    ids = pair.tokenizer(PROMPTS[1])["input_ids"]
    res = dec.generate(ids, fixed_length_policy(4), request_id="d018",
                       policy_name="fixed_4")
    assert res.rounds
    for rt in res.rounds:
        assert len(rt.target_argmax_ids) == rt.realized_draft_len + 1
        match = rt.per_position_match()
        k = 0
        for m in match:
            if not m:
                break
            k += 1
        assert k == rt.accepted_prefix_len
        assert rt.target_entropy_frontier is not None
        assert rt.target_entropy_frontier >= 0.0
        assert rt.target_margin_frontier is not None
        assert 0.0 <= rt.target_margin_frontier <= 1.0
        # per-round latency deltas, not cumulative: no prefill key in rounds
        assert "prefill" not in rt.latency_ns


def test_fp_divergence_rate(decoder):
    """Characterize (not gate) the floating-point tie divergence rate across
    prompts and lengths, so the 'lossless' claim can be stated honestly."""
    dec, pair = decoder
    total = mism = 0
    for length in (2, 4, 8):
        for prompt in PROMPTS:
            ids = pair.tokenizer(prompt)["input_ids"]
            ref = dec.greedy_reference(ids)
            res = dec.generate(ids, fixed_length_policy(length),
                               request_id="fp", policy_name=f"fixed_{length}")
            total += 1
            if res.output_ids != ref:
                mism += 1
    rate = mism / total if total else 0.0
    print(f"fp-tie divergence rate: {mism}/{total} = {rate:.4f}")
    assert rate < 0.05  # generous ceiling; expected <0.001


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    return len(a) if len(a) != len(b) else None
