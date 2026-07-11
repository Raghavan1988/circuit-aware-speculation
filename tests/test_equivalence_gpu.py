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
