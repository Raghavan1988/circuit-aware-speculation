"""I26 (CPU): the static-KV draft step must be token-identical to the eager
DynamicCache step for a fresh (no-rollback) draft.

This is the D021 forward-step re-verification for the Tier-1 latency path
(``cas.static_decode``). It runs anywhere: a tiny random Llama on CPU in fp32,
no model download, so it belongs to the pure-logic suite (unlike the GPU
equivalence gate in ``test_equivalence_gpu.py``).

A second test *records* the known Tier-2 obstacle: naive StaticCache rollback
(rewriting from an earlier ``cache_position``) leaks stale K/V. It is marked
xfail so the limitation is asserted in code and will flag loudly if a future
transformers version silently fixes it (xpass) -- the cue to promote a lossless
static-generate path out of Tier-2.
"""
import pytest

torch = pytest.importorskip("torch")

from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from cas.static_decode import StaticDraftStepper, make_static_cache, static_forward


def _tiny_model():
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval().to(torch.float32)


def _dynamic_forward(model, ids, cache, past):
    seq = ids.shape[1]
    pos = torch.arange(past, past + seq).unsqueeze(0)
    attn = torch.ones((1, past + seq), dtype=torch.long)
    out = model(input_ids=ids, past_key_values=cache, position_ids=pos,
                attention_mask=attn, use_cache=True)
    return out.logits, out.past_key_values, past + seq


def _dynamic_draft(model, prompt_ids, n):
    """Greedy n-token draft via the eager DynamicCache path (the reference)."""
    with torch.no_grad():
        ids = torch.tensor([prompt_ids])
        logits, cache, past = _dynamic_forward(model, ids, DynamicCache(), 0)
        cur = logits[0, -1]
        toks = []
        for _ in range(n):
            t = int(cur.argmax())
            toks.append(t)
            logits, cache, past = _dynamic_forward(
                model, torch.tensor([[t]]), cache, past)
            cur = logits[0, -1]
    return toks


def test_static_step_matches_dynamic_fresh_draft():
    """Static single-token decode reproduces the eager draft token-for-token."""
    model = _tiny_model()
    prompt = [int(x) for x in torch.randint(0, 256, (12,))]
    n = 8
    ref = _dynamic_draft(model, prompt, n)

    stepper = StaticDraftStepper(model, max_cache_len=64, compile_mode=None)
    with torch.no_grad():
        cur = stepper.prefill(prompt)
        got = []
        for _ in range(n):
            t = int(cur.argmax())
            got.append(t)
            cur = stepper.step(t)
    assert got == ref, f"static draft {got} != eager draft {ref}"


def test_static_step_logits_close_to_dynamic():
    """The per-position logits (not just argmax) agree to fp32 tolerance, so the
    equivalence is genuine, not an argmax coincidence."""
    model = _tiny_model()
    prompt = [int(x) for x in torch.randint(0, 256, (10,))]
    with torch.no_grad():
        _, dc, past = _dynamic_forward(model, torch.tensor([prompt]), DynamicCache(), 0)
        d_logits, dc, past = _dynamic_forward(model, torch.tensor([[5]]), dc, past)

        sc = make_static_cache(model, 64)
        static_forward(model, torch.tensor([prompt]), sc,
                       torch.arange(0, len(prompt)))
        s_logits = static_forward(model, torch.tensor([[5]]), sc,
                                  torch.arange(len(prompt), len(prompt) + 1))
    diff = (d_logits[0, -1] - s_logits[0, -1]).abs().max().item()
    assert diff < 1e-4, f"static vs dynamic logit gap {diff:.2e} too large"


@pytest.mark.xfail(reason="I26 Tier-2 obstacle: naive StaticCache rollback leaks "
                          "stale K/V (measured 2026-09-06, transformers 5.13). A "
                          "lossless static-generate path needs version-specific "
                          "KV-mask work; recorded here so an xpass flags a fix.",
                   strict=False)
def test_static_rollback_is_lossless():
    """Rewriting from an earlier cache_position should ignore the stale higher
    slots. It does not -- this documents why static rollback is deferred."""
    model = _tiny_model()
    prompt = [int(x) for x in torch.randint(0, 256, (10,))]
    maxlen = 64
    with torch.no_grad():
        # Cache A: prefill, write drafts at positions 10..14, then roll back to
        # 12 and write a different token X.
        A = make_static_cache(model, maxlen)
        static_forward(model, torch.tensor([prompt]), A, torch.arange(0, 10))
        for p, t in zip(range(10, 15), [100, 101, 102, 103, 104]):
            static_forward(model, torch.tensor([[t]]), A, torch.arange(p, p + 1))
        X = 200
        lgA = static_forward(model, torch.tensor([[X]]), A,
                             torch.arange(12, 13))[0, -1]
        # Cache B: the clean reference -- prefill, 100,101, then X at 12.
        B = make_static_cache(model, maxlen)
        static_forward(model, torch.tensor([prompt]), B, torch.arange(0, 10))
        for p, t in zip(range(10, 12), [100, 101]):
            static_forward(model, torch.tensor([[t]]), B, torch.arange(p, p + 1))
        lgB = static_forward(model, torch.tensor([[X]]), B,
                             torch.arange(12, 13))[0, -1]
    assert int(lgA.argmax()) == int(lgB.argmax())
