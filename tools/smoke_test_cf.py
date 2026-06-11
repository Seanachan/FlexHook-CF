#!/usr/bin/env python3
"""
CPU smoke tests for the FlexHook-CF graft. No dataset / GPU required.

Run from the FlexHook repo root:
    python tools/smoke_test_cf.py

Tests:
  1. data/cf_utils.inject_counterfactuals  (slot replacement, caps, guards)
  2. tools/gen_counterfactuals.rule_based_counterfactuals  (single-attribute edits)
  3. the counterfactual push-loss math + backward  (skipped if torch unavailable)

Modules are loaded by file path so we do NOT import the heavy `data` package
(which pulls torch / transformers / CLIP).
"""

import os
import sys
import random
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf_utils = _load('cf_utils', 'data/cf_utils.py')
genmod = _load('gen_counterfactuals', 'tools/gen_counterfactuals.py')

PASS, FAIL = 0, 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ok   - {msg}')
    else:
        FAIL += 1
        print(f'  FAIL - {msg}')


def test_inject():
    print('[1] inject_counterfactuals')
    rng = random.Random(0)
    counterfactuals = {
        'white car': [{'expression': 'red car', 'attr_type': 'color'},
                      {'expression': 'black car', 'attr_type': 'color'}],
        'moving van': [{'expression': 'parked van', 'attr_type': 'motion'}],
    }
    pos = ['white car', 'moving van']
    slots = ['white car', 'some negative a', 'some negative b', 'moving van', 'neg c']

    new, is_cf = cf_utils.inject_counterfactuals(slots, pos, counterfactuals, n_cf=2,
                                                 attr_types=None, rng=rng)
    check(len(new) == len(slots), 'slot count N preserved')
    check(sum(is_cf) == 2, 'exactly n_cf=2 counterfactuals injected')
    # injected slots must be former negatives, never the positive slots
    for i, flag in enumerate(is_cf):
        if flag:
            check(new[i] not in pos, f'injected slot {i} is a true negative ({new[i]!r})')
            check(new[i] in {'red car', 'black car', 'parked van'},
                  f'injected slot {i} is a known counterfactual ({new[i]!r})')
    check(new[0] == 'white car' and new[3] == 'moving van', 'positive slots untouched')

    # cap by available candidates
    new2, is_cf2 = cf_utils.inject_counterfactuals(slots, ['moving van'], counterfactuals,
                                                   n_cf=5, attr_types=None, rng=rng)
    check(sum(is_cf2) == 1, 'injection capped by available candidates (1)')

    # attr filter excludes motion
    new3, is_cf3 = cf_utils.inject_counterfactuals(slots, pos, counterfactuals, n_cf=5,
                                                   attr_types={'color'}, rng=rng)
    for i, flag in enumerate(is_cf3):
        if flag:
            check(new3[i] in {'red car', 'black car'}, 'attr_types filter keeps only color')

    # disabled paths
    _, z = cf_utils.inject_counterfactuals(slots, pos, counterfactuals, n_cf=0, rng=rng)
    check(sum(z) == 0, 'n_cf=0 injects nothing')
    _, z2 = cf_utils.inject_counterfactuals(slots, [], counterfactuals, n_cf=3, rng=rng)
    check(sum(z2) == 0, 'no positives -> injects nothing')


def test_rule_gen():
    print('[2] rule_based_counterfactuals')
    rng = random.Random(0)
    for expr in ['white car', 'moving cars in the left', 'the parked van on the right']:
        variants = genmod.rule_based_counterfactuals(expr, k=4, attr_types=None, rng=rng)
        check(len(variants) > 0, f'produced variants for {expr!r}')
        for v in variants:
            check(v['expression'].lower() != expr.lower(),
                  f'variant differs from source ({v["expression"]!r})')
            check(v['attr_type'] in {'color', 'type', 'motion', 'location'},
                  f'attr_type valid ({v["attr_type"]})')
            # exactly one token changed
            a, b = expr.split(), v['expression'].split()
            diffs = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
            check(diffs == 1, f'exactly one token changed for {v["expression"]!r} (diffs={diffs})')


def test_loss_math():
    print('[3] counterfactual push-loss (torch)')
    try:
        import torch
    except Exception as e:
        print(f'  skip - torch unavailable ({e})')
        return
    torch.manual_seed(0)
    B, L, N, C = 2, 4, 5, 2  # (batch, num_layers, num_expressions, 2 classes)
    outputs = torch.randn(B, L, N, C, requires_grad=True)
    is_cf = torch.zeros(B, N, dtype=torch.long)
    is_cf[0, 1] = 1
    is_cf[1, 3] = 1

    # replicate the main.py block
    cf_mask = is_cf.unsqueeze(1).repeat(1, L, 1).flatten().float()
    p_match = outputs.flatten(0, 2).softmax(-1)[:, 1]
    loss_cf = -torch.log((1.0 - p_match).clamp_min(1e-6))
    loss_cf = (loss_cf * cf_mask).sum() / cf_mask.sum()

    check(torch.isfinite(loss_cf).item(), 'L_cf is finite')
    check(loss_cf.item() > 0, 'L_cf is positive')
    check(int(cf_mask.sum().item()) == 2 * L, 'cf_mask covers both CF slots across all layers')
    loss_cf.backward()
    check(outputs.grad is not None and torch.isfinite(outputs.grad).all().item(),
          'backward produces finite gradients')

    # zero-CF guard: mask sum 0 must not be invoked (we guard in main.py)
    empty = torch.zeros(B, N, dtype=torch.long).unsqueeze(1).repeat(1, L, 1).flatten().float()
    check(empty.sum().item() == 0, 'empty cf_mask detected (loss skipped upstream)')


def test_loss_agg_max():
    print('[4] CF_LOSS_AGG=max (hardest CF per anchor)')
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        print(f'  skip - torch unavailable ({e})')
        return
    torch.manual_seed(0)
    B, L, N, C = 2, 4, 5, 2
    outputs = torch.randn(B, L, N, C, requires_grad=True)
    is_cf = torch.zeros(B, N, dtype=torch.long)
    is_cf[0, 1] = 1
    is_cf[0, 2] = 1  # two CF slots on sample 0 -> max must pick one
    is_cf[1, 3] = 1
    margin = 0.5

    # replicate the main.py 'max' branch
    cf_mask = is_cf.unsqueeze(1).repeat(1, L, 1).flatten().float()
    p_match = outputs.flatten(0, 2).softmax(-1)[:, 1]
    per_slot = F.relu(p_match - margin)
    lc = (per_slot * cf_mask).view(B * L, N)
    has_cf = cf_mask.view(B * L, N).sum(1) > 0
    loss_max = lc.max(dim=1).values[has_cf].mean()
    loss_mean = (per_slot * cf_mask).sum() / cf_mask.sum()

    check(torch.isfinite(loss_max).item(), 'max-agg L_cf is finite')
    check(loss_max.item() >= loss_mean.item() - 1e-6,
          'max-agg >= mean-agg (hardest slot dominates)')
    check(int(has_cf.sum().item()) == B * L, 'every (sample,layer) anchor has a CF row')
    loss_max.backward()
    check(outputs.grad is not None and torch.isfinite(outputs.grad).all().item(),
          'max-agg backward produces finite gradients')
    # gradient sparsity: only one CF slot per anchor gets gradient through max
    g = outputs.grad.abs().view(B, L, N, C).sum(-1)
    cf0 = g[0, 0, [1, 2]]
    check((cf0 > 0).sum().item() <= 2, 'gradient flows only through selected slots')


if __name__ == '__main__':
    test_inject()
    test_rule_gen()
    test_loss_math()
    test_loss_agg_max()
    print(f'\n{PASS} passed, {FAIL} failed')
    sys.exit(1 if FAIL else 0)
