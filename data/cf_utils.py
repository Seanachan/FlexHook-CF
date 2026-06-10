"""
Counterfactual hard-negative utilities for FlexHook-CF.

Pure standard-library only (no torch / transformers) so it can be imported and
unit-tested without the heavy training environment. Used by:
  - data/mydataloader.py  (train-time injection of counterfactual negatives)
  - tools/gen_counterfactuals.py (offline generation)
  - tools/smoke_test_cf.py (tests)

A "counterfactual negative" for a tracked object o+ is a single-attribute
perturbation of one of o+'s ground-truth referring expressions (e.g.
"white car" -> "red car"). It is guaranteed to be a *true negative* for o+
(the perturbed attribute is one o+ actually has), which forces the matcher to
verify that specific attribute instead of relying on shortcut co-occurrence.
"""

import os
import json


def load_counterfactuals(path):
    """Load a counterfactuals.json mapping {base_expression: [variant, ...]}.

    Each variant is either a string or a dict {"expression": str, "attr_type": str}.
    Returns {} if the path is empty / missing so callers degrade gracefully.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def load_captions(path):
    """Load a captions.json mapping {trajectory_key: caption_string} for ESI.

    trajectory_key is ``f'{video}_{obj}'`` (matches data/mydataloader.py __getitem__).
    Returns {} if the path is empty / missing so callers degrade to a no-op.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def _variant_string(v):
    """Accept either a raw string variant or a {"expression", "attr_type"} dict."""
    if isinstance(v, dict):
        return v.get('expression')
    return v


def _variant_attr(v):
    if isinstance(v, dict):
        return v.get('attr_type')
    return None


def collect_cf_candidates(pos_exps, counterfactuals, attr_types=None):
    """Gather counterfactual negative strings derived from a trajectory's positives.

    pos_exps:       list[str] ground-truth positive expressions for this object.
    counterfactuals: dict[str, list] base_expression -> variants.
    attr_types:     optional iterable of allowed attribute categories
                    (e.g. {"color","type"}); None = allow all.
    Returns a de-duplicated list[str] of candidate negatives, excluding any
    string that is itself one of pos_exps (guard against false negatives).
    """
    pos_set = set(pos_exps)
    allowed = set(attr_types) if attr_types else None
    out, seen = [], set()
    for p in pos_exps:
        for v in counterfactuals.get(p, []):
            if allowed is not None and _variant_attr(v) not in allowed:
                continue
            s = _variant_string(v)
            if not s or s in pos_set or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def inject_counterfactuals(sampled_target_exp, pos_exps, counterfactuals,
                           n_cf, attr_types=None, rng=None):
    """Replace up to ``n_cf`` negative slots with counterfactual hard negatives.

    Keeps the slot count N fixed (so the model's fixed N-expression path is
    unchanged): only *negative* slots (expressions not in ``pos_exps``) are
    overwritten with counterfactual perturbations of the object's positives.

    sampled_target_exp: list[str] length N (already-sampled pos/neg mix).
    pos_exps:           list[str] ground-truth positives for this object.
    counterfactuals:    dict[str, list] base_expression -> variants.
    n_cf:               int, max counterfactual negatives to inject.
    attr_types:         optional iterable of allowed attribute categories.
    rng:                random.Random-like (defaults to the ``random`` module).

    Returns (new_list, is_cf) where ``is_cf`` is a list[int] length N marking
    which slots hold an injected counterfactual negative.
    """
    if rng is None:
        import random as rng  # module-level default

    n = len(sampled_target_exp)
    is_cf = [0] * n
    if n_cf <= 0 or not counterfactuals or not pos_exps:
        return sampled_target_exp, is_cf

    candidates = collect_cf_candidates(pos_exps, counterfactuals, attr_types)
    if not candidates:
        return sampled_target_exp, is_cf

    rng.shuffle(candidates)

    pos_set = set(pos_exps)
    new = list(sampled_target_exp)
    neg_slots = [i for i, e in enumerate(new) if e not in pos_set]

    k = min(n_cf, len(candidates), len(neg_slots))
    for j in range(k):
        idx = neg_slots[j]
        new[idx] = candidates[j]
        is_cf[idx] = 1
    return new, is_cf
