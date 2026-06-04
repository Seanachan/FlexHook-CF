#!/usr/bin/env python3
"""
Offline counterfactual hard-negative generator for FlexHook-CF.

For every unique referring expression in a Refer-KITTI / Refer-KITTI-V2 train
split, produce K single-attribute counterfactual variants (each perturbs exactly
ONE attribute named in the source expression -> a guaranteed true negative for
the object the source describes) and write them to a JSON consumed by the
dataloader at train time.

Output schema (consumed by data/cf_utils.load_counterfactuals):
    { "<base expression>": [ {"expression": "<variant>", "attr_type": "color"}, ... ], ... }

Backends (``--backend``):
    rule  (default) : zero-dependency, deterministic, domain-aware word swaps.
                      Refer-KITTI expressions are templated (color/type/motion/
                      location), so controlled swaps are reliable and need no LLM.
    openai          : query an OpenAI-compatible chat API (needs OPENAI_API_KEY).
    local           : query a local HuggingFace causal LM via transformers.

Typical use (run on the GPU machine where the dataset lives):
    python tools/gen_counterfactuals.py \
        --data-root datasets/refer-kitti --dataset kitti-1 \
        --backend rule --k 4 \
        --out datasets/refer-kitti/counterfactuals.json
"""

import os
import re
import json
import argparse
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Attribute vocabulary for the rule-based backend.
# Each category maps to a swap group; a perturbation replaces one whole-word
# occurrence with a *different* member of the same group.
# --------------------------------------------------------------------------- #
ATTRIBUTE_VOCAB = {
    'color': ['black', 'white', 'red', 'silver', 'blue', 'green', 'gray', 'grey', 'golden', 'dark'],
    'type': [
        # singular and plural kept as separate swap groups (see _SWAP_GROUPS)
        'car', 'van', 'truck', 'bus', 'person', 'pedestrian', 'cyclist', 'vehicle',
        'cars', 'vans', 'trucks', 'buses', 'people', 'pedestrians', 'cyclists', 'vehicles',
    ],
    'motion': ['moving', 'parked', 'turning', 'walking', 'standing', 'stopping', 'running', 'static'],
    'location': ['left', 'right', 'front'],
}

# Finer swap groups so a plural stays plural, singular stays singular, etc.
_SWAP_GROUPS = {
    'color': [['black', 'white', 'red', 'silver', 'blue', 'green', 'gray', 'golden', 'dark']],
    'type': [
        ['car', 'van', 'truck', 'bus', 'vehicle'],
        ['cars', 'vans', 'trucks', 'buses', 'vehicles'],
        ['person', 'pedestrian', 'cyclist', 'man', 'woman'],
        ['people', 'pedestrians', 'cyclists', 'men', 'women'],
    ],
    'motion': [['moving', 'parked', 'turning', 'walking', 'standing', 'stopping', 'running', 'static']],
    'location': [['left', 'right', 'front']],
}


def _find_group(word, category):
    for group in _SWAP_GROUPS[category]:
        if word in group:
            return group
    return None


def rule_based_counterfactuals(expression, k=4, attr_types=None, rng=None):
    """Generate up to ``k`` single-attribute counterfactuals via word swaps.

    Returns list[{"expression", "attr_type"}]. Each variant differs from the
    source by exactly one whole-word token. Deterministic given ``rng``.
    """
    if rng is None:
        import random as rng
    cats = list(attr_types) if attr_types else list(_SWAP_GROUPS.keys())
    tokens = expression.split()
    lower = expression.lower()
    out, seen = [], set()

    # Enumerate (token_index, category, replacement) options, one attribute at a time.
    options = []
    for i, tok in enumerate(tokens):
        bare = re.sub(r'[^a-zA-Z]', '', tok).lower()
        if not bare:
            continue
        for cat in cats:
            group = _find_group(bare, cat)
            if not group:
                continue
            for repl in group:
                if repl == bare:
                    continue
                options.append((i, cat, tok, repl))

    rng.shuffle(options)
    for i, cat, tok, repl in options:
        # Preserve surrounding punctuation / capitalization of the original token.
        new_tok = re.sub(re.escape(re.sub(r'[^a-zA-Z]', '', tok)), repl, tok, count=1)
        new_tokens = list(tokens)
        new_tokens[i] = new_tok
        variant = ' '.join(new_tokens)
        if variant.lower() == lower or variant in seen:
            continue
        seen.add(variant)
        out.append({'expression': variant, 'attr_type': cat})
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------- #
# LLM backends (optional). Lazy imports so the rule backend needs nothing.
# --------------------------------------------------------------------------- #
_LLM_PROMPT = (
    "You rewrite referring expressions for multi-object tracking. "
    "Given an expression describing some objects, produce {k} variants. "
    "In each variant change EXACTLY ONE attribute (color, object type, motion, "
    "or location) to describe a DIFFERENT object; keep every other word identical. "
    "Return strict JSON: a list of objects with keys 'expression' and 'attr_type' "
    "(one of color/type/motion/location). Expression: \"{expr}\""
)


def _llm_counterfactuals(expression, k, attr_types, backend, model):
    prompt = _LLM_PROMPT.format(k=k, expr=expression)
    if backend == 'openai':
        from openai import OpenAI  # lazy
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or 'gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content
    elif backend == 'ollama':
        text = _ollama_chat(prompt, model or 'qwen3-vl:8b')
    elif backend == 'local':
        from transformers import pipeline  # lazy
        gen = pipeline('text-generation', model=model or 'Qwen/Qwen2.5-3B-Instruct')
        text = gen(prompt, max_new_tokens=256)[0]['generated_text']
    else:
        raise ValueError(f'unknown backend: {backend}')

    variants = _parse_json_list(text)
    # Validate + filter to allowed attribute types and true one-attribute edits.
    allowed = set(attr_types) if attr_types else None
    clean = []
    for v in variants:
        s = v.get('expression') if isinstance(v, dict) else None
        a = v.get('attr_type') if isinstance(v, dict) else None
        if not s or s.lower() == expression.lower():
            continue
        if allowed is not None and a not in allowed:
            continue
        clean.append({'expression': s, 'attr_type': a or 'unknown'})
    return clean[:k]


def _ollama_chat(prompt, model):
    """Query a local ollama server's OpenAI-free /api/chat endpoint (stdlib only).

    Host from $OLLAMA_HOST (default http://localhost:11434). format=json
    constrains the model to emit valid JSON so _parse_json_list rarely falls back.
    """
    import urllib.request  # lazy, stdlib

    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    if not host.startswith('http'):
        host = 'http://' + host
    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.7},
    }).encode()
    req = urllib.request.Request(
        host.rstrip('/') + '/api/chat',
        data=payload, headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())['message']['content']


def _parse_json_list(text):
    """Best-effort extraction of a JSON list from an LLM response."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\[.*\]', text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
    return []


# --------------------------------------------------------------------------- #
# Expression collection (mirrors data/mydataloader.py parsing).
# --------------------------------------------------------------------------- #
VIDEO_SPLITS = {
    # Minimal embedded split lists so this script is standalone. If your repo's
    # data/utils.py exposes VIDEOS, prefer --use-repo-splits to read those.
}


def collect_expressions(data_root, dataset, videos=None):
    """Collect unique referring expressions from <data_root>/expression/<video>/*.json."""
    exp_dir = os.path.join(data_root, 'expression')
    exprs = set()
    vids = videos if videos else sorted(os.listdir(exp_dir))
    for video in vids:
        vdir = os.path.join(exp_dir, video)
        if not os.path.isdir(vdir):
            continue
        for fn in os.listdir(vdir):
            path = os.path.join(vdir, fn)
            if fn.endswith('.json'):
                try:
                    s = json.load(open(path))['sentence']
                    exprs.add(s)
                except Exception:
                    continue
            else:
                # kitti-1 stores some expressions as bare filenames
                exprs.add(fn)
    return sorted(exprs)


def main():
    ap = argparse.ArgumentParser(description='Generate counterfactual hard negatives for FlexHook-CF')
    ap.add_argument('--data-root', required=True, help='dataset root containing expression/ and labels.json')
    ap.add_argument('--dataset', default='kitti-1', choices=['kitti-1', 'kitti-2', 'dance'])
    ap.add_argument('--out', required=True, help='output counterfactuals.json path')
    ap.add_argument('--backend', default='rule', choices=['rule', 'openai', 'ollama', 'local'])
    ap.add_argument('--model', default=None, help='model name for openai/ollama/local backends')
    ap.add_argument('--k', type=int, default=4, help='variants per expression')
    ap.add_argument('--attr-types', nargs='+', default=None,
                    help='restrict to these attribute categories (color type motion location)')
    ap.add_argument('--videos', nargs='+', default=None, help='restrict to these video folders')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    exprs = collect_expressions(args.data_root, args.dataset, args.videos)
    print(f'collected {len(exprs)} unique expressions from {args.data_root}')

    result = {}
    n_variants = 0
    n_empty = 0
    for expr in exprs:
        if args.backend == 'rule':
            variants = rule_based_counterfactuals(expr, args.k, args.attr_types, rng=rng)
        else:
            variants = _llm_counterfactuals(expr, args.k, args.attr_types, args.backend, args.model)
        # Validation: drop any variant equal to the source or not a single edit.
        variants = [v for v in variants if v['expression'].lower() != expr.lower()]
        if not variants:
            n_empty += 1
            continue
        result[expr] = variants
        n_variants += len(variants)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print(f'wrote {args.out}: {len(result)} expressions with counterfactuals, '
          f'{n_variants} total variants, {n_empty} expressions had no perturbable attribute')


if __name__ == '__main__':
    main()
