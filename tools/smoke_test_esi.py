"""
CPU smoke tests for the ESI+HMSI graft (no GPU / no full model build).

Covers the highest-risk edits:
  1. mask anchor: the new anchored-obj mask == original trailing-slice mask when
     ESI is OFF (cap_len=0), and is structurally correct when ON.
  2. HMSI fusion module: dummy obj_f + caption -> holistic token, shape + backward.
  3. caption tokenization shape; load_captions round-trip.

Run:  python tools/smoke_test_esi.py
"""
import os
import sys
import json
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
def check(name, ok):
    global PASS, FAIL
    print(f"  {'ok  ' if ok else 'FAIL'} - {name}")
    PASS += int(bool(ok)); FAIL += int(not ok)


def original_mask(b, n, text_len, condition_num, Sobj, text_mask):
    """The pre-ESI mask (trailing obj slice)."""
    m = torch.zeros((b, 1, n, text_len*n + condition_num*n + Sobj))
    for j in range(n):
        m[:, 0, j, j*text_len:(j+1)*text_len] = text_mask[:, j]
        m[:, 0, j, n*text_len+j*condition_num:n*text_len+(j+1)*condition_num] = 1
    m[:, :, :, -Sobj:] = 1
    return m.bool()


def new_mask(b, n, text_len, condition_num, Sobj, cap_len, text_mask):
    """The ESI mask (anchored obj + optional caption block)."""
    obj_start = text_len*n + condition_num*n
    m = torch.zeros((b, 1, n, obj_start + Sobj + cap_len))
    for j in range(n):
        m[:, 0, j, j*text_len:(j+1)*text_len] = text_mask[:, j]
        m[:, 0, j, n*text_len+j*condition_num:n*text_len+(j+1)*condition_num] = 1
    m[:, :, :, obj_start:obj_start+Sobj] = 1
    if cap_len:
        m[:, :, :, obj_start+Sobj:] = 1
    return m.bool()


print("[1] mask anchor equivalence + correctness")
b, n, text_len, condition_num = 2, 3, 25, 10
for Sobj in (48, 12, 432):                      # obj length varies per pyramid layer
    tm = (torch.rand(b, n, text_len) > 0.3).float()
    om = original_mask(b, n, text_len, condition_num, Sobj, tm)
    nm0 = new_mask(b, n, text_len, condition_num, Sobj, 0, tm)
    check(f"ESI-off mask == original (Sobj={Sobj})", torch.equal(om, nm0))
    nm1 = new_mask(b, n, text_len, condition_num, Sobj, 1, tm)
    obj_start = text_len*n + condition_num*n
    check(f"ESI-on length = +cap (Sobj={Sobj})", nm1.shape[-1] == obj_start + Sobj + 1)
    check(f"ESI-on obj block all-True (Sobj={Sobj})", bool(nm1[:, :, :, obj_start:obj_start+Sobj].all()))
    check(f"ESI-on caption col all-True for all N (Sobj={Sobj})", bool(nm1[:, :, :, obj_start+Sobj:].all()))
    # text/conditional region identical to ESI-off
    check(f"ESI-on preserves text+cond region (Sobj={Sobj})",
          torch.equal(nm1[:, :, :, :obj_start], nm0[:, :, :, :obj_start]))


print("\n[2] HMSI fusion module: dummy obj_f + caption -> holistic token")
try:
    from models.utils import CATransformerBlockTest
    from models.mymodel import Mlp_resid
    import torch.nn as nn
    C, cap_len, Sobj = 32, 25, 48
    B = 2
    hmsi_cap_norm = nn.LayerNorm(C)
    hmsi_cap_proj = Mlp_resid(C, C, C)
    hmsi_ca = CATransformerBlockTest(layer_id=0, dim=C, n_heads=4, norm_eps=None, drop_out=0.0)
    hmsi_gate = nn.Linear(C*2, C)
    hmsi_fuse_norm = nn.LayerNorm(C)
    hmsi_out = Mlp_resid(C, C, C)

    cap_feat = torch.randn(B, cap_len, C, requires_grad=True)
    obj_f = torch.randn(B, Sobj, C, requires_grad=True)
    cap_mask = (torch.rand(B, cap_len) > 0.3).float()

    cap_l = hmsi_cap_proj(hmsi_cap_norm(cap_feat))
    cap_refined = hmsi_ca(cap_l, obj_f, None, None, None)
    m = cap_mask.unsqueeze(-1).float()
    cap_vec = (cap_refined*m).sum(1, keepdim=True)/m.sum(1, keepdim=True).clamp_min(1e-6)
    obj_vec = obj_f.mean(1, keepdim=True)
    g = torch.sigmoid(hmsi_gate(torch.cat([obj_vec, cap_vec], dim=-1)))
    caption_f = hmsi_out(hmsi_fuse_norm(obj_vec + g*cap_vec))

    check("caption_f shape (B,1,C)", tuple(caption_f.shape) == (B, 1, C))
    check("caption_f finite", bool(torch.isfinite(caption_f).all()))
    loss = caption_f.pow(2).mean()
    loss.backward()
    gp = list(hmsi_gate.parameters())[0].grad
    check("backward -> finite grad on HMSI gate", gp is not None and bool(torch.isfinite(gp).all()))
    check("backward -> grad flows to obj_f", obj_f.grad is not None and bool(torch.isfinite(obj_f.grad).all()))
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"HMSI module ({type(e).__name__})", False)


print("\n[3] caption tokenization shape + load_captions round-trip")
try:
    from data.cf_utils import load_captions
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump({'0005_1': 'silver moving car on the right'}, f)
        p = f.name
    d = load_captions(p)
    check("load_captions returns dict", isinstance(d, dict) and d.get('0005_1', '').startswith('silver'))
    check("load_captions missing path -> {}", load_captions('') == {})
    os.unlink(p)
    try:
        from transformers import RobertaTokenizerFast
        tok = RobertaTokenizerFast.from_pretrained('pretrained/roberta-base')
        out = tok.batch_encode_plus(['silver moving car on the right'], padding='max_length',
                                    return_tensors='pt', truncation=True, max_length=25)
        check("caption tokenizes to (1,25)", tuple(out['input_ids'].shape) == (1, 25))
        check("squeeze -> (25,)", tuple(out['input_ids'][0].shape) == (25,))
    except Exception as e:
        print(f"  (skip tokenizer test: {e})")
except Exception as e:
    import traceback; traceback.print_exc()
    check(f"caption utils ({type(e).__name__})", False)


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
