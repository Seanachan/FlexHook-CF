"""
CPU smoke tests for the L1 QCOND graft (query-conditioned obj_f via per-expression
FiLM, block-diagonal obj block). Mirrors smoke_test_esi.py: standalone reimplementation
of the decode obj-block + mask logic, no full model build.

Covers the highest-risk edits:
  1. identity init: zero-init gamma/beta => obj_block == obj_f broadcast across N
     (FiLM is a no-op at step 0, so an untrained QCOND head == vanilla).
  2. shapes: obj_block (B, N*Lobj, C); mask (B,1,N, obj_start + N*Lobj + cap).
  3. block-diagonal mask: expression j sees ONLY its own obj sub-block.
  4. qcond-OFF mask == vanilla shared-obj mask (visible to all N).
  5. backward -> finite grads to gamma/beta.

Run:  python tools/smoke_test_qcond.py
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0; FAIL = 0
def check(name, ok):
    global PASS, FAIL
    print(f"  {'ok  ' if ok else 'FAIL'} - {name}")
    PASS += int(bool(ok)); FAIL += int(not ok)


def qcond_obj_block(obj_f, text, text_mask, gamma, beta, tnorm):
    """obj_f (B,Lobj,C), text (B,N,Tl,C), text_mask (B,N,Tl) -> obj_block (B,N*Lobj,C)."""
    tm = text_mask.unsqueeze(-1).to(obj_f.dtype)
    t_n = (text*tm).sum(2)/tm.sum(2).clamp_min(1e-6)          # (B,N,C)
    t_n = tnorm(t_n)
    g = gamma(t_n).unsqueeze(2); b = beta(t_n).unsqueeze(2)   # (B,N,1,C)
    return (obj_f.unsqueeze(1)*(1+g)+b).flatten(1,2)          # (B,N*Lobj,C)


def build_mask(b, n, text_len, condition_num, Lobj, cap_len, text_mask, qcond, residual=False):
    obj_start = text_len*n + condition_num*n
    if residual:
        obj_total = Lobj + n*Lobj          # shared + per-query
    elif qcond:
        obj_total = n*Lobj                 # replace
    else:
        obj_total = Lobj                   # vanilla shared
    m = torch.zeros((b,1,n, obj_start + obj_total + cap_len))
    cond_off = obj_start + (Lobj if residual else 0)
    for j in range(n):
        m[:,0,j, j*text_len:(j+1)*text_len] = text_mask[:,j]
        m[:,0,j, n*text_len+j*condition_num:n*text_len+(j+1)*condition_num] = 1
        if qcond:
            m[:,0,j, cond_off+j*Lobj:cond_off+(j+1)*Lobj] = 1
    if residual or not qcond:
        m[:,:,:, obj_start:obj_start+Lobj] = 1
    if cap_len:
        m[:,:,:, obj_start+obj_total:] = 1
    return m.bool()


B,N,Tl,C = 2,3,25,32
cond, Lobj = 10, 48

print("[1] identity init: zero gamma/beta => obj_block == obj_f broadcast (no-op at step 0)")
torch.manual_seed(0)
gamma = nn.Linear(C,C); nn.init.zeros_(gamma.weight); nn.init.zeros_(gamma.bias)
beta  = nn.Linear(C,C); nn.init.zeros_(beta.weight);  nn.init.zeros_(beta.bias)
tnorm = nn.LayerNorm(C)
obj_f = torch.randn(B,Lobj,C, requires_grad=True)
text  = torch.randn(B,N,Tl,C, requires_grad=True)
text_mask = (torch.rand(B,N,Tl) > 0.3).float()
ob = qcond_obj_block(obj_f, text, text_mask, gamma, beta, tnorm)
check("obj_block shape (B,N*Lobj,C)", tuple(ob.shape) == (B, N*Lobj, C))
ob_view = ob.view(B,N,Lobj,C)
# zero gamma/beta -> obj_f*(1+0)+0 == obj_f for every expression slot
check("identity-init: every expr's obj block == obj_f", torch.allclose(ob_view, obj_f.unsqueeze(1).expand(B,N,Lobj,C), atol=1e-5))

print("\n[2] FiLM actually conditions once gamma/beta are non-trivial")
nn.init.normal_(gamma.weight, std=0.1); nn.init.normal_(beta.weight, std=0.1)
ob2 = qcond_obj_block(obj_f, text, text_mask, gamma, beta, tnorm).view(B,N,Lobj,C)
# different expressions (different pooled text) -> different obj blocks
check("non-identity FiLM makes per-expr obj blocks differ", not torch.allclose(ob2[:,0], ob2[:,1], atol=1e-4))

print("\n[3] block-diagonal mask: expr j sees ONLY its own obj sub-block")
mq = build_mask(B,N,Tl,cond,Lobj,0,text_mask,qcond=True)
obj_start = Tl*N + cond*N
okdiag = True
for j in range(N):
    own = mq[:,0,j, obj_start+j*Lobj:obj_start+(j+1)*Lobj].all()
    others = [mq[:,0,j, obj_start+k*Lobj:obj_start+(k+1)*Lobj].any() for k in range(N) if k!=j]
    okdiag &= bool(own) and not any(bool(o) for o in others)
check("each expr's own obj block all-True, others all-False", okdiag)
check("qcond mask width = obj_start + N*Lobj", mq.shape[-1] == obj_start + N*Lobj)

print("\n[4] qcond-OFF mask == vanilla shared-obj mask (visible to all N)")
moff = build_mask(B,N,Tl,cond,Lobj,0,text_mask,qcond=False)
check("off-mask width = obj_start + Lobj", moff.shape[-1] == obj_start + Lobj)
check("off-mask obj block visible to ALL N", bool(moff[:,:,:, obj_start:obj_start+Lobj].all()))
# text+conditional region identical between qcond on/off
check("text+conditional region identical on/off",
      torch.equal(mq[:,:,:,:obj_start], moff[:,:,:,:obj_start]))

print("\n[5] caption block after obj, visible to all N (qcond ON + ESI)")
mqc = build_mask(B,N,Tl,cond,Lobj,1,text_mask,qcond=True)
check("qcond+cap mask width = obj_start + N*Lobj + 1", mqc.shape[-1] == obj_start + N*Lobj + 1)
check("caption col all-True for all N", bool(mqc[:,:,:, -1:].all()))

print("\n[6] backward -> finite grads to gamma/beta + obj_f")
loss = qcond_obj_block(obj_f, text, text_mask, gamma, beta, tnorm).pow(2).mean()
loss.backward()
gg = gamma.weight.grad
check("grad on gamma finite", gg is not None and bool(torch.isfinite(gg).all()))
check("grad flows to obj_f", obj_f.grad is not None and bool(torch.isfinite(obj_f.grad).all()))
check("grad flows to text", text.grad is not None and bool(torch.isfinite(text.grad).all()))

print("\n[7] RESIDUAL (augment) mode: shared obj_f visible-to-all AND conditioned block-diagonal")
mr = build_mask(B,N,Tl,cond,Lobj,0,text_mask,qcond=True,residual=True)
obj_start = Tl*N + cond*N
check("residual mask width = obj_start + Lobj + N*Lobj",
      mr.shape[-1] == obj_start + Lobj + N*Lobj)
# shared block (first Lobj of obj region) visible to ALL N
check("residual: shared obj block visible to ALL N",
      bool(mr[:,:,:, obj_start:obj_start+Lobj].all()))
# conditioned block (offset by Lobj) is block-diagonal
cond_off = obj_start + Lobj
okdiag = True
for j in range(N):
    own = mr[:,0,j, cond_off+j*Lobj:cond_off+(j+1)*Lobj].all()
    others = [mr[:,0,j, cond_off+k*Lobj:cond_off+(k+1)*Lobj].any() for k in range(N) if k!=j]
    okdiag &= bool(own) and not any(bool(o) for o in others)
check("residual: conditioned block is block-diagonal", okdiag)
# text+conditional region identical to replace/vanilla
check("residual: text+conditional region identical to vanilla",
      torch.equal(mr[:,:,:,:obj_start], moff[:,:,:,:obj_start]))
# residual + caption: caption after the full obj region, visible to all
mrc = build_mask(B,N,Tl,cond,Lobj,1,text_mask,qcond=True,residual=True)
check("residual+cap mask width = obj_start + Lobj + N*Lobj + 1",
      mrc.shape[-1] == obj_start + Lobj + N*Lobj + 1)
check("residual+cap: caption col all-True", bool(mrc[:,:,:, -1:].all()))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
