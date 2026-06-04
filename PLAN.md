# Project Plan — Counterfactual Hard-Negative Learning for Two-Stage RMOT

**Working name:** FlexHook-CF
**Base:** FlexHook (CVPR'2026, `github.com/buptLwz/FlexHook`, MIT) — cloned at `./FlexHook`
**Graft:** COAL (2026) counterfactual learning — *cheap half only* (LLM negatives + push-loss; no DINO-X)
**Compute:** user has GPU + Refer-KITTI / Refer-KITTI-V2 ready
**Date:** 2026-06-04

---

## 1. Thesis / Contribution

Two-stage referring-by-tracking (FlexHook) is **efficient and strong** (HOTA 53.83 on Refer-KITTI, 0.77 h training) but its matcher (PCD) learns **shortcut attribute associations** — it latches onto frequent co-occurring cues instead of verifying each attribute. This is the "Sparsity-Discriminability Paradox" (COAL): fine-grained discrimination is demanded but per-attribute supervision is sparse.

**Claim:** Injecting **LLM-generated single-attribute counterfactual hard negatives** ("white car" → "red car") with a **push-loss** into FlexHook's existing PCD contrastive supervision forces compositional attribute verification, yielding **SOTA on the compositional Refer-KITTI-V2** at near-zero extra training cost and **zero inference overhead** (negatives generated offline, used only at train time).

**Target:** beat both parents on RK-V2 — FlexHook 42.53, COAL 43.46 HOTA. Maintain RK ≈ 53.8.

---

## 2. Why this graft (vs alternatives)

- FlexHook already contains InfoNCE contrastive (`loss.py::multiSimilarityLoss`) with **generic** negatives. COAL's idea = upgrade those to **structured single-attribute** negatives + a directed push-loss. This is an *augmentation of existing supervision*, not a rebuild.
- COAL's expensive dependency (DINO-X for proposals/captions) is **not needed**: FlexHook's off-the-shelf tracker already supplies detection/trajectories. We keep only COAL's counterfactual-learning (CFL) component.
- Effect should be **largest on RK-V2** (compositional expressions) — exactly the open frontier (field plateaus ~52–54 on RK; spread is on V2 35→43).

Rejected alternatives: DKGTrack POS-decouple (lower ceiling), HFF Look-Back (small +0.8 payoff — kept as optional Phase 7 add-on).

---

## 3. Method design

### 3.1 FlexHook base (unchanged, recap)
Frozen ROPE-Swin-T (visual) + frozen RoBERTa (text). Off-the-shelf tracker → bbox trajectories.
- **C-Hook**: bilinear neighboring-grid sample of target features `J` from frozen backbone map at bbox locations (+3 tracking-noise augs); M language-conditioned reference points `P_r`.
- **Temporal Integration**: concat multi-frame `J` + grid displacements → MLP → trajectory feats `F_J`.
- **PCD** (`models/mymodel.py::CATransformerBlockTest`): masked cross-attn, Q=learnable queries, K=V=concat(`F_J`, `F_r`, `F_l`); per-query mask → pairwise discrimination. 4 pyramid levels → match logits `S`.
- **Loss** (`loss.py`): `L_focal(S, S_gt)` + `λ·point_dispersion_loss(P_r)` (boundary barrier).

### 3.2 Counterfactual graft (new) — "FlexHook-CF"
For a referring expression `R` matched to target `o⁺`, generate counterfactual `R̃` = `R` with **exactly one attribute perturbed** (color / type / direction / location), structure preserved. `R̃` is a **true negative** for `o⁺`.

- **Shared compute:** `F_J` (trajectory feats) depend only on image+boxes → computed once. Only the linguistic branch (`F_l`, reference points `F_r`, PCD) is re-run for `R̃`. Cheap.
- **Push-loss** (COAL Eq.): `P = (S+1)/2`; `L_cf = −log(1 − P(F^cf_{o⁺}, F^cf_r))` applied **only on target `o⁺`**; other candidates **masked** (avoid penalizing objects that legitimately match `R̃`).
- **Total:** `L = L_focal(pos) + λ_pd·L_pd + λ_cf·L_cf`.
- **1-Image-2N-Queries batching:** each image carries N positive expressions + N counterfactual negatives (start N per FlexHook's per-image expr count; COAL used N=10).

### 3.3 Graft points (file-by-file)
| File | Change |
|---|---|
| `tools/gen_counterfactuals.py` (new) | offline LLM negative generation + validation, writes JSON keyed by expr id |
| `data/mydataloader.py` | load counterfactual exprs per sample; build 2N-query batch; emit CF labels + masks |
| `loss.py` | add `CounterfactualLoss` (push-loss); wire into total loss |
| `models/mymodel.py` | route CF linguistic feats through C-Hook conditioning + PCD (reuse `F_J`); return CF match logits |
| `main.py` | 2N-query forward, add `λ_cf·L_cf`, separate logging |
| `config.py` / `configs/*` | `λ_cf`, `N_cf`, CF-json path, attribute-type toggles |

---

## 4. Counterfactual generation pipeline

- **Attributes to perturb:** color, object type/category, motion/direction, spatial location. One per negative.
- **LLM:** any capable model (Qwen / local Llama / API). Prompt: "Given expression E describing tracked objects, change exactly ONE attribute to make it describe a *different* object; keep all other words; output K variants labeled by which attribute changed."
- **Validation (mirror COAL 2-stage, lightweight):**
  1. attribute-diff check — negative differs from source by exactly one attribute token-set.
  2. grounding sanity — negative should not be a trivial paraphrase; spot-check ~5% manually.
- **False-negative guard:** at train time, mask any scene object that the CF expression legitimately matches; apply `L_cf` only on `o⁺`.
- Output: `datasets/<ds>/counterfactuals.json`.

---

## 5. Phased execution

| Phase | Goal | Exit check |
|---|---|---|
| **0. Repro-infer** | env (ROPE-ViT, PT2.6/CU12.4), datasets soft-linked, run `infer.sh` w/ provided ckpts | HOTA ≈ paper (RK 53.83 / V2 42.53) via bundled TrackEval |
| **1. Repro-train** | run `train.sh` RK + RK-V2 from scratch | our baseline numbers logged; ~0.77 h/run confirmed |
| **2. CF-gen** | build `gen_counterfactuals.py`, generate + validate negatives for train splits | JSON exists; attribute-diff valid; manual spot-check pass |
| **3. Implement graft** | dataloader + loss + model routing + config | smoke-test tiny batch: shapes OK, loss finite, `backward()` works on CPU |
| **4. Train FlexHook-CF** | RK + RK-V2 | HOTA/DetA/AssA vs Phase-1 baseline |
| **5. Ablations** | isolate the effect (see §6) | tables filled; counterfactual-specificity proven vs random negs |
| **6. Analysis + draft** | qualitative cases, failure analysis, paper draft (ars-plan chapters §8) | draft + figures |
| **7. (opt) Look-Back** | add HFF hard-example mining if time | delta measured |

---

## 6. Ablation matrix

- `λ_cf` sweep {0, 0.1, 0.5, 1.0}
- `N_cf` per expr {1, 3, 5, 10}
- attribute type: color-only / type-only / motion-only / location-only / all
- **counterfactual vs random negatives** (the key claim — does *structured* perturbation beat generic hard negs?)
- with/without other-candidate masking (false-negative guard value)
- **compositional generalization:** train RK → test RK-V2 (COAL's headline strength)

---

## 7. Evaluation protocol

- **Datasets:** Refer-KITTI, Refer-KITTI-V2 (TempRMOT splits). Optional: Refer-Dance, LaMOT for generalization (FlexHook already supports via `-mix`).
- **Metric:** HOTA (primary), DetA, AssA, LocA via bundled `TrackEval`. Use FlexHook `infer.sh`/`eval.sh` path unchanged for comparability.
- **Compare against:** our Phase-1 baseline (primary), published FlexHook + COAL (reference).

---

## 8. Paper structure (ars-plan deliverable)

1. **Introduction** — RMOT task; two-stage revival; sparsity-discriminability paradox; contribution (CF hard negatives, zero-inference-cost, V2 SOTA).
2. **Related Work** — RMOT one-/two-stage; VL grounding; counterfactual & contrastive learning; hard-negative mining.
3. **Method** — FlexHook base recap; counterfactual generation; CF-augmented PCD; push-loss; 2N-query training.
4. **Experiments** — datasets/metrics; baseline repro; main results (RK, RK-V2); ablations; compositional generalization.
5. **Analysis & Discussion** — why it helps on V2; qualitative attribute-discrimination; limitations (LLM-negative noise, dependence on tracker stage).
6. **Conclusion.**

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PCD already contrastive → marginal gain | target V2 compositional; ablate CF-vs-random to prove specificity |
| LLM negatives noisy / false negatives | validation pass + train-time masking + apply `L_cf` only on `o⁺` |
| Env friction (ROPE-ViT/CU12.4/PT2.6) | Phase 0 derisks first; pin versions |
| Baseline won't hit exact paper HOTA | use our own trained baseline as reference; report delta not absolute |
| 0.77 h claim assumes 2×4090 / frozen encoders | confirm in Phase 1; scale batch in `*.sh` to user GPU |

---

## 10. Open decisions (confirm before Phase 2)

- LLM for negative generation: local model vs API? (affects cost/throughput)
- Primary dataset focus: RK-V2 first (frontier) or RK first (sanity)? — plan assumes **RK-V2 primary, RK for sanity**.
- Paper venue/target (drives writing depth) — TBD, not blocking implementation.

---

## 11. Implementation status (2026-06-04)

**Code complete (Phase 2 + 3), unit-verified on CPU. Defaults are a no-op (`LAMBDA_CF=0`, `N_CF=0`) → identical to vanilla FlexHook.** GPU runs (Phases 0/1/4/5) happen on the user's machine where dataset + conda env + FlexHook ckpt live.

Files added / changed under `FlexHook/`:
- `data/cf_utils.py` *(new)* — pure-stdlib `inject_counterfactuals`, `load_counterfactuals`.
- `tools/gen_counterfactuals.py` *(new)* — offline generator; default `rule` backend (zero deps, domain-aware single-attribute swaps) + optional `openai`/`local` LLM backends.
- `tools/smoke_test_cf.py` *(new)* — CPU tests (56 assertions: injection logic, rule generator, push-loss math + backward). **All pass.**
- `data/mydataloader.py` — load `counterfactuals.json` (train only); inject CF negatives into N slots; return `is_cf`.
- `main.py` — `--lambda-cf/--n-cf/--cf-json` flags; unpack `is_cf`; push-loss `L_cf = -log(1-P_match)` over CF slots.
- `config.py` — `LAMBDA_CF`, `N_CF`, `CF_JSON_PATH`, `CF_ATTR_TYPES` + CLI overrides.

**Not yet verified (needs GPU env):** end-to-end real-batch → model forward → loss. First run on the GPU machine: `python tools/smoke_test_cf.py`, then a 1–2 iteration dry train.

### Run commands (on the GPU machine)
```bash
# 0. confirm graft in the real env
python tools/smoke_test_cf.py

# 1. generate counterfactuals (rule backend = no extra deps). Point --data-root
#    at the same data_root your train yaml uses (the dir with expression/ + labels.json).
python tools/gen_counterfactuals.py --data-root datasets/refer-kitti   --dataset kitti-1 --backend rule --k 4 --out datasets/refer-kitti/counterfactuals.json
python tools/gen_counterfactuals.py --data-root datasets/refer-kitti-v2 --dataset kitti-2 --backend rule --k 4 --out datasets/refer-kitti-v2/counterfactuals.json

# 2. train. Append to the main.py args in train.sh:
#   baseline (unchanged):     (nothing — defaults are vanilla FlexHook)
#   CF-as-CE ablation:        --n-cf 3 --lambda-cf 0   --cf-json datasets/<ds>/counterfactuals.json
#   CF-push (full method):    --n-cf 3 --lambda-cf 1.0 --cf-json datasets/<ds>/counterfactuals.json
# 3. eval with the existing eval.sh path (unchanged) → TrackEval HOTA/DetA/AssA.
```
Optionally swap the generator to a real LLM: `--backend openai --model gpt-4o-mini` (needs `OPENAI_API_KEY`) or `--backend local --model Qwen/Qwen2.5-3B-Instruct`.
