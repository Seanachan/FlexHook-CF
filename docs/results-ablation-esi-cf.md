# FlexHook-CF — Full Ablation Report: ESI + CF on Refer-KITTI-V2

**Date:** 2026-06-12 · **Datasets:** Refer-KITTI-V2 (862 test (video, expression)
sequences over 0005/0011/0013/0019) · **Tracker:** `Temp-NeuralSORT-kitti2` ·
**Encoders:** ROPE-Swin-T + RoBERTa · 20 epochs, best ckpt = highest val F1.

This report covers the **two grafts** (CF = counterfactual hard negatives,
ESI = explicit semantic injection + HMSI fusion), in **two training regimes**
(frozen encoders vs the paper's full fine-tune), under **two evaluation
aggregations** (pooled COMBINED vs per-expression mean) — and the methodological
lesson that the aggregation choice flips every conclusion.

It **supersedes the conclusion of `results-cf-ablation.md`** (the "margin loss
beats baseline (+0.29)" finding there was a pooled-COMBINED artifact; see §4).

---

## 1. Metrics: the single most important section

TrackEval scores each (video, expression) pair as its own sequence and also
emits one **pooled COMBINED** row (all detections pooled, then HOTA computed).
Two ways to aggregate 862 sequences:

- **per-expression mean** — mean of per-sequence `HOTA___AUC`. Every language
  query counts equally.
- **pooled COMBINED** — detection-weighted. Big/easy sequences dominate; an
  expression with *zero* predictions barely dents it.

**The published numbers (FlexHook 42.53, COAL 43.46) are pooled** — confirmed
by exact reproduction of FH-V2 at 42.526 pooled in the sibling GMC-Link project.

We report **both**, but trust **per-expression** for method decisions, because
pooled was caught lying three separate times in this project:

| trap | what pooled / val-F1 said | what per-expr said |
|---|---|---|
| CF-margin (frozen, local) | +0.29 "beats baseline" | −5.2, regression |
| ESI+CF (frozen, Colab) | 41.11, **best of all runs** | 31.65, catastrophic (164 dead expressions) |
| ESI (unfrozen) val F1 | 57.98 > baseline 57.22 | HOTA −1.81 vs baseline |

A model can score *high* pooled HOTA by abandoning hard expressions entirely —
zero predictions on a small sequence costs almost nothing in the pool.
Per-expression exposes exactly this failure mode (we count **zero-HOTA
sequences** as a diagnostic below).

---

## 2. Main matrix

All Colab runs: batch 7 (frozen) / batch 14 (unfrozen ≈ paper's 2 GPU × 7),
single GPU, 20 epochs, `DATA.NUM_WORKERS 16`. Same data, same tracker, same
eval path. Frozen = `--freeze-text --freeze-visual`; unfrozen = stock
`train.sh` regime (the paper's `train.sh` does **not** freeze — config defaults
are `freeze=False`; the "frozen encoders" reading of the method was wrong).

| # | regime | method | per-expr HOTA | DetA | AssA | zero-HOTA seqs | pooled HOTA | val F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | frozen | baseline | 38.14 | 30.34 | 52.18 | 22 | 40.57 | 55.32 |
| 2 | frozen | **+ESI** | **40.06** | **32.19** | **54.25** | **8** | 40.35 | 55.32 |
| 3 | frozen | +ESI+CF (qwen, margin m=0.5, λ=1) | 31.65 | 25.94 | 41.02 | 164 | 41.11 | 56.77 |
| 4 | unfrozen | baseline | 39.08 | 31.54 | 52.23 | — | **41.77** | 57.22 |
| 5 | unfrozen | +ESI | 37.27 | 30.50 | 49.67 | 107 | 41.61 | 57.98 |
| — | (paper) | FlexHook | — | — | — | — | 42.53 | — |
| — | (paper) | COAL | — | — | — | — | 43.46 | — |

(An earlier local 8 GB / batch-1 / frozen sweep of CF variants is in
`results-cf-ablation.md`; its push-vs-margin mechanics remain valid, its
"margin beats baseline" headline does not survive per-expression scoring.)

---

## 3. Finding A — ESI works, but only in the frozen (low-resource) regime

**Frozen:** ESI is a clean win on *every* component: HOTA +1.92, DetA +1.85,
AssA +2.07, and zero-HOTA sequences cut 22 → 8. Magnitude matches COAL's
reported standalone ESI gain (+2.1). Mechanism is as designed: with a frozen
backbone the visual features cannot specialize to attributes (color/motion);
the VLM caption stream injects exactly that missing semantics, helping both
detection and association.

**Unfrozen:** the gain *inverts* — HOTA −1.81, AssA −2.56, 107 zero-HOTA
sequences. Once the backbone is trainable it learns the attribute cues itself,
and the caption stream's weakness becomes net noise: captions are generated
from **GT boxes at train time but tracker boxes at eval time**, a domain gap
that the frozen model tolerated (captions were its only attribute source) but
the fine-tuned model does not need and gets misled by.

**Claim to write:** *ESI is a resource-constrained-RMOT technique. When encoder
fine-tuning is infeasible (8–16 GB GPUs), injecting offline VLM captions
recovers +1.9 per-expression HOTA and 3.6× fewer dead expressions. With full
fine-tuning it adds nothing and the train/eval caption domain gap hurts.*

## 4. Finding B — CF (counterfactual negatives) hurts everywhere

Full history (local sweep + Colab):

- **Push loss** `−log(1−P)`: structural DetA tax (CF shares tokens with its
  positive; pushing the CF bleeds onto the positive's logit). AssA ↑ but HOTA ↓
  at every λ and with both rule and LLM negatives. Tuning walks the DetA↔AssA
  curve, never beats baseline.
- **Margin loss** `relu(P−m)`: removes the tax — and *appeared* to beat
  baseline (+0.29 pooled). Per-expression scoring overturned this: −5.2.
- **Stacked on ESI** (run 3): catastrophic. 156 *new* zero-HOTA expressions vs
  ESI alone — and they are precisely the attribute-conjunction expressions CF
  trained on (`parked-white-cars`, `cars-that-are-stationary`,
  `red-autos-which-are-parked`). The discrimination objective teaches the
  matcher to *over-reject* attribute expressions at inference.
- The same run has the **best pooled HOTA (41.11) and best val F1 (56.77)** of
  all frozen runs — the sharpest demonstration that pooled/val-F1 reward
  abandoning hard expressions.

**Claim to write:** *On a two-stage PCD matcher, counterfactual hard-negative
training does not transfer from COAL's architecture: every variant (rule/LLM ×
push/margin × λ ∈ {0.5, 1.0} × with/without ESI) regresses per-expression HOTA,
with failures concentrated exactly on the attribute expressions CF targets.*

## 5. Finding C — the regime gap and the leaderboard

- Unfreezing the baseline (= matching the paper's training) is worth
  +0.94 per-expr / +1.20 pooled. Best pooled = **41.77**, still −0.76 from the
  paper's 42.53 (residual: 1 GPU vs 2, single seed, minor pipeline deltas).
- **COAL's 43.46 (pooled) is not reachable with these levers.** ESI is
  pooled-neutral (−0.2 in both regimes); CF's pooled "gain" is the artifact.
  Independently, GMC-Link's reproduction campaign found the pooled metric
  ~saturated (24+ levers neutral at the ceiling).

## 6. What the paper should be

1. **Contribution:** ESI for resource-constrained two-stage RMOT
   (+1.92 per-expr HOTA frozen, dead expressions 22→8), with the
   regime-dependence result (§3) as an honest scope statement.
2. **Negative result:** CF hard negatives on PCD matchers — complete matrix,
   mechanism (DetA tax / over-rejection), why it diverges from COAL.
3. **Methodology:** pooled COMBINED vs per-expression mean — three documented
   cases where pooled/val-F1 inverted the ranking; recommend per-expression
   (+ zero-HOTA count) as the primary metric for referring tasks.

Open items: multi-seed confirmation of the frozen ESI gain (+1.92, single
seed); 2-GPU reproduction of the unfrozen baseline (close the last −0.76);
caption-from-tracker-boxes at *train* time as a fix for the ESI domain gap
(would test §3's mechanism directly); RK-V1 as a second dataset.

## 7. Reproduce

```bash
# frozen pair (16GB GPU, e.g. Colab T4):
... main.py --cfg configs/train/train-kitti2.yaml --batch-size 7 \
    --freeze-text --freeze-visual [--esi-enabled --esi-cap-train cf_data/captions_train.json \
    --esi-cap-eval cf_data/captions_eval.json]
# unfrozen pair (≥40GB GPU): drop the freeze flags, --batch-size 14
# ESI+CF: + --n-cf 3 --lambda-cf 1.0 --cf-loss margin --cf-margin 0.5 \
#   --cf-json cf_data/counterfactuals-kitti2-qwen.json
# eval: configs/infer/kitti2.yaml --track-root tracker_outputs/Temp-NeuralSORT-kitti2
#   (ESI ckpts need the ESI flags at eval too; CF is train-only)
# per-expression parse: mean of HOTA___AUC over rows seq!=COMBINED in
#   retest-kitti-2/<run>/results/pedestrian_detailed.csv  (and count zero rows)
```

Full setup: `docs/colab_esi_setup.md`. CF mechanics detail:
`docs/results-cf-ablation.md`.

---

## Appendix A — Faithfulness audit vs the COAL paper (2026-06-13)

Audited against Jia et al. 2026, "COAL: Counterfactual and Observation-Enhanced
Alignment Learning…" (Methodology §3, Eq. 1–10, Tables 2–3). Verdict per
component:

### Faithful ✓

| component | COAL | ours |
|---|---|---|
| CF generation | LLM parses attributes, randomly replaces **one**, retains the rest | qwen single-attribute swap — same recipe |
| CF loss form | `L_cf = −log(1 − P(F_o⁺^cf, F_r^cf))` (Eq. 10), applied **only to the target object** o⁺, unweighted (`L = L_m + L_cf` ⇒ λ=1) | our **push loss, λ=1.0, on the perturbed object's own slots** — identical form. NOTE: the *faithful* config is exactly the one that regressed; the margin loss was our (non-COAL) rescue attempt |
| ESI offline + used at inference | VLM/LLM run offline, no online overhead | same (ollama offline, captions at train+eval) |
| main loss | per-object BCE on P=(cos+1)/2 (Eq. 8–9) | FlexHook CE on PCD logits — same role, different parameterization (host difference) |

### Deviated ✗ — four majors

1. **COAL's ESI includes detection, not just description.** Their frozen VLM
   runs **per frame** and produces *both* dense object proposals B *and*
   captions (§3.1); deformable sampling pools features at *VLM* boxes (Eq. 2).
   ESI in COAL densifies the **observation/detection** side. We kept FlexHook's
   tracker boxes and grafted only the caption half, **per trajectory** (one
   caption from 3 crops) instead of per-frame caption sets. Our "ESI" is at
   most half of theirs.

2. **No Pixel-Word Contextualization (Bi-Fusion).** COAL bidirectionally fuses
   pixel and referring-word features *before* sampling (Eq. 1), so object
   features are query-contextualized from the start. Removing it costs them
   **−2.04 HOTA on V2** (Tab. 3: 43.46→41.42). FlexHook's C-Hook is
   *deliberately* text-independent (`obj_f` shared across all N expressions) —
   we never had this component, and adding it would break FlexHook's
   efficiency design.

3. **Our caption denoising attends to the wrong stream.** COAL's HSR (§3.3)
   filters caption words **by the referring words** (FLL: Q=C_w, K=R_w,
   V=R_w^vl) then aggregates (ALG) → O_c is **query-adaptive**. Our HMSI
   denoises captions against **visual** `obj_f` → our caption token is
   **query-independent**. Their Tab. 3 shows caption refinement is worth
   ~0.6 HOTA; ours implements a different (visual) refinement signal entirely.

4. **COAL's CFL acts on a query-conditioned object representation — ours
   can't.** The paper is explicit: *"since the HMSI network is query-guided,
   the target's holistic representation dynamically updates from F_o⁺ to
   F_o⁺^cf when conditioned on the counterfactual query."* The push loss
   therefore moves the **joint** (object, query) embedding. In FlexHook-PCD the
   object features never change with the query — the only thing our push loss
   can move is the matcher/text side. That is a clean mechanistic explanation
   of our DetA-tax / over-rejection failures: **the model satisfies the push
   the only way it can, by suppressing the match head.** This deviation is
   structural (consequence of #2/#3), not a bug.

### Implications for the conclusions

- Our negative CF result is properly stated as: **CFL does not transfer to a
  matcher with query-independent object features.** It does *not* contradict
  COAL (whose CFL pre-supposes query-guided fusion). Arguably it *supports*
  their design: the query-guided pathway is what makes the push loss safe.
- Our frozen-ESI gain (+1.92 per-expr) came from a strictly weaker ESI (no VLM
  proposals, no Bi-Fusion, trajectory-level captions) — consistent with COAL's
  caption-only ablation magnitude (+2.09 on V2, Tab. 2).
- COAL's own no-prior baseline scores **37.29 pooled on V2** — *below*
  FlexHook's 42.53. Their 43.46 is knowledge-priors lifting a weaker base
  architecture; FlexHook+priors ≠ COAL's number, and the two ecosystems are
  not lever-compatible (this audit is the reason why).
- A faithful transplant (per-frame VLM proposals + Bi-Fusion + query-guided
  caption refinement + CFL on the joint representation) would be a different,
  much larger project — effectively re-implementing COAL's matcher inside
  FlexHook's tracking shell.
