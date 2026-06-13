# FlexHook-CF — Counterfactual Hard-Negative Ablation (Refer-KITTI-V2)

> **SUPERSEDED (2026-06-12):** the headline "margin loss beats baseline (+0.29)"
> below is a **pooled-COMBINED artifact** — per-expression scoring shows the same
> run at −5.2 vs baseline. The push-vs-margin *mechanics* (§Findings 1–3) remain
> valid. Full updated matrix + metric methodology: `results-ablation-esi-cf.md`.

**Date:** 2026-06-09 · **Regime:** single RTX 3060 Ti (8 GB), `--batch-size 1`,
frozen encoders (`--freeze-text --freeze-visual`), 20 epochs, ROPE-Swin-T + RoBERTa,
tracker `Temp-NeuralSORT-kitti2`. All runs share this regime, so the valid
comparison is **vs the same-machine baseline**, not vs published FlexHook
(42.53 HOTA, trained 2×GPU / batch 7 / full fine-tune — our whole matrix sits
~2.5 HOTA below it because of the weaker regime).

Best checkpoint per run = highest val F1 (`ckpt_epoch_best_0`). HOTA via bundled
TrackEval over the 4 RK-V2 test sequences (0005/0011/0013/0019), after the
`gt_template_gen` + mkdir eval fix (see commit `2a03cc1`).

## Results

| run | CF source | λ_cf | CF loss | HOTA | DetA | AssA | LocA | val F1 |
|---|---|---|---|---|---|---|---|---|
| **baseline** (vanilla FlexHook) | — | — | — | **40.03** | 28.13 | 57.09 | 89.01 | 54.87 |
| CF | rule | 1.0 | push | 38.64 | 25.58 | 58.49 | 89.20 | 50.27 |
| CF | rule | 0.5 | push | 39.13 | 26.20 | 58.56 | 89.13 | — |
| CF | qwen (clean) | 1.0 | push | 39.36 | 26.15 | 59.36 | 89.57 | 50.89 |
| CF | qwen (clean) | 0.5 | push | 39.77 | 27.00 | 58.69 | 89.39 | — |
| **CF** | **qwen (clean)** | **1.0** | **margin (m=0.5)** | **40.32** | **28.51** | 57.18 | 89.25 | **55.53** |

## Findings

1. **The push-loss `L_cf = −log(1−P_match)` has a structural detection tax.**
   Every push run shows the same signature: **AssA ↑, DetA ↓, HOTA < baseline.**
   Because a counterfactual shares most tokens with its positive, pushing the CF
   match-prob → 0 bleeds onto the positive's match logit → suppresses detection.

2. **Tuning only walks the DetA↔AssA curve; it never beats baseline.**
   - λ 1.0 → 0.5 (rule): +0.49 HOTA (DetA recovery), AssA flat.
   - quality rule → qwen @ λ1.0: +0.72 HOTA (and AssA to 59.36, highest of all).
   - both together (qwen-λ0.5): 39.77 — best push run, still −0.26 < baseline.
   The two levers are roughly **additive / slightly overlapping** (predicted
   ~39.85). Conclusion: **push-loss CF cannot beat baseline by tuning.**

3. **A margin/hinge loss `L_cf = relu(P_match − m)` breaks the tax and beats baseline.**
   Zero gradient below the margin → already-rejected CFs get no push → the match
   head is not dragged down → **DetA recovers to 28.51 (> baseline 28.13).**
   qwen + margin(0.5) = **40.32 HOTA**, first config to exceed baseline on HOTA
   (+0.29), DetA (+0.38), and val F1 (+0.66) simultaneously.

## Caveats / open items

- **+0.29 HOTA is marginal and single-seed** — needs 2–3 seeds to confirm it is
  a real beat rather than run-to-run variance.
- The margin run's **AssA (57.18) is back to baseline level**, not the 58–59 of
  the push runs: m=0.5 is gentle enough that the CF discrimination effect is
  weak. The likely sweet spot — push hard enough to keep the AssA gain *without*
  re-incurring the DetA tax — is an **untested lower margin (m ≈ 0.3–0.4)**.
- All numbers are in the 8 GB / batch-1 / frozen regime; absolute HOTA is not
  comparable to the paper. Re-running the winning config in a full-GPU regime is
  future work.

## Reproduce

CF generation (clean qwen negatives, local ollama):
```
OLLAMA_HOST=http://localhost:11434 python tools/gen_counterfactuals.py \
  --data-root datasets/refer-kitti-v2 --dataset kitti-2 \
  --backend ollama --model qwen2.5:3b-instruct --k 4 \
  --out cf_data/counterfactuals-kitti2-qwen.json
# then filter to <= text_len (25) roberta tokens (see flexhook-cf memory)
```
Winning training run (margin loss):
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --cfg configs/train/train-kitti2.yaml --output kitti-2/cf-push-qwen-margin \
  --batch-size 1 --val-batch-size 8 --visual rope-swin-tiny --text roberta \
  --pretrained src --freeze-text --freeze-visual \
  --n-cf 3 --lambda-cf 1.0 --cf-json cf_data/counterfactuals-kitti2-qwen.json \
  --cf-loss margin --cf-margin 0.5
```
Eval: same as `infer.sh` kitti-2 block with `--resume <best ckpt>`.
