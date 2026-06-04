# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**FlexHook-CF** — a fork of FlexHook (CVPR'2026, "Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again") extended with **counterfactual hard-negative learning** (the `-CF` graft) grafted from COAL (2026). Task: Referring Multi-Object Tracking (RMOT) — given a language expression, decide which tracked object trajectories it refers to.

`origin` → `Seanachan/FlexHook-CF`. `upstream` → `buptLwz/FlexHook` (kept for diffing the graft: `git diff upstream/main -- <file>`).

## Big-picture architecture

This is a **two-stage referring-by-tracking** system, NOT end-to-end. Understanding the data flow across files is essential:

1. **An off-the-shelf tracker runs first** (TempRMOT+NeuralSORT) and its bbox trajectories are pre-computed into `tracker_outputs/<TRACKER_ROOT>/<video>/{car,pedestrian}/predict.txt`. This repo only trains/evaluates the *referring head* that scores trajectory↔expression matches. Inference reads tracker outputs via `--track-root`.

2. **Frozen encoders, cheap training.** Visual = ROPE-Swin-T (`models/swin_transformer_rope.py`, `vit_rope.py`), text = RoBERTa (or CLIP/BERT), both frozen. Only the small referring head trains → the paper's ~0.77 h training.

3. **Per-trajectory matching.** A training sample = **one object trajectory + N expressions** (`sample_expression_num`), with binary match labels. This is built in `data/mydataloader.py::RMOT_Dataset.__getitem__`. Labels: `1 if expr in data['expression'][last_frame] else 0`.

4. **Model forward** (`models/mymodel.py::forward` → `forward_features` → `decode`):
   - **C-Hook**: bilinearly `grid_sample`s target features from the frozen backbone feature map at bbox grid locations (no re-encoding). The resulting trajectory feature `obj_f` is **text-independent** and shared across all N expressions.
   - **Language-conditioned reference points** (`conditional_pos_*`) ARE text-dependent; regularized by `point_dispersion_loss` (boundary barrier so points don't collapse to grid edges).
   - **PCD** (Pairwise Correspondence Decoder, `CATransformerBlockTest` in `models/utils.py`): masked cross-attention matcher that replaces CLIP cosine similarity. Output = `(B, num_layers=4, N, 2)` match logits.

5. **Loss is plain `torch.nn.CrossEntropyLoss(reduction='none')`** (`main.py`, the `criterion`), reweighted by `config.POSW`/`config.entropy`, plus the model's `regular` term. ⚠️ `loss.py`'s `SimilarityLoss`/`multiSimilarityLoss` (InfoNCE) are **dead code — not used in training**.

6. **Eval path**: `main.py::inference` produces per-`(video, obj, frame, expression)` logits → `test_utils.py::generate_final_results*` → bundled `TrackEval` (`run_mot_challenge.py --METRICS HOTA`). Metric is HOTA/DetA/AssA.

### The `-CF` graft (this fork's contribution)
Counterfactual hard negatives: single-attribute perturbations of an object's positive expressions ("white car" → "red car") are injected as negatives and pushed away, forcing per-attribute verification. Default config is a **no-op** (`LAMBDA_CF=0, N_CF=0` ⇒ identical to vanilla FlexHook).
- `data/cf_utils.py` — pure-stdlib injection (`inject_counterfactuals`) + JSON loading. Imported by the dataloader.
- `data/mydataloader.py` — loads `counterfactuals.json` (train only), swaps negative slots for CF negatives keeping N fixed, returns an extra `is_cf` tensor.
- `main.py` — unpacks `is_cf`, adds push-loss `L_cf = -log(1 - P_match)` over CF slots, weighted by `LAMBDA_CF`.
- `config.py` — `LAMBDA_CF`, `N_CF`, `CF_JSON_PATH`, `CF_ATTR_TYPES`.
- `tools/gen_counterfactuals.py` — offline negative generator. `tools/smoke_test_cf.py` — CPU tests.

## Datasets & config

Dataset keys (set by `--cfg`/`config.DATA.DATASET`; video splits in `data/utils.py::VIDEOS`):
`kitti-1` = Refer-KITTI · `kitti-2` = Refer-KITTI-V2 · `dance` = Refer-Dance · `mix` = LaMOT (all `-mix`/`*-tiny` scripts and `data/mydataloader_mix.py`).

- Config is **yacs** (`config.py` defaults) + per-run YAML in `configs/train/*.yaml` and `configs/infer/*.yaml`; CLI flags and `--opts KEY VALUE` override. `data_root`/`track_root` come from the YAML — edit there.
- Datasets are **soft-linked**, not copied — see `datasets/DATASET.md` (download Refer-KITTI/v2 from TempRMOT, Refer-Dance from iKUN, LaMOT from its repo; `ln -s` image dirs into placeholders).
- Pretrained encoders go in `pretrained/` per `pretrained/PRETRAIN.md`. SOTA checkpoints (`SOTA_ckpts/`) and tracker outputs (`tracker_outputs/`) come from the repo's Baidu link.
- `.gitignore` excludes weights (`*.pth/*.pt/*.bin/*.safetensors`), output dirs (`/kitti-1/`, `/kitti-2/`, `/dance/`, `/mix/`, `/retest-*/`), and `counterfactuals.json`.

## Commands

Training/eval use multi-GPU `torch.distributed.launch`; the canonical commands live in the shell scripts — **comment out the unrelated blocks, then `sh xx.sh`**. `--batch-size` is per-GPU; scale `--nproc_per_node` to available GPUs.

```bash
sh train.sh        # train (default block: kitti-2) then auto-eval epochs 0..4
sh eval.sh         # reproduce paper HOTA from SOTA_ckpts (kitti-1/kitti-2/...)
sh infer.sh        # inference with a checkpoint + tracker outputs
sh train-mix.sh    # LaMOT training; sh cpmix.sh / eval.sh aggregate mix subsets
```

Raw training invocation (what the scripts wrap):
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py \
  --cfg configs/train/train-kitti2.yaml --output kitti-2/try \
  --batch-size 7 --visual rope-swin-tiny --text roberta --pretrained src
```
Eval adds `--eval --resume <ckpt>.pth` with a `configs/infer/*.yaml` and `--track-root tracker_outputs/<TRACKER_ROOT>`.

### `-CF` workflow
```bash
python tools/smoke_test_cf.py          # CPU unit tests (no dataset/GPU needed); run after touching the graft
python tools/gen_counterfactuals.py --data-root <data_root> --dataset kitti-2 \
       --backend rule --k 4 --out <data_root>/counterfactuals.json   # rule backend = zero extra deps
```
Enable CF during training by appending to the `main.py` args:
- baseline (vanilla FlexHook): add nothing — defaults are a no-op.
- CF-as-CE ablation: `--n-cf 3 --lambda-cf 0   --cf-json <data_root>/counterfactuals.json`
- CF-push (full method): `--n-cf 3 --lambda-cf 1.0 --cf-json <data_root>/counterfactuals.json`

Project plan, phase status, and the full ablation/run matrix live in `PLAN.md`.

## Gotchas

- Editing the dataloader's train return tuple requires matching the unpack in `main.py::train_one_epoch` (currently 11 fields incl. `is_cf`); val/test unpacks differ — don't conflate them.
- `tracker_outputs` frame ids are shifted by `-1` for the TempRMOT tracker in `data/mydataloader.py` (test branch) — preserve this when changing parsing.
- `set_epoch()` re-parses data each epoch to reshuffle sampled frames; shuffling in the DataLoader only shuffles trajectory order.
