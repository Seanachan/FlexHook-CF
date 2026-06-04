# FlexHook-CF: Counterfactual Hard-Negative Learning for Two-Stage RMOT

FlexHook-CF extends **FlexHook** (CVPR'2026, [*Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again*](https://arxiv.org/abs/2503.07516)) with **counterfactual hard-negative learning**, a component adapted from COAL (*Counterfactual and Observation-Enhanced Alignment Learning for Discriminative RMOT*, 2026).

Two-stage referring-by-tracking is fast and strong, but its matcher tends to learn shortcut attribute associations. FlexHook-CF injects **single-attribute counterfactual negatives** — minimal perturbations of an object's own referring expressions (e.g. *"white car"* → *"red car"*) — and pushes them away with a dedicated loss, forcing the model to verify each attribute instead of relying on co-occurrence. Negatives are generated **offline** and used only at training time, so there is **no inference-time cost**. The goal is to improve compositional generalization, especially on Refer-KITTI-V2.

> The `-CF` additions are **off by default** (`LAMBDA_CF=0`, `N_CF=0`), so the repository reproduces vanilla FlexHook exactly unless counterfactual learning is explicitly enabled.

<div align="center">
  <img src="./FlexHook.png" width="100%" height="100%"/>
</div><br/>

## Installation

Besides Torch, the core components also include RoBERTa, ROPE Swin-T, and CLIP. We recommend setting up the environment following the guidelines for <a href="https://github.com/naver-ai/rope-vit">ROPE-ViT</a> and installing the Transformers library to support the language model.

> Note: We use `PyTorch 2.6.0` and `CUDA 12.4`. They differ from those in ROPE-ViT, but the ROPE-ViT configurations still apply.

1. Follow <a href="https://github.com/naver-ai/rope-vit">ROPE-ViT</a> to prepare the environment.
2. Prepare the dataset following [here](datasets/DATASET.md).
3. Prepare the pretrained weights following [here](pretrained/PRETRAIN.md).
4. Download the **best weights** and **tracker results** from <a href="https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA?pwd=d3qj" title="model">FlexHook_best</a> and place them in the root directory as `SOTA_ckpts/` and `tracker_outputs/`.
5. Change the necessary items in `configs/` and the batch size in `*.sh`.

## Inference and Training

All training and testing commands are listed in `*.sh`. Comment out unrelated parts, then run `sh xx.sh` (e.g. `sh infer.sh` to reproduce the paper's results).

The `-mix` suffix denotes LaMOT-related code; LaMOT inference runs on subsets by default. To get full LaMOT results, sequentially modify `cpmix.sh` and `eval.sh` to aggregate all videos and compute overall performance.

## Counterfactual learning (`-CF`)

1. **Generate counterfactual negatives** (offline). The default `rule` backend needs no extra dependencies; `openai` / `local` LLM backends are also available. Point `--data-root` at the same `data_root` your training config uses (the directory containing `expression/` and `labels.json`):

   ```bash
   python tools/gen_counterfactuals.py \
       --data-root <data_root> --dataset kitti-2 \
       --backend rule --k 4 --out <data_root>/counterfactuals.json
   ```

2. **Verify the graft** (CPU, no dataset/GPU required):

   ```bash
   python tools/smoke_test_cf.py
   ```

3. **Train** by appending these flags to the `main.py` invocation in `train.sh`:

   | Setting | Flags |
   |---|---|
   | Baseline (vanilla FlexHook) | *(none — defaults are a no-op)* |
   | CF-as-CE ablation | `--n-cf 3 --lambda-cf 0   --cf-json <data_root>/counterfactuals.json` |
   | CF-push (full method) | `--n-cf 3 --lambda-cf 1.0 --cf-json <data_root>/counterfactuals.json` |

   Knobs: `LAMBDA_CF` (push-loss weight), `N_CF` (max counterfactual negatives per sample), `CF_JSON_PATH`, `CF_ATTR_TYPES` (config.py / YAML / `--opts`).

## Acknowledgements

Built on [FlexHook](https://github.com/buptLwz/FlexHook) (MIT). The counterfactual-learning component is adapted from COAL. Please cite the original FlexHook and COAL papers when using this work.
