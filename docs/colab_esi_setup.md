# Colab setup — ESI+HMSI training (Stage 3)

Goal: on a Colab GPU (T4 16 GB, headless), train **baseline** and **ESI+HMSI** at
**batch-7** (recovers the ~2.5 HOTA the 8 GB/batch-1 regime cost us), then eval
**per-expression** (NOT pooled COMBINED — the artifact that burned us). Captioning
already done locally → only the 2 JSONs travel.

## Prerequisites (manual, you)
1. Restart Claude Code so colab-mcp tools load.
2. Colab: Runtime → Change runtime type → **GPU**. Run colab-mcp's connect cell + Google auth.
3. Put on Google Drive (e.g. `MyDrive/flexhook/`):
   - `datasets/refer-kitti-v2/` (KITTI images under `KITTI/training/image_02/`, `labels.json`, `expression/`, `gt_template_gen/`, `gt_template/`)
   - `pretrained/` (`roberta-base/`, `swin_rope_mixed_tiny_patch4_window7_224/`)
   - `tracker_outputs/Temp-NeuralSORT-kitti2/`
   - `cf_data/captions_train.json`, `cf_data/captions_eval.json`

## Cells (run in order once connected)

### 1. GPU + clone
```bash
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone https://github.com/Seanachan/FlexHook-CF.git
%cd FlexHook-CF
!git log --oneline -3   # expect 8f671ce ESI graft, b48e233 GT captioner
```

### 2. Environment (Colab has torch preinstalled; add the rest)
```bash
!pip -q install transformers==4.57.6 timm einops scipy pyyaml yacs termcolor opencv-python
!python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"
```
> If torch/cu mismatch surfaces, pin to match (ROPE-ViT configs still apply per README). Verify CUDA True before training.

### 3. Mount Drive + symlink data (no copy)
```bash
from google.colab import drive; drive.mount('/content/drive')
D='/content/drive/MyDrive/flexhook'   # adjust to your Drive layout
import os
for src,dst in [(f'{D}/datasets/refer-kitti-v2','datasets/refer-kitti-v2'),
                (f'{D}/pretrained/roberta-base','pretrained/roberta-base'),
                (f'{D}/pretrained/swin_rope_mixed_tiny_patch4_window7_224','pretrained/swin_rope_mixed_tiny_patch4_window7_224'),
                (f'{D}/tracker_outputs/Temp-NeuralSORT-kitti2','tracker_outputs/Temp-NeuralSORT-kitti2')]:
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    if not os.path.exists(dst): os.symlink(src,dst)
os.makedirs('cf_data',exist_ok=True)
!cp "$D/cf_data/captions_train.json" "$D/cf_data/captions_eval.json" cf_data/
!ls -la datasets/refer-kitti-v2 pretrained cf_data
```

### 4. Smoke tests (must pass before training)
```bash
!python tools/smoke_test_cf.py | tail -1
!python tools/smoke_test_esi.py | tail -1
```

### 5. Train — baseline then ESI (batch-7; checkpoint to Drive for session limits)
```bash
# BASELINE (no ESI) — the same-machine reference, batch-7
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29711 main.py \
  --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/colab-base \
  --batch-size 7 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
  --freeze-text --freeze-visual

# ESI+HMSI
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29712 main.py \
  --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/colab-esi \
  --batch-size 7 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
  --freeze-text --freeze-visual \
  --esi-enabled --esi-cap-train cf_data/captions_train.json --esi-cap-eval cf_data/captions_eval.json
```
> AUTO_RESUME is on — if a session drops, copy `kitti-2/colab-*` to Drive and re-run the same cell to resume. (Or add `!cp -r kitti-2 $D/` checkpoints periodically.)

### 6. Eval (per-expression) — for each best ckpt
```bash
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29713 main.py \
  --cfg configs/infer/kitti2.yaml --track-root tracker_outputs/Temp-NeuralSORT-kitti2 --data-path 12 \
  --output retest-kitti-2/colab-esi --val-batch-size 40 --visual rope-swin-tiny --text roberta \
  --eval --resume kitti-2/colab-esi/ckpt_epoch_best_0.pth
# repeat for colab-base
```

### 7. Per-expression verdict (NOT pooled COMBINED)
```python
import csv, statistics
def per_expr_hota(p):
    rows=[r for r in csv.DictReader(open(p)) if r['seq']!='COMBINED']
    return statistics.mean(float(r['HOTA___AUC']) for r in rows)*100, float([r for r in csv.DictReader(open(p)) if r['seq']=='COMBINED'][0]['HOTA___AUC'])*100
for tag in ['colab-base','colab-esi']:
    m,c=per_expr_hota(f'retest-kitti-2/{tag}/results/pedestrian_detailed.csv')
    print(f'{tag}: per-expression mean HOTA {m:.2f} | pooled COMBINED {c:.2f}')
# GATE: esi per-expression mean > base per-expression mean
```

## Then Stage 4
Re-run ESI training with `--n-cf 3 --lambda-cf 1.0 --cf-loss margin --cf-margin 0.5
--cf-json <kitti2 counterfactuals>` stacked on `--esi-enabled` → does ESI+CFL beat each alone (per-expression)?

## Notes / risks
- Single GPU (`--nproc_per_node=1`); batch-7 fits 16 GB with ESI (local ESI peak was ~3.8 GB at batch-1).
- Caption JSON keys: train=GT obj ids, eval=tracker ids (already generated correctly).
- If deps fight Colab's torch, the only hard requirement is a working CUDA torch + transformers + timm/einops; the model is otherwise self-contained.
