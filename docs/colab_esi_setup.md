# Colab setup — ESI+HMSI training (Stage 3), streamlined

Validated on a Colab **T4 (16 GB)**, driven cell-by-cell. batch-7 fits in ~4.5 GB.
Captioning is done locally; only annotations + images travel. Evaluate
**per-expression** (NOT pooled COMBINED — a pooling artifact already burned us).

## Data on Google Drive (you upload once, share links)
Two archives (gdown pulls them by file ID — share as "anyone with link"):
- **images**: a 7z of `*/KITTI/training/image_02/<video>/*.png` for videos 0000-0020.
  NOTE: a **Refer-KITTI V1** image archive works fine for V2 — KITTI frames are shared;
  only annotations differ. Confirm it has all 21 video folders.
- **core tarball** (`flexhook_core.tar.gz`, ~137 M, built locally): V2 `labels.json`,
  `expression/`, `gt_template_gen/`, `gt_template/`, `pretrained/swin_*`,
  `tracker_outputs/Temp-NeuralSORT-kitti2/`, `cf_data/captions_{train,eval}.json`.
roberta-base is NOT in either — it auto-downloads from HuggingFace on Colab.

Build the core tarball locally (includes the qwen CF negatives for the CF screens):
```bash
tar -czhf cf_data/flexhook_core.tar.gz \
  pretrained/swin_rope_mixed_tiny_patch4_window7_224 \
  datasets/refer-kitti-v2/{labels.json,expression,gt_template_gen,gt_template} \
  tracker_outputs/Temp-NeuralSORT-kitti2 cf_data/captions_train.json cf_data/captions_eval.json \
  cf_data/counterfactuals-kitti2-qwen.json
```

## Cells

### 0. Set Runtime → GPU (T4) FIRST, then:
```bash
!git clone https://github.com/Seanachan/FlexHook-CF.git
%cd /content/FlexHook-CF
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 1. Deps (one shot — all of them)
```bash
!pip -q install -r requirements-colab.txt
!python -c "import torch,transformers;print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

### 2. Data — gdown both archives, extract (set your file IDs)
```python
import gdown, os
IMG_ID  = "1QmY9nXA-WmBOF44xmTBvOeFrQWLYCHMl"   # images 7z
CORE_ID = "1aFssnRBYsTYWALtgZombwNmEYguk55F2"   # flexhook_core.tar.gz
gdown.download(id=IMG_ID,  output="/content/imgs.7z", quiet=False)
gdown.download(id=CORE_ID, output="/content/core.tar.gz", quiet=False)
!apt-get -qq install -y p7zip-full >/dev/null
# images: extract image_02, symlink into the V2 path (frames are shared V1/V2)
!cd /content && 7z x imgs.7z "*/KITTI/training/image_02/*" -o/content/imgs -y >/dev/null
imgdir = [r for r,_,f in os.walk('/content/imgs') if r.endswith('image_02')][0]
os.makedirs('/content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training', exist_ok=True)
!ln -sfn "$imgdir" /content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training/image_02
# core: V2 annotations + swin + tracker + captions
!tar -xzf /content/core.tar.gz -C /content/FlexHook-CF
print('image videos:', len(os.listdir('/content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training/image_02')))
```

### 3. roberta-base from HuggingFace
```python
from transformers import AutoTokenizer, AutoModel
p='/content/FlexHook-CF/pretrained/roberta-base'
AutoTokenizer.from_pretrained('roberta-base').save_pretrained(p)
AutoModel.from_pretrained('roberta-base').save_pretrained(p)
```

### 4. Smoke tests (gate — must be green before training)
```bash
%cd /content/FlexHook-CF
!python tools/smoke_test_cf.py | tail -1
!python tools/smoke_test_esi.py | tail -1   # expect 23 passed
```

### 5. Drive checkpoint-sync (survives runtime disconnect)
```python
from google.colab import drive; drive.mount('/content/drive')
import os; DST='/content/drive/MyDrive/flexhook/ckpts'; os.makedirs(DST, exist_ok=True)
!nohup bash -c 'while true; do rsync -a /content/FlexHook-CF/kitti-2/ '"$DST"'/kitti-2/ 2>/dev/null; cp /content/*.log '"$DST"'/ 2>/dev/null; sleep 600; done' >/content/cksync.log 2>&1 &
```

### 6. Train (background nohup; survives bridge drops) — baseline then ESI
```bash
%cd /content/FlexHook-CF
# BASELINE
!OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29721 main.py \
 --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/colab-base \
 --batch-size 7 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
 --freeze-text --freeze-visual > /content/base.log 2>&1 &
# ESI (after baseline finishes, or in a later session)
!OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29722 main.py \
 --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/colab-esi \
 --batch-size 7 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
 --freeze-text --freeze-visual \
 --esi-enabled --esi-cap-train cf_data/captions_train.json --esi-cap-eval cf_data/captions_eval.json \
 > /content/esi.log 2>&1 &
```
Tail: `!grep -E "Train: \[|Error|out of memory" /content/base.log | tail -3`
(~30 min/epoch on T4, 1746 iters/epoch at batch-7, 20 epochs ≈ 10 h.)

### 7. Resume after a runtime disconnect
Re-run cells 0-5, then restore checkpoints from Drive and relaunch with the SAME
`--output` (AUTO_RESUME continues from the last epoch):
```bash
!mkdir -p kitti-2 && rsync -a /content/drive/MyDrive/flexhook/ckpts/kitti-2/ kitti-2/
# then re-run the matching train cell in §6
```

### 8. Eval (per-expression) — each best ckpt
```bash
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29731 main.py \
 --cfg configs/infer/kitti2.yaml --track-root tracker_outputs/Temp-NeuralSORT-kitti2 --data-path 12 \
 --output retest-kitti-2/colab-esi --val-batch-size 40 --visual rope-swin-tiny --text roberta \
 --eval --resume kitti-2/colab-esi/ckpt_epoch_best_0.pth
 # (ESI eval also needs --esi-enabled --esi-cap-train ... --esi-cap-eval ...)
```
```python
import csv, statistics
def hota(p):
    rows=[r for r in csv.DictReader(open(p)) if r['seq']!='COMBINED']
    comb=[r for r in csv.DictReader(open(p)) if r['seq']=='COMBINED'][0]
    return statistics.mean(float(r['HOTA___AUC']) for r in rows)*100, float(comb['HOTA___AUC'])*100
for tag in ['colab-base','colab-esi']:
    m,c=hota(f'retest-kitti-2/{tag}/results/pedestrian_detailed.csv')
    print(f'{tag}: per-expression mean {m:.2f} | pooled COMBINED {c:.2f}')
# GATE: esi per-expression mean > base per-expression mean
```

## 9. CF screens — A0 re-baseline + A1 loss round 2 (plan 2026-06-11)

The local 8GB/batch-1 ablation (docs/results-cf-ablation.md) is NOT comparable to this
regime. Re-anchor first (A0), then screen (A1). All runs single-seed; multi-seed only on
the final winner. Common stem (same as §6 baseline cell, change `--output` + append):

```bash
CF="--n-cf 3 --cf-json cf_data/counterfactuals-kitti2-qwen.json"
# A0-1  vanilla re-baseline = §6 BASELINE cell (kitti-2/colab-base) — reuse if already run
# A0-2  margin(0.5) re-anchor (local winner; does +0.29 survive the regime jump?)
--output kitti-2/cf-m05        $CF --lambda-cf 1.0 --cf-loss margin --cf-margin 0.5
# A1-1  margin sweep — recover push-loss AssA without DetA tax
--output kitti-2/cf-m03        $CF --lambda-cf 1.0 --cf-loss margin --cf-margin 0.3
--output kitti-2/cf-m04        $CF --lambda-cf 1.0 --cf-loss margin --cf-margin 0.4
# A1-2  max-over-CF-slots (VSE++ hardest-only; needs commit with --cf-loss-agg)
--output kitti-2/cf-m05-max    $CF --lambda-cf 1.0 --cf-loss margin --cf-margin 0.5 --cf-loss-agg max
```
Eval each via §8 with the matching `--output retest-kitti-2/<tag>` + `--resume
kitti-2/<tag>/ckpt_epoch_best_0.pth` (CF runs need NO extra eval flags — CF is
train-time only). Keep rule: HOTA > cf-m05 AND DetA ≥ colab-base. Record
HOTA/DetA/AssA + val F1 per run in docs/results-cf-ablation.md (new Colab section).

## Notes
- Single GPU (`--nproc_per_node=1`); batch-7 ≈ 4.5 GB on T4 (plenty of headroom).
- ESI eval MUST pass `--esi-enabled` + caption JSONs (captions feed the matcher at inference).
- Stage 4: add `--n-cf 3 --lambda-cf 1.0 --cf-loss margin --cf-margin 0.5 --cf-json <kitti2 cf json>` on top of the ESI train cell.
- `--cf-loss-agg {mean|max}` (default mean) added 2026-06-11; smoke_test_cf.py covers it (61 asserts).
