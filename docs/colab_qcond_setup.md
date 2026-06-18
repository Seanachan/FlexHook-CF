# Colab setup — QCOND-residual UNFROZEN (paper regime), the make-or-break test

Frozen RK-V2 batch-1 A/B already showed **qcond-residual = +1.07 per-expr HOTA**
(clean win on DetA+AssA+HOTA, color collapse fixed). But frozen is the off-paper
8 GB regime; ESI won frozen (+1.92) then **inverted unfrozen (−1.81)**. This screen
answers the only question that matters for a paper: **does the residual-QCOND gain
survive a trainable backbone (unfrozen, batch-14 = paper's 2 GPU × 7)?**

Two runs:
1. **qcond-residual unfrozen** — make-or-break method test.
2. **qcond-residual + CF-push unfrozen** — the COAL thesis: a query-conditioned
   obj_f finally gives the counterfactual push-loss a per-query handle (vanilla
   FlexHook's text-blind obj_f is exactly why CF push could only suppress the
   shared head = the DetA tax). Tests whether query-conditioning turns CF from
   catastrophic (−8.41 on ESI) into positive.

Needs a **big GPU** (A100 / L4 / Blackwell). Unfrozen batch-14 ≈ paper effective
batch; on a T4 (16 GB) it will OOM — drop to `--batch-size 7` and note the
smaller effective batch (not paper-comparable). Evaluate **per-expression**.

## Reference numbers to beat (RK-V2, per-expression mean HOTA)
| run | HOTA | DetA | AssA | regime |
|---|---|---|---|---|
| unfrozen baseline | **39.08** | 31.54 | 52.23 | the gate for run 1 |
| frozen baseline | 37.18 | 29.95 | 49.55 | — |
| frozen qcond-residual | 38.25 | 30.33 | 51.88 | the frozen win |
| frozen ESI (won) → unfrozen ESI (lost) | 40.06 → **37.27** | — | — | the inversion to avoid |

**GATE run 1:** qcond-residual unfrozen per-expr HOTA **> 39.08** → QCOND transfers,
method alive → paper. ≤ 39.08 (esp. an ESI-style inversion) → frozen-only technique,
diagnostic paper. **GATE run 2:** residual+CF ≥ residual → query-conditioning
unlocked CF (COAL thesis confirmed); < residual → CF still harmful even with the
handle.

## Cells

### 0. Runtime → GPU (A100/L4), then clone the **qcond** branch
```bash
!git clone -b qcond https://github.com/Seanachan/FlexHook-CF.git
%cd /content/FlexHook-CF
!git log --oneline -1   # expect: feat(qcond): query-conditioned object representation
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 1. Deps
```bash
!pip -q install -r requirements-colab.txt
!python -c "import torch,transformers;print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

### 2. Data — gdown images + core tarball (SAME archives as ESI screen; core already
holds swin, V2 annotations, tracker outputs, and counterfactuals-kitti2-qwen.json)
```python
import gdown, os
IMG_ID  = "1QmY9nXA-WmBOF44xmTBvOeFrQWLYCHMl"   # images 7z
CORE_ID = "1aFssnRBYsTYWALtgZombwNmEYguk55F2"   # flexhook_core.tar.gz
gdown.download(id=IMG_ID,  output="/content/imgs.7z", quiet=False)
gdown.download(id=CORE_ID, output="/content/core.tar.gz", quiet=False)
!apt-get -qq install -y p7zip-full >/dev/null
!cd /content && 7z x imgs.7z "*/KITTI/training/image_02/*" -o/content/imgs -y >/dev/null
imgdir = [r for r,_,f in os.walk('/content/imgs') if r.endswith('image_02')][0]
os.makedirs('/content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training', exist_ok=True)
!ln -sfn "$imgdir" /content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training/image_02
!tar -xzf /content/core.tar.gz -C /content/FlexHook-CF
# TrackEval reads seq length from the V1 path -> symlink it to the same frames
os.makedirs('/content/FlexHook-CF/datasets/refer-kitti/KITTI/training', exist_ok=True)
!ln -sfn "$imgdir" /content/FlexHook-CF/datasets/refer-kitti/KITTI/training/image_02
print('image videos:', len(os.listdir('/content/FlexHook-CF/datasets/refer-kitti-v2/KITTI/training/image_02')))
```

### 3. roberta-base from HuggingFace
```python
from transformers import AutoTokenizer, AutoModel
p='/content/FlexHook-CF/pretrained/roberta-base'
AutoTokenizer.from_pretrained('roberta-base').save_pretrained(p)
AutoModel.from_pretrained('roberta-base').save_pretrained(p)
```

### 4. Smoke gate (must be green)
```bash
!python tools/smoke_test_qcond.py | tail -1   # expect: 18 passed, 0 failed
!python tools/smoke_test_cf.py   | tail -1     # CF path still intact for run 2
```

### 5. Drive checkpoint-sync (survives disconnect)
```python
from google.colab import drive; drive.mount('/content/drive')
import os; DST='/content/drive/MyDrive/flexhook/ckpts'; os.makedirs(DST, exist_ok=True)
!nohup bash -c 'while true; do rsync -a /content/FlexHook-CF/kitti-2/ '"$DST"'/kitti-2/ 2>/dev/null; cp /content/*.log '"$DST"'/ 2>/dev/null; sleep 600; done' >/content/cksync.log 2>&1 &
```

### 6. Train (UNFROZEN = NO --freeze flags; batch-14; 16 dataloader workers)
```bash
%cd /content/FlexHook-CF
# RUN 1 — qcond-residual unfrozen
!OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29741 main.py \
 --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/qcond-residual-unfrozen \
 --batch-size 14 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
 --qcond-residual --opts DATA.NUM_WORKERS 16 \
 > /content/qres.log 2>&1 &

# RUN 2 — qcond-residual + CF-push (COAL thesis); run AFTER run 1 finishes
!OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup \
 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29742 main.py \
 --cfg configs/train/train-kitti2.yaml --data-path 12 --output kitti-2/qcond-residual-cf \
 --batch-size 14 --val-batch-size 40 --visual rope-swin-tiny --text roberta --pretrained src \
 --qcond-residual \
 --n-cf 3 --lambda-cf 1.0 --cf-loss push --cf-json cf_data/counterfactuals-kitti2-qwen.json \
 --opts DATA.NUM_WORKERS 16 \
 > /content/qrescf.log 2>&1 &
```
Tail: `!grep -E "Train: \[|Error|out of memory" /content/qres.log | tail -3`
(~3 h/run on a Blackwell/A100. If OOM: `--batch-size 7`, note non-paper effective batch.)

### 7. Eval (per-expression) — BOTH need `--qcond-residual` so the model build matches
the checkpoint. CF is train-time only → run 2 eval needs NO CF flags.
```bash
# RUN 1 eval
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29751 main.py \
 --cfg configs/infer/kitti2.yaml --track-root tracker_outputs/Temp-NeuralSORT-kitti2 --data-path 12 \
 --output retest-kitti-2/qcond-residual-unfrozen --val-batch-size 40 --visual rope-swin-tiny --text roberta \
 --eval --resume kitti-2/qcond-residual-unfrozen/ckpt_epoch_best_0.pth --qcond-residual
# RUN 2 eval (same, swap the two paths; still --qcond-residual, no CF flags)
!OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master-port 29752 main.py \
 --cfg configs/infer/kitti2.yaml --track-root tracker_outputs/Temp-NeuralSORT-kitti2 --data-path 12 \
 --output retest-kitti-2/qcond-residual-cf --val-batch-size 40 --visual rope-swin-tiny --text roberta \
 --eval --resume kitti-2/qcond-residual-cf/ckpt_epoch_best_0.pth --qcond-residual
```
```python
import csv, statistics
def hota(p):
    rows=[r for r in csv.DictReader(open(p)) if r['seq']!='COMBINED']
    comb=[r for r in csv.DictReader(open(p)) if r['seq']=='COMBINED'][0]
    mean=lambda k: statistics.mean(float(r[k]) for r in rows)*100
    return mean('HOTA___AUC'), mean('DetA___AUC'), mean('AssA___AUC'), float(comb['HOTA___AUC'])*100
for tag in ['qcond-residual-unfrozen','qcond-residual-cf']:
    h,d,a,c=hota(f'retest-kitti-2/{tag}/results/pedestrian_detailed.csv')
    print(f'{tag}: per-expr HOTA {h:.2f} DetA {d:.2f} AssA {a:.2f} | pooled {c:.2f}')
# GATE 1: qcond-residual-unfrozen per-expr HOTA > 39.08 (unfrozen baseline)
# GATE 2: qcond-residual-cf >= qcond-residual-unfrozen  (CF unlocked?)
```

### 8. Pull the two CSVs back to local for the 3-way cmp
Download `retest-kitti-2/{qcond-residual-unfrozen,qcond-residual-cf}/results/pedestrian_detailed.csv`
(or rsync from Drive), then locally:
```bash
python tools/cmp_perexpr.py retest-kitti-2/colab-base-unfrozen/results/pedestrian_detailed.csv \
       retest-kitti-2/qcond-residual-unfrozen/results/pedestrian_detailed.csv base-unfrozen qcond-res-unfrozen
```

## Notes
- The unfrozen baseline ckpt/eval already exist (`retest-kitti-2/colab-base-unfrozen`,
  per-expr 39.08) — that is the gate; no need to re-run it.
- Why QCOND should transfer where ESI didn't: QCOND injects **pure text conditioning**
  (no caption), learned end-to-end, available identically at train and eval — none of
  ESI's GT-box(train)/tracker-box(eval) caption domain gap. If it still inverts, the
  injected-prior family is dead unfrozen and the paper is the diagnostic.
