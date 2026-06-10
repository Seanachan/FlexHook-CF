#!/usr/bin/env python3
"""
Offline trajectory-caption generator for FlexHook-CF ESI (Explicit Semantic Injection).

Mirrors tools/gen_counterfactuals.py (resumable checkpointed loop, per-item try/except)
but instead crops each tracked object's region from the frame images and asks a local
ollama VLM for one factual caption (color / type / motion / location).

Two modes, because the trainer and the evaluator iterate DIFFERENT trajectories:
  --mode eval   : tracker trajectories (tracker_outputs/<ROOT>/<video>/{car,pedestrian}/predict.txt)
                  keyed f'{video}_{tracker_id}'  (pedestrian id += max(car_ids); image = frame-1)
  --mode train  : GT trajectories from labels.json  (added in Stage 1; key f'{video}_{gt_obj_id}')

Output JSON: { "f'{video}_{obj}'": "<caption string>", ... }  -- keys match
data/mydataloader.py::__getitem__ `f'{video}_{obj}'` (no suffix) for direct lookup.

VLM call: ollama POST /api/generate with base64 `images` (see _ollama_generate).
Pure stdlib + Pillow + numpy.
"""

import os
import io
import json
import base64
import argparse
import urllib.request

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Ollama VLM backend
# --------------------------------------------------------------------------- #
_VLM_PROMPT = (
    "These are cropped images of the SAME tracked object across a few frames of a "
    "street/driving scene. Identify its attributes. Use an empty string for any "
    "attribute you cannot see; do not guess. "
    "Return STRICT JSON with exactly these keys: "
    '{"color": "", "type": "", "motion": "", "position": ""}  where '
    "type is one of car/van/truck/bus/pedestrian/cyclist, "
    "motion is one of moving/parked/turning/stopping/walking, "
    "position is one of left/right/front."
)


def _assemble_caption(d):
    """Assemble a tidy caption string from a {color,type,motion,position} dict."""
    color = str(d.get('color', '')).strip()
    typ = str(d.get('type') or d.get('object_type') or '').strip()
    motion = str(d.get('motion', '')).strip()
    pos = str(d.get('position') or d.get('location') or '').strip()
    parts = [p for p in [color, motion, typ] if p]          # "silver moving car"
    cap = ' '.join(parts)
    if pos:
        cap = f'{cap} on the {pos}' if cap else f'object on the {pos}'
    return cap.strip()


def _ollama_generate(prompt, images_b64, model, host=None, timeout=180):
    """POST to ollama /api/generate with base64 images; return the text response."""
    host = host or os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    if not host.startswith('http'):
        host = 'http://' + host
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'images': images_b64,
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.2},
    }).encode()
    req = urllib.request.Request(
        host.rstrip('/') + '/api/generate',
        data=payload, headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())['response']


def _parse_caption(text):
    """Extract a tidy caption string from the VLM response (structured dict or prose)."""
    text = (text or '').strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # structured attribute form (preferred): {color,type,motion,position}
            if any(k in obj for k in ('color', 'type', 'object_type', 'motion', 'position', 'location')):
                cap = _assemble_caption(obj)
                if cap:
                    return cap
            for k in ('caption', 'description', 'text'):
                if obj.get(k):
                    return str(obj[k]).strip().strip('"').strip()
        if isinstance(obj, str):
            return obj.strip()
    except Exception:
        pass
    # fallback: strip code fences / quotes, take first line
    t = text.replace('```json', '').replace('```', '').strip().strip('"')
    return t.split('\n')[0].strip()


# --------------------------------------------------------------------------- #
# Image cropping (upscale tiny KITTI boxes so the VLM can read them)
# --------------------------------------------------------------------------- #
def crop_box(img, x, y, w, h, margin=0.5, min_side=224):
    """Crop [x,y,w,h] (top-left MOT box) with context margin, upscale to >=min_side."""
    mx, my = w * margin, h * margin
    x1 = max(0, int(x - mx)); y1 = max(0, int(y - my))
    x2 = min(img.width, int(x + w + mx)); y2 = min(img.height, int(y + h + my))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    s = max(1.0, min_side / max(1, min(cw, ch)))
    if s > 1.0:
        crop = crop.resize((int(cw * s), int(ch * s)), Image.LANCZOS)
    return crop


def to_b64(img):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def pick_frames(frame_ids, k):
    """Pick up to k evenly-spaced frames across the trajectory."""
    n = len(frame_ids)
    if n <= k:
        return list(frame_ids)
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [frame_ids[i] for i in idx]


# --------------------------------------------------------------------------- #
# Trajectory collection  (eval = tracker; train = GT/labels.json, Stage 1)
# --------------------------------------------------------------------------- #
def collect_tracker_trajectories(track_root, img_root, videos):
    """eval mode: read predict.txt per video; key f'{video}_{id}', ped id += max(car_ids)."""
    trajs = {}  # key -> list of (frame_id, x, y, w, h)
    vids = videos if videos else sorted(
        d for d in os.listdir(track_root)
        if os.path.isdir(os.path.join(track_root, d))
    )
    for video in vids:
        max_car_id = 0
        for cls in ('car', 'pedestrian'):
            pt = os.path.join(track_root, video, cls, 'predict.txt')
            if not os.path.exists(pt):
                continue
            arr = np.loadtxt(pt, delimiter=',')
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.size == 0:
                continue
            ids = np.unique(arr[:, 1]).astype(int)
            cur_max = int(ids.max()) if len(ids) else 0
            for oid in ids:
                rows = arr[arr[:, 1] == oid]
                key_id = int(oid) if cls == 'car' else int(oid) + max_car_id
                key = f'{video}_{key_id}'
                # frame-1 shift (TempRMOT) so crop matches the image the model sees
                trajs[key] = {
                    'video': video,
                    'boxes': [(int(r[0]) - 1, float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows],
                }
            if cls == 'car':
                max_car_id = cur_max
    return trajs


def caption_trajectory(traj, img_root, model, k_frames, host):
    """Crop k representative frames of a trajectory, send to VLM, return caption."""
    video = traj['video']
    boxes = traj['boxes']
    frame_ids = [b[0] for b in boxes]
    chosen = set(pick_frames(frame_ids, k_frames))
    imgs_b64 = []
    for (fid, x, y, w, h) in boxes:
        if fid not in chosen:
            continue
        ip = os.path.join(img_root, video, f'{fid:06d}.png')
        if not os.path.exists(ip):
            continue
        try:
            crop = crop_box(Image.open(ip), x, y, w, h)
        except Exception:
            crop = None
        if crop is not None:
            imgs_b64.append(to_b64(crop))
        if len(imgs_b64) >= k_frames:
            break
    if not imgs_b64:
        return None
    resp = _ollama_generate(_VLM_PROMPT, imgs_b64, model, host=host)
    return _parse_caption(resp)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='Generate VLM trajectory captions for FlexHook-CF ESI')
    ap.add_argument('--mode', choices=['eval', 'train'], default='eval')
    ap.add_argument('--track-root', help='tracker_outputs/<ROOT> (eval mode)')
    ap.add_argument('--img-root', required=True,
                    help='image dir, e.g. datasets/refer-kitti-v2/KITTI/training/image_02')
    ap.add_argument('--out', required=True, help='output captions json')
    ap.add_argument('--model', default='qwen2.5vl:7b', help='ollama VLM model')
    ap.add_argument('--frames', type=int, default=3, help='crops per trajectory')
    ap.add_argument('--videos', nargs='+', default=None, help='restrict to these videos')
    ap.add_argument('--limit', type=int, default=0, help='Stage-0: cap #trajectories (0=all)')
    args = ap.parse_args()
    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

    if args.mode == 'train':
        raise SystemExit('train mode (GT/labels.json) is added in Stage 1; use --mode eval for the Stage-0 gate.')
    if not args.track_root:
        raise SystemExit('--track-root required for --mode eval')

    trajs = collect_tracker_trajectories(args.track_root, args.img_root, args.videos)
    keys = sorted(trajs)
    if args.limit:
        keys = keys[:args.limit]
    print(f'collected {len(trajs)} trajectories ({args.mode}); captioning {len(keys)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    result = {}
    if os.path.exists(args.out):
        try:
            result = json.load(open(args.out))
            print(f'resuming: {len(result)} already captioned')
        except Exception:
            result = {}

    def _save():
        with open(args.out, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

    n_err = 0
    for i, key in enumerate(keys):
        if key in result:
            continue
        try:
            cap = caption_trajectory(trajs[key], args.img_root, args.model, args.frames, host)
        except Exception as e:
            n_err += 1
            if n_err <= 20 or n_err % 25 == 0:
                print(f'  [warn] {key} failed ({n_err}): {type(e).__name__}: {e}')
            continue
        if cap:
            result[key] = cap
            print(f'  {key}: {cap}')
        if (i + 1) % 25 == 0:
            _save()
    _save()
    print(f'wrote {args.out}: {len(result)} captions, {n_err} errors')


if __name__ == '__main__':
    main()
