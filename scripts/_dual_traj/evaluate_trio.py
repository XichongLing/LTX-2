#!/usr/bin/env python3
"""Regional DINOv2 appearance, RAFT motion, and SAM3 silhouette evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ltx_pipelines.correspondence_mask import load_file_mask
from ltx_pipelines.utils.media_io import resize_and_center_crop

SEG = Path('/media/xichong-drz-inokko/4715db97-9866-4ec1-9eab-1f836151d77b1/segmentation')
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
LABELS = ('image', 'video', 'dual')


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            h.update(str(file.relative_to(path)).encode())
        with file.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def metadata(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames', '-of', 'json', str(path)]))
    s = data['streams'][0]
    a, b = map(int, s['avg_frame_rate'].split('/'))
    return dict(width=int(s['width']), height=int(s['height']), fps=a / b,
                frames=int(s['nb_read_frames']))


def validate_alignment(metas):
    canvas = metas['image']
    for label, m in metas.items():
        if m['frames'] != canvas['frames'] or abs(m['fps'] - canvas['fps']) > 1e-5:
            raise ValueError(f'Temporal mismatch: {label}: {m}, expected {canvas}')
        if label != 'source' and (m['width'], m['height']) != (canvas['width'], canvas['height']):
            raise ValueError(f'Output resolution mismatch: {label}')
    return canvas


def aligned_image(frame, height, width):
    return resize_and_center_crop(torch.as_tensor(frame.copy()).float(), height, width)[0, :, 0] / 255


def frames(path, height, width, limit):
    cap = cv2.VideoCapture(str(path))
    try:
        for i in range(limit):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Failed to decode frame {i}: {path}')
            yield aligned_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), height, width)
    finally:
        cap.release()


def masks(path, count, height, width):
    # Loader returns (1,1,F,H,W); retain CPU pixel masks, never VAE token masks.
    return load_file_mask(path=path, num_frames=count, height=height, width=width)[0, 0]


def appearance_input(frame, mask):
    coords = torch.nonzero(mask > 0)
    if not len(coords):
        return None
    y0, x0 = coords.min(0).values.tolist()
    y1, x1 = (coords.max(0).values + 1).tolist()
    region = frame[:, y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1][None]
    region = region * m + MEAN * (1 - m)
    region = F.interpolate(region[None], (224, 224), mode='bicubic', align_corners=False)[0]
    return ((region - MEAN) / STD)[None]


def appearance_margin(output, reference, source):
    return float(output @ reference), float(output @ source), float(output @ reference - output @ source)


def silhouette_iou(source, output):
    source, output = source >= .5, output >= .5
    union = (source | output).sum().item()
    return float((source & output).sum()) / union if union else None


def resize_flow(flow, height, width):
    h, w = flow.shape[-2:]
    result = F.interpolate(flow, (height, width), mode='bilinear', align_corners=False)
    return result * result.new_tensor([width / w, height / h]).view(1, 2, 1, 1)


def flow_error(source, output, first_mask, next_mask):
    weights = 1 - torch.maximum(first_mask, next_mask)
    denominator = weights.sum()
    error = torch.linalg.vector_norm(output - source, dim=0)
    return (float((error * weights).sum() / denominator) if denominator > 0 else None,
            float(weights.mean()))


def cache_matches(data, sha, concept):
    return data.get('source_video_sha256') == sha and data.get('prompt_or_mask') == concept


def silhouettes(args, videos, hashes, metas, manifest):
    result = {}
    supplied_concepts = set()
    for label, video in videos.items():
        supplied = getattr(args, f'{label}_silhouette_mask')
        if supplied:
            candidate = supplied.parent / 'manifest.json'
            if candidate.exists():
                data = json.loads(candidate.read_text())
                if data.get('model') == 'sam3':
                    concept = data.get('prompt_or_mask')
                    if data.get('source_video_sha256') != hashes[label] or (args.subject_concept and concept != args.subject_concept):
                        raise ValueError(f'Stale supplied SAM3 silhouette: {supplied}')
                    if data.get('frame_count') != metas[label]['frames']:
                        raise ValueError(f'Supplied SAM3 frame count mismatch: {supplied}')
                    supplied_concepts.add(concept)
            result[label] = supplied
            continue
        if not args.subject_concept:
            raise ValueError(f'--subject-concept is required to generate {label} silhouettes')
        root = args.output_dir / 'sam3' / label
        meta_path = root / 'manifest.json'
        if root.exists():
            if not meta_path.exists() or not cache_matches(json.loads(meta_path.read_text()), hashes[label], args.subject_concept):
                raise ValueError(f'Stale/incomplete SAM3 cache: {root}; choose a new output directory')
        else:
            cmd = [str(args.sam3_python), str(args.sam3_tool), '--video', str(video),
                   '--text-prompt', args.subject_concept, '--output-dir', str(root), '--all-instances']
            manifest.setdefault('sam3_commands', []).append(cmd)
            with (args.output_dir / f'sam3_{label}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        pngs = sorted((root / 'masks').glob('*.png'))
        if len(pngs) != metas[label]['frames']:
            raise ValueError(f'SAM3 frame count mismatch: {root}')
        result[label] = root / 'masks'
    if args.subject_concept:
        supplied_concepts.add(args.subject_concept)
    if len(supplied_concepts) > 1:
        raise ValueError(f'Silhouette SAM3 concepts differ: {supplied_concepts}')
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    valid = [x for x in values if x is not None and np.isfinite(x)]
    return dict(mean=float(np.mean(valid)) if valid else None,
                std=float(np.std(valid)) if valid else None, n=len(valid))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-video', 'reference-image', 'routing-mask', 'image-video', 'video-video', 'dual-video', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--reference-mask', type=Path)
    p.add_argument('--subject-concept')
    for label in ('source', *LABELS):
        p.add_argument(f'--{label}-silhouette-mask', type=Path)
    p.add_argument('--dino-repository', type=Path, default=Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main')
    p.add_argument('--dino-checkpoint', type=Path, default=Path.home() / '.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth')
    p.add_argument('--raft-checkpoint', type=Path)
    p.add_argument('--sam3-python', type=Path, default=SEG / 'envs/sam3/bin/python')
    p.add_argument('--sam3-tool', type=Path, default=SEG / 'tools/segment_sam3_text_video.py')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-frames', type=int, help='Explicit smoke-test prefix; full-video metadata must still match')
    return p


def main():
    args = parser().parse_args()
    videos = {'source': args.source_video, **{label: getattr(args, label + '_video') for label in LABELS}}
    metas = {label: metadata(path) for label, path in videos.items()}
    canvas = validate_alignment(metas)
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError('--max-frames must be at least 2')
    n = min(canvas['frames'], args.max_frames or canvas['frames'])
    if n < 2:
        raise ValueError('At least two evaluation frames are required')
    for path in [args.routing_mask, *[getattr(args, f'{k}_silhouette_mask') for k in videos]]:
        if path and path.is_dir():
            count = len([p for p in path.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')])
            if count != canvas['frames']:
                raise ValueError(f'Mask directory has {count} frames, expected {canvas["frames"]}: {path}')
    h, w = canvas['height'], canvas['width']
    if min(h // 2, w // 2) < 128:
        raise ValueError('RAFT-Large requires half-resolution dimensions >=128 pixels')
    hashes = {label: fingerprint(path) for label, path in videos.items()}
    m = masks(args.routing_mask, n, h, w)
    rm = masks(args.reference_mask, 1, h, w)[0] if args.reference_mask else m[0]
    ref = aligned_image(np.asarray(Image.open(args.reference_image).convert('RGB')), h, w)
    if appearance_input(ref, rm) is None:
        raise ValueError('Reference appearance mask is empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(command=sys.argv, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    videos={k: dict(path=str(v), sha256=hashes[k], metadata=metas[k]) for k, v in videos.items()},
                    frames=list(range(n)), canvas=canvas, torch=torch.__version__,
                    preprocessing='inference resize_and_center_crop; shared soft pixel mask; masked bbox DINO 224',
                    reference_sha256=fingerprint(args.reference_image), routing_mask_sha256=fingerprint(args.routing_mask),
                    reference_mask_sha256=fingerprint(args.reference_mask) if args.reference_mask else None)
    sil_paths = silhouettes(args, videos, hashes, metas, manifest)
    manifest['silhouettes'] = {k: dict(path=str(v), sha256=fingerprint(v)) for k, v in sil_paths.items()}
    manifest['sam3_manifests'] = {k: json.loads((v.parent / 'manifest.json').read_text())
                                for k, v in sil_paths.items() if (v.parent / 'manifest.json').exists()}
    print('Loading DINOv2', flush=True)
    model = torch.hub.load(str(args.dino_repository), 'dinov2_vitb14_reg', source='local', weights=str(args.dino_checkpoint)).eval().to(args.device)
    manifest['dino'] = dict(model='dinov2_vitb14_reg', checkpoint_sha256=fingerprint(args.dino_checkpoint),
                            repository=str(args.dino_repository))
    def embed(frame, mask):
        inputs = appearance_input(frame, mask)
        if inputs is None:
            return None
        with torch.inference_mode():
            return F.normalize(model(inputs.to(args.device)).float(), dim=-1)[0].cpu()
    reference = embed(ref, rm)
    source_features = [embed(f, m[t]) for t, f in enumerate(frames(videos['source'], h, w, n))]
    source_sil = masks(sil_paths['source'], n, h, w)
    rows = []
    for label in LABELS:
        print(f'Appearance/silhouette: {label}', flush=True)
        sil = masks(sil_paths[label], n, h, w)
        for t, frame in enumerate(frames(videos[label], h, w, n)):
            out = embed(frame, m[t])
            a, b, margin = appearance_margin(out, reference, source_features[t]) if out is not None else (None, None, None)
            rows.append(dict(branch=label, frame=t, reference_cosine=a, source_cosine=b, appearance_margin=margin,
                             appearance_valid=out is not None, mask_coverage=float(m[t].mean()),
                             silhouette_iou=silhouette_iou(source_sil[t], sil[t]),
                             source_area=int((source_sil[t] >= .5).sum()), output_area=int((sil[t] >= .5).sum()),
                             source_empty=not bool((source_sil[t] >= .5).any()), output_empty=not bool((sil[t] >= .5).any())))
        del sil
    del embed, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    import torchvision
    weights = Raft_Large_Weights.DEFAULT
    if args.raft_checkpoint:
        model = raft_large(weights=None)
        model.load_state_dict(torch.load(args.raft_checkpoint, map_location='cpu', weights_only=True))
        cp = args.raft_checkpoint
    else:
        model = raft_large(weights=weights)
        cp = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
    model = model.eval().to(args.device)
    manifest['raft'] = dict(model='raft_large', weights=weights.name, checkpoint_sha256=fingerprint(cp),
                            torchvision=torchvision.__version__, resolution=[h // 2, w // 2], units='output pixels')
    def flow(first, second):
        pair = F.interpolate(torch.stack([first, second]), (h // 2, w // 2), mode='bilinear', align_corners=False)
        ph, pw = (-pair.shape[-2]) % 8, (-pair.shape[-1]) % 8
        pair = F.pad(pair, (0, pw, 0, ph), mode='replicate').to(args.device)
        first, second = weights.transforms()(pair[:1], pair[1:])
        with torch.inference_mode():
            small = model(first, second)[-1][..., :h // 2, :w // 2]
        return resize_flow(small, h, w)[0].cpu()
    # Cache source flow on disk; never retain a video worth of flows in GPU/host RAM.
    flow_dir = args.output_dir / 'source_flows'
    flow_dir.mkdir(exist_ok=True)
    stream = frames(videos['source'], h, w, n)
    prev = next(stream)
    for t, current in enumerate(stream):
        print(f'Source flow {t + 1}/{n - 1}', flush=True)
        torch.save(flow(prev, current), flow_dir / f'{t:06d}.pt')
        prev = current
    flow_rows = []
    for label in LABELS:
        stream = frames(videos[label], h, w, n)
        prev = next(stream)
        for t, current in enumerate(stream):
            src = torch.load(flow_dir / f'{t:06d}.pt', weights_only=True)
            error, coverage = flow_error(src, flow(prev, current), m[t], m[t + 1])
            flow_rows.append(dict(branch=label, frame=t, next_frame=t + 1, flow_error=error,
                                  outside_coverage=coverage, valid=error is not None))
            prev = current
        print(f'Output flow complete: {label}', flush=True)
    del model, flow
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = []
    for label in LABELS:
        for scope, start in [('all', 0), ('frame17_onward', 17)]:
            for metric, table in [('appearance_margin', rows), ('silhouette_iou', rows), ('flow_error', flow_rows)]:
                summary.append(dict(branch=label, scope=scope, metric=metric,
                                    **stats([r[metric] for r in table if r['branch'] == label and r['frame'] >= start])))
    write_csv(args.output_dir / 'per_frame.csv', rows)
    write_csv(args.output_dir / 'per_pair_flow.csv', flow_rows)
    write_csv(args.output_dir / 'summary.csv', summary)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (args.output_dir / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, (metric, table) in zip(axes, [('appearance_margin', rows), ('flow_error', flow_rows), ('silhouette_iou', rows)]):
        for label in LABELS:
            subset = [r for r in table if r['branch'] == label]
            ax.plot([r['frame'] for r in subset], [np.nan if r[metric] is None else r[metric] for r in subset], label=label)
        ax.set(xlabel='Frame', ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'metrics.png', dpi=150)
    plt.close(fig)
    lines = ['# Trio evaluation', '', '| Branch | Scope | Metric | Mean | Std | Valid samples |', '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['branch']} | {r['scope']} | {r['metric']} | {r['mean']} | {r['std']} | {r['n']} |")
    lines += ['', 'Hypotheses: image/dual appearance margin higher; video/dual outside flow error lower; video silhouette IoU higher.',
              'These hypotheses do not affect scores. DINO also responds to pose/shape. RAFT errors compare estimated flows, not ground truth.',
              'Both-empty silhouettes and empty appearance/flow supports are undefined and excluded from means.',
              'Frame-17 flow summaries include pairs whose first frame is >=17. Standard deviation is over frames, not independent experimental seeds.',
              'Masks supplied without SAM3 manifests are treated as user-provided silhouettes. See evaluation_manifest.json for provenance.',
              '', '![Metric timelines](metrics.png)']
    (args.output_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'Evaluation complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
