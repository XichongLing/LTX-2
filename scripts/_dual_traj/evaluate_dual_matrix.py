"""Reproduce the resolved Anna/banner/kitchen evaluation of dual_matrix."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DISK = Path('/media/xichong-drz-inokko/4715db97-9866-4ec1-9eab-1f836151d77b1')
MATRIX = DISK / 'dual_matrix'
RUNS = Path('/home/xichong-drz-inokko/experimental/Image_seq_transl/_draft_runs/dual_traj')
DATA = Path('/home/xichong-drz-inokko/experimental/data/dual_run')
ANNA = Path('/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/anna_stage2_vcond/intermediates')


def main():
    root = MATRIX / 'evaluation'
    root.mkdir(exist_ok=True)
    source = root / 'anna_source_first129_lossless.mkv'
    if not source.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(ANNA / 'source_video.mp4'),
                        '-map', '0:v:0', '-frames:v', '129', '-c:v', 'ffv1', str(source)], check=True)
    cases = [
        ('anna', source, ANNA / 'flux/translated_first_frame.png',
         RUNS / 'anna_stage_matrix/3b_multiscale_girl_dual_left_two_thirds/masks/left_two_thirds_864_1280x768.png',
         'woman', ['anna_image.mp4', 'anna_video.mp4', 'anna_dual.mp4'],
         ['anna_stage_matrix/3c_multiscale_image_branch_iclora_on', 'anna_stage_matrix/1_single_scale_video_branch',
          'anna_stage_matrix/3d_multiscale_girl_dual_left_two_thirds_kvnone_iclora_off']),
        ('banner', DATA / 'banner/banner_source_2.mp4', DATA / 'banner/banner_reference_2.png',
         RUNS / 'banner_i2_v2/masks/lower_half_1280x768.png', 'person on stage',
         ['banner_image.mp4', 'banner_video.mp4', 'banner_dual.mp4'],
         ['banner_i2_v2/image_branch', 'banner_i2_v2/video_branch', 'banner_i2_v2/dual_partition_lower_half_kv_none']),
        ('kitchen', DATA / 'kitchen/kitchen_source_3.mp4', DATA / 'kitchen/kitchen_reference_2.png',
         RUNS / 'kitchen_i2_v3/masks/left_half_1280x768.png', 'man',
         ['kitchen_image_branch.mp4', 'kitchen_video_branch.mp4', 'kitchen_dual.mp4'],
         ['kitchen_i2_v3/image_branch', 'kitchen_i2_v2/video_branch', 'kitchen_i2_v3/dual_partition_left_half_iclora_off_kv_none']),
    ]
    for name, src, ref, mask, concept, videos, originals in cases:
        dest = root / ('banner_person' if name == 'banner' else name)
        dest.mkdir(exist_ok=True)
        command = [sys.executable, str(REPO / 'scripts/_dual_traj/evaluate_trio.py'),
                   '--source-video', str(src), '--reference-image', str(ref), '--routing-mask', str(mask),
                   '--subject-concept', concept, '--output-dir', str(dest)]
        for label, video in zip(('image', 'video', 'dual'), videos):
            command += [f'--{label}-video', str(MATRIX / video)]
        provenance = {label: dict(original=str(RUNS / path), command=(RUNS / path / 'command.txt').read_text())
                      for label, path in zip(('image', 'video', 'dual'), originals)}
        provenance['evaluation_command'] = command
        provenance['source_alignment'] = ('Lossless FFV1 copy of original source frames 0..128; original has 130 frames. '
                                          'Original: ' + str(ANNA / 'source_video.mp4')) if name == 'anna' else 'Full source; no temporal resampling'
        provenance['comparison_note'] = 'Evaluate selected files as provided. Branch settings/Stage-1 caches may differ; consult original commands.'
        (dest / 'selected_runs.json').write_text(json.dumps(provenance, indent=2))
        (dest / 'command.txt').write_text(shlex.join(command) + '\n')
        if (dest / 'evaluation_manifest.json').exists() and (dest / 'summary.json').exists():
            print(f'Reusing completed evaluation: {name}', flush=True)
            continue
        print(f'Evaluating {name}', flush=True)
        with (dest / 'run.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO)
    rows = []
    for name, *_ in cases:
        dest = root / ('banner_person' if name == 'banner' else name)
        for row in json.loads((dest / 'summary.json').read_text()):
            rows.append(dict(case=name, **row))
    (root / 'combined_summary.json').write_text(json.dumps(rows, indent=2))
    import csv
    with (root / 'combined_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# dual_matrix evaluation', '', 'Frame-17-onward means; full-frame results and standard deviations are in combined_summary.csv.',
             '', '| Case | Branch | Appearance margin | Outside flow error (px) | Source silhouette IoU |',
             '|---|---|---:|---:|---:|']
    for name, *_ in cases:
        for label in ('image', 'video', 'dual'):
            values = {r['metric']: r['mean'] for r in rows if r['case'] == name and r['branch'] == label and r['scope'] == 'frame17_onward'}
            rendered = [('undefined' if values[k] is None else f'{values[k]:.4f}') for k in ('appearance_margin', 'flow_error', 'silhouette_iou')]
            lines.append(f'| {name} | {label} | ' + ' | '.join(rendered) + ' |')
    lines += ['', '## Sample and segmentation audit', '']
    for name, *_ in cases:
        dest = root / ('banner_person' if name == 'banner' else name)
        with (dest / 'per_frame.csv').open() as f:
            per_frame = list(csv.DictReader(f))
        for label in ('image', 'video', 'dual'):
            subset = [r for r in per_frame if r['branch'] == label]
            empty = sum(r['output_empty'] == 'True' for r in subset)
            lines.append(f'- {name}/{label}: {len(subset)} frames; output subject masks empty in {empty} frames. ' +
                         ('Interpret silhouette IoU cautiously because of missed detections.' if empty else ''))
    lines += ['', '## Interpretation', '',
              '- These are the selected files, not a fully controlled ablation. Anna image/dual use multiscale Stage 1, while its video baseline uses single scale.',
              '- All three selected dual runs use KV mode none; Anna/kitchen disable edit-branch IC-LoRA, banner enables it. Original commands are preserved in each selected_runs.json.',
              '- Routing masks are static: Anna left 864/1280 pixels, kitchen left half, banner lower half. Appearance includes background inside those regions.',
              '- Anna source was losslessly truncated to frames 0..128 to match the inference prefix; other sources use their full 81 frames.',
              '- Banner uses person on stage. The initial woman on stage evaluation remains in banner/; it failed to detect the video-branch performer and its IoU is unreliable. Use banner_person/ for the final results.',
              '- DINO responds to pose/shape as well as appearance. RAFT compares estimated flows. Per-frame std is not an uncertainty estimate over independent seeds.',
              '', 'Reports: [Anna](anna/report.md), [Banner](banner_person/report.md), [Kitchen](kitchen/report.md).']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'All evaluations complete: {root}', flush=True)


if __name__ == '__main__':
    main()
