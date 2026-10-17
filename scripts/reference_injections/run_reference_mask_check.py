#!/usr/bin/env python3
"""
Segmentation quality check for LTX-2.3 IC-LoRA edit-anything pipeline.

Modes
-----
discover  — run one forward pass and print all (q_len, k_len) cross-attn pairs
            so you can confirm the text token sequence length.

sweep     — for every active text token, compute a per-frame heatmap and write
            a grid image at <work_dir>/sweep.png.  Slow but gives a full picture
            of which tokens drive spatial attention.

mask      — aggregate cross-attention for the given --word, generate binary mask
            frames and write <work_dir>/mask_<word>.mp4.

Usage examples
--------------
# Find text cross-attention call indices / k_len:
python run_reference_mask_check.py discover \
    --reference-video /path/to/input.mp4 \
    --prompt "..." --gemma-root /path/to/gemma

# Per-token heatmap sweep (writes sweep.png):
python run_reference_mask_check.py sweep \
    --reference-video /path/to/input.mp4 \
    --prompt "..." --gemma-root /path/to/gemma \
    --work-dir /path/to/out --word fountain

# Binary mask for a single word:
python run_reference_mask_check.py mask \
    --reference-video /path/to/input.mp4 \
    --prompt "Convert the fountain ..." --gemma-root /path/to/gemma \
    --word fountain --work-dir /path/to/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path setup — must happen before any LTX import
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
_SRCS = [
    REPO_ROOT / "packages" / "ltx-core" / "src",
    REPO_ROOT / "packages" / "ltx-pipelines" / "src",
]
for _p in _SRCS:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Local imports (no LTX yet)
# ---------------------------------------------------------------------------

import attn_capture as ac
import mask_from_attn as mf
import overlay as ov
from prompt_token_utils import infer_token_indices, inspect_prompt_tokens


# ---------------------------------------------------------------------------
# Stub: build pipeline
# The pipeline is expensive to construct (loads transformer, Gemma, LoRA).
# Build once and reuse across modes.
# ---------------------------------------------------------------------------

def _build_pipeline(args) -> object:
    """
    Construct ICLoraPipeline.  Adjust imports / paths if package layout changes.
    """
    from ltx_pipelines import ICLoraPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_pipelines.utils.args import _resolve_quantization

    lora_kwargs = {"sd_ops": LTXV_LORA_COMFY_RENAMING_MAP}

    # Convert "fp8-cast" string → QuantizationPolicy object (same as _PipelineArgumentParser does)
    _resolve_quantization(args)

    lora = LoraPathStrengthAndSDOps(
        path=str(Path(args.ic_lora_path).expanduser().resolve()),
        strength=1.0,
        **lora_kwargs,
    )

    return ICLoraPipeline(
        distilled_checkpoint_path=str(Path(args.distilled_checkpoint_path).expanduser().resolve()),
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).expanduser().resolve()),
        gemma_root=str(Path(args.gemma_root).expanduser().resolve()),
        loras=[lora],
        device=torch.device("cuda"),
        quantization=getattr(args, "quantization", None),
    )


# ---------------------------------------------------------------------------
# Stub: run one forward pass with attention capture
#
# IMPORTANT — grid_thw:
#   The transformer works in LATENT space.  VideoLatentShape.from_pixel_shape
#   uses floor division (h // 32, w // 32).  The pipeline enforces that h and w
#   are multiples of 32 (or 64 for 2-stage), so this matches the actual latent.
#   If you ever pass non-divisible dims, derive grid_thw from the actual
#   patchified sequence length instead:
#       seq_len = store.records[0].q_len          # 2*T*H*W for IC-LoRA
#       n_target = seq_len // 2
#       H_lat = height // 32   # may be off by 1 — verify against q_len
# ---------------------------------------------------------------------------

def _run_forward(pipeline, args, store: ac.AttentionStore):
    """
    Call pipeline inside capture() context and return (decoded_video, fps, pixel_hw, grid_thw).

    grid_thw is (T_lat, H_lat, W_lat) in latent space.
    pixel_hw is (H_px, W_px) of the pipeline output.
    """
    # blocks.py imports euler_denoising_loop by name, so we must patch it there
    import ltx_pipelines.utils.blocks as _blocks
    import ltx_pipelines.utils.samplers as _samplers

    _orig_loop = _samplers.euler_denoising_loop

    def _loop_with_step_tracking(sigmas, video_state, audio_state, stepper, transformer, denoiser):
        def _tracked_denoiser(t, vs, as_, sigmas, step_idx):
            store.mark_step()
            return denoiser(t, vs, as_, sigmas, step_idx)
        return _orig_loop(sigmas, video_state, audio_state, stepper, transformer, _tracked_denoiser)

    _blocks.euler_denoising_loop = _loop_with_step_tracking
    try:
        with ac.capture(store):
            _, _ = pipeline(
                prompt=args.prompt,
                seed=getattr(args, "seed", 42),
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=getattr(args, "frame_rate", 24.0),
                images=[],
                video_conditioning=[
                    (str(Path(args.reference_video).expanduser().resolve()),
                     getattr(args, "video_strength", 0.85)),
                ],
                video_cfg_scale=getattr(args, "video_cfg_scale", 2.0),
                skip_stage_2=True,
            )
    finally:
        _blocks.euler_denoising_loop = _orig_loop

    # Stage 1 generates at height//2 × width//2; stage 2 upsamples ×2.
    # With skip_stage_2=True the output stays at stage-1 resolution.
    # Derive the actual latent grid from the captured self-attention q_len
    # (self-attn has q_len == k_len == 2*T*H*W for IC-LoRA) — this is robust
    # against any internal resolution choice the pipeline may make.
    T_lat = (args.num_frames - 1) // 8 + 1
    stage1_h = args.height // 2
    stage1_w = args.width // 2
    H_lat = stage1_h // 32
    W_lat = stage1_w // 32
    grid_thw = (T_lat, H_lat, W_lat)

    # Override with observed self-attention q_len if available (discover mode)
    self_attn_q = None
    if store._discover_log:
        for e in store._discover_log:
            if e["q_len"] == e["k_len"] and e["q_len"] > 256:
                self_attn_q = e["q_len"]
                break
    elif store.records:
        # In capture mode: self-attention records won't be stored (k_len not in text_len_set),
        # so fall back to cross-attn q_len from first record
        self_attn_q = store.records[0].q_len   # = 2*T*H*W for IC-LoRA

    if self_attn_q is not None:
        import math
        n_target = self_attn_q // 2   # IC-LoRA halves: n_target = T*H*W
        hw = n_target // T_lat
        aspect = stage1_h / stage1_w
        W_lat = max(1, round(math.sqrt(hw / aspect)))
        H_lat = max(1, hw // W_lat)
        grid_thw = (T_lat, H_lat, W_lat)
    print(f"[INFO] grid_thw = {grid_thw}  (T={T_lat}, H={H_lat}, W={W_lat})", file=sys.stderr)

    fps = getattr(args, "frame_rate", 24.0)
    pixel_hw = (args.height, args.width)
    return fps, pixel_hw, grid_thw


# ---------------------------------------------------------------------------
# Token-index <-> context-column mapping
#
# `agg`'s k_len columns come from attn2's context (= video_encoding), which is
# NOT in the same order as the left-padded sequence that inspect_prompt_tokens()
# / infer_token_indices() index into. EmbeddingsProcessor.create_embeddings()
# right-pad-reorders the sequence before the connector
# (_compute_right_pad_order / _apply_right_pad_order in
# ltx_core/text_encoders/gemma/embeddings_processor.py). For left-padded input
# this is a cyclic rotation: the n_active active tokens move from
# [k_len - n_active, k_len) to [0, n_active), preserving relative order. So a
# tokenizer index `p` maps to context column `p - (k_len - n_active)`.
# ---------------------------------------------------------------------------

def _context_offset(prompt: str, gemma_root: str, k_len: int) -> int:
    tokens = inspect_prompt_tokens(prompt, gemma_root, max_length=k_len)
    n_active = sum(1 for t in tokens if t.active)
    return k_len - n_active


def _to_context_indices(token_indices: list[int], offset: int, k_len: int) -> list[int]:
    remapped = [i - offset for i in token_indices]
    valid = [i for i in remapped if 0 <= i < k_len]
    dropped = [i for i in remapped if not (0 <= i < k_len)]
    if dropped:
        print(f"[WARN] Dropped out-of-range context indices: {dropped}")
    return valid


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _run_discover(pipeline, args):
    store = ac.AttentionStore(text_len_set=set(range(1, 2048)), discover=True)
    store.reset()
    _run_forward(pipeline, args, store)

    out_path = Path(args.work_dir) / "discover.txt"
    import io, sys as _sys
    buf = io.StringIO()
    _orig_stdout = _sys.stdout
    _sys.stdout = buf
    store.print_discover_table()
    _sys.stdout = _orig_stdout
    table_text = buf.getvalue()
    print(table_text)
    out_path.write_text(table_text)
    print(f"[discover] Table saved to {out_path}")


def _run_sweep(pipeline, args):
    work_dir = Path(args.work_dir)

    store = ac.AttentionStore(
        text_len_set={getattr(args, "text_len", 256)},
        discover=False,
    )
    store.reset()
    fps, pixel_hw, grid_thw = _run_forward(pipeline, args, store)

    # Decode reference video frames for compositing
    ref_path = Path(args.reference_video).expanduser().resolve()
    frames_bgr, _ = ov.read_video_frames_bgr(ref_path)
    if not frames_bgr:
        import numpy as np
        print("[WARN] Could not read reference video; using blank frames.")
        H_px, W_px = pixel_hw
        frames_bgr = [np.zeros((H_px, W_px, 3), dtype="uint8")] * grid_thw[0]

    T_lat, H_lat, W_lat = grid_thw
    n_target = T_lat * H_lat * W_lat

    agg = mf.aggregate(store.records)   # (q_len, k_len)
    k_len = agg.shape[1]
    print(f"[sweep] k_len={k_len}  q_len={agg.shape[0]}  grid_thw={grid_thw}")

    tokens = inspect_prompt_tokens(args.prompt, args.gemma_root)
    active = [t for t in tokens if t.active]
    offset = k_len - len(active)
    print(f"[sweep] {len(active)} active tokens, context offset={offset} "
          f"(context_col = tokenizer_index - {offset}) — computing per-token heatmaps ...")

    import cv2
    import numpy as np

    sweep_dir = work_dir / "sweep_tokens"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    token_heatmaps = []
    token_labels = []
    frame_idx = 0   # use first latent frame for the grid
    ref_h, ref_w = frames_bgr[0].shape[:2] if frames_bgr else pixel_hw

    for pt in active:
        ctx_idx = pt.index - offset
        if not (0 <= ctx_idx < k_len):
            print(f"[WARN] token {pt.index}:{pt.text!r} -> context index {ctx_idx} "
                  f"out of range [0,{k_len}); skipping")
            continue
        vec = mf.token_map(agg, [ctx_idx])
        grid = mf.reshape_to_grid(vec, grid_thw, n_target_tokens=n_target)
        heat, _ = mf.to_pixel_mask(grid, pixel_hw, threshold="quantile", keep_frac=0.25)
        # Resize heat[frame_idx] to reference frame size for grid
        h_t = heat[frame_idx].unsqueeze(0).unsqueeze(0)
        h_t_up = torch.nn.functional.interpolate(h_t, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
        heat_np = h_t_up.squeeze().numpy()
        token_heatmaps.append(heat_np)
        token_labels.append(f"{pt.index}->{ctx_idx} {pt.text!r}")

        # Per-token overlay, saved separately from the combined sweep grid.
        comp = ov.overlay_frame(frames_bgr[0], heat_np)
        safe_text = "".join(c if c.isalnum() else "_" for c in pt.text).strip("_") or "tok"
        cv2.imwrite(str(sweep_dir / f"{pt.index:04d}_ctx{ctx_idx:04d}_{safe_text}.png"), comp)

    if not token_heatmaps:
        print("[WARN] No heatmaps generated — check text_len and k_len.")
        return

    token_heatmaps_arr = np.stack(token_heatmaps, axis=0)
    grid_out = work_dir / "sweep.png"
    ov.write_sweep_grid(frames_bgr[0], token_heatmaps_arr, token_labels, grid_out)
    print(f"[sweep] Wrote sweep grid to {grid_out}")
    print(f"[sweep] Wrote {len(token_heatmaps)} per-token overlays to {sweep_dir}/")

    if getattr(args, "word", None):
        _save_word_mask(agg, grid_thw, n_target, pixel_hw, frames_bgr, fps, args, offset=offset)


def _save_word_mask(agg, grid_thw, n_target, pixel_hw, frames_bgr, fps, args, offset: int | None = None):
    word = args.word
    try:
        token_indices = infer_token_indices(args.prompt, word, args.gemma_root)
    except ValueError as e:
        print(f"[WARN] {e}")
        return
    if not token_indices:
        print(f"[WARN] No token indices found for word '{word}'.")
        return

    k_len = agg.shape[1]
    if offset is None:
        offset = _context_offset(args.prompt, args.gemma_root, k_len)

    ctx_indices = _to_context_indices(token_indices, offset, k_len)
    if not ctx_indices:
        print(f"[WARN] Word '{word}' tokenizer indices {token_indices} all map "
              f"out of range [0,{k_len}) with offset={offset}.")
        return
    print(f"[mask] Word '{word}' → tokenizer indices {token_indices} → context indices {ctx_indices}")

    vec = mf.token_map(agg, ctx_indices)
    grid = mf.reshape_to_grid(vec, grid_thw, n_target_tokens=n_target)
    heat, mask = mf.to_pixel_mask(grid, pixel_hw, threshold="quantile", keep_frac=0.25)

    # Resize frames to pixel_hw if needed
    import cv2
    H_px, W_px = pixel_hw
    frames_resized_all = [cv2.resize(f, (W_px, H_px)) for f in frames_bgr]

    # Map each latent frame t to its corresponding raw frame via the causal VAE's
    # 8x temporal downsampling: t=0 -> raw frame 0; t>=1 -> raw frames 8(t-1)+1..8t.
    # Use the last raw frame in that window (clamped) as the representative frame.
    T = heat.shape[0]
    n_raw = len(frames_resized_all)
    frames_resized = [
        frames_resized_all[min(0 if t == 0 else 8 * t, n_raw - 1)]
        for t in range(T)
    ]

    out_path = Path(args.work_dir) / f"mask_{word}.mp4"
    ov.write_heatmap_video(frames_resized, heat.numpy(), mask.numpy(), out_path, fps=fps)
    print(f"[mask] Wrote mask video to {out_path}")

    # Single representative-frame overlay PNG, separate from the mp4.
    overlay_png = Path(args.work_dir) / f"mask_{word}_frame0.png"
    comp = ov.overlay_frame(frames_resized[0], heat[0].numpy(), mask[0].numpy())
    cv2.imwrite(str(overlay_png), comp)
    print(f"[mask] Wrote representative frame overlay to {overlay_png}")


def _run_mask(pipeline, args):
    if not getattr(args, "word", None):
        print("[ERROR] --word is required for mask mode.", file=sys.stderr)
        sys.exit(1)

    store = ac.AttentionStore(text_len_set={getattr(args, "text_len", 256)})
    store.reset()
    fps, pixel_hw, grid_thw = _run_forward(pipeline, args, store)

    T_lat, H_lat, W_lat = grid_thw
    n_target = T_lat * H_lat * W_lat

    agg = mf.aggregate(store.records)
    print(f"[mask] {len(store.records)} records, q_len={agg.shape[0]}, k_len={agg.shape[1]}, grid={grid_thw}")

    ref_path = Path(args.reference_video).expanduser().resolve()
    frames_bgr, _ = ov.read_video_frames_bgr(ref_path)
    if not frames_bgr:
        import numpy as np
        H_px, W_px = pixel_hw
        frames_bgr = [np.zeros((H_px, W_px, 3), dtype="uint8")] * T_lat

    _save_word_mask(agg, grid_thw, n_target, pixel_hw, frames_bgr, fps, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--reference-video", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--gemma-root", required=True)
    p.add_argument("--distilled-checkpoint-path", required=True)
    p.add_argument("--spatial-upsampler-path", required=True)
    p.add_argument("--ic-lora-path", required=True)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num-frames", type=int, default=97)
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--video-cfg-scale", type=float, default=2.0)
    p.add_argument("--video-strength", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantization", default="fp8-cast")
    p.add_argument("--text-len", type=int, default=1024,
                   help="Expected k_len for text cross-attention. Hardcoded to 1024 in the encoder (base_encoder.py:182).")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--word", default=None,
                   help="Word/phrase to localise (required for mask mode).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    for name in ("discover", "sweep", "mask"):
        sp = sub.add_parser(name)
        _add_common_args(sp)

    args = parser.parse_args()
    # Convert hyphenated attr names to underscores
    args.num_frames = args.num_frames
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(args)

    if args.mode == "discover":
        _run_discover(pipeline, args)
    elif args.mode == "sweep":
        _run_sweep(pipeline, args)
    elif args.mode == "mask":
        _run_mask(pipeline, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
