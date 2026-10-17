#!/usr/bin/env python3
"""
Show how source-video and image-condition pixels map to output-video pixels,
following the exact resize_and_center_crop path used by the IC-LoRA pipeline.

Both stage 1 (H//2 x W//2, where the RGB conditioning video is active) and
stage 2 (H x W, the final output resolution) are shown, since image conditions
are applied at both stages.

Usage:
    python pixel_correspondence.py \\
        --source-frame  frame_000027.png  \\
        --cond-image    cond_000027.png   \\
        --output-height 768               \\
        --output-width  1280              \\
        [--output-dir   ./pixel_maps]     \\
        [--point X Y]   # source-frame pixel to track; repeatable
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class CropParams:
    scale: float
    crop_top: int
    crop_left: int
    src_h: int
    src_w: int
    tgt_h: int
    tgt_w: int

    def src_to_out(self, y: float, x: float) -> tuple[float, float]:
        return y * self.scale - self.crop_top, x * self.scale - self.crop_left

    def out_to_src(self, y: float, x: float) -> tuple[float, float]:
        return (y + self.crop_top) / self.scale, (x + self.crop_left) / self.scale

    def in_bounds(self, y: float, x: float) -> bool:
        return 0 <= y < self.tgt_h and 0 <= x < self.tgt_w


def compute_crop_params(src_h: int, src_w: int, tgt_h: int, tgt_w: int) -> CropParams:
    """Replicate the scale-to-fill + center-crop from resize_and_center_crop in media_io.py."""
    scale = max(tgt_h / src_h, tgt_w / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    crop_top  = (new_h - tgt_h) // 2
    crop_left = (new_w - tgt_w) // 2
    return CropParams(scale, crop_top, crop_left, src_h, src_w, tgt_h, tgt_w)


def apply_resize_crop(img: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
    """Apply the same resize_and_center_crop logic on a HWC uint8 numpy array."""
    p = compute_crop_params(img.shape[0], img.shape[1], tgt_h, tgt_w)
    new_h = math.ceil(img.shape[0] * p.scale)
    new_w = math.ceil(img.shape[1] * p.scale)
    # PIL BILINEAR matches torch.nn.functional.interpolate(..., mode="bilinear")
    resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    return resized[p.crop_top: p.crop_top + tgt_h, p.crop_left: p.crop_left + tgt_w]


def draw_crosshair(arr: np.ndarray, y: int, x: int,
                   color: tuple[int, int, int] = (255, 0, 0),
                   size: int = 24, thickness: int = 2) -> np.ndarray:
    pil = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(pil)
    draw.line([(x - size, y), (x + size, y)], fill=color, width=thickness)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=thickness)
    return np.array(pil)


def label_image(arr: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.array(pil)


def save(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


DIV = "─" * 64


def print_params(label: str, p: CropParams) -> None:
    print(f"\n  {label}  ({p.src_w}×{p.src_h} → {p.tgt_w}×{p.tgt_h})")
    print(f"    scale      = {p.scale:.6f}")
    print(f"    crop_top   = {p.crop_top}  crop_left = {p.crop_left}")
    print(f"    x_out = x_src * {p.scale:.6f} - {p.crop_left}")
    print(f"    y_out = y_src * {p.scale:.6f} - {p.crop_top}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-frame",  required=True,
                    help="Single frame extracted from the source video (or the video itself if readable by PIL)")
    ap.add_argument("--cond-image",    required=True, help="Image condition file")
    ap.add_argument("--output-height", type=int, required=True, help="Output video height  (e.g. 768)")
    ap.add_argument("--output-width",  type=int, required=True, help="Output video width   (e.g. 1280)")
    ap.add_argument("--output-dir",    default="pixel_maps")
    ap.add_argument(
        "--point",
        nargs=2,
        type=int,
        action="append",
        metavar=("X", "Y"),
        default=[],
        help="Source-frame pixel coordinate to track (X Y, 0-indexed). Repeatable.",
    )
    args = ap.parse_args()

    out_h, out_w = args.output_height, args.output_width
    s1_h,  s1_w  = out_h // 2, out_w // 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src  = load_rgb(args.source_frame)
    cond = load_rgb(args.cond_image)
    src_h,  src_w  = src.shape[:2]
    cond_h, cond_w = cond.shape[:2]

    # Crop params for all (image, resolution) combinations
    p_src_s1   = compute_crop_params(src_h,  src_w,  s1_h,  s1_w)
    p_cond_s1  = compute_crop_params(cond_h, cond_w, s1_h,  s1_w)
    p_src_s2   = compute_crop_params(src_h,  src_w,  out_h, out_w)
    p_cond_s2  = compute_crop_params(cond_h, cond_w, out_h, out_w)

    print(f"\n{DIV}")
    print(f"  Source frame  : {src_w}×{src_h}   AR={src_w/src_h:.4f}")
    print(f"  Condition img : {cond_w}×{cond_h}   AR={cond_w/cond_h:.4f}")
    print(f"  Output (stage2): {out_w}×{out_h}   AR={out_w/out_h:.4f}")
    print(f"  Stage 1       : {s1_w}×{s1_h}   AR={s1_w/s1_h:.4f}")
    print(DIV)
    print("\nResize / crop parameters")
    print_params("source  → stage 1  [RGB conditioning video]", p_src_s1)
    print_params("cond    → stage 1  [image conditioning]",     p_cond_s1)
    print_params("source  → stage 2  [reference only, no IC-LoRA video]", p_src_s2)
    print_params("cond    → stage 2  [image conditioning]",     p_cond_s2)

    # Cross-mapping formulas: cond pixel → source pixel that maps to the same output pixel
    print(f"\n{DIV}")
    print("Cross-map [stage 1]: source pixel that sees the same output location as cond pixel (x_c, y_c)")
    cs, cl, ct = p_cond_s1.scale, p_cond_s1.crop_left, p_cond_s1.crop_top
    ss, sl, st = p_src_s1.scale, p_src_s1.crop_left, p_src_s1.crop_top
    print(f"    x_src = (x_cond * {cs:.6f} - {cl} + {sl}) / {ss:.6f}")
    print(f"    y_src = (y_cond * {cs:.6f} - {ct} + {st}) / {ss:.6f}")

    print(f"\nCross-map [stage 2]: source pixel that sees the same output location as cond pixel (x_c, y_c)")
    cs, cl, ct = p_cond_s2.scale, p_cond_s2.crop_left, p_cond_s2.crop_top
    ss, sl, st = p_src_s2.scale, p_src_s2.crop_left, p_src_s2.crop_top
    print(f"    x_src = (x_cond * {cs:.6f} - {cl} + {sl}) / {ss:.6f}")
    print(f"    y_src = (y_cond * {cs:.6f} - {ct} + {st}) / {ss:.6f}")

    # ── produce visual outputs ─────────────────────────────────────────────
    src_s1   = apply_resize_crop(src,  s1_h,  s1_w)
    cond_s1  = apply_resize_crop(cond, s1_h,  s1_w)
    src_s2   = apply_resize_crop(src,  out_h, out_w)
    cond_s2  = apply_resize_crop(cond, out_h, out_w)

    # ── specific point tracking ────────────────────────────────────────────
    src_s1_ann  = src_s1.copy()
    cond_s1_ann = cond_s1.copy()
    src_s2_ann  = src_s2.copy()
    cond_s2_ann = cond_s2.copy()

    if args.point:
        print(f"\n{DIV}")
        print("Tracked point mapping")
        print(DIV)

    colors = [(255, 50, 50), (50, 200, 50), (80, 150, 255), (255, 200, 0)]
    for idx, (x_src, y_src) in enumerate(args.point):
        color = colors[idx % len(colors)]
        print(f"\n  Source ({x_src}, {y_src}), color {'RGBY'[idx % 4]}:")

        # Source → output (stage 1 and 2)
        for stage_label, p_src, p_cond, s_ann, c_ann in [
            ("stage1", p_src_s1, p_cond_s1, src_s1_ann, cond_s1_ann),
            ("stage2", p_src_s2, p_cond_s2, src_s2_ann, cond_s2_ann),
        ]:
            y_out, x_out = p_src.src_to_out(y_src, x_src)
            yi, xi = int(round(y_out)), int(round(x_out))
            in_b = p_src.in_bounds(yi, xi)
            note = "" if in_b else "  ← cropped away"
            print(f"    {stage_label}: out=({x_out:.1f}, {y_out:.1f}){note}")
            if in_b:
                if stage_label == "stage1":
                    src_s1_ann  = draw_crosshair(src_s1_ann,  yi, xi, color)
                else:
                    src_s2_ann  = draw_crosshair(src_s2_ann,  yi, xi, color)

            # Where is the equivalent location in the condition image?
            y_cond_src = (y_out + p_cond.crop_top) / p_cond.scale
            x_cond_src = (x_out + p_cond.crop_left) / p_cond.scale
            in_cond_src = (0 <= y_cond_src < cond_h) and (0 <= x_cond_src < cond_w)
            y_c_out, x_c_out = p_cond.src_to_out(y_cond_src, x_cond_src)
            yci, xci = int(round(y_c_out)), int(round(x_c_out))
            in_c_out = p_cond.in_bounds(yci, xci)
            print(f"    {stage_label}: cond src=({x_cond_src:.1f}, {y_cond_src:.1f})", end="")
            if not in_cond_src:
                print("  ← outside cond source image")
            else:
                print(f" → cond out=({x_c_out:.1f}, {y_c_out:.1f})", end="")
                if not in_c_out:
                    print("  ← cropped away in condition")
                else:
                    print()
            if in_c_out:
                if stage_label == "stage1":
                    cond_s1_ann = draw_crosshair(cond_s1_ann, yci, xci, color)
                else:
                    cond_s2_ann = draw_crosshair(cond_s2_ann, yci, xci, color)

    # ── save images ────────────────────────────────────────────────────────
    def _sbs(a: np.ndarray, b: np.ndarray, la: str, lb: str) -> np.ndarray:
        return np.concatenate([label_image(a, la), label_image(b, lb)], axis=1)

    def _blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(float) * 0.5 + b.astype(float) * 0.5).astype(np.uint8)

    save(src_s1,  out_dir / "source_stage1.png")
    save(cond_s1, out_dir / "cond_stage1.png")
    save(src_s2,  out_dir / "source_stage2.png")
    save(cond_s2, out_dir / "cond_stage2.png")

    save(_sbs(src_s1,  cond_s1,  "source @ stage1",  "cond @ stage1"),
         out_dir / "side_by_side_stage1.png")
    save(_sbs(src_s2,  cond_s2,  "source @ stage2",  "cond @ stage2"),
         out_dir / "side_by_side_stage2.png")
    save(label_image(_blend(src_s1,  cond_s1),  "50/50 blend @ stage1"),
         out_dir / "blend_stage1.png")
    save(label_image(_blend(src_s2,  cond_s2),  "50/50 blend @ stage2"),
         out_dir / "blend_stage2.png")

    if args.point:
        save(src_s1_ann,  out_dir / "source_stage1_annotated.png")
        save(cond_s1_ann, out_dir / "cond_stage1_annotated.png")
        save(src_s2_ann,  out_dir / "source_stage2_annotated.png")
        save(cond_s2_ann, out_dir / "cond_stage2_annotated.png")
        save(_sbs(src_s1_ann, cond_s1_ann, "source @ stage1", "cond @ stage1"),
             out_dir / "side_by_side_stage1_annotated.png")
        save(_sbs(src_s2_ann, cond_s2_ann, "source @ stage2", "cond @ stage2"),
             out_dir / "side_by_side_stage2_annotated.png")

    print(f"\n{DIV}")
    print(f"Saved to {out_dir}/")
    print("  source_stage1.png          source_stage2.png")
    print("  cond_stage1.png            cond_stage2.png")
    print("  side_by_side_stage1.png    side_by_side_stage2.png")
    print("  blend_stage1.png           blend_stage2.png")
    if args.point:
        print("  *_annotated.png  — same images with tracked points marked")


if __name__ == "__main__":
    main()
