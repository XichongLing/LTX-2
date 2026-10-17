#!/usr/bin/env python3
"""Draw a Stage-2 routing mask on the resized/cropped reference first frame."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

DEFAULT_REFERENCE_IMAGE = (
    "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/"
    "anna_stage2_vcond/intermediates/flux/translated_first_frame.png"
)
DEFAULT_OUTPUT_MASK = (
    "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/"
    "anna_stage2_vcond/intermediates/stage2_hand_drawn_mask.png"
)
DEFAULT_OUTPUT_PREVIEW = (
    "/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/"
    "anna_stage2_vcond/intermediates/stage2_hand_drawn_mask_preview.png"
)


def resize_and_center_crop(img: Image.Image, height: int, width: int) -> Image.Image:
    src_h, src_w = img.height, img.width
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    return img.crop((crop_left, crop_top, crop_left + width, crop_top + height))


def load_canvas(reference_image: str, height: int, width: int) -> Image.Image:
    image_path = Path(reference_image).expanduser().resolve()
    if not image_path.exists():
        raise gr.Error(f"Reference image does not exist: {image_path}")
    return resize_and_center_crop(Image.open(image_path).convert("RGB"), height, width)


def select_reference_image(uploaded_image: str | None, current_reference: str, height: int, width: int):
    reference_image = uploaded_image or current_reference
    return reference_image, load_canvas(reference_image, height, width)


def _extract_layer_mask(editor_value: dict | Image.Image | None, height: int, width: int) -> Image.Image:
    if editor_value is None:
        raise gr.Error("No drawing found. Draw a white mask layer first.")
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers") or []
        if not layers or layers[0] is None:
            raise gr.Error("No mask layer found. Draw a white mask layer first.")
        mask = layers[0]
    else:
        mask = editor_value
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(np.asarray(mask))
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.BILINEAR)
    return mask


def save_mask(editor_value, reference_image: str, output_mask: str, output_preview: str, height: int, width: int):
    canvas = load_canvas(reference_image, height, width)
    mask = _extract_layer_mask(editor_value, height, width)
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.max() == 0:
        raise gr.Error("Mask is empty. Draw the dress region in white.")

    mask_path = Path(output_mask).expanduser().resolve()
    preview_path = Path(output_preview).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)

    canvas_arr = np.asarray(canvas).astype(np.float32)
    alpha = (arr.astype(np.float32) / 255.0)[..., None]
    overlay_color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    preview = canvas_arr * (1.0 - 0.45 * alpha) + overlay_color * (0.45 * alpha)
    preview_img = Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8))
    preview_img.save(preview_path)

    status = f"Saved mask: {mask_path}\nSaved preview: {preview_path}\nMask size: {mask.width}x{mask.height}"
    return mask, preview_img, status


def build_app(args: argparse.Namespace) -> gr.Blocks:
    initial_canvas = load_canvas(args.reference_image, args.height, args.width)
    with gr.Blocks(title="Stage-2 Routing Mask Drawer") as demo:
        gr.Markdown("# Stage-2 Routing Mask Drawer")
        gr.Markdown("Draw the dress region in white. The canvas is already resized and center-cropped to the LTX output size.")
        with gr.Row():
            with gr.Column(scale=2):
                editor = gr.ImageEditor(
                    label="Reference first frame, output-aligned",
                    value=initial_canvas,
                    type="pil",
                    image_mode="RGB",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=args.brush_size, color_mode="fixed"),
                    eraser=True,
                    interactive=True,
                )
            with gr.Column(scale=1):
                reference_picker = gr.Image(
                    label="Select reference image",
                    sources=["upload"],
                    type="filepath",
                    interactive=True,
                    show_download_button=False,
                )
                reference = gr.Textbox(label="Reference image", value=args.reference_image)
                output_mask = gr.Textbox(label="Output mask", value=args.output_mask)
                output_preview = gr.Textbox(label="Output preview", value=args.output_preview)
                height = gr.Number(label="Height", value=args.height, precision=0)
                width = gr.Number(label="Width", value=args.width, precision=0)
                reload_btn = gr.Button("Reload Canvas")
                save_btn = gr.Button("Save Mask", variant="primary")
                status = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Row():
            mask_preview = gr.Image(label="Saved mask", type="pil", interactive=False)
            overlay_preview = gr.Image(label="Overlay preview", type="pil", interactive=False)

        reload_btn.click(
            fn=lambda ref, h, w: load_canvas(ref, int(h), int(w)),
            inputs=[reference, height, width],
            outputs=[editor],
        )
        reference_picker.change(
            fn=lambda uploaded, ref, h, w: select_reference_image(uploaded, ref, int(h), int(w)),
            inputs=[reference_picker, reference, height, width],
            outputs=[reference, editor],
        )
        save_btn.click(
            fn=lambda edited, ref, out, prev, h, w: save_mask(edited, ref, out, prev, int(h), int(w)),
            inputs=[editor, reference, output_mask, output_preview, height, width],
            outputs=[mask_preview, overlay_preview, status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--output-mask", default=DEFAULT_OUTPUT_MASK)
    parser.add_argument("--output-preview", default=DEFAULT_OUTPUT_PREVIEW)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--brush-size", type=int, default=35)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    build_app(parsed).launch(server_name=parsed.server_name, server_port=parsed.server_port, share=parsed.share)
