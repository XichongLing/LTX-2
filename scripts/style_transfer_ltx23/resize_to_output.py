#!/usr/bin/env python3
"""
Resize a source frame or video to the output dimensions using the same
resize_and_center_crop logic as the IC-LoRA pipeline.

Usage:
    python resize_to_output.py source.png output.png --height 768 --width 1280
    python resize_to_output.py source.mp4 output.mp4 --height 768 --width 1280
    python resize_to_output.py source.mp4 output.mp4 --height 384 --width 640  # stage 1
"""
import argparse
import math
from pathlib import Path

import av
from PIL import Image


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def resize_and_center_crop(img: Image.Image, height: int, width: int) -> Image.Image:
    src_h, src_w = img.height, img.width
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    crop_top  = (new_h - height) // 2
    crop_left = (new_w - width)  // 2
    return img.crop((crop_left, crop_top, crop_left + width, crop_top + height))


def convert_image(src: str, dst: str, height: int, width: int) -> None:
    img = Image.open(src).convert("RGB")
    out = resize_and_center_crop(img, height, width)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    print(f"{img.width}x{img.height} → {out.width}x{out.height}  saved to {dst}")


def convert_video(src: str, dst: str, height: int, width: int) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with av.open(src) as in_container:
        in_stream = next(s for s in in_container.streams if s.type == "video")
        fps = float(in_stream.average_rate or in_stream.base_rate or 24)
        total = in_stream.frames or "?"
        src_w = int(in_stream.codec_context.width or in_stream.width)
        src_h = int(in_stream.codec_context.height or in_stream.height)
        print(f"{src_w}x{src_h} → {width}x{height}  fps={fps:.2f}  frames={total}")

        with av.open(dst, "w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=int(round(fps)))
            out_stream.width  = width
            out_stream.height = height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18", "preset": "fast"}

            frame_idx = 0
            for packet in in_container.demux(in_stream):
                for frame in packet.decode():
                    pil = frame.to_image().convert("RGB")
                    pil = resize_and_center_crop(pil, height, width)
                    av_frame = av.VideoFrame.from_image(pil)
                    av_frame.pts = frame_idx
                    for out_packet in out_stream.encode(av_frame):
                        out_container.mux(out_packet)
                    frame_idx += 1
                    if frame_idx % 10 == 0:
                        print(f"  frame {frame_idx}/{total}", flush=True)

            for out_packet in out_stream.encode():
                out_container.mux(out_packet)

    print(f"saved to {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="Source image or video path")
    ap.add_argument("output", help="Output path")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width",  type=int, required=True)
    args = ap.parse_args()

    if Path(args.input).suffix.lower() in VIDEO_SUFFIXES:
        convert_video(args.input, args.output, args.height, args.width)
    else:
        convert_image(args.input, args.output, args.height, args.width)


if __name__ == "__main__":
    main()
