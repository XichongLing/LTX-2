#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av

from style_transfer_preprocess import add_preprocess_args, prepare_style_transfer_inputs, read_video_metadata
from translate_single_sequence import DEFAULT_PROMPT, build_style_transfer_command


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame


def run_command(command: list[str]) -> None:
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def plan_chunks(reference_frame_indices: list[int], total_frames: int) -> list[ChunkSpec]:
    starts = sorted(set(reference_frame_indices))
    if not starts or starts[0] != 0:
        raise ValueError("--reference-frame-list must include 0")
    if starts[-1] >= total_frames:
        raise ValueError(f"last reference frame {starts[-1]} is outside {total_frames} available frames")

    ends = starts[1:] + [total_frames]
    chunks = [ChunkSpec(index=i, start_frame=start, end_frame=end) for i, (start, end) in enumerate(zip(starts, ends))]
    empty_chunks = [chunk for chunk in chunks if chunk.num_frames <= 0]
    if empty_chunks:
        raise ValueError(f"reference-frame-list produced empty chunks: {empty_chunks}")
    return chunks


def slice_video_by_frames(
    input_path: str, output_path: Path, start_frame: int, end_frame: int, rate: int | Fraction | None = None
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        print(f"Reusing existing video chunk: {output_path}")
        return output_path

    with av.open(input_path) as input_container, av.open(str(output_path), mode="w") as output_container:
        input_stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
        if input_stream is None:
            raise ValueError(f"No video stream found in {input_path}")

        if rate is None:
            fps_value = input_stream.average_rate or input_stream.base_rate
            rate = fps_value if fps_value is not None else 24
        width = int(input_stream.codec_context.width or input_stream.width)
        height = int(input_stream.codec_context.height or input_stream.height)

        output_stream = output_container.add_stream("libx264", rate=rate)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_time_base = Fraction(1, rate) if isinstance(rate, int) else Fraction(rate.denominator, rate.numerator)
        output_stream.time_base = output_time_base

        next_pts = 0
        for frame_idx, frame in enumerate(input_container.decode(video=0)):
            if frame_idx < start_frame:
                continue
            if frame_idx >= end_frame:
                break
            frame = frame.reformat(width=width, height=height, format="yuv420p")
            frame.pts = next_pts
            frame.time_base = output_time_base
            next_pts += 1
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)

    return output_path


def concatenate_videos(chunk_paths: list[Path], output_path: Path, fps: float) -> None:
    if not chunk_paths:
        raise ValueError("No chunk outputs to concatenate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(chunk_paths[0])) as first_container:
        first_stream = next((stream for stream in first_container.streams if stream.type == "video"), None)
        if first_stream is None:
            raise ValueError(f"No video stream found in {chunk_paths[0]}")
        width = int(first_stream.codec_context.width or first_stream.width)
        height = int(first_stream.codec_context.height or first_stream.height)

    output_fps = max(1, int(round(fps)))
    output_time_base = Fraction(1, output_fps)

    with av.open(str(output_path), mode="w") as output_container:
        output_stream = output_container.add_stream("libx264", rate=output_fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = "yuv420p"
        output_stream.time_base = output_time_base

        next_pts = 0
        for chunk_path in chunk_paths:
            with av.open(str(chunk_path)) as input_container:
                for frame in input_container.decode(video=0):
                    frame = frame.reformat(width=width, height=height, format="yuv420p")
                    frame.pts = next_pts
                    frame.time_base = output_time_base
                    next_pts += 1
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate one source sequence/video in chunks. The reference-frame list defines chunk starts; "
            "each chunk is conditioned on its translated start frame at local frame 0."
        )
    )
    add_preprocess_args(parser)
    parser.set_defaults(prompt=DEFAULT_PROMPT)
    parser.add_argument("--output", required=True, help="Final concatenated stylized MP4 output path.")
    parser.add_argument("--distilled-checkpoint-path", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--ic-lora-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--image-frame-idx", type=int, default=0)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument(
        "--reference-image-replace",
        type=int,
        nargs="*",
        default=None,
        metavar="FRAME_IDX",
        help=(
            "Global reference-frame indices to inject as in-place latent replacements. "
            "In chunked mode each matching chunk start is passed as local frame 0."
        ),
    )
    parser.add_argument("--video-strength", type=float, default=1.0)
    parser.add_argument("--conditioning-attention-strength", type=float, default=1.0)
    parser.add_argument("--skip-stage-2", action="store_true")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--quiet-layer-streaming", action="store_true")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None)
    parser.add_argument("--quantization", nargs="+", default=None)
    parser.add_argument(
        "--print-prepared-inputs",
        action="store_true",
        help="Print the prepared artifact paths before running chunked IC-LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)

    prepared = prepare_style_transfer_inputs(args)
    if args.print_prepared_inputs:
        print(f"Prepared inputs: {asdict(prepared)}")

    _width, _height, source_frames, source_fps = read_video_metadata(prepared.source_video)
    total_frames = args.num_frames if args.num_frames is not None else source_frames
    if total_frames > source_frames:
        raise ValueError(f"--num-frames={total_frames} exceeds source video length {source_frames}")
    frame_rate = args.frame_rate if args.frame_rate is not None else source_fps
    args.frame_rate = frame_rate

    chunks = plan_chunks(prepared.reference_frame_indices, total_frames)
    reference_by_start = dict(zip(prepared.reference_frame_indices, prepared.reference_images, strict=True))
    if args.reference_image_replace is not None:
        chunk_starts = set(reference_by_start)
        missing = [idx for idx in sorted(set(args.reference_image_replace)) if idx not in chunk_starts]
        if missing:
            raise ValueError(
                "--reference-image-replace can only name chunk starts from --reference-frame-list in chunked mode. "
                f"Missing from --reference-frame-list: {missing}"
            )
    reference_image_replace_set = set(args.reference_image_replace or [])

    chunk_outputs: list[Path] = []
    chunks_dir = Path(prepared.work_dir) / "chunks"
    for chunk in chunks:
        chunk_dir = chunks_dir / f"chunk_{chunk.index:03d}_{chunk.start_frame:06d}"
        source_chunk = slice_video_by_frames(
            prepared.source_video,
            chunk_dir / "source.mp4",
            chunk.start_frame,
            chunk.end_frame,
        )
        control_chunk = slice_video_by_frames(
            prepared.control_video,
            chunk_dir / "control.mp4",
            chunk.start_frame,
            chunk.end_frame,
        )
        chunk_output = chunk_dir / "output.mp4"
        reference_image = reference_by_start[chunk.start_frame]
        if args.reference_image_replace is None:
            chunk_reference_image_replace = None
        elif chunk.start_frame in reference_image_replace_set:
            chunk_reference_image_replace = [0]
        else:
            chunk_reference_image_replace = []

        command = build_style_transfer_command(
            args,
            str(source_chunk),
            str(control_chunk),
            reference_image,
            image_conditions=[(reference_image, 0)],
            num_frames=chunk.num_frames,
            output=str(chunk_output),
            reference_image_replace=chunk_reference_image_replace,
        )
        print(f"=== Chunk {chunk.index}: frames {chunk.start_frame}..{chunk.end_frame - 1} ===")
        run_command(command)
        chunk_outputs.append(chunk_output)

    concatenate_videos(chunk_outputs, output_path, frame_rate)

    print("Chunked single-sequence translation completed successfully.")
    print(f"Chunks: {len(chunk_outputs)}")
    print(f"Reference frame indices: {prepared.reference_frame_indices}")
    print(
        "Reference image replace: "
        f"{args.reference_image_replace if args.reference_image_replace is not None else 'default(chunk starts)'}"
    )
    print(f"Final output: {output_path}")


if __name__ == "__main__":
    main()
