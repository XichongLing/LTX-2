#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


LTX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = LTX_ROOT / "models"

SIMPLE_FILES = [
    (
        "Alissonerdx/LTX-LoRAs",
        "ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors",
    ),
    (
        "Alissonerdx/LTX-LoRAs",
        "ltx23_edit_anything_global_rank128_v1_6000steps_prodigy.safetensors",
    ),
]

WORKFLOW_FILES = [
    ("Alissonerdx/LTX-LoRAs", "workflows/ltx23_edit_anything_v1.1.json"),
    ("Alissonerdx/LTX-LoRAs", "workflows/ltx23_inpaint_v1.json"),
    ("Alissonerdx/EditAnything", "workflows/edit_anything_with_ref_or_motion_transfer_v0.1.json"),
]

EXPERIMENTAL_FILES = [
    ("Alissonerdx/EditAnything", "edit_anything_v1.1_r256.safetensors"),
    ("Alissonerdx/EditAnything", "edit_anything_30k_v0.1_motion_transfer_r128.safetensors"),
    ("Alissonerdx/EditAnything", "edit_anything_30k_v0.1_motion_transfer_r256.safetensors"),
    ("Alissonerdx/EditAnything", "edit_anything_reference_first_version_r128_ref_adaln_proj-role_embedding.all.safetensors"),
    ("Alissonerdx/EditAnything", "edit_anything_reference_v0.1_r128_ref_adaln_proj-role_embedding.standard.safetensors"),
    ("Alissonerdx/EditAnything", "edit_anything_reference_v0.1_r128_ref_adaln_proj-role_embedding.module.safetensors"),
    (
        "Alissonerdx/EditAnything",
        "edit_anything_reference_v0.1_r128_ref_adaln_proj-role_embedding-ref_attn-ref_visual_proj.standard.safetensors",
    ),
    (
        "Alissonerdx/EditAnything",
        "edit_anything_reference_v0.1_r128_ref_adaln_proj-role_embedding-ref_attn-ref_visual_proj.module.safetensors",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EditAnything/LTX-2.3 LoRAs into the local LTX-2 models folder.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Destination folder for downloaded files.")
    parser.add_argument(
        "--set",
        choices=("simple", "workflows", "experimental", "all"),
        default="simple",
        help="simple downloads the global Edit Anything LoRAs that the local Python runner supports directly.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated/private access.")
    return parser.parse_args()


def selected_files(name: str) -> list[tuple[str, str]]:
    if name == "simple":
        return SIMPLE_FILES
    if name == "workflows":
        return WORKFLOW_FILES
    if name == "experimental":
        return EXPERIMENTAL_FILES
    return SIMPLE_FILES + WORKFLOW_FILES + EXPERIMENTAL_FILES


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, filename in selected_files(args.set):
        local_name = Path(filename).name
        print(f"Downloading {repo_id}/{filename} -> {model_dir / local_name}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=args.token,
        )
        downloaded_path = Path(downloaded)
        target = model_dir / local_name
        if downloaded_path.resolve() != target.resolve() and downloaded_path.is_file():
            target.write_bytes(downloaded_path.read_bytes())
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
