# Local EditAnything Pipeline

This folder contains a lightweight local runner for LTX-2.3 EditAnything-style IC-LoRA video editing.

## 1. Download supported LoRAs

The local Python runner supports the simple global Edit Anything LoRAs from `Alissonerdx/LTX-LoRAs` directly:

```bash
/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python3 \
  /home/xichong-drz-inokko/experimental/LTX-2/scripts/local_edit/download_loras.py \
  --set simple
```

Files are saved into:

```text
/home/xichong-drz-inokko/experimental/LTX-2/models
```

To also fetch workflows and experimental/reference/motion-transfer variants:

```bash
.../download_loras.py --set all
```

Note: `.standard` + `.module` reference variants from `Alissonerdx/EditAnything` are intended for BFSnodes/ComfyUI-style loading and are not guaranteed to work through this plain Python IC-LoRA runner.

## 2. Run one edit

```bash
/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python3 \
  /home/xichong-drz-inokko/experimental/LTX-2/scripts/local_edit/run_edit_anything.py \
  --input-path /path/to/source_frames_or_video \
  --work-dir /tmp/ltx_edit_work \
  --output /tmp/ltx_edit_output.mp4 \
  --reference-image-path /path/to/reference.png \
  --prompt "make the video look like the reference image while preserving motion and composition" \
  --width 1280 \
  --height 768
```

If you do not have a separate reference image, use the source first frame as the image condition:

```bash
.../run_edit_anything.py \
  --input-path /path/to/source.mp4 \
  --work-dir /tmp/ltx_edit_work \
  --output /tmp/ltx_edit_output.mp4 \
  --use-source-first-frame \
  --prompt "turn the scene into a rainy night cinematic shot"
```

The runner copies the provided reference image to:

```text
<work-dir>/flux/translated_first_frame.png
```

No FLUX.2 reference generation is used in this local edit runner.

## Notes

- Default backbone: `models/ltx-2.3-22b-distilled-1.1.safetensors`
- Default LoRA: `models/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors`
- Default conditioning mode: RGB source video as control video
- RTX 5090 is supported by the local `torch 2.9.1+cu128` stack with `sm_120` support.
