# LTX-2 IC-LoRA Detailer Scripts

These scripts run `ltx-2-19b-ic-lora-detailer.safetensors` as an IC-LoRA adapter.
The main input is an existing video, which is used as RGB video conditioning.

## Single Video

```bash
python scripts/detailer/run_ic_lora_detailer.py \
  --distilled-checkpoint-path /path/to/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upsampler-path /path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /path/to/gemma-3 \
  --detailer-lora-path /path/to/ltx-2-19b-ic-lora-detailer.safetensors \
  --input-video /path/to/input.mp4 \
  --output /path/to/output_detail.mp4 \
  --width 1280 \
  --height 704 \
  --num-frames 97 \
  --frame-rate 24 \
  --detailer-lora-strength 1.0 \
  --video-strength 1.0 \
  --conditioning-attention-strength 1.0
```

For quick previews, add `--skip-stage-2`. The preview output is half resolution.

## Inputs

- `--input-video`: the video to refine; by default it is also the IC-LoRA conditioning video.
- `--conditioning-video`: optional separate control video if you want the detailer to follow a preprocessed RGB/depth/edge clip.
- `--prompt`: describes the desired refined result.
- `--detailer-lora-path`: the detailer adapter weights.
- `--image-condition PATH FRAME_IDX STRENGTH [CRF]`: optional image anchor, repeatable.

## Useful Starting Values

- Subtle detail preservation: `--detailer-lora-strength 0.6 --conditioning-attention-strength 0.8`
- Strong detail pass: `--detailer-lora-strength 1.0 --conditioning-attention-strength 1.0`
- More freedom to regenerate texture: lower `--conditioning-attention-strength` to `0.5-0.7`

Keep dimensions divisible by 64. If dimensions are omitted, the script uses the input video's size rounded down to a multiple of 64.
