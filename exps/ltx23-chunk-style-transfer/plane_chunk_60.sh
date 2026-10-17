LTX_ROOT="/home/xichong-drz-inokko/experimental/LTX-2"

python $LTX_ROOT/scripts/run_chunked_style_transfer.py \
  --source-video /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane_flux/intermediates/source_video.mp4 \
  --target-reference /home/xichong-drz-inokko/experimental/data/disney/images/display_0000.png \
  --work-dir /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-chunk \
  --output /home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-chunk/output.mp4 \
  --chunk-len 60 \
  --prompt "convert the video into a photorealistic style, meanwhile preserving the original composition, structure, and camera motions." \
  --distilled-checkpoint-path "$LTX_ROOT/models/ltx-2.3-22b-distilled-1.1.safetensors" \
  --spatial-upsampler-path $LTX_ROOT/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors\
  --gemma-root "/home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized" \
  --ic-lora-path "$LTX_ROOT/models/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
  --quantization fp8-cast \
  --quiet-layer-streaming 