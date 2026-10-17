export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REFERENCE_VIDEO=/home/xichong-drz-inokko/experimental/Image_seq_transl/ltx23-style-transfer-rgb-sintel/plane/intermediates/source_video.mp4
EDIT_DIR=/home/xichong-drz-inokko/experimental/Image_seq_transl/local_edit/
FILE_NAME=$(basename "$REFERENCE_VIDEO")
WORK_DIR=$EDIT_DIR/plane

/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python3 \
  /home/xichong-drz-inokko/experimental/LTX-2/scripts/local_edit/run_edit_anything.py \
  --distilled-checkpoint-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upsampler-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized \
  --ic-lora-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors \
  --reference-video $REFERENCE_VIDEO \
  --work-dir $WORK_DIR \
  --output $WORK_DIR/$FILE_NAME \
  --prompt "remove the aircraft, while following the original motions, scene structures, and camera poses of the source video." \
  --width 1280 \
  --height 768 \
  --video-cfg-scale 2.0 \
  --skip-stage-2 \
  --quiet-layer-streaming \
  --quantization fp8-cast
