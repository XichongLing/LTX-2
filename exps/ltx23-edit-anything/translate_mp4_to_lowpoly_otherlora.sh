export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REFERENCE_VIDEO=/home/xichong-drz-inokko/experimental/data/pexelvideos/clips/028_cinematic_139076_592820390.mp4
EDIT_DIR=/home/xichong-drz-inokko/experimental/Image_seq_transl/local_edit/
FILE_NAME=$(basename "$REFERENCE_VIDEO")
WORK_DIR=$EDIT_DIR/dog_nolora

/home/xichong-drz-inokko/experimental/LTX-2/.venv/bin/python3 \
  /home/xichong-drz-inokko/experimental/LTX-2/scripts/local_edit/run_edit_anything.py \
  --distilled-checkpoint-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upsampler-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /home/xichong-drz-inokko/experimental/gemma-3-12b-it-qat-q4_0-unquantized \
  --ic-lora-path /home/xichong-drz-inokko/experimental/LTX-2/models/ltx-2.3-ils22b-lora_weights_step_03000.safetensors \
  --reference-video $REFERENCE_VIDEO \
  --work-dir $WORK_DIR \
  --output $WORK_DIR/$FILE_NAME \
  --prompt "Convert dog into a low-poly, proxy-mesh-alike model, while following the original motions, scene structures, and camera poses of the source video." \
  --width 1280 \
  --height 768 \
  --video-cfg-scale 2.0 \
  --skip-stage-2 \
  --quiet-layer-streaming \
  --quantization fp8-cast
