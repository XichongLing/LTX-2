# Intermediate single frame in-place replacement
## Descriptions
The normal practice is to in-place insert the reference first frame token into the corresponding tokens places. Now instead of substituting the first frame, I want to see the results of substituting a single frame in the other places (intermediate, end, etc).

## settings
1. run the experiments on ic detailor lora, use reference videos, and reference images in /home/xichong-drz-inokko/experimental/data/detailer_experiments, try experiments on first frame, two different intermediate frames, and the last frame (so 4 sets in total).

## output
save the output videos, and exp configurations in separate folders under /home/xichong-drz-inokko/experimental/Image_seq_transl/replace_non_first_frame