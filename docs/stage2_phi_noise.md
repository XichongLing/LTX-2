# Stage-2 Φ-Noise

The LTX-2.3 IC-LoRA pipeline can manipulate only the Stage-2 target-video noise. The transformer, prediction conversion, IC-LoRA routing, audio/reference trajectories, and Euler updates are unchanged. `gaussian` is the default and uses the existing `GaussianNoiser` directly when no diagnostics sink is requested; the diagnostic observer is sample- and generator-state-identical.

## Initialization order

1. Keep the raw spatially upscaled Stage-1 video latent as the spectral reference.
2. Build the existing Stage-2 clean state, mask, positions, and conditioning tokens.
3. Draw the same full Gaussian state tensor with the same generator and draw order.
4. FFT-manipulate only the target-video prefix. Appended source tokens, audio, and optional appearance-reference noise remain Gaussian.
5. Apply the existing blend `noise * (mask * sigma0) + clean * (1 - mask * sigma0)`. Hard conditions (`mask=0`) therefore overwrite the manipulated initialization intentionally.
6. Enter the unchanged Stage-2 solver at `sigma0=0.909375`, unless the explicitly labeled lower-noise control starts at `0.8` and then follows `0.725, 0.421875, 0`.

The raw Stage-1 latent is the phase source, before edited first-frame replacement. Stage-2 noise parameters are stored in output manifests and are not part of the Stage-1 cache fingerprint.

## CLI

The main runner and `translate_single_sequence.py` expose:

```text
--stage-2-noise-mode gaussian|phi|raw_reference_coefficients|matched_reference_coefficients
--stage-2-noise-transform spatial
--stage-2-noise-mask-mode area_fraction|official_alpha
--stage-2-noise-area-fraction 0.05
--stage-2-noise-alpha 3
--stage-2-noise-gamma 5
--stage-2-noise-phase-source stage1
--stage-2-start-sigma 0.909375
--stage-2-noise-diagnostics PATH.json
```

`area_fraction` must be strictly between zero and one. The implementation selects complete equal-radius rings using corrected `torch.fft.fftfreq` coordinates, reports requested and actual mask sizes, and never permits a full mask. `official_alpha` uses the released cutoff parameterization with corrected coordinates, so it is not bit-exact with the utility’s `linspace` grid.

The four Stage-2 conditioning conditions are represented by controlled branch mode and adapter strength:

| Condition | Arguments |
|---|---|
| Image-only | `--stage-2-branch-mode image --stage-2-ic-lora-strength 0` |
| Source without LoRA | `--stage-2-branch-mode video --stage-2-ic-lora-strength 0` |
| Weak video | `--stage-2-branch-mode video --stage-2-ic-lora-strength 0.25` |
| Full video | `--stage-2-branch-mode video --stage-2-ic-lora-strength 1` |

Do not add `--no-source-video-conditioning` to the image-only Stage-2 ablation: that flag also removes source conditioning from Stage 1. The controlled `image` branch retains normal source-conditioned Stage 1 while constructing no Stage-2 source tokens and running Stage-2 runtime LoRA at zero.

## Ablation runner

Create a JSON file with runner arguments shared by every run:

```json
{
  "output_dir": "/absolute/path/to/phi_noise_outputs",
  "base_args": [
    "--distilled-checkpoint-path", "/path/model.safetensors",
    "--spatial-upsampler-path", "/path/upsampler.safetensors",
    "--gemma-root", "/path/gemma",
    "--ic-lora-path", "/path/ic_lora.safetensors",
    "--reference-video", "/path/source.mp4",
    "--conditioning-video", "/path/control.mp4",
    "--reference-image", "/path/edited_first_frame.png",
    "--prompt", "prompt",
    "--seed", "42"
  ]
}
```

Generate the diagnostic plan without inference, then execute it:

```bash
scripts/exp/phi_noise/run_stage2_phi_ablation.py --config experiment.json --suite diagnostic
scripts/exp/phi_noise/run_stage2_phi_ablation.py --config experiment.json --suite diagnostic --execute
```

After reviewing the nine-run gate, use `--suite focused --execute` for the exact 59-run matrix. Completed diagnostic run IDs are reused. `--suite official-alpha` runs the separate alpha 3/4 presets. The runner resumes completed outputs, shares a verified Stage-2 input cache, stores commands/manifests and first-step comparisons, emits stable blinded review IDs, and builds contact sheets from identical normalized frame positions.

Review motion/layout/silhouette, target appearance, source leakage (lower is better), and temporal stability on a 1–5 scale. A promising image-only or weak-source result improves median structure by at least one point over matched-conditioning Gaussian, reduces leakage by at least one point versus full source control, loses no more than one point on appearance or stability, and has finite, energy-stable initialization/model diagnostics.

## Numerical method

Spatial `fft2`/`ifft2` uses float32 and `norm="ortho"` independently for each batch/channel/frame slice. Phase mode keeps Gaussian magnitude and substitutes reference phase only in the low band. Raw coefficient mode inserts Stage-1 coefficients; matched coefficient mode first matches their low-band energy to the Gaussian low band.

One compensation scalar is computed per batch element over all channels, frames, and spatial frequencies. The selected low band is divided by `gamma`, while unchanged Gaussian high frequencies are multiplied by the compensation scalar needed to restore total float32 FFT energy. Diagnostics report mask geometry, energies, beta, mean/std, lag-one correlations, inverse-FFT imaginary residual, checksums, and the first model response. Energy matching does not imply IID Gaussianity.

References: [Φ-Noise paper](https://arxiv.org/abs/2605.24509), [official utility](https://huggingface.co/ofirab/phi-noise/blob/main/phi_noise_utils.py), and [NeuralRemaster](https://arxiv.org/abs/2512.05106).
