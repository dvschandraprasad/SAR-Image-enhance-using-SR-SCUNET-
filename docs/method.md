# Method: technical notes

This page explains the pipeline in more detail than the README. Every value here was read from the committed code, unless it is marked **author-reported**.

## Contents

- [Problem setup](#problem-setup)
- [Patch sampling](#patch-sampling)
- [Degradation model](#degradation-model)
- [Training curriculum](#training-curriculum)
- [Normalisation](#normalisation)
- [SCUNet architecture](#scunet-architecture)
- [Loss functions](#loss-functions)
- [Training setup](#training-setup)
- [Committed config vs author-reported config](#committed-config-vs-author-reported-config)
- [Metrics](#metrics)
- [Evaluation scripts](#evaluation-scripts)
- [Full-scene inference](#full-scene-inference)
- [Known issues](#known-issues)

---

## Problem setup

- **Task:** single-image super-resolution (SISR) at **2x**. Input: a 1 x 64 x 64 low-resolution (LR) patch. Output: a 1 x 128 x 128 super-resolved (SR) patch.
- **Data:** band 1 of Capella **GEO** GeoTIFFs (X-band, Spotlight, geocoded and terrain-corrected amplitude). The scripts read it as `uint16`.
- **Supervision:** there are no real LR/HR pairs. Each training pair is made on the fly from one real image:
  `HR = random 128 x 128 window`, `LR = degrade(HR)`.

## Patch sampling

Code: `SARDataset` in `src/sar_dataset.py`.

- Each `__getitem__` call opens the GeoTIFF with `rasterio` and reads a **new random** 128 x 128 window. The index only decides *which image* is used (`idx % number_of_images`), not *where* in the image. So the same index gives a different patch every time.
- A window is accepted when at least `MIN_NONZERO = 0.8` of its pixels are non-zero. This skips no-data borders. The code tries up to `max_attempts = 200` times, then falls back to a centre crop.
- Augmentation (training only): a random 90-degree rotation (k = 0..3), then a random left-right flip and a random up-down flip (each with probability 0.5). The same transform is applied to LR and HR.
- `scunet_train.py` sets the number of patches per image (per epoch) to `min(patches_per_image = 200, (H // 128) * (W // 128))`.

## Degradation model

Code: `sar_dataset.degrade`. It works on **raw amplitude**, before the log step, because speckle is multiplicative on physical values.

Let `x` be the HR patch (float32, NaN/inf set to 0). The steps are:

| Step | Formula | Code detail |
|---|---|---|
| 1. Sensor blur (PSF) | `x_b = G_sigma * x` | `scipy.ndimage.gaussian_filter`, `sigma = PSF_SIGMA` in HR pixels |
| 2. Downsample by 2 | `y = D_2(x_b)` | `skimage.transform.resize`, `order=3` (bicubic), `anti_aliasing=True` |
| 3. Speckle | `s ~ clip(Gamma(k = L, theta = 1/L), 0.8, 1.25)` | `L = SPECKLE_LOOKS`. Before the clip, mean = 1 and variance = 1/L |
| 4. Additive noise | `n ~ N(0, sigma_n^2)` | `sigma_n = NOISE_STD` |
| 5. Noisy LR | `y_noisy = y * s + n` | element-wise |
| 6. Blend | `y_lr = b * y_noisy + (1 - b) * y` | `b = DEGRADE_BLEND` |
| 7. Clip | `LR = max(y_lr, 0)` | no upper clip |

Steps 5 to 7 together give: `LR = max(0, y * (1 + b * (s - 1)) + b * n)`.

**Module defaults** (`src/sar_dataset.py`): `PSF_SIGMA = 0.20`, `SPECKLE_LOOKS = 40`, `NOISE_STD = 0.0002`, `DEGRADE_BLEND = 0.5`, `SCALE_FACTOR = 2`, `PATCH_SIZE = 128`.

`inference.py` keeps its **own copy** of these constants (same values). If you change one, change both. The test `test_inference_degradation_constants_match_sar_dataset` checks that they agree.

What these numbers mean in practice:

- **Effective blur is larger than `PSF_SIGMA`.** With `anti_aliasing=True`, `skimage` adds its own Gaussian blur before resizing, with sigma = (factor - 1) / 2 = **0.5 HR pixels** for 2x. So the total blur is about sqrt(0.20^2 + 0.5^2), roughly 0.54 HR pixels.
- **Speckle is mild.** With L = 40, the gamma noise has a standard deviation of 1/sqrt(40), about 0.16, but it is clipped to [0.8, 1.25]. After the blend with b = 0.5, each pixel is multiplied by a factor between 0.9 and 1.125.
- **Additive noise is almost zero.** `NOISE_STD` is in raw pixel units. The pixels are integers, and `normalize_joint` clips them at 10,000, so a standard deviation of 0.0002 (or 0.003 in the author-reported setting) is far below one grey level.

## Training curriculum

Code: `DEGRADE_CURRICULUM`, `get_curriculum_stage` and `apply_degradation_stage` in `src/scunet_train.py`.

Anchor points `(epoch, sigma, looks, noise, blend)`:

| From epoch | PSF sigma | Looks L | Noise std | Blend b |
|---|---|---|---|---|
| 1 | 0.22 | 28 | 3e-4 | 0.65 |
| 21 | 0.18 | 40 | 2e-4 | 0.50 |
| 41 | 0.14 | 64 | 1e-4 | 0.35 |
| 71 and later | 0.10 | 96 | 5e-5 | 0.25 |

- With `curriculum_interp = 'linear'`, the values move linearly between anchors (looks are rounded to an integer). Example: epoch 11 gives sigma 0.20, 34 looks, noise 2.5e-4, blend 0.575.
- At the start of each epoch, `apply_degradation_stage` writes the values into the `sar_dataset` module globals.
- The **validation set is built once, before training, at the final (easiest) stage**, and kept in memory. So validation PSNR always measures the same task, but it is an easier task than most of the training epochs.
- The DataLoaders use `num_workers=0` (the `CFG['num_workers'] = 4` value is not used). This matters: with worker processes, each worker has its own copy of the module, and the curriculum changes would not reach the dataset.

## Normalisation

Code: `sar_dataset.normalize_joint`.

1. Clip both HR and LR to [0, 10000], then take `h = log1p(HR)` and `l = log1p(LR)`.
2. Compute `p1` and `p99`, the 1st and 99th percentiles of `h` (the **HR** patch only).
3. `HR_n = (clip(h, p1, p99) - p1) / (p99 - p1)`
4. `LR_n = clip((l - p1) / (p99 - p1), 0, 1)` (same `p1` and `p99`).
5. If `p99 - p1 < 1e-6` (a flat patch), both outputs are all zeros.

Why joint: if HR and LR are normalised separately, both are stretched to [0, 1] and the brightness change caused by the degradation is hidden. The code comments say this gave a PSNR ceiling of about 15 dB in earlier runs.

Note: the percentiles are taken **per patch**, so absolute radiometry is not kept.

## SCUNet architecture

Code: `src/scunet_model.py`. This is the author's own **simplified re-implementation**, adapted for SR. It is not the official SCUNet code ([Zhang et al.](https://arxiv.org/abs/2203.16921)), and some docstrings in the file still say "4x" and "faithful implementation".

- **Shallow features:** 3x3 conv, 1 to `nc = 96` channels.
- **Encoder:** 4 levels with channels `[96, 192, 384, 768]` and feature sizes `[64, 32, 16, 8]` for a 64 px input. Each level has `depth[i] = 2` Swin-Conv blocks (SCB). A 2x2 stride-2 conv does the downsampling between levels.
- **SCB (Swin-Conv block):** split the channels in half.
  - Swin half: LayerNorm, window self-attention (window 8, relative position bias, shifted windows in every second block), then an MLP (ratio 4). It uses `num_heads // 2 = 3` heads at every level.
  - Conv half: a residual `conv3x3 -> GELU -> conv3x3` block.
  - Concatenate the halves, then `Linear + LayerNorm`. There is no extra skip connection around the whole SCB.
  - When the feature map is not larger than the window (the 8 x 8 level), there is one window and no shift.
- **Decoder:** 3 up-steps. Each one is conv3x3 + PixelShuffle 2x, then concatenation with the encoder skip, a linear projection, and 2 SCBs.
- **Head:** LayerNorm, conv3x3, then `SRUpsample` (conv3x3, PixelShuffle 2x, LeakyReLU 0.2, conv3x3 to 1 channel).
- **Residual mode** (`residual_output=True`, used by every SCUNet script): no sigmoid, and the final conv weights and bias start at **zero**. The scripts add bicubic: `sr = bicubic(lr) + model(lr)`. So at step 0 the network is an exact bicubic upsampler and only learns the correction. (A test checks this.)
- **Fixed input size:** the attention masks and reshapes are built for `img_size = 64`. Other input sizes fail. `inference.py` handles full scenes by cutting them into 64 x 64 tiles.
- **Size:** 32,952,907 trainable parameters for `nc=96, depth=(2,2,2,2), scale=2` (counted on CPU; checked by a test).

## Loss functions

`scunet_train.py` defines `CombinedLoss(mse_weight, l1_weight, fft_weight)`:

- `MSE = mean((sr - hr)^2)` and `L1 = mean(|sr - hr|)` on pixels.
- `FFT = mean(| min(|F(sr)|, 100) - min(|F(hr)|, 100) |)`, where `F` is `torch.fft.rfft2` with `norm='ortho'`, computed in float32.
- If the FFT weight and the L1 weight are both 0, the loss is just `mse_weight * MSE`.
- Otherwise the weights are rescaled so they add up to 1: `w_mse = (1 - w_fft) * mse_weight / (mse_weight + l1_weight)`, and the same for L1. During training the FFT weight warms up as `fft_weight * min(1, epoch / 80)^2`.

The **committed** config uses MSE only (`mse_weight = 1.0, l1_weight = 0.0, fft_weight = 0.0`). A `CharbonnierLoss` class is defined but not used.

`train.py` (RRDBNet) and `swinIR_train.py` use `0.8 * L1 + 0.2 * FFT`.

## Training setup

Committed values in `src/scunet_train.py`:

| Setting | Value |
|---|---|
| Patch size (HR / LR) | 128 / 64 |
| Patches per image per epoch | `min(200, (H // 128) * (W // 128))` |
| Split | 90% train / 10% validation (random split of patch indices, seed 42) |
| Batch size | 4 |
| Epochs | 100 |
| Optimiser | AdamW, lr 2e-4, betas (0.9, 0.999), weight decay 1e-5 (not on biases or norm layers) |
| LR schedule | MultiStepLR, milestones [30, 60, 90, 120], gamma 0.5 (epoch 120 is never reached in a 100-epoch run) |
| Loss | MSE |
| Gradient clipping | max norm 1.0 |
| Mixed precision | fp16 autocast + GradScaler, only when CUDA is available |
| NaN guard | any batch with a NaN or inf loss is skipped |
| Seed | 42 (`torch`, `numpy`, CUDA) |
| Logging | TensorBoard, `runs/scunet-128patch-2x-v17-progressive-ease` |
| Checkpoints | `checkpoints/scunet_epochXXX.pth` every 10 epochs (model + optimiser), `checkpoints/best_scunet.pth` for the best validation **PSNR gain over bicubic** (ties broken by higher PSNR) |
| Resume | `CFG['resume'] = True` continues from the last `scunet_epoch*.pth` |

Validation reports PSNR and SSIM for both SCUNet and bicubic on the same fixed patches, and their difference ("gain").

## Committed config vs author-reported config

| Setting | Committed `scunet_train.py` | Author-reported run |
|---|---|---|
| Learning rate | 2e-4 | 1e-4 |
| Weight decay | 1e-5 | 5e-3 |
| Loss | 1.0 MSE | 0.8 L1 + 0.2 FFT |
| LR milestones | [30, 60, 90, 120], gamma 0.5 | [60, 100], gamma 0.5 |
| Degradation | curriculum (see above), module default 0.20 / 40 / 0.0002, blend 0.5 | PSF sigma 0.5, 12 looks, noise std 0.003 |
| Epochs | 100 | best at epoch 86 |
| Best validation PSNR | not stored in the repository | about 21.46 dB |
| Patch / scale / nc / depth | 128 / 2 / 96 / (2,2,2,2) | same |

Clues in the code that the file was changed many times:

- The comment above `milestones` says "decay by 0.5 at epochs 60 and 100", but the value is `[30, 60, 90, 120]`.
- The loss comments say "v1: 0.8/0.2 — plateaued at 16.8 dB", "v3: 0.7/0.3 + warm restart — hit 21.46 dB but collapsed from NaN cascade" and "v4 (this run): 0.8/0.2, lower LR", but the weights below them are MSE only.
- The run name is `scunet-128patch-2x-v17-progressive-ease`.
- The reported weight decay (5e-3) and loss (0.8 L1 + 0.2 FFT) are the same values that `swinIR_train.py` uses.

So the committed file is the last version that was tried. It is not necessarily the version that gave the reported result. The only place "21.46 dB" appears in the code is the "v3" comment above.

## Metrics

All metrics are computed on **normalised** patches (log scale, [0, 1]). SR and bicubic outputs are clamped to [0, 1] first.

**PSNR** (`skimage.metrics.peak_signal_noise_ratio`, `data_range = 1.0`):

```
MSE  = mean((HR - SR)^2)
PSNR = 10 * log10(1 / MSE)   [dB]
```

Higher is better. +1 dB means the MSE went down by about 21%.

**SSIM** (`skimage.metrics.structural_similarity`, `data_range = 1.0`, default 7 x 7 uniform window):

```
SSIM(x, y) = ((2 mu_x mu_y + C1) (2 sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) (sigma_x^2 + sigma_y^2 + C2))
C1 = (0.01 * 1)^2,  C2 = (0.03 * 1)^2
```

computed in each local window and averaged over the patch. The range is up to 1 (identical images). Higher is better.

**ENL** (Equivalent Number of Looks, `compute_enl` in `evaluate.py`):

```
ENL = mean(P)^2 / var(P)     over the whole patch P (NaN if var < 1e-10)
```

ENL is the classic speckle measure: a higher value means a smoother image with less speckle. Two cautions:

- In SAR practice, ENL is measured on **homogeneous** areas of **linear intensity** images. Here it is measured on whole patches with mixed content, after the log and percentile scaling. So the values are not comparable with ENL values from SAR literature.
- Higher is not automatically better for SR. Compare the ENL of SR with the ENL of HR: an SR ENL far above HR means the model is over-smoothing.

**Gain over bicubic:** mean SR metric minus mean bicubic metric, on the same patches. Bicubic is `F.interpolate(lr, scale_factor=2, mode='bicubic', align_corners=False)`.

## Evaluation scripts

- `evaluate.py` and `quick_benchmark.py` draw patches with the **module-default** degradation (sigma 0.20, 40 looks, noise 2e-4, blend 0.5). This is **not** the final curriculum stage used for validation during training (0.10, 96, 5e-5, 0.25). So the numbers will not match the training log.
- Patches are drawn at random every time (see [Patch sampling](#patch-sampling)), and the scripts do not seed NumPy. So results change a little from run to run. The `--seed` option of `quick_benchmark.py` and the fixed `default_rng(42)` in `evaluate.py` only fix *which image indices* are picked.
- `quick_benchmark.py` loops over the data once **per model**, so each model is scored on a different random set of patches. Treat small differences between models as noise.
- `quick_benchmark.py` feeds the same 64 px LR (made with 2x degradation) to the 4x models (SwinIR, RRDBNet). Their 256 px output is resized to 128 px with bicubic before the metrics. The first `--warmup_batches` batches are used for timing warm-up only and are not scored.
- `evaluate.py --save_images N` draws **new** random patches for the pictures. They are not the same patches as the ones in the metric summary.

## Full-scene inference

Code: `src/inference.py`.

1. Read band 1 of the scene.
2. **Default mode:** degrade the whole scene (same chain as training, module-default constants) and normalise HR and LR jointly with scene-level percentiles. **`--use_raw_lr`:** use the scene itself as the LR input, with HR-style normalisation (`log1p`, 1st/99th percentile stretch).
3. Reflect-pad the LR scene to a multiple of 64 and cut it into non-overlapping 64 x 64 tiles.
4. Run the tiles in batches: `sr = clamp(bicubic(tile) + model(tile), 0, 1)`. Paste each 128 x 128 result into the output canvas.
5. Crop the padding, apply optional masked Gaussian smoothing (`--smooth_sigma`), then undo the normalisation back to `uint16` with `expm1`.
6. Write a GeoTIFF with the pixel size halved (`a / 2`, `e / 2`, same origin and CRS), LZW compression and 256 x 256 internal tiles. Also write a contrast-stretched display GeoTIFF (`*_display.tif`) and a PNG preview (`*_preview.png`).

**Georeferencing caveat:** in default mode the scene is first shrunk to half size, so the SR output has about the **same** number of pixels as the input. The pixel size is still halved, so the output claims to cover only a quarter of the real ground area. Only `--use_raw_lr` gives a true 2x enlargement with correct georeferencing.

## Known issues

These are documented, not fixed, so that the original program logic stays unchanged.

**Training and configuration**

- The committed SCUNet config is not the author-reported config (see [above](#committed-config-vs-author-reported-config)). No checkpoint or log of the reported run is in the repository.
- `train.py` (RRDBNet) and `swinIR_train.py` were written for 4x degradation, but `sar_dataset.SCALE_FACTOR` is now 2. With 256 px HR patches the LR patch is 128 px. RRDBNet then outputs 512 px against a 256 px target (the loss call fails), and SwinIR, built for a 64 px input, fails in its forward pass. To retrain them, set `SCALE_FACTOR = 4` in `sar_dataset.py` first (and set it back to 2 for SCUNet).
- In `train.py`, a comment says 23 RRDB blocks are "~16.7M params". The real count for this implementation is 22,006,689.
- The training and validation patches come from the **same scenes** (random patch split, not a split by scene). Windows can even overlap. This can overestimate performance on new scenes.

**Evaluation**

- Evaluation degradation (module defaults) is different from the training-validation degradation (final curriculum stage).
- In `quick_benchmark.py`, the 4x models are resized to the 2x target, and each model sees different random patches.

**Inference**

- Default mode halves the pixel size even though the pixel count does not double (see [Full-scene inference](#full-scene-inference)). Use `--use_raw_lr` when georeferencing matters.
- Tiles do not overlap, so tile borders can sometimes be visible.

**Data and utility scripts**

- `data/explore_dataset_filtered.csv` holds only a header row. (The notebook output shows that the same Los Angeles filter found 8 items: 4 GEO and 4 SLC.)
- `notebooks/STAC_to_CSV.ipynb` uses `ThreadPoolExecutor` and `as_completed` without importing them, so it fails on a fresh kernel. `src/explore_dataset.py` does the same job and works.
- `download_diverse.py` takes one scene from each occupied cell of a 10 x 10 grid over the data's latitude/longitude range. With the committed CSV only 13 cells are occupied, so the other 12 of the 25 scenes are random GEO scenes (seed 42).
- `download_geotiff_la.py` saves its scene into `data/samples/geotiffs/`. If you run it before training, that scene becomes part of the training set.
- `cuda_test.py` raises an error when no NVIDIA GPU is present (it calls `torch.cuda.get_device_name(0)` without a check).
- `approch_other.py` and `evaluate.py --image_out_dir` use paths relative to the **current folder**, not to the script. Run them from the project root.
- `scunet_model.py`, run directly, tests the sigmoid mode (`residual_output=False`), not the residual mode used in training.
- `degradation.py` shows an older 4x setting (sigma 1.2, 4 looks, noise 0.01) that is applied **after** normalisation. It is not used in training.
