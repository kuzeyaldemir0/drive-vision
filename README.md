# Drive Vision

Semantic segmentation for autonomous driving perception — a from-scratch implementation and comparative study of U-Net architectures on the KITTI dataset.

## Overview

This project explores how architectural choices, loss functions, and hyperparameters affect semantic segmentation performance on urban driving scenes. The baseline U-Net is implemented from scratch in TensorFlow/Keras and trained on the KITTI Semantic Segmentation dataset (200 images, 19 classes + ignore class). All models are trained and evaluated on an Apple M4 GPU.

## Architecture

The baseline U-Net follows the original Ronneberger et al. (2015) paper:

- **Encoder:** 4 levels of double Conv2D (3×3, ReLU, same padding) + MaxPooling2D, filter progression 64 → 128 → 256 → 512
- **Bottleneck:** double Conv2D with 1024 filters
- **Decoder:** 4 levels of Conv2DTranspose (2×2, stride 2) + skip connection concatenation + double Conv2D
- **Output:** 1×1 Conv2D with 20 channels (19 classes + ignore), raw logits
- **Loss:** SparseCategoricalCrossentropy (from_logits=True)
- **Optimizer:** Adam (lr=0.001)
- **Input resolution:** 128×384
- **Total parameters:** ~31M

## Experiments & Results

All experiments use 20 epochs, batch size 8, 160/40 train/val split unless noted otherwise.

### Current best: MobileNet U-Net with skip connections + flip augmentation + encoder fine-tuning (lr=1e-4)

| Metric | Train (epoch 7) | Validation (epoch 7, best) |
|--------|-------|------------|
| Accuracy | 91.2% | 82.6% |
| mIoU | 39.4% | **28.3%** |
| Loss | 0.275 | 0.689 |

Saved via `save_best_only=True`. Final validation mIoU after a sequence of architectural and data interventions (skip connections, horizontal flip augmentation, encoder unfreezing). Roughly +8.4 points over the from-scratch U-Net baseline.

### Learning rate comparison

| Learning Rate | Val Accuracy | Val mIoU | Val Loss | Notes |
|---------------|-------------|----------|----------|-------|
| 0.0001 | 71.9% | 17.5% | 0.958 | Too slow |
| **0.001** | **73.7%** | **19.9%** | **0.891** | **Best** |
| 0.01 | 32.1% | 1.8% | 2.056 | Gradient explosion, model collapsed |

### Batch normalization experiment

| Config | Train mIoU | Val mIoU | Notes |
|--------|-----------|----------|-------|
| No batch norm | 21.9% | 19.9% | Stable generalization |
| With batch norm | 27.8% | 6.6% | Catastrophic overfitting — batch statistics unreliable with batch size 8 on 160 images |

### Transfer learning: MobileNetV2 encoder (no skip connections)

First step toward transfer learning. The hand-written encoder is replaced with a frozen ImageNet-pretrained MobileNetV2 (`include_top=False`, `weights='imagenet'`). The decoder is built from scratch as 5 Conv2DTranspose + Conv2D blocks with no skip connections — deliberately, so the effect of the pretrained encoder can be measured in isolation before adding skips.

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 73.5% | 66.5% |
| mIoU | 18.0% | 15.9% |
| Loss | 0.840 | 1.177 |

Result is ~4 points lower val mIoU than the from-scratch U-Net baseline. Two factors at play: (1) val mIoU was still climbing at epoch 20 (0.140 → 0.142 → 0.146 → 0.159 across epochs 17–20), so this run was undertrained; (2) without skips, the decoder reconstructs 128×384 masks from a 4×12 feature map with only deep semantic channels — no edges or textures. Skip connections are the obvious next step.

### Transfer learning: MobileNetV2 encoder with skip connections

Added 4 skip connections to the frozen MobileNetV2 encoder, tapping `block_1_expand_relu`, `block_3_expand_relu`, `block_6_expand_relu`, and `block_13_expand_relu`. The decoder uses 5 Conv2DTranspose blocks: 4 with concatenated skips + 1 final no-skip upsample (MobileNetV2's first layer is stride-2, so there's no full-resolution encoder feature map to skip from). Encoder remains frozen.

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 84.0% | 80.0% |
| mIoU | 26.4% | 24.0% |
| Loss | 0.537 | 0.725 |

Val mIoU improved +8.1 points over the no-skip version and +4.1 over the from-scratch U-Net baseline — a clean confirmation that the decoder needs intermediate encoder features to reconstruct spatial detail. Train mIoU kept climbing past epoch 15 while val mIoU plateaued around 24%, indicating the model has reached the data ceiling for a 160-image dataset. Best validation was actually epoch 18 (val mIoU 25.0%); final saved weights are from epoch 20 due to lack of `save_best_only`. The next high-leverage move is data augmentation, not further architectural changes.

### Transfer learning: MobileNetV2 encoder with skip connections + horizontal flip augmentation

Added deterministic horizontal flip augmentation to the prior skip-connection setup. Every training image is mirrored via `tf.image.flip_left_right` (image and mask synchronously) and concatenated with the originals, doubling the effective training set from 160 to 320 examples. The validation set is left unaugmented to keep comparisons against prior baselines clean. Encoder still frozen, all other settings (skip connections, lr=1e-3, batch size 8, 20 epochs) match the prior run. `ModelCheckpoint(save_best_only=True, monitor='val_miou')` was added to preserve the best epoch's weights independent of the final epoch.

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 88.2% | 81.8% |
| mIoU | 31.7% | 26.9% |
| Loss | 0.381 | 0.643 |

Best epoch was **18** (val mIoU 27.4%, val accuracy 82.2%, val loss 0.641) — saved separately via the checkpoint callback. Final epoch metrics shown above.

Augmentation lifted val mIoU by +2.4 points on the best epoch (25.0% → 27.4%) and +2.9 points on the final epoch (24.0% → 26.9%) versus the no-flip skip-connection run. Faster early convergence is also visible: epoch 1 val mIoU was 9.5% vs. 4.3% in the no-flip run, suggesting the augmented dataset gave the model meaningfully more diverse signal per epoch. Train/val gap widened slightly (~5 points), consistent with 2× the gradient updates per epoch on the same underlying 160-image information content. Diminishing returns relative to the +8.1 jump from skip connections — the bottleneck is now closer to dataset ceiling than architectural deficit.

### Transfer learning: MobileNetV2 encoder fine-tuning

After establishing skip + flip as the strongest frozen-encoder configuration (val mIoU 27.4% best), the natural next step was to unfreeze the MobileNetV2 encoder and fine-tune it at a low learning rate, allowing the ImageNet features to refine toward the driving-scene domain. Two A/B runs at different learning rates were performed; both load the same `mobilenet_frozen_skip_flip_lr0.001_best.keras` checkpoint and differ only in LR, isolating the LR effect.

#### Fine-tuning at lr=1e-5 (10 epochs)

The canonical "safe" fine-tuning LR — 100× lower than the original training LR. Encoder fully unfrozen and the model recompiled with Adam(lr=1e-5).

| Metric | Train (epoch 10) | Validation (epoch 10, best) |
|--------|-------|------------|
| Accuracy | 90.2% | 82.9% |
| mIoU | 35.1% | 27.8% |
| Loss | 0.312 | 0.633 |

Val mIoU climbed monotonically every epoch from 27.4% → 27.8% (+0.4 points). No overfitting: train/val gap stable at ~7 points, val loss only crept slightly. The encoder made conservative refinements to ImageNet features without disrupting them — but progress was glacial.

#### Fine-tuning at lr=1e-4 (10 epochs)

A 10× higher LR than the canonical fine-tuning rate, but still 10× lower than the original training LR. Same starting checkpoint, same architecture, same dataset, same epoch budget.

| Metric | Train (epoch 7) | Validation (epoch 7, best) |
|--------|-------|------------|
| Accuracy | 91.2% | 82.6% |
| mIoU | 39.4% | **28.3%** |
| Loss | 0.275 | 0.689 |

Val mIoU peaked at 28.3% on epoch 7, then declined to 28.2% by epoch 10. Train mIoU kept climbing aggressively (35.1% → 42.2% across the run) and train/val gap doubled to ~14 points — early overfitting clearly visible. `save_best_only=True` preserved the epoch-7 weights as the run's saved output.

**LR comparison takeaway.** 1e-4 achieved a higher peak (28.3% vs 27.8%) but began overfitting around epoch 7, while 1e-5 was still climbing slowly at epoch 10. The +0.5 mIoU point gain from 1e-4 came at the cost of significantly faster overfitting — the canonical "fine-tuning LR tradeoff" in transfer learning. Combined with diminishing returns from earlier interventions (skip +8.1, flip +2.4, fine-tune +0.5–0.9), this confirms the model is essentially at the data ceiling for KITTI's 160-image training set.

### Loss function experiments (from earlier 30-epoch runs, directionally valid)

| Loss Function | Val mIoU | Notes |
|---------------|----------|-------|
| **Cross-entropy** | **~20%** | Best — simple and stable |
| Focal loss (γ=2) | ~18.5% | Too aggressive for small dataset |
| Weighted CE (cap 10) | ~15.8% | Weights too extreme |

Finding: on a 160-image dataset, class imbalance techniques hurt rather than help. The model lacks sufficient examples of rare classes regardless of loss weighting.

### Class distribution (KITTI training set)

The top 4 classes (vegetation, road, sky, terrain) cover 74% of all pixels. The bottom 10 classes combined cover less than 3%. Motorcycle has just 917 pixels (0.01%) across all training images.

### Summary

| Run | Val mIoU |
|---|---|
| Baseline U-Net (from scratch) | 19.9% |
| MobileNetV2 encoder, no skips | 15.9% |
| MobileNetV2 encoder, with skips | 25.0% |
| MobileNetV2 + skips + flip aug | 27.4% |
| MobileNetV2 + skips + flip + fine-tune (lr=1e-5) | 27.8% |
| **MobileNetV2 + skips + flip + fine-tune (lr=1e-4)** | **28.3%** |

The progression follows a clean diminishing-returns curve. Skip connections (+8.1 pts) were the biggest single gain, addressing a clear architectural deficit — the decoder needed intermediate encoder features to reconstruct spatial detail. Augmentation (+2.4 pts) and encoder fine-tuning (+0.5–0.9 pts depending on LR) confirmed the bottleneck shifting from architecture to data. The remaining gap to production-quality segmentation isn't an architecture or hyperparameter problem; it's a dataset-size problem. KITTI's 160 training images cap practical performance regardless of further model tweaks.

### Possible future directions

These weren't pursued in this project but are natural follow-ups for pushing past the data ceiling:

- Pretraining on Cityscapes (5,000 images, same Cityscapes label scheme) before fine-tuning on KITTI
- Horizontal flip augmentation on the from-scratch U-Net for direct A/B comparison
- Brightness / contrast jitter to simulate weather and time-of-day variation
- Attention U-Net or DeepLabV3+ as alternative architectures
- Higher resolution training (256×768) using more of KITTI's native 1242px width
- Encoder fine-tuning with selective unfreezing (e.g. only top blocks) and LR warmup schedules

## Dataset

KITTI Semantic Segmentation — 200 annotated urban driving images from Karlsruhe, Germany. 160 train / 40 validation split. 19 evaluation classes following the Cityscapes label mapping. Images resized to 128×384 for fast iteration (original ~375×1242).

## Project Structure

```
drive-vision/
├── notebook.ipynb      # Data pipeline, training, evaluation, visualization
├── unet.py             # From-scratch U-Net architecture + preprocessing
├── mobile_u_net.py     # MobileNetV2-based U-Net + flip augmentation helper
├── checkpoints/        # Saved model weights (gitignored)
├── logs/               # TensorBoard logs (gitignored)
├── data/               # KITTI dataset (gitignored)
├── requirements.txt
└── README.md
```

## Stack

Python 3.11, TensorFlow 2.18.1, Keras, NumPy, Matplotlib, Apple M4 GPU (tensorflow-metal)

## Usage

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation in `notebook.ipynb`. Model architecture in `unet.py`.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.

## License

MIT