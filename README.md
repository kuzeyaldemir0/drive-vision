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

### Current best: vanilla U-Net, LR 0.001

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 79.4% | 73.7% |
| mIoU | 21.9% | 19.9% |
| Loss | 0.691 | 0.891 |

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

### Loss function experiments (from earlier 30-epoch runs, directionally valid)

| Loss Function | Val mIoU | Notes |
|---------------|----------|-------|
| **Cross-entropy** | **~20%** | Best — simple and stable |
| Focal loss (γ=2) | ~18.5% | Too aggressive for small dataset |
| Weighted CE (cap 10) | ~15.8% | Weights too extreme |

Finding: on a 160-image dataset, class imbalance techniques hurt rather than help. The model lacks sufficient examples of rare classes regardless of loss weighting.

### Class distribution (KITTI training set)

The top 4 classes (vegetation, road, sky, terrain) cover 74% of all pixels. The bottom 10 classes combined cover less than 3%. Motorcycle has just 917 pixels (0.01%) across all training images.

### Planned experiments

- Increased filter counts for more model capacity
- Attention U-Net variant
- Lightweight U-Net (depthwise separable convolutions)
- Higher resolution training (256×768)
- Horizontal flip augmentation
- DeepLabV3+ pretrained comparison

## Dataset

KITTI Semantic Segmentation — 200 annotated urban driving images from Karlsruhe, Germany. 160 train / 40 validation split. 19 evaluation classes following the Cityscapes label mapping. Images resized to 128×384 for fast iteration (original ~375×1242).

## Project Structure

```
drive-vision/
├── notebook.ipynb      # Data pipeline, training, evaluation, visualization
├── unet.py             # U-Net model architecture
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