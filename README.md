# Drive Vision

Semantic segmentation for autonomous driving perception — a from-scratch implementation and comparative study of U-Net architectures on the KITTI dataset.

## Overview

This project explores how architectural choices, loss functions, and hyperparameters affect semantic segmentation performance on urban driving scenes. The baseline U-Net is implemented from scratch in TensorFlow/Keras and trained on the KITTI Semantic Segmentation dataset (200 images, 19 classes + ignore class). All models are built, trained, and evaluated on an Apple M4 GPU.

## Architecture

The baseline U-Net follows the original Ronneberger et al. (2015) paper:

- **Encoder:** 4 levels of double Conv2D (3×3, ReLU, same padding) + MaxPooling2D, filter progression 64 → 128 → 256 → 512
- **Bottleneck:** double Conv2D with 1024 filters
- **Decoder:** 4 levels of Conv2DTranspose (2×2, stride 2) + skip connection concatenation + double Conv2D
- **Output:** 1×1 Conv2D with 20 channels (19 classes + ignore), raw logits
- **Loss:** SparseCategoricalCrossentropy (from_logits=True)
- **Optimizer:** Adam
- **Total parameters:** ~31M

## Experiments & Results

### Current best: LR 0.001, plain cross-entropy, 20 epochs

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 79.4% | 73.7% |
| mIoU | 21.9% | 19.9% |
| Loss | 0.691 | 0.891 |

### Learning rate experiment

| Learning Rate | Val Accuracy | Val mIoU | Val Loss |
|---------------|-------------|----------|----------|
| 0.0001 | 71.9% | 17.5% | 0.958 |
| **0.001** | **73.7%** | **19.9%** | **0.891** |

Higher learning rate converges ~3x faster and reaches better final performance.

### Loss function experiments

| Loss Function | Val mIoU | Notes |
|---------------|----------|-------|
| **Cross-entropy** | **20.2%** | Best performer — simple and stable |
| Focal loss (γ=2) | 18.5% | Too aggressive for small dataset |
| Focal loss + dropout 0.4 | 18.0% | Dual constraints hurt capacity |
| Weighted CE (cap 10) | 15.8% | Weights too extreme even when capped |

Finding: on a 160-image dataset, class imbalance techniques hurt rather than help. The model doesn't have enough examples of rare classes to learn them regardless of loss weighting.

### Regularization experiments

| Regularization | Val mIoU | Notes |
|----------------|----------|-------|
| None | 20.2% | Best |
| Dropout 0.4 at bottleneck | 20.2% | No measurable difference |

### Class distribution (KITTI training set)

| Class | Pixels | Percentage |
|-------|--------|------------|
| Vegetation (8) | 2,349,785 | 29.9% |
| Road (0) | 1,832,617 | 23.3% |
| Sky (10) | 876,581 | 11.2% |
| Terrain (9) | 767,460 | 9.8% |
| Building (2) | 609,024 | 7.7% |
| Car (13) | 482,763 | 6.1% |
| Sidewalk (1) | 313,266 | 4.0% |
| Ignore (19) | 294,016 | 3.7% |
| Pole (5) | 118,929 | 1.5% |
| Fence (4) | 54,935 | 0.7% |
| Traffic sign (7) | 40,967 | 0.5% |
| Wall (3) | 39,104 | 0.5% |
| Traffic light (6) | 27,666 | 0.4% |
| Truck (14) | 19,878 | 0.3% |
| Train (16) | 17,726 | 0.2% |
| Bus (15) | 6,122 | 0.1% |
| Bicycle (18) | 5,254 | 0.1% |
| Person (11) | 4,789 | 0.1% |
| Rider (12) | 2,521 | 0.03% |
| Motorcycle (17) | 917 | 0.01% |

Extreme imbalance: the top 4 classes cover 74% of all pixels. The bottom 10 classes combined cover less than 3%.

### Planned experiments

- Higher resolution training (256×512 or 368×1232)
- Horizontal flip augmentation
- Learning rate scheduling (CosineDecay)
- Attention U-Net variant
- Lightweight U-Net (depthwise separable convolutions)
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