# Drive Vision

Semantic segmentation for autonomous driving perception — a from-scratch implementation and comparative study of U-Net architectures on the KITTI dataset.

## Overview

This project explores how architectural choices affect semantic segmentation performance on urban driving scenes. The baseline U-Net is implemented from scratch in TensorFlow/Keras and trained on the KITTI Semantic Segmentation dataset (200 images, 19 classes + ignore). Future variants will benchmark architectural modifications against this baseline.

### Planned model comparison

| Model | Status | Description |
|-------|--------|-------------|
| U-Net (from scratch) | ✅ Trained | Baseline encoder-decoder with skip connections |
| Attention U-Net | 🔲 Planned | Learned attention gates on skip connections |
| Lightweight U-Net | 🔲 Planned | Depthwise separable convolutions (MobileNet-style) |
| DeepLabV3+ (fine-tuned) | 🔲 Planned | Pretrained SOTA reference |

## Results

### Baseline U-Net — 30 epochs, batch size 8, 128×384 input

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 82% | 74% |
| Loss | 0.59 | 0.89 |

The model successfully learns to segment major scene classes (road, vegetation, terrain, sky, buildings) within 30 epochs. Training curves show the onset of overfitting after ~20 epochs — the train/val gap widens as training accuracy continues climbing while validation accuracy plateaus around 74%.

### Batch size experiment (5 epochs each)

| Batch size | Steps/epoch | Val accuracy | Val loss |
|------------|-------------|--------------|----------|
| 4 | 40 | 63.7% | 1.328 |
| 8 | 20 | 62.0% | 1.421 |
| 16 | 10 | 35.8% | 2.115 |

Batch size 8 selected for the tighter train/val gap despite slightly slower early convergence.

### Next steps

- Add dropout and batch normalization to address overfitting
- Experiment with class weights for underrepresented classes
- Train at higher resolution (256×512 or 368×1232)
- Implement architectural variants for comparison

## Architecture

The baseline U-Net follows the original Ronneberger et al. (2015) architecture:

- **Encoder:** 4 levels of double Conv2D (3×3, ReLU, same padding) + MaxPooling2D, filter progression 64 → 128 → 256 → 512
- **Bottleneck:** double Conv2D with 1024 filters
- **Decoder:** 4 levels of Conv2DTranspose (2×2, stride 2) + skip connection concatenation + double Conv2D
- **Output:** 1×1 Conv2D with 20 channels (19 classes + ignore), raw logits with SparseCategoricalCrossentropy(from_logits=True)

Total parameters: ~31M

## Dataset

KITTI Semantic Segmentation — 200 annotated urban driving images from Karlsruhe, Germany. Split: 160 train / 40 validation. 19 evaluation classes following the Cityscapes label mapping (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle).

## Stack

Python 3.11, TensorFlow 2.18.1, Keras, NumPy, Matplotlib, Apple M4 GPU (tensorflow-metal)

## Usage

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training and evaluation are in `notebook.ipynb`. Model architecture is in `unet.py`.

## License

MIT