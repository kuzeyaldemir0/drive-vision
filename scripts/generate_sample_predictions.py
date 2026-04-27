"""Generate sample prediction images for the README from the best fine-tuned model.

Runs inference on the validation set, computes per-image mIoU, picks a few
high-mIoU examples, and saves side-by-side input / ground-truth / prediction
visualizations to images/.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mobile_u_net import load_preprocess_mobilenet

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "checkpoints" / "mobilenet_finetuned_skip_flip_lr1e-4_best.keras"
OUTPUT_DIR = REPO_ROOT / "images"
NUM_SAMPLES = 3
NUM_CLASSES_EVAL = 19  # ignore class 19


def per_image_miou(gt: np.ndarray, pred: np.ndarray) -> float:
    ious = []
    for c in range(NUM_CLASSES_EVAL):
        gt_c = gt == c
        pred_c = pred == c
        if not gt_c.any():
            continue
        intersection = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {CHECKPOINT}")
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)

    image_ds = tf.data.Dataset.list_files(
        str(REPO_ROOT / "data/kitti/training/image_2/*.png"), shuffle=False
    )
    mask_ds = tf.data.Dataset.list_files(
        str(REPO_ROOT / "data/kitti/training/semantic/*.png"), shuffle=False
    )
    dataset = tf.data.Dataset.zip((image_ds, mask_ds))
    dataset = dataset.map(load_preprocess_mobilenet)
    val_ds = dataset.skip(160).batch(1).prefetch(tf.data.AUTOTUNE)

    samples = []
    print("Running inference on validation set...")
    for idx, (image, mask) in enumerate(val_ds):
        logits = model.predict(image, verbose=0)
        pred = tf.argmax(logits, axis=-1).numpy()[0]
        gt = tf.squeeze(mask).numpy()
        miou = per_image_miou(gt, pred)
        samples.append(
            {
                "idx": idx,
                "miou": miou,
                "image": image[0].numpy(),
                "gt": gt,
                "pred": pred,
            }
        )

    samples.sort(key=lambda s: -s["miou"])

    print("\nTop 10 validation images by mIoU:")
    for s in samples[:10]:
        print(f"  val[{s['idx']:02d}]  mIoU = {s['miou']:.4f}")

    for rank, s in enumerate(samples[:NUM_SAMPLES], start=1):
        fig, axes = plt.subplots(3, 1, figsize=(10, 7))
        img = np.clip((s["image"] + 1) / 2, 0, 1)

        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(s["gt"], cmap="tab20", vmin=0, vmax=19)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(s["pred"], cmap="tab20", vmin=0, vmax=19)
        axes[2].set_title(f"Prediction (per-image mIoU: {s['miou']:.1%})")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"sample_prediction_{rank}.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
