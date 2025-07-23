import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from model import ml_net_model
import config

# Paths to image/map folders
image_dir = "/kaggle/input/salicon/test_npys/data/processed/images/test/"
map_dir   = "/kaggle/input/salicon/test_npys/data/processed/maps/test/"

def load_input_image(image_path):
    img = np.load(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = cv2.resize(img, (config.shape_c, config.shape_r))
    return img.astype(np.float32)

def load_map(map_path):
    sal_map = np.load(map_path)
    sal_map = cv2.resize(sal_map, (config.shape_c, config.shape_r))
    return sal_map.astype(np.float32)

def compare_saliency_from_index(index):
    # Sort filenames to ensure consistent matching
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
    map_files   = sorted([f for f in os.listdir(map_dir) if f.endswith('.npy')])

    # Safety check
    assert len(image_files) == len(map_files), "Mismatch in image/map count"
    assert 0 <= index < len(image_files), f"Index {index} out of range"

    # Get full paths to selected sample
    image_path = os.path.join(image_dir, image_files[index])
    map_path   = os.path.join(map_dir, map_files[index])

    # Load data
    image = load_input_image(image_path)
    gt_map = load_map(map_path)
    image_batch = np.expand_dims(image, axis=0)

    # Load models and weights
    model_frozen = ml_net_model(freeze_vgg=True)
    model_frozen.load_weights("weights/frozen/mlnet.h5")

    model_trainable = ml_net_model(freeze_vgg=False)
    model_trainable.load_weights("weights/trainable/mlnet.h5")

    # Predict saliency maps
    pred_frozen = model_frozen.predict(image_batch)[0, :, :, 0]
    pred_trainable = model_trainable.predict(image_batch)[0, :, :, 0]

    # Normalize for visualization
    pred_frozen = cv2.normalize(pred_frozen, None, 0, 255, cv2.NORM_MINMAX)
    pred_trainable = cv2.normalize(pred_trainable, None, 0, 255, cv2.NORM_MINMAX)
    gt_map = cv2.normalize(gt_map, None, 0, 255, cv2.NORM_MINMAX)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Input Image")

    axes[1].imshow(gt_map, cmap='hot')
    axes[1].set_title("Ground Truth Map")

    axes[2].imshow(pred_frozen, cmap='hot')
    axes[2].set_title("Prediction (Frozen VGG)")

    axes[3].imshow(pred_trainable, cmap='hot')
    axes[3].set_title("Prediction (Trainable VGG)")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ✅ Example usage: index 5
# Run this to test one sample:
compare_saliency_from_index(index=5)
