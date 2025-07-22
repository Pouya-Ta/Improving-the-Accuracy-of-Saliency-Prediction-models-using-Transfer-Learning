import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import ml_net_model
import config
import os

def load_input(path):
    if path.endswith('.npy'):
        return np.load(path)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.shape_c, config.shape_r))
        return img.astype(np.float32)

def compare_saliency(image_path):
    image = load_input(image_path)
    image_batch = np.expand_dims(image, axis=0)

    # Load both models
    model_frozen = ml_net_model(freeze_vgg=True)
    model_frozen.load_weights("weights/frozen/mlnet.h5")

    model_trainable = ml_net_model(freeze_vgg=False)
    model_trainable.load_weights("weights/trainable/mlnet.h5")

    pred_frozen = model_frozen.predict(image_batch)[0, :, :, 0]
    pred_trainable = model_trainable.predict(image_batch)[0, :, :, 0]

    # Normalize for visualization
    pred_frozen = cv2.normalize(pred_frozen, None, 0, 255, cv2.NORM_MINMAX)
    pred_trainable = cv2.normalize(pred_trainable, None, 0, 255, cv2.NORM_MINMAX)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Input Image")
    axes[1].imshow(pred_frozen, cmap='hot')
    axes[1].set_title("Saliency (VGG Frozen)")
    axes[2].imshow(pred_trainable, cmap='hot')
    axes[2].set_title("Saliency (VGG Trainable)")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
compare_saliency("path/to/test/image.npy")
compare_saliency("path/to/test/image.jpg")
