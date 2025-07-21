# Improving the Accuracy of Saliency Prediction Models Using Transfer Learning

## Overview
This project investigates whether transfer learning can enhance the accuracy of saliency prediction models. Saliency prediction aims to estimate which regions of an image attract human visual attention. We utilize deep learning techniques, specifically ML-Net architecture with a VGG16 backbone, to improve saliency map predictions by leveraging pretrained weights and custom prior learning layers.

To assess the role of transfer learning, we implement **two training strategies**:

### 🔹 1. Frozen VGG-16 (Feature Extractor Mode)
- VGG-16 layers are initialized with ImageNet-pretrained weights and **frozen** (not trainable).
- Only the ML-Net-specific layers are trained.
- This tests how well general features transfer to the saliency task.

### 🔹 2. Fine-tuned VGG-16 (Full Training Mode)
- VGG-16 layers are initialized with ImageNet weights but **remain trainable**.
- Both VGG-16 and ML-Net layers are trained together.
- This allows the backbone to adapt its features specifically for saliency prediction.

By comparing these two settings, we aim to measure the effect of transfer learning and fine-tuning on prediction quality.

## Key Features
- Uses **transfer learning** with pretrained VGG16 as the feature extractor.
- Implements **ML-Net** architecture adapted for saliency prediction.
- Incorporates a **prior learning layer** (custom `EltWiseProduct`) to refine saliency maps.
- Supports training and testing pipelines with dataset preprocessing.
- Evaluates effectiveness of transfer learning on public datasets (e.g., SALICON).
- Provides visual and performance comparison between frozen and fine-tuned VGG backbones.

## Installation and Setup
- Requires Python 3.x with TensorFlow/Keras and necessary dependencies.
- Supports training and testing on GPUs (recommended).
- Download dataset from [Google Drive](https://drive.google.com/drive/folders/1FUFiysRjSVP4344WVPmzvDDYtAKCCqNt?usp=drive_link).
- Follow usage instructions in `main.py` to train or predict saliency maps.

## Usage
Run the main script with the appropriate phase argument:

- **Training (Frozen VGG16):**
```bash
python main.py train --freeze_backbone True
````


- **Training (Fine-tuned VGG16):**
```bash
python main.py train --freeze_backbone False
````

* **Testing:**

```bash
python main.py test path_to_test_images/
```

Model weights will be saved during training and loaded for testing.


## Project Structure

* `main.py`: Entry point for training and testing.
* `model.py`: Defines ML-Net model with VGG16 backbone and prior learning layer.
* `eltwise_product.py`: Custom Keras layer implementing prior learning.
* `utilities.py`: Image and saliency map preprocessing and postprocessing functions.
* `config.py`: Configuration variables for paths, image sizes, batch sizes, etc.
* `weights/`: Folder for pretrained or saved model weights.



## Contributors

* Pouya Taghipour — LinkedIn: , Email: , GitHub:
* Ali Kargar — LinkedIn: , Email: , GitHub:
* Arsalan
* Amirhossein
* Mahtab
* Mahsa
* Niloofar
* Mohammad Hossein
* Amitis

## Supervisors

* Alireza Ghafouri — LinkedIn: , Email: , GitHub:
* Morteza Moziri — LinkedIn: , Email: , GitHub:


