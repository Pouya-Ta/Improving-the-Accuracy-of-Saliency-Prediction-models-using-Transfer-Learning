from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os, cv2, sys
import numpy as np
from config import *
from model import ml_net_model, loss
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def generator(b_s, phase_gen='train'):
    """
    Data generator function for training and validation phases.
    Yields batches of preprocessed images and corresponding ground truth saliency maps.
    Now loads .npy files directly (already preprocessed).

    Args:
        b_s (int): Batch size.
        phase_gen (str): 'train' or 'val' to determine dataset to load.

    Yields:
        Tuple of (images_batch, maps_batch), both numpy arrays ready for model input.
    """
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith('.npy')]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith('.npy')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith('.npy')]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith('.npy')]
    else:
        raise NotImplementedError("Generator phase must be 'train' or 'val'")

    images.sort()
    maps.sort()
    target_size = (28, 28)  # Resize target for saliency maps

    counter = 0
    while True:
        batch_images = [np.load(f) for f in images[counter:counter + b_s]]
        batch_maps = [
            cv2.resize(np.load(f), target_size, interpolation=cv2.INTER_AREA)
            for f in maps[counter:counter + b_s]
        ]
        yield np.stack(batch_images), np.expand_dims(np.stack(batch_maps), axis=-1)
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    """
    Data generator function for test phase.
    Yields batches of preprocessed test images without ground truth.
    Now loads .npy files directly (already preprocessed).

    Args:
        b_s (int): Batch size.
        imgs_test_path (str): Folder path of test images.

    Yields:
        Batch of preprocessed images.
    """
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith('.npy')]
    images.sort()

    counter = 0
    while True:
        batch_images = [np.load(f) for f in images[counter:counter + b_s]]
        yield np.stack(batch_images)
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    phase = sys.argv[1]  # Either 'train' or 'test'

    # Whether to freeze VGG-16 layers or not
    freeze_backbone = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True  # default: True

    # Build the ML-Net model
    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10, freeze_vgg=freeze_backbone)

    # Compile the model
    sgd = SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(optimizer=sgd, loss=loss)

    if phase == 'train':
        print("Training ML-Net")
        model.fit(
            generator(b_s=b_s),
            steps_per_epoch=nb_imgs_train // b_s,
            epochs=nb_epoch,
            validation_data=generator(b_s=b_s, phase_gen='val'),
            validation_steps=nb_imgs_val // b_s,
            callbacks=[
                EarlyStopping(patience=10),
                ModelCheckpoint('weights/frozen/mlnet.h5', save_best_only=True, monitor='val_loss')
            ]
        )

    elif phase == "test":
        output_folder = ''
        if len(sys.argv) < 3:
            raise SyntaxError("For test phase, provide test images path as second argument.")
        imgs_test_path = sys.argv[2]
        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith('.npy')]
        file_names.sort()
        nb_imgs_test = len(file_names)
        print("Load trained weights")
        model.load_weights('mlnet_best.h5')
        print(f"Predict saliency maps for {imgs_test_path}")
        predictions = model.predict(generator_test(b_s=1, imgs_test_path=imgs_test_path), steps=nb_imgs_test)
        for pred, name in zip(predictions, file_names):
            # Assume original shape is stored in .npy or known, otherwise skip resizing
            res = pred[0]
            cv2.imwrite(output_folder + name.replace('.npy', '.png'), (res * 255).astype(np.uint8))
    else:
        raise NotImplementedError("Phase must be 'train' or 'test'")