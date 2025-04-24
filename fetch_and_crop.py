import tensorflow as tf
from keras import layers, Model, Sequential, optimizers, applications
import numpy as np
from tqdm import tqdm
import os
from keras import backend as K
import csv
from PIL import Image
from tqdm import tqdm


def crop_patch_pairs(n_random_crops_train, n_random_crops_val, train_hr_dir="amls2_coursework/datasets/DIV2K_train_HR",
                     train_lr_dir="amls2_coursework/datasets/DIV2K_train_LR_x8", val_hr_dir="amls2_coursework/datasets/DIV2K_valid_HR", 
                     val_lr_dir="amls2_coursework/datasets/DIV2K_valid_LR_x8", scale=8):
    # # -----------------------------
    # # Directories for training data
    # train_hr_dir = "amls2_coursework/datasets/DIV2K_train_HR"
    # train_lr_dir = "amls2_coursework/datasets/DIV2K_train_LR_x8"

    # # Directories for validation data (paired HR and LR)
    # val_hr_dir = "amls2_coursework/datasets/DIV2K_valid_HR"
    # val_lr_dir = "amls2_coursework/datasets/DIV2K_valid_LR_x8"

    # Sorted lists of file names for training
    train_hr_files = sorted(os.listdir(train_hr_dir))
    train_lr_files = sorted(os.listdir(train_lr_dir))
    train_hr_paths = [os.path.join(train_hr_dir, f) for f in train_hr_files]
    train_lr_paths = [os.path.join(train_lr_dir, f) for f in train_lr_files]

    print("Number of training HR images:", len(train_hr_paths))
    print("Number of training LR images:", len(train_lr_paths))

    # Sorted lists for validation
    val_hr_files = sorted(os.listdir(val_hr_dir))
    val_lr_files = sorted(os.listdir(val_lr_dir))
    val_hr_paths = [os.path.join(val_hr_dir, f) for f in val_hr_files]
    val_lr_paths = [os.path.join(val_lr_dir, f) for f in val_lr_files]

    print("Number of validation HR images:", len(val_hr_paths))
    print("Number of validation LR images:", len(val_lr_paths))

    # -----------------------------
    # Function to extract random patches from a LR/HR pair.
    def random_crops(lr_img, hr_img, n_crops, lr_crop=32, scale=scale):
        """
        Randomly crop n patches from the LR/HR image pair.
        """
        hr_crop = lr_crop * scale
        h_lr, w_lr = lr_img.shape[0], lr_img.shape[1]
        lr_patches = []
        hr_patches = []
        
        for _ in range(n_crops):
            # Random top-left corner for the LR crop
            x = np.random.randint(0, h_lr - lr_crop + 1)
            y = np.random.randint(0, w_lr - lr_crop + 1)
            
            lr_patch = lr_img[x: x + lr_crop, y: y + lr_crop, :]
            hr_patch = hr_img[x * scale: x * scale + hr_crop, y * scale: y * scale + hr_crop, :]
            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)
        
        return lr_patches, hr_patches

    # -----------------------------
    train_lr_cropped = []
    train_hr_cropped = []

    for lr_path, hr_path in tqdm(list(zip(train_lr_paths, train_hr_paths)), total=len(train_lr_paths), desc="Processing training pairs"):
        # Load and preprocess LR image
        lr_img = tf.io.read_file(lr_path)
        lr_img = tf.image.decode_png(lr_img, channels=3)
        lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)
        
        # Load and preprocess HR image
        hr_img = tf.io.read_file(hr_path)
        hr_img = tf.image.decode_png(hr_img, channels=3)
        hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)
        
        lr_patches, hr_patches = random_crops(lr_img, hr_img, n_crops=n_random_crops_train, lr_crop=32, scale=scale)
        train_lr_cropped.extend(lr_patches)
        train_hr_cropped.extend(hr_patches)

    print("Total training cropped LR patches:", len(train_lr_cropped))
    print("Total training cropped HR patches:", len(train_hr_cropped))

    # -----------------------------
    # Create random crops for validation data.
    val_lr_cropped = []
    val_hr_cropped = []

    for lr_path, hr_path in tqdm(list(zip(val_lr_paths, val_hr_paths)), total=len(val_lr_paths), desc="Processing validation pairs"):
        # Load and preprocess LR image (validation)
        lr_img = tf.io.read_file(lr_path)
        lr_img = tf.image.decode_png(lr_img, channels=3)
        lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)
        
        # Load and preprocess HR image (validation)
        hr_img = tf.io.read_file(hr_path)
        hr_img = tf.image.decode_png(hr_img, channels=3)
        hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)
        
        lr_patches, hr_patches = random_crops(lr_img, hr_img, n_crops=n_random_crops_val, lr_crop=32, scale=scale)
        val_lr_cropped.extend(lr_patches)
        val_hr_cropped.extend(hr_patches)

    print("Total validation cropped LR patches:", len(val_lr_cropped))
    print("Total validation cropped HR patches:", len(val_hr_cropped))

    return train_lr_cropped, train_hr_cropped, val_lr_cropped, val_hr_cropped
# -----------------------------


def create_batches(lr_cropped, hr_cropped, batch_size):

    lr_batches = []
    hr_batches = []
    for i in range(0, len(lr_cropped), batch_size):
        lr_batch = np.stack(lr_cropped[i: i + batch_size], axis=0)
        hr_batch = np.stack(hr_cropped[i: i + batch_size], axis=0)
        lr_batches.append(lr_batch)
        hr_batches.append(hr_batch)
    return lr_batches, hr_batches
