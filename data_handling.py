import tensorflow as tf
import os
from tqdm import tqdm

# Paths to the LR / HR image directories
hr_dir = "amls2_coursework/datasets/DIV2K_train_HR"
lr_dir = "amls2_coursework/datasets/DIV2K_train_LR_x8"

hr_files = sorted(os.listdir(hr_dir))
lr_files = sorted(os.listdir(lr_dir))

hr_paths = [os.path.join(hr_dir, f) for f in hr_files]
lr_paths = [os.path.join(lr_dir, f) for f in lr_files]

print("Number of HR images:", len(hr_paths))
print("Number of LR images:", len(lr_paths))

def periodic_crops(lr_img, hr_img, lr_crop=32, scale=8, stride=None):
    """
    Periodically (systematically) crop LR/HR images into patches.
    
    By default (stride=None), the stride equals 'lr_crop' which 
    creates non-overlapping patches that tile the entire image.
    """
    if stride is None:
        stride = lr_crop  # Non-overlapping patches

    # Calculate HR patch size
    hr_crop = lr_crop * scale
    
    # Dimensions of LR image
    h_lr, w_lr = tf.shape(lr_img)[0], tf.shape(lr_img)[1]

    # Collect all patches here
    lr_patches = []
    hr_patches = []

    # Slide over the LR image
    for x in range(0, h_lr - lr_crop + 1, stride):
        for y in range(0, w_lr - lr_crop + 1, stride):
            # LR patch
            lr_patch = lr_img[x : x + lr_crop, y : y + lr_crop, :]
            # Corresponding region in HR
            hr_patch = hr_img[x*scale : x*scale + hr_crop, y*scale : y*scale + hr_crop, :]

            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)

    return lr_patches, hr_patches

# Create new directories for the cropped patches
lr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_LR_cropped"
hr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_HR_cropped"
os.makedirs(lr_cropped_dir, exist_ok=True)
os.makedirs(hr_cropped_dir, exist_ok=True)

print("Loading images and creating periodic crops...")

for i, (lr_path, hr_path) in tqdm(enumerate(zip(lr_paths, hr_paths)), total=len(lr_paths)):
    # Load LR
    lr_img = tf.io.read_file(lr_path)
    lr_img = tf.image.decode_png(lr_img, channels=3)
    lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)  # scale to [0,1]

    # Load HR
    hr_img = tf.io.read_file(hr_path)
    hr_img = tf.image.decode_png(hr_img, channels=3)
    hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)  # scale to [0,1]

    # Get periodic (systematic) patches
    lr_patches, hr_patches = periodic_crops(lr_img, hr_img, lr_crop=32, scale=8, stride=None)

    # Save each patch pair
    for j, (lr_patch, hr_patch) in enumerate(zip(lr_patches, hr_patches)):
        # Convert from float32 [0,1] to uint8 [0,255]
        lr_patch_uint8 = tf.image.convert_image_dtype(lr_patch, tf.uint8)
        hr_patch_uint8 = tf.image.convert_image_dtype(hr_patch, tf.uint8)

        # Encode as PNG
        lr_png = tf.image.encode_png(lr_patch_uint8)
        hr_png = tf.image.encode_png(hr_patch_uint8)

        # Construct paths for saving
        
        lr_save_path = os.path.join(lr_cropped_dir, f"lr_cropped_{i}_{j}.png")
        hr_save_path = os.path.join(hr_cropped_dir, f"hr_cropped_{i}_{j}.png")

        # Write to disk
        tf.io.write_file(lr_save_path, lr_png)
        tf.io.write_file(hr_save_path, hr_png)