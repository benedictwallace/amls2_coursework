import tensorflow as tf
import os
from tqdm import tqdm
import shutil

# Paths to the LR / HR image directories
hr_dir = "amls2_coursework/datasets/DIV2K_train_HR"
lr_dir = "amls2_coursework/datasets/DIV2K_train_LR_x8"


hr_files = sorted(os.listdir(hr_dir))
lr_files = sorted(os.listdir(lr_dir))

hr_paths = [os.path.join(hr_dir, f) for f in hr_files]
lr_paths = [os.path.join(lr_dir, f) for f in lr_files]

print("Number of HR images:", len(hr_paths))
print("Number of LR images:", len(lr_paths))

# Define a random crop function
def random_crop_pair(lr_img, hr_img, lr_crop=32, scale=8):
    """
    Randomly crop an LR patch of size `lr_crop`×`lr_crop`,
    and the corresponding HR patch of size (lr_crop*scale)×(lr_crop*scale).
    """
    hr_crop = lr_crop * scale

    h_lr, w_lr = tf.shape(lr_img)[0], tf.shape(lr_img)[1]
    # Choose a random top-left for the LR patch
    x_lr = tf.random.uniform([], 0, h_lr - lr_crop, dtype=tf.int32)
    y_lr = tf.random.uniform([], 0, w_lr - lr_crop, dtype=tf.int32)

    lr_cropped = lr_img[x_lr : x_lr + lr_crop, y_lr : y_lr + lr_crop, :]

    # Corresponding region in HR
    x_hr = x_lr * scale
    y_hr = y_lr * scale
    hr_cropped = hr_img[x_hr : x_hr + hr_crop, y_hr : y_hr + hr_crop, :]

    return lr_cropped, hr_cropped

# Arrays to store the loaded/cropped images
lr_images = []
hr_images = []

print("Loading and cropping images...")
for lr_path, hr_path in tqdm(zip(lr_paths, hr_paths), total=len(lr_paths)):
    # Load LR
    lr_img = tf.io.read_file(lr_path)
    lr_img = tf.image.decode_png(lr_img, channels=3)
    lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)  # scale to [0,1]

    # Load HR
    hr_img = tf.io.read_file(hr_path)
    hr_img = tf.image.decode_png(hr_img, channels=3)
    hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)  # scale to [0,1]

    # Random crop
    lr_crop, hr_crop = random_crop_pair(lr_img, hr_img, lr_crop=32, scale=8)

    # Convert to NumPy
    lr_images.append(lr_crop.numpy())
    hr_images.append(hr_crop.numpy())

# Create new directories for the cropped patches.
lr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_LR_cropped"
hr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_HR_cropped"

os.makedirs(lr_cropped_dir, exist_ok=True)
os.makedirs(hr_cropped_dir, exist_ok=True)

for i, (lr_img, hr_img) in enumerate(zip(lr_images, hr_images)):

    # Convert float32 images in [0, 1] to uint8 in [0, 255]
    lr_img_uint8 = tf.image.convert_image_dtype(lr_img, tf.uint8)
    hr_img_uint8 = tf.image.convert_image_dtype(hr_img, tf.uint8)

    # Encode as PNG
    lr_png = tf.image.encode_png(lr_img_uint8) 
    hr_png = tf.image.encode_png(hr_img_uint8)

    # Construct paths for saving
    lr_save_path = os.path.join(lr_cropped_dir, f"lr_cropped_{i}.png")
    hr_save_path = os.path.join(hr_cropped_dir, f"hr_cropped_{i}.png")

    # Write to disk
    tf.io.write_file(lr_save_path, lr_png)
    tf.io.write_file(hr_save_path, hr_png)