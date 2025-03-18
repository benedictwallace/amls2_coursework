
import tensorflow as tf
from keras import layers, Model, Sequential, optimizers, applications
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras import backend as K

def upscale(initial):
    up_model = layers.Conv2D(256, (3,3), padding="same")(initial)
    up_model = layers.UpSampling2D(size=2)(up_model)
    up_model = layers.PReLU(shared_axes=[1,2])(up_model)

    return up_model

def residual_block(initial):
    res = layers.Conv2D(64, (3,3), padding="same")(initial)
    res = layers.BatchNormalization(momentum=0.5)(res)
    res = layers.PReLU(shared_axes = [1,2])(res)
    res = layers.Conv2D(64, (3,3), padding="same")(res)
    res = layers.BatchNormalization(momentum=0.5)(res)

    return layers.add([initial, res])

def build_generator(input_shape, res_blocks):
    
    gen_in = layers.Input(shape=input_shape)

    gen = layers.Conv2D(64, (9,9), padding="same")(gen_in)
    gen = layers.PReLU(shared_axes=[1,2])(gen)

    temp = gen

    for i in range(0, res_blocks):
        gen = residual_block(gen)

    gen = layers.Conv2D(64, (3,3), padding="same")(gen)
    gen = layers.BatchNormalization(momentum=0.5)(gen)
    gen = layers.add([gen, temp])

    gen = upscale(gen)
    gen = upscale(gen)
    gen = upscale(gen)

    op = layers.Conv2D(3, (9,9), padding="same", activation="sigmoid")(gen)

    model = Model(inputs=gen_in, outputs=op)

    return model


generator = build_generator(res_blocks=8, input_shape=(32, 32, 3))

generator.compile(
    loss=tf.keras.losses.MeanSquaredError(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
)


# Pull cropped images and batch-------------------------------------


# Paths to your cropped directories
lr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_LR_cropped"
hr_cropped_dir = "amls2_coursework/datasets/DIV2K_train_HR_cropped"

# Get list of file names
lr_cropped_files = sorted(os.listdir(lr_cropped_dir))
hr_cropped_files = sorted(os.listdir(hr_cropped_dir))

# Lists to store the loaded images
lr_cropped_images = []
hr_cropped_images = []

# Read each PNG, decode, and convert to float32 in [0,1]
for lr_file, hr_file in zip(lr_cropped_files, hr_cropped_files):
    lr_path = os.path.join(lr_cropped_dir, lr_file)
    hr_path = os.path.join(hr_cropped_dir, hr_file)

    # Read LR image
    lr_img = tf.io.read_file(lr_path)
    lr_img = tf.image.decode_png(lr_img, channels=3)
    lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)  # now in [0,1]

    # Read HR image
    hr_img = tf.io.read_file(hr_path)
    hr_img = tf.image.decode_png(hr_img, channels=3)
    hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)  # now in [0,1]

    lr_cropped_images.append(lr_img)
    hr_cropped_images.append(hr_img)


# Now create batches
batch_size = 8

train_lr_batches = []
train_hr_batches = []

for i in range(0, len(lr_cropped_images), batch_size):
    # Slice out a batch of LR / HR patches
    lr_batch = lr_cropped_images[i : i + batch_size]
    hr_batch = hr_cropped_images[i : i + batch_size]

    # Stack them into (batch_size, H, W, 3)
    lr_batch = np.stack(lr_batch, axis=0)
    hr_batch = np.stack(hr_batch, axis=0)
    
    train_lr_batches.append(lr_batch)
    train_hr_batches.append(hr_batch)

#--------------------------------------------------------------



epochs = 30

for epoch in range(epochs):
    epoch_losses = []
    
    for b in tqdm(range(len(train_lr_batches))):
        lr_imgs = train_lr_batches[b]  # shape [batch_size, 32, 32, 3]
        hr_imgs = train_hr_batches[b]  # shape [batch_size, 256, 256, 3]
        
        # Train generator on this batch
        loss = generator.train_on_batch(lr_imgs, hr_imgs)
        epoch_losses.append(loss)
    
    print(f"Pretrain Epoch {epoch+1}/{epochs}, G_MSE_Loss: {np.mean(epoch_losses):.4f}")

generator.save_weights("generator_pretrained.keras")

sample_lr_img = lr_cropped_images[0] 
sample_hr_img = hr_cropped_images[0]

input_lr = np.expand_dims(sample_lr_img, axis=0)
upscaled_img = generator.predict(input_lr)
upscaled_img = np.squeeze(upscaled_img, axis=0)

resized_lr = tf.image.resize(sample_lr_img, (256, 256)).numpy()
combined_img = np.concatenate([resized_lr, upscaled_img, sample_hr_img], axis=1)

plt.figure()
plt.imshow(combined_img)
plt.title("Left: LR resized | Middle: SRGAN Output | Right: Original HR")
plt.show()