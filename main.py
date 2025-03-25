import tensorflow as tf
from keras import layers, Model, Sequential, optimizers
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def build_discriminator(in_shape):
    model = Sequential()
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding="same", input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


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

    op = layers.Conv2D(3, (9,9), padding="same")(gen)

    model = Model(inputs=gen_in, outputs=op)

    return model
    


def residual_block(initial):
    res = layers.Conv2D(64, (3,3), padding="same")(initial)
    res = layers.BatchNormalization(momentum=0.5)(res)
    res = layers.PReLU(shared_axes = [1,2])(res)
    res = layers.Conv2D(64, (3,3), padding="same")(res)
    res = layers.BatchNormalization(momentum=0.5)(res)

    return layers.add([initial, res])

def upscale(initial):
    up_model = layers.Conv2D(256, (3,3), padding="same")(initial)
    up_model = layers.UpSampling2D(size=2)(up_model)
    up_model = layers.PReLU(shared_axes=[1,2])(up_model)

    return up_model

def build_gan_simple(generator, discriminator, lr_shape):
    """
    Build a combined model that stacks generator and discriminator
    for adversarial training. 
    """
    discriminator.trainable = False
    
    # Low-resolution input
    input_lr = layers.Input(shape=lr_shape)
    
    # Generate super-res output
    gen_hr = generator(input_lr)
    
    # Discriminator result on fake images
    validity = discriminator(gen_hr)

    # Combined model: Low-resolution -> (Generator -> Discriminator)
    gan_model = Model(inputs=input_lr, outputs=validity)

    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    gan_model.compile(loss="binary_crossentropy", optimizer=opt)
    return gan_model

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

print("Number of batches:", len(train_lr_batches))
print("Shape of first LR batch:", train_lr_batches[0].shape)
print("Shape of first HR batch:", train_hr_batches[0].shape)

lr_shape = (32, 32, 3)

hr_shape = (256, 256, 3)

generator = build_generator(res_blocks=4, input_shape=lr_shape)
discriminator = build_discriminator(hr_shape)
gan_model = build_gan_simple(generator, discriminator, lr_shape)

epochs = 20


for epoch in range(epochs):

    g_losses = []
    d_losses = []

    for b in tqdm(range(len(train_hr_batches))):

        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = generator.predict_on_batch(lr_imgs)

        fake_label = np.zeros((batch_size, 1))
        real_label = np.ones((batch_size, 1))

        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)

        discriminator.trainable = False

        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real) # avg loss for show
        d_losses.append(d_loss[0])

        g_loss = gan_model.train_on_batch(lr_imgs, real_label)
        g_losses.append(g_loss)
    
    print(f"Epoch {epoch+1}/{epochs}  [D loss: {np.mean(d_losses):.4f}]  [G loss: {np.mean(g_losses):.4f}]")



sample_lr_img = lr_cropped_images[0]  # shape: (16, 16, 3)
sample_hr_img = hr_cropped_images[0]

# Prepare it for the generator
input_lr = np.expand_dims(sample_lr_img, axis=0)  # (1, 16, 16, 3)

# Generate upscaled image
upscaled_img = generator.predict(input_lr)        # (1, 128, 128, 3)
upscaled_img = np.squeeze(upscaled_img, axis=0)   # now (128, 128, 3)

# Resize the original 16×16 LR image to match the upscaled size for visual comparison
resized_lr = tf.image.resize(sample_lr_img, (256, 256)).numpy()

# Concatenate side-by-side: left half = LR (resized), right half = upscaled
combined_img = np.concatenate([resized_lr, upscaled_img, sample_hr_img], axis=1)

# Display them in a single figure
plt.figure()
plt.imshow(combined_img)
plt.title("Left: LR resized to 128x128   |   Right: Upscaled (Predicted)")
plt.show()

generator.save("generator_model.keras")
discriminator.save("discriminator_model.keras")
gan_model.save("gan_model.keras")