import tensorflow as tf
from keras import layers, Model, Sequential, optimizers, applications
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras import backend as K

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

    op = layers.Conv2D(3, (9,9), padding="same", activation="sigmoid")(gen)

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

def build_vgg_for_content_loss():
    """
    Loads VGG19, and outputs feature maps from a chosen convolutional layer.
    Typically block5_conv4 is used for perceptual loss in SRGAN.
    """
    vgg = applications.VGG19(weights='imagenet', include_top=False)
    
    # Pick a layer to extract features from. 
    # For example, 'block5_conv4' is near the end of VGG19
    # (feel free to experiment with other layers).
    chosen_layer = 'block5_conv4'
    
    # Create a model that outputs the feature maps of that layer.
    vgg_model = Model(inputs=vgg.input, 
                      outputs=vgg.get_layer(chosen_layer).output)
    
    # We don’t want to train VGG19’s weights!
    vgg_model.trainable = False
    
    return vgg_model



def vgg_content_loss(vgg_extractor, hr, sr):
    hr_proc = preprocess_for_vgg(hr)
    sr_proc = preprocess_for_vgg(sr)
    hr_features = vgg_extractor(hr_proc)
    sr_features = vgg_extractor(sr_proc)
    return tf.reduce_mean(tf.square(hr_features - sr_features))



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

def preprocess_for_vgg(img):
    # img in [0,1]
    img = img * 255.0  # back to [0,255]
    img = tf.keras.applications.vgg19.preprocess_input(img)  
    return img

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

generator = build_generator(res_blocks=8, input_shape=lr_shape)

generator.load_weights("generator_pretrained.keras")

discriminator = build_discriminator(hr_shape)

#gan_model = build_gan_simple(generator, discriminator, lr_shape)

vgg_extractor = build_vgg_for_content_loss()

# 3. Create optimizers for G and D 
opt_g = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
opt_d = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)


# 4. A helpful function for adversarial label-based BCE
bce = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(lr_imgs, hr_imgs, lambda_vgg=1.0):
    """
    One training step for both generator and discriminator,
    returning d_loss and g_loss.
    """
    batch_size = tf.shape(lr_imgs)[0]
    
    # ---------------
    # Train Discriminator
    # ---------------
    # (a) Generate fake images
    with tf.GradientTape() as tape_d:
        fake_imgs = generator(lr_imgs, training=True)
        
        # Labeling:
        #   real -> ones
        #   fake -> zeros
        real_labels = tf.ones((batch_size,1), dtype=tf.float32)
        fake_labels = tf.zeros((batch_size,1), dtype=tf.float32)
        
        # (b) Discriminator outputs
        d_real = discriminator(hr_imgs, training=True)
        d_fake = discriminator(fake_imgs, training=True)
        
        # (c) Loss for D = average of real/fake BCE
        d_loss_real = bce(real_labels, d_real)
        d_loss_fake = bce(fake_labels, d_fake)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        
    # (d) Update D
    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    opt_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    
    # ---------------
    # Train Generator
    # ---------------
    with tf.GradientTape() as tape_g:
        fake_imgs = generator(lr_imgs, training=True)
        
        # (a) adversarial loss (wants D(fake)=1)
        d_fake_for_g = discriminator(fake_imgs, training=False)
        adv_loss = bce(tf.ones((batch_size,1)), d_fake_for_g)
        
        # (b) VGG content loss
        content_loss = vgg_content_loss(vgg_extractor, hr_imgs, fake_imgs)
        
        # (c) total generator loss
        g_loss = adv_loss + lambda_vgg * content_loss
    
    # (d) Update G
    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss

# Now run an epoch loop
epochs = 5

for epoch in range(epochs):
    g_losses = []
    d_losses = []
    
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        
        d_loss, g_loss = train_step(lr_imgs, hr_imgs, lambda_vgg=1.0)
        g_losses.append(g_loss.numpy())
        d_losses.append(d_loss.numpy())
    
    print(f"Epoch {epoch+1}/{epochs} "
          f"[D loss: {np.mean(d_losses):.4f}] "
          f"[G loss: {np.mean(g_losses):.4f}]")

        # ----- Save a sample image after each epoch -----
    # Here, we just use the first image from the original loaded set
    # You can choose any index or random sample
    sample_lr_img = lr_cropped_images[0]  # shape [H, W, 3]
    sample_hr_img = hr_cropped_images[0]  # shape [H, W, 3]
    
    # Create batch dimension
    input_lr = np.expand_dims(sample_lr_img, axis=0)
    # Generate SR image
    upscaled_img = generator.predict(input_lr)
    upscaled_img = np.squeeze(upscaled_img, axis=0)  # remove batch dimension

    # Resize LR for visual comparison
    resized_lr = tf.image.resize(sample_lr_img, (256, 256)).numpy()

    # Concatenate side-by-side: [resized LR | SR result | original HR]
    combined_img = np.concatenate([resized_lr, upscaled_img, sample_hr_img], axis=1)

    # Save the combined image
    out_path = f"amls2_coursework/datasets/training_samples/sample_epoch_{epoch+1}.png"
    plt.imsave(out_path, combined_img)


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