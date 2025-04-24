import tensorflow as tf
from keras import layers, Model, optimizers, Sequential, applications
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv


def build_discriminator(in_shape):
    inputs = layers.Input(in_shape)
    
    # Example from the SRGAN paper (simplified)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Flatten + dense
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Final
    out = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, out)
    opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def build_generator(input_shape, res_blocks, upscale_2_power=3, final_activation="sigmoid", clip_output=False):
    
    gen_in = layers.Input(shape=input_shape)

    gen = layers.Conv2D(64, (9,9), padding="same")(gen_in)
    gen = layers.PReLU(shared_axes=[1,2])(gen)

    temp = gen

    for i in range(0, res_blocks):
        gen = residual_block_se(gen, filters=64, se_ratio=16)

    gen = layers.Conv2D(64, (3,3), padding="same")(gen)
    gen = layers.BatchNormalization(momentum=0.5)(gen)
    gen = layers.add([gen, temp])

    for i in range(0, upscale_2_power):
        gen = subpixel_conv(gen, scale=2, filters=64)

    if final_activation != "None":
        out = layers.Conv2D(3, (9,9), padding="same", activation=final_activation)(gen)
    else:
        # No activation, raw outputs (could be in (-inf, +inf))
        out = layers.Conv2D(3, (9,9), padding="same")(gen)

    # Optionally clip to [0,1]
    if clip_output:
        # We can use a Lambda layer
        out = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(out)

    model = Model(inputs=gen_in, outputs=out)

    return model

def squeeze_excitation_block(x, reduction_ratio=16):
    """
    Squeeze-and-Excitation block to recalibrate channel features.
    """
    channels = x.shape[-1]
    
    # Squeeze: Global average pool spatially
    se = layers.GlobalAveragePooling2D()(x)  # shape: (batch, channels)
    
    # Two fully-connected layers
    se = layers.Dense(channels // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    
    # Reshape to (batch, 1, 1, channels)
    se = layers.Reshape((1, 1, channels))(se)
    
    # Excite: multiply original feature maps by the learned channel weights
    x = layers.Multiply()([x, se])
    
    return x

def residual_block_se(x_in, filters=64, se_ratio=16):
    """
    Residual block with BatchNorm + PReLU + Squeeze-Excitation
    """
    # First conv
    x = layers.Conv2D(filters, (3,3), padding="same")(x_in)
    #x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    
    # Second conv
    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    #x = layers.BatchNormalization(momentum=0.5)(x)
    
    # Squeeze-and-Excitation
    x = squeeze_excitation_block(x, reduction_ratio=se_ratio)
    
    # Residual add
    x_out = layers.Add()([x_in, x])
    
    return x_out

def subpixel_conv(x, scale=2, filters=64):
    """
    Takes a tensor `x` of shape (batch, H, W, filters)
    and performs:
      1) A Conv2D with `filters * (scale^2)` channels
      2) tf.nn.depth_to_space(...) to rearrange channels into
         an upscaled output of shape (batch, scale*H, scale*W, filters)
    """
    # 1) Convolution to increase channels
    x = layers.Conv2D(filters * (scale**2), kernel_size=3, padding='same')(x)
    # 2) Pixel shuffle
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, block_size=scale))(x)
    # 3) Activation (commonly PReLU in SRGAN)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    return x

# def residual_block(initial):
#     res = layers.Conv2D(64, (3,3), padding="same")(initial)
#     res = layers.BatchNormalization(momentum=0.5)(res)
#     res = layers.PReLU(shared_axes = [1,2])(res)
#     res = layers.Conv2D(64, (3,3), padding="same")(res)
#     res = layers.BatchNormalization(momentum=0.5)(res)

#     return layers.add([initial, res])

# def upscale(initial):
#     up_model = layers.Conv2D(256, (3,3), padding="same")(initial)
#     up_model = layers.UpSampling2D(size=2)(up_model)
#     up_model = layers.PReLU(shared_axes=[1,2])(up_model)

#     return up_model

def preprocess_for_vgg(img):
    """
    Convert [0,1] images -> [0,255], then apply vgg19.preprocess_input.
    """
    img = img * 255.0
    return tf.keras.applications.vgg19.preprocess_input(img)

def vgg_content_loss(vgg_extractor, hr, sr):
    """
    Compute MSE of VGG feature maps.
    """
    hr_proc = preprocess_for_vgg(hr)
    sr_proc = preprocess_for_vgg(sr)
    hr_features = vgg_extractor(hr_proc)
    sr_features = vgg_extractor(sr_proc)
    return tf.reduce_mean(tf.square(hr_features - sr_features))

def build_vgg_for_content_loss():
    vgg = applications.VGG19(weights='imagenet', include_top=False)
    chosen_layer = 'block5_conv4'
    vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer(chosen_layer).output)
    vgg_model.trainable = False
    return vgg_model

class SRGANTrainer:
    """
    Handles training steps for both generator and discriminator
    with combined MSE + Adversarial + VGG loss.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 vgg_extractor,
                 lr=1e-4,
                 beta_1=0.5,
                 lambda_adv=0.01,
                 lambda_vgg=1,
                 lambda_mse=0.1):
        """
        lambda_adv, lambda_vgg, lambda_mse are weights for each loss component.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_extractor = vgg_extractor

        self.opt_g = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.opt_d = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.bce = tf.keras.losses.BinaryCrossentropy()

        # Weights for each loss term
        self.lambda_adv = lambda_adv
        self.lambda_vgg = lambda_vgg
        self.lambda_mse = lambda_mse

    @tf.function
    def train_step(self, lr_imgs, hr_imgs):
        """
        Returns: (d_loss, g_loss) as floats (or tensors).
        """
        batch_size = tf.shape(lr_imgs)[0]

        # ----- Train Discriminator -----
        with tf.GradientTape() as tape_d:
            # Generate super-res images
            fake_imgs = self.generator(lr_imgs, training=True)

            # Real = 1, Fake = 0
            real_labels = tf.ones((batch_size,1), dtype=tf.float32)
            fake_labels = tf.zeros((batch_size,1), dtype=tf.float32)

            # Evaluate real and fake
            d_real = self.discriminator(hr_imgs, training=True)
            d_fake = self.discriminator(fake_imgs, training=True)

            d_loss_real = self.bce(real_labels, d_real)
            d_loss_fake = self.bce(fake_labels, d_fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        grads_d = tape_d.gradient(d_loss, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))

        # ----- Train Generator -----
        with tf.GradientTape() as tape_g:
            fake_imgs = self.generator(lr_imgs, training=True)

            # (1) Adversarial loss (wants D(fake) = 1)
            d_fake_for_g = self.discriminator(fake_imgs, training=False)
            adv_loss = self.bce(tf.ones((batch_size,1)), d_fake_for_g)

            # (2) VGG content loss
            content_loss = vgg_content_loss(self.vgg_extractor, hr_imgs, fake_imgs)

            # (3) MSE loss (pixel-wise)
            mse_loss = tf.reduce_mean(tf.square(hr_imgs - fake_imgs))

            # Weighted sum
            g_loss = (self.lambda_adv * adv_loss
                      + self.lambda_vgg * content_loss
                      + self.lambda_mse * mse_loss)

        grads_g = tape_g.gradient(g_loss, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        return d_loss, g_loss, adv_loss, content_loss, mse_loss

    @tf.function
    def val_step(self, lr_imgs, hr_imgs):
        """
        Returns: (d_loss, g_loss) for validation (forward-only).
        """
        batch_size = tf.shape(lr_imgs)[0]
        fake_imgs = self.generator(lr_imgs, training=False)

        # Discriminator forward
        d_real = self.discriminator(hr_imgs, training=False)
        d_fake = self.discriminator(fake_imgs, training=False)

        real_labels = tf.ones((batch_size,1), dtype=tf.float32)
        fake_labels = tf.zeros((batch_size,1), dtype=tf.float32)

        d_loss_real = self.bce(real_labels, d_real)
        d_loss_fake = self.bce(fake_labels, d_fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Adversarial
        adv_loss = self.bce(tf.ones((batch_size,1)), d_fake)
        # VGG
        content_loss = vgg_content_loss(self.vgg_extractor, hr_imgs, fake_imgs)
        # MSE
        mse_loss = tf.reduce_mean(tf.square(hr_imgs - fake_imgs))

        g_loss = (self.lambda_adv * adv_loss
                  + self.lambda_vgg * content_loss
                  + self.lambda_mse * mse_loss)
        return d_loss, g_loss, adv_loss, content_loss, mse_loss

