import tensorflow as tf
from keras import layers, Model, Sequential, optimizers, applications
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras import backend as K
import csv
import random
from model import SRGANTrainer, build_discriminator, build_generator, build_vgg_for_content_loss
from fetch_and_crop import crop_patch_pairs, create_batches

# -----------------------------
# Create random crops for training data.
n_random_crops_train = 15  # number of crops per training image pair

n_random_crops_val = 2  # number of crops per validation image pair

train_lr_cropped, train_hr_cropped, val_lr_cropped, val_hr_cropped = crop_patch_pairs(n_random_crops_train, n_random_crops_val, val_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_valid_LR_mild", train_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_train_LR_mild", scale=4)

# -----------------------------
# Create batches from the cropped patches.
batch_size = 16

#train_lr_batches, train_hr_batches = create_batches(train_lr_cropped, train_hr_cropped, batch_size)
val_lr_batches, val_hr_batches = create_batches(val_lr_cropped, val_hr_cropped, batch_size)

lr_shape = (32, 32, 3)

hr_shape = (128, 128, 3)

generator = build_generator(res_blocks=16, input_shape=lr_shape, final_activation="sigmoid", upscale_2_power=2)

generator.load_weights("generator_pretrained_B.weights.h5")

discriminator = build_discriminator(hr_shape)

vgg_extractor = build_vgg_for_content_loss()

sr_trainer = SRGANTrainer(
    generator,
    discriminator,
    vgg_extractor,
    lr=1e-4,
    beta_1=0.5,
    lambda_adv=1e-3,  # Weight of adversarial
    lambda_vgg=0.5,  # Weight of VGG content
    lambda_mse=1   # Weight of MSE
)


# Now run an epoch loop
epochs = 30

train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    # Shuffle
    combined = list(zip(train_lr_cropped, train_hr_cropped))
    random.shuffle(combined)
    # Unzip the list back into LR / HR arrays
    train_lr_cropped_shuffled, train_hr_cropped_shuffled = zip(*combined)

    # Rebatch
    train_lr_batches, train_hr_batches = create_batches(
        train_lr_cropped_shuffled, 
        train_hr_cropped_shuffled, 
        batch_size
    )

    g_losses = []
    d_losses = []
    adv_losses = []
    content_losses = [] 
    mse_losses = []

    if epoch == 10:
        sr_trainer.opt_g.learning_rate.assign(5e-5)
        sr_trainer.lambda_adv = 5e-3
        print("Changed learning rate to 0.00005 and adversarial to 0.005")

    if epoch == 20:
        sr_trainer.lambda_adv = 0.01
        print("Changed adversarial loss to 0.01")

    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        
        d_loss, g_loss, adv_loss, content_loss, mse_loss = sr_trainer.train_step(lr_imgs, hr_imgs)
        g_losses.append(g_loss.numpy())
        d_losses.append(d_loss.numpy())
        adv_losses.append(adv_loss.numpy())
        content_losses.append(content_loss.numpy())
        mse_losses.append(mse_loss.numpy())

    train_d_mean = np.mean(d_losses)
    train_g_mean = np.mean(g_losses)
    train_adv_mean = np.mean(adv_losses)
    train_vgg_mean = np.mean(content_losses)
    train_mse_mean = np.mean(mse_losses)


    train_loss_history.append((train_d_mean, train_g_mean, train_adv_mean, train_vgg_mean, train_mse_mean))

    val_d_losses = []
    val_g_losses = []
    val_adv_losses = []
    val_vgg_losses = []
    val_mse_losses = []

    for b in range(len(val_hr_batches)):
        lr_imgs_val = val_lr_batches[b]
        hr_imgs_val = val_hr_batches[b]
        
        d_loss_val, g_loss_val, adv_loss_val, content_loss_val, mse_loss_val = sr_trainer.val_step(lr_imgs_val, hr_imgs_val)

        val_d_losses.append(d_loss_val.numpy())
        val_g_losses.append(g_loss_val.numpy())
        val_adv_losses.append(adv_loss_val.numpy())
        val_vgg_losses.append(content_loss_val.numpy())
        val_mse_losses.append(mse_loss_val.numpy())

    # Compute mean validation losses for the epoch
    val_d_mean = np.mean(val_d_losses)
    val_g_mean = np.mean(val_g_losses)
    val_adv_mean = np.mean(val_adv_losses)
    val_vgg_mean = np.mean(val_vgg_losses)
    val_mse_mean = np.mean(val_mse_losses)

    val_loss_history.append((val_d_mean, val_g_mean, val_adv_mean, val_vgg_mean, val_mse_mean))

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train => D: {train_d_mean:.4f}, Adv: {train_adv_mean:.4f}, "
        f"VGG: {train_vgg_mean:.4f}, MSE: {train_mse_mean:.4f}, G: {train_g_mean:.4f} || "
        f"Val => D: {val_d_mean:.4f}, Adv: {val_adv_mean:.4f}, "
        f"VGG: {val_vgg_mean:.4f}, MSE: {val_mse_mean:.4f}, G: {val_g_mean:.4f}"
    )


generator.save_weights("generator_vgg_B.weights.h5")

# Extract the separate losses from your history
train_adv = [t[2] for t in train_loss_history]  # Adversarial loss is the 3rd element
train_vgg = [t[3] for t in train_loss_history]  # VGG/content loss is the 4th element
train_mse = [t[4] for t in train_loss_history]  # MSE is the 5th element

val_adv = [v[2] for v in val_loss_history]
val_vgg = [v[3] for v in val_loss_history]
val_mse = [v[4] for v in val_loss_history]

# Create a figure with 3 subplots (one per loss type)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

# 1) Adversarial loss
axes[0].plot(train_adv, label='Train')
axes[0].plot(val_adv,   label='Val')
axes[0].set_title('Adversarial Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# 2) VGG/content loss
axes[1].plot(train_vgg, label='Train')
axes[1].plot(val_vgg,   label='Val')
axes[1].set_title('VGG Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

# 3) MSE loss
axes[2].plot(train_mse, label='Train')
axes[2].plot(val_mse,   label='Val')
axes[2].set_title('MSE Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].legend()

plt.tight_layout()
plt.savefig("amls2_coursework/training_losses_B.png", dpi=300)