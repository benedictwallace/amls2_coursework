
import tensorflow as tf
from keras import layers, Model, Sequential, optimizers, applications
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from keras import backend as K
import random
from model import SRGANTrainer, build_discriminator, build_generator, build_vgg_for_content_loss
from fetch_and_crop import crop_patch_pairs, create_batches

# -----------------------------
# Create random crops for training data.
n_random_crops_train = 15  # number of crops per training image pair

n_random_crops_val = 1  # number of crops per validation image pair

train_lr_cropped, train_hr_cropped, val_lr_cropped, val_hr_cropped = crop_patch_pairs(n_random_crops_train, n_random_crops_val)

# -----------------------------
# Create batches from the cropped patches.
batch_size = 32

#train_lr_batches, train_hr_batches = create_batches(train_lr_cropped, train_hr_cropped, batch_size)
val_lr_batches, val_hr_batches = create_batches(val_lr_cropped, val_hr_cropped, batch_size)

lr_shape = (32, 32, 3)

hr_shape = (128, 256, 3)

generator = build_generator(res_blocks=16, input_shape=lr_shape, final_activation="sigmoid")

generator.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
)

epochs = 200

train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    epoch_losses = []
    val_losses = []
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

    for b in tqdm(range(len(train_lr_batches))):
        lr_imgs = train_lr_batches[b]  # shape [batch_size, 32, 32, 3]
        hr_imgs = train_hr_batches[b]  # shape [batch_size, 256, 256, 3]
        
        # Train generator on this batch
        loss = generator.train_on_batch(lr_imgs, hr_imgs)
        epoch_losses.append(loss)

        
    for b in range(len(val_hr_batches)):
        lr_imgs_val = val_lr_batches[b]
        hr_imgs_val = val_hr_batches[b]
        
        loss = generator.test_on_batch(lr_imgs, hr_imgs)
        val_losses.append(loss)
    
    train_loss_epoch = np.mean(epoch_losses)
    val_loss_epoch   = np.mean(val_losses)

    train_loss_history.append(train_loss_epoch)
    val_loss_history.append(val_loss_epoch)

    print(f"Pretrain Epoch {epoch+1}/{epochs}, G_MSE_Loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}")

generator.save_weights("generator_pretrained.weights.h5")


plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_loss_history, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss_history,   label='Validation Loss')
plt.title('Pre-training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig('pre_training_losses.png')
plt.close()