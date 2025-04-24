#!/usr/bin/env python
"""
Combined training script for SRGAN coursework.

Usage examples
--------------
Pre-train Track 1 (×8 up‑scale):
    python train_srgan_combined.py --track 1 --phase pretrain

Adversarial/VGG fine‑tune Track 1:
    python train_srgan_combined.py --track 1 --phase vgg

Pre-train Track 2 (×4 up‑scale):
    python train_srgan_combined.py --track 2 --phase pretrain

Adversarial/VGG fine‑tune Track 2:
    python train_srgan_combined.py --track 2 --phase vgg

The script intentionally *does not* redefine any function that is imported from
`model` or `fetch_and_crop` – it only wraps them in a training pipeline that
mirrors the four original files.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, Model, optimizers

# ────────────────────────────────────────────────────────────────────────────────
# Internal helper – **not** imported from external modules
# (Used only for Track 1 pre‑training which originally defined a custom model.)
# ────────────────────────────────────────────────────────────────────────────────

def _upscale(x: tf.Tensor) -> tf.Tensor:
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.UpSampling2D(size=2)(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def _residual_block(x: tf.Tensor) -> tf.Tensor:
    res = layers.Conv2D(64, (3, 3), padding="same")(x)
    res = layers.BatchNormalization(momentum=0.5)(res)
    res = layers.PReLU(shared_axes=[1, 2])(res)
    res = layers.Conv2D(64, (3, 3), padding="same")(res)
    res = layers.BatchNormalization(momentum=0.5)(res)
    return layers.add([x, res])


def _build_generator_track1(input_shape: Tuple[int, int, int], res_blocks: int = 8) -> Model:
    """Generator exactly as in *file 1* (×8 up‑scale)."""
    gen_in = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (9, 9), padding="same")(gen_in)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    skip = x

    for _ in range(res_blocks):
        x = _residual_block(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(momentum=0.5)(x)
    x = layers.add([x, skip])

    # 2×‑up‑scale three times → 8× total (32 → 256)
    for _ in range(3):
        x = _upscale(x)

    out = layers.Conv2D(3, (9, 9), padding="same", activation="sigmoid")(x)
    return Model(gen_in, out)


# ────────────────────────────────────────────────────────────────────────────────
# Import *after* defining helpers to avoid circular deps in some environments.
# Note: the script never rewrites the imported functions/classes.
# ────────────────────────────────────────────────────────────────────────────────
from model import (
    build_generator,          # Used for Track 1 VGG fine‑tune & Track 2 flows
    build_discriminator,
    build_vgg_for_content_loss,
    SRGANTrainer,
)
from fetch_and_crop import crop_patch_pairs, create_batches


# ════════════════════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════════════════════


def _load_cropped_png_pairs(lr_dir: Path, hr_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Read *already‑cropped* LR/HR PNGs from disk into memory (Track 1)."""
    lr_files = sorted(os.listdir(lr_dir))
    hr_files = sorted(os.listdir(hr_dir))

    lr_imgs, hr_imgs = [], []
    for lr_file, hr_file in zip(lr_files, hr_files):
        lr_img = tf.io.read_file(lr_dir / lr_file)
        lr_img = tf.image.decode_png(lr_img, channels=3)
        lr_img = tf.image.convert_image_dtype(lr_img, tf.float32)

        hr_img = tf.io.read_file(hr_dir / hr_file)
        hr_img = tf.image.decode_png(hr_img, channels=3)
        hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)

        lr_imgs.append(lr_img.numpy())  # store as NumPy arrays to save RAM later
        hr_imgs.append(hr_img.numpy())
    return lr_imgs, hr_imgs


def _batched(arrays: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    return [np.stack(arrays[i : i + batch_size], axis=0) for i in range(0, len(arrays), batch_size)]


# ════════════════════════════════════════════════════════════════════════════════
# Track 1 – ×8 up‑scale (32 → 256)
# ════════════════════════════════════════════════════════════════════════════════


def pretrain_track1(epochs: int = 30, batch_size: int = 8) -> None:
    """Stage‑1 MSE pre‑training for Track 1."""
    # ------------------------------------------------------------------ data ---
    root = Path("amls2_coursework/datasets")
    lr_dir = root / "DIV2K_train_LR_cropped"
    hr_dir = root / "DIV2K_train_HR_cropped"

    lr_imgs, hr_imgs = _load_cropped_png_pairs(lr_dir, hr_dir)
    lr_batches = _batched(lr_imgs, batch_size)
    hr_batches = _batched(hr_imgs, batch_size)

    # ------------------------------------------------------------- model/optim ---
    generator = _build_generator_track1((32, 32, 3))
    generator.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
    )

    # ----------------------------------------------------------------- train ---
    for epoch in range(epochs):
        losses = []
        for lr_b, hr_b in tqdm(zip(lr_batches, hr_batches), total=len(hr_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            losses.append(generator.train_on_batch(lr_b, hr_b))
        print(f"Epoch {epoch+1:02d} – MSE: {np.mean(losses):.4f}")

    # ----------------------------------------------------------------- save  ---
    generator.save_weights("generator_pretrained_track1.h5")



def vgg_track1(epochs: int = 30, batch_size: int = 16) -> None:
    """Stage‑2 adversarial/VGG fine‑tune for Track 1."""
    # --------------------------------------------------------------- dataset ---
    n_random_crops_train, n_random_crops_val = 15, 2
    (
        train_lr,
        train_hr,
        val_lr,
        val_hr,
    ) = crop_patch_pairs(n_random_crops_train, n_random_crops_val)

    val_lr_batches, val_hr_batches = create_batches(val_lr, val_hr, batch_size)

    # ------------------------------------------------------------- models ---
    lr_shape, hr_shape = (32, 32, 3), (256, 256, 3)
    generator = build_generator(res_blocks=16, input_shape=lr_shape, final_activation="sigmoid")
    generator.load_weights("generator_pretrained_track1.h5")

    discriminator = build_discriminator(hr_shape)
    vgg_extractor = build_vgg_for_content_loss()

    trainer = SRGANTrainer(
        generator,
        discriminator,
        vgg_extractor,
        lr=1e-4,
        beta_1=0.5,
        lambda_adv=1e-3,
        lambda_vgg=0.5,
        lambda_mse=1.0,
    )

    # -------------------------------------------------------------- train ---
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        # Shuffle per‑epoch so we can re‑batch on the fly
        combined = list(zip(train_lr, train_hr))
        random.shuffle(combined)
        tlr, thr = zip(*combined)
        train_lr_batches, train_hr_batches = create_batches(tlr, thr, batch_size)

        d_losses, g_losses, adv_l, content_l, mse_l = [], [], [], [], []

        if epoch == 10:
            trainer.lambda_adv = 5e-3
            print("[Info] Changed adversarial weight → 0.005")
        if epoch == 20:
            trainer.opt_g.learning_rate.assign(5e-5)
            trainer.lambda_adv = 0.01
            print("[Info] Learning‑rate 5e‑5, adversarial weight 0.01")

        for lr_b, hr_b in tqdm(
            zip(train_lr_batches, train_hr_batches), total=len(train_hr_batches), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            d, g, adv, content, mse = trainer.train_step(lr_b, hr_b)
            d_losses.append(d.numpy())
            g_losses.append(g.numpy())
            adv_l.append(adv.numpy())
            content_l.append(content.numpy())
            mse_l.append(mse.numpy())

        # — validate —
        vd_l, vg_l, vadv_l, vcontent_l, vmse_l = [], [], [], [], []
        for lr_b, hr_b in zip(val_lr_batches, val_hr_batches):
            d, g, adv, content, mse = trainer.val_step(lr_b, hr_b)
            vd_l.append(d.numpy())
            vg_l.append(g.numpy())
            vadv_l.append(adv.numpy())
            vcontent_l.append(content.numpy())
            vmse_l.append(mse.numpy())

        # — history —
        train_hist.append((np.mean(d_losses), np.mean(g_losses), np.mean(adv_l), np.mean(content_l), np.mean(mse_l)))
        val_hist.append((np.mean(vd_l), np.mean(vg_l), np.mean(vadv_l), np.mean(vcontent_l), np.mean(vmse_l)))

        print(
            f"Epoch {epoch+1:02d} | Train: D {train_hist[-1][0]:.4f}  Adv {train_hist[-1][2]:.4f}  "
            f"VGG {train_hist[-1][3]:.4f}  MSE {train_hist[-1][4]:.4f}  G {train_hist[-1][1]:.4f} || "
            f"Val: D {val_hist[-1][0]:.4f}  Adv {val_hist[-1][2]:.4f}  VGG {val_hist[-1][3]:.4f}  "
            f"MSE {val_hist[-1][4]:.4f}  G {val_hist[-1][1]:.4f}"
        )

    generator.save_weights("generator_vgg_track1.h5")
    _plot_losses(train_hist, val_hist, "training_losses_track1.png")


# ════════════════════════════════════════════════════════════════════════════════
# Track 2 – ×4 up‑scale (32 → 128)
# ════════════════════════════════════════════════════════════════════════════════


def pretrain_track2(epochs: int = 100, batch_size: int = 32) -> None:
    n_random_crops_train, n_random_crops_val = 15, 1
    train_lr, train_hr, val_lr, val_hr = crop_patch_pairs(
        n_random_crops_train,
        n_random_crops_val,
        val_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_valid_LR_mild",
        train_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_train_LR_mild",
        scale=4,
    )

    val_lr_batches, val_hr_batches = create_batches(val_lr, val_hr, batch_size)

    generator = build_generator(
        res_blocks=16,
        upscale_2_power=2,  # 2^2 = 4 (32 → 128)
        input_shape=(32, 32, 3),
        final_activation="sigmoid",
    )
    generator.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(1e-4, beta_1=0.5))

    train_hist, val_hist = [], []
    for epoch in range(epochs):
        # shuffle + re‑batch each epoch
        combined = list(zip(train_lr, train_hr))
        random.shuffle(combined)
        tlr, thr = zip(*combined)
        train_lr_batches, train_hr_batches = create_batches(tlr, thr, batch_size)

        train_losses = []
        for lr_b, hr_b in tqdm(
            zip(train_lr_batches, train_hr_batches), total=len(train_hr_batches), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            train_losses.append(generator.train_on_batch(lr_b, hr_b))

        val_losses = [generator.test_on_batch(lr_b, hr_b) for lr_b, hr_b in zip(val_lr_batches, val_hr_batches)]
        train_mse, val_mse = np.mean(train_losses), np.mean(val_losses)
        train_hist.append(train_mse)
        val_hist.append(val_mse)
        print(f"Epoch {epoch+1:03d} – Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f}")

    generator.save_weights("generator_pretrained_track2.h5")
    _plot_simple(train_hist, val_hist, "pre_training_losses_track2.png")



def vgg_track2(epochs: int = 30, batch_size: int = 16) -> None:
    n_random_crops_train, n_random_crops_val = 15, 2
    train_lr, train_hr, val_lr, val_hr = crop_patch_pairs(
        n_random_crops_train,
        n_random_crops_val,
        val_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_valid_LR_mild",
        train_lr_dir="/home/uceebah/~AMLS2/amls2_coursework/datasets/DIV2K_train_LR_mild",
        scale=4,
    )

    val_lr_batches, val_hr_batches = create_batches(val_lr, val_hr, batch_size)

    lr_shape, hr_shape = (32, 32, 3), (128, 128, 3)
    generator = build_generator(res_blocks=16, input_shape=lr_shape, final_activation="sigmoid", upscale_2_power=2)
    generator.load_weights("generator_pretrained_track2.h5")

    discriminator = build_discriminator(hr_shape)
    vgg_extractor = build_vgg_for_content_loss()

    trainer = SRGANTrainer(
        generator,
        discriminator,
        vgg_extractor,
        lr=1e-4,
        beta_1=0.5,
        lambda_adv=1e-3,
        lambda_vgg=0.5,
        lambda_mse=1.0,
    )

    train_hist, val_hist = [], []
    for epoch in range(epochs):
        combined = list(zip(train_lr, train_hr))
        random.shuffle(combined)
        tlr, thr = zip(*combined)
        train_lr_batches, train_hr_batches = create_batches(tlr, thr, batch_size)

        d_losses, g_losses, adv_l, content_l, mse_l = [], [], [], [], []

        if epoch == 10:
            trainer.opt_g.learning_rate.assign(5e-5)
            trainer.lambda_adv = 5e-3
            print("[Info] LR 5e‑5, adversarial weight 0.005")
        if epoch == 20:
            trainer.lambda_adv = 0.01
            print("[Info] Adversarial weight 0.01")

        # — training loop —
        for lr_b, hr_b in tqdm(
            zip(train_lr_batches, train_hr_batches), total=len(train_hr_batches), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            d, g, adv, content, mse = trainer.train_step(lr_b, hr_b)
            d_losses.append(d.numpy())
            g_losses.append(g.numpy())
            adv_l.append(adv.numpy())
            content_l.append(content.numpy())
            mse_l.append(mse.numpy())

        # — validation —
        vd_l, vg_l, vadv_l, vcontent_l, vmse_l = [], [], [], [], []
        for lr_b, hr_b in zip(val_lr_batches, val_hr_batches):
            d, g, adv, content, mse = trainer.val_step(lr_b, hr_b)
            vd_l.append(d.numpy())
            vg_l.append(g.numpy())
            vadv_l.append(adv.numpy())
            vcontent_l.append(content.numpy())
            vmse_l.append(mse.numpy())

        train_hist.append((np.mean(d_losses), np.mean(g_losses), np.mean(adv_l), np.mean(content_l), np.mean(mse_l)))
        val_hist.append((np.mean(vd_l), np.mean(vg_l), np.mean(vadv_l), np.mean(vcontent_l), np.mean(vmse_l)))

        print(
            f"Epoch {epoch+1:02d} | Train: D {train_hist[-1][0]:.4f}  Adv {train_hist[-1][2]:.4f}  "
            f"VGG {train_hist[-1][3]:.4f}  MSE {train_hist[-1][4]:.4f}  G {train_hist[-1][1]:.4f} || "
            f"Val: D {val_hist[-1][0]:.4f}  Adv {val_hist[-1][2]:.4f}  VGG {val_hist[-1][3]:.4f}  "
            f"MSE {val_hist[-1][4]:.4f}  G {val_hist[-1][1]:.4f}"
        )

    generator.save_weights("generator_vgg_track2.h5")
    _plot_losses(train_hist, val_hist, "training_losses_track2.png")


# ════════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ════════════════════════════════════════════════════════════════════════════════


def _plot_losses(train_hist: List[Tuple[float, ...]], val_hist: List[Tuple[float, ...]], out_png: str) -> None:
    """Plot adversarial, VGG and MSE curves (3 sub‑plots)."""
    train_adv = [t[2] for t in train_hist]
    train_vgg = [t[3] for t in train_hist]
    train_mse = [t[4] for t in train_hist]

    val_adv = [v[2] for v in val_hist]
    val_vgg = [v[3] for v in val_hist]
    val_mse = [v[4] for v in val_hist]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(train_adv, label="Train")
    axes[0].plot(val_adv, label="Val")
    axes[0].set_title("Adversarial")

    axes[1].plot(train_vgg, label="Train")
    axes[1].plot(val_vgg, label="Val")
    axes[1].set_title("VGG / content")

    axes[2].plot(train_mse, label="Train")
    axes[2].plot(val_mse, label="Val")
    axes[2].set_title("MSE")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _plot_simple(train_loss: List[float], val_loss: List[float], out_png: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.title("Pre‑training MSE loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# Entry‑point
# ════════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SRGAN Training – combined tracks")
    parser.add_argument("--track", type=int, choices=[1, 2], required=True, help="Track number (1 or 2)")
    parser.add_argument(
        "--phase", choices=["pretrain", "vgg"], required=True, help="Training phase: pretrain (MSE) or vgg (adversarial)"
    )
    args = parser.parse_args()

    if args.track == 1 and args.phase == "pretrain":
        pretrain_track1()
    elif args.track == 1 and args.phase == "vgg":
        vgg_track1()
    elif args.track == 2 and args.phase == "pretrain":
        pretrain_track2()
    elif args.track == 2 and args.phase == "vgg":
        vgg_track2()
    else:
        raise ValueError("Unsupported combination of track and phase")


if __name__ == "__main__":
    main()
