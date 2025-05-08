import time
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from metrics import calculate_fid, discriminator_loss, generator_loss
import wandb
from models import build_discriminator,build_generator
import numpy as np

def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, images, latent_dim):
    #performing one training step for the GAN
    #creating fake images from noise
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train_model(train_dataset, val_data, test_data, train_images, val_images, test_images, latent_dim, generator_filters, dropout_rate, learning_rate, run_name):
    from config import OUTPUT_DIR, CHECKPOINT_DIR, EPOCHS
    # Creating a subfolder for this run
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    # Building models
    generator = build_generator(latent_dim, generator_filters, dropout_rate)
    discriminator = build_discriminator(generator_filters, dropout_rate)
    # Setup optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    # Initialize metrics tracking
    gen_losses, disc_losses = [], []
    train_fids, val_fids, test_fids = [], [], []
    # Create a fixed seed for visualization
    fixed_seed = tf.random.normal([16, latent_dim])
    images_for_gif = []
    # Start training
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} - {run_name}")
        start_time = time.time()
        # Training on batches
        epoch_gen_loss = []
        epoch_disc_loss = []
        for image_batch in train_dataset:
            gen_loss, disc_loss = train_step(
                generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                image_batch, latent_dim
            )
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)
        # Average losses for the epoch
        avg_gen_loss = tf.reduce_mean(epoch_gen_loss).numpy()
        avg_disc_loss = tf.reduce_mean(epoch_disc_loss).numpy()
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        # Generate images from fixed seed
        generated_images = generator(fixed_seed, training=False)
        # Save one sample image
        img = (generated_images[0].numpy() + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        images_for_gif.append(img)
        imageio.imwrite(os.path.join(run_dir, f"epoch_{epoch+1}.png"), img)
        # Calculate FID every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            # Training FID
            train_sample = train_images[:100]
            noise_sample = tf.random.normal([100, latent_dim])
            gen_sample = generator(noise_sample, training=False).numpy()
            train_fid = calculate_fid(train_sample, gen_sample)
            train_fids.append(train_fid)
            # Validation FID
            val_sample = val_images[:100]
            val_fid = calculate_fid(val_sample, gen_sample)
            val_fids.append(val_fid)
            # Log FID to wandb
            wandb.log({
                f"{run_name}/train_fid": train_fid,
                f"{run_name}/val_fid": val_fid,
                "epoch": epoch + 1
            })
            # Save checkpoint
            generator.save(os.path.join(CHECKPOINT_DIR, f"{run_name}_generator_epoch_{epoch+1}.h5"))
        # Log losses
        wandb.log({
            f"{run_name}/gen_loss": avg_gen_loss,
            f"{run_name}/disc_loss": avg_disc_loss,
            "epoch": epoch + 1
        })
        print(f"  Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {time.time() - start_time:.2f}s")
    # Create GIF for this run
    gif_path = os.path.join(run_dir, f"{run_name}_progress.gif")
    imageio.mimsave(gif_path, images_for_gif, fps=5)
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title(f'Training Losses - {run_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot FID curves
    plt.subplot(1, 2, 2)
    epochs_measured = [e*5 for e in range(len(train_fids))]
    plt.plot(epochs_measured, train_fids, label='Train FID')
    plt.plot(epochs_measured, val_fids, label='Validation FID')
    plt.title(f'FID Scores - {run_name}')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{run_name}_metrics.png"))
    plt.close()
    # Evaluate on test set
    test_noise = tf.random.normal([100, latent_dim])
    test_generated = generator(test_noise, training=False)
    test_fid = calculate_fid(test_images[:100], test_generated.numpy())
    print(f"Final Test FID for {run_name}: {test_fid:.4f}")
    # Save test samples
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((test_generated[i].numpy() + 1) * 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{run_name}_test_samples.png"))
    plt.close()
    # Return results
    return {
        'run_name': run_name,
        'latent_dim': latent_dim,
        'generator_filters': generator_filters,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'final_gen_loss': gen_losses[-1],
        'final_disc_loss': disc_losses[-1],
        'test_fid': test_fid,
        'val_fid': val_fids[-1] if val_fids else None,
        'train_fid': train_fids[-1] if train_fids else None
    }