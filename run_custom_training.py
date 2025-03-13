#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom training script for BiGAN on user-provided images
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
from tensorflow.keras import mixed_precision
import sys
from bigan_improved_colors import BiGAN, LATENT_DIM, BATCH_SIZE, IMG_SHAPE, WEIGHT_RECON, WEIGHT_ADVERSARIAL

# Import the EPOCHS variable
from importlib import reload
import bigan_improved_colors
EPOCHS = bigan_improved_colors.EPOCHS

def parse_args():
    parser = argparse.ArgumentParser(description='Train BiGAN on custom image dataset')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu', 'mps', 'cuda'], 
                        help='Device to use for training')
    parser.add_argument('--beta', type=float, default=1.0, 
                        help='Weight factor for reconstruction loss (higher = more focus on reconstruction)')
    parser.add_argument('--image_size', type=int, default=32, help='Size to resize images to (square)')
    parser.add_argument('--save_dir', type=str, default='bigan_custom_output', help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval for saving results')
    parser.add_argument('--subset_ratio', type=float, default=1.0, help='Ratio of dataset to use (0-1)')
    
    return parser.parse_args()

def load_images_from_directory(directory, image_size=32):
    """
    Load images from a directory and convert them to the format expected by BiGAN.
    """
    print(f"Loading images from {directory}...")
    
    # Find all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
        image_files.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    
    if not image_files:
        raise ValueError(f"No image files found in {directory}")
    
    print(f"Found {len(image_files)} images")
    
    # Load and preprocess images
    images = []
    for img_path in image_files:
        try:
            # Open and resize the image
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((image_size, image_size), Image.LANCZOS)
            
            # Convert to numpy array and normalize to [-1, 1]
            img_array = np.array(img)
            img_array = (img_array.astype(np.float32) - 127.5) / 127.5
            
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    # Stack into a batch
    images = np.stack(images, axis=0)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    return images

def configure_device(device_name):
    """Configure TensorFlow to use the specified device."""
    if device_name == 'cpu':
        print("Using CPU for training")
        # Force CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    elif device_name in ['gpu', 'cuda']:
        # Check for CUDA GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU for training: {gpus}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found, falling back to CPU")
            
    elif device_name == 'mps':
        # Check for Apple Silicon MPS
        try:
            # Try to enable Metal Performance Shaders (MPS) for Apple Silicon
            if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'set_visible_devices'):
                devices = tf.config.list_physical_devices('MPS')
                if devices:
                    tf.config.experimental.set_visible_devices(devices[0], 'MPS')
                    print("Using MPS (Apple Silicon GPU) for training")
                else:
                    print("MPS device not found, falling back to CPU")
            else:
                # Alternative approach for older TF versions
                # Set the environment variable to enable MPS
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                print("Attempting to use MPS through environment variable")
        except Exception as e:
            print(f"Error configuring MPS: {e}")
            print("Falling back to CPU")

def main():
    """Main function for custom BiGAN training."""
    args = parse_args()
    
    # Configure device
    configure_device(args.device)
    
    # Load images
    x_train = load_images_from_directory(args.img_dir, args.image_size)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Modify global constants in the bigan_improved_colors module
    # This is a bit hacky but allows us to control the constants
    # that are used throughout the bigan_improved_colors.py file
    bigan_improved_colors.EPOCHS = args.epochs
    bigan_improved_colors.BATCH_SIZE = args.batch_size
    bigan_improved_colors.WEIGHT_RECON = bigan_improved_colors.WEIGHT_RECON * args.beta
    reload(bigan_improved_colors)  # Reload to ensure changes take effect
    
    # Print training configuration
    print("\n=== BiGAN Training Configuration ===")
    print(f"Image directory: {args.img_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Beta (reconstruction weight factor): {args.beta}")
    print(f"Reconstruction weight: {bigan_improved_colors.WEIGHT_RECON}")
    print(f"Adversarial weight: {bigan_improved_colors.WEIGHT_ADVERSARIAL}")
    print(f"Subset ratio: {args.subset_ratio}")
    print(f"Save directory: {args.save_dir}")
    print(f"Save interval: {args.save_interval}")
    print("=====================================\n")
    
    # Create and train the model
    model = BiGAN()
    
    # Override the save location in the model
    # First, monkey patch the save methods to use our custom directory
    original_save_samples = model.save_samples
    original_save_reconstructions = model.save_reconstructions
    original_save_real_samples = model.save_real_samples
    original_save_training_curves = model.save_training_curves
    
    def patched_save_samples(epoch, n_samples=25):
        """Patched save_samples method to use custom save directory"""
        z_samples = np.random.normal(0, 1, (n_samples, LATENT_DIM))
        gen_imgs = model.generator.predict(z_samples, verbose=0)
        
        # Display statistics
        print(f"Generated images - Min/Max: {gen_imgs.min():.4f}/{gen_imgs.max():.4f}")
        print(f"Generated images - Mean/Std: {gen_imgs.mean():.4f}/{gen_imgs.std():.4f}")
        
        # Use original function but modify the save path
        import matplotlib.pyplot as plt
        # 5x5 display grid
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        
        # Convert from [-1, 1] to [0, 1] for display
        gen_imgs_display = (gen_imgs + 1) / 2.0
        
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs_display[cnt])
                axs[i, j].axis('off')
                cnt += 1
        
        plt.savefig(f"{args.save_dir}/generated_epoch_{epoch}.png")
        plt.close()
    
    def patched_save_real_samples(samples):
        """Patched save_real_samples method to use custom save directory"""
        import matplotlib.pyplot as plt
        n_samples = len(samples)
        n_row = int(np.sqrt(n_samples))
        n_col = n_samples // n_row
        
        fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
        
        # Convert to [0, 1] for display
        display_samples = (samples + 1) / 2.0
        
        cnt = 0
        for i in range(n_row):
            for j in range(n_col):
                axs[i, j].imshow(display_samples[cnt])
                axs[i, j].axis('off')
                cnt += 1
        
        plt.savefig(f"{args.save_dir}/real_samples.png")
        plt.close()
    
    def patched_save_reconstructions(samples, epoch):
        """Patched save_reconstructions method to use custom save directory"""
        import matplotlib.pyplot as plt
        # Encode samples
        encoded = model.encoder.predict(samples, verbose=0)
        
        # Regenerate images
        reconstructed = model.generator.predict(encoded, verbose=0)
        
        # Prepare display
        plt.figure(figsize=(20, 4))
        
        # Convert from [-1, 1] to [0, 1] for display
        samples_display = (samples + 1) / 2.0
        reconstructed_display = (reconstructed + 1) / 2.0
        
        # Display originals and reconstructions
        for i in range(len(samples)):
            # Original
            plt.subplot(2, len(samples), i+1)
            plt.imshow(samples_display[i])
            plt.title("Original")
            plt.axis('off')
            
            # Reconstruction
            plt.subplot(2, len(samples), i+len(samples)+1)
            plt.imshow(reconstructed_display[i])
            plt.title("Reconstruction")
            plt.axis('off')
        
        plt.savefig(f"{args.save_dir}/reconstruction_epoch_{epoch}.png")
        plt.close()
    
    def patched_save_training_curves(d_losses, g_losses, r_losses, d_accs, epoch):
        """Patched save_training_curves method to use custom save directory"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        
        # Adversarial losses
        plt.subplot(2, 2, 1)
        plt.plot(d_losses, label='Discriminator')
        plt.plot(g_losses, label='Generator')
        plt.title('Adversarial Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Reconstruction losses
        plt.subplot(2, 2, 2)
        plt.plot(r_losses, label='MSE', color='blue')
        plt.title('Reconstruction Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Discriminator accuracy
        plt.subplot(2, 2, 3)
        plt.plot(np.array(d_accs) * 100)
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim([0, 100])
        
        # Pixel distribution for the last generation
        plt.subplot(2, 2, 4)
        z = np.random.normal(0, 1, (1, LATENT_DIM))
        img = model.generator.predict(z, verbose=0)[0]
        
        # Histogram per color channel
        for c, color in enumerate(['r', 'g', 'b']):
            values = img[:,:,c].flatten()
            plt.hist(values, bins=50, color=color, alpha=0.7)
            
        plt.title(f'Pixel Distribution (min={img.min():.2f}, max={img.max():.2f})')
        plt.xlim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"{args.save_dir}/training_curves_epoch_{epoch}.png")
        plt.close()
    
    # Apply the monkey patches
    model.save_samples = patched_save_samples
    model.save_real_samples = patched_save_real_samples
    model.save_reconstructions = patched_save_reconstructions
    model.save_training_curves = patched_save_training_curves
    
    # Train the model
    model.train(
        x_train=x_train,
        subset_ratio=args.subset_ratio
    )
    
    print("\n==== Training completed ====")
    print(f"Results are saved in the '{args.save_dir}' folder")

if __name__ == "__main__":
    main() 