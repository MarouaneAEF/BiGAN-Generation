#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom training script for BiGAN with command line arguments
"""

import argparse
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import bigan_improved_colors
from bigan_improved_colors import BiGAN
import os
import platform

def setup_device(device):
    """Setup training device (CPU, GPU, or MPS)"""
    if device == 'cpu':
        print("\nForcing CPU usage...")
        tf.config.set_visible_devices([], 'GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return "CPU"
        
    elif device == 'gpu':
        print("\nChecking for GPU...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for better memory management
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s):")
                for gpu in gpus:
                    print(f"  - {gpu}")
                return "GPU"
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
                print("Falling back to CPU...")
                return "CPU"
        else:
            print("No GPU found, falling back to CPU...")
            return "CPU"
            
    elif device == 'mps':
        print("\nChecking for Apple Metal (MPS)...")
        # Check if running on macOS and Apple Silicon
        is_macos = platform.system() == 'Darwin'
        is_arm = platform.machine().startswith('arm64')
        
        if is_macos and is_arm:
            try:
                # Try to get MPS device
                devices = tf.config.list_physical_devices()
                has_mps = any('MPS' in str(device) for device in devices)
                if has_mps:
                    print("MPS (Metal) device found and will be used!")
                    # Set memory growth
                    for device in devices:
                        try:
                            tf.config.experimental.set_memory_growth(device, True)
                        except:
                            pass
                    return "MPS"
            except:
                pass
        
        print("MPS (Metal) not available, falling back to CPU...")
        return "CPU"

def load_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    (x_train, _), (x_test, _) = cifar10.load_data()
    # Normalize to [-1, 1]
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    return x_train, x_test

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BiGAN on CIFAR-10')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use (currently only cifar10 supported)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for reconstruction loss')
    parser.add_argument('--adversarial_weight', type=float, default=1.0, help='Weight for adversarial loss')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'gpu', 'mps'], 
                        help='Device to use for training (cpu, gpu, or mps)')
    
    args = parser.parse_args()
    
    # Setup device for training
    device = setup_device(args.device)
    
    print("\nTraining configuration:")
    print("=" * 50)
    print(f"Device: {device}")
    for arg in vars(args):
        if arg != 'device':  # Skip device since we already printed it
            print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    # Load dataset
    if args.dataset == 'cifar10':
        print("\nLoading CIFAR-10 dataset...")
        x_train, x_test = load_cifar10()
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Set global variables in bigan_improved_colors
    bigan_improved_colors.EPOCHS = args.epochs
    bigan_improved_colors.BATCH_SIZE = args.batch_size
    bigan_improved_colors.LEARNING_RATE = args.learning_rate
    bigan_improved_colors.WEIGHT_RECON = args.beta
    bigan_improved_colors.WEIGHT_ADVERSARIAL = args.adversarial_weight
    bigan_improved_colors.LATENT_DIM = args.latent_dim
    
    # Create and train model
    print("\nInitializing BiGAN model...")
    model = BiGAN()
    
    print("\nStarting training...")
    print(f"Using device: {device}")
    model.train(x_train)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 