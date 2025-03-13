#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and prepare datasets for BiGAN training
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for BiGAN training')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'mnist', 'fashion_mnist', 'flowers'],
                        help='Dataset to download')
    parser.add_argument('--category', type=int, default=None, 
                        help='Specific category to extract (e.g., 0-9 for CIFAR-10)')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Directory to save extracted images')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Maximum number of samples to extract')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Size to resize images to')
    return parser.parse_args()

def download_with_progress(url, destination):
    """Download a file with a progress bar"""
    
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    
    # Create a progress bar
    response = urllib.request.urlopen(url)
    file_size = int(response.headers.get('Content-Length', 0))
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)
    
    # Download the file
    with open(destination, 'wb') as f:
        while True:
            buffer = response.read(8192)
            if not buffer:
                break
            f.write(buffer)
            progress_bar.update(len(buffer))
    
    progress_bar.close()
    print(f"Download completed: {destination}")

def prepare_cifar10(output_dir, category=None, num_samples=1000, image_size=32):
    """Extract images from CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), _ = cifar10.load_data()
    
    # Classes in CIFAR-10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Filter by category if specified
    if category is not None:
        if category < 0 or category >= len(classes):
            raise ValueError(f"Invalid category {category}. Must be between 0 and {len(classes)-1}")
        
        # Get indices for this category
        indices = np.where(y_train.flatten() == category)[0]
        class_name = classes[category]
        print(f"Extracting category {category} ({class_name})")
        
        # Limit number of samples
        if len(indices) > num_samples:
            indices = indices[:num_samples]
        
        # Create output directory
        output_subdir = os.path.join(output_dir, f"cifar10_{class_name}")
        os.makedirs(output_subdir, exist_ok=True)
        
        # Save images
        for i, idx in enumerate(indices):
            img = Image.fromarray(x_train[idx])
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(os.path.join(output_subdir, f"{class_name}_{i:04d}.png"))
        
        print(f"Saved {len(indices)} images to {output_subdir}")
    else:
        # Extract samples from each class
        samples_per_class = num_samples // len(classes)
        
        for class_idx, class_name in enumerate(classes):
            # Get indices for this class
            indices = np.where(y_train.flatten() == class_idx)[0]
            
            # Limit number of samples
            if len(indices) > samples_per_class:
                indices = indices[:samples_per_class]
            
            # Create output directory
            output_subdir = os.path.join(output_dir, f"cifar10_{class_name}")
            os.makedirs(output_subdir, exist_ok=True)
            
            # Save images
            for i, idx in enumerate(indices):
                img = Image.fromarray(x_train[idx])
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(os.path.join(output_subdir, f"{class_name}_{i:04d}.png"))
            
            print(f"Saved {len(indices)} images of class {class_name} to {output_subdir}")

def prepare_mnist(output_dir, category=None, num_samples=1000, image_size=32, fashion=False):
    """Extract images from MNIST or Fashion-MNIST dataset"""
    if fashion:
        print("Loading Fashion-MNIST dataset...")
        (x_train, y_train), _ = fashion_mnist.load_data()
        dataset_name = "fashion_mnist"
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        print("Loading MNIST dataset...")
        (x_train, y_train), _ = mnist.load_data()
        dataset_name = "mnist"
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Filter by category if specified
    if category is not None:
        if category < 0 or category >= len(classes):
            raise ValueError(f"Invalid category {category}. Must be between 0 and {len(classes)-1}")
        
        # Get indices for this category
        indices = np.where(y_train.flatten() == category)[0]
        class_name = classes[category].replace('/', '_')  # Replace slashes for file paths
        print(f"Extracting category {category} ({class_name})")
        
        # Limit number of samples
        if len(indices) > num_samples:
            indices = indices[:num_samples]
        
        # Create output directory
        output_subdir = os.path.join(output_dir, f"{dataset_name}_{class_name}")
        os.makedirs(output_subdir, exist_ok=True)
        
        # Save images
        for i, idx in enumerate(indices):
            # Convert grayscale to RGB
            img_array = np.stack([x_train[idx]] * 3, axis=-1)
            img = Image.fromarray(img_array)
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(os.path.join(output_subdir, f"{class_name}_{i:04d}.png"))
        
        print(f"Saved {len(indices)} images to {output_subdir}")
    else:
        # Extract samples from each class
        samples_per_class = num_samples // len(classes)
        
        for class_idx, class_name in enumerate(classes):
            class_name = class_name.replace('/', '_')  # Replace slashes for file paths
            
            # Get indices for this class
            indices = np.where(y_train.flatten() == class_idx)[0]
            
            # Limit number of samples
            if len(indices) > samples_per_class:
                indices = indices[:samples_per_class]
            
            # Create output directory
            output_subdir = os.path.join(output_dir, f"{dataset_name}_{class_name}")
            os.makedirs(output_subdir, exist_ok=True)
            
            # Save images
            for i, idx in enumerate(indices):
                # Convert grayscale to RGB
                img_array = np.stack([x_train[idx]] * 3, axis=-1)
                img = Image.fromarray(img_array)
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(os.path.join(output_subdir, f"{class_name}_{i:04d}.png"))
            
            print(f"Saved {len(indices)} images of class {class_name} to {output_subdir}")

def download_and_extract_flowers(output_dir, num_samples=1000, image_size=32):
    """Download and extract the Flowers dataset"""
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    download_path = os.path.join(output_dir, "flower_photos.tgz")
    extract_path = os.path.join(output_dir, "flower_photos")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(extract_path):
        if not os.path.exists(download_path):
            download_with_progress(url, download_path)
        
        # Extract the dataset
        print(f"Extracting {download_path} to {output_dir}")
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    
    # Process each class
    samples_per_class = num_samples
    total_samples = 0
    
    # Create an organized output structure
    organized_dir = os.path.join(output_dir, "flowers_organized")
    os.makedirs(organized_dir, exist_ok=True)
    
    # Get the list of classes (subdirectories)
    class_dirs = [d for d in os.listdir(extract_path) 
                 if os.path.isdir(os.path.join(extract_path, d)) and not d.startswith('.')]
    
    for class_name in class_dirs:
        class_dir = os.path.join(extract_path, class_name)
        output_class_dir = os.path.join(organized_dir, f"flowers_{class_name}")
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit to number of samples
        if len(image_files) > samples_per_class:
            image_files = image_files[:samples_per_class]
        
        print(f"Processing {len(image_files)} images of {class_name}")
        
        # Process each image
        for i, filename in enumerate(image_files):
            try:
                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(os.path.join(output_class_dir, f"{class_name}_{i:04d}.png"))
                total_samples += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"Saved a total of {total_samples} flower images to {organized_dir}")
    return organized_dir

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download and prepare the selected dataset
    if args.dataset == 'cifar10':
        prepare_cifar10(args.output_dir, args.category, args.num_samples, args.image_size)
    elif args.dataset == 'mnist':
        prepare_mnist(args.output_dir, args.category, args.num_samples, args.image_size)
    elif args.dataset == 'fashion_mnist':
        prepare_mnist(args.output_dir, args.category, args.num_samples, args.image_size, fashion=True)
    elif args.dataset == 'flowers':
        flowers_dir = download_and_extract_flowers(args.output_dir, args.num_samples, args.image_size)
        print(f"Flowers dataset prepared in: {flowers_dir}")
    else:
        print(f"Unknown dataset: {args.dataset}")
        return
    
    print("\nDataset preparation complete!")
    print(f"To train BiGAN on this dataset, run:")
    
    if args.dataset == 'flowers':
        print(f"python run_custom_training.py --img_dir {args.output_dir}/flowers_organized/flowers_[class_name]")
    elif args.category is not None:
        if args.dataset == 'cifar10':
            class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck'][args.category]
            print(f"python run_custom_training.py --img_dir {args.output_dir}/cifar10_{class_name}")
        elif args.dataset == 'mnist':
            print(f"python run_custom_training.py --img_dir {args.output_dir}/mnist_{args.category}")
        elif args.dataset == 'fashion_mnist':
            class_name = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'][args.category]
            print(f"python run_custom_training.py --img_dir {args.output_dir}/fashion_mnist_{class_name}")
    else:
        print(f"python run_custom_training.py --img_dir {args.output_dir}/[class_directory]")

if __name__ == "__main__":
    main() 