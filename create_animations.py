#!/usr/bin/env python3
import os
import glob
from PIL import Image
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_gif(image_dir, pattern, output_path, duration=500):
    """Create a GIF from images matching the pattern in the directory."""
    # Get list of images and sort them
    images = glob.glob(os.path.join(image_dir, pattern))
    images.sort(key=natural_sort_key)
    
    if not images:
        print(f"No images found matching pattern {pattern} in {image_dir}")
        return
    
    # Open all images
    frames = [Image.open(image) for image in images]
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Created animation: {output_path}")

def main():
    # Create animations directory if it doesn't exist
    os.makedirs("animations", exist_ok=True)
    
    # Create GIFs for different aspects of training
    create_gif(
        "bigan_output",
        "generated_epoch_*.png",
        "animations/generated_samples.gif"
    )
    
    create_gif(
        "bigan_output",
        "reconstruction_epoch_*.png",
        "animations/reconstruction.gif"
    )
    
    create_gif(
        "bigan_output",
        "training_curves_epoch_*.png",
        "animations/training_curves.gif"
    )

if __name__ == "__main__":
    main() 