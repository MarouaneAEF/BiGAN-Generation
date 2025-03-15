#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from PIL import Image
import natsort

def create_reconstruction_animation():
    # Create output directory if it doesn't exist
    os.makedirs("animations", exist_ok=True)
    
    # Get all reconstruction images
    image_files = glob.glob("bigan_custom_output/reconstruction_epoch_*.png")
    image_files = natsort.natsorted(image_files)  # Sort naturally by epoch number
    
    if not image_files:
        print("No reconstruction images found!")
        return
    
    print(f"Found {len(image_files)} reconstruction images")
    
    # Load all images
    images = []
    for img_file in image_files:
        img = Image.open(img_file)
        images.append(img)
    
    # Save as GIF
    output_path = "animations/custom_reconstruction.gif"
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 500ms per frame
        loop=0
    )
    print(f"Created animation: {output_path}")

if __name__ == "__main__":
    create_reconstruction_animation() 