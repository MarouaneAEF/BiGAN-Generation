#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate synthetic images for BiGAN training
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import random
import math
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic images for BiGAN training')
    parser.add_argument('--output_dir', type=str, default='synthetic_data',
                        help='Directory to save generated images')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=32,
                        help='Size of generated images')
    parser.add_argument('--pattern', type=str, default='shapes',
                        choices=['shapes', 'gradient', 'noise', 'circles', 'mixed'],
                        help='Pattern to generate')
    parser.add_argument('--color_mode', type=str, default='rgb',
                        choices=['rgb', 'grayscale'],
                        help='Color mode of generated images')
    return parser.parse_args()

def generate_random_color(grayscale=False):
    """Generate a random color"""
    if grayscale:
        val = random.randint(0, 255)
        return (val, val, val)
    else:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_shapes(image_size, color_mode='rgb'):
    """Generate an image with random shapes"""
    grayscale = (color_mode == 'grayscale')
    img = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Number of shapes to draw
    num_shapes = random.randint(1, 5)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'circle', 'line', 'triangle'])
        color = generate_random_color(grayscale)
        
        if shape_type == 'rectangle':
            x1 = random.randint(0, image_size - 1)
            y1 = random.randint(0, image_size - 1)
            x2 = random.randint(x1, image_size - 1)
            y2 = random.randint(y1, image_size - 1)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        elif shape_type == 'circle':
            x = random.randint(0, image_size - 1)
            y = random.randint(0, image_size - 1)
            radius = random.randint(5, image_size // 4)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        
        elif shape_type == 'line':
            x1 = random.randint(0, image_size - 1)
            y1 = random.randint(0, image_size - 1)
            x2 = random.randint(0, image_size - 1)
            y2 = random.randint(0, image_size - 1)
            width = random.randint(1, 5)
            draw.line([x1, y1, x2, y2], fill=color, width=width)
        
        elif shape_type == 'triangle':
            x1 = random.randint(0, image_size - 1)
            y1 = random.randint(0, image_size - 1)
            x2 = random.randint(0, image_size - 1)
            y2 = random.randint(0, image_size - 1)
            x3 = random.randint(0, image_size - 1)
            y3 = random.randint(0, image_size - 1)
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=color)
    
    return img

def generate_gradient(image_size, color_mode='rgb'):
    """Generate an image with a random gradient"""
    grayscale = (color_mode == 'grayscale')
    img = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
    pixels = img.load()
    
    # Choose gradient type
    gradient_type = random.choice(['linear', 'radial', 'angular'])
    
    if grayscale:
        start_val = random.randint(0, 255)
        end_val = random.randint(0, 255)
        start_color = (start_val, start_val, start_val)
        end_color = (end_val, end_val, end_val)
    else:
        start_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        end_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    if gradient_type == 'linear':
        # Linear gradient
        angle = random.uniform(0, 2 * math.pi)
        for x in range(image_size):
            for y in range(image_size):
                # Calculate distance along the gradient direction
                dist = (x * math.cos(angle) + y * math.sin(angle)) / (image_size * math.sqrt(2))
                # Normalize to [0, 1]
                dist = (dist + 1) / 2
                # Interpolate colors
                r = int(start_color[0] * (1 - dist) + end_color[0] * dist)
                g = int(start_color[1] * (1 - dist) + end_color[1] * dist)
                b = int(start_color[2] * (1 - dist) + end_color[2] * dist)
                pixels[x, y] = (r, g, b)
    
    elif gradient_type == 'radial':
        # Radial gradient
        center_x = random.randint(0, image_size - 1)
        center_y = random.randint(0, image_size - 1)
        max_dist = math.sqrt(image_size**2 + image_size**2)
        
        for x in range(image_size):
            for y in range(image_size):
                # Calculate distance from center
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
                # Interpolate colors
                r = int(start_color[0] * (1 - dist) + end_color[0] * dist)
                g = int(start_color[1] * (1 - dist) + end_color[1] * dist)
                b = int(start_color[2] * (1 - dist) + end_color[2] * dist)
                pixels[x, y] = (r, g, b)
    
    else:  # angular
        # Angular gradient
        center_x = random.randint(0, image_size - 1)
        center_y = random.randint(0, image_size - 1)
        
        for x in range(image_size):
            for y in range(image_size):
                # Calculate angle from center
                angle = (math.atan2(y - center_y, x - center_x) + math.pi) / (2 * math.pi)
                # Interpolate colors
                r = int(start_color[0] * (1 - angle) + end_color[0] * angle)
                g = int(start_color[1] * (1 - angle) + end_color[1] * angle)
                b = int(start_color[2] * (1 - angle) + end_color[2] * angle)
                pixels[x, y] = (r, g, b)
    
    return img

def generate_noise(image_size, color_mode='rgb'):
    """Generate an image with random noise"""
    grayscale = (color_mode == 'grayscale')
    img = Image.new('RGB', (image_size, image_size))
    pixels = img.load()
    
    noise_type = random.choice(['uniform', 'perlin-like'])
    
    if noise_type == 'uniform':
        # Uniform random noise
        for x in range(image_size):
            for y in range(image_size):
                if grayscale:
                    val = random.randint(0, 255)
                    pixels[x, y] = (val, val, val)
                else:
                    pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    else:  # perlin-like
        # Simplified Perlin-like noise
        scale = random.randint(1, 10)
        octaves = random.randint(1, 3)
        persistence = random.random() * 0.5 + 0.25
        
        def generate_random_grid(size, scale):
            grid = np.zeros((size // scale + 2, size // scale + 2, 3), dtype=np.float32)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grayscale:
                        val = random.random()
                        grid[i, j] = [val, val, val]
                    else:
                        grid[i, j] = [random.random(), random.random(), random.random()]
            return grid
        
        def interpolate(a, b, x):
            # Smoothstep interpolation
            ft = x * np.pi
            f = (1 - np.cos(ft)) * 0.5
            return a * (1 - f) + b * f
        
        # Generate noise for each octave
        result = np.zeros((image_size, image_size, 3), dtype=np.float32)
        
        for octave in range(octaves):
            current_scale = scale * (2 ** octave)
            amplitude = persistence ** octave
            
            grid = generate_random_grid(image_size, current_scale)
            
            for x in range(image_size):
                for y in range(image_size):
                    # Grid cell coordinates
                    grid_x = x // current_scale
                    grid_y = y // current_scale
                    
                    # Local coordinates [0,1]
                    local_x = (x % current_scale) / current_scale
                    local_y = (y % current_scale) / current_scale
                    
                    # Get values at corners of the current grid cell
                    c00 = grid[grid_y, grid_x]
                    c10 = grid[grid_y, grid_x + 1]
                    c01 = grid[grid_y + 1, grid_x]
                    c11 = grid[grid_y + 1, grid_x + 1]
                    
                    # Interpolate along x
                    i0 = interpolate(c00, c10, local_x)
                    i1 = interpolate(c01, c11, local_x)
                    
                    # Interpolate along y
                    value = interpolate(i0, i1, local_y)
                    
                    # Add to result
                    result[y, x] += value * amplitude
        
        # Normalize and convert to RGB
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # Create image from the result array
        img = Image.fromarray(result, 'RGB')
    
    return img

def generate_circles(image_size, color_mode='rgb'):
    """Generate an image with concentric circles"""
    grayscale = (color_mode == 'grayscale')
    img = Image.new('RGB', (image_size, image_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Random center
    center_x = random.randint(image_size // 4, 3 * image_size // 4)
    center_y = random.randint(image_size // 4, 3 * image_size // 4)
    
    # Number of circles
    num_circles = random.randint(3, 10)
    
    # Max radius
    max_radius = random.randint(image_size // 4, image_size // 2)
    
    for i in range(num_circles):
        radius = max_radius * (1 - i / num_circles)
        color = generate_random_color(grayscale)
        draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=color, width=2)
        
        # Sometimes fill the circle
        if random.random() < 0.3:
            fill_color = generate_random_color(grayscale)
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=fill_color)
    
    return img

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"synthetic_{args.pattern}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} {args.pattern} images...")
    
    pattern_generators = {
        'shapes': generate_shapes,
        'gradient': generate_gradient,
        'noise': generate_noise,
        'circles': generate_circles
    }
    
    for i in tqdm(range(args.num_samples)):
        if args.pattern == 'mixed':
            # Choose a random pattern type
            pattern_type = random.choice(list(pattern_generators.keys()))
            generator = pattern_generators[pattern_type]
            img = generator(args.image_size, args.color_mode)
            img.save(os.path.join(output_dir, f"mixed_{pattern_type}_{i:04d}.png"))
        else:
            generator = pattern_generators[args.pattern]
            img = generator(args.image_size, args.color_mode)
            img.save(os.path.join(output_dir, f"{args.pattern}_{i:04d}.png"))
    
    print(f"Generated {args.num_samples} {args.pattern} images in {output_dir}")
    print("\nTo train BiGAN on this dataset, run:")
    print(f"python run_custom_training.py --img_dir {output_dir}")

if __name__ == "__main__":
    main() 