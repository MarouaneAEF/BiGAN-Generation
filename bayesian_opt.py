#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bayesian optimization for BiGAN hyperparameters using CIFAR-10 dataset
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
import os
import json
from datetime import datetime
import importlib

# Define the hyperparameter search space
space = [
    Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
    Integer(32, 128, name='batch_size'),
    Real(0.1, 10.0, name='weight_recon', prior='log-uniform'),
    Real(0.1, 2.0, name='weight_adversarial', prior='log-uniform'),
    Integer(64, 256, name='latent_dim'),
]

def load_cifar_data():
    """Load and preprocess CIFAR-10 data"""
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    # Normalize to [-1, 1]
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_test = (x_test.astype('float32') - 127.5) / 127.5
    return x_train, x_test

def compute_validation_loss(model, val_data, batch_size=32):
    """Compute validation loss on a subset of data"""
    total_loss = 0
    n_samples = len(val_data)
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = val_data[start_idx:end_idx]
        
        # Get reconstructions
        z = model.encoder(batch, training=False)
        reconstructed = model.generator(z, training=False)
        
        # Compute MSE
        mse = tf.reduce_mean(tf.square(batch - reconstructed))
        total_loss += float(mse.numpy())  # Convert to Python float
    
    return total_loss / n_batches

@use_named_args(space)
def objective(**params):
    """Objective function to minimize"""
    print("\nEvaluating parameters:", params)
    
    try:
        # Load CIFAR data
        x_train, x_test = load_cifar_data()
        
        # Take a small subset for quick evaluation
        train_subset = x_train[:5000]  # Use 5000 images for training
        val_subset = x_test[:1000]    # Use 1000 images for validation
        
        # Create and compile model
        import bigan_improved_colors
        model = bigan_improved_colors.BiGAN()  # Initialize without parameters
        
        # Train for a few epochs
        model.train(
            train_subset,
            batch_size=int(params['batch_size']),  # Convert to Python int
            epochs=3,  # Use just 3 epochs for optimization
            learning_rate=float(params['learning_rate']),  # Convert to Python float
            beta=float(params['weight_recon']),  # Convert to Python float
            adversarial_weight=float(params['weight_adversarial']),  # Convert to Python float
            latent_dimension=int(params['latent_dim'])  # Convert to Python int
        )
        
        # Compute validation loss
        val_loss = compute_validation_loss(model, val_subset, int(params['batch_size']))
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
        
        print(f"Validation loss: {val_loss:.4f}")
        return float(val_loss)  # Convert to Python float
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return float(1e10)  # Convert to Python float

def format_optimal_command(best_params):
    """Format the optimal command with all hyperparameters"""
    cmd = f"""python run_custom_training.py \\
    --dataset cifar10 \\
    --epochs 100 \\
    --batch_size {int(best_params['batch_size'])} \\
    --learning_rate {float(best_params['learning_rate']):.6f} \\
    --beta {float(best_params['weight_recon']):.4f} \\
    --adversarial_weight {float(best_params['weight_adversarial']):.4f} \\
    --latent_dim {int(best_params['latent_dim'])} \\
    --device mps"""
    return cmd

def main():
    """Run Bayesian optimization"""
    # Create output directory for results
    os.makedirs("optimization_results", exist_ok=True)
    
    print("Starting Bayesian optimization for BiGAN on CIFAR-10...")
    print("\nSearch space:")
    for dim in space:
        print(f"- {dim.name}: {dim.bounds} ({type(dim).__name__})")
    
    # Run optimization with minimal iterations
    n_calls = 5  # Use only 5 iterations for quick results
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=2,
        noise=0.1,
        verbose=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimization_results/bigan_opt_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    optimization_results = {
        'best_parameters': {
            dim.name: float(value) if isinstance(value, (np.floating, float)) else int(value)
            for dim, value in zip(space, result.x)
        },
        'best_score': float(result.fun),
        'all_trials': [
            {
                'parameters': {
                    dim.name: float(value) if isinstance(value, (np.floating, float)) else int(value)
                    for dim, value in zip(space, x)
                },
                'score': float(y)
            }
            for x, y in zip(result.x_iters, result.func_vals)
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"\nResults saved to: {results_file}")
    print("\nBest parameters found:")
    for param, value in optimization_results['best_parameters'].items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {optimization_results['best_score']}")
    
    print("\n" + "="*50)
    print("OPTIMAL TRAINING COMMAND")
    print("="*50)
    cmd = format_optimal_command(optimization_results['best_parameters'])
    print("\nRun this command to train with the optimal parameters:")
    print(cmd)
    
    # Also save the command to a file for easy access
    cmd_file = f"optimization_results/optimal_command_{timestamp}.txt"
    with open(cmd_file, 'w') as f:
        f.write(cmd)
    print(f"\nCommand also saved to: {cmd_file}")

if __name__ == "__main__":
    main() 