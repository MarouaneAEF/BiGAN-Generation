#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BiGAN implementation for image reconstruction and anomaly detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, constraints
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import cifar10
import time
from tqdm import tqdm

# Model constants
IMG_SHAPE = (32, 32, 3)
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 200
SAVE_INTERVAL = 5
LEARNING_RATE = 0.0001

# Loss weights
WEIGHT_RECON = 10.0      # Weight for reconstruction loss
WEIGHT_ADVERSARIAL = 1.0 # Weight for adversarial loss

class BiGAN:
    """
    BiGAN implementation for image reconstruction and anomaly detection
    """
    
    def __init__(self):
        """Initialize the model"""
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Weight initialization
        self.kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # Optimizers with gradient clipping
        optimizer_params = {
            'learning_rate': LEARNING_RATE,
            'beta_1': 0.5,
            'clipvalue': 0.01
        }
        
        self.d_optimizer = optimizers.Adam(**optimizer_params)
        self.g_optimizer = optimizers.Adam(**optimizer_params)
        self.e_optimizer = optimizers.Adam(**optimizer_params)
        
        # Build components
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile discriminator
        def mse_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
            
        self.discriminator.compile(
            loss=mse_loss,  # Custom MSE loss implementation
            optimizer=self.d_optimizer,
            metrics=['accuracy']
        )
        
        # Build models
        self.build_adversarial_model()
        self.build_reconstruction_model()
        
    def build_encoder(self):
        """Build the encoder"""
        inputs = layers.Input(shape=IMG_SHAPE)
        
        # Initial processing without reducing resolution to preserve details
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        # Progressive resolution reduction
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Global pooling to retain important features
        x = layers.GlobalAveragePooling2D()(x)
        
        # Final latent vector
        z = layers.Dense(LATENT_DIM, 
                      kernel_initializer=self.kernel_init,
                      kernel_constraint=constraints.MaxNorm(3))(x)
        
        model = models.Model(inputs, z, name="encoder")
        print("Encoder architecture:")
        model.summary()
        
        return model
    
    def build_generator(self):
        """Build the generator"""
        z_input = layers.Input(shape=(LATENT_DIM,))
        
        # Initial dense layer
        x = layers.Dense(4 * 4 * 256, 
                       kernel_initializer=self.kernel_init,
                       kernel_constraint=constraints.MaxNorm(3))(z_input)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((4, 4, 256))(x)
        
        # Progressive upsampling
        x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=self.kernel_init,
                                 kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=self.kernel_init,
                                 kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final upsampling with more filters for better quality
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=self.kernel_init,
                                 kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final output layer
        x = layers.Conv2D(32, kernel_size=3, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Output with tanh activation
        outputs = layers.Conv2D(3, kernel_size=3, padding='same',
                              kernel_initializer=self.kernel_init)(x)
        outputs = layers.Activation('tanh')(outputs)
        
        model = models.Model(z_input, outputs, name="generator")
        print("Generator architecture:")
        model.summary()
        
        return model
    
    def build_discriminator(self):
        """Build the standard BiGAN discriminator"""
        img_input = layers.Input(shape=IMG_SHAPE)
        z_input = layers.Input(shape=(LATENT_DIM,))
        
        # Image processing
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(img_input)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(512, 
                      kernel_initializer=self.kernel_init,
                      kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Latent vector processing
        z = layers.Dense(512, 
                      kernel_initializer=self.kernel_init,
                      kernel_constraint=constraints.MaxNorm(3))(z_input)
        z = layers.LeakyReLU(0.2)(z)
        
        # Concatenation
        concat = layers.Concatenate()([x, z])
        
        # Final layers
        concat = layers.Dense(512, 
                           kernel_initializer=self.kernel_init,
                           kernel_constraint=constraints.MaxNorm(3))(concat)
        concat = layers.LeakyReLU(0.2)(concat)
        
        validity = layers.Dense(1, activation='sigmoid',
                             kernel_initializer=self.kernel_init,
                             kernel_constraint=constraints.MaxNorm(3))(concat)
        
        model = models.Model([img_input, z_input], validity, name="discriminator")
        print("Discriminator architecture:")
        model.summary()
        
        return model
    
    def build_adversarial_model(self):
        """Build the standard BiGAN adversarial model"""
        # Freeze the discriminator for G/E training
        self.discriminator.trainable = False
        
        # Encoder flow
        real_img = layers.Input(shape=IMG_SHAPE)
        latent_z = self.encoder(real_img)
        
        # Generator flow
        random_z = layers.Input(shape=(LATENT_DIM,))
        fake_img = self.generator(random_z)
        
        # Discriminator flow
        valid_real = self.discriminator([real_img, latent_z])
        valid_fake = self.discriminator([fake_img, random_z])
        
        # Define custom MSE loss
        def mse_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Complete model
        self.adversarial_model = models.Model(
            inputs=[real_img, random_z],
            outputs=[valid_real, valid_fake]
        )
        
        self.adversarial_model.compile(
            loss=[mse_loss, mse_loss],  # Custom MSE loss implementation
            loss_weights=[WEIGHT_ADVERSARIAL, WEIGHT_ADVERSARIAL],
            optimizer=self.g_optimizer
        )
        
        # Restore discriminator's trainable state
        self.discriminator.trainable = True
    
    def build_reconstruction_model(self):
        """Build the reconstruction model with MSE loss"""
        # Inputs are real images
        real_img = layers.Input(shape=IMG_SHAPE)
        
        # Ensure encoder and generator are trainable
        self.encoder.trainable = True
        self.generator.trainable = True
        
        # Reconstruction flow
        latent_z = self.encoder(real_img)
        reconstructed_img = self.generator(latent_z)
        
        # Define custom MSE loss
        def mse_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
            
        # This model is used for reconstruction
        self.reconstruction_model = models.Model(real_img, reconstructed_img)
        
        # Compile with weighted MSE loss
        self.reconstruction_model.compile(
            loss=mse_loss,  # Custom MSE loss implementation
            loss_weights=[WEIGHT_RECON],
            optimizer=self.e_optimizer
        )

    def train(self, x_train, validation_data=None, subset_ratio=0.2):
        """Train the BiGAN"""
        # Create output directory
        os.makedirs("bigan_output", exist_ok=True)
        
        # Check for trainable weights
        models_to_check = {
            "Generator": self.generator,
            "Encoder": self.encoder,
            "Discriminator": self.discriminator,
            "Adversarial Model": self.adversarial_model,
            "Reconstruction Model": self.reconstruction_model
        }
        
        for name, model in models_to_check.items():
            trainable_weights = len(model.trainable_weights)
            total_weights = len(model.weights)
            print(f"{name}: {trainable_weights} trainable weights out of {total_weights} total")
            
            if trainable_weights == 0 and total_weights > 0:
                print(f"WARNING: {name} has no trainable weights!")
        
        # Reduce dataset size if requested
        if subset_ratio < 1.0:
            dataset_size = int(len(x_train) * subset_ratio)
            indices = np.random.choice(len(x_train), dataset_size, replace=False)
            x_train = x_train[indices]
            print(f"Using a subset of {dataset_size} images")
        
        dataset_size = len(x_train)
        
        # Display statistics
        print(f"\nDataset: {dataset_size} images, shape: {x_train.shape[1:]}")
        print(f"Min/Max: {x_train.min():.4f}/{x_train.max():.4f}")
        print(f"Mean/Std: {x_train.mean():.4f}/{x_train.std():.4f}")
        
        # Save real samples
        self.save_real_samples(x_train[:25])
        
        # Performance tracking histories
        d_losses = []
        g_losses = []
        r_losses = []
        d_accs = []
        
        # Define training functions with tf.function to avoid retracing
        @tf.function(reduce_retracing=True)
        def train_discriminator(real_imgs, encoded_z, fake_imgs, z_noise, valid_y, fake_y):
            """Optimized function for discriminator training"""
            with tf.GradientTape() as tape:
                # Discriminator predictions
                pred_real = self.discriminator([real_imgs, encoded_z], training=True)
                pred_fake = self.discriminator([fake_imgs, z_noise], training=True)
                
                # Loss calculations - using MSE for Gaussian distribution
                loss_real = tf.reduce_mean(tf.square(valid_y - pred_real))  # MSE implementation
                loss_fake = tf.reduce_mean(tf.square(fake_y - pred_fake))  # MSE implementation
                
                # Calculate mean losses per batch
                loss_real_mean = loss_real  # Already a mean
                loss_fake_mean = loss_fake  # Already a mean
                loss = 0.5 * (loss_real_mean + loss_fake_mean)
                
            # Apply gradients
            grads = tape.gradient(loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            
            # Calculate mean accuracy
            accuracy = tf.reduce_mean(tf.cast(pred_real > 0.5, tf.float32))
            
            return loss, accuracy
        
        @tf.function(reduce_retracing=True)
        def train_generator(real_imgs, z_noise, valid_y, fake_y):
            """Optimized function for generator training"""
            # Ensure discriminator is not trained
            self.discriminator.trainable = False
            
            with tf.GradientTape() as tape:
                # Encoder flow
                latent_z = self.encoder(real_imgs, training=True)
                
                # Generator flow
                fake_img = self.generator(z_noise, training=True)
                
                # Discriminator flow (no training for discriminator)
                valid_real = self.discriminator([real_imgs, latent_z], training=False)
                valid_fake = self.discriminator([fake_img, z_noise], training=False)
                
                # Invert labels to fool the discriminator
                # Using MSE for Gaussian distribution
                loss_real = tf.reduce_mean(tf.square(fake_y - valid_real))  # MSE implementation
                loss_fake = tf.reduce_mean(tf.square(valid_y - valid_fake))  # MSE implementation
                
                # Calculate means for each loss
                loss_real_mean = loss_real  # Already a mean
                loss_fake_mean = loss_fake  # Already a mean
                
                # Total loss (ensuring it's a scalar)
                g_loss = 0.5 * (loss_real_mean + loss_fake_mean) * WEIGHT_ADVERSARIAL
            
            # Get trainable variables of generator and encoder
            trainable_vars = self.generator.trainable_weights + self.encoder.trainable_weights
            
            # Calculate and apply gradients
            gradients = tape.gradient(g_loss, trainable_vars)
            self.g_optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Restore discriminator's trainable state
            self.discriminator.trainable = True
            
            return g_loss
        
        @tf.function(reduce_retracing=True)
        def train_reconstruction(real_imgs):
            """Optimized function for reconstruction training"""
            with tf.GradientTape() as tape:
                # Reconstruction flow
                latent_z = self.encoder(real_imgs, training=True)
                reconstructed_img = self.generator(latent_z, training=True)
                
                # MSE loss calculation
                mse_loss = tf.reduce_mean(tf.square(real_imgs - reconstructed_img)) * WEIGHT_RECON
            
            # Trainable variables
            trainable_vars = self.generator.trainable_weights + self.encoder.trainable_weights
            
            # Calculate and apply gradients
            gradients = tape.gradient(mse_loss, trainable_vars)
            self.e_optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            return mse_loss
        
        # Training loop
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # Epoch metrics
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_r_losses = []
            epoch_d_accs = []
            
            # Number of iterations per epoch
            iterations = dataset_size // BATCH_SIZE
            
            # Progress bar
            progress_bar = tqdm(range(iterations), desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for _ in progress_bar:
                # Select random batch
                idx = np.random.randint(0, dataset_size, BATCH_SIZE)
                real_imgs = x_train[idx]
                
                # Random latent vectors
                z_noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
                
                # Generate fake images
                fake_imgs = self.generator.predict(z_noise, verbose=0)
                
                # Encode real images
                encoded_z = self.encoder.predict(real_imgs, verbose=0)
                
                # Labels with label smoothing
                valid_y = np.ones((BATCH_SIZE, 1)) * 0.9  # Label smoothing
                fake_y = np.zeros((BATCH_SIZE, 1)) + 0.1  # Label smoothing
                
                # Convert to tensors to avoid retracing
                real_imgs_tensor = tf.convert_to_tensor(real_imgs, dtype=tf.float32)
                fake_imgs_tensor = tf.convert_to_tensor(fake_imgs, dtype=tf.float32)
                encoded_z_tensor = tf.convert_to_tensor(encoded_z, dtype=tf.float32)
                z_noise_tensor = tf.convert_to_tensor(z_noise, dtype=tf.float32)
                valid_y_tensor = tf.convert_to_tensor(valid_y, dtype=tf.float32)
                fake_y_tensor = tf.convert_to_tensor(fake_y, dtype=tf.float32)
                
                #----- 1. Train discriminator -----
                d_result = train_discriminator(
                    real_imgs_tensor, 
                    encoded_z_tensor, 
                    fake_imgs_tensor, 
                    z_noise_tensor, 
                    valid_y_tensor, 
                    fake_y_tensor
                )
                # Extract scalar values from tensors
                try:
                    d_loss_tensor = d_result[0]
                    d_acc_tensor = d_result[1]
                    # Check if tensors are size 1 and convert them
                    d_loss_value = float(d_loss_tensor.numpy())
                    d_acc_value = float(d_acc_tensor.numpy())
                except Exception as e:
                    print(f"Error converting discriminator tensors: {e}")
                    print(f"Types: {type(d_result[0])}, {type(d_result[1])}")
                    if hasattr(d_result[0], 'shape'):
                        print(f"Shapes: {d_result[0].shape}, {d_result[1].shape}")
                    # Use default value in case of error
                    d_loss_value = 0.0
                    d_acc_value = 0.5
                
                d_loss = [d_loss_value, d_acc_value]
                
                #----- 2. Train generator/encoder (adversarial) -----
                g_loss = train_generator(
                    real_imgs_tensor, 
                    z_noise_tensor, 
                    valid_y_tensor, 
                    fake_y_tensor
                )
                # Convert generator result
                try:
                    g_loss_value = float(g_loss.numpy())
                except Exception as e:
                    print(f"Error converting generator tensor: {e}")
                    print(f"Type: {type(g_loss)}")
                    if hasattr(g_loss, 'shape'):
                        print(f"Shape: {g_loss.shape}")
                    # Use default value
                    g_loss_value = 0.0
                
                #----- 3. Train reconstruction (MSE) -----
                r_loss = train_reconstruction(real_imgs_tensor)
                # Convert reconstruction result
                try:
                    r_loss_value = float(r_loss.numpy())
                except Exception as e:
                    print(f"Error converting reconstruction tensor: {e}")
                    print(f"Type: {type(r_loss)}")
                    if hasattr(r_loss, 'shape'):
                        print(f"Shape: {r_loss.shape}")
                    # Use default value
                    r_loss_value = 0.0
                
                # Store losses
                epoch_d_losses.append(d_loss[0])
                epoch_g_losses.append(g_loss_value)
                epoch_r_losses.append(r_loss_value)
                epoch_d_accs.append(d_loss[1])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'D Loss': f'{d_loss[0]:.4f}',
                    'G Loss': f'{g_loss_value:.4f}',
                    'R Loss': f'{r_loss_value:.4f}',
                    'D Acc': f'{d_loss[1]*100:.1f}%'
                })
            
            # Calculate averages
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            avg_r_loss = np.mean(epoch_r_losses)
            avg_d_acc = np.mean(epoch_d_accs)
            
            # Store in history
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)
            r_losses.append(avg_r_loss)
            d_accs.append(avg_d_acc)
            
            # Elapsed time
            elapsed = time.time() - start_time
            
            # Display summary
            print(f"Epoch {epoch+1}/{EPOCHS}, "
                  f"D Loss: {avg_d_loss:.4f}, "
                  f"G Loss: {avg_g_loss:.4f}, "
                  f"R Loss: {avg_r_loss:.4f}, "
                  f"D Acc: {avg_d_acc*100:.1f}%, "
                  f"Time: {elapsed:.1f}s")
            
            # Save periodically
            if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == 0:
                self.save_samples(epoch + 1)
                self.save_reconstructions(x_train[:10], epoch + 1)
        
        print("Training completed!")
    
    def save_samples(self, epoch, n_samples=25):
        """Generate and save random samples"""
        z_samples = np.random.normal(0, 1, (n_samples, LATENT_DIM))
        gen_imgs = self.generator.predict(z_samples, verbose=0)
        
        # Display statistics
        print(f"Generated images - Min/Max: {gen_imgs.min():.4f}/{gen_imgs.max():.4f}")
        print(f"Generated images - Mean/Std: {gen_imgs.mean():.4f}/{gen_imgs.std():.4f}")
        
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
        
        plt.savefig(f"bigan_output/generated_epoch_{epoch}.png")
        plt.close()
    
    def save_real_samples(self, samples):
        """Save real samples for comparison"""
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
        
        plt.savefig("bigan_output/real_samples.png")
        plt.close()
    
    def save_reconstructions(self, samples, epoch):
        """Save reconstructions to evaluate quality"""
        # Encode samples
        encoded = self.encoder.predict(samples, verbose=0)
        
        # Regenerate images
        reconstructed = self.generator.predict(encoded, verbose=0)
        
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
        
        plt.savefig(f"bigan_output/reconstruction_epoch_{epoch}.png")
        plt.close()

def load_and_preprocess_cifar10(subset_ratio=1.0):
    """Load and preprocess CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    (x_train, _), (x_test, _) = cifar10.load_data()
    
    # Check shapes
    print(f"Original shape: Train {x_train.shape}, Test {x_test.shape}")
    
    # Convert to float32 and normalize
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    
    # Check normalization
    print(f"After normalization - Min/Max: {x_train.min():.4f}/{x_train.max():.4f}")
    print(f"After normalization - Mean/Std: {x_train.mean():.4f}/{x_train.std():.4f}")
    
    return x_train, x_test

def main():
    """Main function"""
    print("==== BiGAN for image reconstruction and anomaly detection ====")
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("No GPU detected, using CPU.")
        print("Recommended installation: pip install tensorflow-macos tensorflow-metal")
    
    # Load data
    x_train, x_test = load_and_preprocess_cifar10()
    
    # Create and train model
    model = BiGAN()
    
    # Train with a subset of data for faster execution
    model.train(x_train, validation_data=x_test, subset_ratio=0.1)
    
    print("\n==== Training completed ====")
    print("Results are saved in the 'bigan_output' folder")

if __name__ == "__main__":
    main() 