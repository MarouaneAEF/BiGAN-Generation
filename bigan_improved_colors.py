#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BiGAN avec reconstruction fidèle des couleurs pour CIFAR-10
Version spécialement conçue pour maximiser la fidélité des couleurs
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, constraints
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import cifar10
import time
from tqdm import tqdm

# Constantes pour le modèle
IMG_SHAPE = (32, 32, 3)
LATENT_DIM = 128  # Dimension plus grande pour mieux capturer les détails
BATCH_SIZE = 64
EPOCHS = 50
SAVE_INTERVAL = 5
LEARNING_RATE = 0.0001

# Poids des différentes pertes
WEIGHT_RECON = 10.0      # Poids élevé pour la reconstruction
WEIGHT_COLOR = 5.0       # Poids spécifique pour la conservation des couleurs
WEIGHT_ADVERSARIAL = 1.0 # Poids standard pour l'adversarial

class ColorFidelityBiGAN:
    """
    BiGAN optimisé pour la fidélité des couleurs et la qualité de reconstruction
    """
    
    def __init__(self):
        """Initialisation du modèle"""
        # Fixer les graines aléatoires pour la reproductibilité
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Initialisation des poids
        self.kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # Optimiseurs avec clipping de gradient
        optimizer_params = {
            'learning_rate': LEARNING_RATE,
            'beta_1': 0.5,
            'clipvalue': 0.01
        }
        
        self.d_optimizer = optimizers.Adam(**optimizer_params)
        self.g_optimizer = optimizers.Adam(**optimizer_params)
        self.e_optimizer = optimizers.Adam(**optimizer_params)
        
        # Construction des composants
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compilation du discriminateur
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.d_optimizer,
            metrics=['accuracy']
        )
        
        # Construction des modèles
        self.build_adversarial_model()
        self.build_reconstruction_model()
        self.build_color_fidelity_model()
        
    def build_encoder(self):
        """Construction de l'encodeur avec préservation des informations couleur"""
        inputs = layers.Input(shape=IMG_SHAPE)
        
        # Traitement initial sans réduction de résolution pour préserver les détails
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=self.kernel_init,
                        kernel_constraint=constraints.MaxNorm(3))(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        # Diminution progressive de la résolution
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
        
        # Couche globale pour retenir les caractéristiques importantes
        x = layers.GlobalAveragePooling2D()(x)
        
        # Connexion résiduelle pour les caractéristiques de couleur
        color_features = layers.Conv2D(3, kernel_size=1, strides=1, padding='same')(inputs)
        color_features = layers.GlobalAveragePooling2D()(color_features)
        
        # Aplatissement et concaténation avec les caractéristiques de couleur
        x = layers.Concatenate()([x, color_features])
        
        # Vecteur latent final
        z = layers.Dense(LATENT_DIM, 
                      kernel_initializer=self.kernel_init,
                      kernel_constraint=constraints.MaxNorm(3))(x)
        
        model = models.Model(inputs, z, name="encoder")
        print("Architecture de l'encodeur:")
        model.summary()
        
        return model
    
    def build_generator(self):
        """Construction du générateur optimisé pour la fidélité des couleurs"""
        z_input = layers.Input(shape=(LATENT_DIM,))
        
        # Couche dense initiale
        x = layers.Dense(4 * 4 * 256, 
                       kernel_initializer=self.kernel_init,
                       kernel_constraint=constraints.MaxNorm(3))(z_input)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((4, 4, 256))(x)
        
        # Upsampling progressif
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
        
        # Dernier upsampling avec plus de filtres pour meilleure qualité
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=self.kernel_init,
                                 kernel_constraint=constraints.MaxNorm(3))(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Séparation en 2 branches: structure et couleur
        # Branche structure (luminance)
        structure = layers.Conv2D(1, kernel_size=3, padding='same',
                               kernel_initializer=self.kernel_init)(x)
        
        # Branche couleur (chrominance)
        color = layers.Conv2D(3, kernel_size=3, padding='same',
                           kernel_initializer=self.kernel_init)(x)
        color = layers.BatchNormalization(momentum=0.9)(color)
        color = layers.LeakyReLU(0.2)(color)
        color = layers.Conv2D(3, kernel_size=1, padding='same',
                           kernel_initializer=self.kernel_init)(color)
        
        # Fusion des branches
        combined = layers.Concatenate()([structure, color])
        
        # Couche finale avec attention sur les canaux pour préserver les couleurs
        x = layers.Conv2D(8, kernel_size=3, padding='same')(combined)
        x = layers.LeakyReLU(0.2)(x)
        
        # Couche finale avec activation tanh
        outputs = layers.Conv2D(3, kernel_size=3, padding='same',
                              kernel_initializer=self.kernel_init)(x)
        outputs = layers.Activation('tanh')(outputs)
        
        model = models.Model(z_input, outputs, name="generator")
        print("Architecture du générateur:")
        model.summary()
        
        return model
    
    def build_discriminator(self):
        """Construction du discriminateur BiGAN standard"""
        img_input = layers.Input(shape=IMG_SHAPE)
        z_input = layers.Input(shape=(LATENT_DIM,))
        
        # Traitement de l'image
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
        
        # Traitement du vecteur latent
        z = layers.Dense(512, 
                      kernel_initializer=self.kernel_init,
                      kernel_constraint=constraints.MaxNorm(3))(z_input)
        z = layers.LeakyReLU(0.2)(z)
        
        # Concaténation
        concat = layers.Concatenate()([x, z])
        
        # Couches finales
        concat = layers.Dense(512, 
                           kernel_initializer=self.kernel_init,
                           kernel_constraint=constraints.MaxNorm(3))(concat)
        concat = layers.LeakyReLU(0.2)(concat)
        
        validity = layers.Dense(1, activation='sigmoid',
                             kernel_initializer=self.kernel_init,
                             kernel_constraint=constraints.MaxNorm(3))(concat)
        
        model = models.Model([img_input, z_input], validity, name="discriminator")
        print("Architecture du discriminateur:")
        model.summary()
        
        return model
    
    def build_adversarial_model(self):
        """Construction du modèle BiGAN adversarial standard"""
        # Pour l'entraînement G/E, figer le discriminateur
        self.discriminator.trainable = False
        
        # Flux de l'encodeur
        real_img = layers.Input(shape=IMG_SHAPE)
        latent_z = self.encoder(real_img)
        
        # Flux du générateur
        random_z = layers.Input(shape=(LATENT_DIM,))
        fake_img = self.generator(random_z)
        
        # Flux du discriminateur
        valid_real = self.discriminator([real_img, latent_z])
        valid_fake = self.discriminator([fake_img, random_z])
        
        # Modèle complet
        self.adversarial_model = models.Model(
            inputs=[real_img, random_z],
            outputs=[valid_real, valid_fake]
        )
        
        self.adversarial_model.compile(
            loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[WEIGHT_ADVERSARIAL, WEIGHT_ADVERSARIAL],
            optimizer=self.g_optimizer
        )
    
    def build_reconstruction_model(self):
        """Construction du modèle de reconstruction avec perte MSE"""
        # Les entrées sont les images réelles
        real_img = layers.Input(shape=IMG_SHAPE)
        
        # Le flux de reconstruction
        latent_z = self.encoder(real_img)
        reconstructed_img = self.generator(latent_z)
        
        # Ce modèle est utilisé pour la reconstruction
        self.reconstruction_model = models.Model(real_img, reconstructed_img)
        
        # Compilation avec une perte MSE pondérée
        self.reconstruction_model.compile(
            loss='mse',
            loss_weights=[WEIGHT_RECON],
            optimizer=self.e_optimizer
        )
    
    def build_color_fidelity_model(self):
        """Modèle spécifique pour la fidélité des couleurs"""
        # Entrée: image réelle
        real_img = layers.Input(shape=IMG_SHAPE)
        
        # Flux de reconstruction
        latent_z = self.encoder(real_img)
        reconstructed_img = self.generator(latent_z)
        
        # Ce modèle cible spécifiquement la fidélité des couleurs
        self.color_model = models.Model(real_img, reconstructed_img)
        
        # Fonction de perte personnalisée pour la fidélité des couleurs
        def color_loss(y_true, y_pred):
            # Convertir en espace de couleur plus adapté (approximation simplifiée)
            # Calculer la luminance (Y) et chrominance (U, V)
            # Y = 0.299 * R + 0.587 * G + 0.114 * B
            y_true_y = 0.299 * y_true[:,:,:,0] + 0.587 * y_true[:,:,:,1] + 0.114 * y_true[:,:,:,2]
            y_pred_y = 0.299 * y_pred[:,:,:,0] + 0.587 * y_pred[:,:,:,1] + 0.114 * y_pred[:,:,:,2]
            
            # Calculer la perte sur la chrominance (couleur) - plus importante
            color_loss = tf.reduce_mean(tf.square(
                tf.image.rgb_to_hsv(y_true) - tf.image.rgb_to_hsv(y_pred)
            ))
            
            # Calculer la perte sur la luminance (structure) - moins importante
            structure_loss = tf.reduce_mean(tf.square(y_true_y - y_pred_y))
            
            # Combiner avec une pondération favorisant la couleur
            return 0.3 * structure_loss + 0.7 * color_loss
        
        # Compilation avec la perte de couleur personnalisée
        self.color_model.compile(
            loss=color_loss,
            loss_weights=[WEIGHT_COLOR],
            optimizer=self.e_optimizer
        )

    def train(self, x_train, validation_data=None, subset_ratio=0.2):
        """Entraîne le BiGAN avec contrainte de reconstruction et fidélité des couleurs"""
        # Créer le répertoire pour les sorties
        os.makedirs("bigan_color_output", exist_ok=True)
        
        # Réduire la taille des données si demandé
        if subset_ratio < 1.0:
            dataset_size = int(len(x_train) * subset_ratio)
            indices = np.random.choice(len(x_train), dataset_size, replace=False)
            x_train = x_train[indices]
            print(f"Utilisation d'un sous-ensemble de {dataset_size} images")
        
        dataset_size = len(x_train)
        
        # Afficher les statistiques
        print(f"\nDataset: {dataset_size} images, forme: {x_train.shape[1:]}")
        print(f"Min/Max: {x_train.min():.4f}/{x_train.max():.4f}")
        print(f"Moyenne/Écart-type: {x_train.mean():.4f}/{x_train.std():.4f}")
        
        # Sauvegarder des exemples réels
        self.save_real_samples(x_train[:25])
        
        # Historiques pour suivi des performances
        d_losses = []
        g_losses = []
        r_losses = []
        c_losses = []  # Pertes de fidélité des couleurs
        d_accs = []
        
        # Boucle d'entraînement
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            # Métriques de l'époque
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_r_losses = []
            epoch_c_losses = []
            epoch_d_accs = []
            
            # Nombre d'itérations par époque
            iterations = dataset_size // BATCH_SIZE
            
            # Barre de progression
            progress_bar = tqdm(range(iterations), desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for _ in progress_bar:
                # Sélectionner un batch aléatoire
                idx = np.random.randint(0, dataset_size, BATCH_SIZE)
                real_imgs = x_train[idx]
                
                # Vecteurs latents aléatoires
                z_noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
                
                # Générer des images artificielles
                fake_imgs = self.generator.predict(z_noise, verbose=0)
                
                # Encoder les images réelles
                encoded_z = self.encoder.predict(real_imgs, verbose=0)
                
                # Labels avec label smoothing
                valid_y = np.ones((BATCH_SIZE, 1)) * 0.9  # Label smoothing
                fake_y = np.zeros((BATCH_SIZE, 1)) + 0.1  # Label smoothing
                
                #----- 1. Entraîner le discriminateur -----
                d_loss_real = self.discriminator.train_on_batch([real_imgs, encoded_z], valid_y)
                d_loss_fake = self.discriminator.train_on_batch([fake_imgs, z_noise], fake_y)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                #----- 2. Entraîner le générateur/encodeur (adversarial) -----
                g_loss = self.adversarial_model.train_on_batch(
                    [real_imgs, z_noise],
                    [fake_y, valid_y]  # Inverser pour tromper D
                )
                
                #----- 3. Entraîner la reconstruction (MSE) -----
                r_loss = self.reconstruction_model.train_on_batch(real_imgs, real_imgs)
                
                #----- 4. Entraîner la fidélité des couleurs -----
                c_loss = self.color_model.train_on_batch(real_imgs, real_imgs)
                
                # Stocker les pertes
                epoch_d_losses.append(d_loss[0])
                epoch_g_losses.append(np.mean(g_loss))
                epoch_r_losses.append(r_loss)
                epoch_c_losses.append(c_loss)
                epoch_d_accs.append(d_loss[1])
                
                # Mettre à jour la barre de progression
                progress_bar.set_postfix({
                    'D Loss': f'{d_loss[0]:.4f}',
                    'G Loss': f'{np.mean(g_loss):.4f}',
                    'R Loss': f'{r_loss:.4f}',
                    'C Loss': f'{c_loss:.4f}',
                    'D Acc': f'{d_loss[1]*100:.1f}%'
                })
            
            # Calculer les moyennes
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            avg_r_loss = np.mean(epoch_r_losses) 
            avg_c_loss = np.mean(epoch_c_losses)
            avg_d_acc = np.mean(epoch_d_accs)
            
            # Stocker dans l'historique
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)
            r_losses.append(avg_r_loss)
            c_losses.append(avg_c_loss)
            d_accs.append(avg_d_acc)
            
            # Temps écoulé
            elapsed = time.time() - start_time
            
            # Afficher un résumé
            print(f"Epoch {epoch+1}/{EPOCHS}, "
                  f"D Loss: {avg_d_loss:.4f}, "
                  f"G Loss: {avg_g_loss:.4f}, "
                  f"R Loss: {avg_r_loss:.4f}, "
                  f"C Loss: {avg_c_loss:.4f}, "
                  f"D Acc: {avg_d_acc*100:.1f}%, "
                  f"Time: {elapsed:.1f}s")
            
            # Sauvegarder périodiquement
            if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == 0:
                self.save_samples(epoch + 1)
                self.save_reconstructions(x_train[:10], epoch + 1)
                self.save_training_curves(d_losses, g_losses, r_losses, c_losses, d_accs, epoch + 1)
                self.analyze_color_fidelity(x_train[:5], epoch + 1)
        
        print("Entraînement terminé!")
    
    def save_samples(self, epoch, n_samples=25):
        """Génère et sauvegarde des échantillons aléatoires"""
        z_samples = np.random.normal(0, 1, (n_samples, LATENT_DIM))
        gen_imgs = self.generator.predict(z_samples, verbose=0)
        
        # Afficher les statistiques
        print(f"Images générées - Min/Max: {gen_imgs.min():.4f}/{gen_imgs.max():.4f}")
        print(f"Images générées - Moyenne/Écart-type: {gen_imgs.mean():.4f}/{gen_imgs.std():.4f}")
        
        # Grille d'affichage 5x5
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        
        # Convertir de [-1, 1] à [0, 1] pour l'affichage
        gen_imgs_display = (gen_imgs + 1) / 2.0
        
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs_display[cnt])
                axs[i, j].axis('off')
                cnt += 1
        
        plt.savefig(f"bigan_color_output/generated_epoch_{epoch}.png")
        plt.close()
    
    def save_real_samples(self, samples):
        """Sauvegarde des exemples réels pour comparaison"""
        n_samples = len(samples)
        n_row = int(np.sqrt(n_samples))
        n_col = n_samples // n_row
        
        fig, axs = plt.subplots(n_row, n_col, figsize=(10, 10))
        
        # Convertir en [0, 1] pour l'affichage
        display_samples = (samples + 1) / 2.0
        
        cnt = 0
        for i in range(n_row):
            for j in range(n_col):
                axs[i, j].imshow(display_samples[cnt])
                axs[i, j].axis('off')
                cnt += 1
        
        plt.savefig("bigan_color_output/real_samples.png")
        plt.close()
    
    def save_reconstructions(self, samples, epoch):
        """Sauvegarde des reconstructions pour évaluer la qualité"""
        # Encoder les échantillons
        encoded = self.encoder.predict(samples, verbose=0)
        
        # Régénérer les images
        reconstructed = self.generator.predict(encoded, verbose=0)
        
        # Préparer l'affichage
        plt.figure(figsize=(20, 4))
        
        # Convertir de [-1, 1] à [0, 1] pour l'affichage
        samples_display = (samples + 1) / 2.0
        reconstructed_display = (reconstructed + 1) / 2.0
        
        # Afficher les originaux et reconstructions
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
        
        plt.savefig(f"bigan_color_output/reconstruction_epoch_{epoch}.png")
        plt.close()
    
    def analyze_color_fidelity(self, samples, epoch):
        """Analyse spécifique de la fidélité des couleurs"""
        # Encoder et reconstruire
        encoded = self.encoder.predict(samples, verbose=0)
        reconstructed = self.generator.predict(encoded, verbose=0)
        
        plt.figure(figsize=(15, 10))
        
        for i in range(len(samples)):
            # Image originale
            plt.subplot(3, len(samples), i+1)
            plt.imshow((samples[i] + 1) / 2.0)
            plt.title("Original")
            plt.axis('off')
            
            # Image reconstruite
            plt.subplot(3, len(samples), i+len(samples)+1)
            plt.imshow((reconstructed[i] + 1) / 2.0)
            plt.title("Reconstruction")
            plt.axis('off')
            
            # Différence absolue (pour visualiser les erreurs)
            diff = np.abs(samples[i] - reconstructed[i])
            diff_normalized = diff / np.max(diff)
            
            plt.subplot(3, len(samples), i+2*len(samples)+1)
            plt.imshow(diff_normalized)
            plt.title("Différences")
            plt.axis('off')
            
            # Calculer l'erreur moyenne par canal
            error_r = np.mean(np.abs(samples[i,:,:,0] - reconstructed[i,:,:,0]))
            error_g = np.mean(np.abs(samples[i,:,:,1] - reconstructed[i,:,:,1]))
            error_b = np.mean(np.abs(samples[i,:,:,2] - reconstructed[i,:,:,2]))
            
            print(f"Image {i+1}: Erreur R={error_r:.4f}, G={error_g:.4f}, B={error_b:.4f}")
        
        plt.tight_layout()
        plt.savefig(f"bigan_color_output/color_analysis_epoch_{epoch}.png")
        plt.close()
        
        # Histogrammes de couleur
        plt.figure(figsize=(15, 5))
        
        for i in range(min(3, len(samples))):
            # Histogramme original
            plt.subplot(2, 3, i+1)
            for c, color in enumerate(['r', 'g', 'b']):
                values = samples[i,:,:,c].flatten()
                plt.hist(values, bins=50, color=color, alpha=0.5)
            plt.title(f"Original {i+1}")
            plt.xlim(-1, 1)
            
            # Histogramme reconstruit
            plt.subplot(2, 3, i+4)
            for c, color in enumerate(['r', 'g', 'b']):
                values = reconstructed[i,:,:,c].flatten()
                plt.hist(values, bins=50, color=color, alpha=0.5)
            plt.title(f"Reconstruction {i+1}")
            plt.xlim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"bigan_color_output/color_histograms_epoch_{epoch}.png")
        plt.close()
    
    def save_training_curves(self, d_losses, g_losses, r_losses, c_losses, d_accs, epoch):
        """Sauvegarde les courbes d'entraînement"""
        plt.figure(figsize=(15, 10))
        
        # Pertes adversariales
        plt.subplot(2, 2, 1)
        plt.plot(d_losses, label='Discriminateur')
        plt.plot(g_losses, label='Générateur')
        plt.title('Pertes Adversariales')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.legend()
        
        # Pertes de reconstruction et couleur
        plt.subplot(2, 2, 2)
        plt.plot(r_losses, label='MSE', color='blue')
        plt.plot(c_losses, label='Couleur', color='red')
        plt.title('Pertes de Reconstruction')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.legend()
        
        # Précision du discriminateur
        plt.subplot(2, 2, 3)
        plt.plot(np.array(d_accs) * 100)
        plt.title('Précision du Discriminateur')
        plt.xlabel('Époque')
        plt.ylabel('Précision (%)')
        plt.ylim([0, 100])
        
        # Distribution des pixels pour la dernière génération
        plt.subplot(2, 2, 4)
        z = np.random.normal(0, 1, (1, LATENT_DIM))
        img = self.generator.predict(z, verbose=0)[0]
        
        # Histogramme par canal de couleur
        for c, color in enumerate(['r', 'g', 'b']):
            values = img[:,:,c].flatten()
            plt.hist(values, bins=50, color=color, alpha=0.7)
            
        plt.title(f'Distribution des Pixels (min={img.min():.2f}, max={img.max():.2f})')
        plt.xlim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"bigan_color_output/training_curves_epoch_{epoch}.png")
        plt.close()

def load_and_preprocess_cifar10(subset_ratio=1.0):
    """Charge et prétraite le dataset CIFAR-10"""
    print("Chargement du dataset CIFAR-10...")
    (x_train, _), (x_test, _) = cifar10.load_data()
    
    # Vérifier les formes
    print(f"Forme originale: Train {x_train.shape}, Test {x_test.shape}")
    
    # Convertir en float32 et normaliser
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    
    # Vérifier la normalisation
    print(f"Après normalisation - Min/Max: {x_train.min():.4f}/{x_train.max():.4f}")
    print(f"Après normalisation - Moyenne/Écart-type: {x_train.mean():.4f}/{x_train.std():.4f}")
    
    return x_train, x_test

def main():
    """Fonction principale"""
    print("==== BiGAN avec fidélité des couleurs ====")
    
    # Vérifier la version TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    
    # Vérifier si GPU disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU détecté: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Erreur GPU: {e}")
    else:
        print("Aucun GPU détecté, utilisation du CPU.")
        print("Installation recommandée: pip install tensorflow-macos tensorflow-metal")
    
    # Charger les données
    x_train, x_test = load_and_preprocess_cifar10()
    
    # Créer et entraîner le modèle
    model = ColorFidelityBiGAN()
    
    # Entraîner avec un sous-ensemble des données pour plus de rapidité
    model.train(x_train, validation_data=x_test, subset_ratio=0.1)  # Petit sous-ensemble pour rapidité
    
    print("\n==== Entraînement terminé ====")
    print("Les résultats sont sauvegardés dans le dossier 'bigan_color_output'")
    print("Vérifiez particulièrement les fichiers 'color_analysis_*.png' qui montrent la fidélité des couleurs")

if __name__ == "__main__":
    main() 