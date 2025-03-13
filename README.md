# BiGAN Implementation (Bidirectional Generative Adversarial Network)

Ce projet implémente un Bidirectional Generative Adversarial Network (BiGAN) basé sur l'article ["Adversarial Feature Learning"](https://arxiv.org/abs/1605.09782) de Donahue et al.

## Description

Le BiGAN est une extension du GAN (Generative Adversarial Network) qui ajoute un encodeur permettant de mapper les données réelles vers l'espace latent. Cette architecture bidirectionnelle permet :

1. La génération de nouvelles données (comme un GAN classique)
2. L'inférence, c'est-à-dire la projection des données réelles dans l'espace latent
3. L'apprentissage non supervisé de représentations utiles

## Structure du projet

- `bigan.py` : Implémentation principale du modèle BiGAN
- `train.py` : Script d'entraînement du modèle
- `utils.py` : Fonctions utilitaires pour le chargement et la manipulation des données
- `requirements.txt` : Liste des dépendances

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Pour entraîner le modèle :

```bash
python train.py
```

Pour générer des échantillons après entraînement :

```bash
python generate.py
```

## Architecture du BiGAN

Le BiGAN comprend trois composants principaux :
- **Générateur (G)** : Transforme un vecteur latent z en données générées G(z)
- **Encodeur (E)** : Encode les données réelles x en vecteurs latents E(x)
- **Discriminateur (D)** : Essaie de distinguer les paires (x, E(x)) des paires (G(z), z)

## Licence

MIT License 