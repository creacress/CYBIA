
```markdown
# Projet d'Analyse de Toxicité avec CamemBERT

Ce projet vise à entraîner un modèle de classification de séquences (CamemBERT) pour identifier les commentaires toxiques en français. Les commentaires sont classés en deux catégories : toxiques (1) et non toxiques (0).

## Prérequis

- Python 3.8+
- pip (Python package installer)

## Installation

Clonez le dépôt sur votre machine locale :

```bash
git clone <https://github.com/creacress/CYBIA.git>
```

Installez les dépendances requises en utilisant pip :

```bash
pip install -r requirements.txt
```

Les dépendances principales sont les suivantes :

- `transformers` pour les modèles pré-entraînés et les tokeniseurs.
- `datasets` pour le chargement et la manipulation des ensembles de données.
- `sklearn` pour les fonctions de séparation des données d'entraînement et de test et les métriques d'évaluation.
- `pandas` pour la manipulation et l'analyse des données.
- `numpy` pour les calculs numériques.

## Structure du Projet

- `data/`: Dossier contenant le dataset `toxicity_french.csv`.
- `models/`: Dossier où le modèle entraîné sera sauvegardé.
- `train.py`: Script d'entraînement du modèle.
- `requirements.txt`: Fichier contenant toutes les dépendances à installer.

## Utilisation

Pour entraîner le modèle, exécutez le script `train.py` :

```bash
python train.py
```

Le script effectuera les étapes suivantes :

1. Charger les données depuis `data/toxicity_french.csv`.
2. Tokeniser les commentaires en utilisant `CamembertTokenizer`.
3. Séparer les données en ensembles d'entraînement et de validation.
4. Configurer les arguments d'entraînement avec `TrainingArguments`.
5. Initialiser `CamembertForSequenceClassification` avec le nombre correct de labels.
6. Entraîner le modèle en utilisant `Trainer`.
7. Évaluer le modèle et imprimer les métriques de performance.
8. Sauvegarder le modèle entraîné dans `models/`.



## Contact

alexis@webcresson
```
