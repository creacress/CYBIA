from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Chargement des données rééquilibrées
df = pd.read_csv('data/toxicity_french_upsampled.csv', sep=';')  # Utilisation du délimiteur ';'
df = df[['Texte', 'oh_label']]  # Conserver uniquement les colonnes nécessaires

# Analyse rapide pour déterminer la longueur maximale utile
max_length = min(df['Texte'].str.len().quantile(0.95), 512)

# Préparation des données pour l'entraînement
train_df, val_df = train_test_split(df, stratify=df['oh_label'], test_size=0.1)

# Initialisation du tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Fonction pour tokeniser les données
def tokenize_function(examples):
    return tokenizer(examples["Texte"], padding="max_length", truncation=True, max_length=max_length)

# Création des datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Application de la tokenisation
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Création des colonnes 'labels' pour les datasets
train_dataset = train_dataset.rename_column("oh_label", "labels")
val_dataset = val_dataset.rename_column("oh_label", "labels")

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=26,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="steps",
    save_steps=26,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    use_cpu=True
)

# Initialisation du modèle
model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=2)

# Initialisation du Trainer avec la fonction compute_metrics passée en argument
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Entraînement
trainer.train()

# Évaluation
eval_results = trainer.evaluate()
print(eval_results)

# Sauvegarde du modèle entraîné
trainer.save_model('toxicity_model')
