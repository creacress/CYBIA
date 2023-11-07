import logging
import sys
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (CamembertForSequenceClassification, CamembertTokenizer,
                          Trainer, TrainingArguments, TrainerCallback)
import torch
import gc

# Configurer le logging pour afficher les informations d'entraînement
logging.basicConfig(level=logging.INFO)

# Fonction pour gérer l'interruption et sauvegarder le modèle
def save_on_interrupt(signal_received, frame):
    logging.info('Interruption reçue: Sauvegarde du modèle en cours...')
    trainer.save_model('toxicity_model_on_interrupt')
    sys.exit(0)

# Configuration du gestionnaire de signaux pour intercepter Ctrl+C
import signal
signal.signal(signal.SIGINT, save_on_interrupt)

# Fonction pour calculer les métriques de performance
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Charger les données
df = pd.read_csv('data/toxicity_french_upsampled.csv', sep=';')
df = df[['Texte', 'oh_label']]

# Déterminer la longueur maximale pour le tokenizer
max_length = min(df['Texte'].str.len().quantile(0.95), 512)

# Split des données en train et validation sets
train_df, val_df = train_test_split(df, stratify=df['oh_label'], test_size=0.1)

# Initialisation du tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Fonction pour tokeniser les données
def tokenize_function(examples):
    return tokenizer(examples["Texte"], padding="max_length", truncation=True, max_length=max_length)

# Création des datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("oh_label", "labels")
val_dataset = val_dataset.rename_column("oh_label", "labels")

# Configuration de l'entraînement avec le Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=20,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    gradient_accumulation_steps=4,
    use_cpu=False
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

# Callback pour logging personnalisé
class LoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0:
            logging.info(f"Étape {state.global_step}: Entraînement en cours...")

# Entraînement avec gestion des interruptions
try:
    trainer.train(resume_from_checkpoint=True)
except KeyboardInterrupt:
    logging.info("Interruption manuelle détectée pendant l'entraînement.")

# Évaluation et sauvegarde du modèle
eval_results = trainer.evaluate()
logging.info(eval_results)
trainer.save_model('toxicity_model')

# Nettoyage de la mémoire si nécessaire
gc.collect()
if not training_args.use_cpu:
    torch.cuda.empty_cache()
