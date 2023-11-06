import pandas as pd
from google.cloud import translate_v2 as translate
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# Chemin vers votre fichier CSV
file_path = 'data/toxicity_cleaned_final.csv'

# Fonction pour créer un client de l'API Cloud Translation
def create_translate_client():
    return translate.Client()

# Fonction pour traduire un lot de textes
def translate_batch(text_batch, client):
    translated_texts = []
    for text in text_batch:
        if text:  # Vérifier si le texte n'est pas vide
            result = client.translate(text, target_language='fr')
            translated_texts.append(result['translatedText'])
        else:
            translated_texts.append('')
    return translated_texts

# Fonction principale pour effectuer la traduction en parallèle
def translate_in_parallel(file_path, batch_size):
    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Créer un client de l'API Cloud Translation
    client = create_translate_client()

    # Préparer les données pour le multiprocessing
    text_batches = [df['Text'][i:i+batch_size].tolist() for i in range(0, len(df), batch_size)]

    # Créer un pool de threads et effectuer la traduction en parallèle
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(translate_batch, text_batches, [client]*len(text_batches)), total=len(text_batches), desc="Translating"))

    # Concaténer les résultats
    translated_texts = [text for batch in results for text in batch]

    # Mettre à jour le dataframe avec les textes traduits
    df['TranslatedText'] = translated_texts

    # Sauvegarder le DataFrame mis à jour
    df.to_csv('toxicity_translated_to_french.csv', index=False)

# Appeler la fonction principale
def main():
    batch_size = 45  # Ajustez la taille du lot en fonction de la capacité de votre machine

    # Exécuter la fonction de traduction en parallèle
    translate_in_parallel(file_path, batch_size)

if __name__ == '__main__':
    main()
