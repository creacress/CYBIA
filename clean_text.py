import re
import pandas as pd
import unicodedata

def clean_text(text):
    # Supprimer les guillemets spéciaux et backticks
    text = re.sub(r'[`“”]', '', text)

    # Supprimer les barres obliques qui semblent servir de séparateurs
    text = re.sub(r'/', ' ', text)

    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Supprimer les balises HTML et les balises de formatage comme ==books==
    text = re.sub(r'<.*?>|={2,}.*?={2,}', '', text)
    
    # Supprimer les mentions d'utilisateurs
    text = re.sub(r'@\w+', '', text)
    
    # Supprimer les émoticons et les symboles spéciaux, tout en conservant les caractères accentués
    text = re.sub(r'[:;=()\-]{2,}', '', text)
    text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
    
    # Supprimer la ponctuation excessive
    text = re.sub(r'[\.\,]{2,}', ' ', text)
    
    # Normalisation du texte : convertir en minuscules et retirer les espaces superflus
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

try:
    # Chargement des données
    df = pd.read_csv('data/toxicity_parsed_dataset.csv')
    
    # Nettoyage du texte pour chaque entrée dans la colonne 'Text'
    df['Text'] = df['Text'].apply(clean_text)
    
    # Sauvegarde du DataFrame avec la colonne 'Text' nettoyée, remplaçant l'ancien fichier
    df.to_csv('data/toxicity_parsed_dataset.csv', index=False)
except Exception as e:
    print(f"Une erreur est survenue : {e}")