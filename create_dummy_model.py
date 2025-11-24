# create_dummy_model.py
import torch
import json
from model_files.model import SimpleSpeakerIDModel

def create_and_save_dummy_model():
    """Crée un modèle factice et sauvegarde ses poids."""
    # Charger les labels pour connaître le nombre de sorties
    with open("model_files/speaker_labels.json", "r") as f:
        labels = json.load(f)
    num_speakers = len(labels)

    # Instancier le modèle
    dummy_model = SimpleSpeakerIDModel(num_speakers=num_speakers)
    
    # Sauvegarder les poids initiaux (aléatoires)
    model_path = "model_files/dummy_model.pth"
    torch.save(dummy_model.state_dict(), model_path)
    
    print(f"Modèle factice avec {num_speakers} sorties sauvegardé dans '{model_path}'")

if __name__ == "__main__":
    create_and_save_dummy_model()

# --- Exécutez ce fichier avec : python create_dummy_model.py ---