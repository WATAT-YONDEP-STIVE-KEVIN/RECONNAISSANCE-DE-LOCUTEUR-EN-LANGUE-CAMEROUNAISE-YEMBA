# main.py
import torch
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch.nn.functional as F

# Importer nos modules personnalisés
from model_files.model import load_model, SpeakerCNNLSTM
from model_files.preprocessor import preprocess_audio

# --- Configuration de l'application ---
app = FastAPI(
    title="VoiceID Pro API",
    description="API pour la reconnaissance de locuteurs avec un modèle PyTorch (CNNLSTM).",
    version="1.1.0"
)

# Chemins vers les fichiers
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "model_files"
MODEL_PATH = MODEL_DIR / "best_speaker_model.pth"
LABELS_PATH = MODEL_DIR / "speaker_labels.json"

# Monter le dossier static pour servir index.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Dictionnaire pour stocker le modèle et les dépendances
model_pipeline = {}

# --- Événements de démarrage et d'arrêt ---
@app.on_event("startup")
def load_application_dependencies():
    """
    Charge le modèle et les labels au démarrage de l'API.
    Ceci est exécuté une seule fois.
    """
    print("Chargement des dépendances de l'application...")
    
    if not LABELS_PATH.exists():
         raise FileNotFoundError(
            f"Fichier labels non trouvé à {LABELS_PATH}. "
            "Générez-le à partir de votre 'label_encoder.pkl'."
        )
        
    with open(LABELS_PATH, "r") as f:
        speaker_labels = json.load(f)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fichier modèle non trouvé à {MODEL_PATH}.")

    model_pipeline["labels"] = speaker_labels
    model_pipeline["model"] = load_model(MODEL_PATH, num_classes=len(speaker_labels))
    print("Dépendances chargées avec succès.")


# --- Endpoints de l'API ---
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """Sert la page principale de l'interface utilisateur."""
    try:
        with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return JSONResponse(
            status_code=404, 
            content={"message": "Fichier index.html non trouvé dans /static."}
        )

@app.post("/api/recognize", response_class=JSONResponse)
async def recognize_speaker(audio_file: UploadFile = File(...)):
    """
    Reçoit un fichier audio, le traite et retourne le locuteur identifié.
    """
    if "model" not in model_pipeline:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Type de fichier non valide. Veuillez fournir un fichier audio.")

    audio_bytes = await audio_file.read()

    try:
        # 1. Prétraiter l'audio
        input_tensor = preprocess_audio(audio_bytes)

        # 2. Faire la prédiction
        with torch.no_grad():
            model_output = model_pipeline["model"](input_tensor)

        # 3. Appliquer Softmax pour obtenir des probabilités
        probabilities = F.softmax(model_output, dim=1)
        
        # 4. Obtenir la meilleure prédiction
        confidence, predicted_index_tensor = torch.max(probabilities, 1)
        
        predicted_index = str(predicted_index_tensor.item())
        confidence_score = confidence.item()

        # 5. Mapper l'index au nom du locuteur
        speaker_name = model_pipeline["labels"].get(predicted_index, "Inconnu")
        
        return {
            "speaker_name": speaker_name,
            "confidence": round(confidence_score * 100, 2) # Retourne un pourcentage
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        raise HTTPException(status_code=500, detail="Une erreur interne est survenue lors de l'analyse.")