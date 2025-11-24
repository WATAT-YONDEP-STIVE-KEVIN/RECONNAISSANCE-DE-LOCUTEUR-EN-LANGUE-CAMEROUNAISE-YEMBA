# model_files/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ARCHITECTURE EXTRAITE DE VOTRE NOTEBOOK ---
class SpeakerCNNLSTM(nn.Module):
    """Modèle CNN+LSTM pour la reconnaissance de locuteurs"""

    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(SpeakerCNNLSTM, self).__init__()

        self.input_dim = input_dim

        # Couches CNN pour l'extraction de caractéristiques
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # Couches LSTM pour la modélisation temporelle
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Couches de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # *2 pour bidirectionnel
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extraction de caractéristiques CNN
        conv_out = self.conv_layers(x)  # (batch, channels, seq_len)

        # Préparer pour LSTM: (batch, seq_len, features)
        lstm_input = conv_out.transpose(1, 2)

        # Modélisation temporelle LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Utiliser le dernier état caché pour la classification
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concaténer les sorties bidirectionnelles

        # Classification
        output = self.classifier(final_hidden)

        return output

def load_model(model_path: str, num_classes: int) -> SpeakerCNNLSTM:
    """Charge le modèle PyTorch et ses poids."""
    # Hyperparamètres extraits de votre notebook
    N_MFCC = 40
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Instancier le modèle avec les bons paramètres
    model = SpeakerCNNLSTM(
        input_dim=N_MFCC,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    # Charger les poids sur CPU (recommandé pour l'inférence)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Mettre le modèle en mode évaluation (très important !)
    model.eval()
    
    print("Modèle PyTorch personnalisé chargé et en mode évaluation.")
    return model