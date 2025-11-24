# model_files/preprocessor.py
import torch
import torchaudio
import io

# --- À ADAPTER SELON VOTRE MODÈLE ---
# Remplacez cette fonction par votre propre pipeline de prétraitement.
# Les paramètres (n_mels, n_fft, etc.) doivent correspondre à ceux utilisés
# lors de l'entraînement de votre modèle.

TARGET_SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160

def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """
    Charge les bytes d'un fichier audio, le prétraite et le convertit en tenseur.
    
    Args:
        audio_bytes (bytes): Le contenu brut du fichier audio.

    Returns:
        torch.Tensor: Un tenseur prêt à être passé au modèle (ex: mel-spectrogram).
    """
    try:
        # 1. Charger l'audio depuis les bytes en mémoire
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        # 2. Re-échantillonner si nécessaire
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        # 3. Transformer en Mel-spectrogramme
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        mel_spectrogram = mel_spectrogram_transform(waveform)

        # 4. Convertir en échelle logarithmique (dB)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # 5. Ajouter une dimension "batch" et "channel" pour le modèle (ex: [1, 1, n_mels, time])
        log_mel_spectrogram = log_mel_spectrogram.unsqueeze(0)
        
        print(f"Prétraitement réussi. Forme du tenseur: {log_mel_spectrogram.shape}")
        return log_mel_spectrogram

    except Exception as e:
        print(f"Erreur lors du prétraitement audio: {e}")
        # Gérer les erreurs de format de fichier, etc.
        raise ValueError("Impossible de traiter le fichier audio fourni.") from e