# load audio
#convert to mono
# resample to 16kHz
# normalize volume
# trim silence

import librosa
import numpy as np


def preprocess_audio(file_path):
    # Load audio, convert to mono, resample to 16kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Trim silence at beginning and end
    audio, _ = librosa.effects.trim(
        audio,
        top_db=25
    )

    # Normalize volume
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return audio, sample_rate

#Currently not denoising. Why - wavLM is trained on noisy data, so it should be robust to noise.
#  Also, denoising can sometimes remove important speech features, which could negatively impact scoring accuracy.