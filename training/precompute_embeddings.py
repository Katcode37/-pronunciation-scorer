import os
import json
import torch
from backend.preprocess import preprocess_audio
from backend.model import PronunciationScorer

SCORES_PATH = "dataset/scores.json"
AUDIO_DIR = "dataset/audio"
SAVE_PATH = "dataset/embeddings.pt"

model = PronunciationScorer()
model.eval()

embeddings = []
scores = []

with open(SCORES_PATH, "r") as f:
    score_data = json.load(f)

for idx, (file_id, data) in enumerate(score_data.items()):
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    if not os.path.exists(audio_path):
        continue

    audio, _ = preprocess_audio(audio_path)

    with torch.no_grad():
        _, embedding = model(audio)

    embeddings.append(embedding.squeeze(0))
    scores.append(float(data["total"]))

    if idx % 50 == 0:
        print(f"Processed {idx} files")

embeddings = torch.stack(embeddings)
scores = torch.tensor(scores, dtype=torch.float32)

torch.save({
    "embeddings": embeddings,
    "scores": scores
}, SAVE_PATH)

print(f"Saved embeddings to {SAVE_PATH}")
print("Embeddings shape:", embeddings.shape)
print("Scores shape:", scores.shape)