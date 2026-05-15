import os
import json
import re
import torch

from backend.preprocess import preprocess_audio
from backend.alignment import extract_word_intervals, extract_word_audio_segments
from backend.model import PronunciationScorer


SCORES_PATH = "dataset/scores.json"
AUDIO_DIR = "dataset/audio"
TEXTGRID_DIR = "mfa_subset_aligned"
SAVE_PATH = "dataset/word_embeddings_subset.pt"


def normalize_word(word):
    word = word.lower()
    word = re.sub(r"[^a-z0-9]", "", word)
    return word


def get_textgrid_path(file_id):
    return os.path.join(TEXTGRID_DIR, f"{file_id}.TextGrid")


model = PronunciationScorer()
model.eval()

with open(SCORES_PATH, "r") as f:
    scores_data = json.load(f)

embeddings = []
word_scores = []
word_texts = []
file_ids = []

processed_words = 0
skipped_words = 0
processed_files = 0

for file_id, data in scores_data.items():
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")
    textgrid_path = get_textgrid_path(file_id)

    if not os.path.exists(audio_path):
        continue

    if not os.path.exists(textgrid_path):
        continue

    if "words" not in data:
        continue

    audio, sample_rate = preprocess_audio(audio_path)

    intervals = extract_word_intervals(textgrid_path)
    segments = extract_word_audio_segments(audio, sample_rate, intervals)

    json_words = data["words"]

    # Match by position: aligned word 1 ↔ JSON word 1
    min_len = min(len(segments), len(json_words))

    for i in range(min_len):
        aligned_word = normalize_word(segments[i]["word"])
        json_word = normalize_word(json_words[i]["text"])

        if aligned_word != json_word:
            skipped_words += 1
            continue

        word_audio = segments[i]["audio"]
        word_score = float(json_words[i]["total"])
        if len(word_audio) < 3200:
            skipped_words += 1
            continue

        with torch.no_grad():
            _, embedding = model(word_audio)

        embeddings.append(embedding.squeeze(0))
        word_scores.append(word_score)
        word_texts.append(json_word)
        file_ids.append(file_id)

        processed_words += 1

    processed_files += 1

    if processed_files % 5 == 0:
        print(
            f"Processed files: {processed_files}, "
            f"word samples: {processed_words}, "
            f"skipped words: {skipped_words}"
        )


embeddings = torch.stack(embeddings)
word_scores = torch.tensor(word_scores, dtype=torch.float32)

torch.save({
    "embeddings": embeddings,
    "scores": word_scores,
    "words": word_texts,
    "file_ids": file_ids
}, SAVE_PATH)

print(f"Saved word embeddings to {SAVE_PATH}")
print("Embeddings shape:", embeddings.shape)
print("Scores shape:", word_scores.shape)
print("Words saved:", len(word_texts))
print("Skipped words:", skipped_words)