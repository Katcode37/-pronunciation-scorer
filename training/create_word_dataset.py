import os
import json
import re

SCORES_PATH = "dataset/scores.json"
AUDIO_DIR = "dataset/audio"


def normalize_word(word):
    word = word.lower()
    word = re.sub(r"[^a-z0-9]", "", word)
    return word


with open(SCORES_PATH, "r") as f:
    scores = json.load(f)


matched = 0
skipped = 0

for file_id, data in scores.items():
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    if not os.path.exists(audio_path):
        skipped += 1
        continue

    if "words" not in data:
        skipped += 1
        continue

    words = data["words"]

    for word_item in words:
        word_text = normalize_word(word_item["text"])
        word_score = float(word_item["total"])

        if word_text == "":
            continue

        matched += 1

print("Valid word-level samples found:", matched)
print("Skipped utterances/items:", skipped)