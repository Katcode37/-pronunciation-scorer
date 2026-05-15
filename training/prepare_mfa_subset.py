import os
import json
import shutil
import re

SCORES_PATH = "dataset/scores.json"
AUDIO_DIR = "dataset/audio"
MFA_INPUT_DIR = "mfa_subset"
LIMIT = None



def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


os.makedirs(MFA_INPUT_DIR, exist_ok=True)

with open(SCORES_PATH, "r") as f:
    scores = json.load(f)

count = 0

for file_id, data in scores.items():
    audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")

    if not os.path.exists(audio_path):
        continue

    text = clean_text(data["text"])

    shutil.copy(audio_path, os.path.join(MFA_INPUT_DIR, f"{file_id}.wav"))

    with open(os.path.join(MFA_INPUT_DIR, f"{file_id}.txt"), "w") as f:
        f.write(text)

    count += 1

    if LIMIT is not None and count >= LIMIT:
        break

print(f"Prepared {count} files for MFA in {MFA_INPUT_DIR}")