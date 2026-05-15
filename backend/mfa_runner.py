import os
import re
import shutil
import subprocess
import soundfile as sf

from alignment import extract_word_intervals


MFA_INPUT_DIR = "mfa_runtime_input"
MFA_OUTPUT_DIR = "mfa_runtime_output"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_mfa_alignment(audio, sample_rate, expected_text, file_id):
    print("Starting MFA alignment...")

    shutil.rmtree(MFA_INPUT_DIR, ignore_errors=True)
    shutil.rmtree(MFA_OUTPUT_DIR, ignore_errors=True)

    os.makedirs(MFA_INPUT_DIR, exist_ok=True)
    os.makedirs(MFA_OUTPUT_DIR, exist_ok=True)

    wav_path = os.path.join(MFA_INPUT_DIR, f"{file_id}.wav")
    txt_path = os.path.join(MFA_INPUT_DIR, f"{file_id}.txt")

    sf.write(wav_path, audio, sample_rate)

    with open(txt_path, "w") as f:
        f.write(clean_text(expected_text))

    subprocess.run(
        [
            "conda", "run", "-n", "aligner",
            "mfa", "align",
            MFA_INPUT_DIR,
            "english_us_arpa",
            "english_us_arpa",
            MFA_OUTPUT_DIR,
            "--clean"
        ],
        check=True
    )

    textgrid_path = os.path.join(MFA_OUTPUT_DIR, f"{file_id}.TextGrid")

    print("MFA finished:", textgrid_path)

    return extract_word_intervals(textgrid_path)