from transformers import pipeline
from jiwer import wer
import re


stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en"
)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe_audio(audio, sample_rate):
    result = stt_pipeline({
        "array": audio,
        "sampling_rate": sample_rate
    })

    return result["text"]


def compare_with_expected(expected_text, transcribed_text):
    expected_clean = normalize_text(expected_text)
    transcribed_clean = normalize_text(transcribed_text)

    error_rate = wer(expected_clean, transcribed_clean)

    match_score = max(0, 100 * (1 - error_rate))

    return round(match_score, 2)