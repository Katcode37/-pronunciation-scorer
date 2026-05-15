from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from uuid import uuid4
from preprocess import preprocess_audio
from model import predict_score
from stt import transcribe_audio, compare_with_expected
from mfa_runner import run_mfa_alignment
from alignment import extract_word_audio_segments
from word_model import predict_weakest_words

app = Flask(__name__)
CORS(app)

upload_foulder = "uploads"
os.makedirs(upload_foulder, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Pronunciation scorer backend is running"
    })


@app.route("/score", methods=["POST"])

def score_audio():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file uploaded"
        }), 400

    expected_text = request.form.get("expected_text")

    if not expected_text:
        return jsonify({
            "error": "No expected text provided"
        }), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    safe_filename = secure_filename(audio_file.filename)
    unique_filename = f"{uuid4()}_{safe_filename}"

    save_path = os.path.join(upload_foulder, unique_filename)
    audio_file.save(save_path)
    processed_audio, sample_rate = preprocess_audio(save_path)

    transcribed_text = transcribe_audio(
        processed_audio,
        sample_rate
    )

    text_match_score = compare_with_expected(
        expected_text,
        transcribed_text
    )

    predicted_score, embedding = predict_score(processed_audio)
    fake_score = round(predicted_score, 2)

    print("Running MFA...")

    word_intervals = run_mfa_alignment(
        processed_audio,
        sample_rate,
        expected_text,
        unique_filename.replace(".", "_")
    )

    word_segments = extract_word_audio_segments(
        processed_audio,
        sample_rate,
        word_intervals
    )

    weakest_words = predict_weakest_words(
        word_segments,
        top_k=3
    )

    print("Weakest words:", weakest_words)

    if text_match_score < 65:
        feedback = "Please, read the text exactly as you see on the screen."

    elif fake_score >= 70:
        feedback = "Great pronunciation!"

    elif fake_score >= 55:
        feedback = "You can do better."

    else:
        feedback = "Needs some work."
    

    return jsonify({
        "score": fake_score,
        "feedback": feedback,
        "filename": unique_filename,
        "expected_text": expected_text,
        "sample_rate": sample_rate,
        "processed_audio_length": len(processed_audio),
        "embedding_shape": list(embedding.shape),
        "transcribed_text": transcribed_text,
        "text_match_score": text_match_score,
        "weakest_words": weakest_words,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)