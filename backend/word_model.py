from pathlib import Path
import torch
import torch.nn as nn
from model import scorer_model


class WordRegressionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


BASE_DIR = Path(__file__).resolve().parent.parent
WORD_MODEL_PATH = BASE_DIR / "saved_models" / "word_regression_head.pt"

word_model = WordRegressionHead()
word_model.load_state_dict(
    torch.load(WORD_MODEL_PATH, map_location="cpu")
)
word_model.eval()

def score_word_embeddings(word_embeddings, word_texts, top_k=3):
    """
    word_embeddings: tensor shaped [num_words, 768]
    word_texts: list of words, same order as embeddings

    returns lowest-scoring words
    """

    with torch.no_grad():
        scores = word_model(word_embeddings)

    word_results = []

    for word, score in zip(word_texts, scores):
        word_results.append({
            "word": word,
            "score": round(score.item(), 2)
        })

    word_results = sorted(
        word_results,
        key=lambda item: item["score"]
    )

    weak_words = [
        item for item in word_results
        if item["score"] < 9
    ]
    return weak_words[:top_k]

def predict_weakest_words(word_segments, top_k=3):

    embeddings = []
    word_texts = []

    for segment in word_segments:
        word_audio = segment["audio"]

        # skip very short segments
        if len(word_audio) < 3200:
            continue

        with torch.no_grad():
            _, embedding = scorer_model(word_audio)

        embeddings.append(embedding.squeeze(0))
        word_texts.append(segment["word"])

    if len(embeddings) == 0:
        return []

    word_embeddings = torch.stack(embeddings)

    weakest_words = score_word_embeddings(
        word_embeddings,
        word_texts,
        top_k=top_k
    )

    return weakest_words