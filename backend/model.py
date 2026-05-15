from pathlib import Path
import torch
import torch.nn as nn
from transformers import WavLMModel


class PronunciationScorer(nn.Module):
    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

        for param in self.wavlm.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, audio):
        if isinstance(audio, torch.Tensor):
            input_values = audio.float()

            if input_values.dim() == 1:
                input_values = input_values.unsqueeze(0)

        else:
            input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = self.wavlm(input_values)

        hidden_states = outputs.last_hidden_state
        pooled_embedding = hidden_states.mean(dim=1)

        score = self.regressor(pooled_embedding)

        return score, pooled_embedding


scorer_model = PronunciationScorer()

BASE_DIR = Path(__file__).resolve().parent.parent
HEAD_PATH = BASE_DIR / "saved_models" / "regression_head.pt"

scorer_model.regressor.load_state_dict(
    torch.load(HEAD_PATH, map_location="cpu")
)

scorer_model.eval()


def predict_score(audio):
    with torch.no_grad():
        score, embedding = scorer_model(audio)

    score_0_10 = score.item()

    # dataset score is 0–10, frontend wants 0–100
    score_0_100 = max(0, min(100, score_0_10 * 10))

    return score_0_100, embedding