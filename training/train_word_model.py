import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


DATA_PATH = "dataset/word_embeddings_subset.pt"
SAVE_PATH = "saved_models/word_regression_head.pt"


data = torch.load(DATA_PATH)

embeddings = data["embeddings"]
scores = data["scores"]



X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    scores,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

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


model = WordRegressionHead()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

epochs = 20

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for batch_embeddings, batch_scores in train_loader:
        predictions = model(batch_embeddings)

        loss = loss_fn(predictions, batch_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_embeddings, batch_scores in test_loader:
            test_predictions = model(batch_embeddings)
            loss = loss_fn(test_predictions, batch_scores)
            test_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Test Loss: {avg_test_loss:.4f}"
    )
    
torch.save(
    model.state_dict(),
    SAVE_PATH
)

print(f"Saved word model to {SAVE_PATH}")