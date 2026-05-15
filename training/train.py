import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DATA_PATH = "dataset/embeddings.pt"
SAVE_PATH = "saved_models/regression_head.pt"

data = torch.load(DATA_PATH)

embeddings = data["embeddings"]
scores = data["scores"]

dataset = TensorDataset(embeddings, scores)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

regressor = nn.Sequential(
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)

epochs = 30

for epoch in range(epochs):
    regressor.train()
    train_loss = 0

    for batch_embeddings, batch_scores in train_loader:
        predictions = regressor(batch_embeddings).squeeze(1)
        loss = loss_fn(predictions, batch_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    regressor.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_embeddings, batch_scores in test_loader:
            predictions = regressor(batch_embeddings).squeeze(1)
            loss = loss_fn(predictions, batch_scores)
            test_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Test Loss: {test_loss / len(test_loader):.4f}"
    )

torch.save(regressor.state_dict(), SAVE_PATH)
print(f"Saved regression head to {SAVE_PATH}")