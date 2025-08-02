import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os

dataset = 'mr'
loss= 'cos-sim'
emb_path = f'../embeddings/{dataset}/{loss}'
# Safely load .pt files regardless of original device
device = torch.device("cpu")
train_emb = torch.load(os.path.join(emb_path, "train_emb.pt"), map_location=device)
train_labels = torch.load(os.path.join(emb_path,"train_labels.pt"), map_location=device)
val_emb = torch.load(os.path.join(emb_path,"val_emb.pt"), map_location=device)
val_labels = torch.load(os.path.join(emb_path,"val_labels.pt"), map_location=device)
test_emb = torch.load(os.path.join(emb_path,"test_emb.pt"), map_location=device)
test_labels = torch.load(os.path.join(emb_path,"test_labels.pt"), map_location=device)

# DataLoader setup
batch_size = 128
train_loader = DataLoader(TensorDataset(train_emb, train_labels), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_emb, val_labels), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_emb, test_labels), batch_size=batch_size)

# Linear classification model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Model initialization
input_dim = train_emb.shape[1]
num_classes = len(torch.unique(train_labels))
model = LinearClassifier(input_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# Training with early stopping
num_epochs = 300
best_val_acc = 0.0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_true.extend(y_batch.tolist())

    val_acc = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1} — Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_linear_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break



# Test Evaluation
model.load_state_dict(torch.load("best_linear_model.pt", map_location=device))
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.tolist())
        test_true.extend(y_batch.tolist())

test_acc = accuracy_score(test_true, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
