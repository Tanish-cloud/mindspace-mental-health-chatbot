import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
import json

BASE       = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
PROCESSED  = BASE / 'data' / 'processed'
SAVE_PATH  = BASE / 'models' / 'speech_model_v2'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load cached features ──────────────────────────────────────
print("\nLoading cached features...")
data = np.load(PROCESSED / 'speech_features_cache.npz')
X    = data['X'].astype(np.float32)
y    = data['y']

print(f"X shape: {X.shape}")
print(f"Unique labels: {set(y)}")
print(f"Label counts:")
for label in set(y):
    print(f"  {label}: {sum(y == label)}")

# ── Encode labels ─────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes   = le.classes_
print(f"\nEncoded classes: {list(enumerate(classes))}")

# ── Normalize features ────────────────────────────────────────
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_norm = (X - mean) / std

# ── Train/val/test split ──────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_encoded, test_size=0.2,
    random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── Dataset ───────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = SpeechDataset(X_train, y_train)
val_ds   = SpeechDataset(X_val,   y_val)
test_ds  = SpeechDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128)
test_loader  = DataLoader(test_ds,  batch_size=128)

# ── Model ─────────────────────────────────────────────────────
INPUT_SIZE  = X.shape[1]   # 222
NUM_CLASSES = len(classes)  # 4

class SpeechEmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        conv_out = (input_size // 16) * 512
        self.classifier = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),      nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

model = SpeechEmotionCNN(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Class weights ─────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_train
)
class_weights = torch.tensor(cw, dtype=torch.float).to(DEVICE)
print(f"Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, verbose=True
)

# ── Training ──────────────────────────────────────────────────
print("\nTraining...")
best_val_acc = 0
best_epoch   = 0
patience     = 7
patience_counter = 0

for epoch in range(50):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (out.argmax(1) == y_batch).sum().item()
        train_total   += len(y_batch)

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            val_loss    += loss.item()
            val_correct += (out.argmax(1) == y_batch).sum().item()
            val_total   += len(y_batch)

    train_acc = train_correct / train_total
    val_acc   = val_correct   / val_total
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1:2d} | "
          f"Train: {train_acc:.4f} | "
          f"Val: {val_acc:.4f} | "
          f"Loss: {avg_val_loss:.4f}")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch + 1
        torch.save(model.state_dict(),
                   SAVE_PATH / 'best_speech_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

# ── Save config and normalization ─────────────────────────────
SAVE_PATH.mkdir(parents=True, exist_ok=True)
np.save(SAVE_PATH / 'feature_mean.npy', mean)
np.save(SAVE_PATH / 'feature_std.npy',  std)

config = {
    'input_size':   INPUT_SIZE,
    'num_classes':  NUM_CLASSES,
    'class_names':  list(classes),
    'sample_rate':  22050,
    'duration':     3,
    'feature_mean': 'feature_mean.npy',
    'feature_std':  'feature_std.npy',
}
with open(SAVE_PATH / 'speech_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Config saved: {config}")

# ── Final evaluation ──────────────────────────────────────────
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(SAVE_PATH / 'best_speech_model.pt'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        preds   = model(X_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_batch.numpy())

print("\n" + "="*50)
print("FINAL SPEECH MODEL RESULTS")
print("="*50)
print(classification_report(
    all_true, all_preds,
    target_names=classes
))

# ── Emotion to Mindify mapping ────────────────────────────────
emotion_map = {
    'sad':     'Depression',
    'anxious': 'Anxiety',
    'stressed':'Stress',
    'calm':    'Depression',
}
print("\nEmotion → Mindify mapping:")
for k, v in emotion_map.items():
    print(f"  {k:10s} → {v}")