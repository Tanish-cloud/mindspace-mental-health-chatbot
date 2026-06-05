import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# ── Paths ─────────────────────────────────────────────────────
BASE       = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
DATA_PATH  = BASE / 'data' / 'text' / 'mental_health_conditions.csv'
SAVE_PATH  = BASE / 'models' / 'mindify-retrained'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Load & clean data ─────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)

# Keep only depression, anxiety, stress
label_map = {
    'depression': 0,
    'anxiety':    1,
    'stress':     2,
}
df = df[df['status'].isin(label_map.keys())].copy()
df['label'] = df['status'].map(label_map)
df = df[['text', 'label']].copy()

# Clean
df = df.drop_duplicates(subset='text')
df = df[df['text'].str.len() >= 50]
df = df.dropna(subset=['text'])
df = df.reset_index(drop=True)

print(f"\nDataset after cleaning:")
print(f"  Total: {len(df)}")
print(f"  Depression (0): {len(df[df['label']==0])}")
print(f"  Anxiety    (1): {len(df[df['label']==1])}")
print(f"  Stress     (2): {len(df[df['label']==2])}")

# ── Train/test split ──────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

# ── Tokenizer ─────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# ── Dataset class ─────────────────────────────────────────────
class MindifyDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts),
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

print("Tokenizing datasets...")
train_dataset = MindifyDataset(train_df['text'], train_df['label'])
test_dataset  = MindifyDataset(test_df['text'],  test_df['label'])
print(f"Train tokens: {len(train_dataset)} | Test tokens: {len(test_dataset)}")

# ── Class weights ─────────────────────────────────────────────
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_df['label'].values
)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"\nClass weights: {class_weights}")

# ── Model ─────────────────────────────────────────────────────
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)
model.config.id2label = {0: 'Depression', 1: 'Anxiety', 2: 'Stress'}
model.config.label2id = {'Depression': 0, 'Anxiety': 1, 'Stress': 2}

# ── Weighted trainer ──────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Metrics ───────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy':      accuracy_score(labels, preds),
        'macro_f1':      f1_score(labels, preds, average='macro'),
        'f1_depression': f1_score(labels, preds, average=None)[0],
        'f1_anxiety':    f1_score(labels, preds, average=None)[1],
        'f1_stress':     f1_score(labels, preds, average=None)[2],
    }

# ── Training args ─────────────────────────────────────────────
SAVE_PATH.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(SAVE_PATH),
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    logging_steps=50,
    fp16=True,
    dataloader_num_workers=0,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ── Train ─────────────────────────────────────────────────────
print("\nStarting training...")
trainer.train()

# ── Save properly ─────────────────────────────────────────────
print("\nSaving model...")
trainer.save_model(str(SAVE_PATH))
tokenizer.save_pretrained(str(SAVE_PATH))
print(f"Model saved to: {SAVE_PATH}")

# ── Final evaluation ──────────────────────────────────────────
print("\nRunning final evaluation...")
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = test_df['label'].values

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(classification_report(
    y_true, y_pred,
    target_names=['Depression', 'Anxiety', 'Stress']
))

# ── Quick inference test ──────────────────────────────────────
print("Quick inference test:")
model.eval()
tests = [
    ("I feel completely hopeless, nothing matters anymore", "Depression"),
    ("I keep having panic attacks, my heart is racing",     "Anxiety"),
    ("I have so many deadlines, completely overwhelmed",    "Stress"),
]
for text, expected in tests:
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    pred = ['Depression','Anxiety','Stress'][probs.argmax().item()]
    status = "✓" if pred == expected else "✗"
    print(f"  {status} Expected: {expected:12s} | Got: {pred:12s} | "
          f"D:{probs[0]*100:.1f}% A:{probs[1]*100:.1f}% S:{probs[2]*100:.1f}%")