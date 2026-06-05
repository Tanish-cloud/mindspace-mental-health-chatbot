# retrain_final2.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset

BASE      = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
SAVE_PATH = BASE / 'models' / 'mindify-final-v2'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────
print("\nLoading data...")

# Real Reddit depression + anxiety
main = pd.read_csv(BASE / 'data' / 'text' / 'final_combined_dataset.csv')
main = main[main['label'].isin([0, 1])][['text', 'label']].copy()
main = main.drop_duplicates(subset='text')
main = main[main['text'].str.len() >= 50]
print(f"Reddit depression+anxiety: {len(main)}")

# Add diverse anxiety from mental_health_conditions
mhc = pd.read_csv(BASE / 'data' / 'text' / 'mental_health_conditions.csv')

# Add extra anxiety samples (shorter, more diverse)
extra_anxiety = mhc[mhc['status'] == 'anxiety'][['text']].copy()
extra_anxiety = extra_anxiety[extra_anxiety['text'].str.len() >= 50]
extra_anxiety = extra_anxiety[extra_anxiety['text'].str.len() <= 500]
extra_anxiety['label'] = 1
extra_anxiety = extra_anxiety.sample(n=3000, random_state=42)
print(f"Extra anxiety samples: {len(extra_anxiety)}")

# Add extra depression samples for balance
extra_dep = mhc[mhc['status'] == 'depression'][['text']].copy()
extra_dep = extra_dep[extra_dep['text'].str.len() >= 50]
extra_dep = extra_dep[extra_dep['text'].str.len() <= 500]
extra_dep['label'] = 0
extra_dep = extra_dep.sample(n=3000, random_state=42)
print(f"Extra depression samples: {len(extra_dep)}")

# Stress from dreaddit
dreaddit = pd.read_csv(BASE / 'data' / 'text' / 'dreaddit.csv')
stress = dreaddit[dreaddit['label'] == 1][['text']].copy()
stress = stress[stress['text'].str.len() >= 50]
stress['label'] = 2
stress = stress.sample(n=5000, replace=True, random_state=42)
print(f"Stress (upsampled): {len(stress)}")

# Combine all
df = pd.concat([main, extra_anxiety, extra_dep, stress], ignore_index=True)
df = df.drop_duplicates(subset='text')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal distribution:")
print(f"  Depression (0): {len(df[df['label']==0])}")
print(f"  Anxiety    (1): {len(df[df['label']==1])}")
print(f"  Stress     (2): {len(df[df['label']==2])}")
print(f"  Total         : {len(df)}")

# ── Split ─────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

# ── Tokenizer ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# ── Dataset ───────────────────────────────────────────────────
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

print("\nTokenizing...")
train_dataset = MindifyDataset(train_df['text'], train_df['label'])
test_dataset  = MindifyDataset(test_df['text'],  test_df['label'])

# ── Class weights ─────────────────────────────────────────────
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_df['label'].values
)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print(f"Class weights: {class_weights}")

# ── Model ─────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)
model.config.id2label = {0: 'Depression', 1: 'Anxiety', 2: 'Stress'}
model.config.label2id = {'Depression': 0, 'Anxiety': 1, 'Stress': 2}

# ── Weighted trainer ──────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(
            weight=class_weights
        )(outputs.logits, labels)
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

# ── Training ──────────────────────────────────────────────────
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

print("\nTraining...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────
print("\nSaving...")
trainer.save_model(str(SAVE_PATH))
tokenizer.save_pretrained(str(SAVE_PATH))
print(f"Saved to: {SAVE_PATH}")

# ── Evaluate ──────────────────────────────────────────────────
preds_out = trainer.predict(test_dataset)
y_pred = np.argmax(preds_out.predictions, axis=1)
y_true = test_df['label'].values

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(classification_report(
    y_true, y_pred,
    target_names=['Depression', 'Anxiety', 'Stress']
))

# ── Anxiety inference test ────────────────────────────────────
model.eval()
anxiety_tests = [
    ("I keep having panic attacks and I don't know how to stop them",  "Anxiety"),
    ("My heart is racing and I can't breathe, something is wrong",     "Anxiety"),
    ("I am constantly worried about everything, can't turn brain off", "Anxiety"),
    ("I feel terrified for no reason, my hands are shaking",           "Anxiety"),
    ("I am scared to go outside, what if something goes wrong",        "Anxiety"),
    ("I can't speak in public, I freeze up completely",                "Anxiety"),
    ("I lie awake worrying about things I cannot control",             "Anxiety"),
    ("I feel on edge all the time, like something bad will happen",    "Anxiety"),
    ("I feel completely hopeless, nothing matters anymore",            "Depression"),
    ("I have so many deadlines, completely overwhelmed at work",       "Stress"),
]

labels = ['Depression', 'Anxiety', 'Stress']
correct = 0
print("\nAnxiety inference test:")
for text, expected in anxiety_tests:
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    pred = labels[probs.argmax().item()]
    status = "✓" if pred == expected else "✗"
    if pred == expected:
        correct += 1
    print(f"  {status} {pred:12s} | D:{probs[0]*100:.1f}%"
          f" A:{probs[1]*100:.1f}% S:{probs[2]*100:.1f}%"
          f" | {text[:50]}...")

print(f"\nInference accuracy: {correct}/{len(anxiety_tests)}")