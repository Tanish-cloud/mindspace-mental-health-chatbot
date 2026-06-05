# retrain2.py - uses real Reddit data like before
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
SAVE_PATH = BASE / 'models' / 'mindify-final'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load real Reddit data ─────────────────────────────────────
print("\nLoading data...")

# Depression + Anxiety from final_combined_dataset
main_df = pd.read_csv(BASE / 'data' / 'text' / 'reddit_mental_health.csv')
print("reddit_mental_health columns:", main_df.columns.tolist())
print(main_df.head(3))