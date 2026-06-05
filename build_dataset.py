import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

BASE      = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
SAVE_PATH = BASE / 'data' / 'text' / 'final_combined_dataset.csv'

print("Building dataset...\n")
dfs = []

# ── Source 1: Depression + Anxiety from HuggingFace ──────────
try:
    print("Loading depression/anxiety...")
    ds = load_dataset("loads/depression-and-anxiety-reddit")
    df = pd.DataFrame(ds['train'])
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Sample:\n{df.head(2)}")
except Exception as e:
    print(f"  Failed: {e}")

try:
    print("Trying backup...")
    ds = load_dataset("tyqiangz/multilingual-sentiments", "english")
    df = pd.DataFrame(ds['train'])
    print(f"  Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"  Failed: {e}")

# ── Source 2: Dreaddit stress (already have this) ────────────
print("\nLoading dreaddit stress...")
dreaddit = pd.read_csv(BASE / 'data' / 'text' / 'dreaddit.csv')
stress = dreaddit[dreaddit['label'] == 1][['text']].copy()
stress = stress[stress['text'].str.len() >= 50]
stress['label'] = 2
print(f"  Stress samples: {len(stress)}")
dfs.append(stress)

print("\nDone loading what we can.")
print(f"So far: {sum(len(d) for d in dfs)} samples")