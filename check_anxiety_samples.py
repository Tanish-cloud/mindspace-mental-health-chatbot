# check_anxiety_samples.py
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify\data\text\final_combined_dataset.csv'))

print("Sample ANXIETY posts from training data:")
anxiety = df[df['label'] == 1]['text'].sample(20, random_state=42)
for i, t in enumerate(anxiety):
    print(f"\n[{i+1}] {t[:200]}")