# check_stress_samples.py
import pandas as pd
from pathlib import Path

mhc = pd.read_csv(Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify\data\text\mental_health_conditions.csv'))
stress = mhc[mhc['status'] == 'stress']
print(f"Stress samples: {len(stress)}")
print("\nSamples:")
for t in stress['text'].sample(15, random_state=1):
    print(f"- {t[:200]}")
    print()