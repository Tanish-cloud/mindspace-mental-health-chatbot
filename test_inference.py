import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE       = Path(r'C:\Users\TANISH VERMA\OneDrive\Desktop\Mindify')
SAVE_PATH  = BASE / 'models' / 'mindify-retrained'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved model
tokenizer  = AutoTokenizer.from_pretrained(str(SAVE_PATH))
model      = AutoModelForSequenceClassification.from_pretrained(str(SAVE_PATH))
model      = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}\n")

labels = ['Depression', 'Anxiety', 'Stress']

def predict(text):
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    pred = labels[probs.argmax().item()]
    print(f"  {pred:12s} | D:{probs[0]*100:.1f}%"
          f" A:{probs[1]*100:.1f}% S:{probs[2]*100:.1f}%"
          f" | {text[:60]}...")

# test_inference.py - add these stress sentences
tests = [
    # Work stress
    "I have three project deadlines this week and my manager keeps adding more tasks",
    "I have been working 12 hour days for a month straight, I am completely burnt out",
    "My boss expects me to be available 24/7 and I have no time for myself anymore",
    "I have so much work piled up I don't even know where to start",
    
    # Academic stress
    "Finals week is killing me, I have 4 exams in 3 days and haven't slept properly",
    "I have assignments due every single day this week and I am falling behind",
    "The pressure to get good grades is overwhelming, my parents expect too much",
    "I have a major presentation tomorrow and I am not prepared at all",
    
    # Life pressure
    "I can't afford my rent this month, bills are piling up and I don't know what to do",
    "I am juggling work, family, and studies and I feel like I am about to collapse",
    "Everything is happening at once and I have no time to breathe or recover",
    "The responsibilities keep piling up and I feel completely overwhelmed",
    
    # Dreaddit style (matches training data)
    "I don't know how much longer I can keep doing this, the pressure is unbearable",
    "Between work and family obligations I barely have time to breathe anymore",
    "I feel like I am about to snap, too much pressure from all sides",
]

for t in tests:
    predict(t)