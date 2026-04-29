import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os, json

EMOTION_LABELS = [
    "joy","sadness","anger","fear","surprise","disgust",
    "anticipation","trust","love","optimism","pessimism","anxiety",
    "embarrassment","gratitude","relief","confusion"
]
NUM_LABELS = 16

GOEMOTIONS_TO_16 = {
    0:8,1:0,2:2,3:2,4:6,5:7,6:15,7:6,
    8:8,9:1,10:5,11:5,12:12,13:0,14:3,15:13,
    16:1,17:0,18:8,19:11,20:9,21:7,22:15,23:14,
    24:1,25:1,26:4,27:9
}

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data():
    try:
        from datasets import load_dataset
        print("[INFO] Downloading GoEmotions...")
        ds = load_dataset("go_emotions", "simplified")
        texts, labels = [], []
        for split in ['train','validation','test']:
            for item in ds[split]:
                if len(item['labels']) > 0:
                    texts.append(item['text'])
                    labels.append(GOEMOTIONS_TO_16.get(item['labels'][0], 0))
        print(f"[INFO] Loaded {len(texts)} samples!")
        return texts, labels
    except Exception as e:
        print(f"[WARN] Using synthetic data: {e}")
        import random; random.seed(42)
        t = {
            0:["So happy!","Best day ever!"],
            1:["Feeling sad","Heartbroken"],
            2:["So angry!","How dare!"],
            3:["Scared","Terrified"],
            4:["What a surprise!","Unexpected!"],
            5:["Disgusting","Revolting"],
            6:["Can't wait!","So excited!"],
            7:["I trust this","Reliable"],
            8:["I love you!","Full of love"],
            9:["Staying positive!","Hopeful"],
            10:["Given up","Nothing works"],
            11:["So anxious","Overthinking"],
            12:["Embarrassed","Mortifying"],
            13:["So grateful","Thank you!"],
            14:["What relief!","Finally done!"],
            15:["Confused","No sense"]
        }
        texts, labels = [], []
        for _ in range(5000):
            l = __import__('random').randint(0,15)
            texts.append(__import__('random').choice(t[l]))
            labels.append(l)
        return texts, labels

def train_model(epochs=3, batch_size=16, save_dir='./saved_model'):
    print("="*50)
    print("  BERT EMOTION DETECTOR - TRAINING")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    texts, labels = load_data()
    X_train,X_test,y_train,y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train,X_val,y_train,y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    print(f"[INFO] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=NUM_LABELS).to(device)

    train_dl = DataLoader(EmotionDataset(X_train,y_train,tokenizer),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(EmotionDataset(X_val,y_val,tokenizer),
                          batch_size=batch_size)
    test_dl  = DataLoader(EmotionDataset(X_test,y_test,tokenizer),
                          batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*len(train_dl)*epochs),
        num_training_steps=len(train_dl)*epochs)

    for epoch in range(1, epochs+1):
        model.train(); total_loss=0
        for batch in train_dl:
            optimizer.zero_grad()
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += out.loss.item()

        model.eval(); correct=total=0
        with torch.no_grad():
            for batch in val_dl:
                out = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device))
                preds = out.logits.argmax(dim=1)
                correct += (preds==batch['labels'].to(device)).sum().item()
                total += batch['labels'].size(0)
        print(f"✅ Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {correct/total:.4f}")

    # Test evaluation
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device))
            all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir,'label_map.json'),'w') as f:
        json.dump({'labels': EMOTION_LABELS}, f)
    print(f"\n🎉 Model saved to {save_dir}")

if __name__ == '__main__':
    train_model(epochs=3, batch_size=16, save_dir='./saved_model')