import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = './saved_model'
EMOTION_LABELS = [
    "joy","sadness","anger","fear","surprise","disgust",
    "anticipation","trust","love","optimism","pessimism","anxiety",
    "embarrassment","gratitude","relief","confusion"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

def predict_emotion(text):
    enc = tokenizer(text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(input_ids=enc['input_ids'].to(device),
                    attention_mask=enc['attention_mask'].to(device))
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx = probs.argmax()
    return EMOTION_LABELS[idx], float(probs[idx])

if __name__ == '__main__':
    while True:
        text = input("\nEnter tweet (or 'quit'): ")
        if text == 'quit': break
        emotion, confidence = predict_emotion(text)
        print(f"Emotion: {emotion} ({confidence:.2%})")