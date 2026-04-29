from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json, os

app = Flask(__name__)
CORS(app)

MODEL_DIR = './saved_model'
EMOTION_LABELS = [
    "joy","sadness","anger","fear","surprise","disgust",
    "anticipation","trust","love","optimism","pessimism","anxiety",
    "embarrassment","gratitude","relief","confusion"
]

EMOTION_COLORS = {
    "joy":"#FFD700","sadness":"#4169E1","anger":"#FF4500",
    "fear":"#8B008B","surprise":"#FF69B4","disgust":"#228B22",
    "anticipation":"#FF8C00","trust":"#20B2AA","love":"#FF1493",
    "optimism":"#32CD32","pessimism":"#708090","anxiety":"#9400D3",
    "embarrassment":"#FF6347","gratitude":"#00CED1",
    "relief":"#90EE90","confusion":"#D2691E"
}

EMOTION_EMOJIS = {
    "joy":"😊","sadness":"😢","anger":"😠","fear":"😨",
    "surprise":"😲","disgust":"🤢","anticipation":"🤩",
    "trust":"🤝","love":"❤️","optimism":"🌟","pessimism":"😞",
    "anxiety":"😰","embarrassment":"😳","gratitude":"🙏",
    "relief":"😌","confusion":"😕"
}

print("Loading BERT model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    encoding = tokenizer(
        text, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    emotion = EMOTION_LABELS[top_idx]
    confidence = float(probs[top_idx])

    all_emotions = [
        {
            'emotion': EMOTION_LABELS[i],
            'probability': float(probs[i]),
            'emoji': EMOTION_EMOJIS[EMOTION_LABELS[i]],
            'color': EMOTION_COLORS[EMOTION_LABELS[i]]
        }
        for i in range(len(EMOTION_LABELS))
    ]
    all_emotions.sort(key=lambda x: x['probability'], reverse=True)

    return jsonify({
        'text': text,
        'emotion': emotion,
        'confidence': confidence,
        'emoji': EMOTION_EMOJIS[emotion],
        'color': EMOTION_COLORS[emotion],
        'all_emotions': all_emotions[:5]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'BERT Emotion Detector'})

if __name__ == '__main__':
    print("🧠 BERT Emotion Detector — Flask API")
    print("✅ Server running at: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)