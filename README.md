just save the front end code in your using .html and you can now get the acess to the emosense website on your local browser without any bacend.
the directory structure would be like this:-
emosense-
---backend----
saved_model
>-app.py
>evaluate.py
>predict.py
>requirements.txt
>train_model.py
----frontend----
>emosense.html
># EmoSense – Emotion Detection from Tweets using BERT

# Overview
EmoSense is a deep learning-based system that detects emotions from tweets using a fine-tuned BERT model.
The project supports real-time emotion prediction through an interactive frontend interface.
# Features
* Emotion classification from tweets (Joy, Sadness, Anger, Fear,and other total 16 emotion classes etc.)
* BERT-based NLP model
* Interactive frontend dashboard (emotion wheel + charts)
* Bulk tweet analysis support & Image tweet analysis support
* Model evaluation with precision, recall, F1-score
# Model Details

* Model: BERT (bert-base-uncased)
* Framework: PyTorch + HuggingFace Transformers
* Classes: 7–16 emotion categories
* Dataset: (Go-Emotions)

# project Structure

* `backend/` → model training, evaluation, inference
* `frontend/` → UI (runs independently in browser)

# How to Run

# frontend 

Simply open:

```bash
frontend/emosense.html
```

No backend required.

# Backend (Model Training / Evaluation)

```bash
cd backend
pip install -r requirements.txt
python src/train.py
```

# Screenshots

<img width="1920" height="1080" alt="Screenshot (272)" src="https://github.com/user-attachments/assets/03ac346b-f71d-4cdd-902d-eefa0274807f" />

# Note:-

The frontend runs independently using precomputed predictions / logic. Backend is used for training and model experimentation.

# Future Improvements

* API integration (Flask/FastAPI)
* Multimodal emotion detection (BERT + ViT)
* Deployment (Streamlit / Web App)

# Author
RamsaiGoud Bollampally
