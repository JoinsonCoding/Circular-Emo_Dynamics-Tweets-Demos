from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import statistics

# -------------------
# Initialize Flask app
# -------------------
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# -------------------
# Load Twitter-roBERTa model once
# -------------------
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# -------------------
# Store previous sentiment scores (for mean & SD)
# -------------------
history_positive = []
history_negative = []

# -------------------
# Landing page
# -------------------
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Choose Tool</title>
        <style>
            body { display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; font-family:system-ui; }
            button { padding:20px 40px; margin:20px; font-size:20px; cursor:pointer; border-radius:8px; border:none; background:#3498db; color:white; }
            button:hover { background:#2980b9; }
        </style>
    </head>
    <body>
        <h1>Select a Tool</h1>
        <button onclick="location.href='/sentiment'">Sentiment Analyzer</button>
        <button onclick="location.href='/clock'">Circular Clock</button>
    </body>
    </html>
    """

# -------------------
# Sentiment Analyzer route
# -------------------
@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

# -------------------
# Circular Clock route
# -------------------
@app.route('/clock')
def clock():
    return render_template('clock.html')

# -------------------
# Sentiment API using Twitter-roBERTa
# -------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

    # Model classes: [negative, neutral, positive]
    negative, neutral, positive = probs
    score = round(positive * 100)  # main display score
    label = "Positive" if positive >= negative else "Negative"
    emoji = "😊" if label == "Positive" else "😞"

    # Save history
    history_positive.append(positive*100)
    history_negative.append(negative*100)

    # Compute mean & SD
    mean_positive = round(statistics.mean(history_positive), 1)
    sd_positive = round(statistics.stdev(history_positive), 1) if len(history_positive) > 1 else 0.0
    mean_negative = round(statistics.mean(history_negative), 1)
    sd_negative = round(statistics.stdev(history_negative), 1) if len(history_negative) > 1 else 0.0

    return jsonify({
        'score': score,
        'label': label,
        'emoji': emoji,
        'positive': round(positive*100),
        'negative': round(negative*100),
        'mean_positive': mean_positive,
        'sd_positive': sd_positive,
        'mean_negative': mean_negative,
        'sd_negative': sd_negative
    })

# -------------------
# Run the app
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
