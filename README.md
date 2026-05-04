# 🎯 TriageAI — Thai Seller Review Dashboard

A Streamlit web app that automatically analyzes Thai product reviews for online sellers using sentiment analysis and AI-powered insights.

Built as a CS462 Artificial Intelligence course project.

---

## What it does

Upload a CSV of customer reviews and the app will:
- Classify each review as **positive**, **neutral**, or **negative** using a trained ML model
- Show a sentiment breakdown with charts and metrics
- Use **Gemini 2.5 Flash** to summarize what customers are complaining about and give actionable recommendations in Thai

---

## How it works

1. **Preprocessing** — Thai text is cleaned using PyThaiNLP (unicode normalization, repeated character reduction, URL/hashtag removal, emoji stripping)
2. **Tokenization** — PyThaiNLP word tokenizer
3. **Vectorization** — TF-IDF with unigrams and bigrams
4. **Classification** — Logistic Regression trained on the Wisesight Sentiment Dataset (~27,000 Thai slang words), achieving ~70% accuracy on 3-class sentiment (pos/neg/neu)
5. **AI Analysis** — Negative reviews are passed to Gemini 2.5 Flash for theme extraction and business recommendations

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| Thai NLP | PyThaiNLP |
| ML Model | scikit-learn (Logistic Regression + TF-IDF) |
| AI Insights | Google Gemini 2.5 Flash |
| Data | pandas |

---

## CSV Format

Your input CSV must have these two columns:

| column | description |
|---|---|
| `comment` | The review text (Thai) |
| `rating_star` | Star rating (1–5) |

A sample file `mock_reviews.csv` is included.

---

## Setup

1. Clone the repo
```bash
git clone https://github.com/tecchuuu/ai-complaint-triage-thai-sellers.git
cd ai-complaint-triage-thai-sellers
```

2. Install dependencies
```bash
pip install streamlit pythainlp scikit-learn pandas google-generativeai python-dotenv emoji joblib
```

3. Create a `.env` file with your Gemini API key
```
GEMINI_API_KEY=your_api_key_here
```

4. Run the app
```bash
streamlit run app.py
```

---

## Model Performance

Trained on Thai social media text (slang-heavy), 3-class classification:

| Class | F1 Score |
|---|---|
| Negative | 0.69 |
| Neutral | 0.77 |
| Positive | 0.40 |
| **Overall Accuracy** | **0.70** |

> Positive class underperforms due to class imbalance in training data. Future improvement would involve WangchanBERTa fine-tuning for better contextual understanding.

---

## Training Notebook

The full model training pipeline (preprocessing, TF-IDF, model comparison, evaluation) is available on Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17vZNa0tkJYsA7IOXvRcznHjDUd3knwJv?authuser=1#scrollTo=qNfctlMuSxcA)

---

## Limitations

- Traditional ML (TF-IDF + LR) cannot capture context — "ของดี" and "ของไม่ดี" may be misclassified
- Positive sentiment detection is weaker than negative/neutral
- Model trained on social media slang; may underperform on formal text
