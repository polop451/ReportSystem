from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Load word lists from files (if available)
def load_word_list(filename):
    try:
        with open(filename, encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

BADWORDS_TH = load_word_list('src/badwords_th.txt')
BADWORDS_EN = load_word_list('src/badwords_en.txt')
SPAM_KEYWORDS = load_word_list('src/spam_keywords.txt')

MODEL_PATH = 'src/model_nlp.pkl'
VECTORIZER_PATH = 'src/vectorizer.pkl'

class TextRequest(BaseModel):
    text: str

class CheckResult(BaseModel):
    badwords: List[str]
    spam_keywords: List[str]
    is_gambling: bool
    nlp_pred: int = None

def predict_nlp(text: str):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        return int(pred)
    return None

@app.post('/check_text', response_model=CheckResult)
def check_text(req: TextRequest):
    text = req.text.lower()
    badwords = [w for w in BADWORDS_TH.union(BADWORDS_EN) if w in text]
    spam = [k for k in SPAM_KEYWORDS if k in text]
    is_gambling = bool(re.search(r'(พนัน|casino|bet|แทงบอล|slot|หวย)', text))
    nlp_pred = predict_nlp(text)
    return CheckResult(badwords=badwords, spam_keywords=spam, is_gambling=is_gambling, nlp_pred=nlp_pred)

@app.post('/train')
async def train_model(file: UploadFile = File(...)):
    import pandas as pd
    df = pd.read_csv(file.file)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=500)
    model.fit(X, labels)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return {'message': 'Model trained and saved.'}

