import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

DATA_PATH = 'src/test_set.csv'
MODEL_PATH = 'src/model_nlp.pkl'
VECTORIZER_PATH = 'src/vectorizer.pkl'

# Load dataset
df = pd.read_csv(DATA_PATH)
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(int).tolist()

# Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load model/vectorizer if exists, else train
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=500)
    model.fit(X_vec, y_train)

# Transform test set
X_test_vec = vectorizer.transform(X_test)
preds = model.predict(X_test_vec)

# Metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")