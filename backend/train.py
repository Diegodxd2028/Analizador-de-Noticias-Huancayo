# backend/train.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data" / "dataset.csv"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

if not DATASET.exists():
    df = pd.DataFrame({
        "text": [
            "Vacuna causa efectos mortales sin evidencia científica",
            "Municipalidad de Huancayo aprueba presupuesto para obras locales"
        ],
        "label": ["Fake", "Real"]
    })
else:
    df = pd.read_csv(DATASET).dropna(subset=["text", "label"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42,
    stratify=df["label"] if df["label"].nunique() > 1 and len(df) > 5 else None
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
    ("clf", ComplementNB())
])

pipe.fit(X_train, y_train)
if len(X_test) > 0 and df["label"].nunique() > 1:
    print(classification_report(y_test, pipe.predict(X_test)))

joblib.dump(pipe, ARTIFACTS / "model.joblib")
print(f"✅ Modelo TF-IDF+NB guardado en {ARTIFACTS / 'model.joblib'}")
