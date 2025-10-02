# backend/train_sbert.py (robusto)
from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "dataset.csv"
ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True, parents=True)

ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

if not DATA.exists():
    raise FileNotFoundError(f"No existe {DATA}. Crea data/dataset.csv con columnas text,label (Fake/Real).")

df = pd.read_csv(DATA).dropna(subset=["text","label"])
labels = df["label"].unique().tolist()
if len(labels) < 2:
    raise ValueError(f"Se requiere al menos 2 clases. Encontradas: {labels}")

# Oversampling mínimo: asegurar >= 4 por clase
MIN_PER_CLASS = 4
parts = []
for lab, g in df.groupby("label"):
    if len(g) < MIN_PER_CLASS:
        g = g.sample(n=MIN_PER_CLASS, replace=True, random_state=42)
    parts.append(g)
dfb = pd.concat(parts, ignore_index=True)

# Intentar split estratificado; si no se puede, split simple
try:
    X_train, X_test, y_train, y_test = train_test_split(
        dfb["text"].tolist(), dfb["label"].tolist(),
        test_size=0.2, random_state=42, stratify=dfb["label"]
    )
    eval_ok = True
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        dfb["text"].tolist(), dfb["label"].tolist(),
        test_size=0.2, random_state=42
    )
    eval_ok = False

encoder = SentenceTransformer(ENCODER_NAME)
def enc(batch): return encoder.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

Xtr = enc(X_train); Xte = enc(X_test)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(Xtr, y_train)

if eval_ok and len(X_test) > 0:
    y_pred = clf.predict(Xte)
    print(classification_report(y_test, y_pred, digits=4))
else:
    print("Aviso: no se pudo estratificar por tamaño/clases; se entrenó sin reporte de test.")

joblib.dump(clf, ART / "sbert_logreg.joblib")
(ART / "sbert_config.json").write_text(json.dumps({"encoder_name": ENCODER_NAME}, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ Guardados: artifacts/sbert_logreg.joblib y artifacts/sbert_config.json")
