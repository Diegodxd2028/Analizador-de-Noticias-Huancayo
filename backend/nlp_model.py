# backend/nlp_model.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

ART = Path(__file__).resolve().parent / "artifacts"

# Carga perezosa global
_cache = {"encoder": None, "clf": None, "tfidf_nb": None, "tfidf": None}

# --- Patrones simples (mejorables a futuro) ---
PATTERNS = {
    "salud": ["vacuna","covid","hospital","salud","síntoma","médico","casos","epidemia","dengue"],
    "política": ["congreso","alcalde","gobierno","elecciones","decreto","ministro","partido","corrupción"],
    "economía": ["inflación","precio","dólar","pbi","desempleo","impuesto","importación","minería","comercio"],
}

def detect_pattern(text: str) -> str | None:
    t = text.lower()
    best, hits = None, 0
    for k, words in PATTERNS.items():
        c = sum(w in t for w in words)
        if c > hits:
            best, hits = k, c
    return best if hits > 0 else None

def load_sbert_stack():
    if _cache["clf"] is None:
        _cache["clf"] = joblib.load(ART / "sbert_logreg.joblib")
        cfg = json.loads((ART / "sbert_config.json").read_text(encoding="utf-8"))
        _cache["encoder"] = SentenceTransformer(cfg["encoder_name"])
    return _cache["encoder"], _cache["clf"]

def load_tfidf_nb():
    # modelo pipeline: ('tfidf', TfidfVectorizer), ('clf', ComplementNB)
    if _cache["tfidf_nb"] is None:
        _cache["tfidf_nb"] = joblib.load(ART / "model.joblib")
        _cache["tfidf"] = _cache["tfidf_nb"].named_steps["tfidf"]
    return _cache["tfidf_nb"], _cache["tfidf"]

def sbert_embed(texts):
    enc, _ = load_sbert_stack()
    return enc.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

def predict_main(text: str, abstain_threshold: float = 0.55):
    # Predicción principal con SBERT + LogReg
    enc, clf = load_sbert_stack()
    vec = sbert_embed([text])
    proba = clf.predict_proba(vec)[0]
    classes = clf.classes_
    idx = int(np.argmax(proba))
    label = classes[idx]
    score = float(proba[idx])

    # Abstenerse si la confianza es baja (opcional y honesto)
    abstain = score < abstain_threshold

    # Patrón temático rápido
    pattern = detect_pattern(text)

    return {"label": label, "score": round(score, 4), "abstain": abstain, "pattern": pattern}

def explain_terms(text: str, top_k: int = 8):
    """
    Explicabilidad con el modelo TF-IDF + NB:
    extrae las n-gramas con mayor peso para la clase predicha.
    """
    tfidf_nb, tfidf = load_tfidf_nb()
    # predicción del NB para saber a qué clase apuntar
    pred = tfidf_nb.predict([text])[0]
    # Representación TF-IDF del documento
    X = tfidf.transform([text])
    feature_names = np.array(tfidf.get_feature_names_out())
    indices = X.nonzero()[1]
    vals = X.data

    # Para NB: feature_log_prob_ [clase, feature]
    clf = tfidf_nb.named_steps["clf"]
    classes = clf.classes_
    cls_idx = int(np.where(classes == pred)[0][0])
    log_probs = clf.feature_log_prob_[cls_idx]

    # Score local = tfidf_val * log_prob_feature
    local_scores = vals * log_probs[indices]
    order = np.argsort(local_scores)[::-1]
    top_terms = []
    for i in order[:top_k]:
        top_terms.append({"term": feature_names[indices[i]], "contrib": float(local_scores[i])})

    return {"explainer_model": "tfidf+ComplementNB", "predicted_by_tfidf": pred, "top_terms": top_terms}
