# backend/app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

from .db import Base, engine, SessionLocal
from .models import Prediction
from .nlp_model import predict_main, explain_terms

app = FastAPI(title="Analizador de Noticias Huancayo", version="1.0.0")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

class PredictIn(BaseModel):
    text: str | None = None
    url: str | None = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def extract_text_from_url(url: str) -> str:
    r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # intenta priorizar <article>, si existe
    article = soup.find("article")
    base = article or soup
    paragraphs = [p.get_text(" ", strip=True) for p in base.find_all("p")]
    text = " ".join(paragraphs)
    return text[:20000]

@app.get("/")
def root():
    return {"ok": True, "service": "analizador-noticias", "version": "1.0.0"}

@app.post("/predict")
def predict(inp: PredictIn, db: Session = Depends(get_db)):
    text = (inp.text or "").strip()
    src = "text"
    src_val = None

    if not text and inp.url:
        try:
            text = extract_text_from_url(inp.url)
            src = "url"
            src_val = inp.url
        except Exception as e:
            raise HTTPException(400, f"No se pudo leer la URL: {e}")

    if not text or len(text) < 30:
        raise HTTPException(400, "Proporciona texto suficiente o una URL válida con contenido.")

    res = predict_main(text)  # SBERT + LogReg
    # registro en BD
    record = Prediction(
        source=src,
        source_value=src_val if src == "url" else text[:280],
        label=res["label"],
        score=res["score"]
    )
    db.add(record)
    db.commit()

    return res

@app.post("/explain")
def explain(inp: PredictIn):
    text = (inp.text or "").strip()
    if not text and inp.url:
        try:
            text = extract_text_from_url(inp.url)
        except Exception as e:
            raise HTTPException(400, f"No se pudo leer la URL: {e}")
    if not text or len(text) < 30:
        raise HTTPException(400, "Proporciona texto suficiente o una URL válida con contenido.")

    return explain_terms(text)

@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    # Métrica simple: conteo por etiqueta, promedios de score
    from sqlalchemy import func
    total = db.query(func.count()).select_from(Prediction).scalar() or 0
    by_label = db.query(Prediction.label, func.count(), func.avg(Prediction.score))\
                 .group_by(Prediction.label).all()
    series = [{"label": l, "count": int(c), "avg_score": float(a or 0)} for l, c, a in by_label]
    return {"total": total, "by_label": series}
