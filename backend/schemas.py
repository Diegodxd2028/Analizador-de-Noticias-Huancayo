from pydantic import BaseModel

class PredictIn(BaseModel):
    text: str | None = None
    url: str | None = None

class PredictOut(BaseModel):
    label: str
    score: float
