import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictStr
from transformers import AutoModel, AutoTokenizer

from text2topic_model import Text2TopicModel, normalize_text

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent / "outputs/text2topic")))
THRESHOLD = 0.5
logger = logging.getLogger(__name__)
lock = Lock()


class PaperRequest(BaseModel):
    title: StrictStr = Field(..., min_length=1, max_length=10000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp checkpoint và tính trước vector chủ đề một lần."""
    app.state.ready = False
    try:
        metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        config = metadata["args"]
        names, ids, texts = (metadata[key] for key in ("topic_names", "label_ids", "topic_texts"))
        if not names or not len(names) == len(ids) == len(texts):
            raise ValueError("Metadata labels are empty or inconsistent.")
        if not all(isinstance(t, str) and t.strip() for t in names + texts):
            raise ValueError("Metadata contains invalid topic names or texts.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "encoder", local_files_only=True)
        model = Text2TopicModel(
            AutoModel.from_pretrained(MODEL_DIR / "encoder", local_files_only=True),
            config["dropout"],
        )
        model.classifier.load_state_dict(torch.load(
            MODEL_DIR / "classifier.pt", map_location="cpu", weights_only=True,
        ))
        model.to(device=device, dtype=dtype).eval()
        topics = tokenizer(texts, padding=True, truncation=True,
                           max_length=config["topic_max_length"], return_tensors="pt")
        with torch.inference_mode():
            vectors = model.encode({key: value.to(device) for key, value in topics.items()})
        app.state.predictor = (model, tokenizer, device, vectors, names, ids, config["max_length"])
        app.state.ready = True
    except Exception:
        logger.exception("Cannot load Text2Topic checkpoint")
    yield
    app.state.ready = False
    app.state.predictor = None


app = FastAPI(title="Paper Topic API", lifespan=lifespan)


@app.post("/predict")
def predict(paper: PaperRequest):
    """Trả nhãn đạt ngưỡng 0.5 hoặc nhãn có xác suất cao nhất."""
    title = normalize_text(paper.title)
    if not title:
        raise HTTPException(422, "Title must not be blank.")
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "Model unavailable. Check checkpoint and server logs.")
    try:
        model, tokenizer, device, vectors, names, ids, max_length = app.state.predictor
        with lock, torch.inference_mode():
            tokens = tokenizer(title, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            scores = model.score_pairs(model.encode(tokens), vectors).float().sigmoid()[0].cpu()
        if scores.ndim != 1 or scores.numel() != len(names) or not torch.isfinite(scores).all():
            raise ValueError("Model returned invalid probabilities.")
        selected = (scores >= THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        fallback = not selected
        if fallback:
            selected = [scores.argmax().item()]
        return {"title": title, "labels": [
            {"id": ids[i], "name": names[i], "probability": scores[i].item()} for i in selected
        ], "fallback": fallback}
    except torch.cuda.OutOfMemoryError:
        logger.exception("Insufficient GPU memory")
        raise HTTPException(503, "Insufficient GPU memory. Try again later.") from None
    except Exception:
        logger.exception("Paper prediction failed")
        raise HTTPException(500, "Prediction failed. Check server logs.") from None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
