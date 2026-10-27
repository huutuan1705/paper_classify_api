import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator
from transformers import AutoModel, AutoTokenizer

from expert_topic_inference import load_expertise_rules, recommend_expertise
from text2topic_model import Text2TopicModel, normalize_text

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent / "outputs/text2topic")))
THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 32
logger = logging.getLogger(__name__)
lock = Lock()


class PaperRequest(BaseModel):
    title: StrictStr = Field(..., min_length=1, max_length=10000)


class ExpertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    titles: list[Annotated[StrictStr, Field(min_length=1, max_length=10000)]] = Field(
        ..., min_length=1, max_length=1000,
        description="Publication titles for one expert; normalized duplicates count once.",
    )

    @field_validator("titles")
    @classmethod
    def normalize_titles(cls, titles: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for title in titles:
            title = normalize_text(title)
            if not title:
                raise ValueError("Titles must not be blank.")
            unique.setdefault(title.casefold(), title)
        return list(unique.values())


class ExpertTopic(BaseModel):
    topic_id: int
    topic_name: str
    expertise_score: float
    expertise_type: Literal["direct", "related"]
    paper_count: int


class ExpertResponse(BaseModel):
    publication_count: int
    topics: list[ExpertTopic]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp checkpoint và tính trước vector chủ đề một lần."""
    app.state.ready = False
    app.state.expert_rules = None
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
    if app.state.ready:
        try:
            app.state.expert_rules = load_expertise_rules(MODEL_DIR / "expert_topic_rules.json", metadata)
        except Exception:
            logger.exception("Cannot load expert topic rules")
    try:
        yield
    finally:
        app.state.ready = False
        app.state.predictor = None
        app.state.expert_rules = None


app = FastAPI(title="Paper and Expert Topic API", lifespan=lifespan)


def _classify_papers(titles: list[str]) -> list[dict[str, Any]]:
    """Share the existing threshold/fallback logic, using bounded title batches."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "Model unavailable. Check checkpoint and server logs.")
    try:
        model, tokenizer, device, vectors, names, ids, max_length = app.state.predictor
        predictions = []
        for start in range(0, len(titles), INFERENCE_BATCH_SIZE):
            batch = titles[start:start + INFERENCE_BATCH_SIZE]
            with lock, torch.inference_mode():
                if len(batch) == 1:
                    tokens = tokenizer(batch[0], truncation=True, max_length=max_length, return_tensors="pt")
                else:
                    tokens = tokenizer(batch, padding=True, truncation=True,
                                       max_length=max_length, return_tensors="pt")
                tokens = {key: value.to(device) for key, value in tokens.items()}
                probabilities = model.score_pairs(model.encode(tokens), vectors).float().sigmoid().cpu()
            if probabilities.shape != (len(batch), len(names)) or not torch.isfinite(probabilities).all():
                raise ValueError("Model returned invalid probabilities.")
            for title, scores in zip(batch, probabilities):
                selected = (scores >= THRESHOLD).nonzero(as_tuple=True)[0].tolist()
                fallback = not selected
                if fallback:
                    selected = [scores.argmax().item()]
                predictions.append({"title": title, "labels": [
                    {"id": ids[i], "name": names[i], "probability": scores[i].item()} for i in selected
                ], "fallback": fallback})
        return predictions
    except torch.cuda.OutOfMemoryError:
        logger.exception("Insufficient GPU memory")
        raise HTTPException(503, "Insufficient GPU memory. Try again later.") from None
    except Exception:
        logger.exception("Paper prediction failed")
        raise HTTPException(500, "Prediction failed. Check server logs.") from None


@app.post("/predict")
def predict(paper: PaperRequest) -> dict[str, Any]:
    """Trả nhãn đạt ngưỡng 0.5 hoặc nhãn có xác suất cao nhất."""
    title = normalize_text(paper.title)
    if not title:
        raise HTTPException(422, "Title must not be blank.")
    return _classify_papers([title])[0]


@app.post("/predict/expert", response_model=ExpertResponse)
def predict_expert(expert: ExpertRequest) -> dict[str, Any]:
    """Recommend direct and one-step related topics using the configured rules.

    Scores are relative paper counts for direct topics, and source score times
    relationship weight for related topics. Publications are counted once per
    selected label, including the existing classifier's highest-score fallback.
    """
    rules = getattr(app.state, "expert_rules", None)
    if rules is None:
        raise HTTPException(503, "Expert rules unavailable. Check configuration and server logs.")
    papers = _classify_papers(expert.titles)
    try:
        result = recommend_expertise([[label["id"] for label in paper["labels"]] for paper in papers], rules)
        topics = result["direct_topics"] + result["related_topics"]
        topics.sort(key=lambda topic: (-topic["expertise_score"], topic["topic_name"]))
        return {"publication_count": result["publication_count"], "topics": topics}
    except Exception:
        logger.exception("Expert prediction failed")
        raise HTTPException(500, "Expert prediction failed. Check server logs.") from None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
