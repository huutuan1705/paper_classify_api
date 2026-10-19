import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from text2topic_model import Text2TopicModel, normalize_text


model_dir = Path("outputs/text2topic")
metadata = json.loads((model_dir / "metadata.json").read_text())
config = metadata["args"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder", local_files_only=True)
model = Text2TopicModel(
    AutoModel.from_pretrained(model_dir / "encoder", local_files_only=True),
    config["dropout"],
)
model.classifier.load_state_dict(torch.load(
    model_dir / "classifier.pt", map_location="cpu", weights_only=True,
))
model.to(device=device, dtype=dtype).eval()

title = normalize_text(input("Title: "))
title_tokens = tokenizer(title, truncation=True, max_length=config["max_length"], return_tensors="pt")
topic_tokens = tokenizer(
    metadata["topic_texts"], padding=True, truncation=True,
    max_length=config["topic_max_length"], return_tensors="pt",
)
title_tokens = {key: value.to(device) for key, value in title_tokens.items()}
topic_tokens = {key: value.to(device) for key, value in topic_tokens.items()}

with torch.inference_mode():
    scores = model(title_tokens, topic_tokens).float().sigmoid()[0].cpu().tolist()

print(", ".join(name for name, score in zip(metadata["topic_names"], scores)
                if score >= config["threshold"]))
