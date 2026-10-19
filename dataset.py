import random
import re
import unicodedata
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LABEL_COLUMNS = [
    "subfield_id",
    "other_subfield_id_1",
    "other_subfield_id_2",
    "other_subfield_id_3",
    "other_subfield_id_4",
    "other_subfield_id_5",
    "other_subfield_id_6",
]

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
STEMMER = PorterStemmer()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_papers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"title", "subfield_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).map(normalize_text)
    df["subfield_id"] = pd.to_numeric(df["subfield_id"], errors="raise").astype(int)
    return df


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without removing scientific symbols."""
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def tokenize_title(text: str) -> list[str]:
    """Lowercase, tokenize, remove English stopwords, then apply Porter stemming."""
    words = TOKEN_PATTERN.findall(normalize_text(text).lower())
    return [STEMMER.stem(word) for word in words if word not in ENGLISH_STOP_WORDS]


def extract_titles(df: pd.DataFrame) -> np.ndarray:
    """Return normalized titles used as the only model input."""
    return df["title"].to_numpy()


def load_multilabel_data(
    train_csv_path: str,
    test_csv_path: str,
    taxonomy_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """Load normalized texts and multi-hot targets for classical classifiers."""
    train_df = load_papers(train_csv_path)
    test_df = load_papers(test_csv_path)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    label2id, id2label = build_label_mapping(all_df, taxonomy_path)
    train_df = add_multihot_labels(train_df, label2id)
    test_df = add_multihot_labels(test_df, label2id)

    x_train = extract_titles(train_df)
    x_test = extract_titles(test_df)
    y_train = np.stack(train_df["labels"].to_numpy()).astype(np.int32)
    y_test = np.stack(test_df["labels"].to_numpy()).astype(np.int32)
    return x_train, x_test, y_train, y_test, label2id, id2label


def build_label_mapping(
    papers_df: pd.DataFrame,
    taxonomy_path: Optional[str] = None,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    if taxonomy_path:
        taxonomy_df = pd.read_csv(taxonomy_path)
        if "subfield_id" not in taxonomy_df.columns:
            raise ValueError(f"Missing subfield_id column in {taxonomy_path}")
        label_ids = (
            pd.to_numeric(taxonomy_df["subfield_id"], errors="coerce")
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
    else:
        label_ids = []

    seen = set(label_ids)
    for col in LABEL_COLUMNS:
        if col not in papers_df.columns:
            continue
        values = (
            pd.to_numeric(papers_df[col], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        for value in values:
            if value not in seen:
                label_ids.append(value)
                seen.add(value)

    label_ids = sorted(label_ids)
    label2id = {label: index for index, label in enumerate(label_ids)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def row_to_labels(row: pd.Series, label2id: Dict[int, int]) -> np.ndarray:
    target = np.zeros(len(label2id), dtype=np.float32)
    for col in LABEL_COLUMNS:
        if col not in row.index or pd.isna(row[col]):
            continue
        try:
            label = int(float(row[col]))
        except (TypeError, ValueError):
            continue
        if label in label2id:
            target[label2id[label]] = 1.0
    return target


def add_multihot_labels(df: pd.DataFrame, label2id: Dict[int, int]) -> pd.DataFrame:
    df = df.copy()
    df["labels"] = [row_to_labels(row, label2id) for _, row in df.iterrows()]
    return df


def split_by_primary_topic(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in df.groupby("subfield_id", sort=True):
        if len(group) < 2:
            train_parts.append(group)
            continue

        train_group, test_group = train_test_split(
            group,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        train_parts.append(train_group)
        test_parts.append(test_group)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Keep duplicate titles out of both splits at the same time to avoid leakage.
    key_cols = ["title"]
    train_keys = set(map(tuple, train_df[key_cols].astype(str).to_numpy()))
    test_keys = set(map(tuple, test_df[key_cols].astype(str).to_numpy()))
    overlapping_keys = train_keys.intersection(test_keys)
    if overlapping_keys:
        test_key_series = list(map(tuple, test_df[key_cols].astype(str).to_numpy()))
        overlap_mask = pd.Series(
            [key in overlapping_keys for key in test_key_series],
            index=test_df.index,
        )
        train_df = pd.concat([train_df, test_df[overlap_mask]], ignore_index=True)
        test_df = test_df[~overlap_mask].copy()
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


class PapersDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        encoding = self.tokenizer(
            row["title"],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(row["labels"], dtype=torch.float32)
        return item


def create_dataloaders(
    train_csv_path: str,
    test_csv_path: str,
    tokenizer,
    taxonomy_path: Optional[str] = None,
    max_length: int = 256,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], Dict[int, int]]:
    train_df = load_papers(train_csv_path)
    test_df = load_papers(test_csv_path)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    label2id, id2label = build_label_mapping(all_df, taxonomy_path)
    train_df = add_multihot_labels(train_df, label2id)
    test_df = add_multihot_labels(test_df, label2id)

    train_dataset = PapersDataset(train_df, tokenizer=tokenizer, max_length=max_length)
    test_dataset = PapersDataset(test_df, tokenizer=tokenizer, max_length=max_length)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, label2id, id2label
