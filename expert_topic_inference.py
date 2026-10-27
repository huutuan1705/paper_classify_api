"""Aggregate existing paper topic predictions into direct and related expertise.

No model training or external dependencies. Call after the existing classifier.
"""

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def load_expertise_rules(path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Load rules once and validate topic IDs against the model metadata."""
    rules = json.loads(path.read_text(encoding="utf-8"))
    ids, names = metadata["label_ids"], metadata["topic_names"]
    if (not ids or len(ids) != len(names) or any(type(topic) is not int for topic in ids)
            or len(set(ids)) != len(ids)
            or any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("Metadata topic IDs and names must be nonempty, unique, and aligned.")
    expected_names = {str(topic): name for topic, name in zip(ids, names)}
    if not isinstance(rules, dict) or rules.get("topic_names") != expected_names:
        raise ValueError("Expert rule topics must match metadata IDs and names.")
    minimum_papers = rules.get("minimum_papers_for_direct_topic")
    if type(minimum_papers) is not int or minimum_papers < 1:
        raise ValueError("Minimum paper count must be a positive integer.")
    for key in (
        "minimum_relative_frequency", "minimum_source_relative_frequency_for_inference",
        "minimum_relationship_for_inference", "minimum_inferred_score",
    ):
        value = rules.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1):
            raise ValueError(f"{key} must be a finite number in [0, 1].")
    relationships = rules.get("relationships")
    if not isinstance(relationships, dict):
        raise ValueError("Expert relationships must map source topics to weighted targets.")
    for source, targets in relationships.items():
        if source not in expected_names or not isinstance(targets, dict):
            raise ValueError("Expert relationships contain an invalid source topic.")
        for target, weight in targets.items():
            if (target not in expected_names or type(weight) not in (int, float)
                    or not math.isfinite(weight) or not 0 <= weight <= 1):
                raise ValueError("Expert relationships contain an invalid target or weight.")
    return rules


def recommend_expertise(
    paper_topics: Sequence[Iterable[int]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    """Each paper contains the topic IDs already predicted by the classifier.

    Multiple labels on one paper are supported; duplicate labels are ignored.
    A related topic is not treated as proven direct expertise.
    """
    names = rules["topic_names"]
    total = len(paper_topics)
    counts: Counter[str] = Counter()
    for topics in paper_topics:
        unique = {str(topic) for topic in topics}
        invalid = unique - names.keys()
        if invalid:
            raise ValueError(f"Unknown topic IDs: {', '.join(sorted(invalid))}")
        counts.update(unique)

    if not counts:
        return {"publication_count": total, "direct_topics": [], "related_topics": []}

    largest_count = max(counts.values())
    direct: dict[str, dict[str, Any]] = {}
    sources: dict[str, float] = {}

    for topic, count in counts.items():
        relative_frequency = count / largest_count
        if (count < rules["minimum_papers_for_direct_topic"]
                or relative_frequency < rules["minimum_relative_frequency"]):
            continue
        direct[topic] = {
            "topic_id": int(topic),
            "topic_name": names[topic],
            "expertise_type": "direct",
            "paper_count": count,
            "paper_ratio": round(count / total, 4),
            "expertise_score": round(relative_frequency, 4),
        }
        if relative_frequency >= rules["minimum_source_relative_frequency_for_inference"]:
            sources[topic] = relative_frequency

    related: dict[str, dict[str, Any]] = {}
    for source, source_score in sources.items():
        for target, weight in rules["relationships"].get(source, {}).items():
            if target in direct or weight < rules["minimum_relationship_for_inference"]:
                continue
            score = source_score * weight
            if score < rules["minimum_inferred_score"]:
                continue
            if target not in related or score > related[target]["expertise_score"]:
                related[target] = {
                    "topic_id": int(target),
                    "topic_name": names[target],
                    "expertise_type": "related",
                    "paper_count": counts[target],
                    "expertise_score": round(score, 4),
                    "inferred_from": names[source],
                }

    def rank(item: dict[str, Any]) -> tuple[float, str]:
        return (-item["expertise_score"], item["topic_name"])

    return {
        "publication_count": total,
        "direct_topics": sorted(direct.values(), key=rank),
        "related_topics": sorted(related.values(), key=rank),
    }
