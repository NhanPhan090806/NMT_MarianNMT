"""SQuAD 1.1 normalized EM and word-overlap F1 (0-100)."""

import re
import string
from collections import Counter


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0  # matches the SQuAD 1.1 scorer, including two empty strings
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_predictions(predictions, references):
    if not predictions or set(predictions) != set(references):
        raise ValueError("Predictions must cover exactly the nonempty reference ID set.")
    em = f1 = 0.0
    for key, answers in references.items():
        if not answers:
            raise ValueError("SQuAD 1.1 requires at least one reference per example.")
        em += max(exact_match(predictions[key], answer) for answer in answers)
        f1 += max(token_f1(predictions[key], answer) for answer in answers)
    count = len(predictions)
    return {"em": 100 * em / count, "f1": 100 * f1 / count, "examples": count,
            "empty_answer_percent": 100 * sum(not p.strip() for p in predictions.values()) / count}

