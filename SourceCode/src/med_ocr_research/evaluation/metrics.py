from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Calcula distancia de edicao entre duas strings."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            substitution = previous[hyp_index - 1] + int(ref_char != hyp_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return sequence_edit_distance(reference_words, hypothesis_words) / len(reference_words)


def sequence_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            substitution = previous[hyp_index - 1] + int(ref_token != hyp_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class AggregateMetrics:
    samples: int
    exact_match_accuracy: float
    cer: float
    wer: float
    correct: int
    total_char_errors: int
    total_reference_chars: int
    total_word_errors: int
    total_reference_words: int


def aggregate_metrics(
    references: Iterable[str],
    hypotheses: Iterable[str],
) -> AggregateMetrics:
    refs = [str(value) for value in references]
    hyps = [str(value) for value in hypotheses]
    if len(refs) != len(hyps):
        raise ValueError("references e hypotheses precisam ter o mesmo tamanho.")

    correct = 0
    total_char_errors = 0
    total_reference_chars = 0
    total_word_errors = 0
    total_reference_words = 0

    for reference, hypothesis in zip(refs, hyps):
        correct += int(exact_match(reference, hypothesis))
        total_char_errors += levenshtein_distance(reference, hypothesis)
        total_reference_chars += len(reference)

        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        total_word_errors += sequence_edit_distance(reference_words, hypothesis_words)
        total_reference_words += len(reference_words)

    samples = len(refs)
    return AggregateMetrics(
        samples=samples,
        exact_match_accuracy=correct / samples if samples else 0.0,
        cer=total_char_errors / total_reference_chars if total_reference_chars else 0.0,
        wer=total_word_errors / total_reference_words if total_reference_words else 0.0,
        correct=correct,
        total_char_errors=total_char_errors,
        total_reference_chars=total_reference_chars,
        total_word_errors=total_word_errors,
        total_reference_words=total_reference_words,
    )


def label_distribution(values: Iterable[str], limit: int = 20) -> list[dict[str, int | str]]:
    counter = Counter(str(value) for value in values)
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(limit)
    ]
