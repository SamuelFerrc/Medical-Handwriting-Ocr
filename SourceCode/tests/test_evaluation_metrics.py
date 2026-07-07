from med_ocr_research.evaluation.metrics import (
    aggregate_metrics,
    character_error_rate,
    exact_match,
    word_error_rate,
)


def test_exact_match_accuracy_cer_and_wer_for_word_predictions() -> None:
    references = ["amoxicilina", "dipirona", "500 mg"]
    hypotheses = ["amoxicilina", "diprona", "500mg"]

    metrics = aggregate_metrics(references, hypotheses)

    assert metrics.samples == 3
    assert metrics.correct == 1
    assert metrics.exact_match_accuracy == 1 / 3
    assert metrics.total_char_errors == 2
    assert metrics.total_reference_chars == 25
    assert metrics.cer == 2 / 25
    assert metrics.total_word_errors == 3
    assert metrics.total_reference_words == 4
    assert metrics.wer == 3 / 4


def test_individual_metric_helpers() -> None:
    assert exact_match("abc", "abc") is True
    assert exact_match("abc", "abd") is False
    assert character_error_rate("abc", "abd") == 1 / 3
    assert word_error_rate("duas palavras", "duas") == 1 / 2
