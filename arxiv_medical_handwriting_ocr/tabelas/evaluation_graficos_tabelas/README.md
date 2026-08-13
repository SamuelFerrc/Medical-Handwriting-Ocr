# Evaluation artifacts

Images: `arxiv_medical_handwriting_ocr\imagens\evaluation_graficos_tabelas`

LaTeX tables: `arxiv_medical_handwriting_ocr\tabelas\evaluation_graficos_tabelas`

The 24 CSV files from `SourceCode/outputs/metrics/*_evaluation/` were processed:

- `evaluated_predictions.csv`: one complete LaTeX table and one chart dashboard per evaluation.
- `per_label_metrics.csv`: one complete LaTeX table and one chart dashboard per evaluation.

Accuracy is the mean of `exact_match`. CER is the sum of `char_errors` divided by the sum of `ref_chars`. WER is the sum of `word_errors` divided by the sum of `ref_words`. Per-sample CER/WER can exceed 1 when insertions occur.

Per-label charts show readable subsets; the `*_per_label_metrics.tex` tables preserve all rows. Long tables require `longtable`; raw prediction tables use `pdflscape` and `url`. `validation_metrics_recalculation.csv` records the difference between metrics recalculated from the CSVs and `metrics.json`.
