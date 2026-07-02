---
tipo: dataset
status: gerado
tags:
  - dados
  - split
  - estatisticas
---

# Estatisticas dos Dados Processados e Splitados

Gerado em: 2026-06-30T22:11:08

## Configuracao

- Seed: 42
- Proporcao alvo para splits novos: 70% treino, 15% validacao, 15% teste.
- Split oficial preservado quando ja havia `train`, `val` e `test` no manifesto.
- Manifesto splitado: `data/processed/WritingProcessed_split.csv`
- Arquivos de split: `data/splits`
- Datasets por cenario: `data/splits/scenarios`
- Mapa operacional dos experimentos: `data/splits/experiment_datasets.csv`
- Estatisticas em JSON: `outputs/reports/split_statistics.json`

## Metodo por dataset

| dataset | metodo |
| --- | --- |
| PrescriptionBD | split original preservado (train/val/test ja presente) |
| RxHandBD | split 70/15/15 gerado por amostra, seed 757 |

## Amostras por dataset

| dataset | samples |
| --- | --- |
| PrescriptionBD | 4680 |
| RxHandBD | 5578 |

## Amostras por split

| split | samples |
| --- | --- |
| test | 1616 |
| train | 7025 |
| val | 1617 |

## Amostras por dataset e split

| dataset | split | samples |
| --- | --- | --- |
| PrescriptionBD | test | 780 |
| PrescriptionBD | train | 3120 |
| PrescriptionBD | val | 780 |
| RxHandBD | test | 836 |
| RxHandBD | train | 3905 |
| RxHandBD | val | 837 |

## Datasets por cenario experimental

| scenario | split | samples | path |
| --- | --- | --- | --- |
| RxHandBD | all | 5578 | data/splits/scenarios/rxhandbd/all.csv |
| RxHandBD | train | 3905 | data/splits/scenarios/rxhandbd/train.csv |
| RxHandBD | val | 837 | data/splits/scenarios/rxhandbd/val.csv |
| RxHandBD | test | 836 | data/splits/scenarios/rxhandbd/test.csv |
| PrescriptionBD | all | 4680 | data/splits/scenarios/prescriptionbd/all.csv |
| PrescriptionBD | train | 3120 | data/splits/scenarios/prescriptionbd/train.csv |
| PrescriptionBD | val | 780 | data/splits/scenarios/prescriptionbd/val.csv |
| PrescriptionBD | test | 780 | data/splits/scenarios/prescriptionbd/test.csv |
| Combinado | all | 10258 | data/splits/scenarios/combined/all.csv |
| Combinado | train | 7025 | data/splits/scenarios/combined/train.csv |
| Combinado | val | 1617 | data/splits/scenarios/combined/val.csv |
| Combinado | test | 1616 | data/splits/scenarios/combined/test.csv |

## Matriz operacional dos experimentos

| id | model | train_dataset | test_dataset | train_manifest | val_manifest | test_manifest |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | cnn | RxHandBD | RxHandBD | data/splits/scenarios/rxhandbd/train.csv | data/splits/scenarios/rxhandbd/val.csv | data/splits/scenarios/rxhandbd/test.csv |
| E02 | cnn | PrescriptionBD | PrescriptionBD | data/splits/scenarios/prescriptionbd/train.csv | data/splits/scenarios/prescriptionbd/val.csv | data/splits/scenarios/prescriptionbd/test.csv |
| E03 | cnn | Combinado | Combinado | data/splits/scenarios/combined/train.csv | data/splits/scenarios/combined/val.csv | data/splits/scenarios/combined/test.csv |
| E04 | crnn | RxHandBD | RxHandBD | data/splits/scenarios/rxhandbd/train.csv | data/splits/scenarios/rxhandbd/val.csv | data/splits/scenarios/rxhandbd/test.csv |
| E05 | crnn | PrescriptionBD | PrescriptionBD | data/splits/scenarios/prescriptionbd/train.csv | data/splits/scenarios/prescriptionbd/val.csv | data/splits/scenarios/prescriptionbd/test.csv |
| E06 | crnn | Combinado | Combinado | data/splits/scenarios/combined/train.csv | data/splits/scenarios/combined/val.csv | data/splits/scenarios/combined/test.csv |
| E07 | trocr | RxHandBD | RxHandBD | data/splits/scenarios/rxhandbd/train.csv | data/splits/scenarios/rxhandbd/val.csv | data/splits/scenarios/rxhandbd/test.csv |
| E08 | trocr | PrescriptionBD | PrescriptionBD | data/splits/scenarios/prescriptionbd/train.csv | data/splits/scenarios/prescriptionbd/val.csv | data/splits/scenarios/prescriptionbd/test.csv |
| E09 | trocr | Combinado | Combinado | data/splits/scenarios/combined/train.csv | data/splits/scenarios/combined/val.csv | data/splits/scenarios/combined/test.csv |
| E10 | parseq | RxHandBD | RxHandBD | data/splits/scenarios/rxhandbd/train.csv | data/splits/scenarios/rxhandbd/val.csv | data/splits/scenarios/rxhandbd/test.csv |
| E11 | parseq | PrescriptionBD | PrescriptionBD | data/splits/scenarios/prescriptionbd/train.csv | data/splits/scenarios/prescriptionbd/val.csv | data/splits/scenarios/prescriptionbd/test.csv |
| E12 | parseq | Combinado | Combinado | data/splits/scenarios/combined/train.csv | data/splits/scenarios/combined/val.csv | data/splits/scenarios/combined/test.csv |

## Comprimento dos textos normalizados

| dataset | chars_mean | chars_median | chars_min | chars_max | words_mean | words_median | words_min | words_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PrescriptionBD | 6.33 | 6.0 | 2 | 11 | 1.01 | 1.0 | 1 | 2 |
| RxHandBD | 6.81 | 7.0 | 2 | 18 | 1.13 | 1.0 | 1 | 4 |

## Resolucao das imagens processadas

| resolution | samples |
| --- | --- |
| 512x512 | 10258 |

## Controle de qualidade

| metrica | valor |
| --- | --- |
| duplicated_sample_id | 0 |
| empty_text_raw | 0 |
| empty_text_normalized | 0 |
| missing_before_repair | 0 |
| repaired | 0 |
| missing_after_repair | 0 |

## Frequencia de caracteres

| character | count |
| --- | --- |
| a | 6879 |
| e | 6305 |
| o | 5713 |
| i | 5657 |
| n | 5375 |
| l | 4614 |
| t | 4512 |
| r | 3610 |
| c | 2973 |
| x | 2637 |
| m | 2594 |
| d | 2029 |
| s | 1990 |
| f | 1880 |
| p | 1807 |
| u | 1492 |
| b | 1454 |
| z | 1111 |
| v | 842 |
| <space> | 783 |
| k | 601 |
| g | 572 |
| h | 560 |
| y | 503 |
| - | 468 |
| j | 120 |
| 0 | 117 |
| . | 102 |
| w | 69 |
| 2 | 65 |
