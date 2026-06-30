---
tipo: experimento
status: rascunho
tags:
  - experimentos
  - matriz
---

# Matriz de Experimentos

## Experimentos principais

| ID | Modelo | Dataset treino | Dataset teste | Objetivo | Status |
|---|---|---|---|---|---|
| E01 | CNN | RxHandBD | RxHandBD | Baseline in-domain | Pendente |
| E02 | CNN | PrescriptionBD | PrescriptionBD | Baseline in-domain | Pendente |
| E03 | CNN | Combinado | Combinado | Baseline combinado | Pendente |
| E04 | CRNN | RxHandBD | RxHandBD | Sequencial in-domain | Pendente |
| E05 | CRNN | PrescriptionBD | PrescriptionBD | Sequencial in-domain | Pendente |
| E06 | CRNN | Combinado | Combinado | Sequencial combinado | Pendente |
| E07 | TrOCR | RxHandBD | RxHandBD | Transformer in-domain | Pendente |
| E08 | TrOCR | PrescriptionBD | PrescriptionBD | Transformer in-domain | Pendente |
| E09 | TrOCR | Combinado | Combinado | Transformer combinado | Pendente |
| E10 | PARSeq | RxHandBD | RxHandBD | Transformer in-domain | Pendente |
| E11 | PARSeq | PrescriptionBD | PrescriptionBD | Transformer in-domain | Pendente |
| E12 | PARSeq | Combinado | Combinado | Transformer combinado | Pendente |

## Experimentos opcionais

| ID | Modelo | Treino | Teste | Objetivo |
|---|---|---|---|---|
| O01 | CRNN | RxHandBD | PrescriptionBD | Transferencia de dominio |
| O02 | CRNN | PrescriptionBD | RxHandBD | Transferencia de dominio |
| O03 | TrOCR | RxHandBD | PrescriptionBD | Transferencia de dominio |
| O04 | TrOCR | PrescriptionBD | RxHandBD | Transferencia de dominio |
| O05 | PARSeq | RxHandBD | PrescriptionBD | Transferencia de dominio |
| O06 | PARSeq | PrescriptionBD | RxHandBD | Transferencia de dominio |

## Prioridade

1. Validar pipeline com uma pequena amostra.
2. Executar CNN e CRNN como baselines.
3. Executar TrOCR e PARSeq.
4. Repetir melhores configuracoes com seeds adicionais, se houver tempo.
