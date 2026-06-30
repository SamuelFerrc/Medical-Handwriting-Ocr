---
tipo: modelo
status: rascunho
tags:
  - baseline
  - comparacao
---

# Baseline e Comparacao

## Objetivo

Garantir que a comparacao entre modelos seja metodologicamente justa.

## Baselines possiveis

| Baseline | Funcao | Observacao |
|---|---|---|
| CNN simples | Referencia minima | Pode ser limitada para texto livre |
| CRNN | Baseline sequencial forte | Muito usado em HTR/OCR |
| Modelo pre-treinado sem fine-tuning | Medir transferencia direta | Opcional |

## Regras de comparacao

- Mesmos dados de teste.
- Mesmo texto alvo normalizado.
- Mesmo script de metricas.
- Mesma politica de selecao de checkpoint.
- Reportar quando um modelo usar pre-treinamento externo.

## Tabela de configuracao comparavel

| Modelo | Pre-treinado? | Input | Decoding | Loss | Observacao |
|---|---|---|---|---|---|
| CNN | A definir | A definir | A definir | A definir | Baseline |
| CRNN | A definir | A definir | CTC | CTC | Sequencial |
| TrOCR | Sim/Nao | A definir | Autoregressivo | Cross-entropy | Transformer |
| PARSeq | Sim/Nao | A definir | Modelo especifico | Cross-entropy | Transformer |
