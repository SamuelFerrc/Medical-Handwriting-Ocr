---
tipo: escrita
status: rascunho
tags:
  - artigo
  - introducao
---

# Introducao

## Pontos a desenvolver

- Importancia da digitalizacao de documentos medicos.
- Desafio especifico de prescricoes manuscritas.
- Limites de OCR tradicional em caligrafia variavel.
- Avanco de CNNs, CRNNs e Transformers em OCR/HTR.
- Necessidade de comparacao controlada em datasets de dominio medico.

## Problema

Ainda nao esta claro qual familia de modelos oferece melhor desempenho para reconhecimento textual em imagens medicas manuscritas quando avaliada sob o mesmo protocolo e com metricas diretamente comparaveis.

## Objetivo no texto

Este trabalho compara CNN, CRNN (CNN + BiLSTM), TrOCR e PARSeq nos datasets RxHandBD e Doctor's Handwritten PrescriptionBD, medindo Accuracy, CER e WER.

## Contribuicoes

- Comparacao experimental entre quatro arquiteturas.
- Protocolo padronizado de avaliacao.
- Analise dos erros por caractere, palavra e exemplos qualitativos.
