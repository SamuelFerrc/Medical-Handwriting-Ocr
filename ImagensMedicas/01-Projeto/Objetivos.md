---
tipo: projeto
status: rascunho
tags:
  - objetivos
---

# Objetivos

## Objetivo geral

Comparar o desempenho de CNN, CRNN (CNN + BiLSTM), TrOCR e PARSeq no reconhecimento de texto manuscrito em imagens medicas/prescricoes, usando RxHandBD e Doctor's Handwritten PrescriptionBD, medindo Accuracy, CER e WER.

## Objetivos especificos

- Levantar trabalhos relacionados sobre OCR/HTR em prescricoes medicas e documentos manuscritos.
- Auditar os datasets quanto a formato, qualidade das imagens, transcricoes, duplicidades e granularidade dos rotulos.
- Definir uma estrategia unica de preprocessamento, normalizacao textual e particionamento.
- Implementar ou adaptar os quatro metodos comparados.
- Treinar e avaliar todos os modelos sob protocolo controlado.
- Calcular Accuracy, CER e WER para cada metodo, dataset e cenario experimental.
- Realizar analise qualitativa dos erros.
- Elaborar conclusao sobre melhor metodo e condicoes de aplicacao.

## Criterios de sucesso

- Todos os modelos avaliados no mesmo conjunto de teste.
- Splits reproduziveis e armazenados.
- Metricas calculadas com o mesmo normalizador textual.
- Resultados apresentados por dataset e por modelo.
- Limitacoes documentadas de forma explicita.
