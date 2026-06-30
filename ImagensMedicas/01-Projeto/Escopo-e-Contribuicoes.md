---
tipo: projeto
status: rascunho
tags:
  - escopo
  - contribuicoes
---

# Escopo e Contribuicoes

## Dentro do escopo

- Reconhecimento de texto em imagens medicas/prescricoes.
- Comparacao de quatro metodos: CNN, CRNN, TrOCR e PARSeq.
- Avaliacao quantitativa com Accuracy, CER e WER.
- Analise qualitativa dos erros.
- Documentacao de pipeline reprodutivel.

## Fora do escopo

- Diagnostico clinico automatizado.
- Interpretacao farmacologica completa da prescricao.
- Validacao regulatoria ou estudo com usuarios em ambiente hospitalar.
- Correcao automatica usando conhecimento medico externo, a menos que seja registrada como experimento separado.

## Possiveis contribuicoes cientificas

- Evidencia comparativa entre arquiteturas em bases de prescricoes manuscritas.
- Protocolo replicavel para avaliacao OCR/HTR em documentos medicos.
- Discussao sobre impacto de normalizacao textual e composicao de dataset.
- Mapeamento dos erros que mais afetam CER/WER.

## Ameacas a validade

- Datasets pequenos ou desbalanceados.
- Vazamento por escritor, imagem duplicada ou documento parcialmente replicado.
- Diferencas entre datasets que prejudiquem comparacao direta.
- Rotulos inconsistentes, abreviados ou incompletos.
- Dependencia de hiperparametros e pesos pre-treinados.
