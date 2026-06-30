---
tipo: projeto
status: rascunho
tags:
  - pergunta
  - hipoteses
---

# Pergunta de Pesquisa e Hipoteses

## Pergunta principal

Qual arquitetura entre CNN, CRNN (CNN + BiLSTM), TrOCR e PARSeq apresenta melhor desempenho no reconhecimento de texto manuscrito em imagens medicas/prescricoes dos datasets RxHandBD e Doctor's Handwritten PrescriptionBD, considerando Accuracy, CER e WER?

## Perguntas secundarias

1. O uso de modelos sequenciais ou Transformer reduz CER e WER em relacao a uma CNN baseline?
2. O desempenho muda significativamente entre RxHandBD e Doctor's Handwritten PrescriptionBD?
3. A combinacao dos datasets melhora a generalizacao ou introduz degradacao por diferencas de dominio?
4. Quais classes de erro sao mais frequentes: substituicao, insercao, delecao, abreviacoes, segmentacao ou caracteres ambiguos?
5. O melhor modelo quantitativo tambem apresenta erros qualitativamente menos criticos para uso em contexto medico?

## Hipoteses

### H1

Modelos com capacidade sequencial ou encoder-decoder, como CRNN, TrOCR e PARSeq, terao CER e WER menores do que uma CNN baseline.

### H2

TrOCR e PARSeq tendem a superar CRNN quando houver volume suficiente de dados ou quando for possivel aproveitar pesos pre-treinados.

### H3

O treinamento com dados combinados pode melhorar a robustez, mas somente se os rotulos forem padronizados e o particionamento evitar vazamento entre treino e teste.

### H4

WER sera mais sensivel do que CER para erros clinicamente relevantes, pois substituicoes de palavras podem alterar a interpretacao da prescricao.

## Criterio de resposta

A arquitetura recomendada sera aquela que apresentar:

- Maior Accuracy.
- Menor CER.
- Menor WER.
- Boa estabilidade entre datasets.
- Menor incidencia de erros qualitativamente graves.
