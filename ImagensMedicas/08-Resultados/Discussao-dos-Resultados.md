---
tipo: resultado
status: rascunho
tags:
  - discussao
---

# Discussao dos Resultados

## Roteiro de interpretacao

1. Qual modelo teve melhor Accuracy?
2. Qual modelo teve menor CER?
3. Qual modelo teve menor WER?
4. O ranking muda entre os datasets?
5. O conjunto combinado melhorou ou piorou o desempenho?
6. O melhor modelo numerico comete menos erros graves?
7. O ganho justifica custo computacional maior?

## Comparacao esperada

- CNN deve funcionar como referencia minima.
- CRNN deve capturar melhor a natureza sequencial da escrita.
- TrOCR pode se beneficiar de pre-treinamento e modelagem textual.
- PARSeq pode ser competitivo para reconhecimento de palavras/linhas.

## Pontos a discutir

- Impacto da qualidade dos dados.
- Efeito da normalizacao textual.
- Erros em nomes de medicamentos e doses.
- Diferenças entre CER e WER.
- Limitacoes de generalizacao.
- Implicacoes para uso real.

## Frases a completar

- "O modelo com melhor desempenho geral foi..."
- "Apesar de apresentar menor CER, o modelo..."
- "A principal fonte de erro observada foi..."
- "Os resultados sugerem que..."
