---
tipo: metricas
status: rascunho
tags:
  - analise
  - estatistica
---

# Plano de Analise Estatistica

## Objetivo

Verificar se as diferencas observadas entre modelos sao consistentes e nao apenas variacao aleatoria.

## Analises minimas

- Reportar Accuracy, CER e WER por modelo e cenario.
- Reportar media e desvio padrao quando houver multiplas seeds.
- Reportar intervalo de confianca por bootstrap, se possivel.
- Comparar desempenho por dataset.

## Bootstrap sugerido

1. Amostrar o conjunto de teste com reposicao.
2. Recalcular Accuracy, CER e WER.
3. Repetir 1.000 vezes.
4. Estimar intervalo de confianca de 95%.

## Comparacoes

| Comparacao | Objetivo |
|---|---|
| CNN vs CRNN | Medir ganho de modelagem sequencial |
| CRNN vs TrOCR | Comparar sequencial classico com Transformer |
| CRNN vs PARSeq | Comparar sequencial classico com Transformer |
| TrOCR vs PARSeq | Comparar Transformers |

## Relato no artigo

Se nao houver teste estatistico formal, indicar que a conclusao se baseia em diferenca observada e analise qualitativa, evitando afirmar significancia estatistica.
