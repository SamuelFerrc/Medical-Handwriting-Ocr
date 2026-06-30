---
tipo: metricas
status: rascunho
tags:
  - erros
  - analise-qualitativa
---

# Erros Qualitativos

## Objetivo

Entender que tipos de erro os modelos cometem alem dos valores agregados de Accuracy, CER e WER.

## Categorias de erro

| Categoria | Exemplo | Impacto |
|---|---|---|
| Substituicao de caractere | `m` por `n` | Pode alterar palavra |
| Substituicao de palavra | Medicamento A por B | Alto impacto |
| Delecao | Omitir dose ou frequencia | Alto impacto |
| Insercao | Adicionar termo inexistente | Medio/alto |
| Segmentacao | Juntar ou separar palavras | Afeta WER |
| Abreviacao | Interpretar abreviacao de forma errada | Alto impacto |
| Numeros/unidades | Erro em dose | Alto impacto |

## Amostragem sugerida

- Top 20 piores exemplos por CER.
- Top 20 piores exemplos por WER.
- Exemplos onde CER e baixo, mas erro e clinicamente relevante.
- Exemplos onde modelos discordam entre si.

## Tabela de analise

| sample_id | referencia | CNN | CRNN | TrOCR | PARSeq | categoria | observacao |
|---|---|---|---|---|---|---|---|
| A preencher |  |  |  |  |  |  |  |

## Perguntas para discussao

- O erro altera medicamento, dose ou frequencia?
- O erro vem de caligrafia, baixa qualidade da imagem ou limitacao do modelo?
- O modelo com melhor CER tambem evita erros graves?
