---
tipo: metodologia
status: rascunho
tags:
  - treinamento
---

# Protocolo de Treinamento

## Objetivo

Definir regras comuns para treinar todos os modelos de forma comparavel.

## Configuracoes comuns

| Item | Valor |
|---|---|
| Seed | A definir |
| Otimizador | AdamW ou Adam |
| Scheduler | A definir |
| Batch size | Ajustado por memoria |
| Early stopping | Com base em CER de validacao |
| Max epochs | A definir |
| Criterio de melhor checkpoint | Menor CER validacao |

## Registro minimo por execucao

- Modelo.
- Dataset/cenario.
- Configuracao de preprocessamento.
- Hiperparametros.
- Tempo de treino.
- Melhor epoca.
- Loss de treino/validacao.
- Accuracy, CER e WER na validacao.
- Accuracy, CER e WER no teste.

## Comparacao justa

- Todos os modelos devem usar o mesmo split.
- O teste final deve ser executado uma unica vez por configuracao escolhida.
- Se houver ajuste fino por modelo, registrar o espaco de busca.
- Modelos pre-treinados devem ser identificados claramente.

## Repeticoes

Idealmente executar cada configuracao com 3 seeds e reportar media e desvio. Se nao for viavel, justificar o uso de uma seed unica.
