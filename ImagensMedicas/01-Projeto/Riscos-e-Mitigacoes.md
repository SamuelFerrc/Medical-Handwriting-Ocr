---
tipo: planejamento
status: rascunho
tags:
  - riscos
---

# Riscos e Mitigacoes

| Risco | Impacto | Probabilidade | Mitigacao |
|---|---:|---:|---|
| Rotulos inconsistentes | Alto | Media | Criar regras de normalizacao e amostrar casos manualmente |
| Datasets pequenos | Alto | Media | Usar pesos pre-treinados, aumento de dados e validacao cuidadosa |
| Vazamento entre treino e teste | Alto | Media | Separar por escritor/documento quando houver metadados |
| CNN baseline inadequada para sequencias | Medio | Alta | Definir claramente a granularidade ou usar CNN como encoder baseline |
| Diferencas fortes entre datasets | Medio | Media | Reportar resultados separados e combinados |
| Treinamento Transformer caro | Medio | Media | Usar fine-tuning, early stopping e batches ajustados |
| OCR confundir abreviacoes medicas | Alto | Alta | Fazer analise qualitativa e discutir risco clinico |

## Plano de contingencia

- Se um modelo nao convergir, registrar configuracoes testadas e usar checkpoint pre-treinado mais estavel.
- Se a base combinada degradar desempenho, reportar datasets separadamente como resultado principal.
- Se faltarem rotulos por linha/palavra, adaptar a tarefa para o nivel de granularidade disponivel.
