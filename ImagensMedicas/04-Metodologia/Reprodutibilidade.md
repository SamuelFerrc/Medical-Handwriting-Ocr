---
tipo: metodologia
status: rascunho
tags:
  - reproducibilidade
---

# Reprodutibilidade

## Itens obrigatorios

- Codigo versionado.
- Seeds registradas.
- Versoes das bibliotecas salvas.
- Ambiente documentado.
- Splits salvos.
- Configuracoes de treino versionadas.
- Scripts de avaliacao unicos para todos os modelos.
- Predicoes brutas armazenadas.

## Ambiente

| Item | Valor |
|---|---|
| Sistema operacional | A preencher |
| GPU | A preencher |
| CPU | A preencher |
| RAM | A preencher |
| Python | A preencher |
| PyTorch/TensorFlow | A preencher |
| CUDA | A preencher |

## Estrutura sugerida para codigo

```text
data/
  raw/
  processed/
  splits/
configs/
src/
  data/
  models/
  train/
  eval/
outputs/
  runs/
  predictions/
  metrics/
```

## Checklist

- [ ] Rodar uma avaliacao completa do zero usando apenas README/config.
- [ ] Conferir se metricas sao reproduzidas.
- [ ] Salvar predicoes usadas na tabela final.
