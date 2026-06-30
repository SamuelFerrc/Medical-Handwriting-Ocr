---
tipo: metodologia
status: rascunho
tags:
  - augmentation
---

# Aumento de Dados

## Objetivo

Melhorar generalizacao dos modelos simulando variacoes comuns em documentos manuscritos.

## Transformacoes candidatas

- Rotacao leve.
- Translacao leve.
- Alteracao de brilho/contraste.
- Ruido gaussiano leve.
- Desfoque leve.
- Elastic transform com cautela.
- Perspectiva leve.
- Crop/padding controlado.

## Regras

- Aplicar apenas no treino.
- Nunca alterar validacao e teste.
- Manter o texto semanticamente legivel.
- Usar a mesma politica entre modelos quando possivel.
- Registrar parametros em cada experimento.

## Matriz de decisao

| Transformacao | Usar? | Intervalo | Observacao |
|---|---|---|---|
| Rotacao | A definir | -3 a +3 graus | Evitar deformar linhas |
| Brilho/contraste | A definir | Leve | Simula scanner/camera |
| Ruido | A definir | Baixo | Nao apagar tracos |
| Blur | A definir | Baixo | Simula foco ruim |
| Perspectiva | A definir | Baixo | Usar com cautela |

## Experimentos sugeridos

- Sem augmentation.
- Augmentation leve.
- Augmentation moderada.

Comparar impacto em CER/WER, nao apenas em loss de treino.
