---
tipo: metodologia
status: rascunho
tags:
  - preprocessamento
  - imagens
---

# Pre-processamento

## Objetivo

Padronizar as imagens para reduzir variabilidade irrelevante sem apagar informacao textual importante.

## Operacoes candidatas

- Conversao para RGB ou escala de cinza.
- Correcao de orientacao.
- Redimensionamento preservando proporcao.
- Padding para dimensoes fixas.
- Normalizacao de intensidade.
- Remocao leve de ruido.
- Binarizacao, apenas se melhorar empiricamente.
- Correcao de inclinacao, se necessaria.

## Cuidados

- Evitar preprocessamentos que apaguem tracos finos.
- Nao aplicar operacoes diferentes por modelo sem registrar justificativa.
- Manter copia dos dados originais.
- Validar visualmente uma amostra apos cada transformacao.

## Parametros a definir

| Parametro | Valor candidato | Decisao |
|---|---|---|
| Altura padrao | 32, 48, 64 ou 128 px | A definir |
| Largura maxima | Dependente do modelo | A definir |
| Canal | Grayscale ou RGB | A definir |
| Padding | Branco ou valor medio | A definir |
| Normalizacao | ImageNet ou dataset | A definir |

## Checklist

- [ ] Testar preprocessamento em exemplos faceis e dificeis.
- [ ] Verificar se caracteres pequenos continuam legiveis.
- [ ] Documentar dimensoes finais por modelo.
- [ ] Salvar script de preprocessamento.
