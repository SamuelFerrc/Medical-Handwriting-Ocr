---
tipo: metricas
status: rascunho
tags:
  - metricas
  - accuracy
  - cer
  - wer
---

# Acuracia, CER e WER

## Accuracy

Nesta pesquisa, usar como definicao principal:

```text
Accuracy = numero de predicoes exatamente corretas / numero total de amostras
```

A comparacao deve ser feita apos aplicar a mesma normalizacao ao texto alvo e a predicao.

## CER

Character Error Rate:

```text
CER = distancia_de_edicao_caracteres(predicao, referencia) / numero_de_caracteres_da_referencia
```

Operacoes consideradas: insercao, delecao e substituicao de caracteres.

## WER

Word Error Rate:

```text
WER = distancia_de_edicao_palavras(predicao, referencia) / numero_de_palavras_da_referencia
```

Operacoes consideradas: insercao, delecao e substituicao de palavras.

## Regras de calculo

- Calcular metricas no mesmo conjunto de teste para todos os modelos.
- Usar `text_normalized` como referencia principal.
- Reportar tambem resultados por dataset.
- Manter predicoes brutas para auditoria.
- Documentar se pontuacao, acentos e maiusculas foram removidos.

## Interpretacao

- Accuracy alta indica predicoes inteiramente corretas.
- CER baixo indica poucos erros por caractere.
- WER baixo indica melhor preservacao das palavras.
- Em contexto medico, WER e analise qualitativa podem ser mais relevantes que CER isolado.

## Tabela padrao

| Modelo | Cenario | Accuracy | CER | WER | Melhor criterio |
|---|---|---:|---:|---:|---|
| CNN | A preencher |  |  |  |  |
| CRNN | A preencher |  |  |  |  |
| TrOCR | A preencher |  |  |  |  |
| PARSeq | A preencher |  |  |  |  |
