---
tipo: modelo
status: rascunho
tags:
  - modelo
  - cnn
---

# CNN

## Papel no estudo

Baseline visual para comparar com modelos sequenciais e Transformers.

## Observacao metodologica

Uma CNN pura e naturalmente adequada para classificacao de classes fixas. Para reconhecimento de texto livre, e preciso definir como a CNN gerara sequencias ou restringir a tarefa a unidades como palavra/token. Esta decisao deve ser registrada antes dos experimentos.

## Possiveis configuracoes

| Configuracao | Descricao | Quando usar |
|---|---|---|
| CNN classificadora | Classifica a imagem em uma palavra/classe | Se a tarefa for token/palavra com vocabulario fechado |
| CNN encoder + cabeca sequencial simples | Extrai features e prediz caracteres em posicoes fixas | Se houver comprimento maximo controlado |
| CNN como baseline de features | Usa CNN para extracao e decodificador simples | Se quiser baseline minimo comparavel |

## Hiperparametros a registrar

- Arquitetura base.
- Resolucao de entrada.
- Numero de camadas/blocos.
- Tamanho do vocabulario.
- Loss.
- Otimizador.
- Learning rate.
- Batch size.
- Epochs.

## Avaliacao

Usar as mesmas metricas:

- Accuracy de texto completo ou classe.
- CER.
- WER.

## Riscos

- Comparacao injusta se CNN for usada para uma tarefa mais simples que os demais modelos.
- Baixo desempenho esperado em sequencias longas.
- Vocabulario fechado pode mascarar dificuldade real de OCR/HTR.
