---
tipo: metodologia
status: rascunho
tags:
  - metodologia
  - experimento
---

# Desenho Experimental

## Tipo de estudo

Estudo experimental comparativo entre arquiteturas de reconhecimento de texto em imagens medicas/prescricoes.

## Variavel independente

Arquitetura/modelo utilizado:

- CNN.
- CRNN (CNN + BiLSTM).
- TrOCR.
- PARSeq.

## Variaveis dependentes

- Accuracy.
- CER.
- WER.
- Tempo de treinamento, se medido.
- Tempo de inferencia, se medido.
- Consumo de memoria, se medido.

## Controles

Para garantir comparacao justa:

- Mesmos splits de treino, validacao e teste.
- Mesmo normalizador textual.
- Mesma politica de preprocessamento, quando compativel com o modelo.
- Mesmo criterio de early stopping.
- Mesmo conjunto de metricas.
- Mesmo ambiente computacional ou ambiente documentado.

## Cenarios

| Cenario | Treino | Teste | Objetivo |
|---|---|---|---|
| A | RxHandBD | RxHandBD | Desempenho in-domain |
| B | PrescriptionBD | PrescriptionBD | Desempenho in-domain |
| C | RxHandBD + PrescriptionBD | Combinado | Beneficio do treinamento combinado |
| D opcional | RxHandBD | PrescriptionBD | Transferencia de dominio |
| E opcional | PrescriptionBD | RxHandBD | Transferencia de dominio |

## Saida esperada

Uma tabela comparativa com linhas por modelo e cenario, contendo Accuracy, CER e WER, alem de observacoes sobre estabilidade, erros e custo computacional.
