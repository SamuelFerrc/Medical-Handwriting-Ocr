---
tipo: metodologia
status: rascunho
tags:
  - split
  - avaliacao
---

# Divisao Treino Validacao Teste

## Objetivo

Criar particionamento justo e reproduzivel para comparar todos os modelos.

## Estrategia preferencial

Quando houver metadados suficientes, dividir por escritor ou documento, evitando que exemplos do mesmo medico/documento aparecam em treino e teste.

## Proporcao sugerida

| Split | Proporcao | Uso |
|---|---:|---|
| Treino | 70% | Ajuste dos pesos |
| Validacao | 15% | Escolha de hiperparametros e early stopping |
| Teste | 15% | Avaliacao final |

Se o dataset ja tiver split oficial, registrar e justificar se sera usado.

## Cenarios de avaliacao

1. Treinar e testar em RxHandBD.
2. Treinar e testar em Doctor's Handwritten PrescriptionBD.
3. Treinar no conjunto combinado e testar em teste combinado.
4. Opcional: treino em um dataset e teste no outro para avaliar transferencia de dominio.

## Regras

- O conjunto de teste nao deve ser usado para escolha de hiperparametros.
- Todos os modelos devem usar exatamente os mesmos splits.
- Salvar a seed e os arquivos de split.
- Documentar qualquer amostra removida.

## Checklist

- [ ] Seed definida.
- [ ] Split salvo em arquivo.
- [ ] Distribuicao por dataset conferida.
- [ ] Distribuicao de comprimento conferida.
- [ ] Duplicidades cruzadas removidas ou documentadas.
