---
tipo: revisao
status: rascunho
tags:
  - conceitos
  - ocr
---

# Conceitos: OCR em Imagens Medicas

## Problema

O reconhecimento de texto em prescricoes manuscritas combina dificuldades de OCR, reconhecimento de escrita manual e dominio medico. O texto pode conter abreviacoes, unidades, nomes de medicamentos, posologias e rasuras.

## Desafios tecnicos

- Caligrafia altamente variavel.
- Baixa resolucao, sombras, inclinacao e cortes.
- Linhas sobrepostas ou segmentacao inconsistente.
- Vocabulário medico especializado.
- Rotulos com abreviacoes e inconsistencias.
- Ambiguidade entre caracteres parecidos.

## Nivel da tarefa

Definir a unidade de reconhecimento e fundamental:

- Imagem inteira da prescricao.
- Linha de texto.
- Palavra/token.
- Campo especifico, como medicamento ou dose.

O nivel escolhido afeta arquitetura, preprocessamento e interpretacao de Accuracy, CER e WER.

## Implicacao para os modelos

- CNN isolada pode ser util como baseline de classificacao quando a saida for uma classe/token.
- CRNN e adequada para sequencias com comprimento variavel.
- TrOCR e PARSeq tendem a ser mais fortes quando ha pre-treinamento e fine-tuning adequado.

## Ponto critico

Uma pequena diferenca de CER pode esconder erros graves se envolver nome de medicamento, dose ou frequencia. Por isso a analise qualitativa deve acompanhar a tabela numerica.
