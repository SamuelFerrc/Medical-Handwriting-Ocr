---
tipo: metodologia
status: rascunho
tags:
  - dados
  - preprocessamento
---

# Preparacao dos Dados

## Objetivo

Transformar RxHandBD e Doctor's Handwritten PrescriptionBD em um formato unico, auditavel e reproduzivel para treinamento e avaliacao dos modelos.

## Etapas

- [x] Baixar datasets e registrar fonte.
- [x] Conferir licencas e permissao de uso academico.
- [x] Converter anotacoes para tabela unica.
- [x] Normalizar caminhos de imagem.
- [x] Remover ou marcar amostras sem rotulo.
- [x] Verificar duplicidades.
- [x] Definir normalizacao textual.
- [x] Criar splits fixos.
- [x] Salvar estatisticas de dados.

## Normalizacao textual inicial

Definir e manter igual para todos os modelos:

- Converter para lowercase? Sim / Nao.
- Remover acentos? Sim / Nao.
- Remover pontuacao? Sim / Nao.
- Padronizar espacos multiplos? Sim.
- Manter numeros e unidades? Sim.
- Manter barras, hifens e simbolos de dose? A decidir.

## Estatisticas a gerar

- Numero de amostras por dataset e split.
- Comprimento medio do texto em caracteres.
- Comprimento medio em palavras.
- Distribuicao de resolucao das imagens.
- Frequencia de caracteres.
- Percentual de amostras vazias ou invalidas.
