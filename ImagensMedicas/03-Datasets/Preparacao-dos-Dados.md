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

## Formato padrao sugerido

| coluna | descricao |
|---|---|
| sample_id | Identificador unico da amostra |
| dataset | RxHandBD ou Doctor's Handwritten PrescriptionBD |
| image_path | Caminho relativo da imagem |
| text_raw | Transcricao original |
| text_normalized | Transcricao apos normalizacao |
| split | train, val ou test |
| writer_id | Identificador do escritor, se existir |
| document_id | Identificador do documento, se existir |
| notes | Observacoes de auditoria |

## Etapas

- [ ] Baixar datasets e registrar fonte.
- [ ] Conferir licencas e permissao de uso academico.
- [ ] Converter anotacoes para tabela unica.
- [ ] Normalizar caminhos de imagem.
- [ ] Remover ou marcar amostras sem rotulo.
- [ ] Verificar duplicidades.
- [ ] Definir normalizacao textual.
- [ ] Criar splits fixos.
- [ ] Salvar estatisticas de dados.

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
