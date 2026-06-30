---
tipo: dataset
status: rascunho
tags:
  - dicionario
  - dados
---

# Dicionario de Dados

## Tabela principal de amostras

| Campo | Tipo | Obrigatorio | Descricao |
|---|---|---:|---|
| sample_id | string | Sim | Identificador unico e estavel da amostra |
| dataset | string | Sim | Nome da base de origem |
| image_path | string | Sim | Caminho da imagem |
| text_raw | string | Sim | Rotulo original sem normalizacao |
| text_normalized | string | Sim | Rotulo usado nas metricas |
| split | string | Sim | train, val ou test |
| writer_id | string | Nao | Autor/escritor, se disponivel |
| document_id | string | Nao | Documento original, se disponivel |
| width | integer | Nao | Largura da imagem em pixels |
| height | integer | Nao | Altura da imagem em pixels |
| notes | string | Nao | Observacoes de qualidade |

## Regras de identificacao

- `sample_id` deve ser unico mesmo apos combinar datasets.
- `image_path` deve ser relativo ao diretorio raiz do projeto de dados.
- `text_raw` nunca deve ser sobrescrito.
- `text_normalized` deve ser gerado por script versionado.

## Controle de qualidade

- [ ] Nenhum `sample_id` duplicado.
- [ ] Nenhuma imagem ausente.
- [ ] Nenhum rotulo nulo no conjunto usado para treino/teste.
- [ ] Splits sem sobreposicao.
- [ ] Metadados de origem preservados.
