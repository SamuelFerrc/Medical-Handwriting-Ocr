---
tipo: projeto
status: rascunho
tags:
  - projeto
  - resumo
---

# Resumo Executivo

## Titulo provisorio

Comparacao de arquiteturas CNN, CRNN, TrOCR e PARSeq para reconhecimento de texto manuscrito em imagens medicas.

## Contexto

Prescricoes e outros documentos medicos manuscritos apresentam grande variabilidade de caligrafia, ruido visual, abreviacoes e termos tecnicos. Essa variabilidade dificulta a digitalizacao automatica e torna relevante comparar abordagens classicas e modernas de reconhecimento de texto.

## Proposta

Avaliar quatro metodos de reconhecimento em imagens medicas/prescricoes:

- CNN como baseline visual.
- CRNN com CNN + BiLSTM para sequencias.
- TrOCR como abordagem Transformer encoder-decoder.
- PARSeq como abordagem Transformer para reconhecimento de texto.

Os modelos serao avaliados nos datasets RxHandBD e Doctor's Handwritten PrescriptionBD, separadamente e, se metodologicamente valido, em um conjunto combinado.

## Resultado esperado

Identificar qual metodo oferece melhor desempenho geral e melhor equilibrio entre Accuracy, CER e WER, descrevendo tambem os tipos de erro mais frequentes e as limitacoes do protocolo.

## Contribuicoes esperadas

- Comparacao controlada entre quatro familias de modelos.
- Pipeline padronizado de preprocessamento e avaliacao.
- Analise quantitativa e qualitativa em dados de prescricoes manuscritas.
- Discussao sobre viabilidade de uso em contexto medico.

## Fronteiras do estudo

- O estudo compara desempenho de reconhecimento textual, nao faz validacao clinica.
- A avaliacao depende da qualidade e granularidade dos rotulos disponiveis.
- A pesquisa nao substitui revisao profissional de prescricoes.
