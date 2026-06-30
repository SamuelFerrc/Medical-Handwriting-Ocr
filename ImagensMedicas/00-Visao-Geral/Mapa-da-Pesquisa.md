---
tipo: mapa
status: ativo
tags:
  - pesquisa
  - ocr
  - imagens-medicas
---

# Mapa da Pesquisa

## Tema

Comparacao de diferentes arquiteturas de reconhecimento de texto em imagens medicas, com foco em prescricoes manuscritas e bases RxHandBD + Doctor's Handwritten PrescriptionBD.

## Pergunta central

Entre CNN, CRNN (CNN + BiLSTM), TrOCR e PARSeq, qual metodo apresenta melhor desempenho para reconhecimento de texto em imagens medicas/prescricoes considerando Accuracy, CER e WER?

## Notas principais

- Projeto: [[01-Projeto/Resumo-Executivo|Resumo Executivo]]
- Problema e hipoteses: [[01-Projeto/Pergunta-de-Pesquisa-e-Hipoteses|Pergunta de Pesquisa e Hipoteses]]
- Objetivos: [[01-Projeto/Objetivos|Objetivos]]
- Datasets: [[03-Datasets/RxHandBD|RxHandBD]] e [[03-Datasets/Doctors-Handwritten-PrescriptionBD|Doctor's Handwritten PrescriptionBD]]
- Metodologia: [[04-Metodologia/Desenho-Experimental|Desenho Experimental]]
- Modelos: [[05-Modelos/CNN|CNN]], [[05-Modelos/CRNN-CNN-BiLSTM|CRNN]], [[05-Modelos/TrOCR|TrOCR]], [[05-Modelos/PARSeq|PARSeq]]
- Metricas: [[07-Metricas/Acuracia-CER-WER|Acuracia, CER e WER]]
- Resultados: [[08-Resultados/Tabela-Comparativa|Tabela Comparativa]]
- Artigo: [[09-Escrita/Estrutura-do-Artigo|Estrutura do Artigo]]

## Entregaveis da pesquisa

- Protocolo experimental reprodutivel.
- Pipeline padronizado de preprocessamento.
- Treinamento/fine-tuning dos quatro metodos.
- Comparacao quantitativa por dataset e no conjunto combinado.
- Analise qualitativa dos erros mais recorrentes.
- Artigo cientifico com resultados, limitacoes e conclusoes.

## Decisoes pendentes

- [ ] Confirmar se a tarefa sera reconhecimento de linha, palavra, token ou texto completo da prescricao.
- [ ] Confirmar se os datasets possuem transcricoes alinhadas por imagem, linha ou palavra.
- [ ] Definir estrategia de divisao evitando vazamento por medico/escritor/paciente/documento.
- [ ] Escolher idioma e normalizacao textual final.
- [ ] Definir hardware disponivel para treinamento.
