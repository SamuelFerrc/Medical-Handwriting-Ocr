# Pesquisa: CNNs para reconhecimento em imagens medicas

Este vault organiza o dimensionamento completo da pesquisa sobre comparacao de modelos para reconhecimento de texto manuscrito em imagens medicas/prescricoes, usando os datasets RxHandBD e Doctor's Handwritten PrescriptionBD.

## Entrada principal

- [[00-Visao-Geral/Mapa-da-Pesquisa|Mapa da Pesquisa]]
- [[00-Visao-Geral/Mapa-Visual.canvas|Mapa Visual]]
- [[01-Projeto/Resumo-Executivo|Resumo Executivo]]
- [[04-Metodologia/Desenho-Experimental|Desenho Experimental]]
- [[06-Experimentos/Matriz-de-Experimentos|Matriz de Experimentos]]
- [[08-Resultados/Tabela-Comparativa|Tabela Comparativa]]

## Fluxo sugerido

1. Definir o problema, objetivos e hipoteses em [[01-Projeto/Pergunta-de-Pesquisa-e-Hipoteses|Pergunta de Pesquisa e Hipoteses]].
2. Validar datasets, rotulos e particionamento em [[03-Datasets/Preparacao-dos-Dados|Preparacao dos Dados]].
3. Executar os modelos: [[05-Modelos/CNN]], [[05-Modelos/CRNN-CNN-BiLSTM]], [[05-Modelos/TrOCR]] e [[05-Modelos/PARSeq]].
4. Avaliar com [[07-Metricas/Acuracia-CER-WER|Acuracia, CER e WER]].
5. Consolidar analise em [[08-Resultados/Discussao-dos-Resultados|Discussao dos Resultados]] e escrever o artigo em [[09-Escrita/Estrutura-do-Artigo|Estrutura do Artigo]].
