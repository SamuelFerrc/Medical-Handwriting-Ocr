---
tipo: referencia
status: rascunho
tags:
  - glossario
---

# Glossario

| Termo | Definicao operacional nesta pesquisa |
|---|---|
| OCR | Reconhecimento optico de caracteres em imagens. |
| HTR | Reconhecimento de texto manuscrito. |
| CNN | Rede neural convolucional usada para extrair caracteristicas visuais. |
| CRNN | Arquitetura que combina CNN para extracao visual e RNN/BiLSTM para modelar sequencias. |
| BiLSTM | LSTM bidirecional, usada para capturar contexto antes e depois de cada posicao da sequencia. |
| CTC | Funcao de perda para alinhar sequencias de entrada e saida sem anotacao caractere-a-caractere. |
| TrOCR | Modelo encoder-decoder baseado em Transformer para OCR. |
| PARSeq | Modelo Transformer para reconhecimento de texto com decodificacao permutada/autoregressiva. |
| Accuracy | Proporcao de predicoes completamente corretas apos normalizacao textual. |
| CER | Character Error Rate, taxa de erro por caractere. |
| WER | Word Error Rate, taxa de erro por palavra. |
| Data leakage | Vazamento de informacao entre treino, validacao e teste. |

## Convencoes

- "Texto alvo" significa a transcricao correta associada a cada imagem.
- "Predicao" significa o texto gerado pelo modelo apos decodificacao.
- "Normalizacao" significa aplicar as mesmas regras ao texto alvo e a predicao antes de calcular metricas.
