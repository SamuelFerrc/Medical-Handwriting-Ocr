---
tipo: metodologia
status: rascunho
tags:
  - pipeline
---

# Pipeline de Pesquisa

```mermaid
flowchart TD
    A["Coleta dos datasets"] --> B["Auditoria dos dados"]
    B --> C["Normalizacao dos rotulos"]
    C --> D["Preprocessamento das imagens"]
    D --> E["Divisao train/val/test"]
    E --> F["Treinamento CNN"]
    E --> G["Treinamento CRNN"]
    E --> H["Fine-tuning TrOCR"]
    E --> I["Fine-tuning PARSeq"]
    F --> J["Predicoes no teste"]
    G --> J
    H --> J
    I --> J
    J --> K["Calculo de Accuracy, CER e WER"]
    K --> L["Analise qualitativa dos erros"]
    L --> M["Discussao e conclusao"]
```

## Artefatos produzidos

- Tabela unica de amostras.
- Arquivos de split.
- Configuracoes de treinamento por modelo.
- Checkpoints.
- Predicoes por modelo.
- Tabelas de metricas.
- Relatorio de erros.

## Controle de versao

Cada execucao deve registrar:

- Data.
- Commit ou versao do codigo.
- Configuracao usada.
- Seed.
- Dataset/split.
- Checkpoint final.
- Metricas finais.
