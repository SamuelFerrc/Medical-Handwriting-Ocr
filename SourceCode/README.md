# Ambiente SourceCode - Pesquisa OCR/HTR Medica

Ambiente Python preparado para voce desenvolver o codigo da pesquisa em OCR/HTR de prescricoes manuscritas descrita em `../ImagensMedicas`.

Este repositorio ainda nao implementa a pesquisa. Ele apenas deixa a estrutura pronta para receber:

- padronizar RxHandBD e Doctor's Handwritten PrescriptionBD em um manifesto unico;
- normalizar rotulos com a mesma regra em todos os modelos;
- gerar splits fixos `train`, `val` e `test`;
- preprocessar imagens preservando proporcao;
- treinar ou adaptar CNN, CRNN, TrOCR e PARSeq;
- avaliar predicoes com Accuracy, CER e WER;
- registrar artefatos em `outputs/`.

## Estrutura

```text
SourceCode/
  configs/
    default.yaml
    data/
    experiments/
    models/
  data/
    raw/
    processed/
    splits/
  outputs/
    runs/
    predictions/
    metrics/
    reports/
  scripts/
  src/med_ocr_research/
  tests/
```

## Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quando voce for implementar os treinos com modelos profundos, instale tambem as dependencias opcionais listadas em `requirements-train.txt`.

```powershell
python -m pip install -r requirements-train.txt
```

## Fluxo planejado

1. Coloque os datasets originais em `data/raw/`.
2. Desenvolva os scripts de conversao para gerar manifestos em `data/processed/`.
3. Salve splits fixos em `data/splits/`.
4. Desenvolva preprocessamento, treino e avaliacao dentro de `src/`.
5. Grave checkpoints, predicoes, metricas e relatorios em `outputs/`.

## Contrato do manifesto

O arquivo principal deve conter:

```text
sample_id,dataset,image_path,text_raw,text_normalized,split,writer_id,document_id,width,height,notes
```

Campos obrigatorios:

- `sample_id`
- `dataset`
- `image_path`
- `text_raw`
- `text_normalized`
- `split`

`image_path` deve ser relativo ao diretorio de dados usado na execucao.

## Planejamento dos modelos

Os arquivos em `configs/models/` guardam pontos de partida para CNN, CRNN, TrOCR e PARSeq. Eles sao apenas configuracoes iniciais para orientar seu desenvolvimento.
