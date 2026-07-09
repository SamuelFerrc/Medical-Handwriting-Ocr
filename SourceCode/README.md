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

## Fine-tuning do TrOCR

O treino do TrOCR fica em `src/med_ocr_research/training/train_trocr.py`.
Use `--dataset` para alternar entre os splits prontos:

```powershell
python src/med_ocr_research/training/train_trocr.py --dataset prescriptionbd
python src/med_ocr_research/training/train_trocr.py --dataset rxhandbd
python src/med_ocr_research/training/train_trocr.py --dataset combined
```

Ele aceita os mesmos argumentos principais usados nos treinos E1/E2:

```powershell
python3 src/med_ocr_research/training/train_trocr.py `
  --run-name e3.03 `
  --dataset combined `
  --epochs 200 `
  --batch-size 2 `
  --gradient-accumulation-steps 8 `
  --learning-rate 0.0001 `
  --augmentation-strength strong `
  --fp16 `
  --gradient-checkpointing
```

Em GPUs com cerca de 6 GB de VRAM, como RTX 3050, `--batch-size 16`
costuma estourar memoria no TrOCR. O exemplo acima usa micro-batch 2 e
acumulacao 8, mantendo batch efetivo 16.

Para um dataset novo, informe os CSVs diretamente, mantendo as colunas de imagem
e texto do manifesto:

```powershell
python src/med_ocr_research/training/train_trocr.py `
  --train-csv data/splits/meu_dataset/train.csv `
  --val-csv data/splits/meu_dataset/val.csv `
  --data-root data `
  --label-column text_normalized `
  --image-column image_path
```

Tambem ha um executor sequencial no mesmo estilo dos treinos CNN:

```powershell
python scripts/train_trocr_sequence.py --datasets prescriptionbd rxhandbd combined
```

Os artefatos sao salvos em `outputs/runs/<run-name>/`, incluindo `trocr_best/`,
`trocr_final/`, historico de treino, resumo da execucao e predicoes da melhor
validacao.

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
