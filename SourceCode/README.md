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

## Experimentos com diferentes seeds

As seeds de treinamento estao em `configs/default.yaml`:

```yaml
reproducibility:
  seed: 42  # seed dos splits; nao muda entre os treinamentos
  baseline_seed: 42
  training_seeds: [42, 123, 456]
evaluation:
  seed_aggregation: median
  require_all_seeds: true
```

Os nomes das execucoes incluem a seed, por exemplo:

```text
outputs/runs/e2.03_seed42/
outputs/metrics/e2.03_seed42_evaluation/
outputs/runs/e2.03_seed123/
outputs/metrics/e2.03_seed123_evaluation/
```

Os treinos individuais tambem acrescentam `_seed<seed>` ao `--run-name`
automaticamente e recusam sobrescrever uma pasta que ja contem artefatos.
Os caminhos padrao de avaliacao e inspecao incluem esse sufixo. A pasta,
o `run_config.csv` e os metadados externos identificam a execucao.

### Protocolo comum de treinamento e validacao

A secao `training` de `configs/default.yaml` define o protocolo da matriz
inteira. Os hiperparametros nao sao mais copiados de uma execucao da seed 42.
O executor aplica as seguintes condicoes aos quatro modelos:

| Condicao | Valor comum |
| --- | --- |
| Dados | Mesmos CSVs de treino, validacao e teste por cenario; splits fixos |
| Inicializacao | Todos os pesos aprendiveis inicializados do zero, inclusive TrOCR e PARSeq |
| Orcamento | 200 epocas completas, sem early stopping |
| Batch | Microbatch 2, acumulacao de 8 passos, batch efetivo nominal 16 |
| Otimizador | AdamW, learning rate `0.0001`, weight decay `0.01` |
| Scheduler | Warmup linear nos primeiros 10% dos passos, seguido de decaimento linear |
| Gradientes | Norma maxima `1.0`; acumulacao ponderada pelas amostras, inclusive no ultimo grupo incompleto |
| Precisao | FP16 com AMP quando executado em CUDA |
| Augmentation | Mesma configuracao `strong`, aplicada somente no treino |
| Aleatoriedade | Seeds 42, 123 e 456; geradores independentes e semeados para os DataLoaders |
| Validacao | Todas as amostras do split, sem augmentation e sem filtrar textos desconhecidos |
| Melhor checkpoint | Menor CER de validacao; maior Accuracy como desempate |
| Teste | Avaliacao do checkpoint escolhido pela validacao sobre o teste completo |

As populacoes esperadas de treino/validacao/teste sao respectivamente
`3120/780/780` para PrescriptionBD, `3905/837/836` para RxHandBD e
`7025/1617/1616` para o cenario combinado. A CNN E1 mantem seu vocabulario
aprendido apenas no treino. Os textos desconhecidos continuam contando nas
metricas de validacao; sua loss de classificacao e calculada somente sobre
os alvos representaveis. As losses nativas dos modelos nao sao usadas para
comparar sua qualidade de transcricao nem selecionar o melhor checkpoint.

As arquiteturas, funcoes de perda e normalizacoes de entrada proprias de
cada modelo sao preservadas. **Todos os modelos sao treinados do zero**:
`training.pretrained_transformers: false` impede carregar pesos aprendidos
para TrOCR e PARSeq. O TrOCR conserva sua configuracao, processor e tokenizer
de referencia, mas inicializa encoder e decoder aleatoriamente; o PARSeq usa
a implementacao oficial com `pretrained=False`. Os pesos de uma seed nunca
sao usados para iniciar outra.

A comparacao controla a inicializacao sem pesos pre-treinados, a exposicao
aos dados e os hiperparametros comuns. As arquiteturas ainda diferem em
objetivos, vocabularios, normalizacao e custo computacional; o protocolo nao
garante que a configuracao comum seja a melhor para cada modelo. A seed do
treino nao modifica os grupos dos splits nem garante separacao por escritor
quando essa identificacao nao esta disponivel nos dados.

O redimensionamento inicial em PIL e `64x512`. O processor do TrOCR transforma
essa imagem novamente em `384x384`, sua entrada nativa. CNN, CNN-BiLSTM e
PARSeq usam `64x512`. A equivalencia do protocolo de treino nao significa
resolucao efetiva identica entre todas as arquiteturas.

### Resultados historicos e novos treinamentos

As 12 execucoes historicas da seed 42 e a execucao parcial `e1.01_seed123`
usaram o protocolo anterior. CNN/CRNN precisam ser refeitas devido ao novo
scheduler, acumulacao, clipping e criterio de checkpoint; TrOCR/PARSeq,
devido a inicializacao do zero e a mudanca de batch efetivo 256 para 16.
A validacao completa tambem
muda a selecao de checkpoint da CNN nos cenarios com textos desconhecidos.
Portanto, nenhuma dessas execucoes compoe as medianas do protocolo atual.

O arquivamento preserva os artefatos incompativeis em
`outputs/archive/<timestamp>/`, com um manifesto, antes de liberar seus
nomes para novas execucoes. Para conferir e aplicar essa operacao:

```bash
python3 scripts/archive_incompatible_runs.py --dry-run
python3 scripts/archive_incompatible_runs.py
```

O protocolo e sua assinatura (`protocol_version` e fingerprint) acompanham
os novos registros. As verificacoes de compatibilidade impedem que resultados
antigos ou de outra configuracao sejam reaproveitados como se pertencessem ao
protocolo atual. Arquivar uma execucao preserva o historico; nao torna seus
resultados compativeis com o novo protocolo.

As avaliacoes tambem registram e conferem o protocolo e o hash dos pesos
efetivamente avaliados. Um resultado antigo nao e aceito apenas porque usa
o mesmo caminho de checkpoint de um novo treinamento.

Para conferir os comandos sem iniciar treinamento:

```bash
python3 scripts/train_all_seeds.py --dry-run
```

Para executar a matriz atual, sao necessarios **36 treinamentos e suas
avaliacoes** (4 modelos x 3 cenarios x 3 seeds), incluindo a repeticao da seed
42. Execute no ambiente com `requirements-train.txt` instalado:

```bash
python3 scripts/train_all_seeds.py
```

O executor le `configs/experiments/matrix.yaml` e `configs/default.yaml`.
Ele nao exige que uma seed de referencia ja esteja treinada. Para a
comparacao entre modelos e seeds, use `train_all_seeds.py`: os executores
individuais e os executores sequenciais antigos podem receber argumentos
diferentes e nao substituem esse protocolo. Cada treino comeca novamente,
sem carregar o resultado de outra seed.

As execucoes sao sequenciais. Apos cada treino, o melhor checkpoint selecionado
na validacao e avaliado no teste completo. Uma nova chamada pula treinamentos
completos compativeis com o protocolo e avaliacoes validas ja existentes.
Se houver uma pasta de treinamento incompleto, o executor para: preserve
essa pasta com outro nome antes de repetir
o treino desde o inicio. Ele nao retoma automaticamente pesos/otimizadores de
treinamentos interrompidos. Uma falha interrompe a sequencia.

Ao concluir todas as avaliacoes, o executor calcula **a mediana entre as seeds
42, 123 e 456 de cada metrica**, separadamente para cada modelo/cenario.
Accuracy, CER e WER sao calculadas primeiro sobre o teste completo de cada
execucao. A mediana e aplicada aos tres valores de cada metrica; nao se agrupam
predicoes entre seeds nem se seleciona uma unica seed vencedora.

Os resultados finais ficam em `outputs/metrics/seed_summary/`:

- `per_seed_metrics.csv`: resultados individuais e caminho de origem;
- `seed_medians.csv`: medianas de Accuracy (`exact_match_accuracy`), CER e WER;
- `seed_medians.json`: resultados individuais, medianas e seeds utilizadas.

Para recalcular a consolidacao depois das avaliacoes:

```bash
python3 scripts/aggregate_seed_metrics.py
```

A consolidacao exige todas as seeds, protocolo e parametros de treinamento
compativeis e exatamente as mesmas amostras/rotulos de teste. Enquanto faltar
qualquer avaliacao requerida, nenhuma mediana final e publicada.

Para migrar outra copia que ainda tenha os nomes antigos, use
`python3 scripts/migrate_seed_runs.py --dry-run` e depois
`python3 scripts/migrate_seed_runs.py`. A migracao atualiza nomes e referencias,
preserva os pesos e recusa colisoes; pode ser repetida sem duplicar sufixos.
Essa migracao historica altera apenas a identificacao por seed: execucoes
do protocolo anterior ainda precisam ser arquivadas e refeitas para entrar
na comparacao atual.

Testes do fluxo, sem GPU ou PyTorch:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_seed_workflow.py -v
```

## Execucao independente do VS Code e painel no navegador (Linux)

O treinamento e o painel podem rodar como dois servicos do `systemd --user`.
Eles continuam executando depois que o terminal e o VS Code sao fechados.
O painel pode ser fechado ou reiniciado sem interromper o treinamento.

No ambiente Python com as dependencias de treinamento instaladas, inicie com:

```bash
python3 scripts/training_service.py start
```

Se as dependencias estiverem em outro ambiente, informe seu interpretador:

```bash
python3 scripts/training_service.py start --python .venv/bin/python
```

Esse comando instala os servicos do usuario, inicia o painel e dispara
`train_all_seeds.py` em segundo plano. O Python escolhido e preservado nos
servicos. O treinamento verifica os artefatos existentes, pula as execucoes
completas do protocolo atual e para se encontrar treinamento parcial ou
incompativel, conforme o fluxo de seeds.

Para abrir apenas o painel, sem iniciar nenhum treinamento:

```bash
python3 scripts/training_service.py monitor
```

O comando imprime um link para `http://127.0.0.1:8765/` contendo o token de
acesso. Abra o link completo. O painel mostra a etapa/modelo/seed atual,
epocas registradas, resultados individuais, medianas finais e as ultimas
linhas do log, com atualizacao a cada 3 segundos. Os campos sem resultado
aparecem como pendentes. As medianas continuam exigindo as tres seeds.

Para acessar a partir de outro dispositivo **na mesma rede local**:

```bash
python3 scripts/training_service.py monitor --host 0.0.0.0
```

Use no navegador o IP local do computador de treinamento no lugar de
`127.0.0.1`, mantendo a porta e o token. `hostname -I` ajuda a identificar
esse IP. O acesso depende de a rede/firewall permitir a porta 8765; o script
nao modifica o firewall. Para tornar esse endereco o padrao das proximas
chamadas, altere `monitoring.host` em `configs/default.yaml` para `0.0.0.0`.
O padrao `127.0.0.1` restringe o acesso ao proprio computador. Nao e necessario
publicar nada na internet. O token deve ser compartilhado apenas com quem
vai acompanhar a execucao; o painel e de leitura, sem botoes para executar
comandos ou parar o treinamento.

Comandos de acompanhamento e parada:

```bash
python3 scripts/training_service.py status        # status e link de acesso
python3 scripts/training_service.py stop          # para treino e subprocessos
python3 scripts/training_service.py stop-monitor  # para so o painel
```

Os logs persistentes ficam em `outputs/monitor/training.log` e
`outputs/monitor/monitor.log`; o estado da execucao fica em `pipeline.json`.
Essa pasta, incluindo o token, e ignorada pelo Git. Uma trava impede dois
executores simultaneos. Se o processo falhar ou for interrompido, o painel
mostra o erro/estado e preserva os resultados ja gravados. Treinamentos nao
sao reiniciados automaticamente apos falha ou reinicializacao do computador.

O computador precisa permanecer ligado e sem suspensao. Fechar o VS Code e
diferente de encerrar a sessao do usuario: para sobreviver tambem ao logout
do Linux, o administrador pode habilitar `loginctl enable-linger <usuario>`.
Essa configuracao nao e alterada pelos scripts. Os servicos tem nomes
`med-ocr-<identificador-do-projeto>-training.service` e
`med-ocr-<identificador-do-projeto>-monitor.service`, exibidos pelo comando
`status`, e sao instalados em `~/.config/systemd/user/` (ou `XDG_CONFIG_HOME`).

Testes do painel e controle de execucao:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_training_monitor.py -v
```

## Treinamento do TrOCR

O treino do TrOCR fica em `src/med_ocr_research/training/train_trocr.py`.
Na matriz comparativa, `train_all_seeds.py` transmite `--no-pretrained` para
inicializar todos os pesos do zero. Exemplos individuais devem incluir essa
opcao para adotar o mesmo regime de inicializacao.
Use `--dataset` para alternar entre os splits prontos:

```powershell
python src/med_ocr_research/training/train_trocr.py --dataset prescriptionbd --no-pretrained
python src/med_ocr_research/training/train_trocr.py --dataset rxhandbd --no-pretrained
python src/med_ocr_research/training/train_trocr.py --dataset combined --no-pretrained
```

Ele aceita os mesmos argumentos principais usados nos treinos E1/E2:

```powershell
python3 src/med_ocr_research/training/train_trocr.py `
  --run-name e3.03 `
  --dataset combined `
  --no-pretrained `
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
  --no-pretrained `
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

## Treinamento do PARSeq

O PARSeq usa a implementacao oficial de `baudm/parseq`, fixada em um commit
compativel com PyTorch Lightning 2.x. Na matriz comparativa, o modelo e
instanciado com `--no-pretrained`, sem carregar os pesos publicados.

O pipeline mantem os mesmos splits, data augmentation, otimizador, scheduler,
metricas, criterio de melhor checkpoint e artefatos do TrOCR. O checkpoint
oficial e nativo em `32x128` e 25 caracteres; este pipeline adapta os
embeddings posicionais inicializados aleatoriamente para `64x512` e expande
o limite para 64, conforme os parametros experimentais usados no TrOCR.

```powershell
python src/med_ocr_research/training/train_parseq.py --dataset prescriptionbd --no-pretrained
python src/med_ocr_research/training/train_parseq.py --dataset rxhandbd --no-pretrained
python src/med_ocr_research/training/train_parseq.py --dataset combined --no-pretrained
```

Para executar os tres cenarios em sequencia, gerando as runs `e4.01_seed42`, `e4.02_seed42`
e `e4.03_seed42` com a seed padrao:

```powershell
python scripts/train_parseq_sequence.py --datasets prescriptionbd rxhandbd combined
```

Exemplo com batch efetivo 16, como no treino longo do TrOCR:

```powershell
python src/med_ocr_research/training/train_parseq.py `
  --run-name e4.03 `
  --dataset combined `
  --data-root data `
  --model-name parseq `
  --no-pretrained `
  --epochs 200 `
  --batch-size 2 `
  --learning-rate 0.0001 `
  --weight-decay 0.01 `
  --num-workers 2 `
  --max-target-length 64 `
  --max-generation-length 64 `
  --num-beams 1 `
  --image-height 64 `
  --image-width 512 `
  --gradient-accumulation-steps 8 `
  --warmup-ratio 0.1 `
  --max-grad-norm 1.0 `
  --label-column text_normalized `
  --image-column image_path `
  --seed 42 `
  --augmentation-strength strong `
  --fp16 `
  --gradient-checkpointing
```

O melhor checkpoint e salvo em `outputs/runs/<run-name>/parseq_best/`. Para
avaliar no teste correspondente ao dataset salvo nos metadados:

```powershell
python src/med_ocr_research/evaluation/evaluate_parseq.py `
  --checkpoint outputs/runs/e4.03_seed42/parseq_best
```

Para inspecionar algumas predicoes manualmente:

```powershell
python tests/test_parseq_predictions.py `
  --checkpoint outputs/runs/e4.03_seed42/parseq_best
```

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
