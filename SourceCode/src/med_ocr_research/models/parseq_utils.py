from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# Commit oficial posterior ao release v1.0.0. Ele mantem os pesos publicados e
# corrige a compatibilidade com PyTorch Lightning 2.x.
DEFAULT_PARSEQ_HUB_REPO = (
    "baudm/parseq:1902db043c029a7e03a3818c616c06600af574be"
)
DEFAULT_PARSEQ_MODEL = "parseq"
SUPPORTED_PARSEQ_MODELS = ("parseq", "parseq_tiny", "parseq_patch16_224")
PARSEQ_WEIGHTS_NAME = "model.pt"
PARSEQ_METADATA_NAME = "training_metadata.json"


def load_official_parseq(
    model_name: str = DEFAULT_PARSEQ_MODEL,
    hub_repo: str = DEFAULT_PARSEQ_HUB_REPO,
    pretrained: bool = True,
    charset: str | None = None,
) -> Any:
    if model_name not in SUPPORTED_PARSEQ_MODELS:
        supported = ", ".join(SUPPORTED_PARSEQ_MODELS)
        raise ValueError(f"Modelo PARSeq desconhecido: {model_name}. Opcoes: {supported}")

    try:
        model = torch.hub.load(
            hub_repo,
            model_name,
            pretrained=pretrained,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Nao foi possivel carregar o PARSeq oficial. Instale "
            "requirements-train.txt e confira o acesso ao Torch Hub na primeira "
            "execucao."
        ) from exc

    if charset is not None:
        adapt_parseq_charset(model, charset)
    return model


def tokenizer_charset(tokenizer: Any) -> str:
    specials = {
        getattr(tokenizer, "BOS", "[B]"),
        getattr(tokenizer, "EOS", "[E]"),
        getattr(tokenizer, "PAD", "[P]"),
    }
    return "".join(token for token in tokenizer._itos if token not in specials)


def charset_with_extras(model: Any, extra_characters: str) -> str:
    charset = tokenizer_charset(model.tokenizer)
    for character in extra_characters:
        if character not in charset:
            charset += character
    return charset


def adapt_parseq_charset(model: Any, charset: str) -> None:
    """Expande o tokenizer/head preservando todos os pesos compativeis.

    O modelo oficial usa 94 caracteres imprimiveis, mas nao inclui espaco. Os
    manifestos medicos mantem espacos em rotulos como ``500 mg``. A expansao
    acrescenta esses caracteres ao fim do vocabulario e preserva exatamente os
    indices e pesos de EOS e dos 94 caracteres pre-treinados.
    """

    old_tokenizer = model.tokenizer
    old_charset = tokenizer_charset(old_tokenizer)
    if charset == old_charset:
        return
    if not charset.startswith(old_charset):
        raise ValueError(
            "O charset adaptado precisa preservar o charset oficial como prefixo."
        )

    new_tokenizer = type(old_tokenizer)(charset)
    parseq_model = model.model
    old_embedding = parseq_model.text_embed.embedding
    old_head = parseq_model.head

    new_embedding = nn.Embedding(
        len(new_tokenizer),
        old_embedding.embedding_dim,
        device=old_embedding.weight.device,
        dtype=old_embedding.weight.dtype,
    )
    new_head = nn.Linear(
        old_head.in_features,
        len(new_tokenizer) - 2,
        bias=old_head.bias is not None,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    nn.init.trunc_normal_(new_embedding.weight, std=0.02)
    nn.init.trunc_normal_(new_head.weight, std=0.02)
    if new_head.bias is not None:
        nn.init.zeros_(new_head.bias)

    with torch.no_grad():
        for token, old_index in old_tokenizer._stoi.items():
            new_index = new_tokenizer._stoi[token]
            new_embedding.weight[new_index].copy_(old_embedding.weight[old_index])
            if old_index < old_head.out_features and new_index < new_head.out_features:
                new_head.weight[new_index].copy_(old_head.weight[old_index])
                if old_head.bias is not None and new_head.bias is not None:
                    new_head.bias[new_index].copy_(old_head.bias[old_index])
        new_embedding.weight[new_tokenizer.pad_id].zero_()

    parseq_model.text_embed.embedding = new_embedding
    parseq_model.head = new_head
    model.tokenizer = new_tokenizer
    model.bos_id = new_tokenizer.bos_id
    model.eos_id = new_tokenizer.eos_id
    model.pad_id = new_tokenizer.pad_id
    try:
        model.hparams.charset_train = charset
    except Exception:
        pass


def adapt_parseq_dimensions(
    model: Any,
    image_size: tuple[int, int],
    max_label_length: int,
) -> None:
    """Adapta embeddings posicionais do checkpoint para outra resolucao/comprimento."""

    if max_label_length < 1:
        raise ValueError("max_label_length precisa ser maior que zero.")

    parseq_model = model.model
    encoder = parseq_model.encoder
    patch_embed = encoder.patch_embed
    old_image_size = tuple(int(value) for value in patch_embed.img_size)
    new_image_size = tuple(int(value) for value in image_size)
    patch_size = tuple(int(value) for value in patch_embed.patch_size)
    if any(size % patch != 0 for size, patch in zip(new_image_size, patch_size)):
        raise ValueError(
            f"Imagem {new_image_size} precisa ser divisivel pelo patch {patch_size}."
        )

    if new_image_size != old_image_size:
        old_grid = tuple(int(value) for value in patch_embed.grid_size)
        new_grid = tuple(
            size // patch for size, patch in zip(new_image_size, patch_size)
        )
        old_pos_embed = encoder.pos_embed
        embedding_dim = old_pos_embed.shape[-1]
        grid_embed = old_pos_embed.reshape(1, old_grid[0], old_grid[1], embedding_dim)
        grid_embed = grid_embed.permute(0, 3, 1, 2)
        resized_embed = F.interpolate(
            grid_embed,
            size=new_grid,
            mode="bicubic",
            align_corners=False,
        )
        resized_embed = resized_embed.permute(0, 2, 3, 1).reshape(
            1, new_grid[0] * new_grid[1], embedding_dim
        )
        encoder.pos_embed = nn.Parameter(resized_embed.to(old_pos_embed.dtype))
        patch_embed.img_size = new_image_size
        patch_embed.grid_size = new_grid
        patch_embed.num_patches = new_grid[0] * new_grid[1]

    old_max_length = int(parseq_model.max_label_length)
    if max_label_length != old_max_length:
        old_queries = parseq_model.pos_queries
        new_queries = old_queries.new_empty(
            (1, max_label_length + 1, old_queries.shape[-1])
        )
        nn.init.trunc_normal_(new_queries, std=0.02)
        positions_to_copy = min(old_queries.shape[1], new_queries.shape[1])
        with torch.no_grad():
            new_queries[:, :positions_to_copy].copy_(
                old_queries[:, :positions_to_copy]
            )
        parseq_model.pos_queries = nn.Parameter(new_queries)
        parseq_model.max_label_length = max_label_length

    try:
        model.hparams.img_size = list(new_image_size)
        model.hparams.max_label_length = max_label_length
    except Exception:
        pass


def parseq_input_size(model: Any) -> tuple[int, int]:
    height, width = model.hparams.img_size
    return int(height), int(width)


def parseq_max_label_length(model: Any) -> int:
    return int(model.hparams.max_label_length)


def validate_parseq_labels(
    labels: Sequence[str],
    model: Any,
    max_target_length: int,
    split_name: str,
) -> None:
    charset = set(tokenizer_charset(model.tokenizer))
    unsupported = sorted({char for label in labels for char in str(label)} - charset)
    if unsupported:
        printable = ", ".join(repr(char) for char in unsupported)
        raise ValueError(
            f"Caracteres nao suportados no split {split_name}: {printable}. "
            "Acrescente-os com --extra-characters."
        )

    longest = max((len(str(label)) for label in labels), default=0)
    model_limit = parseq_max_label_length(model)
    effective_limit = min(max_target_length, model_limit)
    if longest > effective_limit:
        raise ValueError(
            f"Rotulo com {longest} caracteres no split {split_name}; o limite "
            f"efetivo do PARSeq e {effective_limit}."
        )


def configure_parseq_gradient_checkpointing(model: Any, enabled: bool) -> None:
    encoder = model.model.encoder
    if not hasattr(encoder, "set_grad_checkpointing"):
        if enabled:
            raise RuntimeError("O encoder desta variante PARSeq nao suporta checkpointing.")
        return
    encoder.set_grad_checkpointing(enabled)


def freeze_parseq_encoder(model: Any, freeze: bool) -> None:
    if not freeze:
        return
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False


def parseq_permutation_loss(
    model: Any,
    images: torch.Tensor,
    labels: list[str],
) -> torch.Tensor:
    """Permutation language modeling loss da implementacao oficial do PARSeq."""

    tgt = model.tokenizer.encode(labels, images.device)
    memory = model.model.encode(images)
    tgt_perms = model.gen_tgt_perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)

    total_loss = images.new_zeros(())
    loss_numel = 0
    n = int((tgt_out != model.pad_id).sum().item())
    for index, permutation in enumerate(tgt_perms):
        tgt_mask, query_mask = model.generate_attn_masks(permutation)
        decoded = model.model.decode(
            tgt_in,
            memory,
            tgt_mask,
            tgt_padding_mask,
            tgt_query_mask=query_mask,
        )
        logits = model.model.head(decoded).flatten(end_dim=1)
        total_loss = total_loss + n * F.cross_entropy(
            logits,
            tgt_out.flatten(),
            ignore_index=model.pad_id,
        )
        loss_numel += n
        if index == 1:
            tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
            n = int((tgt_out != model.pad_id).sum().item())

    if loss_numel == 0:
        raise ValueError("O batch nao possui tokens validos para calcular a loss.")
    return total_loss / loss_numel


def decode_parseq_logits(
    model: Any,
    logits: torch.Tensor,
) -> tuple[list[str], list[float]]:
    probabilities = logits.softmax(-1)
    predictions, token_probabilities = model.tokenizer.decode(probabilities)
    confidences = [
        float(values.prod().item()) if values.numel() else 0.0
        for values in token_probabilities
    ]
    return [str(value) for value in predictions], confidences


def save_parseq_bundle(
    model: Any,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / PARSEQ_WEIGHTS_NAME)
    (output_dir / PARSEQ_METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_parseq_bundle(
    checkpoint_dir: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[Any, dict[str, Any]]:
    metadata_path = checkpoint_dir / PARSEQ_METADATA_NAME
    weights_path = checkpoint_dir / PARSEQ_WEIGHTS_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados PARSeq ausentes: {metadata_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesos PARSeq ausentes: {weights_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = load_official_parseq(
        model_name=str(metadata.get("model_name", DEFAULT_PARSEQ_MODEL)),
        hub_repo=str(metadata.get("hub_repo", DEFAULT_PARSEQ_HUB_REPO)),
        pretrained=False,
        charset=metadata.get("charset"),
    )
    metadata_image_size = metadata.get("image_size") or parseq_input_size(model)
    metadata_max_length = max(
        int(metadata.get("max_target_length", parseq_max_label_length(model))),
        int(metadata.get("max_generation_length", parseq_max_label_length(model))),
    )
    adapt_parseq_dimensions(
        model,
        (int(metadata_image_size[0]), int(metadata_image_size[1])),
        metadata_max_length,
    )
    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return model, metadata
