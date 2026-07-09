from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class E2CnnBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes=78,
        dropout=0.30,
        blank_idx=0,
        default_return_sequence=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.blank_idx = blank_idx
        self.default_return_sequence = default_return_sequence

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * 2, num_classes)
        self.ctc_criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, x, return_sequence=None):
        if return_sequence is None:
            return_sequence = self.default_return_sequence

        x = self.features(x)

        x = x.mean(dim=2)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = self.dropout(x)
        x = self.fc(x)

        if return_sequence:
            return x

        x = x[:, -1, :]
        return x

    def ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor | Sequence[torch.Tensor],
        target_lengths: torch.Tensor | Sequence[int],
        input_lengths: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Calcula CTCLoss para logits no formato batch-first: [B, T, C]."""
        if logits.ndim != 3:
            raise ValueError(
                "CTC espera logits com shape [batch, time, classes]. "
                f"Recebido: {tuple(logits.shape)}"
            )

        batch_size, time_steps, _ = logits.shape
        target_lengths = torch.as_tensor(
            target_lengths,
            dtype=torch.long,
            device=logits.device,
        )
        if input_lengths is None:
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=time_steps,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            input_lengths = torch.as_tensor(
                input_lengths,
                dtype=torch.long,
                device=logits.device,
            )

        if isinstance(targets, torch.Tensor):
            ctc_targets = targets.to(device=logits.device, dtype=torch.long)
        else:
            ctc_targets = torch.cat(
                [target.to(device=logits.device, dtype=torch.long) for target in targets]
            )

        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        return self.ctc_criterion(
            log_probs,
            ctc_targets,
            input_lengths,
            target_lengths,
        )

    def ctc_decode(
        self,
        logits: torch.Tensor,
        idx_to_token: dict[int | str, str] | None = None,
        input_lengths: torch.Tensor | Sequence[int] | None = None,
        blank_idx: int | None = None,
    ) -> list[str] | list[list[int]]:
        """Decodifica CTC greedy removendo blanks e repeticoes consecutivas."""
        if logits.ndim != 3:
            raise ValueError(
                "CTC decode espera logits com shape [batch, time, classes]. "
                f"Recebido: {tuple(logits.shape)}"
            )

        blank_idx = self.blank_idx if blank_idx is None else blank_idx
        predicted_indices = logits.argmax(dim=-1).detach().cpu()
        if input_lengths is None:
            lengths = [predicted_indices.size(1)] * predicted_indices.size(0)
        else:
            lengths = [
                int(length)
                for length in torch.as_tensor(input_lengths, dtype=torch.long).cpu()
            ]

        decoded: list[str] | list[list[int]]
        decoded = []
        for sequence, length in zip(predicted_indices, lengths):
            collapsed: list[int] = []
            previous_index: int | None = None
            for raw_index in sequence[:length].tolist():
                index = int(raw_index)
                if index != previous_index and index != blank_idx:
                    collapsed.append(index)
                previous_index = index

            if idx_to_token is None:
                decoded.append(collapsed)
            else:
                decoded.append(
                    "".join(
                        idx_to_token.get(index, idx_to_token.get(str(index), ""))
                        for index in collapsed
                    )
                )

        return decoded

    def greedy_decode(
        self,
        logits: torch.Tensor,
        idx_to_token: dict[int | str, str] | None = None,
        input_lengths: torch.Tensor | Sequence[int] | None = None,
        blank_idx: int | None = None,
    ) -> list[str] | list[list[int]]:
        return self.ctc_decode(
            logits=logits,
            idx_to_token=idx_to_token,
            input_lengths=input_lengths,
            blank_idx=blank_idx,
        )
