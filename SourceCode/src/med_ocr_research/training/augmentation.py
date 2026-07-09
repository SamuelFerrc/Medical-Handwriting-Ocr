from __future__ import annotations

from typing import Any


HANDWRITING_AUGMENTATION_CONFIGS: dict[str, dict[str, Any]] = {
    "light": {
        "degrees": 4,
        "translate": (0.02, 0.03),
        "scale": (0.97, 1.03),
        "shear": 2,
        "brightness": 0.10,
        "contrast": 0.15,
        "affine_p": 0.7,
        "jitter_p": 0.4,
        "blur_p": 0.1,
    },
    "medium": {
        "degrees": 8,
        "translate": (0.05, 0.06),
        "scale": (0.90, 1.08),
        "shear": 5,
        "brightness": 0.20,
        "contrast": 0.25,
        "affine_p": 0.8,
        "jitter_p": 0.5,
        "blur_p": 0.25,
    },
    "strong": {
        "degrees": 10,
        "translate": (0.07, 0.08),
        "scale": (0.85, 1.12),
        "shear": 7,
        "brightness": 0.25,
        "contrast": 0.35,
        "affine_p": 0.9,
        "jitter_p": 0.6,
        "blur_p": 0.3,
    },
}


def build_handwriting_train_transform(transforms: Any, strength: str) -> Any:
    if transforms is None:
        raise RuntimeError(
            "torchvision nao esta instalado. Instale requirements-train.txt "
            "ou rode com --no-augmentation."
        )

    config = HANDWRITING_AUGMENTATION_CONFIGS[strength]
    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=config["degrees"],
                        translate=config["translate"],
                        scale=config["scale"],
                        shear=config["shear"],
                        fill=255,
                    )
                ],
                p=config["affine_p"],
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=config["brightness"],
                        contrast=config["contrast"],
                    )
                ],
                p=config["jitter_p"],
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=config["blur_p"],
            ),
        ]
    )
