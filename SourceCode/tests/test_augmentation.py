from __future__ import annotations

import pytest

from med_ocr_research.training.augmentation import (
    HANDWRITING_AUGMENTATION_CONFIGS,
    build_handwriting_train_transform,
)


class FakeCompose:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms


class FakeRandomApply:
    def __init__(self, transforms: list[object], p: float) -> None:
        self.transforms = transforms
        self.p = p


class FakeRandomAffine:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeColorJitter:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeGaussianBlur:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeTransforms:
    Compose = FakeCompose
    RandomApply = FakeRandomApply
    RandomAffine = FakeRandomAffine
    ColorJitter = FakeColorJitter
    GaussianBlur = FakeGaussianBlur


@pytest.mark.parametrize("strength", ("light", "medium", "strong"))
def test_handwriting_augmentation_levels_use_expected_shared_config(
    strength: str,
) -> None:
    config = HANDWRITING_AUGMENTATION_CONFIGS[strength]

    transform = build_handwriting_train_transform(FakeTransforms, strength)

    affine_apply, jitter_apply, blur_apply = transform.transforms
    assert affine_apply.p == config["affine_p"]
    assert affine_apply.transforms[0].kwargs == {
        "degrees": config["degrees"],
        "translate": config["translate"],
        "scale": config["scale"],
        "shear": config["shear"],
        "fill": 255,
    }
    assert jitter_apply.p == config["jitter_p"]
    assert jitter_apply.transforms[0].kwargs == {
        "brightness": config["brightness"],
        "contrast": config["contrast"],
    }
    assert blur_apply.p == config["blur_p"]
    assert blur_apply.transforms[0].kwargs == {
        "kernel_size": 3,
        "sigma": (0.1, 0.8),
    }


def test_handwriting_augmentation_requires_torchvision_when_enabled() -> None:
    with pytest.raises(RuntimeError, match="torchvision"):
        build_handwriting_train_transform(None, "light")
