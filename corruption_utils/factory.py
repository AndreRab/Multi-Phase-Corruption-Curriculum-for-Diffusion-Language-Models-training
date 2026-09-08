"""Shared construction and validation for corruption methods.

Keeping this contract in one place prevents training and evaluation from
silently using different method ordering or method-specific parameters.
"""

from __future__ import annotations

from collections.abc import Sequence

from corruption_utils.base_corruption_method import CorruptionMethod
from corruption_utils.mask_token_corruption_method import MaskTokenCorruption
from corruption_utils.random_token_corruption import RandomTokenCorruption
from corruption_utils.similar_token_corruption_method import SimilarTokenCorruption


CORRUPTION_METHOD_ORDER = ("similar", "mask", "random")
SUPPORTED_CORRUPTION_METHODS = frozenset(CORRUPTION_METHOD_ORDER)


def validate_corruption_methods(method_names: Sequence[str]) -> list[str]:
    """Validate and return corruption method names in the requested order."""

    names = list(method_names)
    unknown = sorted(set(names) - SUPPORTED_CORRUPTION_METHODS)
    if unknown:
        raise ValueError(f"Unknown corruption method(s): {unknown}")
    if not names:
        raise ValueError("At least one corruption method is required.")
    if len(set(names)) != len(names):
        raise ValueError("corruption methods must not contain duplicates.")
    return names


def build_corruption(
    name: str,
    *,
    model,
    tokenizer,
    num_diffusion_steps: int,
    minimum_probability: float,
    maximum_probability: float,
    similar_number_of_neighbors: int = 20,
) -> CorruptionMethod:
    """Build one corruption method using the shared train/eval parameters."""

    if name not in SUPPORTED_CORRUPTION_METHODS:
        raise ValueError(
            f"Unknown corruption method '{name}'. "
            f"Expected one of {sorted(SUPPORTED_CORRUPTION_METHODS)}."
        )
    if not 0 < minimum_probability <= maximum_probability <= 1:
        raise ValueError(
            "corruption probabilities must satisfy "
            "0 < minimum_probability <= maximum_probability <= 1."
        )

    if name == "similar":
        return SimilarTokenCorruption(
            embedding_weight=model.transformer.wte.weight,
            num_diffusion_steps=num_diffusion_steps,
            number_of_neighbors=similar_number_of_neighbors,
            minimum_probability=minimum_probability,
            maximum_probability=maximum_probability,
        )
    if name == "mask":
        return MaskTokenCorruption(
            mask_token_id=tokenizer.mask_token_id,
            num_diffusion_steps=num_diffusion_steps,
            minimum_probability=minimum_probability,
            maximum_probability=maximum_probability,
        )
    return RandomTokenCorruption(
        dictionary_size=len(tokenizer),
        num_diffusion_steps=num_diffusion_steps,
        minimum_probability=minimum_probability,
        maximum_probability=maximum_probability,
    )


def build_corruptions(
    method_names: Sequence[str],
    *,
    model,
    tokenizer,
    num_diffusion_steps: int,
    minimum_probability: float,
    maximum_probability: float,
    similar_number_of_neighbors: int = 20,
) -> list[CorruptionMethod]:
    """Build methods in the exact order used by ``iterations_intervals``."""

    names = validate_corruption_methods(method_names)
    return [
        build_corruption(
            name,
            model=model,
            tokenizer=tokenizer,
            num_diffusion_steps=num_diffusion_steps,
            minimum_probability=minimum_probability,
            maximum_probability=maximum_probability,
            similar_number_of_neighbors=similar_number_of_neighbors,
        )
        for name in names
    ]
