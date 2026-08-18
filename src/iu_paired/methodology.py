"""Small, dataset-free invariants for the thesis-selected paired methodology."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

IMAGE_WEIGHT = 0.75
TEXT_WEIGHT = 0.25
LOCKED_INPUT_COLUMNS = frozenset({"uid", "image_filenames", "indication"})
FORBIDDEN_PREDICTOR_COLUMNS = frozenset({"findings", "impression"})


def late_fuse(
    image_probabilities: Sequence[float],
    text_probabilities: Sequence[float],
    image_weight: float = IMAGE_WEIGHT,
) -> list[float]:
    """Fuse aligned image/text probabilities using a convex late-fusion weight."""
    if len(image_probabilities) != len(text_probabilities):
        raise ValueError("Image and text probability vectors must have equal length.")
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError("image_weight must be between 0 and 1 inclusive.")
    values = []
    for image_value, text_value in zip(image_probabilities, text_probabilities):
        if not 0.0 <= float(image_value) <= 1.0 or not 0.0 <= float(text_value) <= 1.0:
            raise ValueError("Probabilities must be between 0 and 1 inclusive.")
        values.append(image_weight * float(image_value) + (1.0 - image_weight) * float(text_value))
    return values


def validate_locked_predictor_columns(columns: Iterable[str]) -> None:
    """Reject leakage-prone reference fields in the locked-test predictor table."""
    normalized = {str(column).strip().lower() for column in columns}
    forbidden = normalized & FORBIDDEN_PREDICTOR_COLUMNS
    if forbidden:
        raise ValueError(f"Locked predictors contain forbidden reference fields: {sorted(forbidden)}")
    missing = LOCKED_INPUT_COLUMNS - normalized
    if missing:
        raise ValueError(f"Locked predictors are missing required fields: {sorted(missing)}")


def validate_disjoint_ids(development_ids: Iterable[str], locked_test_ids: Iterable[str]) -> None:
    """Require unique, non-overlapping development and locked-test study identifiers."""
    development = [str(value) for value in development_ids]
    locked = [str(value) for value in locked_test_ids]
    if len(development) != len(set(development)) or len(locked) != len(set(locked)):
        raise ValueError("Study identifiers must be unique within each partition.")
    overlap = set(development) & set(locked)
    if overlap:
        raise ValueError(f"Development and locked-test identifiers overlap: {sorted(overlap)[:5]}")
