from __future__ import annotations

import numpy as np

FDI_LABELS = (
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
)
FDI_TO_CLASS = {fdi: class_id for class_id, fdi in enumerate(FDI_LABELS)}
CLASS_TO_FDI = {class_id: fdi for fdi, class_id in FDI_TO_CLASS.items()}
ARCH_CLASS_LABELS = tuple(range(17))
UPPER_ARCH_CLASS_TO_FDI = {
    0: 0,
    1: 18,
    2: 17,
    3: 16,
    4: 15,
    5: 14,
    6: 13,
    7: 12,
    8: 11,
    9: 21,
    10: 22,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
}
LOWER_ARCH_CLASS_TO_FDI = {
    0: 0,
    1: 38,
    2: 37,
    3: 36,
    4: 35,
    5: 34,
    6: 33,
    7: 32,
    8: 31,
    9: 41,
    10: 42,
    11: 43,
    12: 44,
    13: 45,
    14: 46,
    15: 47,
    16: 48,
}
ARCH_CLASS_TO_FDI = {
    "upper": UPPER_ARCH_CLASS_TO_FDI,
    "lower": LOWER_ARCH_CLASS_TO_FDI,
}
FDI_TO_ARCH_CLASS = {
    fdi: class_id
    for mapping in ARCH_CLASS_TO_FDI.values()
    for class_id, fdi in mapping.items()
}


def map_fdi_to_arch_class(
    labels: np.ndarray, mapping: dict[int, int] | None = None
) -> np.ndarray:
    """Map Teeth3DS FDI labels to a jaw-normalized 17-class target."""
    return _map_labels(
        labels, FDI_TO_ARCH_CLASS if mapping is None else mapping, name="FDI"
    )


def map_fdi_to_class(
    labels: np.ndarray, mapping: dict[int, int] | None = None
) -> np.ndarray:
    """Map raw FDI labels to the global contiguous 33-class target."""
    return _map_labels(labels, FDI_TO_CLASS if mapping is None else mapping, name="FDI")


def map_arch_class_to_fdi(labels: np.ndarray, jaw: str) -> np.ndarray:
    """Restore FDI labels from jaw-normalized 17-class predictions."""
    if jaw not in ARCH_CLASS_TO_FDI:
        raise ValueError(f"Unsupported jaw: {jaw!r}")
    return _map_labels(labels, ARCH_CLASS_TO_FDI[jaw], name=f"{jaw} arch class")


def _map_labels(labels: np.ndarray, mapping: dict[int, int], name: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    unknown = sorted(set(labels.reshape(-1).tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected {name} labels: {unknown}")

    mapped = np.empty_like(labels, dtype=np.int64)
    for source, target in mapping.items():
        mapped[labels == source] = target
    return mapped
