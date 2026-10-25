"""Subset coordinates in checkpoint order."""
MODALITIES = ("text", "audio", "vision")
SUBSETS = {
    "T": ("text",), "A": ("audio",), "V": ("vision",),
    "TA": ("text", "audio"), "TV": ("text", "vision"),
    "AV": ("audio", "vision"), "TAV": MODALITIES,
}
EXPERTS = tuple(SUBSETS)


def allowed_experts(available):
    available = frozenset(available)
    if not available or not available.issubset(MODALITIES):
        raise ValueError("Availability must be a nonempty subset of text/audio/vision")
    return tuple(name for name, subset in SUBSETS.items() if set(subset) <= available)
