from __future__ import annotations


def prompt_from_descriptors(descriptors: dict[str, float]) -> str:
    tone = _tone(descriptors)
    contrast = _contrast(descriptors)
    warmth = _warmth(descriptors)
    structure = _structure(descriptors)
    texture = _texture(descriptors)
    return (
        "A clean studio portrait generated from the provided seju-face centroid: "
        f"{tone}, {contrast}, {warmth}, {structure}, {texture}, "
        "balanced frontal composition, natural expression, soft fashion editorial lighting, "
        "high detail face, realistic skin texture, neutral background. "
        "Avoid copying any specific real person; synthesize a new face from aggregate traits only."
    )


def _tone(d: dict[str, float]) -> str:
    luminance = d.get("luminance", 0.5)
    if luminance >= 0.62:
        return "bright and airy facial tones"
    if luminance <= 0.42:
        return "deeper shaded facial tones"
    return "even mid-bright facial tones"


def _contrast(d: dict[str, float]) -> str:
    contrast = d.get("contrast", 0.2)
    if contrast <= 0.16:
        return "low-contrast soft features"
    if contrast >= 0.28:
        return "clear high-contrast feature separation"
    return "moderately soft feature contrast"


def _warmth(d: dict[str, float]) -> str:
    warmth = d.get("warmth", 0.0)
    saturation = d.get("saturation", 0.15)
    if warmth > 0.05:
        base = "warm skin-color balance"
    elif warmth < -0.04:
        base = "cool skin-color balance"
    else:
        base = "neutral skin-color balance"
    if saturation < 0.08:
        return base + " with muted color"
    if saturation > 0.22:
        return base + " with vivid color"
    return base


def _structure(d: dict[str, float]) -> str:
    symmetry = d.get("symmetry", 0.85)
    upper_darkness = d.get("upper_band_darkness", 0.4)
    if symmetry > 0.9 and upper_darkness > 0.45:
        return "symmetrical composition with defined eye-band shadow"
    if symmetry > 0.9:
        return "symmetrical composition with gentle eye definition"
    if upper_darkness > 0.45:
        return "defined eye-band shadow"
    return "natural balanced facial structure"


def _texture(d: dict[str, float]) -> str:
    edge_density = d.get("edge_density", 0.08)
    if edge_density < 0.05:
        return "smooth silhouette and minimal hard edges"
    if edge_density > 0.12:
        return "crisp hairline and facial detail"
    return "soft hairline and readable facial detail"
