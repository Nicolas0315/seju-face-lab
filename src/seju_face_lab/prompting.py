from __future__ import annotations


PROMPT_PROFILES = ("balanced", "detector-friendly")


def prompt_from_descriptors(descriptors: dict[str, float], profile: str = "balanced") -> str:
    if profile not in PROMPT_PROFILES:
        raise ValueError(f"Unsupported prompt profile: {profile}")
    tone = _tone(descriptors)
    contrast = _contrast(descriptors)
    warmth = _warmth(descriptors)
    structure = _structure(descriptors)
    texture = _texture(descriptors)
    prompt = (
        "Aggregate traits seju-face photoreal close-up portrait, "
        "new fictional person, "
        "no celebrity likeness, "
        f"{tone}, {contrast}, {warmth}, {structure}, {texture}, "
        "frontal unobscured face, clear eyes, natural expression, "
        "soft editorial light, realistic skin."
    )
    if profile == "detector-friendly":
        prompt += (
            " Centered head and shoulders, both eyes fully visible, looking into camera, "
            "face fills frame, plain background, no hair covering eyes, no hands near face."
        )
    return prompt


def negative_prompt_for_profile(profile: str) -> str:
    if profile == "balanced":
        return ""
    if profile == "detector-friendly":
        return (
            "profile view, side face, turned head, closed eyes, sunglasses, mask, "
            "hand over face, hair over eyes, cropped face, multiple people"
        )
    raise ValueError(f"Unsupported prompt profile: {profile}")


def _tone(d: dict[str, float]) -> str:
    luminance = d.get("luminance", 0.5)
    if luminance >= 0.62:
        return "bright airy tones"
    if luminance <= 0.42:
        return "deeper shaded tones"
    return "even mid-bright tones"


def _contrast(d: dict[str, float]) -> str:
    contrast = d.get("contrast", 0.2)
    if contrast <= 0.16:
        return "soft low contrast"
    if contrast >= 0.28:
        return "clear feature separation"
    return "moderate soft contrast"


def _warmth(d: dict[str, float]) -> str:
    warmth = d.get("warmth", 0.0)
    saturation = d.get("saturation", 0.15)
    if warmth > 0.05:
        base = "warm skin balance"
    elif warmth < -0.04:
        base = "cool skin balance"
    else:
        base = "neutral skin balance"
    if saturation < 0.08:
        return "muted " + base
    if saturation > 0.22:
        return "vivid " + base
    return base


def _structure(d: dict[str, float]) -> str:
    symmetry = d.get("symmetry", 0.85)
    upper_darkness = d.get("upper_band_darkness", 0.4)
    if symmetry > 0.9 and upper_darkness > 0.45:
        return "symmetric defined eyes"
    if symmetry > 0.9:
        return "symmetric gentle eyes"
    if upper_darkness > 0.45:
        return "defined eye shadow"
    return "balanced facial structure"


def _texture(d: dict[str, float]) -> str:
    edge_density = d.get("edge_density", 0.08)
    if edge_density < 0.05:
        return "smooth silhouette"
    if edge_density > 0.12:
        return "crisp hairline detail"
    return "soft readable hairline"
