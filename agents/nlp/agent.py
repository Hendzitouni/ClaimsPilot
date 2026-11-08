
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize French and English claim descriptions."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


PART_KEYWORDS = {
    "front_bumper": [
        "pare-chocs avant",
        "devant de ma voiture",
        "front bumper",
        "front of my car",
    ],
    "rear_bumper": [
        "pare-chocs arrière",
        "arrière de ma voiture",
        "rear bumper",
        "back bumper",
    ],
    "door": [
        "portière",
        "porte de mon véhicule",
        "car door",
        "my door",
        "side door",
    ],
    "headlight": [
        "phare",
        "headlight",
        "front light",
    ],
    "windshield": [
        "pare-brise",
        "windshield",
    ],
}


def extract_damaged_parts(text: str) -> list[str]:
    """Extract damaged vehicle parts from a claim description."""
    normalized = normalize_text(text)
    detected = []

    for part, keywords in PART_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = normalize_text(keyword)

            pattern = (
                r"(?<!\w)"
                + re.escape(normalized_keyword)
                + r"(?!\w)"
            )

            if re.search(pattern, normalized):
                detected.append(part)
                break

    return detected
def check_consistency(
    described_parts: list[str],
    vision_parts: list[str]
) -> str:
    """Compare the claim description with detected visual damage."""

    if not described_parts or not vision_parts:
        return "uncertain"

    if set(described_parts) == set(vision_parts):
        return "consistent"

    return "inconsistent"
def analyze_claim(text: str, vision_parts: list[str]) -> dict:
    """Analyze a claim description against detected visual damage."""

    described_parts = extract_damaged_parts(text)

    consistency = check_consistency(
        described_parts,
        vision_parts
    )

    return {
        "described_parts": described_parts,
        "consistency": consistency,
    }
