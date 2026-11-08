
from agents.nlp.agent import (
    analyze_claim,
    check_consistency,
    extract_damaged_parts,
)


def test_french_extraction():
    result = extract_damaged_parts("Le pare-chocs avant est rayé.")
    assert result == ["front_bumper"]


def test_english_extraction():
    result = extract_damaged_parts("My windshield is broken.")
    assert result == ["windshield"]


def test_consistent_claim():
    result = check_consistency(
        ["front_bumper"], ["front_bumper"]
    )
    assert result == "consistent"


def test_inconsistent_claim():
    result = check_consistency(
        ["front_bumper"], ["rear_bumper"]
    )
    assert result == "inconsistent"


def test_uncertain_claim():
    result = check_consistency(
        ["front_bumper"], []
    )
    assert result == "uncertain"


def test_complete_workflow():
    result = analyze_claim(
        "La portière est rayée.",
        ["door"]
    )
    assert result == {
        "described_parts": ["door"],
        "consistency": "consistent",
    }
