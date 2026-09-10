"""NLP utilities — parse free-text symptoms into feature adjustments."""

SYMPTOM_KEYWORDS = {
    "frequent urination": {"Glucose": 15},
    "excessive thirst": {"Glucose": 10},
    "fatigue": {"Glucose": 5, "BMI": 2},
    "blurred vision": {"Glucose": 10},
    "chest pain": {"trestbps": 10, "thalach": -5},
    "shortness of breath": {"thalach": -10},
    "dizziness": {"trestbps": 5},
    "high blood pressure": {"trestbps": 15},
    "numbness": {"Glucose": 5},
    "slow healing": {"Glucose": 8},
}


def parse_symptoms(text: str) -> dict:
    """
    Scans free-text symptom description for known keywords and returns
    a dict of feature -> adjustment value to nudge model inputs.
    """
    text = (text or "").lower()
    detected = {}
    for symptom, adjustments in SYMPTOM_KEYWORDS.items():
        if symptom in text:
            for feature, delta in adjustments.items():
                detected[feature] = detected.get(feature, 0) + delta
    return detected
