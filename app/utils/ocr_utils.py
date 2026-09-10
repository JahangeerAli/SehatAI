"""OCR utilities — extract numeric health values from an uploaded lab report image."""

import pytesseract
from PIL import Image
import re


def extract_report_values(image_path: str):
    """
    Reads text from a lab report image and extracts common health metrics.
    Returns (extracted_dict, raw_text).
    """
    text = pytesseract.image_to_string(Image.open(image_path))

    patterns = {
        "Glucose": r"glucose[:\s]+(\d+\.?\d*)",
        "trestbps": r"(?:blood pressure|bp)[:\s]+(\d+)",
        "chol": r"cholesterol[:\s]+(\d+\.?\d*)",
        "BMI": r"bmi[:\s]+(\d+\.?\d*)",
    }

    extracted = {}
    text_lower = text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            extracted[key] = float(match.group(1))

    return extracted, text
