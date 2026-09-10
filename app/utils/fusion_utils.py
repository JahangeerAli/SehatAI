"""Fusion layer — combines manual form input, OCR-extracted values, and
symptom-based adjustments into one final feature dict for the ML models."""


def fuse_features(base_features: dict, symptom_adjustments: dict, ocr_values: dict) -> dict:
    """
    base_features: manual form input (dict of all required model features)
    symptom_adjustments: output of parse_symptoms()
    ocr_values: output of extract_report_values()

    OCR values (if present) override manual input, since they come from an
    actual lab report. Symptom adjustments then nudge the values slightly.
    """
    final = dict(base_features)

    for k, v in ocr_values.items():
        if k in final:
            final[k] = v

    for k, delta in symptom_adjustments.items():
        if k in final:
            final[k] += delta

    return final
