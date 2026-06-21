"""
spaCy Entity Extractor
Extracts named entities using spaCy en_core_web_sm.

FIX: Only keep useful label types.
     CARDINAL, DATE, TIME, MONEY, PERCENT, ORDINAL were causing
     noise like "Battery → LOC", "Celeron → PERSON".
     Now filtered to ORG, PERSON, GPE, FAC, NORP only.
"""

import spacy

# Load model once at module level
nlp = spacy.load("en_core_web_sm")

# Only keep these spaCy label types — everything else is noise for tech docs
# Removed: CARDINAL, DATE, TIME, MONEY, PERCENT, ORDINAL, QUANTITY, LANGUAGE
ALLOWED_SPACY_LABELS = {"ORG", "PERSON", "GPE", "FAC", "NORP"}

# Map spaCy labels → our unified schema
TYPE_MAP = {
    "ORG":    "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE":    "LOCATION",
    "FAC":    "LOCATION",
    "NORP":   "ORGANIZATION",
}


def extract_entities_spacy(text: str) -> list:
    doc = nlp(text)
    entities = []

    for ent in doc.ents:

        # Skip noisy label types entirely
        if ent.label_ not in ALLOWED_SPACY_LABELS:
            continue

        mapped_type = TYPE_MAP.get(ent.label_, "ORGANIZATION")

        entities.append({
            "name": ent.text.strip(),
            "type": mapped_type,
        })

    return entities