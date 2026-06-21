"""
Entity Merger
Merges spaCy entities + tech entities, deduplicates, cleans types.

FIX: Added fix_entity_type() to correct wrong NER labels.
     spaCy en_core_web_sm misclassifies tech products as PERSON/LOC
     because it's trained on general text, not tech manuals.
     Example fixes: Celeron → PRODUCT, Battery → PRODUCT, DVD+RW → PRODUCT
"""

import re


# ─────────────────────────────────────────────────────────────
# TECH PRODUCT KEYWORDS
# If any of these appear in an entity name → force type to PRODUCT
# This corrects spaCy misclassifications for known tech terms.
# ─────────────────────────────────────────────────────────────
TECH_PRODUCT_KEYWORDS = [
    # Dell product lines
    "inspiron", "latitude", "optiplex", "dimension", "xps",
    "alienware", "vostro", "precision", "streak", "venue",
    # Intel / AMD processors
    "celeron", "pentium", "core i3", "core i5", "core i7", "core i9",
    "xeon", "atom", "athlon", "sempron", "turion",
    # Microsoft software
    "windows", "ms office", "microsoft office", "outlook", "excel",
    "word", "powerpoint", "internet explorer", "windows media player",
    "windows xp", "windows vista", "windows 7", "windows 10",
    # Hardware components / ports
    "dvd", "cd-rom", "cd/dvd", "usb", "bios", "sata", "pata",
    "ide", "pci", "agp", "hdmi", "vga", "dvi", "s-video",
    # Peripherals
    "battery", "keyboard", "display", "monitor", "touchpad",
    "webcam", "microphone", "speaker", "headphone",
    # Networking
    "wlan", "bluetooth", "ethernet", "modem", "router",
    "adapter", "wireless", "wi-fi",
    # GPU
    "geforce", "radeon", "directx", "opengl",
    # Other tech
    "nvidia", "realtek", "sigmatel", "broadcom",
]

# ─────────────────────────────────────────────────────────────
# GARBAGE NAMES
# Entity names that should be removed entirely — they are
# document structure words, not real-world entities.
# ─────────────────────────────────────────────────────────────
GARBAGE_NAMES = {
    # Document structure
    "notice", "caution", "notes", "note", "tip", "warning", "important",
    "table", "figure", "page", "drive", "button", "key",
    "chapter", "section", "step", "index", "contents",
    # View labels from hardware manuals
    "functions", "bottom view", "top view", "left side view",
    "right side view", "front view", "back view",
    # Manual section titles
    "before you reinstall", "icon sizes", "adjusting icon sizes",
    "adjusting font sizes", "computing habits", "scanner problems",
    "hardware incompatibilities", "getting help", "finding information",
    "technical support", "regulatory information",
    # Generic single words that slip through
    "one", "two", "three", "four", "five",
    "computer", "printer", "scanner",
}

# ─────────────────────────────────────────────────────────────
# GENERIC TERMS
# Real words but too vague to be useful in GraphRAG queries.
# Example: "system", "device" → appear everywhere, connect to everything,
# add no signal about the actual document content.
# ─────────────────────────────────────────────────────────────
GENERIC_TERMS = {
    "system", "file", "folder", "screen", "menu", "option",
    "setting", "feature", "mode", "type", "information", "version",
    "service", "support", "problem", "solution", "process", "data",
    "memory", "power", "network", "device", "window", "software",
    "hardware", "program", "application", "user", "password",
    "model", "number", "item", "list", "area", "panel",
}


# ─────────────────────────────────────────────────────────────
# FIX ENTITY TYPE
# ─────────────────────────────────────────────────────────────

def fix_entity_type(name: str, current_type: str):
    """
    Returns corrected type string, or None to remove entity entirely.

    Logic:
      1. Remove garbage names (document structure words)
      2. Remove generic terms (too vague for GraphRAG)
      3. Remove purely numeric strings
      4. Fix wrong NER types for known tech products
      5. Otherwise keep original type
    """
    name_lower = name.lower().strip()

    # 1. Remove garbage document structure words
    if name_lower in GARBAGE_NAMES:
        return None

    # 2. Remove generic terms
    if name_lower in GENERIC_TERMS:
        return None

    # 3. Remove purely numeric strings + junk like "10,000 ft", "1-Year"
    if re.match(r'^[\d\s,\.\-\/\%]+$', name):
        return None

    # 4. Fix wrong NER types for known tech products
    for keyword in TECH_PRODUCT_KEYWORDS:
        if keyword in name_lower:
            return "PRODUCT"

    # 5. Keep original type
    return current_type


# ─────────────────────────────────────────────────────────────
# VALIDITY CHECK
# ─────────────────────────────────────────────────────────────

def is_valid_entity(name: str) -> bool:
    name = name.strip()

    if not name:
        return False

    if name.isdigit():
        return False

    if len(name) < 3:
        return False

    # More than 6 words → likely a sentence fragment, not an entity
    if len(name.split()) > 6:
        return False

    # OCR noise filter: all-letter multi-word strings with very short avg word length
    if re.fullmatch(r"[a-zA-Z\s\.]+", name):
        words = name.split()
        if len(words) > 2:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len <= 2:
                return False

    return True


# ─────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────

def merge_entities(spacy_entities: list, tech_entities: list) -> list:
    """
    Merge tech entities + spaCy entities.
    Tech entities take priority (pattern-based = more reliable for this domain).
    Applies type correction and garbage filtering to both.
    """
    merged = []
    seen = set()

    # ── Tech entities first (higher confidence for this domain)
    for entity in tech_entities:

        if not is_valid_entity(entity["name"]):
            continue

        corrected = fix_entity_type(
            entity["name"],
            entity.get("type", "PRODUCT")
        )
        if corrected is None:
            continue

        entity = dict(entity)
        entity["type"] = corrected
        key = entity["name"].lower()

        if key not in seen:
            merged.append(entity)
            seen.add(key)

    # ── spaCy entities — deduplicate against tech
    for entity in spacy_entities:

        key = entity["name"].lower()

        # Skip compound "X and Y" — spaCy sometimes joins two entities
        if " and " in key:
            continue

        if not is_valid_entity(entity["name"]):
            continue

        corrected = fix_entity_type(
            entity["name"],
            entity.get("type", "ORGANIZATION")
        )
        if corrected is None:
            continue

        if key not in seen:
            entity = dict(entity)
            entity["type"] = corrected
            merged.append(entity)
            seen.add(key)

    return merged