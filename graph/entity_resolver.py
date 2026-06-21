"""
graph/entity_resolver.py  —  Phase 9: Better Entity Resolution

Merges entity name variants into canonical forms:
  "Dell" + "DELL" + "Dell Computer Corporation"  →  "Dell"
  "Windows XP Professional" + "WinXP"            →  "Windows XP"
  "Intel Corp" + "Intel Corporation"             →  "Intel"
  "Celeron" + "Intel Celeron M"                  →  "Intel Celeron M"

Three-pass approach:
  Pass 1 — Hardcoded canonical lookup  (known common variants)
  Pass 2 — Normalization grouping      (strip legal suffixes, case-insensitive)
  Pass 3 — Fuzzy matching              (difflib, within same entity type only)

No extra dependencies — uses only Python stdlib (difflib, re, collections).

Usage in run_graph.py:
    resolver = EntityResolver()
    entities, name_map = resolver.resolve(entities)
    ...
    relationships = resolver.apply_to_relationships(relationships, name_map)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  HARDCODED CANONICAL NAMES  (Pass 1)
#  lowercase variant → canonical display name
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_MAP: dict[str, str] = {
    # ── Dell
    "dell inc":                   "Dell",
    "dell inc.":                  "Dell",
    "dell computer":              "Dell",
    "dell computers":             "Dell",
    "dell computer corporation":  "Dell",
    "dell corp":                  "Dell",
    "dell corporation":           "Dell",
    "dell technologies":          "Dell",
    # ── Intel
    "intel corp":                 "Intel",
    "intel corp.":                "Intel",
    "intel corporation":          "Intel",
    "intel inc":                  "Intel",
    # ── Microsoft
    "microsoft corp":             "Microsoft",
    "microsoft corp.":            "Microsoft",
    "microsoft corporation":      "Microsoft",
    "microsoft inc":              "Microsoft",
    # ── Windows
    "win xp":                     "Windows XP",
    "winxp":                      "Windows XP",
    "windows xp professional":    "Windows XP",
    "windows xp home edition":    "Windows XP",
    "windows xp home":            "Windows XP",
    "windows xp pro":             "Windows XP",
    "microsoft windows xp":       "Windows XP",
    "windows® xp":                "Windows XP",
    # ── Office
    "ms office":                  "Microsoft Office",
    "microsoft office xp":        "Microsoft Office",
    "microsoft office 2003":      "Microsoft Office",
    # ── Processors
    "celeron":                    "Intel Celeron",
    "intel celeron":              "Intel Celeron",
    "celeron m":                  "Intel Celeron M",
    "intel celeron m":            "Intel Celeron M",
    "pentium m":                  "Intel Pentium M",
    "intel pentium m":            "Intel Pentium M",
    "intel pentium":              "Intel Pentium",
    # ── Other common tech
    "nvidia corp":                "NVIDIA",
    "nvidia corporation":         "NVIDIA",
    "amd inc":                    "AMD",
    "advanced micro devices":     "AMD",
}

# ─────────────────────────────────────────────────────────────────────────────
#  LEGAL SUFFIX PATTERN  (Pass 2)
#  These are stripped before grouping, so "Dell Inc" groups with "Dell"
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_SUFFIX_RE = re.compile(
    r'\s+(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|'
    r'llc|l\.l\.c\.?|co\.?|company|computer|computers|'
    r'technologies|technology|systems|group|international|worldwide)$',
    re.IGNORECASE,
)

# Fuzzy similarity threshold for Pass 3 (0–100)
# 88 = high enough to avoid false merges, low enough to catch real variants
FUZZY_THRESHOLD = 88


# ─────────────────────────────────────────────────────────────────────────────
#  ENTITY RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

class EntityResolver:

    def resolve(self, entities: list) -> tuple[list, dict]:
        """
        Main entry point. Returns:
            resolved_entities — deduplicated list with canonical names
            name_map          — dict[original_lower → canonical]
                                (pass this to apply_to_relationships)
        """
        before   = len(entities)
        name_map = self._build_name_map(entities)

        # Apply canonical names and deduplicate
        seen:     set  = set()
        resolved: list = []

        for entity in entities:
            canonical = name_map.get(entity["name"].lower(), entity["name"])
            key       = canonical.lower()

            if key not in seen:
                seen.add(key)
                entity         = dict(entity)
                entity["name"] = canonical
                resolved.append(entity)

        after = len(resolved)
        logger.info(
            f"Entity resolution: {before} → {after} entities "
            f"({before - after} merged)"
        )
        return resolved, name_map

    # ─────────────────────────────────────────────────────────────────────────

    def apply_to_relationships(
        self,
        relationships: list,
        name_map:      dict,
    ) -> list:
        """
        Update source/target in every relationship using the canonical name map.
        Also removes self-loops that can be created when two different entity
        names are merged into the same canonical name.
        """
        for rel in relationships:
            rel["source"] = name_map.get(rel["source"].lower(), rel["source"])
            rel["target"] = name_map.get(rel["target"].lower(), rel["target"])

        # Remove self-loops (e.g., Dell → Dell after merging "Dell Inc" → "Dell")
        before = len(relationships)
        relationships = [
            r for r in relationships
            if r["source"].lower() != r["target"].lower()
        ]
        loops = before - len(relationships)
        if loops:
            logger.info(f"Removed {loops} self-loop relationship(s) after merging")

        return relationships

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNAL: BUILD NAME MAP
    # ─────────────────────────────────────────────────────────────────────────

    def _build_name_map(self, entities: list) -> dict[str, str]:
        """
        Three-pass algorithm that builds original_lower → canonical mapping.
        """
        name_map:       dict[str, str] = {}
        entity_names:   list[str]      = [e["name"] for e in entities]

        # For Pass 3: remember each entity's type (same-type fuzzy matching only)
        type_map: dict[str, str] = {
            e["name"].lower(): e.get("type", "")
            for e in entities
        }

        # ── Pass 1: Hardcoded canonical lookup ────────────────────────────────
        for name in entity_names:
            key = name.lower()
            if key in CANONICAL_MAP:
                name_map[key] = CANONICAL_MAP[key]

        unmapped = [n for n in entity_names if n.lower() not in name_map]

        # ── Pass 2: Normalization grouping ────────────────────────────────────
        # Strip legal suffixes, lowercase → group variants together
        groups: dict[str, list[str]] = defaultdict(list)
        for name in unmapped:
            normalized = self._normalize(name)
            groups[normalized].append(name)

        for normalized, variants in groups.items():
            canonical = self._pick_canonical(variants)
            for variant in variants:
                name_map[variant.lower()] = canonical

        # ── Pass 3: Fuzzy matching (same entity type only) ────────────────────
        # After passes 1+2, all canonical names are determined.
        # Now find pairs of canonicals that are very similar but weren't caught.
        canonical_names = list({v for v in name_map.values()})

        by_type: dict[str, list[str]] = defaultdict(list)
        for cname in canonical_names:
            etype = type_map.get(cname.lower(), "")
            by_type[etype].append(cname)

        for etype, names in by_type.items():
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    n1, n2 = names[i], names[j]

                    if n1.lower() == n2.lower():
                        continue

                    # Skip pairs with very different lengths (unrelated entities)
                    if min(len(n1), len(n2)) / max(len(n1), len(n2)) < 0.5:
                        continue

                    if self._fuzzy_ratio(n1, n2) >= FUZZY_THRESHOLD:
                        # Keep the shorter / cleaner name as canonical
                        keep = n1 if len(n1) <= len(n2) else n2
                        drop = n2 if keep == n1 else n1

                        # Remap everything pointing to `drop` → `keep`
                        for k in list(name_map.keys()):
                            if name_map[k].lower() == drop.lower():
                                name_map[k] = keep

        return name_map

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """Strip legal suffixes and lowercase for grouping comparison."""
        name = name.strip()
        name = LEGAL_SUFFIX_RE.sub("", name)
        return name.strip().lower()

    @staticmethod
    def _fuzzy_ratio(s1: str, s2: str) -> float:
        """0–100 similarity using Python stdlib SequenceMatcher."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100

    @staticmethod
    def _pick_canonical(variants: list[str]) -> str:
        """
        From a list of name variants, pick the best canonical form.
        Strategy: most frequent → break ties by shortest (cleaner, more general).
        Example: ["Dell", "Dell", "Dell Computer Corporation"] → "Dell"
        """
        freq     = Counter(variants)
        max_freq = max(freq.values())
        top      = [n for n, f in freq.items() if f == max_freq]
        return min(top, key=len)