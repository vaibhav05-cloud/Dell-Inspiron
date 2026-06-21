"""
processor/multimodal_processor.py  —  Step 3: Multimodal Processing

Optimizations vs original:
  1. Entity-focused image prompt  — explicitly asks Mistral to name products,
                                    components, brands, specs (improves entity extraction
                                    from image chunks downstream)
  2. Section context in prompt    — tells Mistral which section the image belongs to
  3. Image size filtering         — skip images < 100×100 px before Mistral API call
                                    (saves tokens + time; small images add no signal)
  4. table text_summary field     — natural-language prose version of each table added
                                    alongside the existing markdown field.
                                    Entity extractor works much better on prose than
                                    on markdown pipe-tables.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VISION_MODEL           = "mistral-small-latest"
REQUEST_DELAY_SECONDS  = 1.0

# Images below these thresholds are too small to contain useful entity info
# (headers, icons, watermarks, decorative rules)
MIN_IMG_DESCRIBE_W = 100  # pixels
MIN_IMG_DESCRIBE_H = 100  # pixels

IMAGE_MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}

# ── ENTITY-FOCUSED IMAGE PROMPT ───────────────────────────────────────────────
# Original prompt asked for a generic description.
# New prompt instructs Mistral to explicitly name:
#   products, models, companies, hardware components, software, specs.
# This makes the downstream image chunks much richer for entity extraction.
IMAGE_PROMPT_TEMPLATE = """\
You are extracting structured information from a figure in a technical product manual.

Document section : "{section_name}"
Caption provided : "{caption}"

Write 3-5 sentences describing exactly what this figure shows.
Follow these rules strictly:

1. NAME all specific products and models visible
   (e.g. "Dell Inspiron 1150 laptop", "Intel Celeron M processor", "Windows XP").
2. NAME all hardware components and ports
   (e.g. "USB 2.0 port", "VGA connector", "BIOS chip", "SATA hard drive").
3. NAME all software elements
   (e.g. "Windows XP Device Manager", "Microsoft Office 2003 setup wizard").
4. For diagrams: name each component and describe how it connects to others.
5. For hardware photos: list every labeled part and its location on the device.
6. Include any part numbers, version numbers, or model numbers you can read.

Write in plain prose only — no bullet points, no markdown, no preamble.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalProcessor:

    def __init__(
        self,
        model:          str       = VISION_MODEL,
        api_key:        str | None = None,
        skip_existing:  bool      = True,
    ):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Add it to a .env file "
                "or pass api_key= explicitly."
            )

        self.model_name    = model
        self.skip_existing = skip_existing
        self.llm = ChatMistralAI(
            model    = model,
            api_key  = api_key,
            temperature = 0.2,
        )
        logger.info(f"MultimodalProcessor ready  (model={model})")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(self, json_path: str) -> str:
        """Load *_parsed.json, enrich images + tables, save as *_processed.json."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Parsed JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded → {json_path.name}")

        data["images"] = self._process_images(data.get("images", []))
        data["tables"] = self._process_tables(data.get("tables", []))

        out_path = self._save(data, json_path)
        return str(out_path)

    # ── Private: Images ───────────────────────────────────────────────────────

    def _process_images(self, images: list) -> list:
        if not images:
            return images

        logger.info(
            f"Processing {len(images)} image(s) with {self.model_name}..."
        )

        described = 0

        for i, img in enumerate(images):
            prefix = f"  [{i + 1}/{len(images)}]"

            # Skip if already has a description (resume-safe)
            if self.skip_existing and img.get("description"):
                logger.info(f"{prefix} Already described — skipping")
                continue

            # Skip images that are too small to contain useful information
            w = img.get("width", 0)
            h = img.get("height", 0)
            if w < MIN_IMG_DESCRIBE_W or h < MIN_IMG_DESCRIBE_H:
                img["description"] = ""   # empty → chunker skips this image
                logger.info(
                    f"{prefix} Skip {w}×{h} px image "
                    f"(below {MIN_IMG_DESCRIBE_W}×{MIN_IMG_DESCRIBE_H} threshold)"
                )
                continue

            try:
                description = self._describe_image(
                    image_path   = img["image_path"],
                    caption      = img.get("caption", ""),
                    section_name = img.get("section_name", "Unknown Section"),
                )
                img["description"] = description
                described += 1
                logger.info(f"{prefix} Described → {img['image_path']}")

            except Exception as exc:
                img["description"] = ""
                logger.warning(f"{prefix} Failed on {img['image_path']}: {exc}")

            time.sleep(REQUEST_DELAY_SECONDS)

        logger.info(
            f"Image processing done: {described}/{len(images)} described"
        )
        return images

    def _describe_image(
        self,
        image_path:   str,
        caption:      str,
        section_name: str = "Unknown Section",
    ) -> str:
        """Send one image to Mistral vision and return a description."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file missing on disk: {path}")

        mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
        b64_image = base64.b64encode(path.read_bytes()).decode("utf-8")

        prompt = IMAGE_PROMPT_TEMPLATE.format(
            section_name = section_name or "Unknown Section",
            caption      = caption      or "No caption available",
        )

        # NOTE: LangChain's newer content-block API is not yet compatible with
        # ChatMistralAI — we pass the Mistral-native shape directly.
        message = HumanMessage(content=[
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_image}"},
        ])

        response = self.llm.invoke([message])
        return response.content.strip()

    # ── Private: Tables ───────────────────────────────────────────────────────

    def _process_tables(self, tables: list) -> list:
        if not tables:
            return tables

        logger.info(f"Converting {len(tables)} table(s) ...")

        for t in tables:
            headers      = t.get("headers", [])
            rows         = t.get("rows",    [])
            section_name = t.get("section_name", "Unknown Section")
            caption      = t.get("caption", "")

            # Existing markdown (structured format for LLM reading)
            t["markdown"]     = self._table_to_markdown(headers, rows, caption)

            # NEW: natural-language prose summary (better for entity extraction)
            t["text_summary"] = self._table_to_text_summary(
                headers, rows, section_name
            )

        return tables

    @staticmethod
    def _table_to_markdown(headers: list, rows: list, caption: str) -> str:
        """Convert table to GitHub-flavoured Markdown."""
        if not headers:
            return "*Empty table*"

        clean_h = [h.strip() or f"Column {i + 1}" for i, h in enumerate(headers)]

        lines = []
        if caption and caption != "No caption found":
            lines.append(f"**{caption}**\n")

        lines.append("| " + " | ".join(clean_h) + " |")
        lines.append("| " + " | ".join(["---"] * len(clean_h)) + " |")

        for row in rows:
            padded = list(row) + [""] * (len(clean_h) - len(row))
            cells  = [
                str(c).replace("\n", " ").replace("|", "\\|")
                for c in padded[:len(clean_h)]
            ]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def _table_to_text_summary(
        headers:      list,
        rows:         list,
        section_name: str,
    ) -> str:
        """
        Convert a table to natural-language prose for entity extraction.

        Markdown pipe-tables contain lots of formatting noise (|, ---, etc.)
        that confuses NER models. This prose version makes entities like
        "Intel Celeron M" and "Windows XP" clearly visible in plain text.

        Example output:
          Table in section 'Technical Specifications' listing: Processor, Speed,
          Cache. Processor: Intel Celeron M 360J, Speed: 1.4 GHz, Cache: 1 MB.
          Processor: Intel Pentium M 740, Speed: 1.73 GHz, Cache: 2 MB.
        """
        if not headers or not rows:
            return ""

        clean_h = [str(h).strip() for h in headers if str(h).strip()]
        if not clean_h:
            return ""

        parts = [
            f"Table in section '{section_name}' "
            f"listing: {', '.join(clean_h)}."
        ]

        for row in rows[:12]:  # cap at 12 rows to keep chunk size reasonable
            row_facts = []
            for i, cell in enumerate(row):
                if i >= len(headers):
                    break
                cell_str = str(cell).strip() if cell else ""
                # Skip empty / placeholder cells
                if not cell_str or cell_str.lower() in (
                    "none", "-", "n/a", "na", "", "—"
                ):
                    continue
                header = str(headers[i]).strip() or f"Column {i + 1}"
                row_facts.append(f"{header}: {cell_str}")
            if row_facts:
                parts.append(", ".join(row_facts) + ".")

        return " ".join(parts)

    # ── Private: Save ─────────────────────────────────────────────────────────

    @staticmethod
    def _save(data: dict, original_path: Path) -> Path:
        stem     = original_path.stem.replace("_parsed", "")
        out_path = original_path.parent / f"{stem}_processed.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Processed JSON saved → {out_path}")
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_JSON = "output/sample_parsed.json"

    processor  = MultimodalProcessor()
    output_path = processor.process_file(INPUT_JSON)

    print("\n" + "─" * 50)
    print("  MULTIMODAL PROCESSING COMPLETE")
    print("─" * 50)
    print(f"  Output saved : {output_path}")
    print("─" * 50)