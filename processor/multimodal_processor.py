"""
processor/multimodal_processor.py

Step 3 of the pipeline — Multimodal Understanding.

Takes the JSON produced by PDFParser (Step 2) and enriches it:
  • Images → sent to Mistral's vision model → natural-language description
  • Tables → converted into clean Markdown for easier LLM/embedding consumption

The output is saved as a new *_processed.json, ready for Step 4 (Chunking).

Run standalone:
    python -m processor.multimodal_processor

Requires:
    MISTRAL_API_KEY set in a .env file in the project root (see .env.example)
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
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Pixtral was merged into Mistral Small 4 (March 2026) — vision is now built
# straight into the small model, no separate "pixtral-*" model needed anymore.
VISION_MODEL = "mistral-small-latest"

REQUEST_DELAY_SECONDS = 1.0  # small pause between image calls to be gentle on rate limits

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

IMAGE_PROMPT_TEMPLATE = """You are analyzing a figure extracted from a document page.

Existing caption (may be missing or unhelpful): "{caption}"

In 2-4 sentences, describe what this image actually shows. If it is a chart \
or graph, name the chart type, what is being measured on each axis, and the \
key trend. If it is a diagram, describe its components and how they connect. \
If it is a photo, describe the main subject.

Only describe what you can actually see in the image — do not invent numbers \
or labels you can't read clearly. Respond with plain text only, no markdown, \
no preamble."""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalProcessor:

    def __init__(
        self,
        model: str = VISION_MODEL,
        api_key: str | None = None,
        skip_existing: bool = True,
    ):
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Add it to a .env file in the project "
                "root (see .env.example) or pass api_key= explicitly."
            )

        self.model_name = model
        self.skip_existing = skip_existing
        self.llm = ChatMistralAI(model=model, api_key=api_key, temperature=0.2)
        logger.info(f"MultimodalProcessor ready  (model={model})")

    # ── Public API ───────────────────────────────────────────────────────────

    def process_file(self, json_path: str) -> str:
        """Load a *_parsed.json file, enrich it, save as *_processed.json."""
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

    # ── Private: Images ──────────────────────────────────────────────────────

    def _process_images(self, images: list) -> list:
        if not images:
            return images

        logger.info(f"Processing {len(images)} image(s) with {self.model_name}...")

        for i, img in enumerate(images):
            if self.skip_existing and img.get("description"):
                logger.info(f"  [{i + 1}/{len(images)}] Already described, skipping")
                continue

            try:
                description = self._describe_image(img["image_path"], img.get("caption", ""))
                img["description"] = description
                logger.info(f"  [{i + 1}/{len(images)}] ✅  {img['image_path']}")
            except Exception as exc:
                img["description"] = "Description unavailable (vision call failed)"
                logger.warning(f"  [{i + 1}/{len(images)}] ⚠️  Failed on {img['image_path']}: {exc}")

            time.sleep(REQUEST_DELAY_SECONDS)

        return images

    def _describe_image(self, image_path: str, caption: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file missing on disk: {path}")

        mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
        b64_image = base64.b64encode(path.read_bytes()).decode("utf-8")

        prompt = IMAGE_PROMPT_TEMPLATE.format(caption=caption or "No caption found")

        # NOTE: Mistral's vision API expects this exact dict shape for image_url.
        # LangChain's newer "standard content block" format (create_image_block)
        # is NOT yet compatible with ChatMistralAI — HumanMessage content is
        # passed straight through to the Mistral API unchanged, so we build the
        # block manually in Mistral's native shape instead.
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_image}"},
        ])

        response = self.llm.invoke([message])
        return response.content.strip()

    # ── Private: Tables ──────────────────────────────────────────────────────

    def _process_tables(self, tables: list) -> list:
        if not tables:
            return tables

        logger.info(f"Converting {len(tables)} table(s) to Markdown...")

        for t in tables:
            t["markdown"] = self._table_to_markdown(
                headers=t.get("headers", []),
                rows=t.get("rows", []),
                caption=t.get("caption", ""),
            )

        return tables

    @staticmethod
    def _table_to_markdown(headers: list, rows: list, caption: str) -> str:
        if not headers:
            return "*Empty table*"

        clean_headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(headers)]

        lines = []
        if caption and caption != "No caption found":
            lines.append(f"**{caption}**\n")

        lines.append("| " + " | ".join(clean_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")

        for row in rows:
            padded = list(row) + [""] * (len(clean_headers) - len(row))
            cells = [str(c).replace("\n", " ").replace("|", "\\|") for c in padded[:len(clean_headers)]]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ── Private: Save ────────────────────────────────────────────────────────

    @staticmethod
    def _save(data: dict, original_path: Path) -> Path:
        stem = original_path.stem.replace("_parsed", "")
        out_path = original_path.parent / f"{stem}_processed.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"💾  Processed JSON saved → {out_path}")
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE RUN  (python -m processor.multimodal_processor)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_JSON = "output/sample_parsed.json"

    processor = MultimodalProcessor()
    output_path = processor.process_file(INPUT_JSON)

    print("\n" + "─" * 50)
    print("  MULTIMODAL PROCESSING COMPLETE")
    print("─" * 50)
    print(f"  Output saved : {output_path}")
    print("─" * 50)
