"""
parser/pdf_parser.py  —  Step 2: PDF Parsing

Optimizations vs original:
  1. Better heading detection  — uses font size + bold flag + ALL-CAPS heuristic
  2. Text cleaning             — fixes soft-hyphen line breaks, normalises whitespace
  3. Section name cleaning     — strips trailing dots / page-number noise, caps length
  4. Image deduplication       — same xref appearing on multiple pages extracted once
  5. Image size filtering      — decorative icons / borders (< 80×80 px or < 2 KB)
                                 are skipped so they never reach Mistral vision
  6. Per-image section name    — derived from the headings on that page, not "Figure N"
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Images smaller than these are decorative (icons, bullets, borders) — skip them
MIN_IMAGE_WIDTH  = 80    # pixels
MIN_IMAGE_HEIGHT = 80    # pixels
MIN_IMAGE_BYTES  = 2048  # 2 KB

# Font-size threshold for headings (spans >= this size are headings)
HEADING_FONT_SIZE = 13.5

# fitz bold flag bitmask (bit 4)
FITZ_BOLD_FLAG = 16


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    page_number:  int
    section_name: str
    content:      str
    block_type:   str = "text"


@dataclass
class TableBlock:
    page_number:  int
    section_name: str
    headers:      list
    rows:         list
    raw_text:     str
    caption:      str = "No caption found"
    block_type:   str = "table"


@dataclass
class ImageBlock:
    page_number:  int
    section_name: str
    image_index:  int
    image_path:   str
    caption:      str
    width:        int
    height:       int
    block_type:   str = "image"


@dataclass
class ParsedDocument:
    file_path:   str
    file_name:   str
    total_pages: int
    texts:  list = field(default_factory=list)
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PARSER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PDFParser:

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self._setup_directories()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder ready → {self.output_dir.resolve()}")

    def _clean_output_dir(self):
        """Wipe images + JSON left over from a previous run."""
        removed = 0
        if self.images_dir.exists():
            for f in self.images_dir.iterdir():
                if f.is_file():
                    f.unlink()
                    removed += 1
        for f in self.output_dir.glob("*.json"):
            f.unlink()
            removed += 1
        if removed:
            logger.info(f"Cleaned {removed} old file(s) from previous run")

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: str, clean: bool = True) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if clean:
            self._clean_output_dir()

        logger.info(f"Opening PDF → {pdf_path.name}")
        doc = fitz.open(str(pdf_path))

        result = ParsedDocument(
            file_path   = str(pdf_path.resolve()),
            file_name   = pdf_path.name,
            total_pages = len(doc),
        )

        result.texts  = self._extract_text(doc)
        result.tables = self._extract_tables(str(pdf_path))
        result.images = self._extract_images(doc, pdf_stem=pdf_path.stem)

        doc.close()

        logger.info("Parsing complete!")
        logger.info(f"  Text blocks : {len(result.texts)}")
        logger.info(f"  Tables      : {len(result.tables)}")
        logger.info(f"  Images      : {len(result.images)}")

        return result

    def save_to_json(self, parsed: ParsedDocument) -> str:
        stem = Path(parsed.file_name).stem
        out  = self.output_dir / f"{stem}_parsed.json"

        data = {
            "file_name":   parsed.file_name,
            "file_path":   parsed.file_path,
            "total_pages": parsed.total_pages,
            "summary": {
                "text_blocks": len(parsed.texts),
                "tables":      len(parsed.tables),
                "images":      len(parsed.images),
            },
            "texts":  [asdict(t) for t in parsed.texts],
            "tables": [asdict(t) for t in parsed.tables],
            "images": [asdict(i) for i in parsed.images],
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON saved → {out}")
        return str(out)

    # ── Helper: Heading Detection ─────────────────────────────────────────────

    def _is_heading_span(self, span: dict) -> bool:
        """
        More accurate heading detection for technical manuals.
        Three signals: large font, smaller-bold, or ALL-CAPS short line.
        """
        size  = span.get("size", 0)
        flags = span.get("flags", 0)
        text  = span.get("text", "").strip()

        if not text or len(text) > 120:
            return False

        is_bold = bool(flags & FITZ_BOLD_FLAG)

        # Signal 1: clearly large font
        if size >= HEADING_FONT_SIZE:
            return True

        # Signal 2: medium size + bold + short  (subsection headings)
        if size >= 11 and is_bold and len(text.split()) <= 10:
            return True

        # Signal 3: ALL-CAPS short lines (common heading style in Dell manuals)
        if (
            text.isupper()
            and 3 < len(text) < 80
            and len(text.split()) <= 8
        ):
            return True

        return False

    # ── Helper: Text Cleaning ─────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Fix common PDF text extraction artefacts:
          - Soft-hyphen line breaks:  "tech-\nnology"  →  "technology"
          - Lone newlines mid-sentence (PDF line-wrap artefact)
          - Repeated whitespace
          - Control characters
        """
        # Soft hyphen at end of line: join "tech-\nnology" → "technology"
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        # Other lone newlines (not paragraph breaks) → space
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Multiple spaces/tabs → single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Control characters (except newlines)
        text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    # ── Helper: Section Name Cleaning ─────────────────────────────────────────

    @staticmethod
    def _clean_section_name(name: str) -> str:
        """
        Normalise section names extracted from headings:
          - Strip trailing dots, spaces
          - Remove trailing TOC page numbers:  "Chapter 3 ........ 45"
          - Cap at 80 characters
        """
        name = name.strip().rstrip('.')
        # TOC-style page number at end
        name = re.sub(r'\s*\.{2,}\s*\d+\s*$', '', name)
        name = name.strip()
        if len(name) > 80:
            name = name[:80].rsplit(' ', 1)[0] + '...'
        return name

    # ── Helper: Page Section ──────────────────────────────────────────────────

    def _get_section_from_page(self, doc: fitz.Document, page_num: int) -> str:
        """
        Derive a section name for a page by finding the first heading-like
        span in the page's text blocks.
        Falls back to "Page N" if no heading is found.
        """
        page = doc[page_num]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if self._is_heading_span(span):
                        text = span.get("text", "").strip()
                        if text and 3 < len(text) < 120:
                            return self._clean_section_name(text)
        return f"Page {page_num + 1}"

    # ── Private: Text Extraction ──────────────────────────────────────────────

    def _extract_text(self, doc: fitz.Document) -> list:
        text_blocks: list[TextBlock] = []
        current_section = "Introduction"

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_number = page_num + 1

            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue

                block_text = ""
                is_heading = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            block_text += text + " "
                            if self._is_heading_span(span):
                                is_heading = True

                block_text = self._clean_text(block_text)

                if not block_text:
                    continue

                # Update current section when a heading is found
                if is_heading and len(block_text) < 120:
                    current_section = self._clean_section_name(block_text)

                # Skip very short non-heading blocks (noise: page numbers, stray chars)
                if len(block_text) < 20 and not is_heading:
                    continue

                text_blocks.append(TextBlock(
                    page_number  = page_number,
                    section_name = current_section,
                    content      = block_text,
                    block_type   = "heading" if is_heading else "text",
                ))

        return text_blocks

    # ── Private: Table Extraction ─────────────────────────────────────────────

    def _extract_tables(self, pdf_path: str) -> list:
        table_blocks: list[TableBlock] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_number  = page_num + 1
                table_objects = page.find_tables()

                for t_idx, table_obj in enumerate(table_objects):
                    table_data = table_obj.extract()
                    if not table_data:
                        continue

                    caption = self._find_table_caption(page, table_obj.bbox)
                    headers = [str(h) if h is not None else "" for h in table_data[0]]
                    rows    = [
                        [str(c) if c is not None else "" for c in row]
                        for row in table_data[1:]
                    ]

                    raw_text  = " | ".join(headers) + "\n"
                    raw_text += "\n".join(" | ".join(row) for row in rows)

                    section = (
                        caption
                        if caption != "No caption found"
                        else f"Table {t_idx + 1} (Page {page_number})"
                    )

                    table_blocks.append(TableBlock(
                        page_number  = page_number,
                        section_name = section,
                        headers      = headers,
                        rows         = rows,
                        raw_text     = raw_text,
                        caption      = caption,
                    ))

        return table_blocks

    def _find_table_caption(self, page, bbox: tuple) -> str:
        x0, top, x1, bottom = bbox
        CAPTION_STARTS = ("table", "tab.", "tab ")

        search_regions = [
            (x0, max(0, top - 50),              x1, top),
            (x0, bottom, x1, min(page.height, bottom + 50)),
        ]

        for region in search_regions:
            try:
                cropped = page.crop(region)
                text    = cropped.extract_text()
                if not text:
                    continue
                for line in text.strip().splitlines():
                    line = line.strip()
                    if line.lower().startswith(CAPTION_STARTS):
                        return line
            except Exception:
                continue

        return "No caption found"

    # ── Private: Image Extraction ─────────────────────────────────────────────

    def _extract_images(self, doc: fitz.Document, pdf_stem: str) -> list:
        image_blocks: list[ImageBlock] = []
        global_idx  = 0
        seen_xrefs: set[int] = set()   # deduplicate: same xref on multiple pages

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_number = page_num + 1

            # Derive section name from headings on this page
            section = self._get_section_from_page(doc, page_num)

            for local_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]

                # Skip images already extracted from a previous page
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]
                    img_ext    = base_image["ext"]

                    # ── Byte-size filter (very small = icon / border)
                    if len(img_bytes) < MIN_IMAGE_BYTES:
                        logger.info(
                            f"  Skip tiny image "
                            f"({len(img_bytes)} bytes) on page {page_number}"
                        )
                        continue

                    # ── Dimension filter
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    w, h    = pil_img.size
                    pil_img.close()

                    if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                        logger.info(
                            f"  Skip small image "
                            f"({w}×{h} px) on page {page_number}"
                        )
                        continue

                    # ── Save to disk
                    filename  = (
                        f"{pdf_stem}_p{page_number}"
                        f"_img{local_idx + 1}.{img_ext}"
                    )
                    save_path = self.images_dir / filename
                    save_path.write_bytes(img_bytes)

                    caption = self._find_image_caption(page)

                    image_blocks.append(ImageBlock(
                        page_number  = page_number,
                        section_name = section,
                        image_index  = global_idx,
                        image_path   = str(save_path),
                        caption      = caption,
                        width        = w,
                        height       = h,
                    ))
                    global_idx += 1
                    logger.info(
                        f"  Saved image ({w}×{h} px) → {filename}"
                    )

                except Exception as exc:
                    logger.warning(
                        f"  Could not extract image {local_idx} "
                        f"on page {page_number}: {exc}"
                    )

        return image_blocks

    def _find_image_caption(self, page: fitz.Page) -> str:
        CAPTION_STARTS = (
            "figure", "fig.", "fig ", "chart", "diagram", "image"
        )
        try:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text.lower().startswith(CAPTION_STARTS):
                            return text
        except Exception:
            pass
        return "No caption found"