from __future__ import annotations

import fitz
import pdfplumber
import json
import logging
import io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    page_number: int
    section_name: str
    content: str
    block_type: str = "text"


@dataclass
class TableBlock:
    page_number: int
    section_name: str
    headers: list
    rows: list
    raw_text: str
    caption: str = "No caption found"
    block_type: str = "table"


@dataclass
class ImageBlock:
    page_number: int
    section_name: str
    image_index: int
    image_path: str
    caption: str
    width: int
    height: int
    block_type: str = "image"


@dataclass
class ParsedDocument:
    file_path: str
    file_name: str
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

    def _setup_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder ready → {self.output_dir.resolve()}")

    def _clean_output_dir(self):
        """Wipe images + JSON left over from a previous PDF before starting a new one."""
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
            logger.info(f"🧹  Cleaned {removed} old file(s) from previous run")

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
            file_path=str(pdf_path.resolve()),
            file_name=pdf_path.name,
            total_pages=len(doc),
        )

        result.texts  = self._extract_text(doc)
        result.tables = self._extract_tables(str(pdf_path))
        result.images = self._extract_images(doc, pdf_stem=pdf_path.stem)

        doc.close()

        logger.info("✅  Parsing complete!")
        logger.info(f"    Text blocks : {len(result.texts)}")
        logger.info(f"    Tables      : {len(result.tables)}")
        logger.info(f"    Images      : {len(result.images)}")

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

        logger.info(f"💾  JSON saved → {out}")
        return str(out)

    # ── Private: Text ─────────────────────────────────────────────────────────

    def _extract_text(self, doc: fitz.Document) -> list:
        text_blocks: list = []
        current_section = "Introduction"

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_number = page_num + 1

            raw_blocks = page.get_text("dict")["blocks"]

            for block in raw_blocks:
                if block["type"] != 0:
                    continue

                block_text = ""
                is_heading = False

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            block_text += text + " "
                            if span["size"] > 14:
                                is_heading = True

                block_text = block_text.strip()
                if not block_text:
                    continue

                if is_heading and len(block_text) < 120:
                    current_section = block_text

                text_blocks.append(TextBlock(
                    page_number=page_number,
                    section_name=current_section,
                    content=block_text,
                    block_type="heading" if is_heading else "text",
                ))

        return text_blocks

    # ── Private: Tables ───────────────────────────────────────────────────────

    def _extract_tables(self, pdf_path: str) -> list:
        table_blocks: list = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_number = page_num + 1

                # find_tables() gives Table objects with .bbox + .extract()
                table_objects = page.find_tables()

                for t_idx, table_obj in enumerate(table_objects):
                    table_data = table_obj.extract()

                    if not table_data:
                        continue

                    caption = self._find_table_caption(page, table_obj.bbox)

                    headers = [str(h) if h is not None else "" for h in table_data[0]]
                    rows    = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in table_data[1:]
                    ]

                    raw_text  = " | ".join(headers) + "\n"
                    raw_text += "\n".join(" | ".join(row) for row in rows)

                    section = caption if caption != "No caption found" \
                              else f"Table {t_idx + 1} (Page {page_number})"

                    table_blocks.append(TableBlock(
                        page_number=page_number,
                        section_name=section,
                        headers=headers,
                        rows=rows,
                        raw_text=raw_text,
                        caption=caption,
                    ))

        return table_blocks

    def _find_table_caption(self, page, bbox: tuple) -> str:
        x0, top, x1, bottom = bbox
        CAPTION_STARTS = ("table", "tab.", "tab ")

        search_regions = [
            (x0, max(0, top - 50),          x1, top),
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

    # ── Private: Images ───────────────────────────────────────────────────────

    def _extract_images(self, doc: fitz.Document, pdf_stem: str) -> list:
        image_blocks: list = []
        global_idx   = 0

        for page_num in range(len(doc)):
            page        = doc[page_num]
            page_number = page_num + 1

            for local_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]

                try:
                    base_image  = doc.extract_image(xref)
                    img_bytes   = base_image["image"]
                    img_ext     = base_image["ext"]

                    filename  = f"{pdf_stem}_p{page_number}_img{local_idx + 1}.{img_ext}"
                    save_path = self.images_dir / filename
                    save_path.write_bytes(img_bytes)

                    pil_img = Image.open(io.BytesIO(img_bytes))
                    w, h    = pil_img.size
                    pil_img.close()

                    caption = self._find_image_caption(page)

                    image_blocks.append(ImageBlock(
                        page_number=page_number,
                        section_name=f"Figure {global_idx + 1}",
                        image_index=global_idx,
                        image_path=str(save_path),
                        caption=caption,
                        width=w,
                        height=h,
                    ))
                    global_idx += 1
                    logger.info(f"  Saved image → {filename}")

                except Exception as exc:
                    logger.warning(f"  Could not extract image {local_idx} on page {page_number}: {exc}")

        return image_blocks

    def _find_image_caption(self, page: fitz.Page) -> str:
        CAPTION_STARTS = ("figure", "fig.", "fig ", "chart", "diagram", "image")
        try:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text.lower().startswith(CAPTION_STARTS):
                            return text
        except Exception:
            pass
        return "No caption found"