from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

_PDF_MIN_CHARS_PER_PAGE = 100


class ZimExtractor:
    """Extract and chunk text from a Kiwix .zim archive."""

    NOISE_TAGS = ["script", "style", "nav", "header", "footer", "figure"]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        ocr_engine=None,
        quality_filter=None,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.ocr_engine = ocr_engine
        self.quality_filter = quality_filter  # ChunkFilter | None

    def sanitize(self, text: str) -> str:
        return "".join(
            ch if ch >= " " or ch in "\n\r\t" else " " for ch in text
        )

    def extract_html_blocks(self, html_bytes: bytes) -> list[dict]:
        soup = BeautifulSoup(html_bytes.decode("utf-8", errors="ignore"), "html.parser")
        blocks = []
        accepted_el = soup.find(class_="accepted-answer")
        if accepted_el:
            for tag in accepted_el(self.NOISE_TAGS):
                tag.decompose()
            text = self.sanitize(accepted_el.get_text(separator=" ", strip=True))
            accepted_el.decompose()
            if text:
                blocks.append({"text": text, "is_accepted": True})
        for tag in soup(self.NOISE_TAGS):
            tag.decompose()
        main_text = self.sanitize(soup.get_text(separator=" ", strip=True))
        if main_text:
            blocks.append({"text": main_text, "is_accepted": False})
        return blocks

    def extract_pdf_text(self, pdf_bytes: bytes) -> tuple[str, bool]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = self.sanitize(" ".join(pages)).strip()
        total = len(reader.pages)
        avg_chars = len(full_text) / total if total else 0
        return full_text, avg_chars < _PDF_MIN_CHARS_PER_PAGE

    def _should_keep(self, chunk: str) -> bool:
        if self.quality_filter is None:
            return True
        return self.quality_filter.is_clean(chunk)

    def _yield_blocks(
        self, blocks: list[dict], source: str, title: str,
        counts: dict | None = None,
    ) -> Iterator[dict]:
        for block in blocks:
            if len(block["text"]) < 150:
                continue
            for chunk in self.splitter.split_text(block["text"]):
                if self._should_keep(chunk):
                    yield {
                        "text": chunk,
                        "source": source,
                        "title": title,
                        "is_accepted": block.get("is_accepted", False),
                    }
                elif counts is not None:
                    counts["filtered"] += 1

    def iter_chunks(
        self,
        archive,
        entry_offset: int = 0,
        entry_limit: int = 0,
    ) -> Iterator[dict]:
        """
        Yield chunk dicts from a libzim.Archive.
        Prints progress and a summary to stdout (same as original extract_zim.py).
        """
        total = archive.all_entry_count
        start = entry_offset
        end = min(entry_offset + entry_limit, total) if entry_limit > 0 else total
        counts: dict[str, int] = {
            "skipped": 0, "html": 0, "pdf": 0,
            "pdf_scanned": 0, "pdf_error": 0, "filtered": 0,
        }

        for i in range(start, end):
            if (i - start) % 500 == 0:
                print(f"\r  {i - start:,} / {end - start:,} entries scanned ...",
                      end="", flush=True)

            try:
                entry = archive._get_entry_by_id(i)
            except Exception:
                counts["skipped"] += 1
                continue

            if entry.is_redirect:
                counts["skipped"] += 1
                continue

            try:
                item = entry.get_item()
            except Exception:
                counts["skipped"] += 1
                continue

            mime = item.mimetype
            content = bytes(item.content)
            title = entry.title or entry.path

            if "text/html" in mime:
                blocks = self.extract_html_blocks(content)
                if not any(len(b["text"]) >= 150 for b in blocks):
                    counts["skipped"] += 1
                    continue
                counts["html"] += 1
                yield from self._yield_blocks(blocks, entry.path, title, counts)

            elif ("application/json" in mime
                  and entry.path.startswith("videos/")
                  and entry.path.endswith(".json")):
                try:
                    data = json.loads(content)
                except Exception:
                    counts["skipped"] += 1
                    continue
                desc = data.get("description", "").strip()
                vtitle = data.get("title", title).strip()
                text = f"{vtitle}\n\n{desc}" if desc else vtitle
                if len(text) < 150:
                    counts["skipped"] += 1
                    continue
                counts["video_desc"] = counts.get("video_desc", 0) + 1
                for chunk in self.splitter.split_text(text):
                    if self._should_keep(chunk):
                        yield {"text": chunk, "source": entry.path, "title": vtitle}
                    else:
                        counts["filtered"] += 1

            elif "application/json" in mime and "page_content_" in entry.path:
                try:
                    html_body = json.loads(content).get("htmlBody", "")
                except Exception:
                    counts["skipped"] += 1
                    continue
                if not html_body:
                    counts["skipped"] += 1
                    continue
                blocks = self.extract_html_blocks(html_body.encode("utf-8"))
                if not any(len(b["text"]) >= 150 for b in blocks):
                    counts["skipped"] += 1
                    continue
                counts["json_html"] = counts.get("json_html", 0) + 1
                yield from self._yield_blocks(blocks, entry.path, title, counts)

            elif "application/pdf" in mime:
                if not _PYPDF_AVAILABLE:
                    counts["skipped"] += 1
                    continue
                try:
                    text, is_scanned = self.extract_pdf_text(content)
                except Exception:
                    counts["pdf_error"] += 1
                    continue
                if is_scanned or len(text) < 150:
                    if self.ocr_engine is not None:
                        try:
                            from kiwix_rag.ocr import ocr_pdf
                            text = self.sanitize(ocr_pdf(content, self.ocr_engine))
                            if len(text) < 150:
                                counts["pdf_scanned"] += 1
                                continue
                            counts["pdf_ocr"] = counts.get("pdf_ocr", 0) + 1
                        except Exception:
                            counts["pdf_scanned"] += 1
                            continue
                    else:
                        counts["pdf_scanned"] += 1
                        continue
                else:
                    counts["pdf"] += 1
                for chunk in self.splitter.split_text(text):
                    if self._should_keep(chunk):
                        yield {"text": chunk, "source": entry.path, "title": title}
                    else:
                        counts["filtered"] += 1

            else:
                counts["skipped"] += 1

        print()
        print(f"  HTML pages:        {counts['html']:,}")
        if counts.get("video_desc"):
            print(f"  Video desc (JSON): {counts['video_desc']:,}")
        if counts.get("json_html"):
            print(f"  JSON pages:        {counts['json_html']:,}")
        print(f"  PDFs (text):       {counts['pdf']:,}")
        if counts.get("pdf_ocr"):
            print(f"  PDFs (OCR'd):      {counts['pdf_ocr']:,}")
        if counts["pdf_scanned"]:
            print(f"  PDFs (scanned/empty, skipped): {counts['pdf_scanned']:,}")
        if counts["pdf_error"]:
            print(f"  PDFs (error, skipped):         {counts['pdf_error']:,}")
        print(f"  Other/redirects:   {counts['skipped']:,}")
        if counts["filtered"]:
            print(f"  Quality-filtered:  {counts['filtered']:,}")
