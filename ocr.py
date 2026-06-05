"""
OCR module for extract_zim.py.

Provides preprocess() and ocr_image() used when --ocr is passed.
Engine selection: --ocr-engine tesseract (default) or --ocr-engine easyocr

Dependencies:
    pip install pymupdf opencv-python-headless pillow pytesseract  # tesseract engine
    pip install easyocr                                             # easyocr engine
    brew install tesseract                                          # tesseract binary (macOS)
    sudo apt install tesseract-ocr                                  # tesseract binary (Pi 5 / Linux)
"""

import io
import os
import shutil
import numpy as np

import cv2
import fitz  # pymupdf


# ── image preprocessing ───────────────────────────────────────────────────────

def preprocess(img: np.ndarray, dpi_scale: float = 1.0) -> np.ndarray:
    """
    Prepare a raw page image for OCR.

    Steps tuned for aged / degraded documents:
      1. Greyscale
      2. Upscale if below 300 DPI (dpi_scale < 1 means we rendered low)
      3. Median denoise  — kills speckle without blurring stroke edges
      4. CLAHE           — local contrast boost for yellowed / faded pages
      5. Adaptive threshold → clean B&W for the OCR engine
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if dpi_scale < 0.9:
        scale = 1.0 / dpi_scale
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    img = cv2.medianBlur(img, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return img


# ── page rendering ─────────────────────────────────────────────────────────────

RENDER_DPI = 300

def render_pages(pdf_bytes: bytes):
    """Yield (page_index, numpy_image) for each page in the PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        yield page.number, img
    doc.close()


# ── tesseract binary resolution ────────────────────────────────────────────────

def _find_tesseract() -> str | None:
    """Return the tesseract binary path, or None if not found."""
    found = shutil.which("tesseract")
    if found:
        return found
    # Fallback search for common install locations not always on PATH
    for candidate in [
        "/opt/homebrew/bin/tesseract",   # macOS Homebrew (Apple Silicon)
        "/usr/local/bin/tesseract",      # macOS Homebrew (Intel) / generic
        "/usr/bin/tesseract",            # Linux apt
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


# ── OCR engines ───────────────────────────────────────────────────────────────

def ocr_tesseract(img: np.ndarray, tesseract_bin: str) -> str:
    import tempfile, subprocess
    from PIL import Image
    # macOS uses /private/tmp to avoid symlink issues with subprocesses
    tmp_dir = "/private/tmp" if os.path.isdir("/private/tmp") else None
    with tempfile.NamedTemporaryFile(suffix=".png", dir=tmp_dir, delete=False) as f:
        Image.fromarray(img).save(f.name)
        img_path = f.name
    result = subprocess.run(
        [tesseract_bin, img_path, "stdout", "--oem", "1", "--psm", "6"],
        capture_output=True,
    )
    os.unlink(img_path)
    if result.returncode != 0:
        raise RuntimeError(f"Tesseract failed (exit {result.returncode}): {result.stderr[:200]}")
    return result.stdout.decode("utf-8", errors="replace")


def ocr_easyocr(img: np.ndarray, reader) -> str:
    results = reader.readtext(img, detail=0, paragraph=True)
    return " ".join(results)


# ── public interface ──────────────────────────────────────────────────────────

def load_engine(engine: str):
    """Return an opaque engine handle — pass to ocr_pdf()."""
    if engine == "tesseract":
        tesseract_bin = _find_tesseract()
        if tesseract_bin is None:
            raise RuntimeError(
                "Tesseract binary not found. Install with:\n"
                "  macOS:  brew install tesseract\n"
                "  Linux:  sudo apt install tesseract-ocr"
            )
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin
            pytesseract.get_tesseract_version()
        except ImportError:
            raise RuntimeError("pytesseract not installed. Run: pip install pytesseract")
        except Exception as e:
            raise RuntimeError(f"Tesseract check failed: {e}")
        return ("tesseract", tesseract_bin)

    if engine == "easyocr":
        try:
            import easyocr
        except ImportError:
            raise RuntimeError("EasyOCR not installed. Run: pip install easyocr")
        print("  Loading EasyOCR model (first run downloads ~200MB)...", flush=True)
        reader = easyocr.Reader(["en"], gpu=False)
        return ("easyocr", reader)

    raise ValueError(f"Unknown OCR engine: {engine!r}. Choose 'tesseract' or 'easyocr'.")


def ocr_pdf(pdf_bytes: bytes, engine_handle) -> str:
    """
    Render each page, preprocess, OCR, and return the full document text.
    Returns empty string if every page produces no output.
    """
    engine_name, engine_obj = engine_handle
    pages = []
    for _, raw_img in render_pages(pdf_bytes):
        processed = preprocess(raw_img)
        if engine_name == "tesseract":
            text = ocr_tesseract(processed, engine_obj)
        else:
            text = ocr_easyocr(processed, engine_obj)
        pages.append(text.strip())
    return "\n\n".join(p for p in pages if p)
