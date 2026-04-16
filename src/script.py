"""
Cookbook OCR pipeline — accuracy-first.

Scans PDFs in ../input, rasterizes with PyMuPDF, preprocesses with OpenCV,
runs Tesseract via pytesseract with OSD rotation + dual-PSM confidence
picking, writes one Markdown file per PDF to ../output.

Requires:
    pip install pymupdf pillow pytesseract opencv-python numpy
    Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
    For best accuracy on old print, also install the `eng_best` traineddata:
        https://github.com/tesseract-ocr/tessdata_best
        drop eng.traineddata into a folder and point --tessdata at it,
        then run with --lang eng.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

# Windows: auto-detect standard Tesseract install.
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(DEFAULT_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

RENDER_DPI = 400                      # high DPI for small serif in old cookbooks
MIN_EFFECTIVE_DPI = 320               # upscale below this
TESS_LANG = "eng"
TESS_OEM = 1                          # LSTM only
# Page segmentation modes to race. 3 = fully automatic; 4 = single column variable sizes.
TESS_PSM_CANDIDATES = (3, 4)

LIGATURE_MAP = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
    "\u2014": "--", "\u2013": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ",
}

log = logging.getLogger("cookbook-ocr")


# ---------- rasterization ----------

def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------- preprocessing ----------

def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate and expand canvas so nothing is cropped."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    m = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw / 2) - cx
    m[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def correct_orientation(bgr: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to auto-rotate 90/180/270 scans. No-op on failure."""
    try:
        osd = pytesseract.image_to_osd(bgr, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0)) % 360
        if rot:
            log.debug("OSD rotate %d", rot)
            return rotate_bound(bgr, rot)
    except pytesseract.TesseractError as e:
        log.debug("OSD failed: %s", e)
    return bgr


def deskew(gray: np.ndarray) -> np.ndarray:
    """Small-angle skew correction via minAreaRect of text pixels."""
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.2 or abs(angle) > 10:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def ensure_min_dpi(gray: np.ndarray, src_dpi: int, min_dpi: int) -> np.ndarray:
    if src_dpi >= min_dpi:
        return gray
    scale = min_dpi / src_dpi
    h, w = gray.shape
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess(bgr: np.ndarray, src_dpi: int) -> Image.Image:
    """
    Pipeline tuned for old, yellowed cookbook pages:
      grayscale → CLAHE (uneven lighting) → denoise → deskew → upscale → adaptive threshold.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    gray = deskew(gray)
    gray = ensure_min_dpi(gray, src_dpi, MIN_EFFECTIVE_DPI)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return Image.fromarray(binarized)


# ---------- OCR ----------

def ocr_with_confidence(img: Image.Image, psm: int, lang: str, extra_cfg: str) -> tuple[str, float]:
    """Run Tesseract at given PSM, return (text, mean word confidence)."""
    config = f"--oem {TESS_OEM} --psm {psm} {extra_cfg}".strip()
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    confidences = [int(c) for c in data["conf"] if c not in ("-1", "") and int(c) >= 0]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    # Rebuild text in reading order from the data dict (preserves blocks/paragraphs).
    lines: dict[tuple, list[tuple[int, str]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((data["left"][i], word))
    ordered = []
    prev_block = None
    for key in sorted(lines):
        block = key[0]
        if prev_block is not None and block != prev_block:
            ordered.append("")  # paragraph break between blocks
        ordered.append(" ".join(w for _, w in sorted(lines[key])))
        prev_block = block
    return "\n".join(ordered), mean_conf


def best_ocr(img: Image.Image, lang: str, extra_cfg: str) -> tuple[str, float, int]:
    best: tuple[str, float, int] = ("", -1.0, -1)
    for psm in TESS_PSM_CANDIDATES:
        try:
            text, conf = ocr_with_confidence(img, psm, lang, extra_cfg)
        except pytesseract.TesseractError as e:
            log.debug("PSM %d failed: %s", psm, e)
            continue
        # Prefer higher confidence; break ties by more text.
        score = conf + (len(text) / 1e6)
        if score > best[1] + (len(best[0]) / 1e6):
            best = (text, conf, psm)
    return best


# ---------- cleanup ----------

def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    for src, dst in LIGATURE_MAP.items():
        text = text.replace(src, dst)

    lines = [ln.rstrip() for ln in text.splitlines()]
    merged: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Join hyphenated line wraps: "exam-\nple" -> "example", but only when lowercase continues.
        if (
            ln.endswith("-")
            and i + 1 < len(lines)
            and lines[i + 1]
            and lines[i + 1][:1].islower()
        ):
            merged.append(ln[:-1] + lines[i + 1].lstrip())
            i += 2
        else:
            merged.append(ln)
            i += 1
    text = "\n".join(merged)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text.strip()


# ---------- driver ----------

def ocr_pdf(pdf_path: Path, out_path: Path, dpi: int, lang: str, extra_cfg: str) -> None:
    log.info("Processing %s", pdf_path.name)
    doc = fitz.open(pdf_path)
    total = doc.page_count

    header = [
        f"# {pdf_path.stem}",
        "",
        f"*OCR transcription of `{pdf_path.name}` ({total} pages). Engine: Tesseract + OpenCV.*",
        "",
        "---",
        "",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header))
        for i, page in enumerate(doc, start=1):
            bgr = render_page(page, dpi)
            bgr = correct_orientation(bgr)
            pil = preprocess(bgr, dpi)
            text, conf, psm = best_ocr(pil, lang, extra_cfg)
            cleaned = clean_text(text)
            log.info("  page %3d/%d  conf=%.1f  psm=%d  chars=%d", i, total, conf, psm, len(cleaned))
            f.write(f"\n## Page {i}\n\n")
            if cleaned:
                f.write(cleaned)
            else:
                f.write("_(no text detected)_")
            f.write("\n")
    doc.close()
    log.info("Wrote %s", out_path)


def check_tesseract() -> None:
    cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(cmd) is None and not Path(cmd).exists():
        raise SystemExit(
            "Tesseract not found. Install UB-Mannheim build:\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki\n"
            f"Or set the executable path: pytesseract.pytesseract.tesseract_cmd = ...\n"
            f"Currently: {cmd}"
        )
    log.info("Tesseract %s", pytesseract.get_tesseract_version())


def main() -> int:
    parser = argparse.ArgumentParser(description="Cookbook OCR pipeline.")
    parser.add_argument("--src", type=Path, default=INPUT_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    parser.add_argument("--lang", default=TESS_LANG, help="Tesseract language (e.g. 'eng', 'eng+nld').")
    parser.add_argument("--tessdata", type=Path, default=None, help="Override TESSDATA_PREFIX (e.g. tessdata_best).")
    parser.add_argument("--tesseract", type=Path, default=None, help="Path to tesseract.exe.")
    parser.add_argument("--force", action="store_true", help="Re-OCR even if markdown exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = str(args.tesseract)

    extra_cfg = ""
    if args.tessdata:
        if not args.tessdata.exists():
            raise SystemExit(f"tessdata path not found: {args.tessdata}")
        os.environ["TESSDATA_PREFIX"] = str(args.tessdata)
        extra_cfg = f'--tessdata-dir "{args.tessdata}"'

    check_tesseract()

    args.out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.src.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found in %s", args.src)
        return 1

    for pdf in pdfs:
        md = args.out / f"{pdf.stem}.md"
        if md.exists() and not args.force:
            log.info("Skipping %s (exists; --force to re-run)", md.name)
            continue
        ocr_pdf(pdf, md, args.dpi, args.lang, extra_cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
