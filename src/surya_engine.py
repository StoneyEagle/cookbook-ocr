"""
Surya OCR engine wrapper.

Surya is a transformer-based document OCR (https://github.com/VikParuchuri/surya).
It performs text detection + recognition + reading-order analysis in one pass,
and emits structured output including bold/italic/math tags — significantly
more accurate than Tesseract for multi-column book layouts with illustrations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_MATH_FRAC = re.compile(r"\\frac\{(\d+)\}\{(\d+)\}")
_FRAC_UNICODE = {
    (1, 2): "½", (1, 3): "⅓", (2, 3): "⅔",
    (1, 4): "¼", (3, 4): "¾",
    (1, 5): "⅕", (2, 5): "⅖", (3, 5): "⅗", (4, 5): "⅘",
    (1, 6): "⅙", (5, 6): "⅚",
    (1, 7): "⅐",
    (1, 8): "⅛", (3, 8): "⅜", (5, 8): "⅝", (7, 8): "⅞",
    (1, 9): "⅑", (1, 10): "⅒",
}
_MATH_TAG = re.compile(r"<math>(.*?)</math>", re.DOTALL)
_BOLD_TAG = re.compile(r"<b>(.*?)</b>", re.DOTALL)
_ITALIC_TAG = re.compile(r"<i>(.*?)</i>", re.DOTALL)
_OTHER_TAGS = re.compile(r"</?[a-zA-Z][^>]*>")


@dataclass
class SuryaLine:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float


class SuryaEngine:
    """Lazy-loaded Surya predictors shared across pages."""

    def __init__(self) -> None:
        self._foundation = None
        self._recognition = None
        self._detection = None

    def _load(self) -> None:
        if self._recognition is not None:
            return
        log.info("Loading Surya models (first run downloads ~1.5GB)...")
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        self._foundation = FoundationPredictor()
        self._recognition = RecognitionPredictor(self._foundation)
        self._detection = DetectionPredictor()
        log.info("Surya models ready.")

    def ocr_page(self, bgr: np.ndarray) -> tuple[list[SuryaLine], float]:
        """OCR a BGR numpy image. Returns (lines_in_reading_order, mean_conf)."""
        self._load()
        # Surya takes PIL RGB.
        if bgr.ndim == 3 and bgr.shape[2] == 3:
            rgb = bgr[:, :, ::-1]
        else:
            rgb = bgr
        pil = Image.fromarray(rgb)
        preds = self._recognition([pil], det_predictor=self._detection)
        result = preds[0]

        lines: list[SuryaLine] = []
        confs: list[float] = []
        for ln in result.text_lines:
            text = _clean_surya_text(ln.text)
            if not text.strip():
                continue
            bbox = tuple(ln.bbox)  # (x0, y0, x1, y1)
            lines.append(SuryaLine(text=text, bbox=bbox, confidence=float(ln.confidence)))
            confs.append(float(ln.confidence))
        mean_conf = float(np.mean(confs)) * 100.0 if confs else 0.0
        return lines, mean_conf


def _math_to_unicode(match: re.Match) -> str:
    """Convert LaTeX math like `\\frac{1}{2}` to Unicode fractions where possible."""
    body = match.group(1)

    def _frac_repl(m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        return _FRAC_UNICODE.get((num, den), f"{num}/{den}")

    body = _MATH_FRAC.sub(_frac_repl, body)
    # Strip any remaining LaTeX control sequences defensively.
    body = re.sub(r"\\[a-zA-Z]+\s*", "", body)
    body = body.replace("{", "").replace("}", "")
    return body


def _clean_surya_text(text: str) -> str:
    """Convert Surya's HTML-ish + LaTeX output to plain markdown-safe text."""
    text = _MATH_TAG.sub(_math_to_unicode, text)
    text = _BOLD_TAG.sub(r"**\1**", text)
    text = _ITALIC_TAG.sub(r"*\1*", text)
    text = _OTHER_TAGS.sub("", text)
    text = _normalize_inline_fractions(text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# Surya often emits mixed-number fractions using the Unicode fraction slash
# (U+2044): "1½" becomes "11⁄2", "2¾" becomes "23⁄4". Also occasionally uses
# ASCII slash: "1 1/2 cups", "23/4 cups".
_FRAC_SLASH = "\u2044"
_UNIT_KW = r"cup|cups|tsp|tbsp|teaspoon|teaspoons|tablespoon|tablespoons|pound|pounds|oz|lb|lbs|quart|pint|gallon|stick|sticks"
_MIXED_FRAC_SLASH = re.compile(rf"(\d)(\d)[{_FRAC_SLASH}/](\d)")
_BARE_FRAC_SLASH = re.compile(rf"(?<!\d)(\d)[{_FRAC_SLASH}](\d)(?!\d)")


def _mixed_frac_repl(m: re.Match) -> str:
    whole, num, den = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = _FRAC_UNICODE.get((num, den))
    return f"{whole}{frac}" if frac else m.group(0)


def _bare_frac_repl(m: re.Match) -> str:
    num, den = int(m.group(1)), int(m.group(2))
    return _FRAC_UNICODE.get((num, den), m.group(0))


def _normalize_inline_fractions(text: str) -> str:
    text = _MIXED_FRAC_SLASH.sub(_mixed_frac_repl, text)
    text = _BARE_FRAC_SLASH.sub(_bare_frac_repl, text)
    # ASCII "1 1/2 cup" / "2 3/4 cups" — require unit context for safety.
    for num, den, uni in [
        (1, 2, "½"), (1, 3, "⅓"), (2, 3, "⅔"),
        (1, 4, "¼"), (3, 4, "¾"),
        (1, 8, "⅛"), (3, 8, "⅜"), (5, 8, "⅝"), (7, 8, "⅞"),
    ]:
        pat = re.compile(rf"(\d)\s+{num}/{den}(?=\s+(?:{_UNIT_KW})\b)", re.I)
        text = pat.sub(rf"\1{uni}", text)
        pat2 = re.compile(rf"(?<!\d){num}/{den}(?=\s+(?:{_UNIT_KW})\b)", re.I)
        text = pat2.sub(uni, text)
    return text


def assemble_markdown(lines: list[SuryaLine], page_width: int) -> str:
    """
    Reassemble Surya's text lines into a reading-order markdown body.

    Workflow:
      1. Cluster all lines into columns by x-start.
      2. Single column → emit linearly by y.
      3. Multi-column → pull full-width "banner" lines out as their own
         one-line sections so they interleave by y-position, and emit each
         column as its own block.
    """
    if not lines:
        return ""

    median_h = float(np.median([ln.bbox[3] - ln.bbox[1] for ln in lines]))
    paragraph_gap = max(8.0, median_h * 1.4)

    columns = _cluster_columns(lines, page_width)
    if len(columns) == 1:
        return _emit_linear(columns[0], paragraph_gap)

    # Multi-column page: pull banners out so they interleave correctly.
    banners: list[SuryaLine] = []
    per_col: list[list[SuryaLine]] = [[] for _ in columns]
    for i, col in enumerate(columns):
        for ln in col:
            x0, _, x1, _ = ln.bbox
            if x1 - x0 > page_width * 0.58:
                banners.append(ln)
            else:
                per_col[i].append(ln)

    sections: list[list[SuryaLine]] = []
    for b in banners:
        sections.append([b])
    for col in per_col:
        if col:
            sections.append(col)

    for s in sections:
        s.sort(key=lambda ln: ln.bbox[1])
    sections.sort(key=lambda s: min(ln.bbox[1] for ln in s))

    out: list[str] = []
    for section in sections:
        if out and out[-1] != "":
            out.append("")
        out.append(_emit_linear(section, paragraph_gap))

    while out and out[-1].strip() == "":
        out.pop()
    return "\n".join(out).strip()


def _emit_linear(lines: list[SuryaLine], paragraph_gap: float) -> str:
    """Emit lines top-to-bottom with paragraph breaks between y-gaps."""
    if not lines:
        return ""
    sorted_lines = sorted(lines, key=lambda l: l.bbox[1])
    out: list[str] = []
    prev_y_bottom: Optional[float] = None
    for ln in sorted_lines:
        if prev_y_bottom is not None and ln.bbox[1] - prev_y_bottom > paragraph_gap:
            if out and out[-1] != "":
                out.append("")
        out.append(ln.text + "  ")
        prev_y_bottom = ln.bbox[3]
    return "\n".join(out).strip()


def _cluster_columns(lines: list[SuryaLine], page_width: int) -> list[list[SuryaLine]]:
    """
    Cluster text lines into columns by their left-edge x-coordinate.

    Returns a single-column result when splits would produce a minor cluster
    that's likely just an indented paragraph or decorative sidebar rather than
    a true second column.
    """
    if not lines:
        return []
    single = [sorted(lines, key=lambda l: l.bbox[1])]
    if len(lines) < 4:
        return single

    x0s = sorted(ln.bbox[0] for ln in lines)
    gap_threshold = page_width * 0.12  # only real column gutters cross this
    cluster_starts = [x0s[0]]
    for prev, curr in zip(x0s, x0s[1:]):
        if curr - prev > gap_threshold:
            cluster_starts.append(curr)
    if len(cluster_starts) == 1:
        return single

    centers = cluster_starts
    clusters: list[list[SuryaLine]] = [[] for _ in centers]
    for ln in lines:
        x0 = ln.bbox[0]
        idx = min(range(len(centers)), key=lambda i: abs(x0 - centers[i]))
        clusters[idx].append(ln)
    clusters = [c for c in clusters if c]

    # Reject the split if any minor cluster is too small — it's probably a
    # caption, pull-quote, or indented block inside a single-column layout.
    total = sum(len(c) for c in clusters)
    min_share = 0.2
    min_count = 4
    if any(len(c) < max(min_count, total * min_share) for c in clusters):
        return single

    clusters.sort(key=lambda c: min(ln.bbox[0] for ln in c))
    return clusters
