# cookbook-ocr

Accuracy-first OCR pipeline for digitizing old cookbooks. Converts scanned PDFs into cleanly formatted Markdown for preservation.

## Pipeline

Two OCR engines are supported. Default is **Surya** — a transformer-based
document OCR that performs text detection, recognition, and reading-order
analysis in a single pass.

### Surya path (default, `--engine surya`)

1. **Rasterize** PDF pages at 400 DPI with PyMuPDF
2. **Auto-rotate** via Tesseract OSD
3. Run Surya detection + recognition → returns per-line bboxes, text, confidence
4. **Column clustering** — split bboxes into banner lines (page-wide) and
   column lines (clustered by x-start), emit each section in top-down order
5. **Fraction normalization** — `11⁄2` → `½`, `23/4` → `¾`, LaTeX `\frac{1}{2}`
   → `½`
6. Preserve inline formatting — `<b>` → `**bold**`, `<i>` → `*italic*`

### Tesseract path (`--engine tesseract`)

1. Rasterize at 400 DPI
2. Auto-rotate via OSD
3. Preprocess — CLAHE, denoise, deskew, upscale, adaptive threshold
4. **Layout variants race**: full-page adaptive, full-page hard-threshold
   (kills illustration bleed-through), per-column (when 2+ columns detected)
5. OCR each variant with a PSM race (1/3/4/6), pick highest mean word confidence
6. `wordfreq` garbage filter drops ornamental borders
7. Line prefix/suffix strip removes margin-rule debris
8. Fraction and ligature normalization

## Layout

```
cookbook-ocr/
├── input/          drop scanned PDFs here (gitignored)
├── output/         OCR markdown lands here (gitignored)
├── src/script.py   pipeline
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Install Tesseract separately: <https://github.com/UB-Mannheim/tesseract/wiki>.

Download [`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best) models (LSTM float-precision; higher accuracy than the int-quantized models bundled with Tesseract):

```bash
mkdir tessdata_best
curl -L -o tessdata_best/eng.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
curl -L -o tessdata_best/osd.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/osd.traineddata
```

The script auto-detects a `tessdata_best/` directory at project root.

## Run

```bash
.venv/Scripts/python.exe src/script.py -v
```

First Surya run downloads ~1.5GB of model weights to a user cache. Subsequent runs reuse them.

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--engine` | `surya` | `surya` (transformer, high accuracy) or `tesseract` (lightweight) |
| `--src` | `./input` | PDF source folder |
| `--out` | `./output` | Markdown destination |
| `--dpi` | 400 | Render DPI |
| `--lang` | `eng` | Tesseract language (Tesseract engine only) |
| `--tessdata` | — | Override tessdata directory (Tesseract engine only) |
| `--tesseract` | auto | Path to `tesseract.exe` (Tesseract engine only) |
| `--force` | off | Re-OCR even if markdown exists |
| `-v` | off | Verbose logging (per-page confidence) |

## Notes

- Copyright: do not commit or publish PDFs you don't have redistribution rights for. `input/*.pdf` is gitignored by default.
- Tuning: if confidence is consistently low, try a larger `--dpi`, switch to `tessdata_best`, or add a custom wordlist.
