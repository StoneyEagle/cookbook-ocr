# cookbook-ocr

Accuracy-first OCR pipeline for digitizing old cookbooks. Converts scanned PDFs into cleanly formatted Markdown for preservation.

## Pipeline

1. **Rasterize** PDF pages at 400 DPI with PyMuPDF
2. **Auto-rotate** via Tesseract OSD (fixes sideways / upside-down scans)
3. **Preprocess** with OpenCV — CLAHE for uneven lighting, non-local means denoise, deskew, adaptive threshold
4. **Upscale** to ≥320 effective DPI for small serif type
5. **OCR** with Tesseract, racing PSM 3 vs PSM 4, pick by mean word confidence
6. **Reconstruct** reading order from `image_to_data` blocks/paragraphs
7. **Clean** Unicode — ligatures (`fi`, `fl`, `ffi`), smart quotes, hyphenated line-wrap joining
8. Write per-PDF Markdown with page headings

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

For best accuracy, grab `eng.traineddata` from [`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best) and pass `--tessdata /path/to/tessdata_best`.

## Run

```bash
.venv/Scripts/python.exe src/script.py -v
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--src` | `./input` | PDF source folder |
| `--out` | `./output` | Markdown destination |
| `--dpi` | 400 | Render DPI |
| `--lang` | `eng` | Tesseract language (e.g. `eng+nld`) |
| `--tessdata` | — | Override tessdata directory |
| `--tesseract` | auto | Path to `tesseract.exe` |
| `--force` | off | Re-OCR even if markdown exists |
| `-v` | off | Verbose logging (per-page confidence) |

## Notes

- Copyright: do not commit or publish PDFs you don't have redistribution rights for. `input/*.pdf` is gitignored by default.
- Tuning: if confidence is consistently low, try a larger `--dpi`, switch to `tessdata_best`, or add a custom wordlist.
