# 📄 PDF OCR Workbench

**From scanned PDF to a searchable, editable knowledge base.**

The PDF OCR Workbench is a local web application designed to turn scanned or image-based PDF pages (especially difficult historical documents) into editable text. It features a human-in-the-loop review process, multiple OCR engines, and a hybrid search architecture. 

Because it runs entirely locally, **your documents never leave your machine** unless you explicitly choose the MinerU Cloud API option.

---

## ✨ Features

- **Multiple OCR Engines:** Choose between MinerU (Local/Cloud), Tesseract, RapidOCR, PaddleOCR, Chandra (VLM), or standard PDF text layer extraction.
- **Human-in-the-Loop:** Uploading a PDF *does not* automatically trigger OCR. You choose the specific pages (or regions) to process.
- **Column Detection:** Automatically detects two-column layouts and reads the left column before the right to maintain reading order.
- **Editable Review Stage:** Compare the original page image side-by-side with the extracted text. Correct errors before approving.
- **Quality Score:** A heuristic indicator (0.0 to 1.0) that flags garbled text or random characters so you know which pages need review.
- **Notebook & Export:** Approve pages into a clean "Notebook" and export the final document as Markdown, Word (.docx), HTML, JSON, or Plain Text.
- **Hybrid Search:** Search your approved documents using a combination of Semantic (45%), Keyword (35%), and Heading (20%) matching.
- **No Database Required:** All data is stored locally in plain files and `state.json` files.

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
It is highly recommended to use a virtual environment (Python 3.10 - 3.12 recommended).

```bash
# Clone the repository
git clone https://github.com/SAMxASH/O-C-R.git
cd O-C-R

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
