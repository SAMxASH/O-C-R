"""
butter-fly: PDF -> hybrid OCR (text layer / Tesseract / RapidOCR / PaddleOCR / VLM / MinerU cloud / MinerU local)
    -> editable notebook (headings, lists, bold, italic) -> hybrid search.
No database: files + one state.json per document under ./data
Run:  python app.py   then open http://127.0.0.1:5000
"""
import base64
import hashlib
import html as htmllib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

try:
    import pymupdf as fitz      # current name
except ImportError:             # older installs
    import fitz  # PyMuPDF
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# ----------------------------------------------------------------- config
# MinerU cloud API
MINERU_TOKEN = os.getenv("MINERU_TOKEN", "")
MINERU_BASE = os.getenv("MINERU_BASE", "https://mineru.net/api/v4")
MINERU_MODEL_VERSION = os.getenv("MINERU_MODEL_VERSION", "pipeline")   # "pipeline" or "vlm"
MINERU_LANGUAGE = os.getenv("MINERU_LANGUAGE", "en")
MINERU_FORMULA = os.getenv("MINERU_FORMULA", "0") == "1"
MINERU_BATCH_PAGES = int(os.getenv("MINERU_BATCH_PAGES", "8"))         # pages per cloud job (1 queue wait per job)
MINERU_API_WORKERS = int(os.getenv("MINERU_API_WORKERS", "3"))         # parallel cloud jobs
MINERU_POLL_SEC = float(os.getenv("MINERU_POLL_SEC", "4"))
OCR_TIMEOUT_SEC = int(os.getenv("OCR_TIMEOUT_SEC", "1800"))

# MinerU local (CLI installed with: pip install -U "mineru[all]")
MINERU_LOCAL_CMD = os.getenv("MINERU_LOCAL_CMD", "mineru")
MINERU_LOCAL_BACKEND = os.getenv("MINERU_LOCAL_BACKEND", "pipeline")   # pipeline (CPU ok) | vlm-* / hybrid-* (GPU)
MINERU_LOCAL_METHOD = os.getenv("MINERU_LOCAL_METHOD", "")             # auto | txt | ocr   (empty = MinerU default)
MINERU_LOCAL_LANG = os.getenv("MINERU_LOCAL_LANG", "")                 # e.g. en  (empty = MinerU default)
MINERU_LOCAL_TEMPLATE = os.getenv("MINERU_LOCAL_TEMPLATE", "")         # full command, see mineru_local_extract
MINERU_LOCAL_ARGS = os.getenv("MINERU_LOCAL_ARGS", "")                 # any extra CLI flags, e.g. "--device cuda"
MINERU_LOCAL_BATCH_PAGES = int(os.getenv("MINERU_LOCAL_BATCH_PAGES", "1"))   # mineru 4.x `parse` returns plain markdown (no per-page data) -> 1 page per call
MINERU_LOCAL_TIER = os.getenv("MINERU_LOCAL_TIER", "")                 # flash | basic | standard | advanced (empty = default)
MINERU_LOCAL_REMOTE = os.getenv("MINERU_LOCAL_REMOTE", "0") == "1"      # pass --remote (mineru.net) instead of local parsing
MINERU_LOCAL_TIMEOUT_SEC = int(os.getenv("MINERU_LOCAL_TIMEOUT_SEC", "7200"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")                         # full path on Windows if not in PATH
TESS_LANG = os.getenv("TESS_LANG", "eng")
OCR_ZOOM = float(os.getenv("OCR_ZOOM", "3.5"))                         # render zoom for local OCR (3.5 ~ 250 dpi)
LOCAL_WORKERS = int(os.getenv("LOCAL_WORKERS", "1"))                   # concurrent local OCR calls

VLM_URL = os.getenv("VLM_URL", "")            # e.g. http://localhost:8000/v1  (vLLM / LM Studio / Ollama OpenAI API)
VLM_MODEL = os.getenv("VLM_MODEL", "")        # e.g. datalab-to/chandra
VLM_KEY = os.getenv("VLM_KEY", "")
VLM_PROMPT = os.getenv("VLM_PROMPT", "OCR this page to Markdown. Read multi-column pages one column at a time, "
                                      "left column first. Output only the text.")

CHANDRA_CMD = os.getenv("CHANDRA_CMD", "chandra")                      # CLI from `pip install chandra-ocr`
CHANDRA_METHOD = os.getenv("CHANDRA_METHOD", "vllm")                  # "vllm" (server running) or "hf" (local GPU)

HEADER_FRAC = float(os.getenv("HEADER_FRAC", "0.055"))                 # top strip treated as full-width running head
AUTO_CHAIN = [e.strip() for e in os.getenv("AUTO_CHAIN", "layer,tesseract,rapid,paddle").split(",") if e.strip()]
LAYER_MIN_CHARS = int(os.getenv("LAYER_MIN_CHARS", "300"))
LAYER_MIN_Q = float(os.getenv("LAYER_MIN_Q", "0.85"))
AUTO_NB_Q = float(os.getenv("AUTO_NB_Q", "0.90"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
SEM_MIN = float(os.getenv("SEM_MIN", "0.55"))     # raw similarity a "related" result must reach
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))

DATA = Path(os.getenv("DATA_DIR", "data"))
DATA.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

_state_lock = threading.RLock()
_pdf_lock = threading.Lock()
_eng_lock = threading.Lock()
_job_lock = threading.Lock()
_mineru_local_lock = threading.Lock()          # only one local MinerU process at a time (RAM / GPU)
_local_sem = threading.Semaphore(LOCAL_WORKERS)
_jobs = {}


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------- markdown-ish patterns (headings / lists)
HEAD_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
LIST_ITEM_RE = re.compile(r"^\s{0,6}([-*•●▪◦]|\d{1,3}[.)])\s+(.+\S)\s*$")
LIST_START = re.compile(r"^(?:[-*•●▪◦]|\d{1,3}[.)])\s+\S")
COL_MARK = "────── column {} ──────"
COL_RE = re.compile(r"^─+ column \d+ ─+$")
COL_LINE_RE = re.compile(r"^─+ column (\d+) ─+$", re.M)


# ----------------------------------------------------------------- storage
def doc_dir(doc_id):
    if not re.fullmatch(r"[0-9a-f]{12}", doc_id):
        abort(404)
    d = DATA / doc_id
    if not d.exists():
        abort(404)
    return d


def load_state(doc_id):
    with _state_lock:
        return json.loads((doc_dir(doc_id) / "state.json").read_text(encoding="utf-8"))


def save_state(doc_id, st):
    with _state_lock:
        tmp = doc_dir(doc_id) / "state.json.tmp"
        tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        tmp.replace(doc_dir(doc_id) / "state.json")


def update_page(doc_id, page, **fields):
    with _state_lock:
        st = load_state(doc_id)
        p = st["pages"].setdefault(str(page), {"text": "", "in_notebook": False})
        p.update(fields)
        if not p["text"].strip():
            p["in_notebook"] = False
            p.pop("engine", None)
            p.pop("quality", None)
        save_state(doc_id, st)
        return p


# ----------------------------------------------------------------- text cleanup + LaTeX -> plain text
def _tbl(src, dst):
    return {ord(a): b for a, b in zip(src, dst)}


_SUP = _tbl("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUB = _tbl("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
          "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
          "sigma": "σ", "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
          "Theta": "Θ", "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}
_TEX_CMD = {"times": "×", "cdot": "·", "pm": "±", "mp": "∓", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
            "neq": "≠", "ne": "≠", "approx": "≈", "sim": "∼", "equiv": "≡", "infty": "∞", "rightarrow": "→",
            "to": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftrightarrow": "⇔", "degree": "°", "circ": "°",
            "ldots": "…", "dots": "…", "cdots": "…", "sum": "∑", "prod": "∏", "int": "∫", "partial": "∂",
            "nabla": "∇", "in": "∈", "div": "÷", "prime": "′", "angle": "∠", "perp": "⊥", "propto": "∝",
            "%": "%", "$": "$", "&": "&", "_": "_", "#": "#", **_GREEK}


def _grp(x):
    x = x.strip()
    return x if re.fullmatch(r"[\w.]+", x) else f"({x})"


def _script(m, table, mark):
    inner = m.group(1).replace(" ", "")
    if inner and all(ord(ch) in table for ch in inner):
        return inner.translate(table)
    return mark + (inner if len(inner) == 1 else f"({inner})")


def latex_to_text(s):
    """Converts the inside of a $...$ span to readable unicode. Returns None if it contains LaTeX we can't flatten."""
    has_script = "^" in s or "_" in s
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)(?![A-Za-z])\s*", "", s)
    s = re.sub(r"\\(?:[,;:! ]|q?quad(?![A-Za-z]))", " ", s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", s)
    for _ in range(3):
        s = re.sub(r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|operatorname|mbox)\s*\{([^{}]*)\}",
                   r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda m: f"{_grp(m.group(1))}/{_grp(m.group(2))}", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: "√" + _grp(m.group(1)), s)
    s = re.sub(r"\\([A-Za-z]+|[%$&_#])", lambda m: _TEX_CMD.get(m.group(1), m.group(0)), s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: _script(m, _SUP, "^"), s)
    s = re.sub(r"\^\s*([0-9A-Za-z+\-])", lambda m: _script(m, _SUP, "^"), s)
    s = re.sub(r"_\s*\{([^{}]*)\}", lambda m: _script(m, _SUB, "_"), s)
    s = re.sub(r"_\s*([0-9A-Za-z])", lambda m: _script(m, _SUB, "_"), s)
    if "\\" in s:
        return None
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    if re.fullmatch(r"[\d\s.,%+\-/]+", s):
        s = s.replace(" ", "")
    elif has_script:
        s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r"\s+(?=[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉°])", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s or None


# $$...$$   \(...\)   \[...\]   $...$  (the single-$ form must hug its content, so "$5 and $10" is left alone)
_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\((.+?)\\\)|\\\[(.+?)\\\]"
                      r"|(?<![\\$\d])\$(?=[^\s$])([^$\n]+?)(?<=[^\s$])\$(?![\d$])", re.S)
_LIGS = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "\u00ad": ""}


def fix_text(text):
    """Cleans OCR text. Returns (text, {what: how_many}) so the UI can say what it did."""
    counts = {}

    def bump(k, c):
        if c:
            counts[k] = counts.get(k, 0) + c

    t = text
    t, c = re.subn(r"!\[[^\]]*\]\([^)]*\)", "", t)
    bump("images", c)
    n = sum(t.count(k) for k in _LIGS)
    for k, v in _LIGS.items():
        t = t.replace(k, v)
    bump("ligatures", n)
    t = t.replace("\u00a0", " ")

    done = [0]

    def rep(m):
        inner = next(g for g in m.groups() if g is not None)
        res = latex_to_text(inner)
        if res is None:
            return m.group(0)
        done[0] += 1
        return res

    t = _MATH_RE.sub(rep, t)
    bump("math", done[0])

    t, c = re.subn(r"(?<=[A-Za-z])-\n(?=[a-z])", "", t)                        # con-\nstruction
    bump("hyphens", c)
    t, c = re.subn(r"(?<=[\w)\]”\"']) +([,.;:])(?=\s|$)", r"\1", t, flags=re.M)  # "word ," -> "word,"
    bump("punct", c)
    t, c = re.subn(r"(?<=\S)[ \t]{2,}(?=\S)", " ", t)
    bump("spaces", c)
    t, c = re.subn(r"^([ \t]*)[•●▪◦‣∙·]\s*(?=\S)", r"\1- ", t, flags=re.M)      # bullet glyphs -> "- "
    bump("bullets", c)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(), counts


def clean_ocr(text):
    return fix_text(text)[0]


def quality(text):
    """0..1 share of tokens that look like real words/numbers. Cheap garbage detector (no dictionary)."""
    text = re.sub(r"^─+ column \d+ ─+$", "", text, flags=re.M)
    toks = [t.strip(".,;:()[]\"'“”‘’—-") for t in text.split()]
    toks = [t for t in toks if t]
    if len(toks) < 15:
        return 0.0
    ok = sum(1 for t in toks if re.fullmatch(r"[A-Za-z]{1,20}|\d[\d,.]*|[A-Za-z]+\d+|[A-Za-z]{1,8}[.\-'][A-Za-z]{0,8}", t))
    return round(ok / len(toks), 2)


# ----------------------------------------------------------------- column-aware reading order
# Everything (PDF text layer, Tesseract, RapidOCR, Paddle) is reduced to word/line boxes
# (x0, y0, x1, y1, text) and then passed through the same layout logic:
#   running head (full width)  ->  left column  ->  right column
def _longest_run(mask, lo, hi):
    best_len = best_start = run = start = 0
    for i in range(lo, hi):
        if mask[i]:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    return best_start, best_len


def find_gutter(boxes, width):
    """boxes: [(x0, x1)] of body text. Returns (gx0, gx1) of an empty vertical band near the centre, or None."""
    if len(boxes) < 40:
        return None
    n = 400
    cov = np.zeros(n, dtype=np.int32)
    for x0, x1 in boxes:
        cov[max(int(x0 / width * n), 0): min(int(x1 / width * n) + 1, n)] += 1
    mask = cov <= 1                                   # tolerate one stray mark crossing the gutter
    s, ln = _longest_run(mask, int(0.35 * n), int(0.65 * n))
    if ln < 0.015 * n:
        return None
    gx0, gx1 = s * width / n, (s + ln) * width / n
    mid = (gx0 + gx1) / 2
    left = sum(1 for x0, x1 in boxes if (x0 + x1) / 2 < mid)
    if min(left, len(boxes) - left) < 0.1 * len(boxes):
        return None
    return gx0, gx1


def words_to_lines(ws):
    ws = sorted(ws, key=lambda w: (w[1] + w[3]) / 2)
    lines = []
    for w in ws:
        cy, h = (w[1] + w[3]) / 2, max(w[3] - w[1], 1)
        if lines and abs(cy - lines[-1]["cy"]) <= 0.6 * h:
            L = lines[-1]
            L["w"].append(w)
            L["cy"] = (L["cy"] * (len(L["w"]) - 1) + cy) / len(L["w"])
        else:
            lines.append({"cy": cy, "w": [w]})
    out = []
    for L in lines:
        L["w"].sort(key=lambda w: w[0])
        out.append({"x0": min(w[0] for w in L["w"]), "x1": max(w[2] for w in L["w"]),
                    "y0": min(w[1] for w in L["w"]), "y1": max(w[3] for w in L["w"]),
                    "cy": L["cy"], "text": " ".join(w[4] for w in L["w"])})
    return out


def _join(texts):
    out = ""
    for t in texts:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t                         # con-/struction -> construction
        else:
            out = f"{out} {t}" if out else t
    return out


def lines_to_paragraphs(lines):
    if not lines:
        return []
    hs = sorted(l["y1"] - l["y0"] for l in lines)
    h = hs[len(hs) // 2] or 1
    xs0 = sorted(l["x0"] for l in lines)
    xs1 = sorted(l["x1"] for l in lines)
    left = xs0[int(len(xs0) * 0.1)]
    right = xs1[min(int(len(xs1) * 0.9), len(xs1) - 1)]
    pitches = sorted(b["cy"] - a["cy"] for a, b in zip(lines, lines[1:]))
    pitch = pitches[len(pitches) // 2] if pitches else h * 1.3
    paras, cur = [], []
    for i, l in enumerate(lines):
        if cur:
            p = lines[i - 1]
            if ((l["x0"] - left > 0.8 * h) or (l["cy"] - p["cy"] > 1.6 * pitch) or (p["x1"] < right - 3 * h)
                    or LIST_START.match(l["text"])):          # every bullet / numbered item starts a new paragraph
                paras.append(_join(cur))
                cur = []
        cur.append(l["text"])
    if cur:
        paras.append(_join(cur))
    return [p for p in paras if p.strip()]


def items_to_text(items, W, H, columns=True):
    if not items:
        return ""
    hdr = HEADER_FRAC * H if columns else 0
    body = [i for i in items if (i[1] + i[3]) / 2 > hdr]
    gutter = find_gutter([(i[0], i[2]) for i in body], W) if columns else None
    if gutter is None:
        regions = [(None, items)]
    else:
        mid = (gutter[0] + gutter[1]) / 2
        regions = [(None, [i for i in items if (i[1] + i[3]) / 2 <= hdr]),
                   (1, [i for i in body if (i[0] + i[2]) / 2 < mid]),
                   (2, [i for i in body if (i[0] + i[2]) / 2 >= mid])]
    parts = []
    for no, reg in regions:
        paras = lines_to_paragraphs(words_to_lines(reg)) if reg else []
        if not paras:
            continue
        if no:
            parts.append(COL_MARK.format(no))          # visible separator: where each column starts
        parts += paras
    return "\n\n".join(parts)


# ----------------------------------------------------------------- pdf helpers
def render_page(doc_id, page, zoom, box=None):
    with _pdf_lock:
        pdf = fitz.open(doc_dir(doc_id) / "source.pdf")
        try:
            png = pdf[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).tobytes("png")
        finally:
            pdf.close()
    if box:
        im = Image.open(io.BytesIO(png))
        w, h = im.size
        x0, y0, x1, y1 = box
        out = io.BytesIO()
        im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(out, "PNG")
        return out.getvalue()
    return png


def multi_page_pdf(doc_id, pages):
    """One PDF holding the given 1-based pages in that order (page_idx in MinerU output == index in `pages`)."""
    with _pdf_lock:
        src = fitz.open(doc_dir(doc_id) / "source.pdf")
        try:
            src.select([p - 1 for p in pages])
            return src.tobytes(garbage=3, deflate=True)
        finally:
            src.close()


def image_bytes_to_pdf(img_bytes):
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode in ("RGBA", "P", "LA"):
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PDF")
    return out.getvalue()


def layer_items(doc_id, page):
    with _pdf_lock:
        pdf = fitz.open(doc_dir(doc_id) / "source.pdf")
        try:
            pg = pdf[page - 1]
            W, H = pg.rect.width, pg.rect.height
            words = [(w[0], w[1], w[2], w[3], w[4]) for w in pg.get_text("words")]
        finally:
            pdf.close()
    return words, W, H


# ----------------------------------------------------------------- engines
def _mineru_cmd():
    """Full path of the local `mineru` CLI (PATH first, then next to the running python / venv Scripts)."""
    p = shutil.which(MINERU_LOCAL_CMD)
    if p:
        return p
    scripts = Path(sys.executable).parent
    for name in (MINERU_LOCAL_CMD, MINERU_LOCAL_CMD + ".exe"):
        if (scripts / name).is_file():
            return str(scripts / name)
    return None


def engine_list():
    def has(m):
        return importlib.util.find_spec(m) is not None
    return [
        {"id": "auto", "label": "Auto (text layer -> local OCR)", "available": True},
        {"id": "layer", "label": "PDF text layer (column-aware)", "available": True},
        {"id": "tesseract", "label": "Tesseract (local)",
         "available": has("pytesseract") and bool(shutil.which(TESSERACT_CMD or "tesseract"))},
        {"id": "rapid", "label": "RapidOCR (local, ONNX, fast)", "available": has("rapidocr_onnxruntime") or has("rapidocr")},
        {"id": "paddle", "label": "PaddleOCR (local)", "available": has("paddleocr")},
        {"id": "mineru_local", "label": "MinerU (local, offline)", "available": bool(_mineru_cmd())},
        {"id": "mineru", "label": "MinerU (cloud API)", "available": bool(MINERU_TOKEN)},
        {"id": "chandra", "label": "Chandra OCR (needs GPU / vLLM)", "available": bool(shutil.which(CHANDRA_CMD))},
        {"id": "vlm", "label": "VLM server (Chandra / olmOCR / ...)", "available": bool(VLM_URL and VLM_MODEL)},
    ]


def _items_tesseract(im):
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    d = pytesseract.image_to_data(im, lang=TESS_LANG, config="--oem 1 --psm 3 --dpi 250",
                                  output_type=pytesseract.Output.DICT)
    items = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) >= 0:
            items.append((d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i], t))
    return items


def _poly_box(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


_rapid = None


def _items_rapid(im):
    global _rapid
    arr = np.array(im.convert("RGB"))
    with _eng_lock:
        if _rapid is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _rapid = (RapidOCR(), 1)
            except ImportError:
                from rapidocr import RapidOCR
                _rapid = (RapidOCR(), 2)
    eng, ver = _rapid
    items = []
    if ver == 1:
        res, _ = eng(arr)
        for box, txt, _s in (res or []):
            items.append((*_poly_box(box), txt))
    else:
        out = eng(arr)
        boxes = out.boxes if out.boxes is not None else []
        for box, txt in zip(boxes, out.txts or []):
            items.append((*_poly_box(box), txt))
    return items


_paddle = None


def _items_paddle(im):
    global _paddle
    arr = np.array(im.convert("RGB"))
    with _eng_lock:
        if _paddle is None:
            from paddleocr import PaddleOCR
            try:    # PaddleOCR 3.x
                _paddle = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                                    use_doc_unwarping=False, use_textline_orientation=False)
            except Exception:  # noqa: BLE001  (2.x or different signature)
                try:
                    _paddle = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
                except Exception:  # noqa: BLE001
                    _paddle = PaddleOCR(lang="en")
    items = []
    if hasattr(_paddle, "predict"):
        for r in _paddle.predict(arr):
            polys = r.get("rec_polys") if r.get("rec_polys") is not None else r.get("dt_polys")
            for poly, txt in zip(polys, r.get("rec_texts", [])):
                items.append((*_poly_box(poly), txt))
    else:
        res = _paddle.ocr(arr, cls=False)
        for line in (res[0] or []):
            box, (txt, _s) = line
            items.append((*_poly_box(box), txt))
    return items


def _chandra(png_bytes):
    """Runs the `chandra` CLI on one page image and returns its markdown."""
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "page.png", Path(td) / "out"
        src.write_bytes(png_bytes)
        r = subprocess.run([CHANDRA_CMD, str(src), str(out), "--method", CHANDRA_METHOD, "--no-images"],
                           capture_output=True, text=True, timeout=OCR_TIMEOUT_SEC)
        if r.returncode != 0:
            raise RuntimeError("chandra failed: " + (r.stderr or r.stdout)[-400:])
        mds = sorted(out.rglob("*.md"))
        if not mds:
            raise RuntimeError("chandra produced no markdown output")
        return mds[0].read_text(encoding="utf-8", errors="replace")


def _vlm(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    headers = {"Content-Type": "application/json"}
    if VLM_KEY:
        headers["Authorization"] = f"Bearer {VLM_KEY}"
    body = {"model": VLM_MODEL, "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
    r = requests.post(VLM_URL.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ----------------------------------------------------------------- MinerU shared parsing
def _session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def _cl_text(it):
    t = it.get("type")
    if t == "image":
        return ""
    if t == "table":
        return (it.get("table_body") or "").strip()
    txt = (it.get("text") or "").strip()
    if not txt:
        return ""
    lvl = it.get("text_level")
    return ("#" * min(int(lvl), 6) + " " + txt) if lvl else txt


def _content_list_to_pages(raw):
    pages = {}
    for it in json.loads(raw):
        if not isinstance(it, dict):
            continue
        t = _cl_text(it)
        if t:
            pages.setdefault(int(it.get("page_idx", 0)), []).append(t)
    return {k: "\n\n".join(v) for k, v in pages.items()}


# ----------------------------------------------------------------- MinerU (cloud)
def mineru_extract(pdf_bytes, filename, formula, report, cancelled):
    """Returns {page_idx: markdown}. page_idx is relative to the uploaded PDF."""
    if not MINERU_TOKEN:
        raise RuntimeError("MINERU_TOKEN is not set (put it in .env)")
    S = _session()
    headers = {"Authorization": f"Bearer {MINERU_TOKEN}", "Content-Type": "application/json"}
    body = {"files": [{"name": filename, "is_ocr": True, "data_id": uuid.uuid4().hex[:16]}],
            "model_version": MINERU_MODEL_VERSION, "language": MINERU_LANGUAGE,
            "enable_formula": bool(formula), "enable_table": True}
    report("MinerU: requesting upload slot")
    r = S.post(f"{MINERU_BASE}/file-urls/batch", headers=headers, json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"MinerU error: {j.get('msg')}")
    batch_id = j["data"]["batch_id"]

    last = None
    for attempt in range(1, 5):                       # the storage host is often flaky from some networks
        if cancelled():
            raise Cancelled()
        try:
            report(f"MinerU: uploading {len(pdf_bytes) / 1e6:.1f} MB (try {attempt}/4)")
            S.put(j["data"]["file_urls"][0], data=pdf_bytes, timeout=(15, 300)).raise_for_status()
            last = None
            break
        except requests.RequestException as e:
            last = e
            time.sleep(3 * attempt)
    if last is not None:
        raise RuntimeError(f"MinerU upload failed after 4 tries ({last.__class__.__name__}). "
                           "Network/DNS problem reaching MinerU storage - try a VPN, or use 'MinerU (local)'.")

    t0 = time.time()
    state = "uploaded"
    while True:
        if cancelled():
            raise Cancelled()
        waited = time.time() - t0
        if waited > OCR_TIMEOUT_SEC:
            raise RuntimeError(f"MinerU timed out after {OCR_TIMEOUT_SEC}s (last state: {state})")
        time.sleep(MINERU_POLL_SEC)
        s = S.get(f"{MINERU_BASE}/extract-results/batch/{batch_id}", headers=headers, timeout=60)
        s.raise_for_status()
        sj = s.json()
        if sj.get("code") != 0:
            raise RuntimeError(f"MinerU error: {sj.get('msg')}")
        results = sj["data"].get("extract_result") or []
        if not results:
            report(f"MinerU: waiting for server to register file ({int(waited)}s)")
            continue
        item = results[0]
        state = item.get("state", "?")
        prog = item.get("extract_progress") or {}
        msg = f"MinerU: {state}"
        if prog.get("total_pages"):
            msg += f" {prog.get('extracted_pages', 0)}/{prog['total_pages']} pages"
        msg += f" ({int(waited)}s)"
        if state in ("pending", "waiting-file") and waited > 120:
            msg += " - queued on MinerU's shared server"
        report(msg)
        if state == "failed":
            raise RuntimeError(f"MinerU failed: {item.get('err_msg')}")
        if state == "done":
            report("MinerU: downloading result")
            z = S.get(item["full_zip_url"], timeout=(10, 300))
            z.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
                names = zf.namelist()
                cl = next((n for n in names if n.endswith("content_list.json")), None)
                if cl:
                    return _content_list_to_pages(zf.read(cl).decode("utf-8", errors="replace"))
                md = next((n for n in names if n.endswith("full.md")), None) or next((n for n in names if n.endswith(".md")), None)
                if not md:
                    raise RuntimeError("No markdown found in MinerU result")
                return {0: zf.read(md).decode("utf-8", errors="replace")}


# ----------------------------------------------------------------- MinerU (local CLI)
def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001
        pass


def mineru_local_extract(pdf_bytes, label, report, cancelled):
    """Runs the local `mineru` CLI on a PDF. Returns {page_idx: markdown} like the cloud version."""
    exe = _mineru_cmd()
    if not exe:
        raise RuntimeError("local MinerU not found: pip install -U \"mineru[all]\" (or set MINERU_LOCAL_CMD)")
    while not _mineru_local_lock.acquire(timeout=1):      # one run at a time, but stay cancellable while waiting
        if cancelled():
            raise Cancelled()
        report(f"MinerU local ({label}): waiting for the previous run to finish")
    try:
        with tempfile.TemporaryDirectory() as td:
            src, out, log = Path(td) / "in.pdf", Path(td) / "out", Path(td) / "run.log"
            src.write_bytes(pdf_bytes)
            res_file = out / "result.md"
            if MINERU_LOCAL_TEMPLATE.strip():
                # placeholders: {exe} {input} {output} {backend}
                sub = {"{exe}": exe, "{input}": str(src), "{output}": str(res_file), "{backend}": MINERU_LOCAL_BACKEND}
                cmd = [sub.get(t, t) for t in shlex.split(MINERU_LOCAL_TEMPLATE, posix=(os.name != "nt"))]
            else:                                         # mineru 4.x:  mineru parse <file> --pages all -o <file.md>
                cmd = [exe, "parse", str(src), "--pages", "all", "--no-marker", "--force",
                       "--wait", str(MINERU_LOCAL_TIMEOUT_SEC), "-o", str(res_file)]
                if MINERU_LOCAL_TIER:
                    cmd += ["--tier", MINERU_LOCAL_TIER]
                if MINERU_LOCAL_REMOTE:
                    cmd += ["--remote"]
            if MINERU_LOCAL_ARGS.strip():
                cmd += shlex.split(MINERU_LOCAL_ARGS, posix=(os.name != "nt"))
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            t0 = time.time()
            with open(log, "wb") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                while proc.poll() is None:
                    if cancelled():
                        _kill_tree(proc)
                        raise Cancelled()
                    el = time.time() - t0
                    if el > MINERU_LOCAL_TIMEOUT_SEC:
                        _kill_tree(proc)
                        raise RuntimeError(f"local MinerU timed out after {MINERU_LOCAL_TIMEOUT_SEC}s")
                    report(f"MinerU local ({label}): running {int(el)}s (the first run loads / downloads models)")
                    time.sleep(1)
            if proc.returncode != 0:
                tail = log.read_text(encoding="utf-8", errors="replace")[-700:]
                raise RuntimeError(f"mineru exited with code {proc.returncode}: {tail}")
            cls = sorted(out.rglob("*content_list.json"), key=lambda p: ("_v2" in p.name, len(p.name)))
            if cls:
                return _content_list_to_pages(cls[0].read_text(encoding="utf-8", errors="replace"))
            mds = [res_file] if res_file.is_file() else sorted(out.rglob("*.md"), key=lambda p: len(p.name))
            if mds:
                return {0: mds[0].read_text(encoding="utf-8", errors="replace")}
            printed = log.read_text(encoding="utf-8", errors="replace").strip()   # `parse` prints the markdown too
            if len(printed) > 20:
                return {0: printed}
            raise RuntimeError("mineru finished but produced no output (run it by hand on a PDF and check the message)")
    finally:
        _mineru_local_lock.release()


# ----------------------------------------------------------------- engine dispatch
def run_engine(engine, doc_id, page, box, formula, report, cancelled):
    """Returns raw text for one page (or one snipped box)."""
    if engine == "layer":
        words, W, H = layer_items(doc_id, page)
        if box:
            x0, y0, x1, y1 = box
            words = [w for w in words if x0 * W <= (w[0] + w[2]) / 2 <= x1 * W and y0 * H <= (w[1] + w[3]) / 2 <= y1 * H]
        return items_to_text(words, W, H, columns=not box)

    if engine in ("mineru", "mineru_local"):
        pdf = image_bytes_to_pdf(render_page(doc_id, page, 3.0, box)) if box else multi_page_pdf(doc_id, [page])
        if engine == "mineru":
            res = mineru_extract(pdf, f"p{page}.pdf", formula, report, cancelled)
        else:
            res = mineru_local_extract(pdf, f"page {page}", report, cancelled)
        return "\n\n".join(res[k] for k in sorted(res))

    png = render_page(doc_id, page, OCR_ZOOM if engine not in ("vlm", "chandra") else 2.5, box)
    if engine == "chandra":
        report("Chandra: running (the first call loads the model)")
        with _local_sem:
            return _chandra(png)
    if engine == "vlm":
        report("VLM: reading page")
        with _local_sem:
            return _vlm(png)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    fn = {"tesseract": _items_tesseract, "rapid": _items_rapid, "paddle": _items_paddle}[engine]
    report(f"{engine}: recognising")
    with _local_sem:
        items = fn(im)
    return items_to_text(items, im.width, im.height, columns=not box)


def run_auto(doc_id, page, box, formula, report, cancelled):
    avail = {e["id"] for e in engine_list() if e["available"]}
    best, errs = ("", -1.0, ""), []
    for eng in AUTO_CHAIN:
        if eng not in avail:
            continue
        if cancelled():
            raise Cancelled()
        try:
            txt = clean_ocr(run_engine(eng, doc_id, page, box, formula, report, cancelled))
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            errs.append(f"{eng}: {e}")
            continue
        q = quality(txt)
        good = (len(txt) >= LAYER_MIN_CHARS and q >= LAYER_MIN_Q) if eng == "layer" else q >= 0.6
        if txt and q > best[1]:
            best = (txt, q, eng)
        if good:
            return txt, eng
    if best[0]:
        return best[0], best[2]
    raise RuntimeError("; ".join(errs) or "no engine produced text (install tesseract/rapidocr or set another engine)")


# ----------------------------------------------------------------- jobs
def _inc_done(job):
    with _job_lock:
        job["done"] += 1


def run_job(jid, doc_id, pages, engine, box, formula, auto_nb):
    job = _jobs[jid]
    report = lambda m: job.__setitem__("msg", m)          # noqa: E731
    cancelled = lambda: job["cancel"]                       # noqa: E731

    def store(page, text, eng):
        text = clean_ocr(text)
        with _state_lock:
            cur = load_state(doc_id)["pages"].get(str(page), {}).get("text", "")
            new = (cur.rstrip() + "\n\n" + text).strip() if (box and cur.strip()) else text
            q = quality(new)
            fields = {"text": new, "engine": eng, "quality": q}
            if not box:                                   # a snip must not kick a page out of the notebook
                fields["in_notebook"] = bool(auto_nb and q >= AUTO_NB_Q)
            update_page(doc_id, page, **fields)
        job["results"][str(page)] = {"engine": eng, "quality": q}

    def one(page):
        if engine == "auto":
            txt, eng = run_auto(doc_id, page, box, formula, report, cancelled)
        else:
            txt, eng = run_engine(engine, doc_id, page, box, formula, report, cancelled), engine
        if not txt.strip():
            raise RuntimeError("no text found")
        store(page, txt, eng)
        _inc_done(job)

    try:
        if engine in ("mineru", "mineru_local") and not box and len(pages) > 1:
            local = engine == "mineru_local"
            size = MINERU_LOCAL_BATCH_PAGES if local else MINERU_BATCH_PAGES
            workers = 1 if local else MINERU_API_WORKERS
            chunks = [pages[i:i + size] for i in range(0, len(pages), size)]

            def do_chunk(chunk):
                try:
                    if job["cancel"]:
                        return
                    pdf = multi_page_pdf(doc_id, chunk)
                    if local:
                        res = mineru_local_extract(pdf, f"pages {chunk[0]}-{chunk[-1]}", report, cancelled)
                    else:
                        res = mineru_extract(pdf, f"p{chunk[0]}-{chunk[-1]}.pdf", formula, report, cancelled)
                    if len(res) == 1 and 0 in res and len(chunk) > 1:
                        raise RuntimeError("MinerU returned no per-page data; set the batch size to 1 "
                                           "(MINERU_LOCAL_BATCH_PAGES / MINERU_BATCH_PAGES=1)")
                    for idx, p in enumerate(chunk):
                        t = res.get(idx, "")
                        if t.strip():
                            store(p, t, engine)
                            _inc_done(job)
                        else:
                            job["errors"][str(p)] = "no text"
                except Cancelled:
                    pass
                except Exception as e:  # noqa: BLE001
                    for p in chunk:
                        job["errors"][str(p)] = str(e)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(do_chunk, chunks))
        else:
            for i, p in enumerate(pages, 1):
                if job["cancel"]:
                    break
                job["msg"] = f"page {p} ({i}/{len(pages)})"
                try:
                    one(p)
                except Cancelled:
                    break
                except Exception as e:  # noqa: BLE001
                    job["errors"][str(p)] = str(e)
        if job["cancel"]:
            job["status"] = "cancelled"
        elif len(pages) == 1 and job["errors"]:
            job.update(status="error", error=next(iter(job["errors"].values())))
        else:
            job["status"] = "done"
            job["msg"] = f"finished: {job['done']} ok, {len(job['errors'])} failed"
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))
    finally:
        job["t"] = time.time()


def _prune_jobs():
    now = time.time()
    for k in [k for k, v in _jobs.items() if v["status"] != "running" and now - v.get("t", now) > 3600]:
        _jobs.pop(k, None)


# ----------------------------------------------------------------- chunking (heading / list aware)
def paragraphs(pages):
    """Splits page texts into paragraphs, remembering the current heading (+level) and whether a paragraph is a list item."""
    out, heading, level, fresh, prev = [], "", 0, False, None
    for page, text in pages:
        if prev is not None and page != prev + 1:
            heading, level, fresh = "", 0, False
        prev = page
        cur, is_list = [], False
        for line in text.splitlines() + [""]:
            s = line.strip()
            if COL_RE.match(s):                            # column separator: paragraph break, never indexed
                line = s = ""
            hm = HEAD_RE.match(line) if s else None
            lm = bool(LIST_ITEM_RE.match(line)) if s and not hm else False
            if hm or lm or not s:
                if cur:
                    out.append({"page": page, "heading": heading, "level": level, "start": fresh,
                                "text": " ".join(cur), "list": is_list})
                    fresh = False
                cur, is_list = [], False
                if hm:
                    heading, level, fresh = hm.group(2).strip(), len(hm.group(1)), True
                elif lm:
                    cur, is_list = [s], True
            else:
                cur.append(s)
    return out


def chunk_pages(pages):
    chunks, cur = [], None
    for p in paragraphs(pages):
        w = p["text"].split()
        if len(w) > CHUNK_WORDS:
            if cur:
                chunks.append(cur)
                cur = None
            step = max(CHUNK_WORDS - CHUNK_OVERLAP, 1)
            for i in range(0, len(w), step):
                chunks.append({"page": p["page"], "heading": p["heading"], "level": p["level"],
                               "start": p["start"] and i == 0, "text": " ".join(w[i:i + CHUNK_WORDS])})
                if i + CHUNK_WORDS >= len(w):
                    break
            continue
        if cur and (cur["page"] != p["page"] or cur["heading"] != p["heading"]
                    or p["start"] or cur["n"] + len(w) > CHUNK_WORDS):
            chunks.append(cur)
            cur = None
        if cur is None:
            cur = {"page": p["page"], "heading": p["heading"], "level": p["level"], "start": p["start"],
                   "text": p["text"], "n": len(w), "ll": p["list"]}
        else:
            cur["text"] += ("\n" if (cur["ll"] and p["list"]) else "\n\n") + p["text"]
            cur["n"] += len(w)
            cur["ll"] = p["list"]
    if cur:
        chunks.append(cur)
    for c in chunks:
        c.pop("n", None)
        c.pop("ll", None)
    return [c for c in chunks if c["text"].strip()]


# ----------------------------------------------------------------- hybrid search
_model = None
_model_lock = threading.Lock()
_emb_cache = {}

STOP = set("a an the of and or to in on for with is are was were be by as at from that this it its into "
           "about what which who how why when where do does did can".split())


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL)
        return _model


def embed(texts, prefix=""):
    keys = [hashlib.sha1((prefix + t).encode()).hexdigest() for t in texts]
    missing = [(k, prefix + t) for k, t in zip(keys, texts) if k not in _emb_cache]
    if missing:
        vecs = get_model().encode([t for _, t in missing], normalize_embeddings=True,
                                  batch_size=32, show_progress_bar=False)
        for (k, _), v in zip(missing, vecs):
            _emb_cache[k] = v
    return np.vstack([_emb_cache[k] for k in keys])


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def toks(s):
    return [_stem(w) for w in re.findall(r"\w+", s.lower()) if w not in STOP and len(w) > 1]


def plain(s):
    """Text without list markers / bold / italic marks, for scoring and embedding."""
    s = re.sub(r"^\s*(?:[-*•●▪◦]|\d{1,3}[.)])\s+", "", s, flags=re.M)
    return re.sub(r"[*`]+", "", s)


def semantic_search(doc_id, query, top_k=10):
    """final = 0.45 * semantic + 0.35 * keyword + 0.20 * heading   (keyword + heading only if the model can't load)"""
    st = load_state(doc_id)
    pages = sorted((int(n), p["text"]) for n, p in st["pages"].items()
                   if p.get("in_notebook") and p.get("text", "").strip())
    chunks = chunk_pages(pages)
    if not chunks:
        return []
    qnorm = " ".join(query.lower().split())
    qset = set(toks(query))
    N = len(chunks)

    bodies = [plain((c["heading"] + "\n" + c["text"]) if c["start"] else c["text"]).strip() for c in chunks]
    use_sem = True
    try:
        sem = embed(bodies, PASSAGE_PREFIX) @ embed([query], QUERY_PREFIX)[0]
        sem_n = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
    except Exception:  # noqa: BLE001   (no sentence-transformers / model download failed) -> keyword-only
        use_sem, sem_n = False, np.zeros(N)

    head = np.zeros(N)
    idx = [i for i, c in enumerate(chunks) if c["start"] and plain(c["heading"]).strip()]
    for i in idx:
        h = plain(chunks[i]["heading"]).lower().strip()
        head[i] = 1.0 if h == qnorm else 0.85 if qnorm in h else 0.0
    if idx and use_sem:
        try:
            hs = embed([plain(chunks[i]["heading"]) for i in idx], PASSAGE_PREFIX) @ embed([query], PASSAGE_PREFIX)[0]
            for i, s in zip(idx, hs):
                head[i] = max(head[i], float(np.clip((s - 0.55) / 0.35, 0, 1)))
        except Exception:  # noqa: BLE001
            pass

    lex, cov_list = np.zeros(N), np.zeros(N)
    for i, c in enumerate(chunks):
        hay = plain((c["heading"] + " " + c["text"]) if c["start"] else c["text"])
        cov = len(qset & set(toks(hay))) / len(qset) if qset else 0.0
        phrase = 1.0 if qnorm and qnorm in " ".join(hay.lower().split()) else 0.0
        lex[i], cov_list[i] = 0.65 * cov + 0.35 * phrase, cov

    final = (0.45 * sem_n + 0.35 * lex + 0.20 * head) if use_sem else (0.75 * lex + 0.25 * head)
    order = np.argsort(-final)[:top_k]
    raw = sem if use_sem else None
    return [{**chunks[i], "score": round(float(final[i]), 3), "keywords": round(float(cov_list[i]), 2),
             "sem": use_sem, "raw": round(float(raw[i]), 3) if raw is not None else None}
            for i in order]


def exact_hits(doc_id, query, limit=40):
    """Literal word / phrase matches over every block (headings included), independent of chunking."""
    st = load_state(doc_id)
    q = " ".join(query.lower().split())
    terms = [t for t in re.findall(r"\w+", q) if len(t) > 1]
    if not terms:
        return []
    out = []
    for n, text in _nb_items(st):
        heading, level = "", 0
        for b in parse_blocks(text):
            if b[0] == "col":
                continue
            if b[0] == "h":
                heading, level = b[2], b[1]
                body, is_head = b[2], True
            elif b[0] in ("ul", "ol"):
                body, is_head = "\n".join("- " + x for x in b[1]), False
            else:
                body, is_head = b[1], False
            low = " ".join(plain(body).lower().split())
            phrase = q in low
            if not phrase and not all(re.search(r"\b" + re.escape(t), low) for t in terms):
                continue
            out.append({"page": n, "heading": heading, "level": level,
                        "start": is_head, "text": "" if is_head else body,
                        "score": 1.0 if phrase else 0.9, "keywords": 1.0, "exact": True})
            if len(out) >= limit:
                return out
    return out


# ----------------------------------------------------------------- block parsing (export)
def parse_blocks(text):
    """Page text -> [("h", level, text) | ("ul"|"ol", [items]) | ("p", text) | ("col", n)]"""
    blocks, para = [], []
    last_list = False

    def flush():
        if para:
            blocks.append(("p", "\n".join(para)))
            para.clear()

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            flush()
            last_list = False
            continue
        m = COL_LINE_RE.match(s)
        if m:
            flush()
            blocks.append(("col", m.group(1)))
            last_list = False
            continue
        h = HEAD_RE.match(raw)
        if h:
            flush()
            blocks.append(("h", len(h.group(1)), h.group(2).strip()))
            last_list = False
            continue
        li = LIST_ITEM_RE.match(raw)
        if li:
            flush()
            kind = "ol" if li.group(1)[0].isdigit() else "ul"
            if blocks and blocks[-1][0] == kind:
                blocks[-1][1].append(li.group(2))
            else:
                blocks.append((kind, [li.group(2)]))
            last_list = True
            continue
        if last_list and raw[:1] in (" ", "\t"):          # wrapped continuation of a list item
            blocks[-1][1][-1] += " " + s
            continue
        last_list = False
        para.append(s)
    flush()
    return blocks


def inline_html(s):
    s = htmllib.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
    return s.replace("\n", "<br>")


def add_runs(par, text):
    for part in re.split(r"(\*\*.+?\*\*|(?<!\*)\*(?!\s)[^*\n]+?(?<!\s)\*(?!\*))", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def blocks_json(blocks):
    out = []
    for b in blocks:
        if b[0] == "h":
            out.append({"type": "heading", "level": b[1], "text": b[2]})
        elif b[0] in ("ul", "ol"):
            out.append({"type": "list", "ordered": b[0] == "ol", "items": b[1]})
        elif b[0] == "col":
            out.append({"type": "column", "n": int(b[1])})
        else:
            out.append({"type": "paragraph", "text": b[1]})
    return out


# ----------------------------------------------------------------- routes
@app.get("/")
def index():
    f = Path(app.root_path) / "templates" / "index.html"     # served raw: JS template literals contain "{{"
    return Response(f.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/engines")
def engines_route():
    return jsonify(engine_list())


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Upload a .pdf file"), 400
    doc_id = uuid.uuid4().hex[:12]
    d = DATA / doc_id
    d.mkdir()
    f.save(d / "source.pdf")
    try:
        with _pdf_lock:
            pdf = fitz.open(d / "source.pdf")
            n = pdf.page_count
            pdf.close()
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error=f"Cannot read PDF: {e}"), 400
    save_state(doc_id, {"id": doc_id, "name": f.filename, "page_count": n, "pages": {}})
    return jsonify(id=doc_id, name=f.filename, page_count=n)


@app.get("/api/docs")
def docs():
    out = []
    for d in sorted((p for p in DATA.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        s = d / "state.json"
        if s.exists():
            try:
                j = json.loads(s.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            pg = j.get("pages", {})
            done = sum(1 for p in pg.values() if p.get("in_notebook") and p.get("text", "").strip())
            out.append({"id": j["id"], "name": j.get("title") or j["name"], "page_count": j["page_count"], "notebook": done})
    return jsonify(out)


@app.get("/api/doc/<doc_id>")
def doc_info(doc_id):
    return jsonify(load_state(doc_id))


@app.delete("/api/doc/<doc_id>")
def delete_doc(doc_id):
    d = doc_dir(doc_id)
    for j in _jobs.values():                      # stop anything still running for this document
        if j.get("doc") == doc_id:
            j["cancel"] = True
    shutil.rmtree(d, ignore_errors=True)
    return jsonify(ok=True)


@app.get("/api/doc/<doc_id>/page/<int:page>")
def page_info(doc_id, page):
    return jsonify(load_state(doc_id)["pages"].get(str(page), {"text": "", "in_notebook": False}))


@app.get("/api/doc/<doc_id>/page/<int:page>/image")
def page_image(doc_id, page):
    if not 1 <= page <= load_state(doc_id)["page_count"]:
        abort(404)
    zoom = min(max(float(request.args.get("zoom", 1.3)), 0.3), 4.0)
    resp = Response(render_page(doc_id, page, zoom), mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


@app.post("/api/doc/<doc_id>/process")
def process(doc_id):
    st = load_state(doc_id)
    j = request.get_json(force=True)
    avail = {e["id"] for e in engine_list() if e["available"]}
    engine = j.get("engine") or ("mineru_local" if "mineru_local" in avail else "auto")
    if engine not in avail:
        return jsonify(error=f"engine '{engine}' is not available (install it or set its env vars)"), 400
    if "pages" in j:
        pages = sorted({int(p) for p in j["pages"]})
    else:
        pages = list(range(int(j.get("from", 1)), int(j.get("to", st["page_count"])) + 1))
    pages = [p for p in pages if 1 <= p <= st["page_count"]]
    if not pages:
        return jsonify(error="no valid pages"), 400
    box = j.get("box")
    if box:
        x0, y0, x1, y1 = [min(max(float(v), 0), 1) for v in box]
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            return jsonify(error="selection too small"), 400
        box = (x0, y0, x1, y1)
        pages = pages[:1]
    formula = bool(j.get("formula", MINERU_FORMULA))
    _prune_jobs()
    jid = uuid.uuid4().hex[:10]
    _jobs[jid] = {"id": jid, "doc": doc_id, "engine": engine, "status": "running", "total": len(pages), "done": 0,
                  "msg": "starting", "results": {}, "errors": {}, "cancel": False, "error": None}
    threading.Thread(target=run_job, args=(jid, doc_id, pages, engine, box, formula, bool(j.get("auto_nb"))),
                     daemon=True).start()
    return jsonify(job_id=jid)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    return jsonify(_jobs.get(job_id) or {"status": "error", "error": "unknown job", "results": {}, "errors": {}})


@app.post("/api/job/<job_id>/cancel")
def job_cancel(job_id):
    if job_id in _jobs:
        _jobs[job_id]["cancel"] = True
    return jsonify(ok=True)


@app.post("/api/doc/<doc_id>/page/<int:page>/layer")
def page_layer(doc_id, page):
    if not 1 <= page <= load_state(doc_id)["page_count"]:
        return jsonify(error="bad page"), 400
    text = clean_ocr(run_engine("layer", doc_id, page, None, False, lambda m: None, lambda: False))
    if len(text) < 20:
        return jsonify(error="No text layer on this page (scanned image). Use OCR instead."), 422
    return jsonify(update_page(doc_id, page, text=text, engine="layer", quality=quality(text), in_notebook=False))


@app.post("/api/doc/<doc_id>/page/<int:page>/text")
def save_text(doc_id, page):
    if not 1 <= page <= load_state(doc_id)["page_count"]:
        return jsonify(error="bad page"), 400
    j = request.get_json(force=True)
    fields = {}
    if "text" in j:
        fields["text"] = str(j["text"])
    if "in_notebook" in j:
        fields["in_notebook"] = bool(j["in_notebook"])
    return jsonify(update_page(doc_id, page, **fields))


@app.post("/api/doc/<doc_id>/bulk")
def bulk(doc_id):
    """action: to_nb (processed -> notebook) | from_nb (notebook -> processed) | delete (drop the text) | clean (fix text/math)."""
    j = request.get_json(force=True)
    action = j.get("action")
    if action not in ("to_nb", "from_nb", "delete", "clean"):
        return jsonify(error="unknown action"), 400
    totals = {}
    with _state_lock:
        st = load_state(doc_id)
        pages = sorted({int(p) for p in j.get("pages", []) if 1 <= int(p) <= st["page_count"]})
        out = {}
        for n in pages:
            p = st["pages"].get(str(n))
            if action == "delete":
                st["pages"].pop(str(n), None)
                out[str(n)] = {"text": "", "in_notebook": False}
            elif p and p.get("text", "").strip():
                if action == "clean":
                    t, ch = fix_text(p["text"])
                    if t.strip():
                        p["text"] = t
                        if "quality" in p:
                            p["quality"] = quality(t)
                    for k, v in ch.items():
                        totals[k] = totals.get(k, 0) + v
                else:
                    p["in_notebook"] = action == "to_nb"
                out[str(n)] = p
        save_state(doc_id, st)
    return jsonify(pages=out, changes=totals)


@app.post("/api/clean")
def clean():
    t, ch = fix_text(str(request.get_json(force=True).get("text", "")))
    return jsonify(text=t, changes=ch)


@app.get("/api/doc/<doc_id>/notebook")
def notebook(doc_id):
    st = load_state(doc_id)
    items = [{"page": int(n), "text": p["text"]} for n, p in st["pages"].items()
             if p.get("in_notebook") and p.get("text", "").strip()]
    return jsonify(sorted(items, key=lambda x: x["page"]))


def _nb_items(st):
    return sorted((int(n), p["text"]) for n, p in st["pages"].items()
                  if p.get("in_notebook") and p.get("text", "").strip())



def plain_text(t):
    t = COL_LINE_RE.sub("", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", plain(t)).strip()


@app.post("/api/doc/<doc_id>/meta")
def set_meta(doc_id):
    title = re.sub(r"\s+", " ", str(request.get_json(force=True).get("title", "")).strip())[:200]
    if not title:
        return jsonify(error="Name is empty"), 400
    with _state_lock:
        st = load_state(doc_id)
        st["title"] = title
        save_state(doc_id, st)
    return jsonify(title=title)


@app.get("/api/doc/<doc_id>/export")
def export_notebook(doc_id):
    st = load_state(doc_id)
    fmt = request.args.get("format", "md").lower()
    pages_sep = request.args.get("pages", "1") != "0"
    cols = request.args.get("columns", "1") != "0"
    items = _nb_items(st)
    if not items:
        return jsonify(error="Notebook is empty"), 400
    base = (st.get("title") or re.sub(r"\.pdf$", "", st["name"], flags=re.I)).strip()
    safe = re.sub(r'[\\/:*?"<>|]+', "_", base)

    def body(t, style):
        if not cols:
            return re.sub(r"\n{3,}", "\n\n", COL_LINE_RE.sub("", t)).strip()
        if style == "md":
            return COL_LINE_RE.sub(lambda m: f"*— column {m.group(1)} —*", t)
        return t

    if fmt in ("md", "txt"):
        parts = []
        for i, (n, t) in enumerate(items):
            if fmt == "md":
                head = (("" if i == 0 else "---\n\n") + f"*[Page {n}]*") if pages_sep else f"<!-- page {n} -->"
            else:
                head = f"{'═' * 16} Page {n} {'═' * 16}" if pages_sep else ""
            parts.append((head + "\n\n" + body(t, fmt)).strip())
        top = f"# {base}\n\n" if fmt == "md" else f"{base}\n{'=' * len(base)}\n\n"
        data, mime = top + "\n\n".join(parts) + "\n", ("text/markdown" if fmt == "md" else "text/plain")
    elif fmt == "json":
        pg, have = [], set()
        for n, t in items:
            b = body(t, "txt")
            meta = st["pages"].get(str(n), {})
            pg.append({"page": n, "text": b, "plain": plain_text(t), "blocks": blocks_json(parse_blocks(b)),
                       "engine": meta.get("engine"), "quality": meta.get("quality")})
            have.add(n)
        data = json.dumps({
            "schema": "butterfly.doc.v1", "doc_id": st["id"], "name": base, "source_file": st["name"],
            "page_count": st["page_count"], "pages_included": len(pg),
            "pages_missing": [n for n in range(1, st["page_count"] + 1) if n not in have],
            "exported_at": datetime.now().isoformat(timespec="seconds"), "pages": pg},
            ensure_ascii=False, indent=1)
        mime = "application/json"
    elif fmt == "html":
        out = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{htmllib.escape(base)}</title>"
               "<style>body{font-family:Georgia,serif;max-width:820px;margin:2em auto;padding:0 1em;line-height:1.6;color:#1b1b1b}"
               "h1,h2,h3,h4{font-family:system-ui,sans-serif;line-height:1.25;margin:1.4em 0 .5em}"
               "h1.doc{border-bottom:2px solid #ddd;padding-bottom:.3em}p{margin:.7em 0}li{margin:.25em 0}"
               "hr{border:0;border-top:2px solid #bbb;margin:2em 0}.pg{color:#888;font:italic 12px sans-serif;margin:1.2em 0 .4em}"
               ".col{color:#777;font:italic 13px sans-serif;border-top:1px dashed #bbb;margin:1.2em 0 .6em;padding-top:4px}"
               "code{background:#f1f1f1;padding:0 4px;border-radius:3px}</style></head><body>"
               f"<h1 class='doc'>{htmllib.escape(base)}</h1>"]
        for i, (n, t) in enumerate(items):
            if pages_sep:
                out.append(("" if i == 0 else "<hr>") + f"<div class='pg' id='p{n}'>Page {n}</div>")
            for b in parse_blocks(body(t, "txt")):
                if b[0] == "h":
                    lv = min(b[1], 6)
                    out.append(f"<h{lv}>{inline_html(b[2])}</h{lv}>")
                elif b[0] in ("ul", "ol"):
                    out.append(f"<{b[0]}>" + "".join(f"<li>{inline_html(x)}</li>" for x in b[1]) + f"</{b[0]}>")
                elif b[0] == "col":
                    out.append(f"<div class='col'>column {b[1]}</div>")
                else:
                    out.append(f"<p>{inline_html(b[1])}</p>")
        data, mime = "".join(out) + "</body></html>", "text/html"
    elif fmt == "docx":
        try:
            import docx
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.shared import Inches, Pt, RGBColor
        except ImportError:
            return jsonify(error="Word export needs:  pip install python-docx"), 400
        d = docx.Document()
        d.add_heading(base, 0)
        for n, t in items:
            if pages_sep:
                pp = d.add_paragraph()
                r = pp.add_run(f"Page {n}")
                r.italic = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            for b in parse_blocks(body(t, "txt")):
                if b[0] == "h":
                    d.add_heading(re.sub(r"[*`]+", "", b[2]), min(b[1], 9))
                elif b[0] == "ul":
                    for it in b[1]:
                        add_runs(d.add_paragraph(style="List Bullet"), it)
                elif b[0] == "ol":
                    for j, it in enumerate(b[1], 1):          # numbered by hand: Word's own list keeps counting across lists
                        pp = d.add_paragraph()
                        pp.paragraph_format.left_indent = Inches(0.35)
                        pp.paragraph_format.first_line_indent = Inches(-0.25)
                        add_runs(pp, f"{j}. {it}")
                elif b[0] == "col":
                    pp = d.add_paragraph()
                    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    pp.add_run(f"— column {b[1]} —").italic = True
                else:
                    add_runs(d.add_paragraph(), b[1])
        buf = io.BytesIO()
        d.save(buf)
        data, mime = buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        return jsonify(error=f"unknown format '{fmt}'"), 400
    return Response(data, mimetype=mime,
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe)}.{fmt}"})


@app.get("/api/doc/<doc_id>/notebook.md")
def notebook_md(doc_id):
    return export_notebook(doc_id)


@app.get("/api/doc/<doc_id>/search")
def search(doc_id):
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    related = request.args.get("related") == "1"
    try:
        ex = exact_hits(doc_id, q)
    except Exception:  # noqa: BLE001
        ex = []
    if ex and not related:
        return jsonify(ex)                                   # exact matches only
    try:
        sem = semantic_search(doc_id, q)
    except Exception as e:  # noqa: BLE001
        if not ex:
            return jsonify(error=str(e)), 500
        sem = []
    if related:                                              # keep only results that are actually close
        sem = [h for h in sem if h["keywords"] > 0 or (h.get("raw") or 0) >= SEM_MIN]
    else:                                                    # no exact hit: allow partial word matches only
        sem = [h for h in sem if h["keywords"] > 0]
    flag = sem[0]["sem"] if sem else True
    for h in ex:
        h["sem"] = flag
    norm = lambda s: re.sub(r"\W+", "", s.lower())
    seen = [(h["page"], norm(h["text"] or h["heading"])) for h in ex]
    rest = [h for h in sem if not any(h["page"] == p and k and k in norm(h["heading"] + h["text"])
                                      for p, k in seen)]
    return jsonify(ex + rest)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)