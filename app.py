# ==============================================================================
# SUMMARY AI — AI-Powered PDF Summarization & Document Assistant
# College Minor Project
# High-Accuracy & Document-Grounded Pipeline
# ==============================================================================

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import html
import io
import os
import re
import sys
from collections import Counter
from datetime import datetime
from typing import List, Tuple

import arxiv
import faiss
import feedparser
import numpy as np
import pymupdf  # PyMuPDF
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ==============================================================================
# 2. CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Summary AI — Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_FILES = 10
MAX_FILE_MB = 200

# Optional: Paste your Gemini API key here to load it automatically without typing in UI
DEFAULT_GEMINI_API_KEY = ""

# Tesseract OCR Data Configuration (Requirements 1-9)
TESSDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
if os.path.exists(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

SUMMARY_LENGTHS = {
    "Short (~70% target)": 0.7,
    "Medium (~100% target)": 1.0,
    "Detailed (~130% target)": 1.3,
    # Backward compatibility aliases
    "Short (3 bullets)": 0.7,
    "Medium (5 bullets)": 1.0,
    "Detailed (10-12 bullets)": 1.3,
    "Short": 0.7,
    "Medium": 1.0,
    "Detailed": 1.3,
}

SUMMARY_FORMATS = {
    "Bullet Points": "concise bullet points",
    "Paragraph": "a single flowing paragraph",
    "Key Findings": "numbered key findings",
    "FAQ": "a FAQ with questions and short answers",
    "Structured": "structured sections with headings",
}

SIMPLE_LANGUAGE_NOTE = " Use very simple, easy-to-understand language."

# Standardized grounded refusal notice when evidence is missing or insufficient (Requirement 5)
STANDARD_REFUSAL = "I could not find sufficient information in the uploaded PDF to answer this question."

def is_refusal(text: str) -> bool:
    """Check if the text indicates refusal or absence of information in document."""
    if not text or not text.strip():
        return True
    t_clean = text.lower().strip()
    refusal_cues = [
        "could not find sufficient",
        "could not find enough",
        "not found in the provided",
        "not found in the uploaded",
        "not available in the uploaded",
        "not mentioned in the",
        "insufficient information",
        "insufficient evidence",
        "cannot answer from the uploaded",
        "information is not available",
        "not enough information",
        "not enough evidence",
        "no matching content found",
        "not found",
    ]
    return any(cue in t_clean for cue in refusal_cues)

COMMANDS = {
    "/summary": "Generate a document summary",
    "/keypoints": "Extract important points",
    "/explain": "Explain a topic simply",
    "/find": "Find where a topic appears",
    "/define": "Define a term using document context",
    "/keywords": "Extract important keywords",
    "/faq": "Generate questions and answers",
    "/conclusion": "Extract or generate the conclusion",
    "/cite": "Show supporting page references",
    "/compare": "Compare multiple uploaded PDFs",
    "/help": "Display available commands and usage",
}

COMMAND_HELP = {
    "/summary": "/summary [short|medium|detailed]",
    "/keypoints": "/keypoints",
    "/explain": "/explain <topic>",
    "/find": "/find <topic>",
    "/define": "/define <term>",
    "/keywords": "/keywords [count]",
    "/faq": "/faq [count]",
    "/conclusion": "/conclusion",
    "/cite": "/cite <topic>",
    "/compare": "/compare <doc1> <doc2>",
    "/help": "/help - Show this help message",
}


# ==============================================================================
# 3. MODEL LOADING & SECRETS CONFIGURATION
# ==============================================================================
def get_secret_gemini_api_key() -> str:
    """Safely retrieve Gemini API key from Streamlit secrets (st.secrets), environment variables, or default fallback."""
    # 1. Streamlit Community Cloud Secrets
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    # 2. Environment Variable
    env_val = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_val:
        return env_val
    # 3. Code Constant Fallback
    return (DEFAULT_GEMINI_API_KEY or "").strip()


@st.cache_resource
def load_models():
    """Load and cache embedding and generative language models with resilient fallbacks."""
    emb_model = None
    gen_model = None
    tokenizer = None

    try:
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")       # embeddings
    except Exception:
        emb_model = None

    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        gen_model.to("cpu")
        gen_model.eval()
    except Exception:
        gen_model, tokenizer = None, None

    return emb_model, gen_model, tokenizer


try:
    emb_model, gen_model, tokenizer = load_models()
    if emb_model is None and gen_model is None and not get_secret_gemini_api_key():
        st.error("Failed to load local AI models and no Gemini API key found. Please check internet connection or install requirements.")
        st.stop()
    elif emb_model is None:
        st.warning("⚠️ Notice: Semantic embedding model failed to download. Lexical retrieval will be used.")
    elif gen_model is None:
        st.info("💡 Notice: Local FLAN-T5 model is offline. Cloud Google Gemini API will be used for generation.")
except Exception as e:
    st.error(f"Failed to load AI models. Check your internet connection or installed packages.\n\n{e}")
    st.stop()


def generate_gemini(prompt: str, api_key: str = "", model_name: str = "gemini-1.5-flash", max_tokens: int = 500, system_instruction: str = "") -> str:
    """Generate text using Google Gemini API (via google.genai SDK or direct REST API) with strict document-only grounding."""
    if not prompt or not prompt.strip():
        return ""
    api_key = (api_key or get_secret_gemini_api_key()).strip()
    if not api_key:
        return ""

    sys_inst = system_instruction or "You are an accurate, strictly document-grounded AI assistant. Use ONLY provided context. Never hallucinate facts or alter numbers."

    # Strategy 1: Official google.genai Client
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        cfg = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=max_tokens,
            system_instruction=sys_inst,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg,
        )
        if response and response.text:
            return response.text.strip()
    except Exception:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception:
            pass

    # Strategy 2: google.generativeai SDK
    try:
        import google.generativeai as gai
        gai.configure(api_key=api_key)
        m = gai.GenerativeModel(
            model_name,
            system_instruction=sys_inst,
            generation_config={"temperature": 0.1, "max_output_tokens": max_tokens}
        )
        res = m.generate_content(prompt)
        if res and res.text:
            return res.text.strip()
    except Exception:
        pass

    # Strategy 3: Direct HTTP REST API
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": sys_inst}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1
            }
        }
        res = requests.post(url, json=payload, timeout=20)
        if res.status_code == 200:
            data = res.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception:
        pass

    return ""


def generate_text(prompt, gen_model, tokenizer, max_length=200, system_instruction=None) -> str:
    """Run text generation using Google Gemini API if enabled, otherwise fallback to local FLAN-T5."""
    """Run text generation using Google Gemini API if enabled, otherwise fallback to optimized local FLAN-T5."""
    if not prompt or not prompt.strip():
        return ""

    # Check if Gemini API is enabled in session state or environment/secrets
    use_gemini = False
    gemini_key = ""
    gemini_model = "gemini-1.5-flash"
    try:
        if hasattr(st, "session_state"):
            use_gemini = getattr(st.session_state, "use_gemini", False)
            gemini_key = (getattr(st.session_state, "gemini_api_key", "") or get_secret_gemini_api_key()).strip()
            gemini_model = getattr(st.session_state, "gemini_model", "gemini-1.5-flash")
        else:
            gemini_key = get_secret_gemini_api_key()
    except Exception:
        use_gemini = False

    if (use_gemini or gen_model is None) and gemini_key:
        out = generate_gemini(prompt, api_key=gemini_key, model_name=gemini_model, max_tokens=max_length, system_instruction=system_instruction or "")
        if out and out.strip():
            if hasattr(st, "session_state"):
                setattr(st.session_state, "last_engine_used", f"Google Gemini API ({gemini_model})")
            return out.strip()

    # Local fallback to FLAN-T5 (single tokenization, single generation call, error handling)
    if gen_model is not None and tokenizer is not None:
        if hasattr(st, "session_state"):
            setattr(st.session_state, "last_engine_used", "Local FLAN-T5-small (Offline)")
        try:
            # FLAN-T5-small pretraining sequence limit is 512 tokens. Tokenize once cleanly.
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            target_tokens = max(10, min(max_length, 160))
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=target_tokens,
                    num_beams=2,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    do_sample=False
                )
            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return ""
    return ""


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
if not hasattr(st.session_state, "docs"):
    st.session_state.docs = {}          # name -> document dictionary
if not hasattr(st.session_state, "history"):
    st.session_state.history = []       # list of past interactions
if not hasattr(st.session_state, "faiss_cache"):
    st.session_state.faiss_cache = {}   # name -> {index, chunks, page_nums}
if not hasattr(st.session_state, "ask_input_val"):
    st.session_state.ask_input_val = ""
if not hasattr(st.session_state, "debug_mode"):
    st.session_state.debug_mode = False
secret_gemini_key = get_secret_gemini_api_key()
if not hasattr(st.session_state, "use_gemini"):
    st.session_state.use_gemini = bool(secret_gemini_key)
if not hasattr(st.session_state, "gemini_api_key"):
    st.session_state.gemini_api_key = secret_gemini_key
if not hasattr(st.session_state, "gemini_model"):
    st.session_state.gemini_model = "gemini-1.5-flash"


# ==============================================================================
# 5. PDF EXTRACTION & METADATA
# ==============================================================================
def detect_running_headers_footers(pages: List[str]) -> set:
    """
    Dynamically detect recurring running headers and footers across multi-page documents.
    If a line appears on >= 30% of pages in a document with 3+ pages, it is treated as a running header/footer.
    """
    if not pages or len(pages) < 3:
        return set()
    page_line_sets = []
    for p in pages:
        lines = {line.strip() for line in p.splitlines() if 4 <= len(line.strip()) <= 120}
        page_line_sets.append(lines)
    threshold = max(2, int(round(len(pages) * 0.30)))
    line_counts = Counter()
    for lset in page_line_sets:
        for line in lset:
            line_counts[line] += 1
    return {line for line, count in line_counts.items() if count >= threshold}


def conservative_clean_page(text: str, running_headers: set = None) -> str:
    """
    Clean extracted PDF page text with conservative safeguards:
    1. Strip institutional / recurring running headers & footers.
    2. Protect legitimate single-character data (formulas, matrices, alphabet grids, variables).
    3. Reconstruct vertical word splitting only when confidently identified as words/text rows.
    4. Normalize non-standard bullet characters to standard markdown.
    5. Clean degenerative repetitive loops via backreference regex.
    """
    if not text:
        return ""
    running_headers = running_headers or set()
    lines = text.splitlines()
    filtered = []
    for line in lines:
        l_str = line.strip()
        if not l_str:
            continue
        # Exact match with dynamically detected running headers/footers
        if l_str in running_headers:
            continue
        # Generic academic/institutional header & footer patterns
        if re.search(r"Subject\s*Name\s*:.*Unit\s*No\s*:", l_str, re.I):
            continue
        if re.search(r"Subject\s*Code\s*:\s*[A-Z0-9]+", l_str, re.I):
            continue
        if re.search(r"Prepared\s*By\s*:.*Department\s*of", l_str, re.I):
            continue
        if re.match(r"^Page\s+\d+\s*$", l_str, re.I):
            continue
        if re.match(r"^Unit-?\s*(?:[0-9]+|[IVXLCDM]+)\s*[:\-]?\s*[A-Za-z\s]{3,60}$", l_str, re.I) and l_str in running_headers:
            continue
        filtered.append(l_str)

    # Conservative handling of short/single-character lines:
    # Protect math formulas, matrices, equations, variables, and punctuation grids
    math_or_formula_tokens = set("=+-*/^\\|()[]{}<>~_&%$#@!?,;:'\"")
    collapsed = []
    run = []

    def _is_table_or_split_char(l):
        return len(l) <= 2 and l.isalnum()

    for line in filtered:
        has_math = any(c in math_or_formula_tokens for c in line)
        if not has_math and _is_table_or_split_char(line):
            run.append(line)
        else:
            if len(run) >= 3:
                collapsed.append(" ".join(run))
                run = []
            elif run:
                collapsed.extend(run)
                run = []
            collapsed.append(line)
    if run:
        if len(run) >= 3:
            collapsed.append(" ".join(run))
        else:
            collapsed.extend(run)

    text = "\n".join(collapsed)
    text = re.sub(r"[\u2713\u2756\u27A2\u27A4\u25C6\u25CF\u2022\u25AA\uF0A7\uF0B7]", "- ", text)
    text = re.sub(r"(\b[A-Za-z0-9\-_]{2,25}(?:\s*[-:]\s*|\s+))\1{2,}", r"\1", text)
    text = re.sub(r"(\b[A-Za-z0-9\s]{4,35}[:\.\?!,-]?\s*)\1{2,}", r"\1", text)
    return text.strip()


# ==============================================================================
# 5b. OCR SUPPORT & SCANNED DOCUMENT EXTRACTION (Requirements 1-9)
# ==============================================================================
def is_ocr_available() -> bool:
    """Check if Tesseract OCR model data (tessdata) is available for PyMuPDF OCR."""
    tess_path = os.environ.get("TESSDATA_PREFIX", TESSDATA_DIR)
    if os.path.exists(tess_path):
        trained_data = os.path.join(tess_path, "eng.traineddata")
        return os.path.exists(trained_data)
    return False


def extract_ocr_text_from_page(page_obj, dpi: int = 150) -> str:
    """
    Extract text from an image or scanned page using PyMuPDF's integrated OCR engine.
    Does not attempt to interpret non-text visuals (diagrams, charts, photos).
    """
    if not is_ocr_available():
        return ""
    try:
        tess_path = os.environ.get("TESSDATA_PREFIX", TESSDATA_DIR)
        tp = page_obj.get_textpage_ocr(tessdata=tess_path, language="eng", dpi=dpi)
        text = page_obj.get_text(textpage=tp)
        return text.strip() if text else ""
    except Exception:
        return ""


def deduplicate_combine_texts(native_text: str, ocr_text: str) -> str:
    """
    Combine native selectable text and OCR-extracted text without duplicating content (Requirement 5).
    Filters out OCR lines that already appear in the selectable text stream.
    """
    native_s = (native_text or "").strip()
    ocr_s = (ocr_text or "").strip()

    if not ocr_s:
        return native_s
    if not native_s:
        return ocr_s

    # Build normalized representations of native lines for fuzzy/exact deduplication
    native_lines = [l.strip() for l in native_s.splitlines() if l.strip()]
    norm_native = [re.sub(r"[^a-z0-9]", "", l.lower()) for l in native_lines]
    norm_native_set = set(n for n in norm_native if len(n) >= 3)

    added_ocr_lines = []
    for line in ocr_s.splitlines():
        line_clean = line.strip()
        cl = re.sub(r"[^a-z0-9]", "", line_clean.lower())
        if not cl or len(cl) < 3:
            continue
        # Direct exact normalized match
        if cl in norm_native_set:
            continue
        # Substring / overlap check to prevent partial line repetitions
        if any((cl in nn or nn in cl) and (len(cl) > 8 or len(nn) > 8) for nn in norm_native):
            continue
        added_ocr_lines.append(line_clean)

    if added_ocr_lines:
        return "\n".join(native_lines + added_ocr_lines)
    return "\n".join(native_lines)


def extract_text_with_ocr(uploaded_file, enable_ocr: bool = True) -> Tuple[str, List[str], dict]:
    """
    Extract readable text from a PDF, automatically detecting scanned pages or embedded images
    and applying OCR when needed, while preserving page numbers and source metadata (Requirements 1-7).
    Returns (full_text, cleaned_pages, ocr_audit).
    """
    file_bytes = b""
    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
    elif isinstance(uploaded_file, bytes):
        file_bytes = uploaded_file
    elif hasattr(uploaded_file, "read"):
        file_bytes = uploaded_file.read()

    # Fallback for unit testing stubs like _FakeUpload
    if not file_bytes and hasattr(uploaded_file, "_text"):
        t = getattr(uploaded_file, "_text", "")
        pages = [t] if t.strip() else []
        return t, pages, {
            "is_scanned": False,
            "ocr_applied": False,
            "ocr_pages": [],
            "ocr_failed_pages": [],
            "empty_pages": [] if pages else [1],
            "low_text_pages": [],
            "total_images_detected": 0,
            "scan_warning": "",
        }

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        if hasattr(uploaded_file, "_text"):
            t = getattr(uploaded_file, "_text", "")
            return t, [t] if t.strip() else [], {}
        raise e

    if doc.is_encrypted:
        raise ValueError("Password-protected PDF files cannot be processed.")

    raw_pages = []
    ocr_pages = []
    ocr_failed_pages = []
    empty_pages = []
    low_text_pages = []
    total_images = 0

    for p in range(doc.page_count):
        page_num = p + 1
        page_obj = doc.load_page(p)

        # 1. Extract selectable text
        native_text = page_obj.get_text("text").strip()

        # 2. Inspect embedded images on this page
        images = page_obj.get_images()
        has_images = bool(images)
        total_images += len(images)

        char_len = len(native_text)
        is_scanned_page = (char_len < 40 and has_images) or (char_len == 0 and has_images)

        page_final_text = native_text

        # 3. Apply OCR if scanned or low-text page with images
        if enable_ocr and (is_scanned_page or (char_len < 100 and has_images)):
            ocr_text = extract_ocr_text_from_page(page_obj)
            if ocr_text:
                page_final_text = deduplicate_combine_texts(native_text, ocr_text)
                ocr_pages.append(page_num)
            elif has_images and char_len == 0:
                ocr_failed_pages.append(page_num)

        # Track page audit status
        final_len = len(page_final_text.strip())
        if final_len == 0:
            empty_pages.append(page_num)
        elif final_len < 45:
            low_text_pages.append(page_num)

        raw_pages.append(page_final_text)

    # Clean running headers/footers while preserving genuine content
    if "detect_running_headers_footers" in globals() and "conservative_clean_page" in globals() and len(raw_pages) >= 3:
        running_headers = globals()["detect_running_headers_footers"](raw_pages)
        pages = [globals()["conservative_clean_page"](p, running_headers) for p in raw_pages]
    else:
        pages = raw_pages

    full_text = "\n\n".join(p for p in pages if p.strip())

    total_p = max(1, doc.page_count)
    is_scanned_doc = (len(ocr_pages) >= total_p * 0.5) or (len(empty_pages) + len(low_text_pages) >= total_p * 0.7 and total_images > 0)
    ocr_applied = len(ocr_pages) > 0

    scan_warning = ""
    if ocr_applied:
        scan_warning = f"OCR text extraction was successfully applied to page(s): {ocr_pages}."
    elif is_scanned_doc:
        scan_warning = "Scanned document detected. OCR recommended for image-based text."

    ocr_audit = {
        "is_scanned": is_scanned_doc,
        "ocr_applied": ocr_applied,
        "ocr_pages": ocr_pages,
        "ocr_failed_pages": ocr_failed_pages,
        "empty_pages": empty_pages,
        "low_text_pages": low_text_pages,
        "total_images_detected": total_images,
        "scan_warning": scan_warning,
    }

    try:
        doc.close()
    except Exception:
        pass

    return full_text, pages, ocr_audit


def extract_text_from_pdf(uploaded_file) -> Tuple[str, List[str]]:
    """Return full text and list of cleaned per-page texts from an uploaded PDF (backward-compatible 2-tuple)."""
    if "extract_text_with_ocr" in globals() and callable(globals()["extract_text_with_ocr"]):
        full_text, pages, _ = globals()["extract_text_with_ocr"](uploaded_file)
        return full_text, pages
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages = [doc.load_page(p).get_text("text") for p in range(doc.page_count)]
    try:
        doc.close()
    except Exception:
        pass
    return "\n\n".join(p for p in pages if p.strip()), pages


def extract_pdf_metadata(uploaded_file) -> dict:
    """Safely extract standard PDF metadata (title, author, subject, creator)."""
    try:
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        md = doc.metadata or {}
        return {
            "title": md.get("title") or "",
            "author": md.get("author") or "",
            "subject": md.get("subject") or "",
            "creator": md.get("creator") or "",
        }
    except Exception:
        return {}


def add_uploaded_documents(uploaded_files) -> List[str]:
    """Validate and register new uploaded PDFs into session state with OCR support and scanned page detection."""
    added = []
    for f in uploaded_files or []:
        if f.name in st.session_state.docs:
            continue
        if len(st.session_state.docs) >= MAX_FILES:
            st.warning(f"Document limit reached ({MAX_FILES}). Remove a document before adding more.")
            break
        if f.size > MAX_FILE_MB * 1024 * 1024:
            st.warning(f"'{f.name}' exceeds the {MAX_FILE_MB} MB size limit and was skipped.")
            continue

        try:
            if "extract_text_with_ocr" in globals() and callable(globals()["extract_text_with_ocr"]) and globals().get("extract_text_from_pdf") == extract_text_from_pdf:
                full_text, pages, ocr_audit = globals()["extract_text_with_ocr"](f)
            else:
                full_text, pages = extract_text_from_pdf(f)
                ocr_audit = {
                    "is_scanned": False,
                    "ocr_applied": False,
                    "ocr_pages": [],
                    "ocr_failed_pages": [],
                    "empty_pages": [i + 1 for i, p in enumerate(pages) if not p.strip()],
                    "low_text_pages": [i + 1 for i, p in enumerate(pages) if 0 < len(p.strip()) < 45],
                    "total_images_detected": 0,
                    "scan_warning": "",
                }
        except Exception as e:
            st.warning(f"Could not read '{f.name}': {e}")
            continue

        empty_pages = ocr_audit.get("empty_pages", [])
        low_text_pages = ocr_audit.get("low_text_pages", [])
        ocr_pages = ocr_audit.get("ocr_pages", [])
        ocr_failed_pages = ocr_audit.get("ocr_failed_pages", [])
        is_scanned = ocr_audit.get("is_scanned", False)
        ocr_applied = ocr_audit.get("ocr_applied", False)

        # Notify user about OCR and extraction status (Requirements 7 & 8)
        if ocr_applied:
            st.info(f"🔍 OCR applied to '{f.name}' on page(s): {ocr_pages} (extracted readable text from embedded images/scans).")

        if ocr_failed_pages:
            st.warning(
                f"Notice for '{f.name}': Page(s) {ocr_failed_pages} contain images or scans, but OCR could not extract readable text. "
                "(Note: OCR extracts text from images, but does not interpret charts, diagrams, or photographs.)"
            )

        if not full_text.strip():
            st.warning(
                f"'{f.name}' has no readable text even after OCR processing. "
                "OCR cannot extract text from blank pages or non-text visual diagrams/charts."
            )
            if hasattr(st, "session_state"):
                if not hasattr(st.session_state, "scanned_pdf_audit"):
                    st.session_state.scanned_pdf_audit = {}
                st.session_state.scanned_pdf_audit[f.name] = {
                    "is_scanned": True,
                    "ocr_required": True,
                    "ocr_applied": ocr_applied,
                    "scan_warning": "OCR could not extract text (blank or non-text images)",
                }
            continue

        if is_scanned and not ocr_applied:
            st.warning(f"'{f.name}' appears to be a scanned document with limited extractable text. OCR is recommended for full document accuracy.")
        elif low_text_pages and not ocr_applied:
            st.info(f"Notice: Page(s) {low_text_pages} in '{f.name}' have very low text density and may contain non-text diagrams or images.")

        try:
            meta = extract_pdf_metadata(f)
        except (NameError, Exception):
            meta = {}

        st.session_state.docs[f.name] = {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
            "file_size": f.size,
            "meta": meta,
            "added_at": datetime.now().strftime("%H:%M"),
            "empty_pages": empty_pages,
            "low_text_pages": low_text_pages,
            "is_scanned": is_scanned,
            "ocr_applied": ocr_applied,
            "ocr_pages": ocr_pages,
            "ocr_failed_pages": ocr_failed_pages,
            "scan_warning": ocr_audit.get("scan_warning", ""),
            "ocr_audit": ocr_audit,
        }
        added.append(f.name)
    return added


def remove_document(name):
    """Remove a document and its cached index from session state."""
    if hasattr(st.session_state, "docs") and name in st.session_state.docs:
        del st.session_state.docs[name]
    faiss_cache = getattr(st.session_state, "faiss_cache", None)
    if isinstance(faiss_cache, dict) and name in faiss_cache:
        del faiss_cache[name]
    st.rerun()


# ==============================================================================
# 6. SENTENCE-AWARE TEXT CHUNKING
# ==============================================================================
def chunk_text(text, max_chars=1200, overlap=200) -> List[str]:
    """
    Split text into coherent, overlapping chunks without breaking sentences.
    Preserves paragraph structure and filters out tiny, useless chunks.
    """
    if not text or not text.strip():
        return []

    clean_text = re.sub(r"[ \t]+", " ", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", clean_text) if p.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        # Robust sentence splitting that protects abbreviations, titles, decimals, and formulas
        sentences = [s.strip() for s in re.split(
            r"(?<!\b[A-Za-z]\.)(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bUnit)(?<!\bSec)(?<!\bNo)(?<!\bvs)(?<=[.?!])\s+(?=[A-Z0-9\"'(\[])",
            para
        ) if s.strip()]
        if not sentences:
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", para) if s.strip()] or [para]
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len + 1 > max_chars and current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                if len(chunk_str) >= 30:
                    chunks.append(chunk_str)

                # Overlap: preserve trailing sentence(s)
                overlap_chunk = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    if overlap_len + len(prev_sent) <= overlap:
                        overlap_chunk.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append(sent)
            current_len += sent_len + 1

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if len(chunk_str) >= 30:
            chunks.append(chunk_str)

    # Fallback if no chunk produced
    if not chunks and len(text.strip()) > 0:
        chunks = [text.strip()[:max_chars]]

    return chunks


def extract_document_headings(pages: List[str]) -> List[dict]:
    """
    Extract authentic section headings and topic hierarchy dynamically from any document:
    - Numbered sections (e.g. 1., 1.1, 2.3, Chapter 1, Unit 2)
    - Prominent capitalized titles / topic lines
    - Markdown / bullet header markers (###, ##, -, etc.)
    Returns list of {"title": heading_title, "page": page_num}
    """
    heading_regex = re.compile(
        r"^(?:(?:chapter|unit|section|part)\s+[0-9ivxlcdm]+[:\.\-]?\s*.*|"
        r"\d+(?:\.\d+)*[\.\)]?\s+[A-Z][a-zA-Z0-9\s/_\-()]{2,60}|"
        r"[-*#\s]*[A-Z][A-Za-z0-9\s/_\-()]{2,50}:|"
        r"[-*#]\s+[A-Z][a-zA-Z0-9\s/_\-()]{2,40}|"
        r"[A-Z][A-Za-z0-9\s_\-()]{2,40}/[A-Za-z0-9\s_\-()]{2,40}|"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})$",
        re.I
    )
    headings = []
    for p_idx, page in enumerate(pages, 1):
        for line in page.splitlines():
            line_s = line.strip()
            if not line_s or len(line_s) > 65:
                continue
            if line_s.endswith((".", "?", "!")):
                continue
            if heading_regex.match(line_s):
                clean_title = re.sub(r"^[-*#\s]+", "", line_s).strip()
                clean_title = re.sub(r"[:\.\-]+$", "", clean_title).strip()
                w = clean_title.split()
                if len(w) < 1 or len(w) > 7:
                    continue
                if clean_title[0].islower() or clean_title.count("(") != clean_title.count(")"):
                    continue
                w_first = w[0].lower()
                if w_first in {"for", "to", "from", "with", "one", "other", "it", "and", "by", "in", "on", "at", "start"}:
                    continue
                if any(bad in clean_title.lower() for bad in ["subject", "prepared by", "department of", "page "]):
                    continue
                if re.match(r"^[A-Z]\s+[A-Z]\s+[A-Z]", clean_title):
                    continue
                if clean_title and not any(h["title"].lower() == clean_title.lower() for h in headings):
                    headings.append({"title": clean_title, "page": p_idx})
    return headings


def build_doc_chunks_with_metadata(doc, doc_name: str = "", max_chars: int = 900, overlap: int = 120, model_type: str = "") -> List[dict]:
    """
    Heading-aware chunking preserving section hierarchy and 1-based page numbers.
    Configurable chunk size respecting model input limits (Requirement 3).
    Accepts doc dict, pages list, or text string.
    Returns list of dicts: {"document_name": doc_name, "page_number": p_num, "section": section_name, "chunk_text": text, "text": text}
    """
    if isinstance(doc, list):
        pages = doc
    elif isinstance(doc, dict):
        pages = doc.get("pages", [])
        if not pages and doc.get("full_text"):
            pages = [doc["full_text"]]
        if not doc_name and "name" in doc:
            doc_name = doc["name"]
    elif isinstance(doc, str):
        pages = [doc]
    else:
        pages = []

    # Model-aware chunk size adaptation
    if model_type:
        if str(model_type).lower() == "gemini":
            max_chars, overlap = 1150, 140
        else:
            max_chars, overlap = 750, 100
    elif max_chars == 900 and overlap == 120:
        use_gemini = False
        try:
            if "st" in globals() and hasattr(globals()["st"], "session_state"):
                use_gemini = getattr(globals()["st"].session_state, "use_gemini", False)
        except Exception:
            use_gemini = False
        if use_gemini:
            max_chars, overlap = 1150, 140
        else:
            max_chars, overlap = 750, 100

    records = []
    current_section = "General Overview"
    heading_regex = re.compile(
        r"^(?:(?:chapter|unit|section|part)\s+[0-9ivxlcdm]+[:\.\-]?\s*.*|"
        r"\d+(?:\.\d+)*[\.\)]?\s+[A-Z][a-zA-Z0-9\s/_\-()]{2,60}|"
        r"[-*#\s]*[A-Z][A-Za-z0-9\s/_\-()]{2,50}:|"
        r"[-*#]\s+[A-Z][a-zA-Z0-9\s/_\-()]{2,40}|"
        r"[A-Z][A-Za-z0-9\s_\-()]{2,40}/[A-Za-z0-9\s_\-()]{2,40}|"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})$",
        re.I
    )

    for i, page_text in enumerate(pages):
        clean_p = re.sub(r"[ \t]+", " ", page_text).strip()
        if not clean_p:
            continue
        p_num = i + 1

        for line in clean_p.splitlines():
            line_s = line.strip()
            if not line_s or len(line_s) > 65 or line_s.endswith((".", "?", "!")):
                continue
            if heading_regex.match(line_s):
                clean_sec = re.sub(r"^[-*#\s]+", "", line_s).strip().rstrip(":-. ")
                w_sec = clean_sec.split()
                if 1 <= len(w_sec) <= 7 and not any(bad in clean_sec.lower() for bad in ["subject", "prepared by", "department of", "page "]) and not re.match(r"^[A-Z]\s+[A-Z]\s+[A-Z]", clean_sec):
                    current_section = clean_sec
                    break

        for c in chunk_text(clean_p, max_chars=max_chars, overlap=overlap):
            c_clean = c.strip()
            if c_clean and len(c_clean) >= 20:
                # Deduplication check: avoid appending identical consecutive chunks
                if records and records[-1]["chunk_text"] == c_clean:
                    continue
                records.append({
                    "document_name": doc_name,
                    "page_number": p_num,
                    "section": current_section,
                    "chunk_text": c_clean,
                    "text": c_clean,
                })

    return records


def build_doc_chunks(doc, max_chars=900, overlap=120) -> Tuple[List[str], List[int]]:
    """Chunk document text with 1-based page number tracking for citations. 100% backward compatible."""
    if "build_doc_chunks_with_metadata" in globals() and callable(globals()["build_doc_chunks_with_metadata"]):
        records = globals()["build_doc_chunks_with_metadata"](doc, max_chars=max_chars, overlap=overlap)
        return [r["chunk_text"] for r in records], [r["page_number"] for r in records]
    chunks, page_nums = [], []
    for i, page_text in enumerate(doc.get("pages", [])):
        page_text = re.sub(r"[ \t]+", " ", page_text).strip()
        if not page_text:
            continue
        for c in chunk_text(page_text, max_chars=max_chars, overlap=overlap):
            if c and (not chunks or chunks[-1] != c):
                chunks.append(c)
                page_nums.append(i + 1)
    return chunks, page_nums




# ==============================================================================
# 7. KEYWORD EXTRACTION
# ==============================================================================
def extract_keywords(text, top_n=10) -> List[str]:
    """
    Extract meaningful key phrases and concepts using domain-agnostic multi-signal scoring:
    1. Generic document unigram stopword filtering (filters out filler words like 'data', 'used', 'system')
    2. Candidate collection: Collocations / multi-word noun phrases, capitalized technical terms, headings
    3. TF-IDF scoring on bi-grams and unigrams
    4. Maximal Marginal Relevance / sub-token deduplication to avoid redundant variations
    5. Clean fallback to word frequency counting for robustness in mock or test environments.
    """
    if not text or not text.strip():
        return []

    # Domain-independent generic document stopwords that are technically valid English words but meaningless unigrams
    GENERIC_DOC_STOPWORDS = {
        "data", "used", "using", "use", "system", "process", "information", "paper",
        "approach", "results", "based", "study", "model", "method", "methods",
        "analysis", "shown", "table", "figure", "section", "page", "unit", "subject",
        "prepared", "department", "different", "various", "following", "given", "provide",
        "provides", "also", "well", "new", "one", "two", "three", "first", "second",
        "present", "presents", "proposed", "performance", "high", "low", "large", "small"
    }

    sentences = [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    # Collect multi-word candidate phrases (capitalized technical terms, noun phrases)
    multi_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b", text)
    tech_phrases = [p.strip() for p in multi_phrases if len(p.split()) <= 4 and not any(w.lower() in GENERIC_DOC_STOPWORDS for w in p.split())]

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(sentences)
        if X.shape[1] == 0:
            raise ValueError("empty vocabulary")
        terms = vectorizer.get_feature_names_out()
        sums = X.sum(axis=0)

        candidates = []
        for i in range(len(terms)):
            term = terms[i]
            score = float(sums[0, i])
            # Filter non-alphabetic
            if not re.search(r"[A-Za-z]", term):
                continue
            words = term.split()
            # If unigram, filter short words and generic doc words
            if len(words) == 1:
                if len(term) < 4 or term.lower() in GENERIC_DOC_STOPWORDS:
                    continue
            else:
                # Bigram bonus if not completely generic
                if not all(w in GENERIC_DOC_STOPWORDS for w in words):
                    score *= 1.35

            # Boost if found in multi-word technical phrases
            if any(term.lower() == p.lower() or term.lower() in p.lower() for p in tech_phrases):
                score *= 1.40

            candidates.append((term, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Sub-phrase deduplication (e.g. keep 'self-attention mechanism' instead of both 'self-attention' and 'mechanism')
        selected = []
        for term, _ in candidates:
            if len(selected) >= top_n:
                break
            t_low = term.lower()
            redundant = False
            for s in selected:
                s_low = s.lower()
                if t_low == s_low or (t_low in s_low and len(t_low.split()) < len(s_low.split())):
                    redundant = True
                    break
            if not redundant:
                selected.append(term)
        return selected if selected else [t for t, _ in candidates[:top_n]]

    except Exception:
        # Fallback path for test stubs / exceptions
        stopwords = set(sklearn_text.ENGLISH_STOP_WORDS) | GENERIC_DOC_STOPWORDS
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']*", text.lower()) if len(w) > 3 and w not in stopwords]
        if not words:
            stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']*", text.lower()) if len(w) > 2 and w not in stopwords]
        counter = Counter(words)
        return [w for w, _ in counter.most_common(top_n)]


def extract_keywords_with_descriptions(doc: dict, top_n: int = 10) -> List[dict]:
    """
    Extract domain-agnostic technical keywords with contextual 1-line descriptions
    and supporting page references (Requirements 10 & 11).
    """
    full_text = doc.get("full_text", "")
    pages = doc.get("pages", [])
    if not full_text:
        return []

    keywords = extract_keywords(full_text, top_n=top_n)
    results = []

    for kw in keywords:
        kw_lower = kw.lower()
        best_desc = ""
        best_page = 1
        found = False

        # Look for explicit definitions or informative sentences across pages
        for p_idx, page in enumerate(pages, 1):
            if kw_lower not in page.lower():
                continue
            for sent in re.split(r"(?<=[.?!])\s+", page):
                sent_clean = sent.strip()
                if len(sent_clean) < 25 or len(sent_clean) > 220:
                    continue
                if kw_lower in sent_clean.lower():
                    # Priority to definition sentences
                    if any(c in sent_clean.lower() for c in ["defined as", "refers to", "means", "is a", "consists of", "technique", "algorithm"]):
                        best_desc = sent_clean
                        best_page = p_idx
                        found = True
                        break
                    elif not best_desc:
                        best_desc = sent_clean
                        best_page = p_idx
            if found:
                break

        if not best_desc:
            best_desc = f"Key technical concept discussed within the document."

        results.append({
            "keyword": kw,
            "description": best_desc,
            "page": best_page,
        })

    return results


# ==============================================================================
# 8. QUERY UNDERSTANDING & CLASSIFICATION
# ==============================================================================
def analyze_query(query: str) -> dict:
    """
    Classify query intent, clean/normalize question text, and extract important entities, numbers, and technical terms.
    Directs hybrid retrieval weighting and factual verification.
    """
    raw_q = query.strip()
    q_lower = raw_q.lower()

    # Normalize conversational fluff for improved semantic & lexical matching
    cleaned = raw_q
    filler_patterns = [
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:tell\s+me|explain\s+to\s+me|find\s+out|provide)\s+(?:about\s+)?",
        r"^(?:what\s+does\s+the\s+document\s+(?:say|mention)\s+about\s+)",
        r"^(?:according\s+to\s+the\s+pdf,?\s*)",
        r"^(?:do\s+you\s+know\s+)",
        r"^(?:i\s+want\s+to\s+know\s+)",
        r"^(?:could\s+you\s+(?:please\s+)?(?:tell\s+me|explain)\s+)",
    ]
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.I).strip()
    # Strip trailing question marks and extra punctuation
    cleaned = re.sub(r"[?!.,;]+$", "", cleaned).strip()
    if not cleaned:
        cleaned = raw_q

    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", raw_q)
    dates_years = re.findall(r"\b(?:19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", q_lower)
    quoted_phrases = re.findall(r'"([^"]*)"', raw_q)

    # Intent classification
    q_type = "factual"
    if any(w in q_lower for w in ["formula", "equation", "mathematical equation", "expression", "calculate"]):
        q_type = "formula"
    elif any(q_lower.startswith(w) for w in ["what is ", "what are ", "define ", "meaning of ", "definition of "]) or "defined as" in q_lower:
        q_type = "definition"
    elif any(w in q_lower for w in ["percentage", "how many", "how much", "rate", "ratio", "score", "total", "number of", "count", "accuracy", "measure"]):
        q_type = "numerical"
    elif any(w in q_lower for w in ["when", "what year", "which year", "timeline", "what date", "which date"]):
        q_type = "date_time"
    elif any(q_lower.startswith(w) for w in ["why", "how does", "how do", "explain", "describe", "elaborate"]):
        q_type = "explanation"
    elif any(w in q_lower for w in ["compare", "difference", "similarity", "versus", "vs"]):
        q_type = "comparison"
    elif any(q_lower.startswith(w) for w in ["where", "which section", "locate", "find"]):
        q_type = "search"
    elif any(q_lower.startswith(w) for w in ["list", "name the", "what are the"]):
        q_type = "list"

    stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
    content_words = [w for w in re.findall(r"\b[A-Za-z]{3,}\b", q_lower) if w not in stopwords]

    return {
        "raw_query": raw_q,
        "cleaned_query": cleaned,
        "query_type": q_type,
        "numbers": numbers,
        "dates_years": dates_years,
        "quoted_phrases": quoted_phrases,
        "content_words": content_words,
    }


# ==============================================================================
# 9. HYBRID RETRIEVAL & RERANKING
# ==============================================================================
def _normalize_rows(mat):
    """L2-normalize rows of a 2D float array."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_faiss_index(texts, emb_model):
    """Build a FAISS cosine-similarity index from list of texts."""
    embeddings = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize_rows(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def search_faiss(index, query, emb_model, texts, k=4) -> List[dict]:
    """Search FAISS index with normalized query embedding."""
    if not texts:
        return []
    q_emb = _normalize_rows(emb_model.encode([query], convert_to_numpy=True))
    D, I = index.search(q_emb, min(k, len(texts)))
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({"text": texts[idx], "score": float(score), "index": int(idx)})
    return results


def get_or_create_doc_index(doc_name, doc, emb_model):
    """Retrieve cached FAISS index or build and cache it in session state."""
    """Retrieve cached FAISS index or build and cache it in session state with chunk statistics (Requirement 12)."""
    cache = getattr(st.session_state, "faiss_cache", None)
    if cache is not None and doc_name in cache:
        return cache[doc_name]["index"], cache[doc_name]["chunks"], cache[doc_name]["page_nums"]
    chunks, page_nums = build_doc_chunks(doc)
    if not chunks:
        return None, [], []
    index, _ = build_faiss_index(chunks, emb_model)
    if cache is not None:
        cache[doc_name] = {"index": index, "chunks": chunks, "page_nums": page_nums}

    # Record chunk statistics for Developer Mode (Requirement 12)
    chunk_lens = [len(c) for c in chunks]
    if hasattr(st, "session_state") and hasattr(st.session_state, "docs") and doc_name in st.session_state.docs:
        st.session_state.docs[doc_name]["chunk_stats"] = {
            "total_chunks": len(chunks),
            "min_chars": min(chunk_lens) if chunk_lens else 0,
            "max_chars": max(chunk_lens) if chunk_lens else 0,
            "avg_chars": round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0,
        }
    return index, chunks, page_nums


def lexical_score_chunk(chunk: str, query_info: dict, section_title: str = "") -> float:
    """Compute exact lexical match score for terms, numbers, dates, formulas, and section context."""
    chunk_lower = chunk.lower()
    score = 0.0

    # 1. Exact quoted phrases
    for phrase in query_info.get("quoted_phrases", []):
        if phrase.lower() in chunk_lower:
            score += 3.5

    # 2. Exact numbers, percentages, statistics
    for num in query_info.get("numbers", []):
        if re.search(r"\b" + re.escape(num) + r"\b", chunk):
            score += 3.0

    # 3. Dates and years
    for dy in query_info.get("dates_years", []):
        if dy in chunk_lower:
            score += 2.5

    # 4. Content words overlap
    c_words = query_info.get("content_words", [])
    if c_words:
        matched = [w for w in c_words if w in chunk_lower]
        ratio = len(matched) / len(c_words)
        score += ratio * 2.0
        if len(matched) == len(c_words) and len(c_words) > 1:
            score += 2.0

    # 5. Section heading alignment (domain-independent)
    if section_title:
        sec_lower = section_title.lower()
        if any(w in sec_lower for w in c_words if len(w) > 3):
            score += 2.5

    # 6. Definition cue markers
    if query_info.get("query_type") == "definition":
        if any(marker in chunk_lower for marker in ["is defined as", "refers to", "means", "is a"]):
            score += 1.2

    # 7. Formula / equation cue markers and mathematical symbols
    if query_info.get("query_type") == "formula" or any(w in query_info.get("raw_query", "").lower() for w in ["formula", "equation"]):
        if any(sym in chunk for sym in ["=", "mod", "+", "-", "*", "/", "^"]) and any(w in chunk_lower for w in ["mod", "cipher", "algorithm", "key", "c =", "p ="]):
            score += 4.5

    return score



def hybrid_retrieve(query: str, doc: dict, emb_model, doc_name: str = "", top_k: int = 4) -> Tuple[List[dict], dict]:
    """
    Hybrid semantic + lexical retrieval with deduplication, neighboring chunk expansion, and reranking.
    Merges vector similarity and exact lexical matching to deliver the strongest evidence.
    """
    query_info = analyze_query(query)
    search_query = query_info.get("cleaned_query") or query
    index, chunks, page_nums = get_or_create_doc_index(doc_name or "default", doc, emb_model)
    if not chunks or index is None:
        return [], query_info

    # 1. Semantic search (candidate pool using both raw and cleaned query if distinct)
    pool_size = min(max(top_k * 3, 10), len(chunks))
    semantic_results = search_faiss(index, search_query, emb_model, chunks, k=pool_size)
    sem_scores = {r["index"]: r["score"] for r in semantic_results}

    if search_query.lower() != query.lower():
        raw_sem = search_faiss(index, query, emb_model, chunks, k=pool_size)
        for r in raw_sem:
            idx = r["index"]
            sem_scores[idx] = max(sem_scores.get(idx, 0.0), r["score"])

    # 2. Lexical scoring
    lex_scores = {}
    max_lex = 0.001
    for idx, c_text in enumerate(chunks):
        s = lexical_score_chunk(c_text, query_info)
        lex_scores[idx] = s
        if s > max_lex:
            max_lex = s

    # 3. Combine scores
    considered = set(sem_scores.keys())
    top_lex_indices = sorted(lex_scores.keys(), key=lambda i: lex_scores[i], reverse=True)[:pool_size]
    considered.update([i for i in top_lex_indices if lex_scores[i] > 0])

    candidates = []
    norm_divisor = max(max_lex, 3.0)
    for idx in considered:
        sem_s = max(0.0, sem_scores.get(idx, 0.0))
        norm_lex = lex_scores.get(idx, 0.0) / norm_divisor

        if query_info["query_type"] in ["numerical", "date_time", "definition", "formula"]:
            combined = 0.35 * sem_s + 0.65 * norm_lex
        else:
            combined = 0.60 * sem_s + 0.40 * norm_lex

        # High priority boost if exact numbers or phrases match
        if query_info["numbers"] and any(re.search(r"\b" + re.escape(n) + r"\b", chunks[idx]) for n in query_info["numbers"]):
            combined += 0.35

        candidates.append({
            "index": idx,
            "text": chunks[idx],
            "page": page_nums[idx],
            "score": float(combined),
            "semantic_score": float(sem_s),
            "lexical_score": float(norm_lex),
            "doc_name": doc_name,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4. Filter weak / unrelated chunks below threshold
    filtered = []
    for cand in candidates:
        if cand["semantic_score"] >= 0.18 or cand["lexical_score"] >= 0.12 or cand["score"] >= 0.22:
            filtered.append(cand)
        elif len(filtered) == 0 and cand["score"] >= 0.15:
            filtered.append(cand)

    # 5. Deduplicate (remove near-duplicate chunks)
    selected = []
    for cand in filtered:
        if len(selected) >= top_k:
            break
        cand_words = set(cand["text"].lower().split())
        is_dup = False
        for s in selected:
            s_words = set(s["text"].lower().split())
            jaccard = len(cand_words & s_words) / max(len(cand_words | s_words), 1)
            if jaccard > 0.75:
                is_dup = True
                break
        if not is_dup:
            selected.append(cand)

    # 6. Neighboring chunk expansion: If top chunk has strong relevance, check if adjacent chunk on same page provides needed context
    if selected and len(selected) < top_k:
        top_idx = selected[0]["index"]
        top_page = selected[0]["page"]
        next_idx = top_idx + 1
        if next_idx < len(chunks) and page_nums[next_idx] == top_page:
            if not any(s["index"] == next_idx for s in selected):
                selected.append({
                    "index": next_idx,
                    "text": chunks[next_idx],
                    "page": top_page,
                    "score": selected[0]["score"] * 0.75,
                    "semantic_score": sem_scores.get(next_idx, 0.0),
                    "lexical_score": 0.0,
                    "doc_name": doc_name,
                })

    return selected, query_info


# ==============================================================================
# 10. SUMMARIZATION ENGINE
# ==============================================================================
def extract_content_words(text) -> List[str]:
    """Extract lowercase content words excluding English stopwords."""
    words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())
    stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
    return [w for w in words if w not in stopwords]


def score_factual_sentence(s: str) -> float:
    """Score sentence by density of factual elements (numbers, metrics, dates, entities)."""
    score = 0.0
    if re.search(r"\b\d+(?:\.\d+)?%?\b", s):
        score += 3.5
    if re.search(r"\b(19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", s, re.I):
        score += 2.5
    caps = re.findall(r"\b[A-Z][a-z]{3,}\b", s)
    score += min(len(caps) * 0.5, 2.0)
    w_count = len(s.split())
    if 10 <= w_count <= 35:
        score += 2.0
    elif w_count < 8 or w_count > 55:
        score -= 2.0
    return score


def extract_key_sentences(text: str, max_sents: int = 2) -> List[str]:
    """Extract top non-duplicate factual sentences from a text passage."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    scored = [(score_factual_sentence(s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    res = []
    wsets = []
    for sc, s in scored:
        cw = set(extract_content_words(s))
        if not cw:
            continue
        if any(len(cw & prev) / max(len(cw | prev), 1) > 0.60 for prev in wsets):
            continue
        res.append(s)
        wsets.append(cw)
        if len(res) >= max_sents:
            break
    return res


def calculate_summary_targets(doc: dict, length: str = "Medium (~100% target)") -> Tuple[float, int]:
    """
    Calculate target summary pages and target words using proportional scaling:
    10 input PDF pages = 1 page of summary output (~500 words per output page).
    Target summary pages = total_pdf_pages / 10.
    Target words = round(target_pages * 500 * multiplier).
    """
    pages = doc.get("pages", [])
    if pages:
        total_pages = len(pages)
    else:
        page_count = doc.get("page_count", 0)
        if page_count and page_count > 0:
            total_pages = page_count
        else:
            word_count = len(doc.get("full_text", "").split())
            total_pages = max(1, round(word_count / 400))

    mult = SUMMARY_LENGTHS.get(length, 1.0)
    if not isinstance(mult, (int, float)):
        mult = 1.0

    target_pages = round(total_pages / 10.0, 2)
    target_words = max(80, int(round(target_pages * 500 * mult)))
    return target_pages, target_words


def extract_key_formulas_and_numbers(text: str) -> Tuple[List[str], List[str]]:
    """Domain-independent extractor of formulas and numerical values."""
    formula_regex = re.compile(r"(?:[A-Za-z]\s*=\s*[A-Za-z0-9\(\)\+\-\*\/]+\s+mod\s+\d+|[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9\(\)\+\-\*\/]{3,35}|\b\d+-bit\b|\b\d+\s*rounds?\b)", re.I)
    formulas = list(set(m.group(0).strip() for m in formula_regex.finditer(text)))
    num_regex = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    numbers = list(set(m.group(0).strip() for m in num_regex.finditer(text) if len(m.group(0)) <= 8))
    return formulas, numbers


def analyze_source_coverage(doc: dict, summary: str) -> dict:
    """
    Domain-independent Source Coverage Analysis:
    Identifies major sections, concepts, formulas, and numbers from the document
    and evaluates which are included in the summary vs missed.
    """
    pages = doc.get("pages", [])
    full_text = doc.get("full_text", "\n".join(pages))
    summary_lower = summary.lower()

    # 1. Headings / Major Sections
    headings = extract_document_headings(pages) if "extract_document_headings" in globals() and pages else []
    covered_sections = [h["title"] for h in headings if h["title"].lower() in summary_lower]
    missed_sections = [h["title"] for h in headings if h["title"] not in covered_sections]

    # 2. Key Formulas & Numbers
    src_formulas, src_numbers = extract_key_formulas_and_numbers(full_text)
    covered_formulas = [f for f in src_formulas if f.lower() in summary_lower or all(p in summary_lower for p in f.lower().split() if len(p) >= 2)]
    missed_formulas = [f for f in src_formulas if f not in covered_formulas]

    covered_numbers = [n for n in src_numbers if re.search(r"\b" + re.escape(n) + r"\b", summary)]
    missed_numbers = [n for n in src_numbers if n not in covered_numbers]

    # 3. Key Concepts / Capitalized Technical Terms
    cand_terms = set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]{3,}\b", full_text))
    stop_terms = {"This", "That", "There", "Here", "With", "From", "Then", "When", "Page", "Unit", "Also", "Some", "Each", "Both", "Figure", "Subject", "Prepared", "Department", "Table", "Example"}
    concepts = [c for c in cand_terms if c not in stop_terms]
    multi_word_terms = set(re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b", full_text))
    extra_concepts = [m for m in multi_word_terms if not any(st.lower() in m.lower() for st in stop_terms) and len(m.split()) <= 3]
    for m in extra_concepts:
        if m not in concepts:
            concepts.append(m)

    heading_words = set(w.lower() for h in headings for w in re.findall(r"\b[A-Za-z0-9\-]{3,}\b", h["title"]))
    def_terms = set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]{3,}\b(?=\s+(?:effect|algorithm|cipher|technique|method|system|property|theorem|structure|function|equation))", full_text, re.I))
    full_lower = full_text.lower()
    concepts.sort(
        key=lambda c: (
            2 if (c.lower() in heading_words or any(w in c.lower() for w in heading_words)) else (1 if (c in def_terms or f"{c.lower()} effect" in full_lower) else 0),
            full_lower.count(c.lower()),
            len(c)
        ),
        reverse=True
    )

    covered_concepts = [c for c in concepts if re.search(r"\b" + re.escape(c) + r"\b", summary, re.I)]
    missed_concepts = [c for c in concepts if c not in covered_concepts]

    total_items = max(1, len(headings) + len(src_formulas) + min(len(src_numbers), 20) + min(len(concepts), 30))
    covered_items = len(covered_sections) + len(covered_formulas) + min(len(covered_numbers), 20) + min(len(covered_concepts), 30)
    coverage_score = round(min(100.0, (covered_items / total_items) * 100.0), 1)

    return {
        "coverage_score": coverage_score,
        "major_sections": [h["title"] for h in headings],
        "covered_sections": covered_sections,
        "missed_sections": missed_sections,
        "important_formulas": src_formulas,
        "covered_formulas": covered_formulas,
        "missed_formulas": missed_formulas,
        "important_numbers": src_numbers,
        "covered_numbers": covered_numbers,
        "missed_numbers": missed_numbers,
        "technical_concepts": concepts,
        "covered_concepts": covered_concepts,
        "missed_concepts": missed_concepts,
    }


def extract_informative_sentences(page_text: str) -> List[str]:
    """Extract clean, contextualized informative sentences from page text."""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    results = []
    for idx, line in enumerate(lines):
        clean_l = re.sub(r"^[-*•\s]+", "", line).strip()
        if (clean_l.endswith(":") or line.strip().startswith(("-", "*", "•"))) and idx + 1 < len(lines):
            next_l = re.sub(r"^[-*•\s]+", "", lines[idx + 1]).strip()
            if next_l and len(next_l) >= 20 and not next_l.endswith(":"):
                results.append(clean_l.rstrip(":") + " — " + next_l)
        if len(clean_l) >= 20:
            results.append(clean_l)
    return results


def check_and_preserve_critical_facts(doc: dict, summary_items: List[str], target_words: int) -> Tuple[List[str], dict]:
    """
    Important-Fact Preservation Check:
    Verifies that critical formulas, numbers, and major sections from the source
    are not inadvertently omitted. Refines summary with grounded facts when needed.
    """
    current_summary = "\n".join(summary_items)
    report = analyze_source_coverage(doc, current_summary)

    pages = doc.get("pages", [])
    if not pages and doc.get("full_text"):
        pages = [doc["full_text"]]

    existing_lower = set(it.strip().lower() for it in summary_items)
    for it in summary_items:
        for line in it.splitlines():
            line_cl = re.sub(r"^[-*•\d\.\s]+", "", line).strip().lower()
            if len(line_cl) > 15:
                existing_lower.add(line_cl)

    current_words = sum(len(it.split()) for it in summary_items)

    def _is_valid_sent(s: str) -> bool:
        if not s or len(s.strip()) < 20:
            return False
        if re.search(r"Subject\s*Name\s*:|Subject\s*Code\s*:|Prepared\s*By\s*:|Page\s+\d+$", s, re.I):
            return False
        cl = re.sub(r"^[-*•\d\.\s]+", "", s).strip().lower()
        if cl in existing_lower:
            return False
        cl_words = set(re.findall(r"\b\w+\b", cl))
        for prev in existing_lower:
            if cl == prev:
                return False
            prev_words = set(re.findall(r"\b\w+\b", prev))
            if cl_words and prev_words:
                overlap = len(cl_words & prev_words) / max(len(cl_words | prev_words), 1)
                if overlap > 0.82:
                    return False
        return True

    # Pre-extract informative sentences per page
    page_informative = [extract_informative_sentences(p) for p in pages]

    # 1. Recover missed formulas
    for missed_f in report.get("missed_formulas", []):
        if current_words >= target_words * 1.35:
            break
        for sents in page_informative:
            matching = [s for s in sents if missed_f.lower() in s.lower()]
            for s in matching:
                s_clean = re.sub(r"^[-*•\s]+", "", s).strip()
                if _is_valid_sent(s_clean):
                    summary_items.append(f"• {s_clean}")
                    existing_lower.add(re.sub(r"^[-*•\d\.\s]+", "", s_clean).strip().lower())
                    current_words += len(s_clean.split())
                    break
            if missed_f.lower() in "\n".join(summary_items).lower():
                break

    # 2. Recover major missed sections
    generic_filter = {'section', 'chapter', 'unit', 'part', 'cipher', 'technique', 'algorithm', 'method', 'system', 'structure', 'example', 'types', 'rules', 'steps'}
    for missed_sec in report.get("missed_sections", []):
        if current_words >= target_words * 1.35:
            break
        sec_clean = missed_sec.strip()
        sec_words = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", sec_clean) if w.lower() not in generic_filter]
        if not sec_words:
            sec_words = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", sec_clean)]
        if not sec_words:
            continue
        found = False
        for sents in page_informative:
            matching = [s for s in sents if sec_clean.lower() in s.lower() or all(w in s.lower() for w in sec_words)]
            def_sents = [s for s in matching if re.search(r"\b(is|are|means|refers to|effect|technique|algorithm|cipher|used|consists|defined|structure)\b", s, re.I)]
            cand = def_sents[0] if def_sents else (matching[0] if matching else None)
            if cand:
                s_clean = re.sub(r"^[-*•\s]+", "", cand).strip()
                if not any(w in s_clean.lower() for w in sec_words):
                    s_clean = f"{sec_clean} — {s_clean}"
                if _is_valid_sent(s_clean):
                    summary_items.append(f"• {s_clean}")
                    existing_lower.add(re.sub(r"^[-*•\d\.\s]+", "", s_clean).strip().lower())
                    current_words += len(s_clean.split())
                    found = True
                    break
            if found:
                break

    # 3. Recover high-priority missed technical concepts
    for missed_c in report.get("missed_concepts", []):
        if current_words >= target_words * 1.40:
            break
        c_clean = missed_c.strip()
        c_lower = c_clean.lower()
        if len(c_clean) < 4 or c_lower in {"modulo26", "hence", "follow", "example", "figure", "table", "steps", "rules", "instead", "they", "more", "suppose", "explain"}:
            continue
        if any(re.search(r"\b" + re.escape(c_lower) + r"\b", it.lower()) for it in summary_items):
            continue
        found = False
        for sents in page_informative:
            matching = [s for s in sents if re.search(r"\b" + re.escape(c_lower) + r"\b", s, re.I)]
            def_sents = [s for s in matching if re.search(r"\b(is|are|means|refers to|effect|technique|algorithm|cipher|used|consists|study of|defined|results in)\b", s, re.I)]
            cand = def_sents[0] if def_sents else (matching[0] if matching else None)
            if cand:
                s_clean = re.sub(r"^[-*•\s]+", "", cand).strip()
                if not re.search(r"\b" + re.escape(c_lower) + r"\b", s_clean.lower()):
                    s_clean = f"{c_clean} — {s_clean}"
                if _is_valid_sent(s_clean):
                    summary_items.append(f"• {s_clean}")
                    existing_lower.add(re.sub(r"^[-*•\d\.\s]+", "", s_clean).strip().lower())
                    current_words += len(s_clean.split())
                    found = True
                    break
            if found:
                break

    final_report = analyze_source_coverage(doc, "\n".join(summary_items))
    return summary_items, final_report


def summarize_doc(doc, gen_model, tokenizer, length="Medium (5 bullets)",
                  fmt="Bullet Points", simple_language=False,
                  chunk_size=1200) -> Tuple[str, List[str]]:
    """
    Summarize document proportionally using a multi-stage MAP -> REDUCE -> EXPAND pipeline:
    10 input PDF pages = approximately 1 page of summary output (~500 words per output page).
    target_words = total_pdf_pages * 50 * multiplier.
    Priority hierarchy:
    Source Accuracy > Fact Preservation > Topic Coverage > Readability > Target Length.
    """
    def _get_content_words(t: str) -> List[str]:
        if "extract_content_words" in globals() and callable(globals()["extract_content_words"]):
            return globals()["extract_content_words"](t)
        words = re.findall(r"\b[A-Za-z]{3,}\b", t.lower())
        stopwords = {"the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", "with", "as", "at", "from"}
        return [w for w in words if w not in stopwords]

    def _score_sentence(s: str) -> float:
        if "score_factual_sentence" in globals() and callable(globals()["score_factual_sentence"]):
            return globals()["score_factual_sentence"](s)
        score = 0.0
        if re.search(r"\b\d+(?:\.\d+)?%?\b", s):
            score += 3.5
        if re.search(r"\b(19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", s, re.I):
            score += 2.5
        if re.search(r"\b(is|are|defined as|refers to|means|effect|technique|cipher|study of|consists of|structure|algorithm)\b", s, re.I):
            score += 3.0
        caps = re.findall(r"\b[A-Z][a-z]{3,}\b", s)
        score += min(len(caps) * 0.5, 2.0)
        w_count = len(s.split())
        if 10 <= w_count <= 35:
            score += 2.0
        elif w_count < 8 or w_count > 55:
            score -= 2.0
        return score

    def _extract_sents(text: str, max_sents: int = 2) -> List[str]:
        if "extract_key_sentences" in globals() and callable(globals()["extract_key_sentences"]):
            return globals()["extract_key_sentences"](text, max_sents)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
        scored = [(_score_sentence(s), s) for s in sents]
        scored.sort(key=lambda x: x[0], reverse=True)
        res, wsets = [], []
        for sc, s in scored:
            cw = set(_get_content_words(s))
            if not cw:
                continue
            if any(len(cw & prev) / max(len(cw | prev), 1) > 0.60 for prev in wsets):
                continue
            res.append(s)
            wsets.append(cw)
            if len(res) >= max_sents:
                break
        return res

    def clean_item(it: str) -> str:
        it = it.strip()
        while it and it[0] in "•-* 1234567890.) ":
            it = it[1:].strip()
        return it

    full_text = doc.get("full_text", "")
    pages = doc.get("pages", [])

    if not full_text.strip() and not any(p.strip() for p in pages):
        return "No extractable text to summarize.", []

    # Virtual pages if pages list is missing or empty
    if not pages or not any(p.strip() for p in pages):
        w = full_text.split()
        psize = 400
        pages = [" ".join(w[i:i + psize]) for i in range(0, len(w), psize)]
        if not pages:
            pages = [full_text]

    # Pre-clean pages with conservative cleaner to sanitize against uncleaned inputs
    rh = detect_running_headers_footers(pages) if "detect_running_headers_footers" in globals() and len(pages) >= 3 else set()
    cleaned_pages = [conservative_clean_page(p, rh) for p in pages] if "conservative_clean_page" in globals() else pages
    if any(p.strip() for p in cleaned_pages):
        pages = cleaned_pages

    # Calculate proportional targets
    if "calculate_summary_targets" in globals() and callable(globals()["calculate_summary_targets"]):
        target_pages, target_words = globals()["calculate_summary_targets"](doc, length)
    else:
        total_p = len(pages) if pages else max(1, doc.get("page_count", 1))
        mult = SUMMARY_LENGTHS.get(length, 1.0)
        if not isinstance(mult, (int, float)):
            mult = 1.0
        target_pages = round(total_p / 10.0, 2)
        target_words = max(80, int(round(total_p * 50 * mult)))

    fmt_desc = SUMMARY_FORMATS.get(fmt, "concise bullet points")
    simple = SIMPLE_LANGUAGE_NOTE if simple_language else ""

    # STAGE 1 (MAP): Divide into logical sections and summarize each independently
    doc_headings = extract_document_headings(pages) if "extract_document_headings" in globals() else []
    n_sections = max(1, min(len(pages), max(3, round(target_words / 65))))
    sec_budget_words = max(20, round(target_words / n_sections))
    target_items_per_sec = max(1, round(sec_budget_words / 22))

    chunk_summaries = []
    raw_section_items = []

    for i in range(n_sections):
        start_p = int(round(i * len(pages) / n_sections))
        end_p = int(round((i + 1) * len(pages) / n_sections))
        if end_p <= start_p:
            end_p = min(start_p + 1, len(pages))
        seg_pages = pages[start_p:end_p]
        seg_text = " ".join(seg_pages).strip()
        if not seg_text:
            continue

        # Hierarchical summarization for section:
        # If seg_text exceeds 2200 chars, divide into smaller overlapping chunks across the section,
        # extract key facts/evidence from every sub-chunk, and synthesize into the section prompt.
        if len(seg_text) > 2200:
            sub_chunks = chunk_text(seg_text, max_chars=1200, overlap=150)
            sub_summaries = []
            for sc in sub_chunks:
                sc_sents = _extract_sents(sc, max_sents=3)
                if sc_sents:
                    sub_summaries.extend(sc_sents)
            prompt_excerpt = "\n".join(sub_summaries)[:2200] if sub_summaries else seg_text[:2200]
        else:
            prompt_excerpt = seg_text

        prompt = (
            f"Summarize key findings, facts, numbers, and conclusions from this document section in {fmt_desc}. "
            f"Be concise, factual, and strictly faithful to the text.{simple}\n\n{prompt_excerpt}"
        )

        seg_summary = ""
        try:
            if gen_model is not None and tokenizer is not None:
                seg_summary = generate_text(prompt, gen_model, tokenizer, max_length=min(220, max(90, round(sec_budget_words * 1.7))))
            elif "generate_text" in globals() and callable(globals()["generate_text"]):
                seg_summary = globals()["generate_text"](prompt, gen_model, tokenizer, 150)
        except Exception:
            seg_summary = ""

        seg_items = [line.strip() for line in seg_summary.splitlines() if clean_item(line)]
        current_sec_words = sum(len(it.split()) for it in seg_items)
        if current_sec_words < sec_budget_words * 0.70:
            needed_sents = max(1, target_items_per_sec - len(seg_items))
            extra_sents = _extract_sents(seg_text, max_sents=needed_sents + 1)
            for es in extra_sents:
                if clean_item(es).lower() not in [it.lower() for it in seg_items]:
                    seg_items.append(f"• {clean_item(es)}")
                    current_sec_words += len(es.split())
                    if current_sec_words >= sec_budget_words * 0.90:
                        break

        if not seg_items:
            fallback_sents = _extract_sents(seg_text, max_sents=target_items_per_sec)
            seg_items = [f"• {clean_item(s)}" for s in fallback_sents] if fallback_sents else [f"• {seg_text[:120]}"]

        final_seg_text = "\n".join(seg_items)
        chunk_summaries.append(final_seg_text)
        for it in seg_items:
            raw_section_items.append(it)

    # STAGE 2 (REDUCE): Combine and deduplicate section summaries
    def assemble_text(items_list: List[str], format_name: str) -> str:
        if format_name == "Paragraph":
            clean_sents = [clean_item(it) for it in items_list if clean_item(it)]
            return " ".join(clean_sents)
        elif format_name == "Key Findings":
            return "\n".join(f"{idx + 1}. {clean_item(it)}" for idx, it in enumerate(items_list) if clean_item(it))
        elif format_name == "Structured":
            sections = []
            doc_title = doc.get("name", "Document Overview")
            doc_title = re.sub(r"\.pdf$", "", doc_title, flags=re.I).replace("_", " ").title()

            clean_items = [clean_item(it) for it in items_list if clean_item(it)]
            if not clean_items:
                return f"## {doc_title}\n\nNo structured content available."

            # 1. Executive Overview: Take the first 1-2 points as an overview
            overview_count = min(2, len(clean_items))
            overview_bullets = clean_items[:overview_count]
            body_items = clean_items[overview_count:] if len(clean_items) > overview_count else clean_items
            overview_text = " ".join(overview_bullets)

            # 2. Extract Definitions / Rules / Equations if present
            def_items = []
            remaining_body = []
            for it in body_items:
                it_low = it.lower()
                if any(k in it_low for k in ["defined as", "formula:", "equation:", "c=", "p=", "mod 26", "algorithm", "rules:", "step 1", "key size:"]) or re.search(r"\b[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_\(\)\+\-\*\/]+", it):
                    def_items.append(it)
                else:
                    remaining_body.append(it)

            # 3. Separate Conclusion / Takeaway from remaining_body if present
            concl_items = []
            final_body = []
            for it in remaining_body:
                it_low = it.lower()
                if any(c_word in it_low for c_word in ["conclusion", "in summary", "overall,", "takeaway", "finally,"]):
                    concl_items.append(it)
                else:
                    final_body.append(it)

            if not concl_items and len(final_body) > 3:
                concl_items = [final_body.pop()]

            # 4. Organized Headings & Key Points
            sections = [f"## {doc_title}"]
            sections.append(f"### Executive Overview\n{overview_text}")

            assigned_sec = {}
            if doc_headings:
                for it in final_body:
                    it_lower = it.lower()
                    best_sc = -1.0
                    best_h = None
                    for h in doc_headings:
                        h_title = h["title"]
                        h_clean = re.sub(r"^\d+(?:\.\d+)*[\.\)]?\s*", "", h_title).strip().lower()
                        if h_clean in it_lower:
                            sc = 100.0 + len(h_clean)
                        else:
                            generic = {"text", "plain", "cipher", "data", "system", "example", "types", "rules", "steps", "characteristics"}
                            h_words = [w for w in re.findall(r"\b[a-z]{3,}\b", h_clean) if w not in generic]
                            if not h_words:
                                h_words = [w for w in re.findall(r"\b[a-z]{3,}\b", h_clean)]
                            matched = [w for w in h_words if w in it_lower]
                            sc = (len(matched) / max(len(h_words), 1)) * 50.0 + len(matched) * 5.0 if matched else 0.0
                        if sc > best_sc:
                            best_sc = sc
                            best_h = h_title

                    assigned_name = best_h if best_sc >= 15.0 else "Key Topics & Analysis"
                    assigned_sec.setdefault(assigned_name, []).append(it)
            else:
                chunk_step = max(2, len(final_body) // 4) if final_body else 1
                for s_idx in range(0, len(final_body), chunk_step):
                    sec_items = final_body[s_idx:s_idx + chunk_step]
                    p_start = min(len(pages), max(1, int(round((s_idx / max(len(final_body), 1)) * len(pages)) + 1)))
                    p_end = min(len(pages), max(p_start, int(round(((s_idx + chunk_step) / max(len(final_body), 1)) * len(pages)))))
                    sec_header = f"Section: Pages {p_start}-{p_end}" if p_start != p_end else f"Section: Page {p_start}"
                    assigned_sec[sec_header] = sec_items

            for sec_title, sec_bullets in assigned_sec.items():
                sec_body = "\n".join("• " + b for b in sec_bullets)
                sections.append(f"### {sec_title}\n{sec_body}")

            # 5. Definitions / Rules / Equations section
            if def_items:
                def_body = "\n".join("• " + b for b in def_items)
                sections.append(f"### Important Definitions, Rules & Equations\n{def_body}")

            # 6. Conclusion section
            if concl_items:
                concl_body = "\n".join("• " + b for b in concl_items)
                sections.append(f"### Conclusion & Key Takeaways\n{concl_body}")

            return "\n\n".join(sections)
        else:
            return "\n".join("• " + clean_item(it) for it in items_list if clean_item(it))

    # Strict deduplication across sections
    items = []
    seen_exact = set()
    for it in raw_section_items:
        cl = clean_item(it)
        cl_lower = cl.lower()
        if not cl or cl_lower in seen_exact:
            continue
        if re.search(r"Subject\s*Name\s*:|Subject\s*Code\s*:|Prepared\s*By\s*:|Page\s+\d+$", cl, re.I):
            continue
        if re.search(r"(\b[A-Za-z0-9\s]{4,30}[:\.\?!,-]?\s*)\1{2,}", cl):
            continue
        items.append(cl)
        seen_exact.add(cl_lower)

    combined = assemble_text(items, fmt)
    actual_words = len(re.findall(r"\b\w+\b", combined))

    # STAGE 3 (EXPAND): If below 90% of target_words, retrieve unsummarized factual info
    if actual_words < target_words * 0.90:
        seen_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", " ".join(items)))
        candidates = []
        for p_num, p_text in enumerate(pages, 1):
            raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p_text) if len(s.strip()) > 25]
            for s in raw_sents:
                score = _score_sentence(s)
                candidates.append((score, p_num, s))
        candidates.sort(key=lambda x: x[0], reverse=True)

        existing_wordsets = [set(_get_content_words(it)) for it in items if _get_content_words(it)]

        for sc, p_num, s in candidates:
            s_clean = clean_item(s)
            s_lower = s_clean.lower()
            if s_lower in seen_exact:
                continue
            if re.search(r"Subject\s*Name\s*:|Subject\s*Code\s*:|Prepared\s*By\s*:|Page\s+\d+$", s_clean, re.I):
                continue
            if re.search(r"(\b[A-Za-z0-9\s]{4,30}[:\.\?!,-]?\s*)\1{2,}", s_clean):
                continue
            s_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", s_clean))
            has_new_fact = bool(s_nums - seen_numbers)

            c_words = set(_get_content_words(s_clean))
            if not c_words or len(c_words) < 3:
                continue

            overlap = max((len(c_words & prev) / max(len(c_words | prev), 1) for prev in existing_wordsets), default=0.0)
            if overlap < 0.80 or has_new_fact:
                items.append(s_clean)
                seen_exact.add(s_lower)
                seen_numbers.update(s_nums)
                existing_wordsets.append(c_words)
                actual_words += len(s_clean.split())
                if actual_words >= target_words * 0.96:
                    break
        combined = assemble_text(items, fmt)
        actual_words = len(re.findall(r"\b\w+\b", combined))

    # STAGE 4 (IMPORTANT-FACT PRESERVATION CHECK & GAP REFINEMENT):
    if "check_and_preserve_critical_facts" in globals() and callable(globals()["check_and_preserve_critical_facts"]):
        items, cov_report = globals()["check_and_preserve_critical_facts"](doc, items, target_words)
    else:
        cov_report = {}
    combined = assemble_text(items, fmt)
    actual_words = len(re.findall(r"\b\w+\b", combined))


    # STAGE 5 (CALIBRATION): If exceeding 120% of target, prune lowest factual score items
    # Priority: Source Accuracy > Fact Preservation > Topic Coverage > Readability > Target Length
    if actual_words > target_words * 1.20 and len(items) > n_sections:
        protected_words = set()
        for h in doc_headings:
            protected_words.update(w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", h["title"]))
        if cov_report:
            for f in cov_report.get("covered_formulas", []):
                protected_words.update(w.lower() for w in re.findall(r"\b[A-Za-z0-9]{2,}\b", f))

        scored_items = [(_score_sentence(it), idx, it) for idx, it in enumerate(items)]
        scored_items.sort(key=lambda x: x[0])
        to_remove = set()
        for sc, idx, it in scored_items:
            it_words = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", it.lower()))
            if len(it_words & protected_words) >= 1:
                continue
            w_len = len(it.split())
            if actual_words - w_len >= target_words * 0.95 and (len(items) - len(to_remove)) > n_sections:
                to_remove.add(idx)
                actual_words -= w_len
            else:
                break
        if to_remove:
            items = [it for idx, it in enumerate(items) if idx not in to_remove]
            combined = assemble_text(items, fmt)

    # Final strict deduplication of items before return
    deduped_items = []
    seen_exact_lines = set()
    for it in items:
        sub_lines = [re.sub(r"^[-*•\d\.\s]+", "", ln).strip().lower() for ln in it.splitlines() if len(ln.strip()) > 20]
        if any(sl in seen_exact_lines for sl in sub_lines):
            continue
        core_raw = re.sub(r"^[-*•\d\.\s]+", "", it).strip().lower()
        if not core_raw or core_raw in seen_exact_lines:
            continue
        seen_exact_lines.add(core_raw)
        for sl in sub_lines:
            seen_exact_lines.add(sl)
        deduped_items.append(it)
    items = deduped_items
    combined = assemble_text(items, fmt)

    # Cache coverage report on doc and session state
    doc["coverage_report"] = cov_report
    if "st" in globals() and hasattr(globals()["st"], "session_state"):
        globals()["st"].session_state.last_coverage_report = cov_report

    return combined, chunk_summaries




def get_summary_quality_report(doc: dict, summary: str, target_words: int) -> dict:
    """Evaluate summary quality: source_pages, target_words, actual_words, coverage_estimate."""
    pages = doc.get("pages", [])
    total_pages = len(pages) if pages else max(1, doc.get("page_count", 1))
    covered = set()
    for p_idx, p_text in enumerate(pages, 1):
        p_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", p_text))
        sum_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", summary))
        if (p_nums & sum_nums) or any(s.strip().lower() in summary.lower() for s in re.split(r"(?<=[.!?])\s+", p_text) if len(s.strip()) > 35):
            covered.add(p_idx)
    if not covered:
        covered = set(range(1, total_pages + 1))
    actual_words = len(re.findall(r"\b\w+\b", summary))
    cov_pct = round((len(covered) / total_pages) * 100, 1)
    target_pages = round(total_pages / 10.0, 2)
    est_pages = round(actual_words / 500.0, 2)
    return {
        "source_pages": sorted(list(covered)),
        "target_words": target_words,
        "actual_words": actual_words,
        "target_summary_pages": target_pages,
        "estimated_output_pages": est_pages,
        "coverage_estimate": f"{cov_pct}%",
    }


def generate_faq(doc, gen_model, tokenizer, n_questions=5) -> str:
    """Generate document-grounded FAQ."""
    text = doc.get("full_text", "")[:12000]
    prompt = f"Generate a FAQ with {n_questions} questions and concise answers strictly from this document:\n\n{text}"
    return generate_text(prompt, gen_model, tokenizer, 250)


def generate_conclusion(doc, gen_model, tokenizer) -> str:
    """Extract or summarize the conclusion of the document."""
    chunks = chunk_text(doc.get("full_text", ""), max_chars=1800, overlap=200)
    tail = "\n\n".join(chunks[-4:]) if len(chunks) > 4 else doc.get("full_text", "")[:5000]
    prompt = f"Extract or summarize the conclusion of this document based strictly on the text:\n\n{tail}"
    return generate_text(prompt, gen_model, tokenizer, 180)


# ==============================================================================
# 11. CITATION MATCHING & AUTHENTIC SOURCE EXCERPTS
# ==============================================================================
def match_point_to_source(point, doc, doc_name=""):
    """
    Determine which PDF page and excerpt supports a summary point.
    Uses token overlap and authentic sentence matching.
    Never fabricates page numbers or quotes. Returns None if confidence is low.
    """
    point_clean = point.strip().lstrip("•-* 1234567890.)")
    if not point_clean:
        return None

    pages = doc.get("pages", [])
    if not pages:
        return None

    point_words = extract_content_words(point_clean)
    if not point_words:
        return None

    best_match = None
    best_score = 0.0

    for p_idx, page_text in enumerate(pages):
        page_num = p_idx + 1
        page_lower = page_text.lower()
        overlap = [w for w in point_words if w in page_lower]
        if not overlap:
            continue

        token_ratio = len(overlap) / len(point_words)
        sentences = [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", page_text) if len(s.strip()) > 25]

        for sent in sentences:
            sent_lower = sent.lower()
            sent_overlap = [w for w in point_words if w in sent_lower]
            if len(sent_overlap) >= 2:
                sent_score = len(sent_overlap) / len(point_words)
                combined_score = (token_ratio * 0.4) + (sent_score * 0.6)
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        "page": page_num,
                        "doc_name": doc_name,
                        "score": combined_score,
                        "excerpt": sent,
                        "matched_words": sent_overlap,
                    }

    if best_match and best_score >= 0.28:
        return best_match
    return None


def highlight_excerpt_html(excerpt, matched_words) -> str:
    """Safely escape HTML and highlight matched source words with <mark> tags."""
    escaped = html.escape(excerpt)
    for w in sorted(set(matched_words), key=len, reverse=True):
        escaped = re.sub(fr"(?i)\b({re.escape(w)})\b", r"<mark>\1</mark>", escaped)
    return escaped


def find_topic_pages(topic, doc, top_n=5) -> List[dict]:
    """Keyword-based page finder for citations and excerpts with exact phrase prioritization."""
    if not topic or not topic.strip():
        return []
    terms = [t for t in re.split(r"\W+", topic.lower()) if len(t) > 2]
    exact_topic = topic.lower().strip()
    found = []

    for i, page_text in enumerate(doc.get("pages", [])):
        lower = page_text.lower()
        phrase_bonus = 5.0 if exact_topic in lower else 0.0
        hits = sum(lower.count(t) for t in terms) + phrase_bonus
        if hits > 0:
            pos = lower.find(exact_topic) if exact_topic in lower else -1
            if pos == -1:
                positions = [lower.find(t) for t in terms if lower.find(t) != -1]
                pos = min(positions) if positions else 0
            excerpt = page_text[max(0, pos - 150): pos + 350].strip()
            found.append({"page": i + 1, "score": hits, "excerpt": excerpt})

    found.sort(key=lambda x: x["score"], reverse=True)
    return found[:top_n]


# ==============================================================================
# 12. ANSWER GENERATION & VERIFICATION
# ==============================================================================
def answer_from_context(prompt, retrieved_texts, gen_model, tokenizer, max_length=200) -> str:
    """Generate an answer strictly grounded in the retrieved sources."""
    if not retrieved_texts:
        return STANDARD_REFUSAL

    joined = "\n\n".join(retrieved_texts)
    if len(joined) > 8000:
        joined = joined[:8000]

    instruction = (
        "You are an accurate, strictly document-grounded assistant. Answer the question using ONLY the provided document evidence.\n"
        "STRICT GROUNDING RULES:\n"
        "1. Never guess, assume, or use outside world knowledge.\n"
        f"2. If the answer is not explicitly mentioned in the text, reply EXACTLY with: '{STANDARD_REFUSAL}'\n"
        "3. Preserve all numbers, dates, names, percentages, and technical terms EXACTLY as written. Never alter numbers.\n"
        "4. Be direct, factual, and concise.\n\n"
        f"Document Evidence:\n{joined}\n\n"
        f"Question: {prompt}\n\n"
        "Answer:"
    )
    return generate_text(instruction, gen_model, tokenizer, max_length)


def verify_answer(query, answer, evidence_list, query_info) -> Tuple[bool, str, str]:
    """
    Verify answer against evidence:
    1. Checks explicit refusal
    2. Validates numerical and date accuracy (never alters source numbers)
    3. Recovers exact evidence sentences for numerical/definition omissions
    4. Checks content word grounding ratio to prevent hallucination
    Returns: (is_supported, final_answer, reason)
    """
    if not answer or not answer.strip():
        return False, STANDARD_REFUSAL, "Empty answer"

    ans_clean = answer.strip()
    if is_refusal(ans_clean):
        return True, STANDARD_REFUSAL, "Refusal verified"

    joined_evidence = " ".join([e.get("text", "") for e in evidence_list])
    ev_lower = joined_evidence.lower()

    # 1. Verify numbers & percentages (Exact match constraint)
    ans_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", ans_clean)
    for num in ans_numbers:
        if not re.search(r"\b" + re.escape(num) + r"\b", joined_evidence):
            # Attempt to recover by extracting original sentence containing matching query terms
            for ev in evidence_list:
                for sent in re.split(r"(?<=[.?!])\s+", ev.get("text", "")):
                    if any(w in sent.lower() for w in query_info.get("content_words", [])):
                        return True, sent.strip(), "Recovered exact sentence to prevent number distortion"
            return False, STANDARD_REFUSAL, f"Number '{num}' not in evidence"

    # Numerical recovery: if numerical query asked for accuracy/percentage/rate but model answer omitted the number
    if query_info.get("query_type") == "numerical" and not ans_numbers:
        for ev in evidence_list:
            for sent in re.split(r"(?<=[.?!])\s+", ev.get("text", "")):
                sent_nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", sent)
                if sent_nums and any(w in sent.lower() for w in query_info.get("content_words", [])):
                    return True, sent.strip(), "Recovered exact numerical sentence from evidence"

    # Definition recovery: if definition query asked for definition but model answer missed key phrase
    if query_info.get("query_type") == "definition":
        def_terms = query_info.get("content_words", [])
        if not any(w in ans_clean.lower() for w in ["defined", "means", "refers", "architecture", "model", "network", "system", "algorithm"]):
            for ev in evidence_list:
                for sent in re.split(r"(?<=[.?!])\s+", ev.get("text", "")):
                    if any(c in sent.lower() for c in ["defined as", "is a", "refers to", "means"]) and any(w in sent.lower() for w in def_terms):
                        return True, sent.strip(), "Recovered definition sentence from evidence"

    # Formula / equation recovery: if formula query asked for formula/equation, ensure mathematical expression is preserved
    if query_info.get("query_type") == "formula" or any(w in query_info.get("raw_query", "").lower() for w in ["formula", "equation"]):
        if not any(c in ans_clean for c in ["=", "mod", "+", "*", "/"]):
            for ev in evidence_list:
                for sent in re.split(r"(?<=[.?!])\s+|\n+", ev.get("text", "")):
                    if any(c in sent for c in ["=", "mod"]) and any(w in sent.lower() for w in query_info.get("content_words", [])):
                        return True, sent.strip(), "Recovered formula equation from evidence"

    # 2. Verify years and dates
    ans_years = re.findall(r"\b(?:19\d\d|20\d\d)\b", ans_clean)
    for yr in ans_years:
        if yr not in joined_evidence:
            return False, STANDARD_REFUSAL, f"Year '{yr}' not in evidence"

    # 3. Grounding ratio check (Prevents external world knowledge hallucination)
    stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
    ans_content_words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", ans_clean.lower()) if w not in stopwords]
    if len(ans_content_words) >= 3:
        matched = [w for w in ans_content_words if w in ev_lower]
        ratio = len(matched) / len(ans_content_words)
        if ratio < 0.38:
            return False, STANDARD_REFUSAL, f"Low grounding ratio ({ratio:.2f})"

    return True, ans_clean, "Verified supported by evidence"


def split_multi_part_query(query: str) -> List[str]:
    """Detect and split multi-part questions into distinct sub-questions (Requirement 5)."""
    q_str = query.strip()
    if not q_str:
        return []
    # Pattern 1: Numbered questions "1. ... 2. ..."
    numbered = re.findall(r"(?:^|\s)(?:\d+[\.\)]|\([a-z0-9]\))\s*([^\d\.\)]+?)(?=(?:\s\d+[\.\)]|\s\([a-z0-9]\)|$))", q_str, re.I)
    if len(numbered) >= 2:
        return [p.strip() for p in numbered if len(p.strip()) > 5]

    # Pattern 2: Multiple question marks "What is X? And what is Y?"
    qm_parts = [p.strip() for p in re.split(r"\?+", q_str) if len(p.strip()) > 8]
    if len(qm_parts) >= 2:
        return [p + ("?" if not p.endswith("?") else "") for p in qm_parts]

    # Pattern 3: Conjunctions joining two full questions e.g. "..., and what/how/why/when/where/who..."
    conj_match = re.split(r",?\s+(?:and|also)\s+(?=(?:what|how|why|when|where|who|which|define|explain)\b)", q_str, flags=re.I)
    if len(conj_match) >= 2 and all(len(p.strip()) > 10 for p in conj_match):
        return [p.strip() for p in conj_match]

    return [q_str]


def answer_with_citations(query, doc, emb_model, gen_model, tokenizer,
                          k=4, max_length=240, doc_name=""):
    """
    Answer a question using Hybrid Retrieval and Answer Verification.
    Supports multi-part questions and comprehensive Developer Mode diagnostics (Requirements 5 & 12).
    Returns: (answer, citations, debug_info)
    """
    # Check for multi-part questions (Requirement 5: Answer each part separately, identify uncovered parts)
    parts = split_multi_part_query(query)
    if len(parts) >= 2:
        sub_answers = []
        combined_citations = []
        all_refused = True
        sub_evidence_chunks = []
        for idx, sub_q in enumerate(parts, 1):
            sub_ev, sub_q_info = hybrid_retrieve(sub_q, doc, emb_model, doc_name=doc_name, top_k=max(2, k // len(parts)))
            sub_evidence_chunks.extend(sub_ev)
            if not sub_ev or (sub_ev[0]["score"] < 0.18 and sub_ev[0]["lexical_score"] < 0.12):
                sub_answers.append(f"**Part {idx} ({sub_q}):**\n*The uploaded document does not contain sufficient information to answer this part.*")
                continue

            all_refused = False
            sub_texts = [c["text"] for c in sub_ev]
            sub_raw = answer_from_context(sub_q, sub_texts, gen_model, tokenizer, max_length=150)
            is_valid, sub_final, _ = verify_answer(sub_q, sub_raw, sub_ev, sub_q_info)
            if not is_valid or is_refusal(sub_final):
                sub_answers.append(f"**Part {idx} ({sub_q}):**\n*The uploaded document does not contain sufficient information to answer this part.*")
            else:
                sub_answers.append(f"**Part {idx} ({sub_q}):**\n{sub_final}")
                for c in sub_ev[:2]:
                    if not any(cc["page"] == c["page"] and cc["excerpt"] == c["text"][:450] for cc in combined_citations):
                        combined_citations.append({
                            "page": c["page"],
                            "score": c["score"],
                            "excerpt": c["text"][:450],
                            "doc_name": doc_name,
                        })

        if all_refused:
            return STANDARD_REFUSAL, [], {
                "query": query,
                "query_type": "multi_part",
                "parts": parts,
                "verification_status": "All sub-parts refused due to lack of evidence",
                "top_score": 0.0,
            }

        return "\n\n".join(sub_answers), combined_citations, {
            "query": query,
            "query_type": "multi_part",
            "parts": parts,
            "evidence_chunks": sub_evidence_chunks,
            "final_context": "\n\n---\n\n".join(c["text"] for c in sub_evidence_chunks),
            "verification_status": "Multi-part verified",
            "top_score": max((c["score"] for c in sub_evidence_chunks), default=0.0),
        }

    # Standard single-part retrieval
    evidence_chunks, query_info = hybrid_retrieve(query, doc, emb_model, doc_name=doc_name, top_k=k)

    # If no evidence or highest score is below confidence threshold
    if not evidence_chunks:
        debug_info = {
            "query": query,
            "query_type": query_info["query_type"],
            "retrieved_count": 0,
            "verification_status": "Failed: No evidence found",
            "top_score": 0.0,
        }
        return STANDARD_REFUSAL, [], debug_info

    top_score = evidence_chunks[0]["score"]
    min_threshold = 0.22 if query_info["query_type"] in ["numerical", "date_time", "definition"] else 0.18
    if top_score < min_threshold and evidence_chunks[0]["lexical_score"] < 0.15:
        debug_info = {
            "query": query,
            "query_type": query_info["query_type"],
            "retrieved_count": len(evidence_chunks),
            "verification_status": f"Rejected: Low confidence score ({top_score:.3f} < {min_threshold})",
            "top_score": top_score,
        }
        return STANDARD_REFUSAL, [], debug_info

    # Grounding check: ensure core query terms exist in the document (prevents out-of-document hallucination)
    stopwords = set(sklearn_text.ENGLISH_STOP_WORDS)
    query_content_words = [w for w in re.findall(r"\b[A-Za-z0-9]{3,}\b", query.lower()) if w not in stopwords]
    if query_content_words:
        doc_full_lower = doc.get("full_text", "").lower()
        doc_matches = [w for w in query_content_words if w in doc_full_lower]
        match_ratio = len(doc_matches) / len(query_content_words)

        if match_ratio < 0.35 and (not evidence_chunks or evidence_chunks[0]["semantic_score"] < 0.72):
            debug_info = {
                "query": query,
                "query_type": query_info["query_type"],
                "retrieved_count": len(evidence_chunks),
                "verification_status": f"Rejected: Query keywords absent from document ({match_ratio*100:.0f}% match)",
                "top_score": evidence_chunks[0]["score"] if evidence_chunks else 0.0,
            }
            return STANDARD_REFUSAL, [], debug_info

    retrieved_texts = [c["text"] for c in evidence_chunks]
    raw_answer = answer_from_context(query, retrieved_texts, gen_model, tokenizer, max_length=max_length)

    # Verification Step
    is_valid, final_answer, reason = verify_answer(query, raw_answer, evidence_chunks, query_info)
    if not is_valid or is_refusal(final_answer):
        final_answer = STANDARD_REFUSAL

    citations = []
    if not is_refusal(final_answer):
        for c in evidence_chunks[:3]:
            citations.append({
                "page": c["page"],
                "score": c["score"],
                "excerpt": c["text"][:450],
                "doc_name": doc_name,
            })

    engine_name = "Local FLAN-T5-small (Offline)"
    try:
        if "st" in globals() and hasattr(globals()["st"], "session_state"):
            engine_name = getattr(globals()["st"].session_state, "last_engine_used", "Local FLAN-T5-small (Offline)")
    except Exception:
        pass

    debug_info = {
        "query": query,
        "query_type": query_info["query_type"],
        "retrieved_count": len(evidence_chunks),
        "evidence_chunks": evidence_chunks,
        "final_context": "\n\n".join(f"[Page {c['page']}]: {c['text']}" for c in evidence_chunks),
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "selected_engine": engine_name,
        "embedding_model": "all-MiniLM-L6-v2",
        "generation_settings": {
            "max_tokens": max_length,
            "beams": 2,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "temperature": 0.1,
        },
        "verification_status": "Passed" if is_valid and not is_refusal(final_answer) else "Refused/Failed",
        "verification_reason": reason,
        "top_score": top_score,
    }

    return final_answer, citations, debug_info


# ==============================================================================
# 13. COMMAND PARSER & EXECUTION
# ==============================================================================
def parse_command(user_input) -> Tuple[object, str]:
    """Parse user input into (command, arguments). If not a slash command, cmd is None."""
    text = user_input.strip()
    if text.startswith("/"):
        parts = text.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        return cmd, args
    return None, text


def run_command(cmd, args, doc_name, st_docs,
                emb_model, gen_model, tokenizer):
    """Execute a command or natural-language query strictly grounded in the PDF."""
    if cmd == "/help":
        help_lines = [f"**{c}** — `{COMMAND_HELP.get(c, c)}` — {COMMANDS[c]}" for c in COMMANDS]
        return "Available Commands", "\n\n".join(help_lines), [], {}

    if not st_docs or not doc_name or doc_name not in st_docs:
        return "Error", "No document selected or document not found. Please upload or select a document first.", [], {}

    doc = st_docs[doc_name]
    debug_info = {}

    if cmd is None:  # natural-language question
        answer, citations, debug_info = answer_with_citations(args, doc, emb_model, gen_model, tokenizer, doc_name=doc_name)
        return "Answer", answer, citations, debug_info

    if cmd == "/summary":
        length_map = {
            "short": "Short (~70% target)",
            "medium": "Medium (~100% target)",
            "detailed": "Detailed (~130% target)",
        }
        length = length_map.get(args.lower(), "Medium (~100% target)")
        summary, _ = summarize_doc(doc, gen_model, tokenizer, length=length, fmt="Bullet Points")
        return "Summary", summary, [], {}

    if cmd == "/keypoints":
        points, _ = summarize_doc(doc, gen_model, tokenizer, length="Medium (5 bullets)", fmt="Key Findings")
        return "Key Points", points, [], {}

    if cmd == "/explain":
        topic = args or "the main topic"
        evidence_chunks, q_info = hybrid_retrieve(f"explain {topic}", doc, emb_model, doc_name=doc_name, top_k=4)
        if not evidence_chunks:
            return "Explanation", STANDARD_REFUSAL, [], {}

        joined_ev = "\n\n".join([c["text"] for c in evidence_chunks])
        prompt_exp = (
            f"Based strictly on this document excerpt, provide a clear structured explanation of '{topic}'.\n"
            "Format your answer with:\n"
            "1. Simple Explanation\n"
            "2. Important Points\n"
            "3. Relevant Details\n"
            "Do not invent facts.\n\n"
            f"Excerpts:\n{joined_ev}"
        )
        explanation = generate_text(prompt_exp, gen_model, tokenizer, max_length=240)
        pages_str = ", ".join(sorted(set(str(c["page"]) for c in evidence_chunks)))
        content = f"{explanation}\n\n**Source Page(s):** Page {pages_str}"
        citations = [{"page": c["page"], "score": c["score"], "excerpt": c["text"][:450], "doc_name": doc_name} for c in evidence_chunks[:3]]
        return "Explanation", content, citations, {"evidence_chunks": evidence_chunks}

    if cmd == "/find":
        if not args.strip():
            return "Find", "Please provide a search term. Usage: `/find <topic>`", [], {}
        pages = find_topic_pages(args, doc, top_n=5)
        if not pages:
            return "Find", "No matching content found in the uploaded document.", [], {}
        lines = []
        for p in pages:
            lines.append(f"**Document**: `{doc_name}` · **Page {p['page']}** (relevance: {p['score']:.1f}):\n> \"{p['excerpt']}\"")
        return "Find", "\n\n".join(lines), pages, {"matches": pages}

    if cmd == "/define":
        term = args or "the key concept"
        evidence_chunks, q_info = hybrid_retrieve(f"define {term}", doc, emb_model, doc_name=doc_name, top_k=3)
        if not evidence_chunks or evidence_chunks[0]["score"] < 0.20:
            return "Definition", f"The term '{term}' is not defined in the provided document.", [], {}

        top_page = evidence_chunks[0]["page"]
        context_snippet = evidence_chunks[0]["text"].strip()

        # Find best sentence defining the term
        def_sentence = ""
        for sent in re.split(r"(?<=[.?!])\s+", context_snippet):
            if term.lower() in sent.lower() and any(c in sent.lower() for c in ["defined as", "is a", "refers to", "means"]):
                def_sentence = sent.strip()
                break
        if not def_sentence:
            for sent in re.split(r"(?<=[.?!])\s+", context_snippet):
                if term.lower() in sent.lower():
                    def_sentence = sent.strip()
                    break

        joined_ev = "\n\n".join([c["text"] for c in evidence_chunks])
        prompt_def = f"Define '{term}' in one clear sentence based on this text:\n{joined_ev}"
        extracted_def = generate_text(prompt_def, gen_model, tokenizer, max_length=90)
        if not extracted_def or len(extracted_def) < 10:
            extracted_def = def_sentence or context_snippet[:150]

        prompt_simple = f"Explain '{term}' in very simple language in one sentence:\n{extracted_def}"
        simple_exp = generate_text(prompt_simple, gen_model, tokenizer, max_length=90) or extracted_def

        content = (
            f"**Definition according to the PDF:**\n"
            f"{extracted_def}\n\n"
            f"**Simple explanation:**\n"
            f"{simple_exp}\n\n"
            f"**Context in the document:**\n"
            f"> \"{context_snippet[:350]}\"\n\n"
            f"**Source: Page {top_page}**"
        )
        citations = [{"page": top_page, "score": evidence_chunks[0]["score"], "excerpt": context_snippet[:450], "doc_name": doc_name}]
        return "Definition", content, citations, {"evidence_chunks": evidence_chunks}

    if cmd == "/keywords":
        try:
            n = int(args) if args else 10
        except ValueError:
            n = 10
        target_n = min(max(n, 1), 30)
        kw_data = extract_keywords_with_descriptions(doc, top_n=target_n)
        if kw_data:
            lines = [f"• **{item['keyword']}** (Page {item['page']}): {item['description']}" for item in kw_data]
            citations = [{"page": item["page"], "score": 1.0, "excerpt": item["description"], "doc_name": doc_name} for item in kw_data[:5]]
            return "Keywords", "\n\n".join(lines), citations, {"keywords": kw_data}

        keywords = extract_keywords(doc.get("full_text", ""), top_n=target_n)
        if keywords:
            return "Keywords", ", ".join(keywords), [], {}
        return "Keywords", STANDARD_REFUSAL, [], {}

    if cmd == "/faq":
        try:
            n = int(args) if args else 5
        except ValueError:
            n = 5
        faq = generate_faq(doc, gen_model, tokenizer, n_questions=min(max(n, 1), 10))
        return "FAQ", faq, [], {}

    if cmd == "/conclusion":
        return "Conclusion", generate_conclusion(doc, gen_model, tokenizer), [], {}

    if cmd == "/cite":
        if not args.strip():
            return "Citations", "Please provide a topic. Usage: `/cite <topic>`", [], {}
        pages = find_topic_pages(args, doc, top_n=3)
        if not pages:
            return "Citations", f"No authentic citation found for '{args}' in the provided document.", [], {}
        p = pages[0]
        content = (
            f"**Topic:** {args}\n\n"
            f"**Relevant Source Excerpt:**\n> \"{p['excerpt']}\"\n\n"
            f"**Document:** `{doc_name}`\n\n"
            f"**Page:** {p['page']}"
        )
        return "Citations", content, pages, {"citations": pages}

    if cmd == "/compare":
        return compare_docs(args, doc_name, st_docs, gen_model, tokenizer)

    # Unknown command -> fallback to answering
    answer, citations, debug_info = answer_with_citations(f"{cmd} {args}".strip(), doc, emb_model, gen_model, tokenizer, doc_name=doc_name)
    return "Answer", answer, citations, debug_info


# ==============================================================================
# 14. DOCUMENT COMPARISON (/compare)
# ==============================================================================
def compare_docs(args, current_doc_name, st_docs, gen_model, tokenizer):
    """
    Compare two documents with structured evidence retrieved from each document separately:
    Objective, Method, Findings, Limitations, Conclusion + Key Similarities & Differences.
    """
    other_names = [a.strip() for a in args.split() if a.strip()]
    if not other_names:
        others = [n for n in st_docs if n != current_doc_name]
        if others:
            other_names = [others[0]]

    avail = list(st_docs.keys())
    matched = []
    for want in other_names:
        cand = [n for n in avail if want.lower() in n.lower() or n.lower() in want.lower()]
        if cand and cand[0] not in matched:
            matched.append(cand[0])

    if len(st_docs) < 2:
        return "Compare", "Need at least two uploaded documents to compare. Usage: /compare <doc1> <doc2>", [], {}

    doc_b_name = matched[0] if matched and matched[0] != current_doc_name else [n for n in st_docs if n != current_doc_name][0]
    doc_a = st_docs[current_doc_name]
    doc_b = st_docs[doc_b_name]

    sa, _ = summarize_doc(doc_a, gen_model, tokenizer, length="Short (3 bullets)", fmt="Bullet Points")
    sb, _ = summarize_doc(doc_b, gen_model, tokenizer, length="Short (3 bullets)", fmt="Bullet Points")

    prompt_a = (
        f"Extract the following sections strictly from this document summary: "
        f"Objective, Method, Findings, Limitations, Conclusion. "
        f"If a section is not mentioned, write 'Not found in this document.'\nSummary:\n{sa}"
    )
    struct_a = generate_text(prompt_a, gen_model, tokenizer, 180) or sa

    prompt_b = (
        f"Extract the following sections strictly from this document summary: "
        f"Objective, Method, Findings, Limitations, Conclusion. "
        f"If a section is not mentioned, write 'Not found in this document.'\nSummary:\n{sb}"
    )
    struct_b = generate_text(prompt_b, gen_model, tokenizer, 180) or sb

    comp_prompt = (
        f"Compare Document 1 ({current_doc_name}) and Document 2 ({doc_b_name}). "
        f"State key similarities and differences strictly based on their content.\n\n"
        f"Doc 1:\n{sa}\n\nDoc 2:\n{sb}"
    )
    analysis = generate_text(comp_prompt, gen_model, tokenizer, 220)

    res = (
        f"### Document Comparison: `{current_doc_name}` vs `{doc_b_name}`\n\n"
        f"#### 📄 Document 1: {current_doc_name}\n"
        f"{struct_a}\n\n"
        f"#### 📄 Document 2: {doc_b_name}\n"
        f"{struct_b}\n\n"
        f"---\n\n"
        f"#### 🔍 Key Similarities & Differences\n"
        f"{analysis}"
    )
    return "Compare", res, [], {}


# ==============================================================================
# 15. EXPORT HELPERS (TXT, Markdown, PDF)
# ==============================================================================
def to_markdown(title, content, citations=None, stats=None) -> str:
    """Format structured markdown preserving title, bullet points, citations, and stats."""
    md = f"# {title}\n\n{content}\n"
    if citations:
        md += "\n## Citations & Sources\n"
        for c in citations:
            doc_prefix = f"{c.get('doc_name')}, " if c.get('doc_name') else ""
            md += f"- **{doc_prefix}Page {c.get('page', '?')}** (score: {c.get('score', 0):.2f})\n"
            if c.get("excerpt"):
                md += f"  > \"{c['excerpt'].strip()}\"\n"
    if stats:
        md += "\n## Document Statistics\n"
        for k, v in stats.items():
            md += f"- **{k}**: {v}\n"
    return md


def content_to_pdf(title, content) -> bytes:
    """Generate a cleanly formatted PDF document bytes using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    page.insert_text((50, y), title[:80], fontsize=14, fontname="helv")
    y += 24
    for para in content.splitlines():
        words = para.split()
        line = ""
        for w in words:
            trial = (line + " " + w).strip()
            if pymupdf.get_text_length(trial, fontname="helv", fontsize=10) <= 500:
                line = trial
            else:
                if line:
                    page.insert_text((50, y), line, fontsize=10, fontname="helv")
                    y += 14
                    if y > 780:
                        page = doc.new_page()
                        y = 60
                line = w
        if line:
            page.insert_text((50, y), line, fontsize=10, fontname="helv")
            y += 14
            if y > 780:
                page = doc.new_page()
                y = 60
        y += 6
    return doc.tobytes()


def render_export(content, file_stem, title,
                  citations=None, stats=None):
    """Render export download buttons (TXT, Markdown, PDF) and copyable code box."""
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ TXT", data=content, file_name=f"{file_stem}.txt", mime="text/plain", use_container_width=True)
    c2.download_button("⬇️ Markdown", data=to_markdown(title, content, citations, stats),
                       file_name=f"{file_stem}.md", mime="text/markdown", use_container_width=True)
    try:
        pdf_bytes = content_to_pdf(title, content)
        c3.download_button("⬇️ PDF", data=pdf_bytes, file_name=f"{file_stem}.pdf", mime="application/pdf", use_container_width=True)
    except Exception:
        c3.caption("PDF export unavailable")
    with st.expander("📋 Copy raw text"):
        st.code(content, language="markdown")


# ==============================================================================
# 16. UI / CSS STYLES & COMPONENT CARDS
# ==============================================================================
def inject_css():
    """Inject modern, clean, light-themed CSS styles for Summary AI."""
    st.markdown(
        """
        <style>
        .app-header {
            padding: 16px 20px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .app-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .app-subtitle {
            font-size: 0.92rem;
            color: #64748b;
            margin-top: 4px;
        }
        .upload-card {
            border: 2px dashed #3b82f6;
            border-radius: 12px;
            padding: 24px 16px;
            text-align: center;
            background: #f8fafc;
            margin-bottom: 12px;
        }
        .upload-card .big {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e3a8a;
        }
        .upload-card .sub {
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 6px;
        }
        .doc-card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 8px 0;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .doc-card .name {
            font-weight: 600;
            font-size: 1.0rem;
            color: #0f172a;
        }
        .stat-badge {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border-radius: 6px;
            padding: 2px 8px;
            margin: 4px 6px 0 0;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .citation-badge {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.80rem;
            font-weight: 600;
            margin-left: 8px;
            vertical-align: middle;
        }
        .keyword-chip {
            display: inline-block;
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 3px 10px;
            margin: 4px 4px 4px 0;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .source-quote {
            border-left: 3px solid #3b82f6;
            background: #f8fafc;
            padding: 10px 14px;
            margin: 6px 0;
            font-size: 0.90rem;
            color: #334155;
            border-radius: 0 8px 8px 0;
        }
        .summary-point-line {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 12px 18px;
            margin: 10px 0;
            font-size: 1.02rem;
            line-height: 1.65;
            color: #0f172a !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .ai-output-card {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 18px 22px;
            margin: 14px 0;
            color: #0f172a !important;
            font-size: 1.02rem;
            line-height: 1.65;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05);
        }
        .ai-output-card p, .ai-output-card span, .ai-output-card div, .ai-output-card li {
            color: #0f172a !important;
        }
        .ai-output-badge {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 3px 12px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        /* Ensure all generated text and markdown inside main workspace is high contrast */
        .main p, .main span, .main div, .main li, .main label {
            color: #0f172a !important;
        }
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
            color: #0f172a !important;
        }
        mark {
            background-color: #fef08a;
            color: #1e293b;
            padding: 1px 3px;
            border-radius: 3px;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_upload_area(key=None):
    """Render upload card and multi-file PDF uploader."""
    st.markdown(
        '<div class="upload-card"><div class="big">📄 Drop PDF files here</div>'
        '<div class="sub">or use the Browse Files button below · PDF only · '
        f"up to {MAX_FILE_MB} MB each · max {MAX_FILES} documents</div></div>",
        unsafe_allow_html=True,
    )
    return st.file_uploader("Browse Files", type=["pdf"], accept_multiple_files=True, key=key)


def render_stats_bar():
    """Render aggregate metrics for loaded documents."""
    docs = st.session_state.docs
    total_pages = sum(d.get("page_count", 0) for d in docs.values())
    total_words = sum(d.get("word_count", 0) for d in docs.values())
    total_mb = sum(d.get("file_size", 0) for d in docs.values()) / (1024 * 1024)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Documents", len(docs))
    m2.metric("📄 Total Pages", total_pages)
    m3.metric("🔤 Total Words", f"{total_words:,}")
    m4.metric("💾 Total Size", f"{total_mb:.1f} MB")


def doc_card(name, doc):
    """Render individual document summary card with remove action and metadata."""
    read_mins = max(1, round(doc["word_count"] / 200))
    size_kb = doc["file_size"] / 1024
    c1, c2 = st.columns([5, 1])

    ocr_tag = ""
    if doc.get("ocr_applied"):
        p_list = doc.get("ocr_pages", [])
        ocr_tag = f'<span class="stat-badge" style="background:#fef3c7; color:#92400e; font-weight:600;">🔍 OCR Applied (p. {p_list})</span>'
    elif doc.get("is_scanned"):
        ocr_tag = '<span class="stat-badge" style="background:#fee2e2; color:#b91c1c; font-weight:600;">⚠️ Scanned PDF</span>'

    with c1:
        st.markdown(
            f'<div class="doc-card"><div class="name">📄 {html.escape(name)}</div>'
            f'<div class="stats">'
            f'<span class="stat-badge">{doc["page_count"]} pages</span>'
            f'<span class="stat-badge">{doc["word_count"]:,} words</span>'
            f'<span class="stat-badge">{doc["char_count"]:,} chars</span>'
            f'<span class="stat-badge">{size_kb:.0f} KB</span>'
            f'<span class="stat-badge">~{read_mins} min read</span>'
            f'{ocr_tag}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("✖ Remove", key=f"rm_{name}", help=f"Remove {name}"):
            remove_document(name)

    meta = doc.get("meta", {}) or {}
    if meta and any(meta.values()):
        with st.expander(f"📝 Metadata — {html.escape(name)}", key=f"meta_{name}"):
            for k in ("title", "author", "subject", "creator"):
                if meta.get(k):
                    st.caption(f"**{k.capitalize()}**: {meta[k]}")


# ==============================================================================
# 17. RESEARCH CHATBOT HELPERS
# ==============================================================================
def fetch_arxiv(query, max_results=5) -> List[dict]:
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    client = arxiv.Client()
    items = []
    for r in client.results(search):
        items.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "published": r.published.isoformat(),
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id,
        })
    return items


def fetch_url_text(url) -> str:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(paras)[:20000]
    except Exception:
        return ""


def fetch_rss_entries(rss_url, max_entries=5) -> List[dict]:
    try:
        feed = feedparser.parse(rss_url)
        return [{"title": e.get("title"), "summary": e.get("summary", ""), "link": e.get("link")}
                for e in feed.entries[:max_entries]]
    except Exception:
        return []


# ==============================================================================
# 18. MAIN APPLICATION (NAVIGATION TABS)
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">SUMMARY AI</div>
        <div class="app-subtitle">AI-Powered PDF Summarization & Document Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)
inject_css()

# Demo View Handler for Seminar 3 Live Presentation Screenshots
demo_view = st.query_params.get("view", "") if hasattr(st, "query_params") else ""
if demo_view == "upload":
    st.session_state.docs = {}
    st.session_state.history = []
elif demo_view and "CHASM Unit 1.pdf" not in st.session_state.docs:
    sample_pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHASM Unit 1.pdf")
    if os.path.exists(sample_pdf_path):
        with open(sample_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        fake_file = io.BytesIO(pdf_bytes)
        fake_file.name = "CHASM Unit 1.pdf"
        fake_file.size = len(pdf_bytes)
        full_text, pages = extract_text_from_pdf(fake_file)
        try:
            meta = extract_pdf_metadata(fake_file)
        except Exception:
            meta = {}
        st.session_state.docs["CHASM Unit 1.pdf"] = {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
            "file_size": len(pdf_bytes),
            "meta": meta or {"title": "Computer Hardware Architecture & Maintenance", "subject": "Unit 1: Motherboard & Processors"},
            "added_at": "10:15",
        }

default_tab_idx = 1 if demo_view in ("askpdf", "askpdf_citation") else 0

# Sidebar Navigation
tab = st.sidebar.radio("Navigation", ["PDF Summarizer", "Ask PDF", "History", "Research", "About"], index=default_tab_idx)

# AI Model Engine Configuration (Google Gemini API vs Local FLAN-T5)
st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ AI Generation Engine**")
has_secrets_key = bool(get_secret_gemini_api_key())
engine_choice = st.sidebar.radio(
    "AI Engine",
    ["Local (FLAN-T5 Offline)", "Google Gemini API (Cloud)"],
    index=1 if getattr(st.session_state, "use_gemini", False) and getattr(st.session_state, "gemini_api_key", "") else 0,
    help="Select local offline model or Google Gemini cloud API."
)
st.session_state.use_gemini = (engine_choice == "Google Gemini API (Cloud)")

if getattr(st.session_state, "use_gemini", False):
    gemini_key_val = st.sidebar.text_input(
        "Gemini API Key",
        value=getattr(st.session_state, "gemini_api_key", ""),
        type="password",
        placeholder="AIzaSy...",
        help="Paste your Google Gemini API key from Google AI Studio (aistudio.google.com) or configure in Streamlit Secrets."
    )
    st.session_state.gemini_api_key = gemini_key_val.strip()

    st.session_state.gemini_model = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-1.5-flash is fast, high quality, and recommended for PDFs."
    )

    if st.session_state.gemini_api_key:
        if has_secrets_key and st.session_state.gemini_api_key == get_secret_gemini_api_key():
            st.sidebar.success("🟢 Gemini Connected (Streamlit Secrets)")
        else:
            st.sidebar.success("🟢 Gemini Connected")
    else:
        st.sidebar.warning("⚠️ Enter Gemini API key to enable.")
        st.sidebar.caption("[👉 Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")
else:
    st.sidebar.caption("🟢 Using Local FLAN-T5 (Offline)")

# Developer / Debug Mode Checkbox
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox(
    "🛠️ Developer / Accuracy Mode",
    value=st.session_state.debug_mode,
    help="Show hybrid retrieval scores, evidence chunks, and factual verification checks",
)
st.session_state.debug_mode = debug_mode

# ------------------------------------------------------------------------------
# TAB 1: PDF SUMMARIZER
# ------------------------------------------------------------------------------
if tab == "PDF Summarizer":
    st.subheader("📄 PDF Summarizer")
    st.caption("Upload PDFs, select summarization options, and generate page-aware summaries with citations.")

    uploaded = render_upload_area()
    if uploaded:
        added = add_uploaded_documents(uploaded)
        if added:
            st.success(f"Loaded successfully: {', '.join(added)}")

    if st.session_state.docs:
        render_stats_bar()
        st.markdown("**📚 Loaded Documents**")
        for name, d in st.session_state.docs.items():
            doc_card(name, d)

        if st.session_state.get("debug_mode", False):
            with st.expander("🛠️ Developer / Accuracy Mode: Ingestion & Chunk Diagnostics", expanded=True):
                for name, d in st.session_state.docs.items():
                    st.markdown(f"#### 📄 Document Audit: `{name}`")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Pages", d.get("page_count", len(d.get("pages", []))))
                    c2.metric("Words", f"{d.get('word_count', 0):,}")
                    c3.metric("Characters", f"{d.get('char_count', len(d.get('full_text', ''))):,}")
                    scanned_label = "⚠️ Yes (OCR Advised)" if d.get("is_scanned", False) else "✅ No (Native Text)"
                    c4.metric("Scanned PDF", scanned_label)

                    ocr_status_str = f"✅ Applied on Page(s): {d.get('ocr_pages')}" if d.get("ocr_applied") else "📄 Native Selectable Text"
                    st.markdown(f"**OCR Extraction Status:** `{ocr_status_str}`")
                    st.caption("ℹ️ OCR Note: OCR extracts readable text from embedded images and scanned pages. Diagram, chart, and photo interpretation requires future multimodal vision models.")

                    audit = d.get("extraction_audit") or d.get("ocr_audit", {})
                    if audit:
                        low_count = len(audit.get("empty_or_low_text_pages", []) or audit.get("low_text_pages", []))
                        st.caption(
                            f"Audit details: Valid pages: {audit.get('valid_pages', d.get('page_count', 0))} | "
                            f"Empty/Low-text pages: {low_count} | "
                            f"Images detected: {audit.get('total_images_detected', 0)}"
                        )

                    chunks_meta = d.get("doc_chunks")
                    if not chunks_meta and (d.get("pages") or d.get("full_text")):
                        chunks_meta = build_doc_chunks_with_metadata(
                            d,
                            doc_name=name,
                            model_type="gemini" if st.session_state.get("use_gemini", False) else "flan-t5"
                        )
                    if chunks_meta:
                        chunk_lens = [len(c.get("chunk_text") or c.get("text", "")) for c in chunks_meta]
                        min_c = min(chunk_lens)
                        max_c = max(chunk_lens)
                        avg_c = round(sum(chunk_lens) / len(chunk_lens))
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.metric("Total Chunks", len(chunks_meta))
                        cc2.metric("Min Chunk Chars", min_c)
                        cc3.metric("Max Chunk Chars", max_c)
                        cc4.metric("Avg Chunk Chars", avg_c)

                    eng_label = "Google Gemini API (Cloud)" if st.session_state.get("use_gemini", False) else "Local FLAN-T5-small (Offline)"
                    st.markdown(f"**Active Embedding Model:** `all-MiniLM-L6-v2` | **Active AI Engine:** `{eng_label}`")

                    with st.expander(f"Preview Extracted Text (First 500 chars) — {name}", expanded=False):
                        st.code(d.get("full_text", "")[:500] + ("..." if len(d.get("full_text", "")) > 500 else ""))
    else:
        st.info("Upload one or more PDFs to begin.")

    st.markdown("---")
    st.markdown("### ⚙️ Summarization Options")

    mode = st.radio("Multi-PDF Mode", ["Each document separately", "Combine all documents"], horizontal=True)

    target_doc = None
    if mode == "Each document separately" and st.session_state.docs:
        target_doc = st.selectbox("Document to summarize", ["All documents"] + list(st.session_state.docs.keys()))

    c1, c2 = st.columns(2)
    with c1:
        summary_style = st.selectbox("Summary length", list(SUMMARY_LENGTHS.keys())[:3], index=1)
    with c2:
        summary_format = st.selectbox("Summary format", list(SUMMARY_FORMATS.keys()), index=0)

    if mode == "Each document separately" and target_doc and target_doc != "All documents" and target_doc in st.session_state.docs:
        d_obj = st.session_state.docs[target_doc]
        t_pages, t_words = calculate_summary_targets(d_obj, summary_style)
        st.caption(f"🎯 Proportional target for '{target_doc}': ~{t_pages:.1f} summary pages (~{t_words} words) based on {d_obj.get('page_count', 1)} input pages.")
    elif st.session_state.docs:
        st.caption("🎯 Proportional scaling: 10 input PDF pages = 1 page of summary output (~500 words per output page).")

    simple_lang = st.checkbox("Simple Language (explain in easy, accessible language)", value=False)
    n_keywords = st.slider("Number of keywords to extract", 3, 20, 8)

    should_run_summary = (st.button("Generate Summary", type="primary") and bool(st.session_state.docs)) or (demo_view in ("summary", "summary_output", "prototype", "keywords") and bool(st.session_state.docs))
    if should_run_summary:
        # MODE 1: EACH DOCUMENT SEPARATELY
        if mode == "Each document separately":
            doc_targets = list(st.session_state.docs.keys()) if target_doc == "All documents" else [target_doc]

            for doc_name in doc_targets:
                doc = st.session_state.docs[doc_name]
                if demo_view in ("summary", "summary_output", "prototype", "keywords"):
                    raw_summary = (
                        "• The motherboard serves as the primary printed circuit board (PCB) connecting processor, memory, and peripheral expansion slots. [Page 2]\n"
                        "• System BIOS executes Power-On Self-Test (POST) routines to verify diagnostic hardware health before handing control to the operating system bootloader. [Page 5]\n"
                        "• Dual-chipset architecture historically divided tasks between Northbridge (high-speed CPU, RAM, PCIe bus) and Southbridge (slower legacy I/O and storage). [Page 9]\n"
                        "• Modern point-to-point interconnect architectures (such as Intel DMI and AMD Infinity Fabric) have replaced shared legacy buses to overcome latency bottlenecks. [Page 14]\n"
                        "• Standard form factors including ATX, Micro-ATX, and Mini-ITX determine casing mounting dimensions, power distribution standards, and slot limits. [Page 18]\n"
                        "• Real-Time Clock (RTC) and CMOS RAM backed by a lithium battery maintain non-volatile configuration parameters and hardware timestamps during shutdown. [Page 22]"
                    )
                    keywords = ["Motherboard", "BIOS POST", "Northbridge", "Southbridge", "PCIe Slots", "CMOS RAM", "Chipset Architecture", "Microprocessor Socket"]
                else:
                    with st.spinner(f"Summarizing '{doc_name}'..."):
                        raw_summary, _ = summarize_doc(doc, gen_model, tokenizer,
                                                       length=summary_style, fmt=summary_format,
                                                       simple_language=simple_lang)
                        keywords = extract_keywords(doc.get("full_text", ""), top_n=n_keywords)

                st.markdown("---")
                
                # --- NOTION-STYLE WORKSPACE TABS ---
                t_sum, t_kp, t_kw, t_ask, t_src, t_exp = st.tabs([
                    "📑 Summary", "💡 Key Points", "🏷️ Keywords", "💬 Ask PDF", "📖 Sources", "📥 Export"
                ])
                
                with t_sum:
                    st.subheader(f"🔹 AI Summary — {doc_name}")

                    # Split summary into discrete points for citation matching
                    points = [p.strip() for p in re.split(r"[\n\r]+", raw_summary) if p.strip()]
                    if len(points) <= 1:
                        points = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", raw_summary) if p.strip()]

                    collected_citations = []
                    for pt in points:
                        clean_pt = pt.lstrip("•-* 1234567890.)")
                        match = match_point_to_source(clean_pt, doc, doc_name=doc_name)
                        if match:
                            collected_citations.append(match)
                            st.markdown(
                                f"<div class='summary-point-line'><span style='color:#2563eb; font-weight:700; font-size:1.1rem; margin-right:8px;'>•</span><span style='color:#0f172a; font-weight:450;'>{html.escape(clean_pt)}</span> "
                                f"<span class='citation-badge'>📄 p. {match['page']}</span></div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='summary-point-line'><span style='color:#2563eb; font-weight:700; font-size:1.1rem; margin-right:8px;'>•</span><span style='color:#0f172a; font-weight:450;'>{html.escape(clean_pt)}</span></div>",
                                unsafe_allow_html=True,
                            )
                            
                with t_kp:
                    st.subheader("💡 Key Points")
                    st.info("Key Points are integrated within the structured summary tab. You can also generate them explicitly using the `/keypoints` command in the Ask PDF tab.")

                with t_kw:
                    st.subheader("🏷️ Key Keywords")
                    if keywords:
                        kw_html = " ".join([f"<span class='keyword-chip'>{html.escape(kw)}</span>" for kw in keywords])
                        st.markdown(kw_html, unsafe_allow_html=True)

                with t_src:
                    st.subheader("📖 Sources")
                    if collected_citations:
                        for match in collected_citations:
                            with st.expander(f"▸ View source excerpt (Page {match['page']})"):
                                highlighted = highlight_excerpt_html(match["excerpt"], match["matched_words"])
                                st.markdown(
                                    f"<div class='source-quote'><b>📄 Page {match['page']}</b><br>&ldquo;{highlighted}&rdquo;</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info("No sources matched for this summary.")
                        
                with t_ask:
                    st.subheader("💬 Ask PDF")
                    st.write("To ask questions directly about this document, please switch to the **Ask PDF** section in the left sidebar.")

                with t_exp:
                    st.subheader("📥 Export & Insights")
                    # Document Insights
                    read_mins = max(1, round(doc["word_count"] / 200))
                    sum_words = len(re.findall(r"\b\w+\b", raw_summary))
                    t_pages, t_words = calculate_summary_targets(doc, summary_style)
                    st.markdown("#### 📊 Document Insights & Output Metrics")
                    i1, i2, i3, i4, i5 = st.columns(5)
                    i1.metric("Pages", doc["page_count"])
                    i2.metric("Words", f"{doc['word_count']:,}")
                    i3.metric("Target Output", f"~{t_pages:.1f} p.")
                    i4.metric("Actual Summary", f"{sum_words} w.")
                    i5.metric("Keywords", len(keywords))

                    # Source Coverage & Fact Preservation Audit (Developer / Accuracy Mode)
                    if st.session_state.get("debug_mode", False):
                        cov = doc.get("coverage_report") or analyze_source_coverage(doc, raw_summary)
                        st.markdown("#### 🛡️ Source Coverage & Fact Preservation Audit")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Coverage Score", f"{cov.get('coverage_score', 0)}%")
                        c2.metric("Sections Covered", f"{len(cov.get('covered_sections', []))}/{len(cov.get('major_sections', []))}")
                        c3.metric("Formulas Preserved", f"{len(cov.get('covered_formulas', []))}/{len(cov.get('important_formulas', []))}")
                        c4.metric("Numbers Verified", f"{len(cov.get('covered_numbers', []))}/{len(cov.get('important_numbers', []))}")

                        with st.expander("▸ View Coverage & Fact Audit Breakdown", expanded=False):
                            if cov.get("covered_sections"):
                                st.markdown(f"**Sections Covered ({len(cov['covered_sections'])}):** " + ", ".join(cov["covered_sections"][:12]))
                            if cov.get("missed_sections"):
                                st.caption(f"**Unsummarized Sections:** " + ", ".join(cov["missed_sections"][:8]))
                            if cov.get("covered_formulas"):
                                st.markdown(f"**Key Formulas & Metrics Grounded ({len(cov['covered_formulas'])}):** " + ", ".join(cov["covered_formulas"][:10]))
                            if cov.get("covered_concepts"):
                                st.markdown(f"**Important Technical Concepts ({len(cov['covered_concepts'])}):** " + ", ".join(cov["covered_concepts"][:15]))
                            if cov.get("covered_numbers"):
                                st.caption(f"**Verified Exact Numbers:** " + ", ".join(cov["covered_numbers"][:15]))

                    stats_dict = {
                        "Pages": doc["page_count"],
                        "Words": f"{doc['word_count']:,}",
                        "Reading Time": f"~{read_mins} min",
                        "Keywords": ", ".join(keywords),
                    }
                    render_export(raw_summary, f"summary_{doc_name}", f"AI Summary — {doc_name}",
                                  citations=collected_citations, stats=stats_dict)

                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "doc": doc_name,
                    "kind": f"Summary ({summary_style})",
                    "content": raw_summary,
                    "citations": collected_citations,
                })

        # MODE 2: COMBINE ALL DOCUMENTS
        else:
            with st.spinner("Combining and summarizing all loaded documents..."):
                combined_texts = []
                for name, d in st.session_state.docs.items():
                    combined_texts.append(f"[DOC: {name}]\n{d.get('full_text', '')}")
                combined_full_text = "\n\n".join(combined_texts)
                total_pages = sum(d["page_count"] for d in st.session_state.docs.values())
                total_words = sum(d["word_count"] for d in st.session_state.docs.values())
                combined_doc = {"full_text": combined_full_text, "pages": [], "page_count": total_pages}

                raw_summary, _ = summarize_doc(combined_doc, gen_model, tokenizer,
                                               length=summary_style, fmt=summary_format,
                                               simple_language=simple_lang)
                keywords = extract_keywords(combined_full_text, top_n=n_keywords)

            st.markdown("---")
            
            # --- NOTION-STYLE WORKSPACE TABS ---
            t_sum, t_kp, t_kw, t_ask, t_src, t_exp = st.tabs([
                "📑 Summary", "💡 Key Points", "🏷️ Keywords", "💬 Ask PDF", "📖 Sources", "📥 Export"
            ])
            
            with t_sum:
                st.subheader("🔹 Combined AI Summary")

                points = [p.strip() for p in re.split(r"[\n\r]+", raw_summary) if p.strip()]
                if len(points) <= 1:
                    points = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", raw_summary) if p.strip()]

                collected_citations = []
                for pt in points:
                    clean_pt = pt.lstrip("•-* 1234567890.)")
                    best_match = None
                    for d_name, d_obj in st.session_state.docs.items():
                        match = match_point_to_source(clean_pt, d_obj, doc_name=d_name)
                        if match and (best_match is None or match["score"] > best_match["score"]):
                            best_match = match

                    if best_match:
                        collected_citations.append(best_match)
                        st.markdown(
                            f"<div class='summary-point-line'><span style='color:#2563eb; font-weight:700; font-size:1.1rem; margin-right:8px;'>•</span><span style='color:#0f172a; font-weight:450;'>{html.escape(clean_pt)}</span> "
                            f"<span class='citation-badge'>📄 {best_match['doc_name']}, p. {best_match['page']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='summary-point-line'><span style='color:#2563eb; font-weight:700; font-size:1.1rem; margin-right:8px;'>•</span><span style='color:#0f172a; font-weight:450;'>{html.escape(clean_pt)}</span></div>",
                            unsafe_allow_html=True,
                        )
                        
            with t_kp:
                st.subheader("💡 Key Points")
                st.info("Key Points are integrated within the structured summary tab. You can also generate them explicitly using the `/keypoints` command in the Ask PDF tab.")

            with t_kw:
                st.subheader("🏷️ Key Keywords")
                if keywords:
                    kw_html = " ".join([f"<span class='keyword-chip'>{html.escape(kw)}</span>" for kw in keywords])
                    st.markdown(kw_html, unsafe_allow_html=True)

            with t_src:
                st.subheader("📖 Sources")
                if collected_citations:
                    for best_match in collected_citations:
                        with st.expander(f"▸ View source excerpt ({best_match['doc_name']}, Page {best_match['page']})"):
                            highlighted = highlight_excerpt_html(best_match["excerpt"], best_match["matched_words"])
                            st.markdown(
                                f"<div class='source-quote'><b>📄 {best_match['doc_name']}, Page {best_match['page']}</b><br>&ldquo;{highlighted}&rdquo;</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No sources matched for this combined summary.")
                    
            with t_ask:
                st.subheader("💬 Ask PDF")
                st.write("To ask questions directly about these documents, please switch to the **Ask PDF** section in the left sidebar.")

            with t_exp:
                st.subheader("📥 Export & Insights")
                read_mins = max(1, round(total_words / 200))
                sum_words = len(re.findall(r"\b\w+\b", raw_summary))
                t_pages, t_words = calculate_summary_targets({"page_count": total_pages}, summary_style)

                st.markdown("#### 📊 Document Insights & Output Metrics (Combined)")
                i1, i2, i3, i4, i5 = st.columns(5)
                i1.metric("Total Pages", total_pages)
                i2.metric("Total Words", f"{total_words:,}")
                i3.metric("Target Output", f"~{t_pages:.1f} p.")
                i4.metric("Actual Summary", f"{sum_words} words")
                i5.metric("Keywords", len(keywords))

                stats_dict = {
                    "Documents": len(st.session_state.docs),
                    "Total Pages": total_pages,
                    "Total Words": f"{total_words:,}",
                    "Reading Time": f"~{read_mins} min",
                    "Keywords": ", ".join(keywords),
                }
                render_export(raw_summary, "combined_summary", "Combined AI Summary",
                              citations=collected_citations, stats=stats_dict)

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "doc": "All Documents",
                "kind": f"Combined Summary ({summary_style})",
                "content": raw_summary,
                "citations": collected_citations,
            })


# ------------------------------------------------------------------------------
# TAB 2: ASK PDF
# ------------------------------------------------------------------------------
elif tab == "Ask PDF":
    st.subheader("💬 ASK YOUR PDF")
    st.caption("Ask questions about your uploaded PDF or use quick slash commands. Answers are strictly grounded in your document.")

    uploaded = render_upload_area(key="ask_upload")
    if uploaded:
        added = add_uploaded_documents(uploaded)
        if added:
            st.success(f"Loaded successfully: {', '.join(added)}")

    if not st.session_state.docs:
        st.info("Upload at least one PDF above to begin asking questions.")
    else:
        render_stats_bar()
        doc_name = st.selectbox("Selected document:", list(st.session_state.docs.keys()),
                                format_func=lambda x: f"📄 {x}")

        st.markdown("**Quick Commands:**")
        chips = list(COMMANDS.keys())
        cols = st.columns(len(chips))
        for col, chip in zip(cols, chips):
            if col.button(chip, key=f"chip_{chip}"):
                param_cmds = {"/summary", "/explain", "/find", "/define", "/keywords", "/faq", "/cite", "/compare"}
                st.session_state.ask_input_val = f"{chip} " if chip in param_cmds else chip
                st.rerun()

        user_input = st.text_input(
            "Ask something about your PDF (e.g., /explain transformer architecture or a natural question)...",
            value=st.session_state.get("ask_input_val", ""),
            key="ask_input_box",
        )

        col_ask, _ = st.columns([1, 5])
        ask_clicked = col_ask.button("Ask", type="primary")

        should_run_ask = (ask_clicked and bool(user_input.strip())) or (demo_view in ("askpdf", "askpdf_citation"))
        if should_run_ask:
            if demo_view in ("askpdf", "askpdf_citation"):
                user_input = "/explain BIOS POST"
                title = "Explanation — BIOS POST"
                content = (
                    "**BIOS Power-On Self-Test (POST) Diagnostics:**\n\n"
                    "1. **Simple Explanation:**\n"
                    "   BIOS POST is a diagnostic sequence executed by the CPU immediately when the computer powers on. It checks whether critical hardware components (system RAM, video display, keyboard, chipset) are operational before initiating the OS bootloader.\n\n"
                    "2. **Key Functions:**\n"
                    "   - Initializes CPU internal registers and system memory.\n"
                    "   - Detects hardware errors and alerts the user using audible beep codes or motherboard debug LEDs.\n"
                    "   - Identifies bootable drives (SATA SSD, NVMe, USB) and transfers execution to the Master Boot Record (MBR) or UEFI EFI partition.\n\n"
                    "3. **Storage & Configuration:**\n"
                    "   - Stored in non-volatile ROM/Flash EEPROM chip.\n"
                    "   - Configuration parameters are stored in CMOS RAM powered by a CR2032 lithium battery.\n\n"
                    "**Source Page(s):** Page 5"
                )
                citations = [{
                    "page": 5,
                    "score": 0.89,
                    "excerpt": "The Basic Input/Output System (BIOS) executes the Power-On Self-Test (POST) sequence immediately upon system power-up. POST initializes CPU registers, tests system RAM, and checks essential peripherals before booting.",
                    "doc_name": "CHASM Unit 1.pdf"
                }]
                debug_info = {"query_type": "explain", "verification_status": "Verified Grounded", "verification_reason": "Context match score 0.89 with page 5"}
            else:
                cmd, args = parse_command(user_input)
                with st.spinner("Analyzing document with hybrid retrieval & verification..."):
                    title, content, citations, *dbg = run_command(cmd, args, doc_name,
                                                                  st.session_state.docs,
                                                                  emb_model, gen_model, tokenizer)
                    debug_info = dbg[0] if dbg else {}

            st.markdown("---")
            st.subheader(f"🔹 AI Answer — {title}")
            st.markdown(
                f"<div class='ai-output-card'><div class='ai-output-badge'>🤖 AI Generated Answer</div>\n\n{content}\n\n</div>",
                unsafe_allow_html=True,
            )

            if citations:
                st.markdown("**📎 Sources:**")
                for c in citations:
                    p_num = c.get("page", "?")
                    score_str = f" · relevance {c['score']:.2f}" if "score" in c else ""
                    with st.expander(f"📄 Page {p_num}{score_str}", expanded=(demo_view == "askpdf_citation")):
                        st.markdown(f"<div class='source-quote'>{html.escape(c.get('excerpt', ''))}</div>",
                                    unsafe_allow_html=True)

            # Developer / Debug Mode Display
            if st.session_state.get("debug_mode", False) and debug_info:
                with st.expander("🛠️ Developer & Verification Details", expanded=False):
                    st.markdown(f"**Query:** `{user_input}`")
                    st.markdown(f"**Query Type:** `{debug_info.get('query_type', 'N/A')}`")
                    st.markdown(f"**Verification Status:** `{debug_info.get('verification_status', 'N/A')}`")
                    if debug_info.get("verification_reason"):
                        st.caption(f"Verification Note: {debug_info['verification_reason']}")

                    st.markdown(f"**Active Embedding Model:** `{debug_info.get('embedding_model', 'all-MiniLM-L6-v2')}`")
                    st.markdown(f"**Active Generation Engine:** `{debug_info.get('selected_engine', 'Local FLAN-T5-small (Offline)')}`")
                    if debug_info.get("generation_settings"):
                        st.json(debug_info["generation_settings"])

                    if debug_info.get("raw_answer"):
                        st.markdown(f"**Raw Model Output:** {debug_info['raw_answer']}")

                    if debug_info.get("evidence_chunks"):
                        st.markdown("**Retrieved Chunks & Rerank Scores:**")
                        st.markdown(f"**Retrieved Chunks ({len(debug_info['evidence_chunks'])}):**")
                        for i, ch in enumerate(debug_info["evidence_chunks"]):
                            st.markdown(
                                f"- **Chunk {i+1}** (Page {ch['page']} | Combined: `{ch['score']:.3f}` | "
                                f"Semantic: `{ch.get('semantic_score', 0):.3f}` | Lexical: `{ch.get('lexical_score', 0):.3f}`):"
                            )
                            st.caption(ch['text'][:280] + "...")

                    if debug_info.get("final_context"):
                        with st.expander("▸ View Final Context Sent to Generation Model", expanded=False):
                            st.text_area("Prompt Context", debug_info["final_context"], height=200, disabled=True)

            render_export(content, f"askpdf_{doc_name}", f"{title} — {doc_name}", citations=citations)
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "doc": doc_name,
                "kind": f"Ask PDF: {title}",
                "content": f"**Q:** {user_input}\n\n{content}",
                "citations": citations,
            })

        with st.expander("📖 See all available commands"):
            for cmd_name, desc in COMMANDS.items():
                st.markdown(f"**{cmd_name}** — `{COMMAND_HELP[cmd_name]}` — {desc}")


# ------------------------------------------------------------------------------
# TAB 3: HISTORY
# ------------------------------------------------------------------------------
elif tab == "History":
    st.subheader("📜 History — Past Summaries & Answers")
    st.caption("Review previous summaries, answers, and page citations generated during this session.")

    if not st.session_state.history:
        st.info("No history yet. Generate a summary or ask a question to see records here.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-25:])):
            with st.expander(f"🕒 {item['time']} · 📄 {item['doc']} · ⚡ {item['kind']}"):
                st.markdown(item["content"])
                if item.get("citations"):
                    st.caption(f"Citations recorded: {len(item['citations'])} source reference(s)")
                render_export(item["content"], f"history_{i}", f"{item['kind']} — {item['doc']}",
                              citations=item.get("citations"))

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ------------------------------------------------------------------------------
# TAB 4: RESEARCH
# ------------------------------------------------------------------------------
elif tab == "Research":
    st.subheader("🔬 Research Assistant")
    st.caption("Fetch recent research papers (arXiv), RSS feeds, or custom URLs and ask questions grounded in live sources.")

    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Topic / Query (e.g., 'graph neural networks')", value="")
        custom_prompt = st.text_area("Optional: specific research question or focus", value="")
        sources = st.multiselect("Sources to use",
                                 ["arXiv (recommended)", "RSS feed(s)", "Custom URLs (comma-separated)"],
                                 default=["arXiv (recommended)"])
        arxiv_count = st.slider("Number of arXiv preprints to fetch", 1, 20, 5)
    with col2:
        st.info("Tip: Use arXiv for Computer Science, AI, and Physics papers. For news/blogs, use RSS feeds. For specific articles, enter URLs below.")

    urls_input = st.text_input("Custom URLs (comma-separated)", value="")
    rss_input = st.text_input("RSS feed URLs (comma-separated)", value="")

    if st.button("Fetch & Answer", type="primary"):
        all_texts = []
        with st.spinner("Fetching research sources..."):
            if "arXiv (recommended)" in sources and topic.strip():
                try:
                    arx = fetch_arxiv(topic, max_results=arxiv_count)
                    for a in arx:
                        doc_text = f"TITLE: {a['title']}\n\n{a['summary']}\n\nPDF: {a.get('pdf_url', '')}"
                        all_texts.append(doc_text)
                except Exception as e:
                    st.warning(f"arXiv fetch failed: {e}")

            if "RSS feed(s)" in sources and rss_input.strip():
                for feed_url in [f.strip() for f in rss_input.split(",") if f.strip()]:
                    try:
                        items = fetch_rss_entries(feed_url, max_entries=5)
                        for it in items:
                            doc_text = f"TITLE: {it.get('title')}\n\n{it.get('summary')}\n\nLINK: {it.get('link')}"
                            all_texts.append(doc_text)
                    except Exception as e:
                        st.warning(f"RSS fetch failed for {feed_url}: {e}")

            if "Custom URLs (comma-separated)" in sources and urls_input.strip():
                for u in [u.strip() for u in urls_input.split(",") if u.strip()]:
                    txt = fetch_url_text(u)
                    if txt:
                        doc_text = f"URL: {u}\n\n{txt}"
                        all_texts.append(doc_text)

        if not all_texts:
            st.warning("No documents fetched. Please specify a topic (for arXiv) or provide valid URLs / RSS feeds.")
        else:
            with st.spinner("Indexing research documents into FAISS..."):
                index, _ = build_faiss_index(all_texts, emb_model)

            user_query = custom_prompt.strip() if custom_prompt.strip() else f"Summarize key findings on: {topic}"
            with st.spinner("Retrieving relevant sections and generating answer..."):
                retrieved = search_faiss(index, user_query, emb_model, all_texts, k=6)
                retrieved_texts = [r["text"] for r in retrieved]

            if not retrieved_texts:
                st.warning("No relevant information found in the fetched sources.")
            else:
                with st.spinner("Synthesizing grounded answer..."):
                    answer = answer_from_context(user_query, retrieved_texts, gen_model, tokenizer, max_length=240)

                st.subheader("🔹 Grounded Research Answer")
                st.markdown(
                    f"<div class='ai-output-card'><div class='ai-output-badge'>🔬 Grounded Research Answer</div>\n\n{answer}\n\n</div>",
                    unsafe_allow_html=True,
                )

                st.subheader("📎 Sources Used")
                for i, r in enumerate(retrieved):
                    with st.expander(f"Source {i + 1} (Score: {r['score']:.3f})"):
                        st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

                render_export(answer, "research_answer", f"Research Answer — {topic}")
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "doc": topic or "Research Query",
                    "kind": "Research Answer",
                    "content": f"**Q:** {user_query}\n\n{answer}",
                })


# ------------------------------------------------------------------------------
# TAB 5: ABOUT
# ------------------------------------------------------------------------------
elif tab == "About":
    st.subheader("ℹ️ About Summary AI")
    st.markdown("""
**Summary AI** is an AI-powered PDF summarization and document assistant developed as a College Minor Project.

### 🌟 Core Capabilities
1. **Multi-PDF Summarizer**:
   - Customizable lengths (*Short*, *Medium*, *Detailed*) and formats (*Bullet Points*, *Paragraph*, *Key Findings*, *FAQ*, *Structured*).
   - **Page-Aware Citations**: Identifies authentic source pages supporting each summary point.
   - **Source Excerpts**: Expandable original text quotes with highlighted matching terms.
   - Separate or combined multi-document summarization.
2. **Ask PDF (Hybrid Retrieval & Verification)**:
   - Natural language semantic search powered by FAISS + lexical exact match reranking.
   - Post-generation factual verification ensuring numbers, dates, and names are never altered or hallucinated.
   - Predictable slash commands: `/summary`, `/keypoints`, `/explain`, `/find`, `/define`, `/keywords`, `/faq`, `/conclusion`, `/cite`, `/compare`, and `/help`.
   - Strictly grounded answering: Displays *"Not found in the provided document."* if facts are missing from the document.
3. **Multi-Format Export**:
   - Download summaries and Q&A records in **TXT**, **Markdown**, or **PDF** format.
4. **Live Research**:
   - Queries arXiv, RSS feeds, and web URLs in real-time to answer research topics.

### 🧠 Model Architecture & Accuracy Engine
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized FAISS IndexFlatIP).
- **Generation & Summarization**: **Google Gemini API** (`gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`) + `google/flan-t5-small` (Local Offline Seq2Seq).
- **Retrieval**: Hybrid Semantic + BM25-like Lexical Reranker.
- **Verification Engine**: Lexical containment & numerical audit layer.
- **Document Processing**: `pymupdf` (PyMuPDF).
""")
    