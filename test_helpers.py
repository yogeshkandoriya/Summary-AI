# Test the pure helper functions in app.py without loading models or streamlit UI.
import sys
import types

# --- Stub streamlit so module-level UI code is importable ---
st_stub = types.ModuleType("streamlit")
st_stub.session_state = types.SimpleNamespace(docs={}, history=[], _hooked=[])

def _noop(*a, **k):
    return None

for name in ("set_page_config", "title", "sidebar", "radio", "header", "write",
             "info", "warning", "error", "success", "markdown", "caption",
             "spinner", "button", "selectbox", "slider", "checkbox", "columns",
             "file_uploader", "text_input", "text_area", "multiselect",
             "expand", "download_button", "code", "rerun", "subheader", "stop"):
    setattr(st_stub, name, _noop)

st_stub.sidebar = types.SimpleNamespace(radio=_noop)
sys.modules["streamlit"] = st_stub

# --- Load only the definitions (skip execution of the model-loading + UI section) ---
src = open(r"C:/BOT/app.py", encoding="utf-8").read()
# Split after the imports so heavy libs (torch, transformers, faiss, etc.) are NOT imported.
# We only want the pure functions that use stdlib + pymupdf.
header = "\n".join(line for line in src.splitlines()[:19])  # only stdlib imports

# Manually define the functions we want to test (copied verbatim from app.py) —
# actually better: exec the function bodies by extracting them via ast.
import ast

tree = ast.parse(src)
func_names = {"chunk_text", "extract_keywords", "find_topic_pages",
              "build_doc_chunks", "to_markdown", "content_to_pdf",
              "parse_command", "generate_text", "answer_from_context",
              "summarize_doc", "extract_text_from_pdf",
              "add_uploaded_documents", "remove_document", "doc_card",
              "render_stats_bar", "render_upload_area", "inject_css"}

def get_function_source(func_node):
    return ast.get_source_segment(src, func_node)

imported_names = {"re", "html", "Counter", "datetime", "pymupdf", "np", "sklearn_text"}
# Build minimal import shims
import re, html
from collections import Counter
from datetime import datetime
from typing import List, Tuple
import pymupdf
import numpy as np
import sys as _sys
_skl = types.ModuleType("sklearn.feature_extraction")
_skl.text = types.ModuleType("sklearn.feature_extraction.text")
_skl.text.ENGLISH_STOP_WORDS = frozenset({"the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", "with", "as", "at", "from"})
_sys.modules["sklearn"] = types.ModuleType("sklearn")
_sys.modules["sklearn.feature_extraction"] = _skl
_sys.modules["sklearn.feature_extraction.text"] = _skl.text

class _TfidfVectorizer:
    def __init__(self, *a, **k):
        pass
    def fit_transform(self, *a, **k):
        raise ValueError("empty vocabulary")  # force fallback path in test
_skl.text.TfidfVectorizer = _TfidfVectorizer
TfidfVectorizer = _TfidfVectorizer
sklearn_text = _skl.text

# exec only the function definitions we need
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in func_names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<app>", "exec"), globals())

# Also exec constants needed by summarize_doc
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("SUMMARY_LENGTHS", "SUMMARY_FORMATS", "SIMPLE_LANGUAGE_NOTE", "MAX_FILES", "MAX_FILE_MB") for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<app>", "exec"), globals())

# --- Tests ---
results = []

def check(name, cond):
    results.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), name)

# 1. chunk_text
c = chunk_text("One sentence here. " * 300, max_chars=1200, overlap=200)
check("chunk_text returns multiple chunks", len(c) > 1)
check("chunk_text chunks non-empty", all(x.strip() for x in c))

# 2. extract_keywords
kw = extract_keywords("Machine learning is a field. Neural networks are part of machine learning. "
                      "Deep learning uses neural networks. Machine learning models are powerful.", top_n=5)
check("extract_keywords returns terms", len(kw) > 0 and isinstance(kw[0], str))

# 3. find_topic_pages
doc = {
    "pages": [
        "Introduction to graphs and graph theory basics.",
        "Machine learning models for graph analysis are popular.",
        "Graph neural networks combine both topics well.",
    ],
    "full_text": " ".join(["Introduction to graphs.", "Machine learning models.", "Graph neural networks."]),
}
pages = find_topic_pages("machine learning", doc, top_n=3)
check("find_topic_pages finds page 2", any(p["page"] == 2 for p in pages))
check("find_topic_pages has excerpts", all(p["excerpt"] for p in pages))

# 4. build_doc_chunks page tracking
chunks, pnums = build_doc_chunks(doc, max_chars=30, overlap=5)
check("build_doc_chunks page numbers align", len(chunks) == len(pnums))
check("build_doc_chunks page1 present", any(p == 1 for p in pnums))

# 5. parse_command
check("parse /summary short", parse_command("/summary short") == ("/summary", "short"))
check("parse natural language", parse_command("What is the main idea?") == (None, "What is the main idea?"))
check("parse /find with topic", parse_command("/find machine learning") == ("/find", "machine learning"))

# 6. to_markdown
check("to_markdown contains title", "# Title\n\nbody\n" == to_markdown("Title", "body"))

# 7. content_to_pdf (uses pymupdf)
pdf = content_to_pdf("Test Title", "Hello world. This is a test paragraph with several words.")
check("content_to_pdf returns bytes", isinstance(pdf, bytes) and pdf.startswith(b"%PDF"))
check("content_to_pdf non-empty", len(pdf) > 500)

# 8. summarize_doc with stub generate_text
_calls = []
def _fake_generate(prompt, gen_model, tokenizer, max_length):
    _calls.append(len(prompt))
    return "- point one\n- point two"
globals()["generate_text"] = _fake_generate
summary, chunk_sums = summarize_doc({"full_text": "A. " * 400}, None, None,
                                    length="Medium (5 bullets)", fmt="Bullet Points")
check("summarize_doc returns summary", isinstance(summary, str) and len(summary) > 0)
check("summarize_doc chunk summaries non-empty", len(chunk_sums) > 0)

# 9. Phase 1 — document upload limits & scanned-PDF handling
class _FakeUpload:
    def __init__(self, name, size, text="Some real text here. " * 5, bad=False):
        self.name, self.size, self._text, self._bad = name, size, text, bad
    def getvalue(self):
        if self._bad:
            raise RuntimeError("corrupt file")
        return self._text.encode("utf-8")

st_stub.session_state.docs = {}
globals()["st"] = st_stub

# scanned/textless PDF (extract returns empty text) -> skipped
_orig_extract = globals()["extract_text_from_pdf"]
def _fake_extract(f):
    if f._bad:
        raise RuntimeError("corrupt file")
    return f._text, [f._text] if f._text.strip() else []
globals()["extract_text_from_pdf"] = _fake_extract

added = add_uploaded_documents([_FakeUpload("a.pdf", 1024)])
check("upload adds valid doc", added == ["a.pdf"] and "a.pdf" in st_stub.session_state.docs)

# duplicate -> not added twice
added = add_uploaded_documents([_FakeUpload("a.pdf", 1024)])
check("upload skips duplicates", added == [])

# empty-text (scanned) -> skipped with warning
added = add_uploaded_documents([_FakeUpload("scan.pdf", 1024, text="")])
check("upload skips scanned/textless PDFs", added == [] and "scan.pdf" not in st_stub.session_state.docs)

# corrupt file -> skipped
added = add_uploaded_documents([_FakeUpload("bad.pdf", 1024, bad=True)])
check("upload skips unreadable PDFs", added == [] and "bad.pdf" not in st_stub.session_state.docs)

# oversize file -> skipped
added = add_uploaded_documents([_FakeUpload("big.pdf", (MAX_FILE_MB + 1) * 1024 * 1024)])
check("upload enforces size limit", added == [] and "big.pdf" not in st_stub.session_state.docs)

# max files cap
for i in range(12):
    add_uploaded_documents([_FakeUpload(f"d{i}.pdf", 1024)])
check("upload enforces max files", len(st_stub.session_state.docs) <= MAX_FILES)

# remove
remove_document("a.pdf")
check("remove deletes doc", "a.pdf" not in st_stub.session_state.docs)

# stats bar renders (metrics called)
class _MetricStub:
    def __init__(self):
        pass
    def metric(self, label, value):
        _metrics.append((label, value))
_metrics = []
st_stub.columns = lambda *a, **k: [_MetricStub() for _ in range(4)]
render_stats_bar()
check("stats bar shows 4 metrics", len(_metrics) == 4)

failed = [n for n, ok in results if not ok]
print("TOTAL", len(results), "FAILED", len(failed))
sys.exit(1 if failed else 0)
