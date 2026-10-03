# test_ocr_suite.py
# Comprehensive automated test suite for OCR and scanned PDF support in Summary AI
import io
import os
import sys
import types
import pymupdf
from PIL import Image, ImageDraw, ImageFont

# Headless streamlit stub
st_stub = types.ModuleType("streamlit")
st_stub.session_state = types.SimpleNamespace(
    docs={},
    history=[],
    faiss_cache={},
    debug_mode=True,
    use_gemini=False,
    gemini_api_key="",
    gemini_model="gemini-1.5-flash",
    last_engine_used="Local FLAN-T5-small (Offline)",
    scanned_pdf_audit={}
)
for name in ("set_page_config", "title", "sidebar", "radio", "header", "write",
             "info", "warning", "error", "success", "markdown", "caption",
             "spinner", "button", "selectbox", "slider", "checkbox", "columns",
             "file_uploader", "text_input", "text_area", "multiselect",
             "expander", "download_button", "code", "rerun", "subheader", "stop"):
    setattr(st_stub, name, lambda *a, **k: None)

class _SidebarStub:
    def __getattr__(self, name):
        return lambda *a, **k: False if name == "checkbox" else "Ask PDF"

st_stub.sidebar = _SidebarStub()
st_stub.cache_resource = lambda f: f
sys.modules["streamlit"] = st_stub

import app

print("======================================================================")
print("SUMMARY AI — COMPREHENSIVE OCR & SCANNED DOCUMENT TEST SUITE")
print("======================================================================\n")

total_tests = 0
passed_tests = 0
failed_tests = 0

def check(test_name: str, condition: bool, details: str = ""):
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f"PASS: [{test_name}] {details}")
    else:
        failed_tests += 1
        print(f"FAIL: [{test_name}] {details}")

# -----------------------------------------------------------------------------
# Helper: Create Synthetic PDFs
# -----------------------------------------------------------------------------
def make_normal_pdf() -> bytes:
    doc = pymupdf.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 80), "Chapter 1: Foundations of Artificial Intelligence\n\nArtificial intelligence enables computers to learn from data and solve complex problems.", fontsize=14)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 80), "Chapter 2: Machine Learning Architectures\n\nDeep learning neural networks utilize backpropagation and gradient descent optimization.", fontsize=14)
    b = doc.tobytes()
    doc.close()
    return b

def make_scanned_pdf_with_text() -> bytes:
    # Create an image containing clear high-contrast text
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Using default font with clear large text
    text_content = "SCANNED DOCUMENT INVOICE\nCustomer: Global Research Institute\nTotal Amount: 7500 USD\nStatus: Verified and Approved"
    draw.text((40, 50), text_content, fill=(0, 0, 0))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    # Create PDF page with image only, no selectable text
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rect = pymupdf.Rect(50, 50, 545, 300)
    page.insert_image(rect, stream=img_bytes)
    b = doc.tobytes()
    doc.close()
    return b

def make_hybrid_pdf() -> bytes:
    # Image with distinct text
    img = Image.new("RGB", (700, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "FIGURE 1 DIAGRAM LABEL: Advanced Encryption Standard AES Key Expansion", fill=(0, 0, 0))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Native selectable text
    page.insert_text((50, 80), "Symmetric key cryptography uses the same key for encryption and decryption.", fontsize=14)
    # Insert image
    rect = pymupdf.Rect(50, 150, 545, 300)
    page.insert_image(rect, stream=img_bytes)
    b = doc.tobytes()
    doc.close()
    return b

def make_diagram_only_pdf() -> bytes:
    # Image with drawing/geometry only (no letters or text)
    img = Image.new("RGB", (600, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], outline=(0, 0, 255), width=4, fill=(180, 200, 255))
    draw.ellipse([300, 50, 500, 250], outline=(255, 0, 0), width=4, fill=(255, 200, 180))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    rect = pymupdf.Rect(50, 100, 545, 400)
    page.insert_image(rect, stream=img_bytes)
    b = doc.tobytes()
    doc.close()
    return b

class MockUpload:
    def __init__(self, name, data):
        self.name = name
        self.size = len(data)
        self._data = data
    def read(self):
        return self._data
    def getvalue(self):
        return self._data


# -----------------------------------------------------------------------------
# Test 1: OCR Availability & Model Assets
# -----------------------------------------------------------------------------
print("--- TEST 1: OCR ENVIRONMENT & TESSDATA ---")
check("is_ocr_available returns True", app.is_ocr_available(), f"TESSDATA_PREFIX={os.environ.get('TESSDATA_PREFIX')}")
check("tessdata directory exists", os.path.exists(app.TESSDATA_DIR), app.TESSDATA_DIR)
check("eng.traineddata exists", os.path.exists(os.path.join(app.TESSDATA_DIR, "eng.traineddata")), "Model file present")


# -----------------------------------------------------------------------------
# Test 2: Normal PDF (Selectable text, No OCR needed)
# -----------------------------------------------------------------------------
print("\n--- TEST 2: NORMAL SELECTABLE PDF ---")
normal_bytes = make_normal_pdf()
full_text, pages, audit = app.extract_text_with_ocr(normal_bytes)
check("Normal PDF extracted pages", len(pages) == 2, f"Got {len(pages)} pages")
check("Normal PDF contains page 1 text", "Foundations of Artificial Intelligence" in full_text)
check("Normal PDF contains page 2 text", "Machine Learning Architectures" in full_text)
check("Normal PDF ocr_applied is False", audit["ocr_applied"] is False)
check("Normal PDF is_scanned is False", audit["is_scanned"] is False)


# -----------------------------------------------------------------------------
# Test 3: Deduplication Functionality
# -----------------------------------------------------------------------------
print("\n--- TEST 3: TEXT DEDUPLICATION (REQUIREMENT 5) ---")
native_sample = "Symmetric encryption uses a single secret shared key.\nData integrity is protected using hash functions."
# OCR finds exact same lines plus one new line
ocr_sample = "Symmetric encryption uses a single secret shared key.\nData integrity is protected using hash functions.\nUnique OCR line: Cipher Block Chaining mode CBC."
combined = app.deduplicate_combine_texts(native_sample, ocr_sample)
check("Deduplication retains unique OCR line", "Cipher Block Chaining" in combined)
count_symmetric = combined.count("Symmetric encryption")
check("Deduplication prevents repeated lines", count_symmetric == 1, f"Found {count_symmetric} occurrences")


# -----------------------------------------------------------------------------
# Test 4: Scanned PDF Extraction with OCR
# -----------------------------------------------------------------------------
print("\n--- TEST 4: SCANNED PDF EXTRACTION WITH OCR ---")
scanned_bytes = make_scanned_pdf_with_text()
scanned_text, scanned_pages, scanned_audit = app.extract_text_with_ocr(scanned_bytes)
check("Scanned PDF audit detects is_scanned or ocr_applied", scanned_audit["ocr_applied"] or scanned_audit["is_scanned"], str(scanned_audit))
check("Scanned PDF page 1 extracted", len(scanned_pages) == 1, f"Got {len(scanned_pages)} pages")
has_ocr_words = any(w in scanned_text.upper() for w in ["INVOICE", "RESEARCH", "TOTAL", "AMOUNT", "7500", "STATUS", "VERIFIED"])
check("OCR extracted readable words from image", has_ocr_words, f"Extracted: '{scanned_text.strip()}'")


# -----------------------------------------------------------------------------
# Test 5: Hybrid PDF (Selectable text + Image text)
# -----------------------------------------------------------------------------
print("\n--- TEST 5: HYBRID PDF (TEXT + EMBEDDED IMAGE) ---")
hybrid_bytes = make_hybrid_pdf()
hybrid_text, hybrid_pages, hybrid_audit = app.extract_text_with_ocr(hybrid_bytes)
check("Hybrid PDF extracts native text", "Symmetric key cryptography" in hybrid_text)
check("Hybrid PDF total images detected >= 1", hybrid_audit["total_images_detected"] >= 1, f"Images: {hybrid_audit['total_images_detected']}")


# -----------------------------------------------------------------------------
# Test 6: Non-text Visual PDF (Diagrams / Charts / Shapes)
# -----------------------------------------------------------------------------
print("\n--- TEST 6: NON-TEXT VISUAL PDF (DIAGRAM ONLY) ---")
diagram_bytes = make_diagram_only_pdf()
diag_text, diag_pages, diag_audit = app.extract_text_with_ocr(diagram_bytes)
check("Diagram PDF total images detected == 1", diag_audit["total_images_detected"] == 1)
check("Diagram PDF flagged in ocr_failed_pages or empty_pages", (1 in diag_audit["ocr_failed_pages"]) or (1 in diag_audit["empty_pages"]), str(diag_audit))
check("OCR does not fabricate text from shapes", len(diag_text.strip()) < 30, f"Got text: '{diag_text.strip()}'")


# -----------------------------------------------------------------------------
# Test 7: Upload Pipeline & Session State Integration
# -----------------------------------------------------------------------------
print("\n--- TEST 7: UPLOAD PIPELINE INTEGRATION ---")
scanned_upload = MockUpload("scanned_invoice_sample.pdf", scanned_bytes)
app.add_uploaded_documents([scanned_upload])

doc_entry = st_stub.session_state.docs.get("scanned_invoice_sample.pdf")
if doc_entry:
    check("Document added to session_state.docs", True)
    check("OCR audit preserved in doc metadata", "ocr_audit" in doc_entry, str(doc_entry.keys()))
    check("Page count accurate", doc_entry["page_count"] == 1)
    check("Word count recorded", doc_entry["word_count"] > 0, f"Words: {doc_entry['word_count']}")
else:
    # If text extraction was completely empty it gets handled by scanned_pdf_audit
    check("Document handled in scanned_pdf_audit if textless", "scanned_invoice_sample.pdf" in st_stub.session_state.scanned_pdf_audit)


# -----------------------------------------------------------------------------
# Test 8: Backward Compatibility with extract_text_from_pdf
# -----------------------------------------------------------------------------
print("\n--- TEST 8: BACKWARD COMPATIBILITY ---")
normal_upload = MockUpload("normal_test.pdf", normal_bytes)
ret_val = app.extract_text_from_pdf(normal_upload)
check("extract_text_from_pdf returns 2-tuple", isinstance(ret_val, tuple) and len(ret_val) == 2)
check("Page texts match", len(ret_val[1]) == 2)


# -----------------------------------------------------------------------------
# Test 9: End-to-End FAISS Retrieval on OCR'd Document
# -----------------------------------------------------------------------------
print("\n--- TEST 9: END-TO-END RETRIEVAL ON OCR DOCUMENT ---")
if doc_entry and doc_entry["full_text"].strip():
    chunks, pages_map = app.build_doc_chunks(doc_entry)
    check("Chunks created from OCR text", len(chunks) >= 1, f"Total chunks: {len(chunks)}")
    faiss_idx, _ = app.build_faiss_index(chunks, app.emb_model)
    check("FAISS index built successfully", faiss_idx is not None and faiss_idx.ntotal == len(chunks))
    
    # Query retrieval via search_faiss
    results = app.search_faiss(
        index=faiss_idx,
        query="What is the invoice customer or total amount?",
        emb_model=app.emb_model,
        texts=chunks,
        k=1
    )
    check("FAISS retrieves matching OCR chunk", len(results) > 0, f"Retrieved: {len(results)} chunks")
    if results:
        check("Retrieved chunk contains invoice info", "7500" in results[0]["text"] or "Global Research" in results[0]["text"])

    # Hybrid retrieval via hybrid_retrieve
    h_chunks, _ = app.hybrid_retrieve(
        query="What is the total amount?",
        doc=doc_entry,
        emb_model=app.emb_model,
        doc_name="scanned_invoice_sample.pdf",
        top_k=2
    )
    check("Hybrid retrieval returns OCR chunk", len(h_chunks) > 0, f"Count: {len(h_chunks)}")
    if h_chunks:
        check("Source page is 1", h_chunks[0]["page"] == 1)

    # End-to-end Q&A generation via answer_with_citations
    ans, citations, _ = app.answer_with_citations(
        query="What is the total amount?",
        doc=doc_entry,
        emb_model=app.emb_model,
        gen_model=app.gen_model,
        tokenizer=app.tokenizer,
        doc_name="scanned_invoice_sample.pdf"
    )
    check("Q&A answer generated on OCR text", len(ans.strip()) > 0, f"Answer: '{ans.strip()}'")
    check("Citations include Page 1", any(c.get("page") == 1 for c in citations), f"Citations: {citations}")



print("\n======================================================================")
print(f"OCR TEST SUITE COMPLETE: {passed_tests} PASSED, {failed_tests} FAILED (TOTAL: {total_tests})")
print("======================================================================")

if failed_tests > 0:
    sys.exit(1)
sys.exit(0)
