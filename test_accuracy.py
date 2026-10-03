# test_accuracy.py
# Comprehensive accuracy test suite validating the 12 required test cases
import sys
import types

# Stub streamlit for isolated execution
st_stub = types.ModuleType("streamlit")
st_stub.session_state = types.SimpleNamespace(docs={}, history=[], faiss_cache={}, debug_mode=False)
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

# Create mock documents with known facts, numbers, dates, definitions, across multiple pages
mock_doc1 = {
    "full_text": (
        "Page 1 content: Deep learning architectures have evolved rapidly. "
        "In 2017, the Transformer architecture was introduced by Vaswani et al. "
        "A Transformer is defined as a neural network architecture based entirely on self-attention mechanisms. "
        "\n\n"
        "Page 2 content: The experimental evaluation was conducted on large benchmark datasets. "
        "The model achieved an accuracy of 94.5% on the test set, outperforming the baseline by 3.2%. "
        "The total training time took 48 hours across 8 GPUs. "
        "\n\n"
        "Page 3 content: In conclusion, the self-attention mechanism enables superior parallelization. "
        "However, quadratic memory complexity remains a key limitation for long sequences."
    ),
    "pages": [
        "Deep learning architectures have evolved rapidly. In 2017, the Transformer architecture was introduced by Vaswani et al. A Transformer is defined as a neural network architecture based entirely on self-attention mechanisms.",
        "The experimental evaluation was conducted on large benchmark datasets. The model achieved an accuracy of 94.5% on the test set, outperforming the baseline by 3.2%. The total training time took 48 hours across 8 GPUs.",
        "In conclusion, the self-attention mechanism enables superior parallelization. However, quadratic memory complexity remains a key limitation for long sequences."
    ],
    "page_count": 3,
    "word_count": 85,
    "char_count": 550,
    "file_size": 1024,
    "added_at": "12:00",
}

mock_doc2 = {
    "full_text": (
        "Convolutional Neural Networks (CNNs) specialize in grid-like image processing. "
        "The objective is feature map extraction using convolution filters. "
        "The method uses 3x3 convolution kernels with batch normalization. "
        "The accuracy reached 89.1% on ImageNet. "
        "The limitation is spatial invariance loss. "
        "In conclusion, CNNs remain the standard for computer vision."
    ),
    "pages": [
        "Convolutional Neural Networks (CNNs) specialize in grid-like image processing. The objective is feature map extraction using convolution filters.",
        "The method uses 3x3 convolution kernels with batch normalization. The accuracy reached 89.1% on ImageNet. The limitation is spatial invariance loss. In conclusion, CNNs remain the standard for computer vision."
    ],
    "page_count": 2,
    "word_count": 55,
    "char_count": 380,
    "file_size": 1024,
    "added_at": "12:01",
}

st_docs = {
    "transformer_paper.pdf": mock_doc1,
    "cnn_paper.pdf": mock_doc2,
}

emb_model = app.emb_model
gen_model = app.gen_model
tokenizer = app.tokenizer

passed = 0
failed = 0

def run_test(name, condition, details=""):
    global passed, failed
    if condition:
        print(f"PASS [{name}] {details}")
        passed += 1
    else:
        print(f"FAIL [{name}] {details}")
        failed += 1

print("\n--- RUNNING ACCURACY TEST SUITE ---\n")

# TEST 1: Direct factual question
ans1, cits1, dbg1 = app.answer_with_citations("Who introduced the Transformer architecture?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 1: Direct factual question", "vaswani" in ans1.lower() or any("vaswani" in c["excerpt"].lower() for c in cits1), f"Ans: {ans1[:70]}")

# TEST 2: Definition question
ans2, cits2, dbg2 = app.answer_with_citations("Define Transformer", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 2: Definition question", "neural network" in ans2.lower() or "self-attention" in ans2.lower(), f"Ans: {ans2[:70]}")

# TEST 3: Explanation question
ans3, cits3, dbg3 = app.answer_with_citations("Why does self-attention enable superior parallelization?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 3: Explanation question", len(ans3) > 10 and not app.is_refusal(ans3), f"Ans: {ans3[:70]}")

# TEST 4: Numerical question (Ensuring 94.5% is preserved and never altered)
ans4, cits4, dbg4 = app.answer_with_citations("What was the accuracy on the test set?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 4: Numerical question (94.5%)", "94.5%" in ans4 or "94.5" in ans4, f"Ans: {ans4}")

# TEST 5: Date/year question (Ensuring 2017 is preserved)
ans5, cits5, dbg5 = app.answer_with_citations("In what year was the Transformer introduced?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 5: Date/year question (2017)", "2017" in ans5, f"Ans: {ans5}")

# TEST 6: Multi-page retrieval
# Query about architecture (page 1) and limitation (page 3)
ans6, cits6, dbg6 = app.answer_with_citations("What is the quadratic memory complexity limitation of the architecture?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 6: Multi-page retrieval", "quadratic" in ans6.lower() or any(c["page"] == 3 for c in cits6), f"Citations pages: {[c['page'] for c in cits6]}")

# TEST 7: Question whose answer is NOT in the PDF (Population of Japan)
ans7, cits7, dbg7 = app.answer_with_citations("What is the population of Japan?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 7: Out-of-document refusal", app.is_refusal(ans7), f"Ans: {ans7}")

# TEST 8: /find command
t8, c8, p8, dbg8 = app.run_command("/find", "Transformer", "transformer_paper.pdf", st_docs, emb_model, gen_model, tokenizer)
run_test("TEST 8: /find command", len(p8) > 0 and p8[0]["page"] in [1, 2, 3], f"Matches count: {len(p8)}")

# TEST 9: /define command
t9, c9, p9, dbg9 = app.run_command("/define", "Transformer", "transformer_paper.pdf", st_docs, emb_model, gen_model, tokenizer)
run_test("TEST 9: /define command", "Definition" in c9 and "Source: Page" in c9, f"Output snippet: {c9[:80]}")

# TEST 10: /cite command
t10, c10, p10, dbg10 = app.run_command("/cite", "accuracy", "transformer_paper.pdf", st_docs, emb_model, gen_model, tokenizer)
run_test("TEST 10: /cite command", "94.5%" in c10 and "Page" in c10, f"Output snippet: {c10[:80]}")

# TEST 11: /compare command with two PDFs
t11, c11, p11, dbg11 = app.run_command("/compare", "transformer_paper.pdf cnn_paper.pdf", "transformer_paper.pdf", st_docs, emb_model, gen_model, tokenizer)
run_test("TEST 11: /compare command", "Document 1" in c11 and "Document 2" in c11 and "Similarities" in c11, f"Comparison generated")

# TEST 12: Consistency test (same question, different phrasing)
ans12a, _, _ = app.answer_with_citations("What was the test accuracy?", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
ans12b, _, _ = app.answer_with_citations("Tell me the accuracy achieved on the test dataset.", mock_doc1, emb_model, gen_model, tokenizer, doc_name="transformer_paper.pdf")
run_test("TEST 12: Paraphrase consistency", ("94.5" in ans12a) and ("94.5" in ans12b), f"AnsA: '{ans12a}' | AnsB: '{ans12b}'")

print(f"\n==========================================")
print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
print(f"==========================================\n")
sys.exit(0 if failed == 0 else 1)
