# Summary AI – AI-Powered PDF Summarization & Document Assistant

Summary AI is an intelligent, high-accuracy document intelligence assistant built with Python and Streamlit. It leverages modern vector retrieval (FAISS), dense embeddings (MiniLM), PyMuPDF text processing, OCR capabilities, and dual AI generation (Google Gemini API & Local FLAN-T5) to deliver grounded summaries, precise Q&A, and citation-backed document research.

---

## 🚀 Key Features

- **Document Extraction & Cleaning**: Fast, layout-aware PDF text extraction with PyMuPDF with clean paragraph preservation and automated memory management.
- **OCR Support**: Detects scanned pages or embedded images and extracts text using Tesseract OCR.
- **Vector Search & Grounded Retrieval**: Chunks text semantically, generates embeddings via `sentence-transformers/all-MiniLM-L6-v2`, and retrieves relevant context with `faiss-cpu`.
- **Dual AI Generation Modes**:
  - **Google Gemini API**: High-speed, nuanced summaries and contextual answers with strict anti-hallucination prompt engineering.
  - **FLAN-T5 Local Model**: Completely offline, private processing without external API calls.
- **Ask PDF with Verifiable Citations**: Returns answers with exact page numbers and chunk references; reliably refuses when information is not present in the document.
- **Multi-Format Export**: Export summaries and research reports to Markdown, Word (`.docx`), and PowerPoint (`.pptx`).
- **Research Assistant**: Integrated ArXiv literature queries and academic context enrichment.

---

## 🛠️ Architecture & Tech Stack

```text
PDF Upload
  ↓
PyMuPDF Text Extraction / Tesseract OCR
  ↓
Text Normalization & Semantic Chunking
  ↓
Sentence-Transformers (all-MiniLM-L6-v2)
  ↓
FAISS Vector Index (faiss-cpu)
  ↓
Similarity-Grounded Context Retrieval
  ↓
Google Gemini 2.5 / Local FLAN-T5
  ↓
Summaries / Ask PDF / Citations / Keyword Analysis
```

- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: `faiss-cpu`
- **Document Processing**: `PyMuPDF` (fitz), `pytesseract`
- **AI Models**: Google GenAI (`gemini-2.5-flash`), HuggingFace (`google/flan-t5-base`)

---

## 📦 Local Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yogeshkandoriya/Summary-AI.git
   cd Summary-AI
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Gemini API Key**:
   Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **Create app** > **Yup, I have an app**.
3. Select your repository:
   - **Repository**: `yogeshkandoriya/Summary-AI`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Expand **Advanced settings** and paste your API key in **Secrets**:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
5. Click **Deploy!**

---

## 📄 License
MIT License. Free for academic, personal, and research use.
