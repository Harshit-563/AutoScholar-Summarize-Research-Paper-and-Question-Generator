# AutoScholar: Research Paper Summarization & Question Generation

End-to-end pipeline to:
1) Extract text from research paper PDFs
2) Generate abstractive summaries
3) Generate study questions (WH + short answer + optional MCQ stub)
4) Evaluate summaries with ROUGE (and questions with BLEU if references are provided)
5) Run as a CLI **or** a Streamlit web app

---

## Features
- PDF parsing using **PyMuPDF** (fallback: pdfminer.six)
- Summarization: **PEGASUS** (`google/pegasus-arxiv`) or **BART** (`facebook/bart-large-cnn`)
- Question Generation: **T5** (`valhalla/t5-base-qg-hl`)
- Chunking for long documents with smart overlap
- Factuality-friendly settings (min/new tokens, beam search)
- Streamlit UI for demos

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### CLI (demo)

```bash
python scripts/demo.py --pdf sample_paper.pdf --sum_model google/pegasus-arxiv --qg_model valhalla/t5-base-qg-hl
```

Outputs will be saved in `outputs/` as JSON and TXT.

### Streamlit App

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal.

## Folder Structure

```
research_summarizer_qg/
├─ app.py
├─ requirements.txt
├─ README.md
├─ scripts/
│  └─ demo.py
├─ src/
│  ├─ pdf_utils.py
│  ├─ summarizer.py
│  ├─ qg.py
│  ├─ pipeline.py
│  └─ evaluate.py
└─ outputs/
```

## Notes
- First run will download Hugging Face models (internet required).
- For long PDFs, summarization runs per chunk; final summary concatenates chunk summaries and (optionally) a "meta-summary".
- Question generation highlights key sentences to steer T5.

## Citation
- PEGASUS: https://arxiv.org/abs/1912.08777
- FactPEGASUS: https://aclanthology.org/2022.naacl-main.74/
- T5 QG (baseline ideas): https://arxiv.org/abs/1910.10683
