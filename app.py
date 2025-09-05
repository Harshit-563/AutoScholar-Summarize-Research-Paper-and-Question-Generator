import streamlit as st
from pathlib import Path
import json, os, tempfile

from transformers import pipeline as hf_pipeline
from transformers import AutoTokenizer

from src.pdf_utils import extract_text_from_pdf
from src.pipeline import SummarizeAndQG


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="AutoScholar",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AutoScholar — Summarize Papers & Generate Questions")


# ---------------- MODEL LOADERS (CACHED) ----------------

@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return hf_pipeline("summarization", model=model_name, tokenizer=tokenizer)


@st.cache_resource(show_spinner="Loading QG pipeline...")
def load_pipeline(sum_model, qg_model, max_input_tokens, chunk_overlap, max_summary_len, do_meta_summary):
    return SummarizeAndQG(
        sum_model_name=sum_model,
        qg_model_name=qg_model,
        max_input_tokens=max_input_tokens,
        chunk_overlap=chunk_overlap,
        max_summary_tokens=max_summary_len,
        do_meta_summary=do_meta_summary
    )


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")

    sum_model = st.selectbox(
        "Summarization model",
        ["facebook/bart-large-cnn", "google/pegasus-arxiv"]
    )

    qg_model = st.selectbox(
        "Question generation model",
        ["valhalla/t5-base-qg-hl"]
    )

    max_input_tokens = st.number_input(
        "Max input tokens per chunk", 256, 4096, 1024, step=128
    )

    chunk_overlap = st.number_input(
        "Chunk overlap (tokens)", 0, 512, 128, step=16
    )

    max_summary_len = st.number_input(
        "Max summary tokens", 32, 1024, 256, step=16
    )

    do_meta_summary = st.checkbox(
        "Meta-summary over chunk summaries", value=True
    )


# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload a research paper PDF",
    type=["pdf"]
)

if uploaded:
    # Save PDF safely using temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "upload.pdf"
        pdf_path.write_bytes(uploaded.read())

        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(str(pdf_path))

    st.subheader("Extracted Text (preview)")
    st.text_area("Raw text", text[:5000], height=200)

    if st.button("Run Summarization + QG"):
        with st.spinner("Running models (first run may take time)..."):
            pipeline_obj = load_pipeline(
                sum_model,
                qg_model,
                max_input_tokens,
                chunk_overlap,
                max_summary_len,
                do_meta_summary
            )

            result = pipeline_obj.process_text(text)

        st.success("Done!")

        # -------- DISPLAY OUTPUT --------
        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Generated Questions")
        for i, q in enumerate(result["questions"], 1):
            st.markdown(f"**{i}. {q['question']}**")
            if q.get("answer"):
                with st.expander("Show answer"):
                    st.write(q["answer"])

        # -------- DOWNLOADS --------
        json_data = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name="autoscholar_output.json"
        )

        txt = "Summary\n-------\n" + result["summary"] + "\n\nQuestions\n---------\n"
        for i, q in enumerate(result["questions"], 1):
            txt += f"{i}. {q['question']}\n"
            if q.get("answer"):
                txt += f"   Answer: {q['answer']}\n"

        st.download_button(
            "Download TXT",
            data=txt,
            file_name="autoscholar_output.txt"
        )
