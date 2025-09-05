import streamlit as st
from src.pdf_utils import extract_text_from_pdf
from src.pipeline import SummarizeAndQG
from pathlib import Path
import json, os
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="AutoScholar", page_icon="📚", layout="wide")
st.title("📚 AutoScholar — Summarize Papers & Generate Questions")

with st.sidebar:
    st.header("Settings")
    sum_model = st.selectbox("Summarization model", ["google/pegasus-arxiv", "facebook/bart-large-cnn"])
    qg_model = st.selectbox("Question generation model", ["valhalla/t5-base-qg-hl"])
    max_input_tokens = st.number_input("Max input tokens per chunk", 256, 4096, 1024, step=128)
    chunk_overlap = st.number_input("Chunk overlap (tokens)", 0, 512, 128, step=16)
    max_summary_len = st.number_input("Max summary tokens", 32, 1024, 256, step=16)
    do_meta_summary = st.checkbox("Meta-summary over chunk summaries", value=True)

uploaded = st.file_uploader("Upload a research paper PDF", type=["pdf"])

if uploaded is not None:
    pdf_bytes = uploaded.read()
    tmp_pdf = Path("tmp_upload.pdf")
    tmp_pdf.write_bytes(pdf_bytes)

    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(str(tmp_pdf))

    st.subheader("Extracted Text (preview)")
    st.text_area("Raw text", text[:5000], height=200)

    pipeline = SummarizeAndQG(
        sum_model_name=sum_model,
        qg_model_name=qg_model,
        max_input_tokens=max_input_tokens,
        chunk_overlap=chunk_overlap,
        max_summary_tokens=max_summary_len,
        do_meta_summary=do_meta_summary
    )

    if st.button("Run Summarization + QG"):
        with st.spinner("Running models... this may take a while on first run."):
            result = pipeline.process_text(text)
        st.success("Done!")

        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Generated Questions")
        for i, q in enumerate(result["questions"], 1):
            st.markdown(f"**{i}. {q['question']}**")
            if q.get("answer"):
                with st.expander("Show answer"):
                    st.write(q["answer"])

        # Save
        os.makedirs("outputs", exist_ok=True)
        out_path = Path("outputs/output.json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        st.download_button("Download JSON", data=json.dumps(result, indent=2, ensure_ascii=False), file_name="autoscholar_output.json")

        # TXT export
        txt = "Summary\n-------\n" + result["summary"] + "\n\nQuestions\n---------\n"
        for i, q in enumerate(result["questions"], 1):
            txt += f"{i}. {q['question']}\n"
            if q.get("answer"):
                txt += f"   Answer: {q['answer']}\n"
        st.download_button("Download TXT", data=txt, file_name="autoscholar_output.txt")
