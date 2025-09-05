import argparse, json, os
from pathlib import Path
from src.pdf_utils import extract_text_from_pdf
from src.pipeline import SummarizeAndQG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF file")
    ap.add_argument("--sum_model", default="google/pegasus-arxiv")
    ap.add_argument("--qg_model", default="valhalla/t5-base-qg-hl")
    ap.add_argument("--max_input_tokens", type=int, default=1024)
    ap.add_argument("--chunk_overlap", type=int, default=128)
    ap.add_argument("--max_summary_len", type=int, default=256)
    ap.add_argument("--no_meta_summary", action="store_true")
    args = ap.parse_args()

    text = extract_text_from_pdf(args.pdf)

    pipeline = SummarizeAndQG(
        sum_model_name=args.sum_model,
        qg_model_name=args.qg_model,
        max_input_tokens=args.max_input_tokens,
        chunk_overlap=args.chunk_overlap,
        max_summary_tokens=args.max_summary_len,
        do_meta_summary=not args.no_meta_summary
    )

    result = pipeline.process_text(text)

    os.makedirs("outputs", exist_ok=True)
    out = Path("outputs") / (Path(args.pdf).stem + "_result.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
