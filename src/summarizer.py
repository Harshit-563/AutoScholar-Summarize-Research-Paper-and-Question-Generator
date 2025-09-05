from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Summarizer:
    def __init__(self, model_name: str = "google/pegasus-arxiv", max_input_tokens: int = 1024, chunk_overlap: int = 128, max_summary_tokens: int = 256, device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.chunk_overlap = chunk_overlap
        self.max_summary_tokens = max_summary_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _chunk_tokens(self, input_ids: List[int]) -> List[List[int]]:
        max_len = self.max_input_tokens
        overlap = self.chunk_overlap
        chunks = []
        i = 0
        while i < len(input_ids):
            chunk = input_ids[i:i+max_len]
            chunks.append(chunk)
            if i + max_len >= len(input_ids):
                break
            i += max_len - overlap
        return chunks

    def summarize(self, text: str, meta_summary: bool = True) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        ids = inputs["input_ids"][0].tolist()

        chunks = self._chunk_tokens(ids)
        summaries: List[str] = []
        for ch in chunks:
            ch_tensor = torch.tensor([ch]).to(self.device)
            out = self.model.generate(
                input_ids=ch_tensor,
                max_new_tokens=self.max_summary_tokens,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            summ = self.tokenizer.decode(out[0], skip_special_tokens=True)
            summaries.append(summ)

        full_summary = " ".join(summaries)

        if meta_summary and len(summaries) > 1:
            inputs2 = self.tokenizer(full_summary, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.device)
            out2 = self.model.generate(
                **inputs2,
                max_new_tokens=self.max_summary_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            full_summary = self.tokenizer.decode(out2[0], skip_special_tokens=True)

        return full_summary.strip()
