from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
import nltk

# Ensure required NLTK data is available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)


# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def _split_into_sentences(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

class QuestionGenerator:
    """Highlight-based QG using T5 model `valhalla/t5-base-qg-hl`"""
    def __init__(self, model_name: str = "valhalla/t5-base-qg-hl", device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, text: str, max_questions: int = 10) -> List[Dict[str, str]]:
        sents = _split_into_sentences(text)
        picks = [s for s in sents if len(s.split()) > 8][:max_questions]

        qa_list: List[Dict[str, str]] = []
        # Use only the summary context for shorter inputs
        context = text
        for s in picks:
            source_text = f"generate questions: {context.replace(s, '<hl> ' + s + ' <hl>')}"
            inputs = self.tokenizer(source_text, return_tensors="pt", truncation=True, max_length=768).to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True
            )
            q = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Try answer generation (heuristic)
            ans_inp = self.tokenizer(f"answer: {s}", return_tensors="pt", truncation=True, max_length=256).to(self.device)
            ans_out = self.model.generate(**ans_inp, max_new_tokens=32, num_beams=4, early_stopping=True)
            a = self.tokenizer.decode(ans_out[0], skip_special_tokens=True)
            qa_list.append({"question": q, "answer": a})
        return qa_list
