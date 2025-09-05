from typing import Dict, Any
from .summarizer import Summarizer
from .qg import QuestionGenerator

class SummarizeAndQG:
    def __init__(self, sum_model_name: str = "google/pegasus-arxiv", qg_model_name: str = "valhalla/t5-base-qg-hl",
                 max_input_tokens: int = 1024, chunk_overlap: int = 128, max_summary_tokens: int = 256,
                 do_meta_summary: bool = True):
        self.summarizer = Summarizer(model_name=sum_model_name, max_input_tokens=max_input_tokens,
                                     chunk_overlap=chunk_overlap, max_summary_tokens=max_summary_tokens)
        self.qg = QuestionGenerator(model_name=qg_model_name)
        self.do_meta = do_meta_summary

    def process_text(self, text: str) -> Dict[str, Any]:
        summary = self.summarizer.summarize(text, meta_summary=self.do_meta)
        questions = self.qg.generate(summary, max_questions=10)
        return {
            "summary": summary,
            "questions": questions
        }
