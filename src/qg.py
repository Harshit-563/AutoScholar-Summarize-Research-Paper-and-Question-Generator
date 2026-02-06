from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# --- 1. Robust Resource Loading ---
def ensure_nltk_resources():
    # We list both the old and new names to be safe across NLTK versions
    resources = [
        "punkt", 
        "punkt_tab",
        "averaged_perceptron_tagger_eng", 
        "averaged_perceptron_tagger"
    ]
    
    print("Checking NLTK resources...")
    for resource in resources:
        try:
            # Check if it exists locally
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                # If not found, download it
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
    print("NLTK resources ready.")

# Call it immediately
ensure_nltk_resources()

class QuestionGenerator:
    """
    Improved QG using `valhalla/t5-base-qg-hl`.
    Logic: Extract Noun Phrases -> Highlight them -> Generate Question.
    """
    def __init__(self, model_name: str = "valhalla/t5-base-qg-hl", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _get_candidate_answers(self, sentence: str) -> List[str]:
        """
        Extracts nouns/noun phrases to serve as the 'answer' input for the model.
        Highlighting the whole sentence usually leads to vague questions.
        """
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        
        # Extract Nouns (NN, NNS, NNP, NNPS) as candidate answers
        # You can improve this with Spacy for full Named Entity Recognition (NER)
        candidates = [word for word, pos in tagged if pos.startswith('NN')]
        
        # Filter: Only keep candidates longer than 2 chars to avoid noise
        return list(set([c for c in candidates if len(c) > 2]))

    def generate(self, text: str, max_questions: int = 10) -> List[Dict[str, str]]:
        sentences = sent_tokenize(text)
        qa_list: List[Dict[str, str]] = []

        count = 0
        for sentence in sentences:
            if count >= max_questions:
                break
            
            # 1. Find potential answers (Keywords) in the sentence
            candidates = self._get_candidate_answers(sentence)
            
            # If no nouns found, skip (or fallback to whole sentence if you prefer)
            if not candidates:
                continue

            # Limit to 1 best candidate per sentence to ensure variety
            # (Or iterate over all candidates if you want exhaustive questions)
            answer_text = candidates[0] 

            # 2. Prepare Input: <hl> answer <hl> context
            # We use the single sentence as context for speed, or the whole text for accuracy.
            # Using the single sentence is faster and reduces hallucination.
            input_text = f"generate question: {sentence.replace(answer_text, f'<hl> {answer_text} <hl>')}"

            # 3. Tokenize
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)

            # 4. Generate Question
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

            question = self.tokenizer.decode(out[0], skip_special_tokens=True)

            qa_list.append({
                "question": question,
                "answer": answer_text,
                "context": sentence # Helpful to know where it came from
            })
            count += 1

        return qa_list

