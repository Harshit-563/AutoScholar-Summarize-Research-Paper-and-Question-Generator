from typing import Dict, List
from rouge_score import rouge_scorer
import sacrebleu

def rouge_scores(pred: str, ref: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: v.fmeasure for k, v in scores.items()}

def bleu_score(preds: List[str], refs: List[str]) -> float:
    return sacrebleu.corpus_bleu(preds, [refs]).score
