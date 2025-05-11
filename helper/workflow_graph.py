from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from langchain.load import dumps, loads

# Hyperparameter for Reciprocal Rank Fusion
RRF_K = 60  # higher = smoother aggregation; range = [0, ∞)

# Max allowable hallucination attempts
MAX_HALL   = 1  # higher = more attempts; range = [0, ∞)
# Max allowable verification attempts
MAX_VERIFY = 1  # higher = more attempts; range = [0, ∞)

class GraphState(TypedDict):
    """
    Schema for the workflow graph's state dictionary.
    """
    question: str
    original: str
    generation: str
    datasource: str
    hall_attempts: int
    verify_attempts: int
    documents: List[Document]
    doc_checker: str
    hall_checker: str
    verify_checker: str

def reciprocal_rank_fusion(results, k: int = RRF_K):
    """
    Merge multiple retrieval lists with Reciprocal Rank Fusion.
    """
    scores = {}
    for docs in results:
        for i, doc in enumerate(docs):
            key = dumps(doc)
            scores[key] = scores.get(key, 0) + 1/(i+1 + k)
    return [loads(doc_str) for doc_str, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
