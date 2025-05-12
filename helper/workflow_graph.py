from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from langchain.load import dumps, loads

RRF_K = 60              # higher = smoother aggregation; range = [0, inf)
MAX_HALL   = 10         # higher = more attempts; range = [0, inf)
MAX_VERIFY = 10         # higher = more attempts; range = [0, inf)

class GraphState(TypedDict):
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
    scores = {}
    for docs in results:
        for i, doc in enumerate(docs):
            key = dumps(doc)
            scores[key] = scores.get(key, 0) + 1/(i+1 + k)
    return [loads(doc_str) for doc_str, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
