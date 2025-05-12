from langchain_core.messages import HumanMessage, SystemMessage
import numpy as np

from helper.prompt_templates import (
    router_template, multi_query_template, relevance_template,
    answer_template, hallucination_template, verification_template,
    rewrite_template, chitchat_template, vectorstore_summary, scope_definition
)
from helper.workflow_graph import MAX_HALL, MAX_VERIFY

RETRIEVER_SEARCH_K    = 5    # higher = consider more docs; range = [1, inf)
RETRIEVER_FETCH_K     = 20   # higher = fetch more docs;     range = [1, inf)
RETRIEVER_LAMBDA_MULT = 0.3  # higher = more diversity;     range = [0.0, 1.0]

def create_multi_query_chain(llm_primary):
    from langchain_core.output_parsers import StrOutputParser
    return (
        multi_query_template
        | llm_primary
        | StrOutputParser()
        | (lambda lines: lines.split("\n"))
    )

def create_retriever(vector_store, multi_chain, rrf_fn):
    return (
        multi_chain
        | vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_SEARCH_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA_MULT
            }
        ).map()
        | rrf_fn
    )

def init_and_route(state, llm_primary):
    if "original" not in state:
        state["original"] = state["question"]
        state["hall_attempts"]  = 0
        state["verify_attempts"] = 0

    out = llm_primary.with_structured_output(method="json_mode").invoke([
        SystemMessage(router_template.format(vectorstore_summary=vectorstore_summary)),
        HumanMessage(state["question"])
    ])
    return out["Datasource"]

def document_retriever(state, retriever):
    docs = retriever.invoke({
        "question": state["question"],
        "num_queries": 3,
        "vectorstore_summary": vectorstore_summary
    })
    return {"documents": docs, "doc_checker": None}

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

RELEVANCE_THRESHOLD = 0.9

def grade_docs_with_embeddings(state, embedding_model):
    question_text = state["question"]
    try:
        question_emb = embedding_model.embed_query(question_text)
    except Exception as e:
        raise RuntimeError(f"Failed to embed question: {e}")
    
    passed = []
    preds = []
    scores = []
    
    for doc in state["documents"]:
        doc_text = doc.page_content
        doc_emb = embedding_model.embed_query(doc_text)
            
        score = cosine_similarity(question_emb, doc_emb)
        scores.append(score)
        
        is_relevant = score > RELEVANCE_THRESHOLD
        preds.append(is_relevant)
        
        if is_relevant:
            passed.append(doc)
    
    # Handle empty results
    if not state["documents"] or len(scores) == 0:
        checker = "fail"
    else:
        fail_count = len(scores) - len(passed)
        checker = "fail" if (fail_count / len(scores)) >= 0.5 else "pass"
    
    state["relevance_scores"] = scores
    state["relevance_preds"] = preds
    return {"documents": passed, "doc_checker": checker}

def web_search_node(state, web_tool):
    hits  = web_tool.invoke(state["question"])
    pages = [{"metadata":{"title":r["title"],"url":r["url"]},"page_content":r["content"]} for r in hits]
    return {"documents": state.get("documents", []) + pages}

def gen_answer(state, llm_primary):
    q      = state.get("original", state["question"])
    prompt = answer_template.format(context=state["documents"], question=q)
    resp   = llm_primary.invoke(prompt)
    return {"generation": resp.content}

def hallucination_checker(state, llm_fast):
    out = llm_fast.with_structured_output(method="json_mode").invoke(
        hallucination_template.format(documents=state["documents"], generation=state["generation"])
    )
    return {"hall_checker": out["binary_score"].lower()}

def answer_verifier(state, llm_fast):
    q   = state.get("original", state["question"])
    out = llm_fast.with_structured_output(method="json_mode").invoke(
        verification_template.format(question=q, generation=state["generation"])
    )
    return {"verify_checker": out["binary_score"].lower()}

def rewrite_query(state, llm_primary):
    out = llm_primary.with_structured_output(method="json_mode").invoke(
        rewrite_template.format(
            question=state["question"],
            generation=state["generation"],
            vectorstore_summary=vectorstore_summary
        )
    )
    return {"question": out["rewritten_question"]}

def chitchat_node(state, llm_fast):
    resp = llm_fast.invoke([
        SystemMessage(chitchat_template.format(scope_definition=scope_definition)),
        HumanMessage(state["question"])
    ])
    return {"generation": resp.content}

def decide_after_docs(state):
    return "Websearch" if state["doc_checker"]=="fail" else "Generate"

def decide_after_hall(state):
    if state["hall_checker"] == "pass":
        return "AnswerVerifier"
    if state["hall_attempts"] >= MAX_HALL:
        return "Chitter"
    return "HallFail"

def decide_after_verify(state):
    if state["verify_checker"] == "pass":
        return "__end__"
    if state["verify_attempts"] >= MAX_VERIFY:
        return "Chitter"
    return "VerFail"

def hall_fail_node(state):
    return {"hall_attempts": state["hall_attempts"] + 1}

def ver_fail_node(state):
    return {"verify_attempts": state["verify_attempts"] + 1}
