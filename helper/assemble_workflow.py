from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import os

from helper.workflow_graph import GraphState, reciprocal_rank_fusion
from helper.node_definitions import (
    create_multi_query_chain, create_retriever,
    init_and_route, document_retriever,
    web_search_node, gen_answer, hallucination_checker,
    answer_verifier, rewrite_query, chitchat_node,
    decide_after_docs, decide_after_hall, decide_after_verify,
    hall_fail_node, ver_fail_node, grade_docs_with_embeddings
)

OPENAI_PRIMARY_MODEL       = "gpt-4o"      # higher = more powerful; range = available OpenAI chat models
OPENAI_PRIMARY_TEMPERATURE = 0.3           # higher = more creative; range = [0.0, 1.0]
OPENAI_FAST_MODEL          = "gpt-4o-mini" # higher = more powerful; range = available OpenAI chat models
OPENAI_FAST_TEMPERATURE    = 0.3           # higher = more creative; range = [0.0, 1.0]
WEB_TOOL_MAX_RESULTS       = 5             # higher = fetch more results; range = [1, ∞)
WEB_TOOL_SEARCH_DEPTH      = "advanced"    # options = ["basic","advanced"]
WEB_TOOL_INCLUDE_ANSWER    = True          # boolean flag; range = [True, False]

def create_workflow(openai_api_key: str, tavily_api_key: str, vector_store):
    llm_primary = ChatOpenAI(
        model=OPENAI_PRIMARY_MODEL,
        temperature=OPENAI_PRIMARY_TEMPERATURE,
        api_key=openai_api_key
    )
    llm_fast = ChatOpenAI(
        model=OPENAI_FAST_MODEL,
        temperature=OPENAI_FAST_TEMPERATURE,
        api_key=openai_api_key
    )
    web_tool = TavilySearchResults(
        max_results=WEB_TOOL_MAX_RESULTS,
        search_depth=WEB_TOOL_SEARCH_DEPTH,
        include_answer=WEB_TOOL_INCLUDE_ANSWER,
        tavily_api_key=tavily_api_key
    )

    # Build the retriever pipeline (multi‐query -> MMR -> RRF)
    multi_chain    = create_multi_query_chain(llm_primary)
    retriever_chain= create_retriever(vector_store, multi_chain, reciprocal_rank_fusion)

    workflow = StateGraph(GraphState)

    router_fn      = lambda state: init_and_route(state, llm_primary)
    retriever_fn   = lambda state: document_retriever(state, retriever_chain)

    embedding_model = None
    
    if hasattr(vector_store, "embeddings"):
        embedding_model = vector_store.embeddings
    
    if embedding_model is None:
        print("Creating new embedding model instance")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    
    def relevance_fn(state):
        return grade_docs_with_embeddings(state, embedding_model)

    websearch_fn   = lambda state: web_search_node(state, web_tool)
    answer_fn      = lambda state: gen_answer(state, llm_primary)
    hall_check_fn  = lambda state: hallucination_checker(state, llm_fast)
    verify_fn      = lambda state: answer_verifier(state, llm_fast)
    rewrite_fn     = lambda state: rewrite_query(state, llm_primary)
    chitchat_fn    = lambda state: chitchat_node(state, llm_fast)

    workflow.add_node("Router", router_fn)
    workflow.add_node("Retriever", retriever_fn)
    workflow.add_node("RelevanceGrader", relevance_fn)        # now sync node
    workflow.add_node("Websearch", websearch_fn)
    workflow.add_node("Generate", answer_fn)
    workflow.add_node("HallucinationChecker", hall_check_fn)
    workflow.add_node("AnswerVerifier", verify_fn)
    workflow.add_node("Rewrite", rewrite_fn)
    workflow.add_node("Chitter", chitchat_fn)
    workflow.add_node("HallFail", hall_fail_node)
    workflow.add_node("VerFail", ver_fail_node)

    workflow.set_conditional_entry_point(
        router_fn,
        {
            "Vectorstore":    "Retriever",
            "Websearch":      "Websearch",
            "Chitter-Chatter":"Chitter",
        }
    )

    # Graph edges and decision logic
    workflow.add_edge("Retriever", "RelevanceGrader")
    workflow.add_conditional_edges(
        "RelevanceGrader",
        decide_after_docs,
        {"Websearch":"Websearch", "Generate":"Generate"}
    )
    workflow.add_edge("Websearch", "Generate")
    workflow.add_edge("Generate", "HallucinationChecker")
    workflow.add_conditional_edges(
        "HallucinationChecker",
        decide_after_hall,
        {"AnswerVerifier":"AnswerVerifier","HallFail":"HallFail","Chitter":"Chitter"}
    )
    workflow.add_edge("HallFail", "Rewrite")
    workflow.add_conditional_edges(
        "AnswerVerifier",
        decide_after_verify,
        {"__end__":END,"VerFail":"VerFail","Chitter":"Chitter"}
    )
    workflow.add_edge("VerFail", "Rewrite")
    workflow.add_edge("Rewrite", "Retriever")
    workflow.add_edge("Chitter", END)

    return workflow

def get_workflow(env_vars, vector_store):
    return create_workflow(
        openai_api_key=env_vars["OPENAI_API_KEY"],
        tavily_api_key=env_vars["TAVILY_API_KEY"],
        vector_store=vector_store
    )
