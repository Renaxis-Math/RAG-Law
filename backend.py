# backend.py

# %% [markdown]
# # Imports and Environment

# %%
from dotenv import load_dotenv
import os
import glob
from langchain_postgres.vectorstores import PGVector

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.load import dumps, loads
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from typing_extensions import TypedDict
from typing import List
import asyncio

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# %%
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
DB_CONNECTION     = os.getenv("DB_CONNECTION")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Insurance-Law-RAG"
from langsmith import trace

# %% [markdown]
# # Connect to Database

# %%
shared_connection_string = DB_CONNECTION
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if not all([OPENAI_API_KEY, shared_connection_string, TAVILY_API_KEY, LANGSMITH_API_KEY]):
    print("Error: Missing one or more required environment variables")
else:
    print("All environment variables loaded successfully")

insurance_vector_store = PGVector(
    embeddings=embeddings,
    collection_name="insurance_code",
    connection=shared_connection_string,
    use_jsonb=True,
)

# %% [markdown]
# # PDF Ingestion & Chunking

# %%
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_pdf(path: str, source: str) -> List[Document]:
    docs: List[Document] = []
    try:
        loader = PyMuPDFLoader(file_path=path, mode="page")
        pages = loader.load()
        for pg in pages:
            pg.metadata = pg.metadata or {}
            pg.metadata["source"] = source
            docs.append(pg)
    except Exception as e:
        print(f"Error parsing {path}: {e}")
    return docs

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

all_pages: List[Document] = []
for pdf in pdf_files:
    name = os.path.splitext(os.path.basename(pdf))[0]
    print(f"Loading {pdf} as '{name}' …")
    all_pages.extend(parse_pdf(pdf, name))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks: List[Document] = []
for page in all_pages:
    for txt in splitter.split_text(page.page_content):
        chunks.append(Document(page_content=txt, metadata=page.metadata.copy()))

if chunks:
    try:
        insurance_vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to 'insurance_code' vector store.")
    except Exception as e:
        print("Error adding chunks:", e)
else:
    print("No PDF chunks found; nothing ingested.")

# %% [markdown]
# # Prompt Templates

# %%
vectorstore_summary = """
California Insurance Code: statutes, regulations, definitions,
policy requirements, and administrative procedures.
"""

scope_definition = """Questions about the California Insurance Code, including statutes,
regulations, definitions, policy requirements, and compliance procedures."""

router_template = PromptTemplate.from_template("""
You are a legal AI specializing in California insurance law.
Analyze the user's question and select exactly one datasource:
1) Vectorstore – local Insurance Code docs ({vectorstore_summary}).
2) Websearch – external case law, commentary, updates.
3) Chitter-Chatter – off-topic or casual queries.

Return JSON: {{"Datasource":"<Vectorstore|Websearch|Chitter-Chatter>"}}
""")

multi_query_template = PromptTemplate.from_template("""
Rewrite the legal question to improve vector retrieval.
Context summary: {vectorstore_summary}
Original question: {question}

Return {num_queries} distinct rewrites, one per line.
""")

relevance_template = PromptTemplate.from_template("""
Assess if the document is relevant to the question.
Document: {document}
Question: {question}

Return JSON: {{"binary_score":"pass" or "fail"}}
""")

answer_template = PromptTemplate.from_template("""
Use the following context to answer the question:
Context: {context}
Question: {question}

Provide a concise legal answer grounded in the context. Cite sources.
""")

hallucination_template = PromptTemplate.from_template("""
Check if the answer is grounded in the provided context.
Context: {documents}
Answer: {generation}

Return JSON: {{"binary_score":"pass" or "fail","explanation":"…"}}
""")

verification_template = PromptTemplate.from_template("""
Verify the answer addresses the question.
Question: {question}
Answer: {generation}

Return JSON: {{"binary_score":"pass" or "fail","explanation":"…"}}
""")

rewrite_template = PromptTemplate.from_template("""
Rewrite the question to improve retrieval.
Original: {question}
Previous answer: {generation}
Context summary: {vectorstore_summary}

Return JSON: {{"rewritten_question":"…","explanation":"…"}}
""")

chitchat_template = PromptTemplate.from_template("""
You are a friendly assistant. Stay on-topic about California insurance law ({scope_definition}).
For off-topic queries, gently redirect back to insurance topics.
""")

# %% [markdown]
# # Workflow Graph

# %%
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

llm_primary = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)
llm_fast    = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
web_tool    = TavilySearchResults(max_results=5, search_depth="advanced", include_answer=True, tavily_api_key=TAVILY_API_KEY)

router_prompt = router_template.format(vectorstore_summary=vectorstore_summary)

multi_chain = (
    multi_query_template
    | llm_primary
    | StrOutputParser()
    | (lambda lines: lines.split("\n"))
)
def rrf(results, k=60):
    scores = {}
    for docs in results:
        for i, d in enumerate(docs):
            key = dumps(d)
            scores[key] = scores.get(key, 0) + 1/(i+1 + k)
    return [loads(doc_str) for doc_str, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

retriever = (
    multi_chain
    | insurance_vector_store.as_retriever(search_type="mmr", search_kwargs={"k":3,"fetch_k":15,"lambda_mult":0.5}).map()
    | rrf
)

MAX_HALL = 1
MAX_VERIFY = 1

# %% [markdown]
# # Node Definitions

# %%
def init_and_route(state):
    if "original" not in state:
        state["original"] = state["question"]
        state["hall_attempts"] = 0
        state["verify_attempts"] = 0
    out = llm_primary.with_structured_output(method="json_mode").invoke(
        [SystemMessage(router_prompt), HumanMessage(state["question"])]
    )
    return out["Datasource"]

def document_retriever(state):
    docs = retriever.invoke({
        "question": state["question"],
        "num_queries": 3,
        "vectorstore_summary": vectorstore_summary
    })
    return {"documents": docs, "doc_checker": None}

async def grade_docs(state):
    tasks = [
        llm_fast.with_structured_output(method="json_mode").ainvoke(
            relevance_template.format(document=d, question=state["question"])
        )
        for d in state["documents"]
    ]
    results = await asyncio.gather(*tasks)
    passed = [d for d, r in zip(state["documents"], results) if r["binary_score"].lower()=="pass"]
    fail_count = len(state["documents"]) - len(passed)
    checker = "fail" if (fail_count/len(state["documents"]))>=0.5 else "pass"
    return {"documents": passed, "doc_checker": checker}

def web_search_node(state):
    hits = web_tool.invoke(state["question"])
    pages = [{"metadata":{"title":r["title"],"url":r["url"]},"page_content":r["content"]} for r in hits]
    return {"documents": state.get("documents", []) + pages}

def gen_answer(state):
    q = state.get("original", state["question"])
    prompt = answer_template.format(context=state["documents"], question=q)
    resp = llm_primary.invoke(prompt)
    return {"generation": resp.content}

def hallucination_checker(state):
    out = llm_fast.with_structured_output(method="json_mode").invoke(
        hallucination_template.format(documents=state["documents"], generation=state["generation"])
    )
    return {"hall_checker": out["binary_score"].lower()}

def answer_verifier(state):
    q = state.get("original", state["question"])
    out = llm_fast.with_structured_output(method="json_mode").invoke(
        verification_template.format(question=q, generation=state["generation"])
    )
    return {"verify_checker": out["binary_score"].lower()}

def rewrite_query(state):
    out = llm_primary.with_structured_output(method="json_mode").invoke(
        rewrite_template.format(
            question=state["question"],
            generation=state["generation"],
            vectorstore_summary=vectorstore_summary
        )
    )
    return {"question": out["rewritten_question"]}

def chitchat_node(state):
    resp = llm_fast.invoke([
        SystemMessage(chitchat_template.format(scope_definition=scope_definition)),
        HumanMessage(state["question"])
    ])
    return {"generation": resp.content}

def decide_after_docs(state):
    return "Websearch" if state["doc_checker"]=="fail" else "Generate"

def decide_after_hall(state):
    if state["hall_checker"]=="pass":
        return "AnswerVerifier"
    if state["hall_attempts"]>=MAX_HALL:
        return "Chitter"
    return "HallFail"

def decide_after_verify(state):
    if state["verify_checker"]=="pass":
        return "__end__"
    if state["verify_attempts"]>=MAX_VERIFY:
        return "Chitter"
    return "VerFail"

# %% [markdown]
# # Assemble Workflow

# %%
workflow = StateGraph(GraphState)

workflow.add_node("Router", init_and_route)
workflow.add_node("Websearch", web_search_node)
workflow.add_node("Retriever", document_retriever)
workflow.add_node("RelevanceGrader", grade_docs)
workflow.add_node("Generate", gen_answer)
workflow.add_node("HallucinationChecker", hallucination_checker)
workflow.add_node("AnswerVerifier", answer_verifier)
workflow.add_node("Rewrite", rewrite_query)
workflow.add_node("Chitter", chitchat_node)
workflow.add_node("HallFail", lambda s: {"hall_attempts": s["hall_attempts"] + 1})
workflow.add_node("VerFail", lambda s: {"verify_attempts": s["verify_attempts"] + 1})

workflow.set_conditional_entry_point(
    init_and_route,
    {
        "Vectorstore": "Retriever",
        "Websearch":   "Websearch",
        "Chitter-Chatter": "Chitter",
    },
)

workflow.add_edge("Retriever", "RelevanceGrader")
workflow.add_conditional_edges(
    "RelevanceGrader",
    decide_after_docs,
    {"Websearch": "Websearch", "Generate": "Generate"},
)

workflow.add_edge("Websearch", "Generate")
workflow.add_edge("Generate", "HallucinationChecker")
workflow.add_conditional_edges(
    "HallucinationChecker",
    decide_after_hall,
    {"AnswerVerifier": "AnswerVerifier", "HallFail": "HallFail", "Chitter": "Chitter"},
)
workflow.add_edge("HallFail", "Rewrite")
workflow.add_conditional_edges(
    "AnswerVerifier",
    decide_after_verify,
    {"__end__": END, "VerFail": "VerFail", "Chitter": "Chitter"},
)
workflow.add_edge("VerFail", "Rewrite")
workflow.add_edge("Rewrite", "Retriever")
workflow.add_edge("Chitter", END)

def get_workflow():
    return workflow