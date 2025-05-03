# %% [markdown]
# # Imports and Environment

# %%
from dotenv import load_dotenv  
import os
from sqlalchemy.engine.url import make_url 
from langchain_postgres.vectorstores import PGVector

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

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

openai_api_key = os.getenv("OPENAI_API_KEY")
connection_string = os.getenv("DB_CONNECTION")
tavily_api_key = os.getenv("TAVILY_API_KEY")

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing for observability/debugging
os.environ["LANGCHAIN_TRACING"] = "true"
# Set the project name for LangSmith, it will create a new project if it doesn't exist
os.environ["LANGCHAIN_PROJECT"] = "GenAI-Class-Lab8-Assignment"

from langsmith import trace

# %% [markdown]
# # Connect to Database

# %%
shared_connection_string = make_url(connection_string)\
    .set(database="GenAI_Spring25_Hoang_Chu").render_as_string(hide_password=False)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

if not openai_api_key or not shared_connection_string or not tavily_api_key or not embedding_model:
    print("Error: Missing one or more required environment variables")
else:
    print("All environment variables loaded successfully")

book_data_vector_store = PGVector(
    embeddings=embedding_model,   
    collection_name="Book_data",
    connection=shared_connection_string,
    use_jsonb=True, 
)

# %% [markdown]
# # PDF Embedding to DB (from lab 6)

# %%
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_pdf(file_path: str, school: str):
    """
    Parse a PDF file using PyMuPDFLoader to extract pages as Document objects.
    Each page is tagged with metadata including the 'school' and 'source' (filename).
    """
    docs = []
    try:
        loader = PyMuPDFLoader(file_path=file_path, mode='page')
        raw_docs = loader.load()
        for doc in raw_docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["school"] = school
            doc.metadata["source"] = os.path.basename(file_path)
            docs.append(doc)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return docs

pdf_files = [
    ("./PhD Student Handbook 2023-2024_8_19_23_php.pdf", "SCGH"),
    ("2024-2025 DPE Student Handbook.pdf", "DPE"),
    ("MPH-Student-Handbook-2023-2024.pdf", "MPH"),
    ("CISAT_Student_Handbook_2024-2025.pdf", "CISAT"),
    ("DBOS PSYCH Handbook 2023-2024 (final).pdf", "DBOS"),
    ("2024-25 Art Department Handbook.pdf", "SAH"),
    ("2024-25 student handbook CGU Math programs.pdf", "IMS"),
]

# Process each PDF file using the parse_pdf function.
all_docs = []
for file_path, school in pdf_files:
    print(f"Processing {file_path} for school {school}...")
    docs = parse_pdf(file_path, school)
    all_docs.extend(docs)

# Further chunk each page using a recursive text splitter.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = []
for doc in all_docs:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata=doc.metadata.copy())
        chunked_docs.append(new_doc)

# Add the document chunks to the vector store.
if chunked_docs:
    try:
        book_data_vector_store.add_documents(chunked_docs)
        print(f"Added {len(chunked_docs)} document chunks to the vector store.")
    except Exception as e:
        print("Error adding documents to the vector store:", e)
else:
    print("No documents were processed.")

# %% [markdown]
# # Prompt Engineering

# %%
vectorstore_content_summary = """
The vectorstore contains CGU Student Handbooks (2023–2025) from various schools (SCGH, DPE, MPH, CISAT, DBOS, SAH, IMS).
These documents include academic policies, registration procedures, degree requirements, and student services.
"""

relevant_scope = """CGU student handbook topics, including academic policies, registration procedures, degree requirements,
and student services."""

query_router_prompt_template = PromptTemplate.from_template("""
You are an expert at analyzing a user question and deciding which data source is best suited to answer it. You must choose **one** of the following options:

1. **Vectorstore**: Use this if the question can be answered by the **existing** content in the vectorstore.
   The vectorstore contains information about **{vectorstore_content_summary}**.
                                                            
---                                                         
                                                                                                                          
2. **Websearch**: Use this if the question is within scope but meets any of the following criteria:
    - The answer cannot be found in the local vectorstore.
    - The question requires more detailed or factual information than what is in the documents.
    - The topic is time-sensitive or current.    
                                                                                                                
---                                                         
                                                              
3. **Chitter-Chatter**: Use this if the question:
   - Is not related to the scope below, or
   - Is too broad, casual, or off-topic to be answered using vectorstore or websearch.
   
   Chitter-Chatter is a fallback agent that gives a friendly response along with a follow-up.
                                                            
---
                                                            
Scope Definition:
Relevant questions are those related to **{relevant_scope}**

---                                                        

Your Task:
Analyze the user's question. Return a JSON object with one key "Datasource" and one value: "Vectorstore", "Websearch", or "Chitter-Chatter".
""")

# %%
multi_query_generation_prompt = PromptTemplate.from_template("""
You are an AI assistant helping improve document retrieval in a vector-based search system.

---
**Context about the database:**
{vectorstore_content_summary}

Your goal is to help retrieve more relevant documents by rewriting a user's question from multiple angles.

---
**Instructions:**
Given the original question and the content summary above:
1. Return the original user question first.
2. Then generate {num_queries} alternative versions of the same question.
   - Rephrase using different word choices, structure, or emphasis while maintaining the original meaning.
   - Ensure all rewrites are topically relevant to the documents.

Format requirements:
- Do not include bullet points or numbers.
- Each version should appear on a separate newline.
- Return exactly {num_queries} + 1 total questions.
  
---
Original user question: {question}
""")

relevance_grader_prompt_template = PromptTemplate.from_template("""
You are a relevance grader evaluating whether a retrieved document is helpful in answering a user question.

---
Retrieved Document: 
{document}

User Question: 
{question}

---
Your Task: Assess whether the document contains any keyword overlap or semantic relevance to the question.
Return a JSON object with key "binary_score" whose value is "pass" or "fail".
""")

answer_generator_prompt_template = PromptTemplate.from_template("""
You are an assistant for question-answering tasks.

---
Context:
Use the following information to help answer the question:
{context}

User Question:
{question}
                                                                
---
Instructions:
1. Base your answer primarily on the context provided.
2. If the answer is not present in the context, say so explicitly.
3. Keep the answer concise and accurate.
4. At the end, include a reference section with source details.

---
Answer:
""")

hallucination_checker_prompt_template = PromptTemplate.from_template("""
You are an AI grader evaluating whether a student's answer is factually grounded in the provided reference materials.

---
Grading Criteria:
- Pass: The answer is fully based on the given facts without fabrication.
- Fail: The answer contains fabricated or unsupported information.

---
Reference Materials (FACTS):
{documents}

Student's Answer:
{generation}

---
Output Instructions:
Return a JSON object with keys "binary_score" and "explanation".
""")

answer_verifier_prompt_template = PromptTemplate.from_template("""
You are an AI grader verifying whether a student's answer correctly addresses the given question.

---
Question: 
{question}

Student's Answer: 
{generation}

---
Output Instructions:
Return a JSON object with keys "binary_score" and "explanation".
""")

query_rewriter_prompt_template = PromptTemplate.from_template("""
You are a query optimization expert tasked with rewriting questions to improve vector database retrieval accuracy.

---
Context:
- Original Question: {question}
- Previous Answer (incomplete or unhelpful): {generation}

Vectorstore Summary:
{vectorstore_content_summary}

Note: Use the summary as context but do not repeat it verbatim.

---
Your Task: Identify missing keywords or ambiguities and rewrite the question to improve search coverage.
Output a JSON object with keys "rewritten_question" and "explanation".
""")

chitterchatter_prompt_template = PromptTemplate.from_template("""
You are a friendly assistant designed to keep conversations within the current scope while maintaining a warm, helpful tone.

---
Current Scope:
{relevant_scope}

Your job is to respond conversationally while guiding the user toward topics related to CGU student information.

---
Response Guidelines:
- For greetings, respond warmly.
- For off-topic questions, gently redirect the conversation.
""")


# %% [markdown]
# # Graph Definition

# %%
class GraphState(TypedDict):
    question: str                        # User question
    original_question: str               # Copy of original question
    generation: str                      # LLM-generated answer
    datasource: str                      # Output from router node
    hallucination_checker_attempts: int  # Hallucination check retry count
    answer_verifier_attempts: int        # Answer verification retry count
    documents: List[str]                 # Retrieved document list
    checker_result: str                  # Relevance check result ('pass' or 'fail')

# %% [markdown]
# # Tools

# %%
# For serious query
llm_gpt = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.7,
    api_key=openai_api_key
)

# Just for chitter chatter
llm_gpt_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_api_key
)

web_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    tavily_api_key=tavily_api_key
)

query_router_prompt = query_router_prompt_template.format(
    relevant_scope = relevant_scope,
    vectorstore_content_summary = vectorstore_content_summary
)

multi_query_generator = (
    multi_query_generation_prompt
    | llm_gpt
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for i, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            rank = i + 1
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_documents = []
    for doc_str, score in reranked_results:
        doc = loads(doc_str)
        doc.metadata["rrf_score"] = score
        reranked_documents.append(doc)
    return reranked_documents

retrieval_chain_rag_fusion_mmr = (
    multi_query_generator  
    | book_data_vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 3,
            'fetch_k': 15,
            "lambda_mult": 0.5
            }
        ).map()
    | reciprocal_rank_fusion
)

# %%
def document_retriever(state):
    question = state["question"]
    rag_fusion_mmr_results = retrieval_chain_rag_fusion_mmr.invoke({
        "question": question,
        "num_queries": 3,
        "vectorstore_content_summary": vectorstore_content_summary
    })
    formatted_doc_results = [
        Document(
            metadata={k: v for k, v in doc.metadata.items() if k != 'rrf_score'},
            page_content=doc.page_content
        )
        for doc in rag_fusion_mmr_results
    ]
    return {"documents": formatted_doc_results}

def answer_generator(state):
    documents = state["documents"]
    original_question = state.get("original_question", 0)
    if original_question != 0:
        question = original_question
    else:
        question = state["question"]
    documents = [
        Document(metadata=doc["metadata"], page_content=doc["page_content"])
        if isinstance(doc, dict) else doc
        for doc in documents
    ]
    answer_generator_prompt = answer_generator_prompt_template.format(
        context=documents,
        question=question
    )
    answer_generation = llm_gpt.invoke(answer_generator_prompt)
    return {"generation": answer_generation.content}

def web_search(state):
    question = state["question"]
    documents = state.get("documents", [])
    web_results = web_search_tool.invoke(question)
    formatted_web_results = [
        {
            "metadata": {
                "title": result["title"],
                "url": result["url"]
            },
            "page_content": result["content"]
        }
        for result in web_results
    ]
    documents = [
        Document(metadata=doc["metadata"], page_content=doc["page_content"])
        if isinstance(doc, dict) else doc
        for doc in documents
    ]
    documents.extend(formatted_web_results)
    print(f"Total number of web search documents: {len(formatted_web_results)}")
    return {"documents": documents}

def chitter_chatter(state):
    print("\n---CHIT-CHATTING---")
    question = state["question"]
    # Format the chitter chatter prompt with the relevant scope
    formatted_prompt = chitterchatter_prompt_template.format(relevant_scope=relevant_scope)
    chitterchatter_response = llm_gpt_mini.invoke(
        [SystemMessage(formatted_prompt),
         HumanMessage(question)]
    )
    return {"generation": chitterchatter_response.content}

def query_rewriter(state):
    print("\n---QUERY REWRITE---")
    original_question = state.get("original_question", 0)
    if original_question != 0:
        question = original_question
    else:
        question = state["question"]
    generation = state["generation"]
    query_rewriter_prompt = query_rewriter_prompt_template.format(
        question=question,
        generation=generation,
        vectorstore_content_summary=vectorstore_content_summary
    )
    query_rewriter_result = llm_gpt.with_structured_output(method="json_mode").invoke(
        query_rewriter_prompt)
    return {"question": query_rewriter_result['rewritten_question'],
            "original_question": question}

def hallucination_checker_tracker(state):
    num_attempts = state.get("hallucination_checker_attempts", 0)
    return {"hallucination_checker_attempts": num_attempts + 1}

def answer_verifier_tracker(state):
    num_attempts = state.get("answer_verifier_attempts", 0)
    return {"answer_verifier_attempts": num_attempts + 1}

async def grade_documents_parallel(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    async def grade_document(doc, question):
        prompt = relevance_grader_prompt_template.format(
            document=doc,
            question=question
        )
        grader_result = await llm_gpt_mini.with_structured_output(method="json_mode").ainvoke(prompt)
        return grader_result
    tasks = [grade_document(doc, question) for doc in documents]
    results = await asyncio.gather(*tasks)
    filtered_docs = []
    for i, score in enumerate(results):
        if score["binary_score"].lower() == "pass":
            print(f"---GRADE: DOCUMENT RELEVANT--- {score['binary_score']}")
            filtered_docs.append(documents[i])
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    total_docs = len(documents)
    relevant_docs = len(filtered_docs)
    if total_docs > 0:
        filtered_out_percentage = (total_docs - relevant_docs) / total_docs
        checker_result = "fail" if filtered_out_percentage >= 0.5 else "pass"
        print(f"---FILTERED OUT {filtered_out_percentage*100:.1f}% OF IRRELEVANT DOCUMENTS---")
        print(f"---**{checker_result}**---")
    else:
        checker_result = "fail"
        print("---NO DOCUMENTS AVAILABLE, WEB SEARCH TRIGGERED---")
    return {"documents": filtered_docs, "checker_result": checker_result}

def route_question(state):
    print("---ROUTING QUESTION---")
    question = state["question"]
    route_question_response = llm_gpt.with_structured_output(method="json_mode").invoke(
        [SystemMessage(query_router_prompt),
         HumanMessage(question)]
    )
    parsed_router_output = route_question_response["Datasource"]
    if parsed_router_output == "Websearch":
        print("---ROUTING QUESTION TO WEB SEARCH---")
        return "Websearch"
    elif parsed_router_output == "Vectorstore":
        print("---ROUTING QUESTION TO VECTORSTORE---")
        return "Vectorstore"
    elif parsed_router_output == "Chitter-Chatter":
        print("---ROUTING QUESTION TO CHITTER-CHATTER---")
        return "Chitter-Chatter"

def combined_retriever(state):
    vector_result = document_retriever(state)
    web_result = web_search(state)
    merged_docs = vector_result["documents"] + web_result["documents"]
    print(f"Merged {len(vector_result['documents'])} vector docs and {len(web_result['documents'])} web docs into {len(merged_docs)} total documents.")
    return {"documents": merged_docs}

def check_generation_vs_documents_and_question(state):
    print("---CHECK HALLUCINATIONS WITH DOCUMENTS---")
    question = state["question"]
    original_question = state.get("original_question", 0)
    if original_question != 0:
        question = original_question
    else:
        question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hallucination_checker_attempts = state.get("hallucination_checker_attempts", 0)
    answer_verifier_attempts = state.get("answer_verifier_attempts", 0)
    hallucination_checker_prompt = hallucination_checker_prompt_template.format(
        documents=documents,
        generation=generation
    )
    hallucination_checker_result = llm_gpt_mini.with_structured_output(method="json_mode").invoke(
        hallucination_checker_prompt)
    def ordinal(n):
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"
    if hallucination_checker_result['binary_score'].lower() == "pass":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---VERIFY ANSWER WITH QUESTION---")
        answer_verifier_prompt = answer_verifier_prompt_template.format(
            question=question,
            generation=generation
        )
        answer_verifier_result = llm_gpt_mini.with_structured_output(method="json_mode").invoke(
            answer_verifier_prompt)
        if answer_verifier_result['binary_score'] == "pass":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif answer_verifier_attempts > 1:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION, RE-WRITE QUERY---")
            print(f"This is the {ordinal(answer_verifier_attempts+1)} attempt.")
            return "not useful"
    elif hallucination_checker_attempts > 1:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        print(f"This is the {ordinal(hallucination_checker_attempts+1)} attempt.")
        return "not supported"

def decide_to_generate(state):
    """
    Conditional edge function used after document relevance grading.

    It checks the `checker_result` from the previous step:
    - If the result is 'fail' (indicating that a majority of documents were irrelevant),
      it triggers a fallback to web search for more reliable context.
    - If the result is 'pass', it proceeds to the answer generation node.

    Args:
        state (GraphState): Includes the 'checker_result' from the document grading step.

    Returns:
        str: Either 'generate' or 'Websearch', used to transition to the next node in the LangGraph.
    """
    print("---CHECK GENERATION CONDITION---")
    checker_result = state["checker_result"]
    
    if checker_result == "fail":
        print(
            "---DECISION: MORE THAN HALF OF THE DOCUMENTS ARE IRRELEVANT TO QUESTION, NOW INCLUDE WEB SEARCH---"
        )
        return "Websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

# %% [markdown]
# # Workflow

# %%
# Initialize the graph with shared state structure
workflow = StateGraph(GraphState)

# === Add agent nodes === 
workflow.add_node("WebSearcher", web_search)                    # web search
workflow.add_node("DocumentRetriever", document_retriever)      # Multi-query RAG + MMR + RRF
workflow.add_node("RelevanceGrader", grade_documents_parallel)  # Async document evaluation
workflow.add_node("AnswerGenerator", answer_generator)          # Generate grounded response
workflow.add_node("QueryRewriter", query_rewriter)              # Rewrite query if generation fails
workflow.add_node("ChitterChatter", chitter_chatter)            # Fallback for unsupported input

# === Add retry tracker nodes ===
workflow.add_node("HallucinationCheckerFailed", hallucination_checker_tracker)
workflow.add_node("AnswerVerifierFailed", answer_verifier_tracker)

# === Entry point: Route query to appropriate agent ===
workflow.set_conditional_entry_point(
    route_question,
    {
        "Websearch": "WebSearcher",
        "Vectorstore": "DocumentRetriever",
        "Chitter-Chatter": "ChitterChatter",
    },
)

# === Node transitions ===
workflow.add_edge("DocumentRetriever", "RelevanceGrader") 
workflow.add_edge("WebSearcher", "AnswerGenerator")             

workflow.add_edge("HallucinationCheckerFailed", "QueryRewriter")    
workflow.add_edge("AnswerVerifierFailed", "QueryRewriter")         

workflow.add_edge("QueryRewriter", "DocumentRetriever")             
workflow.add_edge("ChitterChatter", END)                            

workflow.add_conditional_edges(
    "RelevanceGrader",
    decide_to_generate,
    {
        "Websearch": "WebSearcher",         # Too many irrelevant docs -> Web search
        "generate": "AnswerGenerator",      
    },
)

workflow.add_conditional_edges(
    "AnswerGenerator",
    check_generation_vs_documents_and_question,
    {
        "not supported": "HallucinationCheckerFailed",  # Hallucinated -> Retry generation via rewriting
        "useful": END,                                  
        "not useful": "AnswerVerifierFailed",           
        "max retries": "ChitterChatter"             
    },
)

# --- Compile the graph ---
def get_workflow():
    return workflow