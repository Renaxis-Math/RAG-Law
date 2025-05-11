from langchain_core.prompts import PromptTemplate

# Vectorstore context summary
vectorstore_summary = """
California Insurance Code: statutes, regulations, definitions,
policy requirements, and administrative procedures.
"""

# Scope definition for chitchat
scope_definition = """
Questions about the California Insurance Code, including statutes,
regulations, definitions, policy requirements, and compliance procedures.
"""

# Router: pick data source
router_template = PromptTemplate.from_template("""
You are a legal AI specializing in California insurance law.
Analyze the user's question and select exactly one datasource:
1) Vectorstore - local Insurance Code docs ({vectorstore_summary}).
2) Websearch - external case law, commentary, updates.
3) Chitter-Chatter - off-topic or casual queries.

Return JSON: {{"Datasource":"<Vectorstore|Websearch|Chitter-Chatter>"}}
""")

# Multi‐query rewrites for retrieval
multi_query_template = PromptTemplate.from_template("""
Rewrite the legal question to improve vector retrieval.
Context summary: {vectorstore_summary}
Original question: {question}

Return {num_queries} distinct rewrites, one per line.
""")

# Relevance check
relevance_template = PromptTemplate.from_template("""
Assess if the document is relevant to the question.
Document: {document}
Question: {question}

Return JSON: {{"binary_score":"pass" or "fail"}}
""")

# Answer generation
answer_template = PromptTemplate.from_template("""
Use the following context to answer the question:
Context: {context}
Question: {question}

Provide a concise legal answer grounded in the context. Cite sources.
""")

# Hallucination check
hallucination_template = PromptTemplate.from_template("""
Check if the answer is grounded in the provided context.
Context: {documents}
Answer: {generation}

Return JSON: {{"binary_score":"pass" or "fail","explanation":"…"}}
""")

# Verification check
verification_template = PromptTemplate.from_template("""
Verify the answer addresses the question.
Question: {question}
Answer: {generation}

Return JSON: {{"binary_score":"pass" or "fail","explanation":"…"}}
""")

# Rewrite after failure
rewrite_template = PromptTemplate.from_template("""
Rewrite the question to improve retrieval.
Original: {question}
Previous answer: {generation}
Context summary: {vectorstore_summary}

Return JSON: {{"rewritten_question":"…","explanation":"…"}}
""")

# Chitchat fallback
chitchat_template = PromptTemplate.from_template("""
You are a friendly assistant. Stay on-topic about California insurance law ({scope_definition}).
For off-topic queries, gently redirect back to insurance topics.
""")
