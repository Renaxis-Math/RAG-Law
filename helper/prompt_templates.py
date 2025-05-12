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

Follow these requirements for citations in your answer:
- If you use a sentence or phrase directly from the context, wrap it in double quotes and add a citation in parentheses with the format:
  (Part {{part}}, Chapter {{chapter}}, Article {{article}}, Rule {{rule}})
- If you paraphrase or summarize a sentence or phrase from the context, italicize your paraphrase and add a citation in parentheses, starting with the original phrase/sentence from the document, followed by the part/chapter/article/rule:
  (*your paraphrase* (original phrase from document, Part {{part}}, Chapter {{chapter}}, Article {{article}}, Rule {{rule}}))
- Only cite if the information comes from the context.
- Omit any field that is missing, but always keep the order: Part, Chapter, Article, Rule.

IMPORTANT VERIFICATION STEPS:
1. Before attributing a direct quote to a specific Rule, double-check that the exact text appears in the Rule you're citing.
2. Verify that each Rule number you cite actually contains the content you're referencing.
3. For paraphrases, ensure the original text you're citing comes from the same Rule number.
4. Be extra careful not to mix up Rule numbers - confirm each Rule number directly from the context.

Examples of correct citation:
- Direct quote: "Every interest in property, or any relation thereto, or liability in respect thereof, of such a nature that a contemplated peril might directly damnify the insured, is an insurable interest." (Part 1, Chapter 2, Article 4, Rule 281)
- Paraphrase: *An insurance policy is void if the insured lacks an insurable interest.* (If the insured has no insurable interest, the contract is void., Part 1, Chapter 2, Article 4, Rule 280)

Provide a concise legal answer grounded in the context.
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
