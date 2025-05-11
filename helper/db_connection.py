# Hyperparameters for embedding store initialization
EMBEDDING_MODEL = "text-embedding-3-large"  # higher = more powerful embeddings; range = available OpenAI embedding models
COLLECTION_NAME  = "insurance_code"          # fixed name; range = any valid collection name

from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

def initialize_vector_store(connection_string: str, api_key: str):
    """
    Initialize and return a PGVector vector store.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection_string,
        use_jsonb=True
    )
    return store
