from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-large"  
COLLECTION_NAME  = "insurance_code"       

def initialize_vector_store(connection_string: str, api_key: str):
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
