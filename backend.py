import os
from helper.env_setup           import load_environment
from helper.db_connection      import initialize_vector_store
from helper.pdf_ingestion      import ingest_pdfs
from helper.assemble_workflow  import get_workflow as build_workflow

def build_full_workflow():
    env_vars = load_environment()

    vector_store = initialize_vector_store(
        connection_string=env_vars["DB_CONNECTION"],
        api_key=env_vars["OPENAI_API_KEY"]
    )

    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    ingest_pdfs(data_dir, vector_store)

    return build_workflow(env_vars, vector_store)