from dotenv import load_dotenv
import os

# Load environment variables and configure tracing for LangSmith
def load_environment():
    """
    Load environment variables and set up LangSmith tracing.
    
    Returns:
        dict of required environment variables.
    """
    load_dotenv()
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    DB_CONNECTION    = os.getenv("DB_CONNECTION")
    TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY")
    LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")

    # Enable LangSmith tracing
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Insurance-Law-RAG"

    if not all([OPENAI_API_KEY, DB_CONNECTION, TAVILY_API_KEY, LANGSMITH_API_KEY]):
        print("Error: Missing one or more required environment variables")
    else:
        print("All environment variables loaded successfully")
        
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "DB_CONNECTION": DB_CONNECTION,
        "TAVILY_API_KEY": TAVILY_API_KEY,
        "LANGSMITH_API_KEY": LANGSMITH_API_KEY
    }
