# California Insurance Law Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about California insurance law by retrieving and synthesizing information from the [California Insurance Code](https://leginfo.legislature.ca.gov/faces/codesTOCSelected.xhtml?tocCode=INS&tocTitle=+Insurance+Code+-+INS). It also search the Web using Tavily API for external validations.

## Overview

This chatbot helps users navigate and understand the California Insurance Code through natural language interaction. It uses advanced RAG techniques to provide accurate, cited responses grounded in actual legal text.

Key Features:
- Natural language understanding of legal questions
- Semantic search over California Insurance Code documents
- Hallucination detection and verification
- Proper legal citations in responses
- Fallback to web search for non-statute questions
- User feedback collection for continuous improvement

## Technical Architecture

![Graph Workflow](graph_workflow.png)

- **Frontend**: Streamlit web interface with chat functionality
- **Vector Store**: PostgreSQL with PGVector extension storing document embeddings
- **Workflow Engine**: LangGraph for dynamic query routing and processing
- **LLM Integration**: OpenAI GPT-4o and GPT-4o-mini models
- **Web Search**: Tavily API for fallback information retrieval

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Renaxis-Math/RAG-Law
   cd RAG-Law
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DB_CONNECTION=postgres://username:password@hostname:port/database
   TAVILY_API_KEY=your_tavily_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   ```

4. Prepare your PDF documents in the `Data/` directory

## System Components

- `main.py`: Streamlit web application
- `backend.py`: Core workflow initialization
- `helper/`: Module containing specialized components:
  - `env_setup.py`: Environment configuration
  - `db_connection.py`: Vector database connectivity
  - `pdf_ingestion.py`: Document processing pipeline
  - `assemble_workflow.py`: Workflow orchestration
  - `node_definitions.py`: LangGraph node implementations
  - `workflow_graph.py`: Graph state definitions
  - `prompt_templates.py`: LLM instruction templates
  - `entity_checker.py`: Query relevance verification

## Workflow

1. User submits a question
2. Router analyzes and directs to appropriate retrieval path
3. Vector search retrieves relevant statute passages
4. Relevance grader filters and ranks passages
5. Answer generator creates a response with citations
6. Verification checks for accuracy and hallucinations
7. Response is presented to user with feedback option

## Testing and Usage

(Optional) Generate a test suite:
```
python generate_test.py
```
Test cases are saved in `tests.csv`.

Start the application:
```
streamlit run main.py
```

Look in your terminal. The web interface will be available at localhost.
