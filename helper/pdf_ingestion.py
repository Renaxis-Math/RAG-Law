# Hyperparameters for PDF ingestion chunking
PDF_CHUNK_SIZE    = 500  # higher = larger text chunks; range = [1, ∞)
PDF_CHUNK_OVERLAP = 50   # higher = more overlap;      range = [0, PDF_CHUNK_SIZE]

import os
import glob
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def parse_pdf(path: str, source: str) -> List[Document]:
    """
    Parse a PDF file into Document objects, one per page.
    """
    docs: List[Document] = []
    try:
        loader = PyMuPDFLoader(file_path=path, mode="page")
        pages = loader.load()
        for page in pages:
            page.metadata = page.metadata or {}
            page.metadata["source"] = source
            docs.append(page)
    except Exception as e:
        print(f"Error parsing {path}: {e}")
    return docs

def ingest_pdfs(data_dir: str, vector_store) -> int:
    """
    Load all PDFs from data_dir, chunk them, and add to vector_store.
    """
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    all_pages: List[Document] = []
    for path in pdf_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Loading {path} as '{name}' …")
        all_pages.extend(parse_pdf(path, name))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP
    )
    chunks: List[Document] = []
    for page in all_pages:
        for txt in splitter.split_text(page.page_content):
            chunks.append(Document(page_content=txt, metadata=page.metadata.copy()))

    if chunks:
        try:
            vector_store.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to '{vector_store}' vector store.")
            return len(chunks)
        except Exception as e:
            print("Error adding chunks:", e)
            return 0
    else:
        print("No PDF chunks found; nothing ingested.")
        return 0
