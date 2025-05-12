# Hyperparameters for PDF ingestion chunking
PDF_CHUNK_SIZE    = 500  # higher = larger text chunks; range = [1, ∞)
PDF_CHUNK_OVERLAP = 50   # higher = more overlap;      range = [0, PDF_CHUNK_SIZE]

import os
import glob
import re
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_structure(text: str, last: dict) -> dict:
    """
    Extract Part, Chapter, Article, Rule/Section from text using regex.
    If not found, use last known value.
    """
    structure = last.copy()
    part_match = re.search(r"PART\s+([\w\d\-]+)", text, re.IGNORECASE)
    chapter_match = re.search(r"CHAPTER\s+([\w\d\-]+)", text, re.IGNORECASE)
    article_match = re.search(r"ARTICLE\s+([\w\d\-]+)", text, re.IGNORECASE)
    rule_match = re.search(r"(?:RULE|SECTION)\s+([\w\d\-\.]+)", text, re.IGNORECASE)
    if part_match:
        structure["part"] = part_match.group(1)
    if chapter_match:
        structure["chapter"] = chapter_match.group(1)
    if article_match:
        structure["article"] = article_match.group(1)
    if rule_match:
        structure["rule"] = rule_match.group(1)
    return structure

def parse_pdf(path: str, source: str) -> List[Document]:
    """
    Parse a PDF file into Document objects, one per page, extracting structure info.
    """
    docs: List[Document] = []
    last_structure = {"part": None, "chapter": None, "article": None, "rule": None}
    try:
        loader = PyMuPDFLoader(file_path=path, mode="page")
        pages = loader.load()
        for page in pages:
            page.metadata = page.metadata or {}
            page.metadata["source"] = source
            # Extract structure from page content
            structure = extract_structure(page.page_content, last_structure)
            page.metadata.update(structure)
            last_structure = structure
            docs.append(page)
    except Exception as e:
        print(f"Error parsing {path}: {e}")
    return docs

def check_document_exists(vector_store, source_name: str) -> bool:
    """
    Check if a document with the given source name already exists in the vector store.
    """
    try:
        # Search for documents with matching source metadata
        results = vector_store.similarity_search(
            query="",  # Empty query to get all documents
            k=1,  # We only need to know if any exist
            filter={"source": source_name}
        )
        return len(results) > 0
    except Exception as e:
        print(f"Error checking document existence: {e}")
        return False

def ingest_pdfs(data_dir: str, vector_store) -> int:
    """
    Load all PDFs from data_dir, chunk them, and add to vector_store.
    Skips PDFs that already exist in the database.
    """
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    all_pages: List[Document] = []
    skipped_count = 0
    
    for path in pdf_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        
        # Check if document already exists
        if check_document_exists(vector_store, name):
            print(f"Skipping {path} - already exists in database")
            skipped_count += 1
            continue
            
        print(f"Loading {path} as '{name}' …")
        all_pages.extend(parse_pdf(path, name))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP
    )
    chunks: List[Document] = []
    for page in all_pages:
        for txt in splitter.split_text(page.page_content):
            # Pass along the structure metadata
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
        print("No new PDF chunks found; nothing ingested.")
        return 0
