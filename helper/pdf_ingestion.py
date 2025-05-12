import os
import glob
import re
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

PDF_CHUNK_SIZE    = 500  # higher = larger text chunks; range = [1, ∞)
PDF_CHUNK_OVERLAP = 50   # higher = more overlap;      range = [0, PDF_CHUNK_SIZE]

def extract_structure(text: str, last: dict) -> dict:
    structure = last.copy()
    part_match = re.search(r"PART\s+([\w\d\-]+)", text, re.IGNORECASE)
    chapter_match = re.search(r"CHAPTER\s+([\w\d\-]+)", text, re.IGNORECASE)
    article_match = re.search(r"ARTICLE\s+([\w\d\-]+)", text, re.IGNORECASE)
    rule_match = re.search(r"(?:RULE|SECTION)\s+([\w\d\-\.]+)", text, re.IGNORECASE)
    # Improved regex to match numbered rules (e.g., "280.") but avoid matching decimal numbers
    numbered_rule_match = re.search(r"^\s*(\d{1,4})\.(?!\d)", text, re.MULTILINE)
    if part_match:
        structure["part"] = part_match.group(1)
    if chapter_match:
        structure["chapter"] = chapter_match.group(1)
    if article_match:
        structure["article"] = article_match.group(1)
    if rule_match:
        structure["rule"] = rule_match.group(1)
    elif numbered_rule_match:
        structure["rule"] = numbered_rule_match.group(1)
    return structure

def parse_pdf(path: str, source: str) -> List[Document]:
    docs: List[Document] = []
    last_structure = {"part": None, "chapter": None, "article": None, "rule": None}
    try:
        loader = PyMuPDFLoader(file_path=path, mode="page")
        pages = loader.load()
        for page in pages:
            page.metadata = page.metadata or {}
            page.metadata["source"] = source
            
            # Split page into sentences
            sentences = re.split(r'(?<=[.!?]) +', page.page_content)
            running_structure = last_structure.copy()
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Update structure if this sentence starts a new part/chapter/article/rule
                running_structure = extract_structure(sent, running_structure)
                meta = page.metadata.copy()
                
                # Only include non-None fields, in strict order
                for key in ["part", "chapter", "article", "rule"]:
                    if running_structure.get(key) is not None:
                        meta[key] = running_structure[key]

                docs.append(Document(page_content=sent, metadata=meta))
            last_structure = running_structure

    except Exception as e:
        print(f"Error parsing {path}: {e}")
    return docs

def check_document_exists(vector_store, source_name: str) -> bool:
    try:
        results = vector_store.similarity_search(
            query="",
            k = 1,                              # We only need to know if any exist
            filter={"source": source_name}
        )
        return len(results) > 0

    except Exception as e:
        print(f"Error checking document existence: {e}")
        return False

def ingest_pdfs(data_dir: str, vector_store) -> int:
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    all_sentences: List[Document] = []
    skipped_count = 0
    
    for path in pdf_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        
        # Check if document already exists
        if check_document_exists(vector_store, name):
            print(f"Skipping {path} - already exists in database")
            skipped_count += 1
            continue
            
        print(f"Loading {path} as '{name}' …")
        all_sentences.extend(parse_pdf(path, name))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing documents")

    if all_sentences:
        try:
            vector_store.add_documents(all_sentences)
            print(f"Added {len(all_sentences)} sentences to '{vector_store}' vector store.")
            return len(all_sentences)
        except Exception as e:
            print("Error adding sentences:", e)
            return 0
    else:
        print("No new PDF sentences found; nothing ingested.")
        return 0
