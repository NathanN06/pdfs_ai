# document_loader.py
import os
import fitz  # PyMuPDF
import json
from utils.chunking import (
    chunk_text, semantic_chunk_text, recursive_chunk_text, 
    adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens
)
from config import DATA_FOLDER, INDEX_FILENAME

def load_documents(data_folder, chunk_strategy='character', chunk_size=512, overlap=0, save_file="preprocessed_documents.json"):
    """
    Load and chunk documents from PDFs in the specified data folder using various chunking strategies.

    Args:
        data_folder (str): Path to the folder containing PDF files.
        chunk_strategy (str): Chunking strategy to use ('character', 'semantic', 'recursive', 'adaptive', 'paragraph', 'token').
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of overlapping characters between chunks (applies to certain chunking strategies).
        save_file (str): File path to save the preprocessed chunks.

    Returns:
        list: List of chunked document texts.
    """
    documents = []

    if os.path.exists(save_file):
        print(f"Loading preprocessed documents from {save_file}")
        with open(save_file, "r") as f:
            documents = json.load(f)
        return documents  # Return preprocessed documents directly

    for filename in os.listdir(data_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_folder, filename)
            pdf_document = fitz.open(pdf_path)  # Open the PDF document with PyMuPDF
            text = ""  # Hold extracted text from PDF
            for page in pdf_document:
                text += page.get_text()  # Extract text from each page

            # Choose chunking strategy
            if chunk_strategy == 'semantic':
                chunks = semantic_chunk_text(text, max_length=chunk_size, overlap=overlap)
            elif chunk_strategy == 'recursive':
                chunks = recursive_chunk_text(text, max_length=chunk_size, overlap=overlap)
            elif chunk_strategy == 'adaptive':
                chunks = adaptive_chunk_text(text, default_max_length=chunk_size, overlap=overlap)
            elif chunk_strategy == 'paragraph':
                chunks = chunk_by_paragraph(text, max_length=chunk_size, overlap=overlap)
            elif chunk_strategy == 'token':
                chunks = chunk_by_tokens(text, max_tokens=chunk_size, overlap=overlap)
            else:
                chunks = chunk_text(text, max_length=chunk_size, overlap=overlap)

            for chunk in chunks:
                documents.append((filename, chunk))

    with open(save_file, "w") as f:
        json.dump(documents, f)

    print(f"Preprocessed documents saved to {save_file}")
    return documents  # Return the newly processed and chunked documents