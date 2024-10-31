# compare_chunking_methods.py

import json
from utils.chunking import chunk_text, semantic_chunk_text, recursive_chunk_text, chunk_by_paragraph, adaptive_chunk_text, chunk_by_tokens

def load_preprocessed_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    documents = [doc[1] for doc in data]
    return documents

if __name__ == "__main__":
    filepath = 'preprocessed_documents.json'
    documents = load_preprocessed_data(filepath)
    
    # Select the document index and chunk index to compare
    document_index = 0  # Change this to select a different document
    chunk_index = 0     # Change this to view a different chunk within each method's output

    # Get the specific document text
    document_text = documents[document_index]

    # Define chunking methods
    chunking_methods = {
        'character': chunk_text,
        'semantic': semantic_chunk_text,
        'recursive': recursive_chunk_text,
        'paragraph': chunk_by_paragraph,
        'adaptive': adaptive_chunk_text,
        'token': chunk_by_tokens
    }

    # Apply each chunking method and print the specified chunk
    print(f"Comparing Chunk {chunk_index} from Document {document_index + 1} Using Different Chunking Methods:\n" + "=" * 80)
    for method_name, chunking_function in chunking_methods.items():
        # Apply the chunking function to the document
        if chunking_function.__name__ in ['chunk_text', 'semantic_chunk_text', 'recursive_chunk_text', 'adaptive_chunk_text', 'chunk_by_tokens']:
            chunks = chunking_function(document_text, 500)  # Default chunk size
        else:
            chunks = chunking_function(document_text)
        
        # Print the specified chunk from the current method
        if chunk_index < len(chunks):
            print(f"Method: {method_name}")
            print(f"Chunk {chunk_index}:\n{chunks[chunk_index]}\n" + "-" * 80)
        else:
            print(f"Method: {method_name}")
            print(f"Chunk {chunk_index} not available (only {len(chunks)} chunks generated)\n" + "-" * 80)