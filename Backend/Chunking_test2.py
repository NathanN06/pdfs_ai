import json
from utils.chunking import chunk_text, semantic_chunk_text, recursive_chunk_text, chunk_by_paragraph, adaptive_chunk_text, chunk_by_tokens

def load_preprocessed_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    documents = [doc[1] for doc in data]
    return documents

def compare_chunks(chunks_a, chunks_b, chunks_c):
    differing_chunks = []
    for i in range(min(len(chunks_a), len(chunks_b), len(chunks_c))):
        if chunks_a[i] != chunks_b[i] or chunks_b[i] != chunks_c[i] or chunks_a[i] != chunks_c[i]:
            differing_chunks.append(i)
    return differing_chunks

if __name__ == "__main__":
    filepath = 'preprocessed_documents.json'
    documents = load_preprocessed_data(filepath)
    
    # Iterate over each document to find differing chunks
    for document_index, document_text in enumerate(documents):
        # Define chunking methods
        chunking_methods = {
            'semantic': semantic_chunk_text,
            'recursive': recursive_chunk_text,
            'paragraph': chunk_by_paragraph
        }

        # Apply chunking methods
        semantic_chunks = chunking_methods['semantic'](document_text, 500)
        recursive_chunks = chunking_methods['recursive'](document_text, 500)
        paragraph_chunks = chunking_methods['paragraph'](document_text)

        # Compare and get differing chunk indices
        differing_chunks = compare_chunks(semantic_chunks, recursive_chunks, paragraph_chunks)
        if differing_chunks:
            print(f"Differing chunks found in Document {document_index + 1} at chunk indices: {', '.join(str(i + 1) for i in differing_chunks)}")
        else:
            print(f"No differing chunks found in Document {document_index + 1}.")