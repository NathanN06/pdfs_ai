# Import necessary functions
from services.retrieval_service import retrieve_documents, generate_response
from services.embedding_service import embed_query, embed_documents
from services.indexing_service import index_embeddings, save_index, load_index
from utils.chunking import (
    chunk_text, semantic_chunk_text, recursive_chunk_text, 
    adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens
)
import json

# Define a dictionary of chunking methods for easy access
chunking_methods = {
    "character": chunk_text,
    "semantic": semantic_chunk_text,
    "recursive": recursive_chunk_text,
    "adaptive": adaptive_chunk_text,
    "paragraph": chunk_by_paragraph,
    "tokens": chunk_by_tokens
}

def test_generation(query: str, chunking_strategy: str):
    # Ensure the chunking strategy is valid
    if chunking_strategy not in chunking_methods:
        raise ValueError(f"Invalid chunking strategy. Choose from: {list(chunking_methods.keys())}")
    
    # Step 1: Load and chunk documents using the selected strategy
    print(f"Using {chunking_strategy} chunking strategy...")
    with open("/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data/Preprocessed_Data/preprocessed_documents.json", 'r') as f:
        documents = json.load(f)
    
    # Apply the selected chunking method
    chunks = [(doc_id, chunking_methods[chunking_strategy](doc_text)) for doc_id, doc_text in documents.items()]
    
    # Step 2: Embed document chunks and create the index
    print("Embedding document chunks...")
    embedded_chunks = embed_documents(chunks)
    faiss_index = index_embeddings(embedded_chunks)
    save_index(faiss_index, "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Index/FAISS_index")
    print("Document embeddings indexed and saved.")

    # Step 3: Embed the user query
    print("Embedding query...")
    query_embedding = embed_query(query)

    # Step 4: Retrieve documents using the embedded query
    print("Retrieving relevant documents...")
    retrieved_docs = retrieve_documents(query_embedding, "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Index/FAISS_index")
    
    # Step 5: Generate the response based on retrieved documents
    print("Generating response...")
    response = generate_response(query, retrieved_docs)
    
    print("Response:")
    print(response)

# Example query and chunking strategy to test
test_query = "What are the main benefits of using a retrieval-augmented generation model?"
test_generation(test_query, "semantic")  # Replace "semantic" with the desired chunking strategy