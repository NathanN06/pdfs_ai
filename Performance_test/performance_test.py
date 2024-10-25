import sys
import os
import time
import faiss  # Import faiss to load the index
import nltk

# Import necessary tokenizer
from nltk.tokenize import word_tokenize


sys.path.append('/Users/nathannguyen/Documents/RAG_BOT_1/Bot')

from document_loader import load_documents
from embedding_and_indexing import create_and_save_index
from retrieval_and_generation import generate_response, retrieve_documents, embed_query

# Function to load the FAISS index
def load_index(index_filename):
    return faiss.read_index(index_filename)

def test_performance(data_folder, chunk_sizes, index_filename):
    user_query = "Sample user query for testing"
    strategies = ['character'] 

    for strategy in strategies:
        for chunk_size in chunk_sizes:
            # Measure document loading and chunking time
            start_time = time.time()
            documents = load_documents(data_folder, chunk_strategy=strategy, chunk_size=chunk_size)
            load_time = time.time() - start_time

            # Measure time to create and save the index
            start_time = time.time()
            create_and_save_index(data_folder, index_filename)
            index_time = time.time() - start_time

            # Load the FAISS index for querying
            index = load_index(index_filename)

            # Simulate query and measure retrieval and generation time
            start_time = time.time()
            query_embedding = embed_query(user_query)
            retrieved_docs = retrieve_documents(index, query_embedding, documents)
            response = generate_response(retrieved_docs, user_query)
            response_time = time.time() - start_time
            
            # Print only the chunk size and time results
            print(f"Chunk size: {chunk_size} | "
                  f"Document loading time: {load_time:.4f} seconds | "
                  f"Indexing time: {index_time:.4f} seconds | "
                  f"Response generation time: {response_time:.4f} seconds")


if __name__ == "__main__":
    data_folder = "/Users/nathannguyen/Documents/RAG_BOT_1/Bot/Data"
    index_filename = "document_index.faiss"
    chunk_sizes = [128, 256, 512, 1024]  # List of chunk sizes to test
    
    test_performance(data_folder, chunk_sizes, index_filename)






