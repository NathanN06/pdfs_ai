import json
import time
import statistics
from sklearn.metrics.pairwise import cosine_similarity
from services.retrieval_service import retrieve_documents, handle_query
from services.embedding_service import embed_query
from services.indexing_service import load_index
from services.document_loader import load_documents
from utils.chunking import (
    chunk_text, semantic_chunk_text, recursive_chunk_text, 
    chunk_by_paragraph, adaptive_chunk_text, chunk_by_tokens
)
from config import DATA_FOLDER, INDEX_FILENAME

# Path to preprocessed documents file
preprocessed_file_path = "/Users/nathannguyen/Documents/RAG_BOT_1/preprocessed_documents.json"

# Load preprocessed documents as sample text
def load_sample_text(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    sample_text = " ".join([chunk[1] for chunk in data])  # Combine each document's text content
    return sample_text

# Load sample text from JSON file
sample_text = load_sample_text(preprocessed_file_path)

# Define test queries and expected document IDs (true_docs) for testing
test_queries = [
    "Provide an overview of A Level Economics and its main themes",
    "Explain the purpose and structure of the Pearson Edexcel Level 3 Advanced GCE in Economics",
    "Discuss key microeconomic concepts in the A Level Economics specification",
    "What are the assessment objectives in Edexcel A Level Economics?",
    "Explain the main topics in macroeconomic policy covered in A Level Economics."
]

# Expected document IDs for each query, based on content in JSON
true_docs = [
    ["A_Level_Econ_A_Spec.pdf"],
    ["A_Level_Econ_A_Spec.pdf"],
    ["A_Level_Econ_A_Spec.pdf"],
    ["A_Level_Econ_A_Spec.pdf"],
    ["A_Level_Econ_A_Spec.pdf"]
]

# Helper functions for evaluating chunking
def test_chunk_length_consistency(chunks):
    lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(lengths) / len(lengths)
    length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
    return avg_length, length_variance

def test_semantic_coherence(chunks, embed_function):
    embeddings = [embed_function(chunk) for chunk in chunks]
    coherence_scores = []
    for i in range(1, len(embeddings)):
        coherence_scores.append(cosine_similarity(
            embeddings[i-1].reshape(1, -1), embeddings[i].reshape(1, -1)
        )[0][0])  # Extract similarity score

    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    return avg_coherence

def test_retrieval_performance(index, documents, query, true_doc_ids, k=5, nprobe=10):
    query_embedding = embed_query(query)
    retrieved_docs = retrieve_documents(index, query_embedding, documents, query, k, nprobe)
    retrieved_doc_ids = [doc[0] for doc in retrieved_docs]  # Extract IDs of retrieved documents
    
    # Calculate recall and precision
    true_positives = len(set(true_doc_ids) & set(retrieved_doc_ids))
    recall = true_positives / len(true_doc_ids) if true_doc_ids else 0
    precision = true_positives / len(retrieved_doc_ids) if retrieved_doc_ids else 0
    
    return recall, precision

# Main function to evaluate each chunking method
def evaluate_chunking_methods(text, queries, true_docs, max_length=512):
    results = []
    index = load_index(INDEX_FILENAME)  # Load FAISS index
    documents = load_documents(DATA_FOLDER)  # Load preprocessed documents

    # Define each chunking method in a dictionary
    chunk_methods = {
        "chunk_text": chunk_text,
        "semantic_chunk_text": semantic_chunk_text,
        "recursive_chunk_text": recursive_chunk_text,
        "chunk_by_paragraph": chunk_by_paragraph,
        "adaptive_chunk_text": adaptive_chunk_text,
        "chunk_by_tokens": chunk_by_tokens,
    }

    for method_name, method_func in chunk_methods.items():
        start_time = time.time()
        
        # Call each method with or without max_length, as appropriate
        if method_name in ["chunk_text", "semantic_chunk_text", "recursive_chunk_text", "adaptive_chunk_text", "chunk_by_tokens"]:
            chunks = method_func(text, max_length)
        else:
            chunks = method_func(text)
        
        # Test each metric
        avg_length, length_variance = test_chunk_length_consistency(chunks)
        coherence_score = test_semantic_coherence(chunks, embed_query)
        
        recall_sum, precision_sum = 0, 0
        for query, true_doc_ids in zip(queries, true_docs):  # Loop through each query and its expected results
            recall, precision = test_retrieval_performance(index, documents, query, true_doc_ids)
            recall_sum += recall
            precision_sum += precision
        
        avg_recall = recall_sum / len(queries)
        avg_precision = precision_sum / len(queries)
        processing_time = time.time() - start_time

        # Store results
        results.append({
            "method": method_name,
            "avg_length": avg_length,
            "length_variance": length_variance,
            "coherence_score": coherence_score,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "processing_time": processing_time
        })
    
    return results

# Main execution block
if __name__ == "__main__":
    # Call the evaluate_chunking_methods function and print results
    results = evaluate_chunking_methods(sample_text, test_queries, true_docs)
    for result in results:
        print(result)
    # Run the evaluation
    results = evaluate_chunking_methods(sample_text, test_queries, true_docs)
    
    # Display the results
    for result in results:
        print(f"Method: {result['method']}")
        print(f"  Avg Length: {result['avg_length']}")
        print(f"  Length Variance: {result['length_variance']}")
        print(f"  Coherence Score: {result['coherence_score']}")
        print(f"  Avg Recall: {result['avg_recall']}")
        print(f"  Avg Precision: {result['avg_precision']}")
        print(f"  Processing Time: {result['processing_time']} seconds\n")