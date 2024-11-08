import os
import json
import warnings
import faulthandler
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from services.embedding_service import embed_documents, embed_query
from services.indexing_service import index_embeddings, save_index
from services.retrieval_service import retrieve_documents
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import INDEX_FOLDER, INDEX_FILENAME
from services.embedding_service import model
from utils.chunking import chunk_text, semantic_chunk_text, recursive_chunk_text, adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens

# Enable faulthandler for better debugging
faulthandler.enable()

# Suppress specific resource warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

# Set the directory for saving JSON results
result_dir = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test results"
os.makedirs(result_dir, exist_ok=True)

# Choose the chunking method
CHUNKING_METHOD = "adaptive"  # Set to "semantic", "recursive", etc., based on preference

# Define a function to apply the chosen chunking method
def apply_chunking_method(text, method=CHUNKING_METHOD):
    if method == "character":
        return chunk_text(text)
    elif method == "semantic":
        return semantic_chunk_text(text)
    elif method == "recursive":
        return recursive_chunk_text(text)
    elif method == "adaptive":
        return adaptive_chunk_text(text)
    elif method == "paragraph":
        return chunk_by_paragraph(text)
    elif method == "token":
        return chunk_by_tokens(text)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")

# Download and load the SciFact dataset into the specified directory
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Data"  # Custom path for SciFact dataset
data_folder = os.path.join(data_path, dataset)

print("Downloading SciFact dataset...")
data_path = util.download_and_unzip(url, data_path)
print("SciFact dataset downloaded and unzipped.")

# Load the dataset
print("Loading SciFact dataset...")
corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split="test")
print(f"Loaded {len(corpus)} passages, {len(queries)} queries, and relevance judgments for {len(qrels)} queries.")

# Chunk and embed each corpus passage, then create the FAISS index
print("Chunking and embedding passages...")
passage_texts = []
for doc_id, doc in corpus.items():
    chunks = apply_chunking_method(doc["text"], CHUNKING_METHOD)
    passage_texts.extend((f"{doc_id}_{i}", chunk) for i, chunk in enumerate(chunks))

passage_embeddings = embed_documents(passage_texts)
print("Finished embedding passages.")

print("Indexing embeddings with a Flat index for small dataset...")
index = index_embeddings(passage_embeddings, index_type='Flat')
index_path = os.path.join(INDEX_FOLDER, INDEX_FILENAME)
save_index(index, index_path)
print("Indexing complete.")

# Initialize results dictionary for retrieval
retrieval_results = {}

# Test retrieval with each query in the SciFact dataset
print("Running retrieval...")
for query_id, query_text in queries.items():
    query_embedding = embed_query(query_text)
    print(f"Query ID {query_id} embedding shape: {query_embedding.shape}")

    retrieved_docs = retrieve_documents(index, query_embedding, passage_texts, user_query=query_text)
    retrieved_doc_ids = [doc[0] for doc in retrieved_docs]

    print(f"Query ID {query_id}: Retrieved document IDs: {retrieved_doc_ids}")
    retrieval_results[query_id] = retrieved_doc_ids
print("Retrieval complete.")

# Save retrieval results to JSON for review
with open(os.path.join(result_dir, "retrieval_results_scifact.json"), "w") as f:
    json.dump(retrieval_results, f, indent=4)

# Function to calculate evaluation metrics and check for high similarity but low precision/recall
def calculate_metrics(relevant_texts, retrieved_texts, embeddings_model):
    print(f"Calculating metrics. Number of relevant texts: {len(relevant_texts)}, retrieved texts: {len(retrieved_texts)}")

    y_true = [1] * min(len(relevant_texts), len(retrieved_texts)) + [0] * max(0, len(retrieved_texts) - len(relevant_texts))
    y_pred = [1 if text in relevant_texts else 0 for text in retrieved_texts[:len(y_true)]]
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if relevant_texts and retrieved_texts:
        relevant_embeddings = embeddings_model.encode(relevant_texts)
        retrieved_embeddings = embeddings_model.encode(retrieved_texts)
        similarity = cosine_similarity(
            np.mean(relevant_embeddings, axis=0).reshape(1, -1),
            np.mean(retrieved_embeddings, axis=0).reshape(1, -1)
        )[0][0]
    else:
        similarity = 0.0
    
    return {"precision": precision, "recall": recall, "embedding_similarity": similarity}

# Initialize evaluation results dictionary
evaluation_results = {}
high_similarity_low_performance_count = 0

# For calculating overall averages
total_precision, total_recall, total_similarity = 0, 0, 0
query_count = len(queries)
high_similarity_low_performance_queries = {}

print("Evaluating retrieval results...")
for query_id, retrieved_doc_ids in retrieval_results.items():
    ground_truth_ids = qrels.get(query_id, [])
    
    relevant_texts = [corpus[doc_id]["text"] for doc_id in ground_truth_ids if doc_id in corpus]
    retrieved_texts = [corpus[doc_id.split('_')[0]]["text"] for doc_id in retrieved_doc_ids if doc_id.split('_')[0] in corpus]
    
    eval_result = calculate_metrics(relevant_texts, retrieved_texts, embeddings_model=model)
    evaluation_results[query_id] = eval_result
    
    total_precision += eval_result["precision"]
    total_recall += eval_result["recall"]
    total_similarity += eval_result["embedding_similarity"]
    
    if eval_result["embedding_similarity"] > 0.7 and eval_result["precision"] < 0.5 and eval_result["recall"] < 0.5:
        high_similarity_low_performance_queries[query_id] = {
            "query_text": queries[query_id],
            "precision": eval_result["precision"],
            "recall": eval_result["recall"],
            "embedding_similarity": eval_result["embedding_similarity"]
        }
        high_similarity_low_performance_count += 1

print("Evaluation complete.")

average_results = {
    "average_precision": total_precision / query_count,
    "average_recall": total_recall / query_count,
    "average_similarity": total_similarity / query_count,
    "high_similarity_low_performance_count": high_similarity_low_performance_count
}

print(f"\nOverall Averages:\nPrecision: {average_results['average_precision']:.4f}, Recall: {average_results['average_recall']:.4f}, Similarity: {average_results['average_similarity']:.4f}")
print(f"High Similarity Low Performance Queries Count: {high_similarity_low_performance_count}")

with open(os.path.join(result_dir, "average_results.json"), "w") as f:
    json.dump(average_results, f, indent=4)

def convert_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj

with open(os.path.join(result_dir, "high_similarity_low_performance_queries.json"), "w") as f:
    json.dump(convert_floats(high_similarity_low_performance_queries), f, indent=4)

print("All results saved.")