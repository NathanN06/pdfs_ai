import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate
import json
import os
from datasets import load_dataset

# Load models and metrics
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge_metric = evaluate.load("rouge")

# Load a limited sample from the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
documents = [item["article"] for item, _ in zip(dataset, range(5))]  # Limit to the first 5 articles
summaries = [item["highlights"] for item, _ in zip(dataset, range(5))]  # Corresponding summaries

# Import your chunking methods
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.chunking import (
    chunk_text, semantic_chunk_text, recursive_chunk_text, 
    adaptive_chunk_text, chunk_by_paragraph, chunk_by_tokens
)

# Define your chunking strategies
CHUNKING_METHODS = {
    "character": chunk_text,
    "semantic": semantic_chunk_text,
    "recursive": recursive_chunk_text,
    "adaptive": adaptive_chunk_text,
    "paragraph": chunk_by_paragraph,
    "token": chunk_by_tokens
}

# Evaluate a chunking method
def evaluate_chunking_method(chunks, summary):
    # 1. Coherence - Embedding-based cosine similarity
    chunk_embeddings = embedding_model.encode(chunks)
    chunk_embeddings = [embedding.reshape(1, -1) for embedding in chunk_embeddings]  # Ensure each embedding is 2D
    coherence_scores = [cosine_similarity(chunk_embeddings[i], chunk_embeddings[i + 1])[0][0]
                        for i in range(len(chunk_embeddings) - 1)]
    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0

    # 2. Redundancy - TF-IDF overlap and similarity
    if len(chunks) > 1:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
        redundancy_scores = cosine_similarity(tfidf_matrix)
        avg_redundancy = float(np.mean(redundancy_scores[np.triu_indices(len(chunks), k=1)]))
    else:
        avg_redundancy = 0

    # 3. Relevance - ROUGE between concatenated chunks and summary
    concatenated_chunks = " ".join(chunks)
    relevance_result = rouge_metric.compute(predictions=[concatenated_chunks], references=[summary])
    avg_rouge_score = (
        relevance_result['rouge1'] if isinstance(relevance_result['rouge1'], float)
        else relevance_result['rouge1'].get('fmeasure', 0)
    )
    avg_rouge_score = float(avg_rouge_score)

    return {
        "coherence": avg_coherence,
        "redundancy": avg_redundancy,
        "relevance (ROUGE-1)": avg_rouge_score
    }

# Run evaluation for each chunking method
results = {}
for method_name, method_func in CHUNKING_METHODS.items():
    summary_evaluations = []
    for doc, summary in zip(documents, summaries):
        chunks = method_func(doc)  # Generate chunks from each document
        summary_evaluations.append(evaluate_chunking_method(chunks, summary))

    # Average metrics across all evaluated summaries
    avg_coherence = np.mean([result["coherence"] for result in summary_evaluations])
    avg_redundancy = np.mean([result["redundancy"] for result in summary_evaluations])
    avg_relevance = np.mean([result["relevance (ROUGE-1)"] for result in summary_evaluations])

    # Store results for the method
    results[method_name] = {
        "coherence": avg_coherence,
        "redundancy": avg_redundancy,
        "relevance (ROUGE-1)": avg_relevance
    }

# Save results to JSON
result_path = "/Users/nathannguyen/Documents/RAG_BOT_1/Backend/Test results/chunking_test_results.json"
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {result_path}")