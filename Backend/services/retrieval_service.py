import os
import re
import numpy as np
from services.embedding_service import embed_query
from services.indexing_service import load_index
from services.document_loader import load_documents
from openai import OpenAI
from dotenv import load_dotenv
from config import DATA_FOLDER, INDEX_FILENAME
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Import stop words
import nltk

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize CrossEncoder reranking model
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def expand_query(query):
    """
    Expands the query by adding synonyms or related terms to the main keywords.
    """
    # Extract keywords from the query
    keywords = extract_keywords(query)
    expanded_terms = set(keywords)

    # Add synonyms from WordNet for query expansion
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                if lemma.name().lower() != keyword:  # Avoid adding the same word
                    expanded_terms.add(lemma.name().lower())

    # Join expanded terms into a single expanded query
    return " ".join(expanded_terms)

def extract_keywords(query):
    """
    Extract keywords from the query by removing stop words and non-alphanumeric characters.
    """
    # Remove non-alphanumeric characters and split into words
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return set(keywords)

def retrieve_documents(index, query_embedding, documents, user_query=None, k=10, nprobe=20):
    """
    Retrieve the top k relevant documents based on embedding similarity,
    then refine relevance through reranking using the CrossEncoder model.
    """
    index.nprobe = nprobe  # Increase nprobe to broaden initial retrieval

    # Ensure query_embedding has the shape (1, d)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Selectively expand the query based on query length or structure
    expanded_query_embedding = None
    if user_query and len(user_query.split()) < 5:  # Expand only for short queries
        expanded_query = expand_query(user_query)
        expanded_query_embedding = embed_query(expanded_query)
    
    # Adaptive weighting for original and expanded embeddings
    initial_sim_threshold = 0.8  # Set a similarity threshold
    if expanded_query_embedding is not None and expanded_query_embedding.ndim == 1:
        expanded_query_embedding = expanded_query_embedding.reshape(1, -1)

    # Retrieve a sample to test initial similarity before weighting
    _, initial_indices = index.search(query_embedding, k=3)
    initial_docs = [documents[i] for i in initial_indices[0] if i < len(documents) and len(documents[i]) > 2]

    # Only proceed if initial_docs has embeddings
    if initial_docs:
        initial_sim = np.mean([
            cosine_similarity([query_embedding], [doc[2]])[0][0] 
            for doc in initial_docs if len(doc) > 2
        ])
    else:
        initial_sim = 0  # Fallback if no documents with embeddings are found

    # Adjust weight dynamically based on initial similarity score
    weight_original = 0.9 if initial_sim >= initial_sim_threshold else 0.7
    weight_expanded = 1 - weight_original
    combined_embedding = (weight_original * query_embedding + 
                          weight_expanded * expanded_query_embedding if expanded_query_embedding is not None else query_embedding)

    # Step 1: Initial retrieval from FAISS with larger k for more coverage
    _, indices = index.search(combined_embedding, k)
    initial_retrieved_docs = [documents[i] for i in indices[0] if 0 <= i < len(documents)]

    # Step 2: Apply reranking using the CrossEncoder model, focusing on top candidates
    if user_query and initial_retrieved_docs:
        return rerank_with_model(initial_retrieved_docs[:5], user_query)  # Limit reranking to top 5
    else:
        return initial_retrieved_docs

def rerank_with_model(retrieved_docs, query_text, max_rerank=5):
    """
    Rerank documents using a CrossEncoder model, limiting to top results.
    
    Args:
        retrieved_docs (list): List of initially retrieved document chunks.
        query_text (str): The original user query text.
        max_rerank (int): Limit on the number of documents to rerank for efficiency.
    
    Returns:
        list: The reranked documents based on model scores.
    """
    # Limit to top retrieved documents to rerank
    top_retrieved_docs = retrieved_docs[:max_rerank]
    
    # Prepare inputs for the CrossEncoder (query, document) pairs
    pairs = [(query_text, doc[1]) for doc in top_retrieved_docs]
    
    # Generate reranking scores for each document
    scores = rerank_model.predict(pairs)
    
    # Combine documents with their scores and sort by descending score
    reranked_docs = sorted(zip(top_retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Return the reranked documents
    return [doc for doc, score in reranked_docs]

def generate_general_response(user_query, message_history):
    """
    Generate a response without document context for general questions.
    """
    prompt = f"Answer the following question as a general AI response:\n{user_query}"
    full_message_history = message_history + [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_message_history
        )
        return response.choices[0].message.content, "Bot"
    except Exception as e:
        print(f"Error generating general response: {e}")
        return "There was an error processing your request. Please try again later.", "Bot"
    
def generate_response(retrieved_docs, user_query, message_history):
    """
    Generate a response using context from retrieved documents.
    """
    normalized_query = user_query.strip().lower()
    quick_responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! How can I help?",
        "help": "I'm here to assist you with any questions or information you need. Just ask!",
        "thank you": "You're very welcome!",
        "thanks": "You're very welcome!",
    }

    if normalized_query in quick_responses:
        return quick_responses[normalized_query], "Bot"

    context = "\n\n".join([doc[1] for doc in retrieved_docs])
    sources = ", ".join([doc[0] for doc in retrieved_docs])

    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"Answer the question: {user_query}\n\n"
        "Please provide a structured and well-organized response in the following format:\n\n"
        "### Overview:\nA brief summary of the main points related to the query.\n\n"
        "### Detailed Response:\nUse bullet points, bold headers, and organized sections to answer the question thoroughly.\n\n"
        "### Additional Information (if relevant):\nAny extra details that might be useful to the user.\n\n"
    )

    formatted_history = [
        {"role": "assistant" if msg["role"] == "bot" else msg["role"], "content": msg["content"]}
        for msg in message_history
    ]
    full_message_history = formatted_history + [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_message_history
        )
        response_text = response.choices[0].message.content
        formatted_response = (
            response_text +
            f"\n\n---\n**Source(s):** {sources}"
        )
        return formatted_response, sources
    except Exception as e:
        print(f"Error generating response: {e}")
        return "There was an error processing your request. Please try again later.", sources

def handle_query(user_query, message_history, index_filename=INDEX_FILENAME, data_folder=DATA_FOLDER):
    """
    Handle a user query by deciding if document context is needed and generating a response.
    """
    context_keywords = ["based on documents", "with context", "refer to sources"]
    use_context = any(keyword in user_query.lower() for keyword in context_keywords)

    # If no document context is required, generate a general AI response
    if not use_context:
        return generate_general_response(user_query, message_history)

    # Load the FAISS index and documents
    index = load_index(index_filename)
    documents = load_documents(data_folder)

    # Add context terms to guide the embedding
    context_terms = ["economics", "themes", "Edexcel", "specification"]
    query_embedding = embed_query(user_query, additional_context=context_terms)

    # Retrieve and rerank documents using the CrossEncoder model
    retrieved_docs = retrieve_documents(index, query_embedding, documents, user_query)

    # Check if any documents were retrieved, else return a fallback response
    if not retrieved_docs:
        print(f"No relevant documents found for query '{user_query}'. Returning default response.")
        return "No relevant documents found", []

    # Generate a response based on the retrieved documents
    return generate_response(retrieved_docs, user_query, message_history)