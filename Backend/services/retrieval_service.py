# retrieval_service.py
import os
from services.embedding_service import embed_query
from services.indexing_service import load_index
from services.document_loader import load_documents
from openai import OpenAI
from dotenv import load_dotenv
from config import DATA_FOLDER, INDEX_FILENAME
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_documents(index, query_embedding, documents, k=10, nprobe=10):
    """
    Retrieve the top k relevant documents based on embedding similarity,
    then refine relevance through reranking by keyword matching.
    """
    index.nprobe = nprobe
    distances, indices = index.search(query_embedding, k)
    initial_retrieved_docs = [documents[i] for i in indices[0] if 0 <= i < len(documents)]
    
    # Rerank using combined embedding similarity and keyword matching
    return rerank_documents(initial_retrieved_docs, query_embedding)

def rerank_documents(retrieved_docs, query_embedding):
    """
    Rerank documents by combining cosine similarity with keyword matching.
    """
    keywords = set(query_embedding.strip().lower().split())  # Split query into keywords
    reranked_docs = []

    for doc in retrieved_docs:
        doc_text = doc[1].lower()
        
        # Keyword match score based on occurrences in the document text
        keyword_score = sum(doc_text.count(keyword) for keyword in keywords)
        
        # Cosine similarity score between query embedding and document embedding
        doc_embedding = embed_query(doc_text)
        similarity_score = cosine_similarity(query_embedding, doc_embedding.reshape(1, -1))[0][0]
        
        # Combine similarity and keyword scores with weighting
        combined_score = similarity_score + 0.1 * keyword_score
        reranked_docs.append((doc, combined_score))
    
    # Sort by the combined score in descending order
    reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)
    
    # Return top k reranked documents
    return [doc for doc, _ in reranked_docs[:5]]

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

    if not use_context:
        return generate_general_response(user_query, message_history)

    index = load_index(index_filename)
    documents = load_documents(data_folder)
    
    # Add context terms to guide the embedding
    context_terms = ["economics", "themes", "Edexcel", "specification"]
    query_embedding = embed_query(user_query, additional_context=context_terms)
    
    # Retrieve and rerank documents
    retrieved_docs = retrieve_documents(index, query_embedding, documents)

    return generate_response(retrieved_docs, user_query, message_history)
