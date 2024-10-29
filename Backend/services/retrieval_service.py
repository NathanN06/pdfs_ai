# retrieval_service.py
import os
from services.embedding_service import embed_query
from services.indexing_service import load_index
from services.document_loader import load_documents
from openai import OpenAI
from dotenv import load_dotenv
from config import DATA_FOLDER, INDEX_FILENAME

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_documents(index, query_embedding, documents, k=5, nprobe=10):
    index.nprobe = nprobe
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0] if 0 <= i < len(documents)]
    return retrieved_docs

def generate_general_response(user_query, message_history):
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
    context_keywords = ["based on documents", "with context", "refer to sources"]
    use_context = any(keyword in user_query.lower() for keyword in context_keywords)

    if not use_context:
        return generate_general_response(user_query, message_history)

    index = load_index(index_filename)
    documents = load_documents(data_folder)
    query_embedding = embed_query(user_query)
    retrieved_docs = retrieve_documents(index, query_embedding, documents)

    return generate_response(retrieved_docs, user_query, message_history)
