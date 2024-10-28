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

def generate_response(retrieved_docs, user_query, message_history):
    # Handle basic commands with predefined responses
    normalized_query = user_query.strip().lower()
    quick_responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! How can I help?",
        "help": "I'm here to assist you with any questions or information you need. Just ask!",
        "thank you": "You're very welcome!",
        "thanks": "You're very welcome!",
    }

    if normalized_query in quick_responses:
        # Return predefined response if user query matches without using retrieved docs
        return quick_responses[normalized_query], "Bot"

    # Organize context from retrieved documents
    context = "\n\n".join([doc[1] for doc in retrieved_docs])
    sources = ", ".join([doc[0] for doc in retrieved_docs])

    # Enhanced prompt for structured and consistent responses
    prompt = (
        f"Given the following context:\n{context}\n\n"
        f"Answer the question: {user_query}\n\n"
        "Please provide a structured and well-organized response in the following format:\n\n"
        "### Overview:\nA brief summary of the main points related to the query.\n\n"
        "### Detailed Response:\nUse bullet points, bold headers, and organized sections to answer the question thoroughly.\n\n"
        "### Additional Information (if relevant):\nAny extra details that might be useful to the user.\n\n"
        "Make sure each section is clearly separated with headers and line breaks, "
        "use bullet points or dashes for lists, and bold important headers or terms. "
        "Conclude with a **Source(s):** section at the end listing the sources."
    )

    # Format message history for consistency
    formatted_history = [
        {"role": "assistant" if msg["role"] == "bot" else msg["role"], "content": msg["content"]}
        for msg in message_history
    ]
    full_message_history = formatted_history + [{"role": "user", "content": prompt}]

    try:
        # OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4",
            messages=full_message_history
        )

        # Extract response content
        response_text = response.choices[0].message.content

        # Append source list to maintain consistency in layout
        formatted_response = (
            response_text +
            f"\n\n---\n**Source(s):** {sources}"
        )
        
        return formatted_response, sources

    except Exception as e:
        print(f"Error generating response: {e}")
        return "There was an error processing your request. Please try again later.", sources


# retrieval_service.py

def handle_query(user_query, message_history, index_filename=INDEX_FILENAME, data_folder=DATA_FOLDER):
    # Define quick responses for common phrases and greetings
    normalized_query = user_query.strip().lower()
    quick_responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! How can I help?",
        "help": "I'm here to assist you with any questions or information you need. Just ask!",
        "thank you": "You're very welcome!",
    }

    # Check if the query is a quick response
    if normalized_query in quick_responses:
        return quick_responses[normalized_query], "Bot"

    # If no quick response is found, proceed with retrieval and response generation
    index = load_index(index_filename)
    documents = load_documents(data_folder)

    query_embedding = embed_query(user_query)
    retrieved_docs = retrieve_documents(index, query_embedding, documents)

    response_text, sources = generate_response(retrieved_docs, user_query, message_history)
    return response_text, sources
