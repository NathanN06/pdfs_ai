# embedding_service.py
from sentence_transformers import SentenceTransformer
from config import DATA_FOLDER, INDEX_FILENAME
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the model only once to improve performance
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(documents):
    """
    Embeds a list of document texts.
    
    Args:
        documents (list): A list of tuples where each tuple contains a document identifier and text.
    
    Returns:
        np.ndarray: An array of embeddings corresponding to the document texts.
    """
    texts = [text for _, text in documents]
    embeddings = model.encode(texts)
    return embeddings

def embed_query(query):
    """
    Embeds a single user query.
    
    Args:
        query (str): The user's query string.
    
    Returns:
        np.ndarray: An array of embeddings corresponding to the query.
    """
    query_embedding = model.encode([query])
    return query_embedding

