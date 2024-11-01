# indexing_service.py
import os
import faiss
from services.embedding_service import embed_documents
from services.document_loader import load_documents
from config import DATA_FOLDER, INDEX_FILENAME

def index_embeddings(embeddings, nlist=None, index_type='IVFFlat'):
    """
    Indexes embeddings using a specified FAISS index type.

    Args:
        embeddings (np.ndarray): Array of embeddings to index.
        nlist (int): Number of clusters. Defaults to min(100, len(embeddings) // 2).
        index_type (str): Type of FAISS index to use ('IVFFlat', 'IVFPQ', or 'IVFSQ').

    Returns:
        faiss.Index: The FAISS index with the embeddings added.
    """
    dimension = embeddings.shape[1]
    if nlist is None:
        nlist = min(100, len(embeddings) // 2)

    quantizer = faiss.IndexFlatL2(dimension)

    if index_type == 'IVFFlat':
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    elif index_type == 'IVFPQ':
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8)  # 8 subquantizers as an example
    elif index_type == 'IVFSQ':
        index = faiss.IndexIVFSQ(quantizer, dimension, nlist, faiss.METRIC_L2)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.train(embeddings)
    index.add(embeddings)
    return index
def save_index(index, filename=INDEX_FILENAME):
    """
    Saves the FAISS index to the specified file.

    Args:
        index (faiss.Index): The FAISS index to save.
        filename (str): Path to save the index file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    faiss.write_index(index, filename)

def load_index(filename=INDEX_FILENAME):
    """
    Loads the FAISS index from the specified file.

    Args:
        filename (str): Path to load the index file.

    Returns:
        faiss.IndexIVFFlat: Loaded FAISS index.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The FAISS index file was not found at: {filename}")
    return faiss.read_index(filename)

def create_and_save_index(data_folder=DATA_FOLDER, index_filename=INDEX_FILENAME, chunk_strategy='token', chunk_size=512):
    """
    Loads documents, creates embeddings, indexes them, and saves the index.

    Args:
        data_folder (str): Path to the folder containing documents.
        index_filename (str): Path to save the FAISS index.
        chunk_strategy (str): The chunking strategy to use.
        chunk_size (int): The size limit for each chunk.
    """
    documents = load_documents(data_folder, chunk_strategy=chunk_strategy, chunk_size=chunk_size)
    embeddings = embed_documents(documents)
    index = index_embeddings(embeddings)
    save_index(index, index_filename)


