import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "Data")
INDEX_FOLDER = os.path.join(BASE_DIR, "index")
INDEX_FILENAME = os.path.join(INDEX_FOLDER, "document_index.faiss")

