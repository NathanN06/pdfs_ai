# main.py
from services.indexing_service import create_and_save_index
from app import app
import sys
import os

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    # Set the desired chunking strategy and size directly in the code
    chunk_strategy = 'token'  # Choose between 'token', 'semantic', 'adaptive', etc.
    chunk_size = 512  # Adjust this value as needed for your use case
    
    # Create and save the index with the specified chunking strategy and size
    create_and_save_index(chunk_strategy=chunk_strategy, chunk_size=chunk_size)
    
    # Run the Flask app
    app.run(debug=True)

if __name__ == "__main__":
    main()

