from flask import Flask
from flask_cors import CORS
from controllers.query_controller import query_bp  # Importing Blueprint
import os

app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Apply CORS specifically to the blueprint
CORS(query_bp)

app.register_blueprint(query_bp)  # Register the blueprint with CORS enabled

app.secret_key = os.urandom(24)

@app.route("/")
def home():
    return "Welcome to the RAG Bot API"

if __name__ == "__main__":
    app.run(debug=True)