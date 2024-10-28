# controllers/query_controller.py

from flask import Blueprint, request, jsonify
from services.retrieval_service import handle_query

query_bp = Blueprint('query', __name__)

@query_bp.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_input = data.get("user_input", "")
    print("Received query from frontend:", user_input)  # Log received query

    # Initialize message history if needed
    message_history = []  # or retrieve actual history if you're tracking it
    
    # Call handle_query
    chatbot_response, sources = handle_query(user_input, message_history)
    
    response_data = {
        "response": chatbot_response,
        "sources": sources
    }
    return jsonify(response_data)