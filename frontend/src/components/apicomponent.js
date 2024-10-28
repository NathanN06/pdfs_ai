// src/components/ApiComponent.js
import React, { useState } from "react";
import axios from "axios";

const ApiComponent = () => {
  const [userMessage, setUserMessage] = useState(""); // User's input message
  const [botResponse, setBotResponse] = useState(""); // Bot's response

  // Function to handle sending the user's message to the backend
  const sendMessage = async () => {
    try {
      const response = await axios.post("http://localhost:5000/query", {
        user_input: userMessage, // Use 'user_input' to match the backend
      });
      setBotResponse(response.data.response);
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  return (
    <div>
      <input
        type="text"
        placeholder="Type your message..."
        value={userMessage}
        onChange={(e) => setUserMessage(e.target.value)}
      />
      <button onClick={sendMessage}>Send</button>

      {botResponse && <h1>Bot Response: {botResponse}</h1>}
    </div>
  );
};

export default ApiComponent;
