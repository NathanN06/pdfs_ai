// frontend/src/ChatBot.js
import React, { useState } from "react";
import "./css/chat_bot.css"; // Import external CSS for styling

import ReactDOM from "react-dom/client";
import "./css/global.css"; // te

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();

    const newMessages = [...messages, { text: query, sender: "user" }];
    setMessages(newMessages);
    setQuery(""); // Clear the input field

    // Display "Processing..." message
    setMessages([...newMessages, { text: "Processing...", sender: "bot" }]);

    try {
      // Send the query to the Flask backend
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();

      // Add bot response and sources to the message history
      const botMessages = [
        { text: `Response: ${data.response}`, sender: "bot" },
        { text: `Sources: ${data.sources}`, sender: "source" },
      ];
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        ...botMessages,
      ]);
    } catch (error) {
      // Display error message
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        { text: `Error: ${error.message}`, sender: "bot error" },
      ]);
    }
  };

  return (
    <div className="chat-container">
      <h1>RAG Bot</h1>

      <div id="chat" className="chat-box">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.sender === "user" ? "You: " : ""}
            {message.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          placeholder="Ask your question here..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          requireds
        />
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ChatBot />
  </React.StrictMode>
);
