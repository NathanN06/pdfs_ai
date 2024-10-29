// src/components/ChatBot.js
"use client";

import { useState, useRef, useEffect } from "react";
import { Moon, Sun, Send } from "lucide-react";
import { marked } from "marked";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    console.log("User input:", input);

    const newMessage = { id: Date.now(), text: input, sender: "user" };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setIsTyping(true);

    try {
      console.log("Sending request to backend...");

      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_input: input }),
      });

      console.log("Request sent to backend");

      if (!response.ok) throw new Error("Failed to fetch response from server");

      const data = await response.json();
      console.log("Backend response:", data);

      // Parse the bot's response as Markdown to allow for formatted output
      const formattedResponse = marked(data.response); // Just Markdown parsing, no extra <br /> tags

      const botResponse = {
        id: Date.now(),
        text: formattedResponse,
        sender: "bot",
        sources: data.sources || "No source available", // Add sources
      };
      setMessages((prev) => [...prev, botResponse]);
    } catch (error) {
      console.error("Error fetching response:", error);
      const errorMessage = {
        id: Date.now(),
        text: "Error: Unable to connect to the server. Please try again later.",
        sender: "bot",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div
      className={`h-screen flex flex-col ${
        darkMode
          ? "bg-gray-900 text-white"
          : "bg-gradient-to-br from-purple-100 to-indigo-200 text-gray-900"
      } transition-colors duration-300`}
    >
      <header className="flex justify-between items-center p-4 bg-white dark:bg-gray-800 shadow-md">
        <h1 className="text-2xl font-bold font-mono text-purple-600 dark:text-purple-400">
          RAG Bot
        </h1>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="rounded-full p-2 hover:bg-purple-100 dark:hover:bg-gray-700"
        >
          {darkMode ? (
            <Sun className="h-6 w-6 text-yellow-400" />
          ) : (
            <Moon className="h-6 w-6 text-purple-600" />
          )}
        </button>
      </header>

      <main className="flex-grow overflow-auto p-4">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl p-3 rounded-lg ${
                  message.sender === "user"
                    ? "bg-purple-500 text-white rounded-br-none"
                    : "bg-white text-gray-900 dark:bg-gray-800 dark:text-white rounded-bl-none"
                } shadow-lg transform transition-all duration-300 hover:scale-105`}
              >
                {/* Render parsed HTML for bot responses */}
                {message.sender === "bot" ? (
                  <div
                    className="markdown-content"
                    dangerouslySetInnerHTML={{ __html: message.text }}
                  ></div>
                ) : (
                  <p>{message.text}</p>
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-lg rounded-bl-none shadow-lg">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-purple-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-3 h-3 bg-purple-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-3 h-3 bg-purple-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="p-4 bg-white dark:bg-gray-800 shadow-md">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center space-x-2 max-w-2xl mx-auto"
        >
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-grow bg-gray-100 dark:bg-gray-700 border-purple-300 dark:border-gray-600 focus:ring-purple-500 focus:border-purple-500 p-2 rounded-lg"
          />
          <button
            type="submit"
            className="rounded-full bg-purple-500 hover:bg-purple-600 text-white p-2"
          >
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </button>
        </form>
      </footer>
    </div>
  );
};

export default Chatbot;
