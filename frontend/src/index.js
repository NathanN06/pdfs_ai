import React from "react";
import ReactDOM from "react-dom/client";
import Chatbot from "./components/ChatBot";
import "./css/global.css";
import { Buffer } from "buffer";
import process from "process";

window.Buffer = Buffer;
window.process = process;

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <Chatbot />
  </React.StrictMode>
);
