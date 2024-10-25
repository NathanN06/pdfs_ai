// frontend/src/ApiComponent.js
import React, { useState, useEffect } from "react";
import axios from "axios";

const ApiComponent = () => {
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios
      .get("http://127.0.0.1:5000/api/message")
      .then((response) => setMessage(response.data.message))
      .catch((error) => console.error("Error fetching message:", error));
  }, []);

  return <h1>{message}</h1>;
};

export default ApiComponent;
