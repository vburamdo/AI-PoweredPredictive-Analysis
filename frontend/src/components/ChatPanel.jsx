import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  collection,
  query,
  orderBy,
  onSnapshot,
  addDoc,
} from "firebase/firestore";
import { firestore } from "../firebaseConfig";

/**
 * ChatPanel — Real-time conversational interface with Firestore + FastAPI backend.
 * -----------------------------------------------------
 * - Reads messages live from Firestore using onSnapshot()
 * - Sends user prompts to FastAPI (/ask_ai)
 * - Displays streaming responses automatically when backend writes to Firestore
 * - Falls back to local Firestore write on network error
 */

export default function ChatPanel({
  username = "biswajeet",
  tenantId = "CUST001",
  appId = "APP01",
  convoId = "convo_1",
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const db = firestore
  // 🔥 Real-time listener: auto-updates chat UI from Firestore
  useEffect(() => {
    const messagesRef = collection(
      db,
      "user_conversations",
      username,
      "tenants",
      tenantId,
      "apps",
      appId,
      "conversations",
      convoId,
      "messages"
    );

    const q = query(messagesRef, orderBy("timestamp", "asc"));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const msgs = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));
      setMessages(msgs);
    });

    return () => unsubscribe();
  }, [username, tenantId, appId, convoId]);

  // 🚀 Send user message
  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);

    const payload = {
      username,
      tenant_id: tenantId,
      app_id: appId,
      convo_id: convoId,
      question: input,
    };

    try {
      // Send to backend FastAPI /ask_ai
      await axios.post("http://127.0.0.1:8000/ask_ai", payload, {
        headers: { "Content-Type": "application/json" },
      });
      setInput("");
    } catch (err) {
      console.error("⚠️ API Error:", err.message);

      // Fallback: Write message directly to Firestore (if backend not reachable)
    //   await addDoc(
    //     collection(
    //       db,
    //       "user_conversations",
    //       username,
    //       "tenants",
    //       tenantId,
    //       "apps",
    //       appId,
    //       "conversations",
    //       convoId,
    //       "messages"
    //     ),
    //     {
    //       role: "user",
    //       content: input,
    //       timestamp: new Date().toISOString(),
    //     }
    //   );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "90vh",
        padding: "20px",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <h2 style={{ textAlign: "center" }}>💬 Observability Assistant</h2>

      {/* Chat window */}
      <div
        style={{
          flex: 1,
          backgroundColor: "#f5f5f5",
          borderRadius: "12px",
          padding: "15px",
          marginTop: "15px",
          overflowY: "auto",
          border: "1px solid #ddd",
        }}
      >
        {messages.length === 0 && (
          <p style={{ textAlign: "center", color: "#888" }}>
            Start chatting with the AI assistant...
          </p>
        )}

        {messages.map((msg) => {
  // Convert content to displayable text
  let displayText = "";

  if (typeof msg.content === "string") {
    displayText = msg.content;
  } else if (typeof msg.content === "object") {
    try {
      // Pretty-print JSON content
      {/* displayText = JSON.stringify(msg.content, null, 2); */}
      displayText = msg?.content?.result
    } catch (err) {
      displayText = "[Invalid message format]";
    }
  } else {
    {/* displayText = String(msg.content); */}
    displayText = msg?.content?.result
  }

  return (
    <div
      key={msg.id}
      style={{
        display: "flex",
        justifyContent:
          msg.role === "user" ? "flex-end" : "flex-start",
        marginBottom: "12px",
      }}
    >
      <div
        style={{
          maxWidth: "70%",
          backgroundColor:
            msg.role === "user" ? "#007bff" : "#e6e6e6",
          color: msg.role === "user" ? "white" : "black",
          padding: "10px 14px",
          borderRadius: "12px",
          fontSize: "0.95rem",
          lineHeight: "1.4",
          whiteSpace: "pre-wrap",
          fontFamily: msg.role === "user" ? "inherit" : "monospace",
        }}
      >
        {displayText}
      </div>
    </div>
  );
})}


        {loading && (
          <p style={{ textAlign: "center", color: "#888" }}>
            🧠 AI is thinking...
          </p>
        )}
      </div>

      {/* Input area */}
      <div
        style={{
          display: "flex",
          marginTop: "10px",
          borderTop: "1px solid #ddd",
          paddingTop: "10px",
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about logs, anomalies, or system behavior..."
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "8px",
            border: "1px solid #ccc",
            outline: "none",
            fontSize: "1rem",
          }}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          style={{
            marginLeft: "10px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "8px",
            padding: "10px 20px",
            fontWeight: "bold",
            cursor: "pointer",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
