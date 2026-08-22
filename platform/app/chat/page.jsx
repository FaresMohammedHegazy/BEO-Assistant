"use client";
import { useEffect, useRef, useState } from "react";

const API_BASE = "http://localhost:8000/api/chat";
const POLL_INTERVAL_MS = 5000;

export default function ChatPage() {
  const [catalog, setCatalog] = useState([]);
  const [activeKey, setActiveKey] = useState(null);
  const [sessions, setSessions] = useState({}); // agent_key -> session object
  const [formValues, setFormValues] = useState({}); // agent_key -> {field: value}
  const [draft, setDraft] = useState("");
  const [starting, setStarting] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/agents`)
      .then((res) => res.json())
      .then((data) => {
        setCatalog(data);
        if (data.length) setActiveKey(data[0].key);
      });
  }, []);

  const activeAgent = catalog.find((a) => a.key === activeKey) || null;
  const activeSession = activeKey ? sessions[activeKey] : null;

  // While the active session is paused, poll so an admin decision (or,
  // for billing_dispute, a resolution the graph reached on its own)
  // shows up here without the user having to say anything.
  useEffect(() => {
    if (!activeSession || !activeSession.paused) return;
    const sessionId = activeSession.session_id;
    const agentKey = activeKey;
    const interval = setInterval(async () => {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      setSessions((prev) => ({ ...prev, [agentKey]: data }));
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [activeSession?.session_id, activeSession?.paused, activeKey]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeSession?.messages?.length]);

  const updateField = (agentKey, fieldName, value) => {
    setFormValues((prev) => ({
      ...prev,
      [agentKey]: { ...prev[agentKey], [fieldName]: value },
    }));
  };

  const startSession = async (agentKey) => {
    setStarting(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_key: agentKey, fields: formValues[agentKey] || {} }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Couldn't start this agent.");
        return;
      }
      setSessions((prev) => ({ ...prev, [agentKey]: data }));
    } finally {
      setStarting(false);
    }
  };

  const endSession = async (agentKey) => {
    const session = sessions[agentKey];
    if (!session) return;
    await fetch(`${API_BASE}/sessions/${session.session_id}`, { method: "DELETE" });
    setSessions((prev) => {
      const next = { ...prev };
      delete next[agentKey];
      return next;
    });
    setFormValues((prev) => ({ ...prev, [agentKey]: {} }));
  };

  const sendMessage = async () => {
    const text = draft.trim();
    if (!text || !activeSession || sending) return;
    setDraft("");
    setSending(true);
    setError("");
    const sessionId = activeSession.session_id;
    const agentKey = activeKey;
    // Show the user's message immediately; it gets replaced by the
    // server's copy (which also carries the reply) once that resolves.
    setSessions((prev) => ({
      ...prev,
      [agentKey]: {
        ...prev[agentKey],
        messages: [
          ...prev[agentKey].messages,
          { role: "user", content: text, created_at: new Date().toISOString() },
        ],
      },
    }));
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "That message couldn't be sent.");
        return;
      }
      setSessions((prev) => ({ ...prev, [agentKey]: data }));
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-52px)]">
      {/* Agent switcher */}
      <div className="w-72 border-r overflow-y-auto">
        {catalog.map((agent) => {
          const session = sessions[agent.key];
          return (
            <button
              key={agent.key}
              onClick={() => {
                setActiveKey(agent.key);
                setError("");
              }}
              className={`w-full text-left p-3 border-b hover:bg-gray-100 ${
                agent.key === activeKey ? "bg-gray-100" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">{agent.label}</span>
                {session?.paused && (
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-yellow-100 text-yellow-800">
                    waiting
                  </span>
                )}
                {session?.finished && (
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-green-100 text-green-800">
                    done
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500 mt-1">{agent.description}</p>
            </button>
          );
        })}
      </div>

      {/* Chat pane */}
      <div className="flex-1 flex flex-col">
        {!activeAgent && <p className="text-gray-400 p-8">Loading agents…</p>}

        {activeAgent && (
          <>
            <div className="border-b p-4">
              <h2 className="text-lg font-semibold">{activeAgent.label}</h2>
              <p className="text-xs text-gray-500">{activeAgent.description}</p>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-300 text-red-700 text-sm p-2 mx-4 mt-3 rounded">
                {error}
              </div>
            )}

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {!activeSession && activeAgent.kind === "structured" && (
                <StartForm
                  agent={activeAgent}
                  values={formValues[activeAgent.key] || {}}
                  onChange={(field, value) => updateField(activeAgent.key, field, value)}
                  onStart={() => startSession(activeAgent.key)}
                  starting={starting}
                />
              )}

              {!activeSession && activeAgent.kind === "freeform" && (
                <p className="text-gray-400 text-sm">
                  Send a message below to start chatting with {activeAgent.label}.
                </p>
              )}

              {activeSession?.messages.map((m, i) => (
                <MessageBubble key={i} message={m} />
              ))}

              {activeSession?.paused && (
                <div className="bg-yellow-50 border border-yellow-300 text-yellow-800 text-sm p-3 rounded flex items-center justify-between">
                  <span>
                    Waiting on {activeSession.ticket_id ? `ticket ${activeSession.ticket_id}` : "a response"} — checking automatically.
                  </span>
                </div>
              )}

              <div ref={bottomRef} />
            </div>

            {activeSession && (
              <div className="border-t p-4">
                <div className="flex gap-2 mb-2">
                  <input
                    className="flex-1 border rounded px-3 py-2 text-sm disabled:bg-gray-100"
                    placeholder={
                      activeSession.finished
                        ? "This conversation is finished."
                        : activeSession.paused
                        ? "Waiting on a response — you can still send a message."
                        : "Type a message…"
                    }
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    disabled={sending}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={sending || !draft.trim()}
                    className="bg-blue-600 text-white px-4 py-2 rounded disabled:bg-gray-300"
                  >
                    Send
                  </button>
                </div>
                {activeAgent.kind === "structured" && (
                  <button
                    onClick={() => endSession(activeAgent.key)}
                    className="text-xs text-gray-400 hover:text-gray-600 underline"
                  >
                    Start a new {activeAgent.label.toLowerCase()} conversation
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function StartForm({ agent, values, onChange, onStart, starting }) {
  return (
    <div className="bg-white border rounded-lg p-4 max-w-md">
      <p className="text-sm text-gray-600 mb-3">
        Fill this in to start — this agent needs a bit of structured context before it can run.
      </p>
      <div className="space-y-3">
        {agent.fields.map((field) => (
          <div key={field.name}>
            <label className="text-xs text-gray-500 block mb-1">{field.label}</label>
            <input
              className="w-full border rounded px-3 py-2 text-sm"
              placeholder={field.example}
              value={values[field.name] || ""}
              onChange={(e) => onChange(field.name, e.target.value)}
            />
          </div>
        ))}
      </div>
      <button
        onClick={onStart}
        disabled={starting}
        className="mt-4 bg-gray-800 text-white px-4 py-2 rounded disabled:bg-gray-300"
      >
        {starting ? "Starting…" : "Start"}
      </button>
    </div>
  );
}

function MessageBubble({ message }) {
  if (message.role === "system") {
    return <p className="text-xs text-gray-400 text-center italic">{message.content}</p>;
  }
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-lg rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
          isUser ? "bg-blue-600 text-white" : "bg-white border"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
