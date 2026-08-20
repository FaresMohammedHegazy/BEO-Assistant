"use client";
import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000/api/admin";

const TABS = [
  { key: "active", label: "Active" },
  { key: "pending_admin", label: "Pending Approval" },
  { key: "open", label: "Failures" },
  { key: "resolved", label: "Resolved" },
];

const STATUS_STYLES = {
  pending_admin: "bg-yellow-100 text-yellow-800",
  open: "bg-red-100 text-red-800",
  resolved: "bg-green-100 text-green-800",
};

export default function AdminTicketsDashboard() {
  const [tickets, setTickets] = useState([]);
  const [filter, setFilter] = useState("active");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [modifyOpen, setModifyOpen] = useState(false);
  const [modifyPayload, setModifyPayload] = useState("{}");
  const [actionError, setActionError] = useState("");

  const loadTickets = async () => {
    const res = await fetch(`${API_BASE}/tickets`);
    setTickets(await res.json());
  };

  useEffect(() => {
    loadTickets();
    const interval = setInterval(loadTickets, 15000);
    return () => clearInterval(interval);
  }, []);

  const selectTicket = async (ticketId) => {
    setSelectedId(ticketId);
    setActionError("");
    setModifyOpen(false);
    const res = await fetch(`${API_BASE}/tickets/${ticketId}`);
    if (!res.ok) {
      setDetail(null);
      return;
    }
    setDetail(await res.json());
  };

  const runAction = async (url, body) => {
    setActionError("");
    const res = await fetch(url, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setActionError(err.detail || `Request failed (${res.status})`);
      return;
    }
    setSelectedId(null);
    setDetail(null);
    await loadTickets();
  };

  const submitDecision = (decision, useModifyPayload) => {
    let payload = null;
    if (useModifyPayload) {
      try {
        payload = JSON.parse(modifyPayload || "{}");
      } catch (e) {
        setActionError("Payload must be valid JSON.");
        return;
      }
    }
    runAction(`${API_BASE}/tickets/${selectedId}/decision`, { decision, payload });
  };

  const retryTicket = () => {
    runAction(`${API_BASE}/tickets/${selectedId}/resume`, null);
  };

  const visibleTickets = tickets.filter((t) =>
    filter === "active" ? t.status !== "resolved" : t.status === filter
  );

  let snapshotPretty = detail?.state_snapshot || "";
  try {
    snapshotPretty = JSON.stringify(JSON.parse(detail.state_snapshot), null, 2);
  } catch (e) {
    // leave as raw string if it isn't valid JSON
  }

  return (
    <div className="flex h-[calc(100vh-52px)]">
      {/* List pane */}
      <div className="w-96 border-r overflow-y-auto">
        <div className="flex gap-1 p-2 border-b">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setFilter(tab.key)}
              className={`flex-1 text-xs py-1.5 rounded border ${
                filter === tab.key ? "bg-gray-800 text-white" : "bg-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {visibleTickets.length === 0 && (
          <p className="text-gray-400 text-sm p-6 text-center">No tickets here.</p>
        )}

        {visibleTickets.map((t) => (
          <div
            key={t.ticket_id}
            onClick={() => selectTicket(t.ticket_id)}
            className={`p-3 border-b cursor-pointer hover:bg-gray-100 ${
              t.ticket_id === selectedId ? "bg-gray-100" : ""
            }`}
          >
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_STYLES[t.status]}`}>
              {t.status.replace("_", " ")}
            </span>
            <div className="text-xs text-gray-500 mt-1">
              {t.graph_id} · {t.thread_id}
            </div>
            <div className="text-sm mt-1 truncate">{t.error_message}</div>
          </div>
        ))}
      </div>

      {/* Detail pane */}
      <div className="flex-1 p-8 overflow-y-auto">
        {!detail && <p className="text-gray-400">Select a ticket to see its checkpointed state.</p>}

        {detail && (
          <>
            {actionError && (
              <div className="bg-red-50 border border-red-300 text-red-700 text-sm p-2 rounded mb-4">
                {actionError}
              </div>
            )}
            <h2 className="text-lg font-semibold mb-4">{detail.ticket_id}</h2>

            <div className="mb-4">
              <p className="text-xs text-gray-500">Graph / Thread</p>
              <p>{detail.graph_id} — {detail.thread_id}</p>
            </div>
            <div className="mb-4">
              <p className="text-xs text-gray-500">Status</p>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_STYLES[detail.status]}`}>
                {detail.status.replace("_", " ")}
              </span>
            </div>
            <div className="mb-4">
              <p className="text-xs text-gray-500">Reason / Error</p>
              <p>{detail.error_message || "—"}</p>
            </div>
            <div className="mb-6">
              <p className="text-xs text-gray-500 mb-1">Checkpointed state at time of pause/failure</p>
              <pre className="bg-gray-100 border rounded p-3 text-xs overflow-x-auto">{snapshotPretty}</pre>
            </div>

            {detail.status === "pending_admin" && (
              <div>
                <div className="flex gap-2">
                  <button
                    onClick={() => submitDecision("approve")}
                    className="bg-green-600 text-white px-4 py-2 rounded"
                  >
                    {detail.graph_id === "vendor_logistics" ? "Approve Variance" : "Approve"}
                  </button>
                  <button
                    onClick={() => submitDecision("reject")}
                    className="bg-red-600 text-white px-4 py-2 rounded"
                  >
                    Reject
                  </button>
                  <button
                    onClick={() => setModifyOpen(!modifyOpen)}
                    className="bg-gray-200 px-4 py-2 rounded"
                  >
                    Modify…
                  </button>
                </div>
                {modifyOpen && (
                  <div className="mt-3">
                    <label className="text-xs text-gray-500 block mb-1">
                      Payload (JSON, merged into graph state)
                    </label>
                    <textarea
                      className="w-full border rounded p-2 text-xs font-mono text-black"
                      rows={4}
                      value={modifyPayload}
                      onChange={(e) => setModifyPayload(e.target.value)}
                    />
                    <button
                      onClick={() => submitDecision("modify", true)}
                      className="mt-2 bg-gray-800 text-white px-4 py-2 rounded"
                    >
                      Submit Modification
                    </button>
                  </div>
                )}
              </div>
            )}

            {detail.status === "open" && (
              <button onClick={retryTicket} className="bg-gray-800 text-white px-4 py-2 rounded">
                Retry Step
              </button>
            )}

            {detail.status === "resolved" && (
              <p className="text-sm text-gray-600">
                Decision: <strong>{detail.decision || "—"}</strong> at {detail.resolved_at || "—"}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}