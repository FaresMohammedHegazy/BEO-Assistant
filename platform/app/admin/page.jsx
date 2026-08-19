"use client";
import { useState, useEffect } from "react";

export default function AdminDashboard() {
  const [tools, setTools] = useState([]);
  const [ragText, setRagText] = useState("");
  const [docIdToDelete, setDocIdToDelete] = useState("");

  // Fetch tools on load
  useEffect(() => {
    fetch("http://localhost:8000/api/admin/tools")
      .then((res) => res.json())
      .then((data) => setTools(data));
  }, []);

  // Removed TypeScript type annotations (: string, : number) for pure JS compatibility
  const handleToggle = async (agentName, toolName, currentState) => {
    await fetch("http://localhost:8000/api/admin/tools/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_name: agentName,
        tool_name: toolName,
        is_active: !currentState,
      }),
    });
    // Refresh tools
    const res = await fetch("http://localhost:8000/api/admin/tools");
    setTools(await res.json());
  };

  const handleRagUpload = async () => {
    const res = await fetch("http://localhost:8000/api/admin/rag/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: ragText, metadata: { source: "admin_ui" } }),
    });
    const data = await res.json();
    alert(`Document added! ID: ${data.document_id}`);
    setRagText("");
  };

  const handleRagDelete = async () => {
    const res = await fetch(`http://localhost:8000/api/admin/rag/document/${docIdToDelete}`, {
      method: "DELETE",
    });
    if (res.ok) alert("Document deleted!");
    else alert("Document not found.");
    setDocIdToDelete("");
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Admin Settings Dashboard</h1>

      {/* Tool Management Section */}
      <section className="mb-10 p-6 border rounded shadow-sm">
        <h2 className="text-xl font-semibold mb-4">Agent Tool Management</h2>
        <table className="w-full text-left border-collapse">
          <thead>
            <tr>
              <th className="border-b p-2">Agent Name</th>
              <th className="border-b p-2">Tool Name</th>
              <th className="border-b p-2">Status</th>
            </tr>
          </thead>
          <tbody>
            {tools.map((t, idx) => (
              <tr key={idx}>
                <td className="border-b p-2">{t.agent_name}</td>
                <td className="border-b p-2">{t.tool_name}</td>
                <td className="border-b p-2">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={t.is_active === 1}
                      onChange={() => handleToggle(t.agent_name, t.tool_name, t.is_active)}
                    />
                    <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* RAG Management Section */}
      <section className="p-6 border rounded shadow-sm">
        <h2 className="text-xl font-semibold mb-4">RAG VectorStore Management</h2>
        <div className="mb-6">
          <h3 className="font-medium mb-2">Upload Document</h3>
          <textarea
            className="w-full p-2 border rounded mb-2 text-black"
            rows={4}
            placeholder="Paste policy text here..."
            value={ragText}
            onChange={(e) => setRagText(e.target.value)}
          />
          <button onClick={handleRagUpload} className="bg-green-600 text-white px-4 py-2 rounded">
            Upload to VectorStore
          </button>
        </div>
        <div>
          <h3 className="font-medium mb-2">Remove Document</h3>
          <input
            type="text"
            className="w-full p-2 border rounded mb-2 text-black"
            placeholder="Document ID (e.g., doc-123456...)"
            value={docIdToDelete}
            onChange={(e) => setDocIdToDelete(e.target.value)}
          />
          <button onClick={handleRagDelete} className="bg-red-600 text-white px-4 py-2 rounded">
            Delete Document
          </button>
        </div>
      </section>
    </div>
  );
}