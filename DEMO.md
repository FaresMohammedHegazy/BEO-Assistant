# Aurelia Hotels BEO Assistant: Quick Start & Demo Guide

## Phase 1: Fresh Installation & Database Setup (Do this once)

1. **Activate your environment:** Open a terminal at the project root (`BEO-Assistant`).
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   pip install fastapi uvicorn python-dotenv
   ```
2. **Verify API Key:** Ensure your `.env` file is in the root directory with:
   ```env
   GROQ_API_KEY=your_key_here
   MODEL_NAME=openai/gpt-oss-120b
   ```
3. **Seed the fresh database & RAG:** 
   ```bash
   python db/setup_db.py
   python populate_rag.py
   ```
   *(This creates clean `EVT_1`, `EVT_2`, and `EVT_3` environments).*

---

## Phase 2: Booting the System (The 3 Terminals)
Open three separate terminal windows. Ensure your virtual environment (`.\venv\Scripts\activate`) is active in all three!

### 🖥️ Terminal 1: The MCP Server (Run from project root)
```bash
set MCP_TRANSPORT=http
python -m mcp_server.server
```

### 🖥️ Terminal 2: The FastAPI Backend (Run from platform folder)
```bash
cd platform
uvicorn main:app --port 8000 --reload
```

### 🖥️ Terminal 3: The Next.js Frontend (Run from platform folder)
```bash
cd platform
npm install
npm run dev
```

---

## Phase 3: The Live Presentation Script
Open your browser. Keep two tabs open side-by-side or easily switchable:
*   **User UI:** http://localhost:3000/chat
*   **Admin UI:** http://localhost:3000/admin/tickets

### Demo 1: VIP Dietary Handoff (Focus: LATS & Life-Safety HITL)
*Proves: LATS, Tool Calling, and `interrupt_before` pauses.*

*   **In User UI:** Select **VIP Dietary Handoff**.
*   **Input:** Event ID: `EVT_1` | Guest ID: `GUEST_VIP_1` -> Click **Start**.
*   **What's happening:** *"For life-safety constraints like severe allergies, we don't trust a single-pass LLM. The agent uses LATS to explore candidate menus, checks actual inventory via an MCP tool, and then hits a mandatory `interrupt_before` pause because Executive Chef sign-off is required."*
*   **Admin UI:** Go to Tickets. Show the `pending_admin` ticket. Click **Approve**.
*   **User UI:** Switch back. Show the chat automatically updating to *"Confirmed dishes..."*.

### Demo 2: Post-Event Billing Dispute (Focus: Task Decomp & ToT)
*Proves: Task Decomposition, Tree of Thoughts, and User Negotiation Deadlocks.*

*   **In User UI:** Select **Post-Event Billing**.
*   **Input:** Event ID: `EVT_2` -> Click **Start**.
*   **What's happening:** *"The agent uses Task Decomposition to safely compute the ledger and generates an invoice. Let's dispute it."*
*   **User UI:** Type: `"The headcount looks way too high, I dispute this."`
*   **What's happening:** *"The agent uses Tree of Thoughts to generate multiple negotiation strategies, scores them based on the discrepancy, and drafts a response."*
*   **User UI:** Type: `"Absolutely not. This is my final answer, I will not pay this."`
*   **What's happening:** *"The graph detects a deadlock. It locks the user out of the chat and escalates to Finance."*
*   **Admin UI:** Find the new ticket. Click **Reject**.
*   **User UI:** Refresh or watch it update to show the final resolution.

### Demo 3: Vendor Logistics (Focus: RAG & Asynchronous Webhooks)
*Proves: RAG, sleeping for external systems, and cross-platform state injection.*

*   **In User UI:** Select **Vendor Logistics**.
*   **Input:** Event ID: `EVT_3` | Vendor: `Acme Linens` | Goal: `Need 250 linens` | Budget: `1000` -> Click **Start**.
*   **What's happening:** *"The agent uses RAG to pull Acme Linens' policies, drafts the request, sends it, and goes to sleep, freeing up compute. Let's simulate the vendor's webhook replying with a quote that is over our $1000 budget."*
*   **Terminal (Root):** Open a 4th terminal and run:
    ```bash
    python simulate_vendor.py EVT_3 1500
    ```
*   **What's happening:** The simulator script acts as an asynchronous webhook. It accesses the SQLite checkpointer, finds the sleeping LangGraph thread for `EVT_3`, and injects the $1500 quote directly into the agent's memory. When the graph wakes up, its internal logic detects the quote exceeds the $1000 budget, so it safely halts execution and opens a `pending_admin` ticket instead of finalizing the contract.
*   **User UI:** The Agent will update the message to *"Waiting on ticket TICKET_### — checking automatically"*.
*   **Admin UI:** Find the ticket. Click **Approve**.
*   **User UI:** Show the logistics are successfully finalized.