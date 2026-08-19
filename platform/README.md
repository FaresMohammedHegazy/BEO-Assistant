### Step 1: Prerequisites & Dependencies

1. Ensure **Python** and **Node.js** (with `npm`) are installed on your machine.
2. Open your terminal at the root of the repository (`E:\BEO-Assistant-main`) and activate your virtual environment:
```bash
venv\Scripts\activate

```


3. Install the required Python packages (including the new `fastapi` and `uvicorn` web server packages):
```bash
pip install fastapi uvicorn

```



---

### Step 2: Initialize the Database

1. Run the database setup script to create the SQLite database (`db/aurelia.db`) and seed the initial tables (including the `agent_tools` table and LangGraph checkpointer tables):
```bash
python db/setup_db.py

```


2. You should see a success message indicating the database was created and seeded.

---

### Step 3: Run the FastAPI Backend

1. Navigate into the `platform` directory:
```bash
cd platform

```


2. Start the local Python backend server using Uvicorn:
```bash
uvicorn main:app --port 8000 --reload

```


3. Leave this terminal open. It will listen for API requests on `[http://127.0.0.1:8000]`.

---

### Step 4: Install Frontend Dependencies

1. Open a **second, separate terminal** window.
2. Navigate to the `platform` folder:
```bash
cd platform

```


3. Install the Node.js dependencies (Next.js and React):
```bash
npm install

```



---

### Step 5: Run the Next.js Frontend

1. In that same second terminal (inside the `platform` folder), start the Next.js development server:
```bash
npm run dev

```


2. The server will start locally at **`http://localhost:3000`**.

---

### Step 6: Access the Admin Dashboard

1. Open your web browser and navigate to:
**`http://localhost:3000/admin`**