# platform/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from admin_api import router  # This imports the router from the previous step
from chat_api import router as chat_router  # Issue #74: end-user chat interface

app = FastAPI()

# Allow your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the routers
app.include_router(router, prefix="/api/admin")
app.include_router(chat_router, prefix="/api/chat")