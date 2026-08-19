# platform/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from admin_api import router  # This imports the router from the previous step

app = FastAPI()

# Allow your Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the router
app.include_router(router, prefix="/api/admin")