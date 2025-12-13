from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your main app
from app import app as fastapi_app

# Create Vercel-compatible app
app = FastAPI()

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your FastAPI app
app.mount("/", fastapi_app)

# Vercel handler
def handler(request, context):
    return app(request, context)