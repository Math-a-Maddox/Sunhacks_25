from fastapi import FastAPI
from pydantic import BaseModel

#initialize fastapi app
app = FastAPI(title = "Found and Lost")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = GeminiClient(api_key=GEMINI_API_KEY)
logger.info("Gemini client initialized successfully")
