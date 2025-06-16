from log_config import setup_logging
import logging
from lang_model import LangModel
import uvicorn
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

setup_logging()
load_dotenv()

API_PORT = int(os.getenv("API_PORT"))
FRONT_PORT = int(os.getenv("FRONT_PORT"))

logger = logging.getLogger(__name__)
logger.info("Старт сервера")

app = FastAPI(title="Bisty API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{FRONT_PORT}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

AI = LangModel()

@app.get("/text")
async def text_chat():
    return FileResponse("static/text_chat.html")

@app.get("/model")
async def text_chat():
    return FileResponse("static/model_chat.html")

@app.post("/api/chat")
async def get_answer(request: ChatRequest):
    try:
        return {
            "response": AI.ask(request.message)
        }
    except Exception as err:
        logger.error(f"Ошибка при выволнении запроса: {err}")
        raise HTTPException(status_code=500, detail="Server-side error")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=FRONT_PORT, reload=True)
