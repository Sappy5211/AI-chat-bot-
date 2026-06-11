from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from manualmind.chat_store import (
    append_message,
    clear_all_chats,
    create_chat,
    delete_chat,
    get_chat,
    load_chat_store,
    rename_chat,
)
from manualmind.ingestion import (
    clear_manual_records,
    ingest_uploaded_pdfs,
    list_loaded_manuals,
    remove_manual_record,
)
from manualmind.llm import get_active_route_details
from manualmind.retrieval import clear_documents, remove_documents_for_file, store_documents
from manualmind.service import answer_query


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web"
EXTRACTED_DIR = BASE_DIR / "extracted"
UPLOADS_DIR = BASE_DIR / "uploads"

STATIC_DIR.mkdir(exist_ok=True)
EXTRACTED_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ManualMind", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/files/extracted", StaticFiles(directory=EXTRACTED_DIR), name="extracted")
app.mount("/files/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


class BufferedUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._buffer = BytesIO(content)

    def getbuffer(self):
        return self._buffer.getbuffer()


def _serialize_sources(sources: list[dict]) -> list[dict]:
    serialized = []
    for source in sources:
        item = dict(source)
        image_path = item.get("image_path")
        if image_path:
            item["image_url"] = f"/files/extracted/{Path(image_path).name}"
        else:
            item["image_url"] = ""
        serialized.append(item)
    return serialized


def _serialize_chat(chat: dict) -> dict:
    messages = []
    for message in chat.get("messages", []):
        message_copy = dict(message)
        message_copy["sources"] = _serialize_sources(message_copy.get("sources", []))
        messages.append(message_copy)
    chat_copy = dict(chat)
    chat_copy["messages"] = messages
    return chat_copy


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/bootstrap")
def bootstrap():
    chats = [_serialize_chat(chat) for chat in load_chat_store()]
    return {
        "chats": chats,
        "manuals": list_loaded_manuals(),
        "route": get_active_route_details(),
    }


@app.post("/api/chats")
def api_create_chat():
    return _serialize_chat(create_chat())


@app.post("/api/chats/{chat_id}/rename")
async def api_rename_chat(chat_id: str, payload: dict):
    title = str(payload.get("title", "")).strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required.")
    try:
        chat = rename_chat(chat_id, title)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return _serialize_chat(chat)


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    chats = delete_chat(chat_id)
    return {"chats": [_serialize_chat(chat) for chat in chats]}


@app.post("/api/chats/clear")
def api_clear_chats():
    chats = clear_all_chats()
    return {"chats": [_serialize_chat(chat) for chat in chats]}


@app.post("/api/chats/{chat_id}/ask")
async def api_ask(chat_id: str, payload: dict):
    query = str(payload.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    if get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found.")

    append_message(chat_id, "user", query)
    answer = answer_query(query, chat_id=chat_id)
    answer["sources"] = _serialize_sources(answer.get("sources", []))
    chat = append_message(chat_id, "assistant", answer["content"], answer["sources"])
    return {"chat": _serialize_chat(chat), "answer": answer}


@app.post("/api/manuals/upload")
async def api_upload_manuals(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    buffered_uploads = []
    for upload in files:
        content = await upload.read()
        buffered_uploads.append(BufferedUpload(upload.filename, content))

    documents, stats, file_results = ingest_uploaded_pdfs(buffered_uploads)
    if documents:
        store_documents(documents)

    return {
        "stats": asdict(stats),
        "file_results": [asdict(result) for result in file_results],
        "manuals": list_loaded_manuals(),
    }


@app.delete("/api/manuals/{file_id}")
def api_remove_manual(file_id: str):
    removed = remove_manual_record(file_id)
    if removed:
        remove_documents_for_file(file_id)
    return {"manuals": list_loaded_manuals()}


@app.post("/api/manuals/clear")
def api_clear_manuals():
    removed_records = clear_manual_records()
    if removed_records:
        clear_documents()
    return {"manuals": list_loaded_manuals()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=False)
