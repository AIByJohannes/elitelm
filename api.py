import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Literal, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from elitelm import ChatSession, load_config, AppConfig

# --- Pydantic Models for OpenAI API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "default"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[dict] = None

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

# --- Global State & Lifespan ---

class AppState:
    session: Optional[ChatSession] = None
    lock: asyncio.Lock = asyncio.Lock()

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("ELITELM_CONFIG", "llama3-qa.yaml")
    print(f"Loading configuration from {config_path}...")
    try:
        config = load_config(config_path)
        # Override config with env vars if needed, or just rely on yaml
        state.session = ChatSession(config)
        print(f"Model loaded on {state.session.device_label}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e
    yield
    # Cleanup if needed
    state.session = None

app = FastAPI(title="EliteLM API", lifespan=lifespan)

# --- Endpoints ---

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if state.session is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    async with state.lock:
        # Prepare session state
        state.session.reset_history()
        
        # Extract user message and history
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        last_message = request.messages[-1]
        if last_message.role != "user":
             # Handle case where last message is not user? 
             # For now assume standard chat format ending with user
             pass

        # Populate history with previous messages
        history_dicts = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
        state.session.chat_history.extend(history_dicts)
        
        user_text = last_message.content

        # Prepare generation arguments
        gen_kwargs = {}
        if request.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            gen_kwargs["top_p"] = request.top_p
        
        if request.stream:
            return StreamingResponse(
                stream_generator(user_text, request.model or "default", **gen_kwargs),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming
            result = await asyncio.to_thread(
                state.session.generate, 
                user_text, 
                **gen_kwargs
            )
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=request.model or "default",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result.text),
                        finish_reason="length" if result.interrupted else "stop"
                    )
                ],
                usage={
                    "prompt_tokens": result.stats.prompt_length if result.stats else 0,
                    "completion_tokens": result.stats.new_tokens if result.stats else 0,
                    "total_tokens": (result.stats.prompt_length + result.stats.new_tokens) if result.stats else 0
                }
            )
            return response


async def stream_generator(user_text: str, model_name: str, **gen_kwargs) -> AsyncGenerator[str, None]:
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def on_token(token: str):
        loop.call_soon_threadsafe(queue.put_nowait, token)

    def run_gen():
        try:
            state.session.generate(user_text, on_token=on_token, **gen_kwargs)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start generation in a separate thread
    asyncio.create_task(asyncio.to_thread(run_gen))

    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    while True:
        token = await queue.get()
        if token is None:
            break
        
        chunk = ChatCompletionChunk(
            id=chat_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=token),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=chat_id,
        created=created,
        model=model_name,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
