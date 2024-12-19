from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="OpenAI Compatible API")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gpt-3.5-turbo"
    stream: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str
    content: str

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # TODO: implement actual LLM logic
    # This is a simple echo response for demonstration
    response_content = f"Echo: {request.messages[-1].content}"
    
    return {
        "id": "chatcmpl-" + str(int(time.time())),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_content
            },
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
