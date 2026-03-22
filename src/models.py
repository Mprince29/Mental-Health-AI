# src/models.py
from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum

class Role(str, Enum):
    user      = "user"
    assistant = "assistant"

class Message(BaseModel):
    role:    Role
    content: str

# ── Request models ────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 2000:
            raise ValueError("Text too long — max 2000 characters")
        return v.strip()

class SupportRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 2000:
            raise ValueError("Text too long — max 2000 characters")
        return v.strip()

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None  # for conversation history

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class BatchClassifyRequest(BaseModel):
    texts: List[str]

    @field_validator('texts')
    @classmethod
    def validate_batch(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > 20:
            raise ValueError("Max 20 texts per batch")
        return [t.strip() for t in v if t.strip()]

# ── Response models ───────────────────────────────────────────

class ClassifyResponse(BaseModel):
    text:      str
    condition: str

class SupportResponse(BaseModel):
    text:               str
    detected_condition: str
    response:           str
    response_time_ms:   int
    in_scope:           bool

class ChatResponse(BaseModel):
    text:               str
    response:           str
    session_id:         Optional[str]
    history_length:     int
    detected_condition: Optional[str]

class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]
    total:   int

class HealthResponse(BaseModel):
    status:     str
    classifier: str
    chatbot:    str
    classes:    List[str]
    version:    str