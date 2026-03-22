# src/app.py
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import config
from src.logger import get_logger
from src.models import (
    ClassifyRequest, ClassifyResponse,
    SupportRequest,  SupportResponse,
    ChatRequest,     ChatResponse,
    BatchClassifyRequest, BatchClassifyResponse,
    HealthResponse, Message
)

logger = get_logger("api")

# ── In-memory conversation store ──────────────────────────────
# Maps session_id → list of messages
# In production, replace with Redis or a database
conversation_store: Dict[str, List[Message]] = {}

# ── ML models (loaded at startup) ────────────────────────────
classifier_model    = None
classifier_tokenizer = None
chatbot_model       = None
chatbot_tokenizer   = None

# ── System prompts ────────────────────────────────────────────
CLASSIFIER_SYSTEM = (
    "You are a mental health text classifier. "
    "Read the given text and classify it into exactly one of "
    f"these categories: {', '.join(config.CLASSES)}. "
    "Respond with only the category name, nothing else. "
    "No explanation, no punctuation, just the category."
)

CHATBOT_SYSTEM = (
    "You are a compassionate support companion. "
    "You listen carefully and respond like a caring human friend — "
    "warm, brief, and genuine. You never give unsolicited advice. "
    "You never dismiss emotions. You respond in 2-4 sentences."
)

# ── Lifespan — load models once at startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_model, classifier_tokenizer
    global chatbot_model, chatbot_tokenizer

    from mlx_lm import load
    logger.info("Loading classifier model...")
    classifier_model, classifier_tokenizer = load(config.CLASSIFIER_MODEL)
    logger.info(f"Classifier loaded from {config.CLASSIFIER_MODEL}")

    logger.info("Loading chatbot model...")
    chatbot_model, chatbot_tokenizer = load(config.CHATBOT_MODEL)
    logger.info(f"Chatbot loaded from {config.CHATBOT_MODEL}")

    logger.info("Both models ready. API is live.")
    yield

    logger.info("Shutting down...")

# ── App init ──────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health AI API",
    description=(
        "A two-model NLP system for mental health text analysis.\n\n"
        "**Models:**\n"
        "- Classifier: Qwen2.5-3B fine-tuned → 14-class condition detection\n"
        "- Chatbot: Qwen2.5-3B fine-tuned → empathetic support responses\n\n"
        "**Built with:** MLX-LM, LoRA fine-tuning, FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} [{duration_ms}ms]"
    )
    return response

# ── Global error handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# ── Core ML functions ─────────────────────────────────────────
def classify_text(text: str) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user",   "content": f"Classify this text:\n\n{text}"}
    ]
    prompt = classifier_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        classifier_model, classifier_tokenizer,
        prompt=prompt,
        max_tokens=10,
        sampler=make_sampler(temp=config.CLASSIFIER_TEMP),
        verbose=False
    )
    prediction = response.strip().split()[0].strip(".,!?")
    if prediction not in config.CLASSES:
        for cls in config.CLASSES:
            if cls.lower() in response.lower():
                return cls
        return "Unknown"
    return prediction

def generate_response(
    text: str,
    condition: str,
    history: List[Message] = None
) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    # Build system prompt based on condition
    if condition in config.NON_CLINICAL:
        system = (
            "You are a warm, thoughtful friend. "
            "Respond naturally and briefly. "
            "You can offer perspective or ask questions."
        )
    else:
        system = CHATBOT_SYSTEM
        if condition and condition != "Unknown":
            system += f" The person may be experiencing: {condition}."

    # Build messages with history
    messages = [{"role": "system", "content": system}]

    if history:
        # Include last N turns for context
        for msg in history[-(config.MAX_HISTORY_TURNS * 2):]:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": text})

    prompt = chatbot_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        chatbot_model, chatbot_tokenizer,
        prompt=prompt,
        max_tokens=config.CHATBOT_MAX_TOKENS,
        sampler=make_sampler(temp=config.CHATBOT_TEMP),
        verbose=False
    )
    return response.strip()

# ── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "name":      "Mental Health AI API",
        "version":   "1.0.0",
        "docs":      "/docs",
        "endpoints": [
            "GET  /health",
            "POST /classify",
            "POST /support",
            "POST /chat",
            "POST /batch-classify",
            "GET  /history/{session_id}",
            "DELETE /history/{session_id}"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status     = "ok",
        classifier = config.CLASSIFIER_MODEL,
        chatbot    = config.CHATBOT_MODEL,
        classes    = config.CLASSES,
        version    = "1.0.0"
    )

@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(request: ClassifyRequest):
    """
    Classify a piece of text into one of 14 mental health conditions.

    Returns a single condition label from:
    ADHD, Addiction, Anxiety, Bipolar, Depression, EatingDisorder,
    Loneliness, Normal, OCD, PTSD, PersonalityDisorder,
    Schizophrenia, Stress, Suicidal
    """
    logger.info(f"Classify request: {request.text[:60]}...")
    condition = classify_text(request.text)
    logger.info(f"Classified as: {condition}")
    return ClassifyResponse(text=request.text, condition=condition)

@app.post("/support", response_model=SupportResponse, tags=["Support"])
def support(request: SupportRequest):
    """
    Full pipeline: classify the text then generate an empathetic response.

    - Detects the mental health condition automatically
    - Generates a response calibrated to that condition
    - Returns in_scope=false if text is not mental health related
    """
    logger.info(f"Support request: {request.text[:60]}...")
    start = time.time()

    condition = classify_text(request.text)
    in_scope  = condition not in config.NON_CLINICAL

    if not in_scope:
        response_text = (
            "This doesn't seem to be related to mental health. "
            "This system is designed to support people dealing with "
            "conditions like anxiety, depression, stress, and similar. "
            "Is there something deeper going on you'd like to talk about?"
        )
    else:
        response_text = generate_response(request.text, condition)

    duration_ms = int((time.time() - start) * 1000)
    logger.info(f"Support response in {duration_ms}ms | condition={condition}")

    return SupportResponse(
        text               = request.text,
        detected_condition = condition,
        response           = response_text,
        response_time_ms   = duration_ms,
        in_scope           = in_scope
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Conversational chat with memory across multiple turns.

    - Pass session_id to continue a conversation
    - Leave session_id empty to start a new conversation
    - History is kept for up to 5 turns (configurable)
    - Returns session_id to use in subsequent requests
    """
    # Create or retrieve session
    session_id = request.session_id or str(uuid.uuid4())
    history    = conversation_store.get(session_id, [])

    logger.info(
        f"Chat request | session={session_id} | "
        f"history_turns={len(history)//2} | "
        f"text={request.text[:60]}..."
    )

    # Classify to add context
    condition = classify_text(request.text)

    # Generate response with history
    response_text = generate_response(
        request.text, condition, history
    )

    # Update history
    history.append(Message(role="user",      content=request.text))
    history.append(Message(role="assistant", content=response_text))

    # Trim history to max turns
    max_messages = config.MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history = history[-max_messages:]

    conversation_store[session_id] = history

    logger.info(
        f"Chat response | session={session_id} | "
        f"condition={condition} | history_len={len(history)}"
    )

    return ChatResponse(
        text               = request.text,
        response           = response_text,
        session_id         = session_id,
        history_length     = len(history) // 2,
        detected_condition = condition
    )

@app.post(
    "/batch-classify",
    response_model=BatchClassifyResponse,
    tags=["Classification"]
)
def batch_classify(request: BatchClassifyRequest):
    """
    Classify multiple texts in one request.

    - Max 20 texts per request
    - Returns condition label for each text
    - Useful for bulk analysis of datasets
    """
    logger.info(f"Batch classify: {len(request.texts)} texts")
    results = []
    for text in request.texts:
        condition = classify_text(text)
        results.append(ClassifyResponse(text=text, condition=condition))
    logger.info(f"Batch complete: {len(results)} classified")
    return BatchClassifyResponse(results=results, total=len(results))

@app.get("/history/{session_id}", tags=["Chat"])
def get_history(session_id: str):
    """
    Retrieve full conversation history for a session.
    """
    history = conversation_store.get(session_id)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session: {session_id}"
        )
    return {
        "session_id":    session_id,
        "total_turns":   len(history) // 2,
        "messages":      [m.model_dump() for m in history]
    }

@app.delete("/history/{session_id}", tags=["Chat"])
def clear_history(session_id: str):
    """
    Clear conversation history for a session.
    """
    if session_id not in conversation_store:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session: {session_id}"
        )
    del conversation_store[session_id]
    logger.info(f"Cleared history for session: {session_id}")
    return {"message": f"History cleared for session: {session_id}"}