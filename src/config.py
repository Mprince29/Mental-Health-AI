# src/config.py
import os

class Config:
    # Model paths
    CLASSIFIER_MODEL   = os.getenv("CLASSIFIER_MODEL", "./fused_classifier")
    CHATBOT_MODEL      = os.getenv("CHATBOT_MODEL",    "./fused_chatbot")

    # Generation settings
    CLASSIFIER_TEMP    = float(os.getenv("CLASSIFIER_TEMP", "0.1"))
    CHATBOT_TEMP       = float(os.getenv("CHATBOT_TEMP",    "0.8"))
    CHATBOT_MAX_TOKENS = int(os.getenv("CHATBOT_MAX_TOKENS", "150"))

    # Conversation history
    MAX_HISTORY_TURNS  = int(os.getenv("MAX_HISTORY_TURNS", "5"))

    # API
    HOST               = os.getenv("HOST", "0.0.0.0")
    PORT               = int(os.getenv("PORT", "8000"))

    # Valid condition classes
    CLASSES = [
        'ADHD', 'Addiction', 'Anxiety', 'Bipolar', 'Depression',
        'EatingDisorder', 'Loneliness', 'Normal', 'OCD', 'PTSD',
        'PersonalityDisorder', 'Schizophrenia', 'Stress', 'Suicidal'
    ]

    # Out of scope — won't use crisis chatbot tone
    NON_CLINICAL = {'Normal'}

config = Config()