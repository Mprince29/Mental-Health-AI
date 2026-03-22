import pandas as pd
import json
import requests
import time
import os
from tqdm import tqdm

print("=" * 60)
print("STEP 3: SYNTHETIC REPLY GENERATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"
OUTPUT_FILE   = "data_chatbot_raw/all_replies.json"
CHECKPOINT_DIR = "data_chatbot_raw/checkpoints"
CHECKPOINT_EVERY = 50   # save progress every 50 examples

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD SOURCE TEXTS
# ─────────────────────────────────────────────────────────────

df = pd.read_csv('data_chatbot_raw/source_texts.csv')
print(f"Source texts loaded: {len(df)} rows")
print(f"Classes: {df['label'].unique().tolist()}")

# ─────────────────────────────────────────────────────────────
# RESUME LOGIC
# If the script was interrupted, pick up from where it stopped.
# This is critical for a 4-6 hour job — you don't want to 
# start from scratch if your Mac sleeps or Ollama crashes.
# ─────────────────────────────────────────────────────────────

completed_texts = set()

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        existing = json.load(f)
    completed_texts = {item['user_text'] for item in existing}
    results = existing
    print(f"\nResuming — {len(results)} already generated, "
          f"{len(df) - len(results)} remaining")
else:
    results = []
    print("\nStarting fresh generation...")

# Filter to only unprocessed rows
df_remaining = df[~df['text'].isin(completed_texts)].copy()
print(f"To generate: {len(df_remaining)} replies\n")

# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
#
# This is the most important piece. The prompt determines the 
# quality of EVERY training example. A bad prompt here means 
# a bad chatbot at the end.
#
# Design principles:
# 1. Condition-aware: different conditions need different tones
#    (Suicidal needs more careful handling than Stress)
# 2. Human-sounding: explicitly forbid AI phrases
# 3. Brief: 2-4 sentences max — chatbot responses should be
#    focused, not overwhelming when someone is distressed
# 4. Non-clinical: no advice unless asked, no "you should see
#    a therapist" in every reply
# ─────────────────────────────────────────────────────────────

# Condition-specific tone guidance
# These are injected into the prompt based on the condition label
CONDITION_TONE = {
    'Suicidal': (
        "This person may be in serious distress. "
        "Your response must be gentle, non-judgmental, and make them "
        "feel heard without minimizing what they feel. "
        "Do NOT give advice. Do NOT mention therapy or hotlines. "
        "Just make them feel less alone in this moment."
    ),
    'Depression': (
        "This person is struggling with depression. "
        "Acknowledge the heaviness they feel. Don't try to fix it or "
        "suggest solutions. Just validate that what they feel is real "
        "and that they're not weak for feeling it."
    ),
    'Anxiety': (
        "This person is experiencing anxiety. "
        "Acknowledge the worry without dismissing it. "
        "Don't tell them to 'calm down' or 'stop overthinking'. "
        "Make them feel understood."
    ),
    'PTSD': (
        "This person is dealing with trauma. "
        "Be gentle. Don't push them to share more than they have. "
        "Validate that their reactions make sense given what they've been through."
    ),
    'Bipolar': (
        "This person is dealing with bipolar disorder. "
        "Acknowledge the exhaustion of the highs and lows. "
        "Be grounded and steady in your response."
    ),
    'Loneliness': (
        "This person feels deeply lonely. "
        "Make them feel seen and less alone. "
        "A warm, genuine response matters more than any advice."
    ),
    'Stress': (
        "This person is overwhelmed with stress. "
        "Acknowledge the pressure they're under. "
        "You can be slightly more practical here but still lead with empathy."
    ),
    'EatingDisorder': (
        "This person is struggling with an eating disorder. "
        "Be very careful — do NOT comment on food, weight, or body. "
        "Focus only on the emotional pain they're expressing."
    ),
    'OCD': (
        "This person is struggling with OCD. "
        "Acknowledge how exhausting intrusive thoughts can be. "
        "Don't tell them their fears are irrational — they know, "
        "and it doesn't help."
    ),
    'ADHD': (
        "This person is struggling with ADHD-related challenges. "
        "Acknowledge the frustration without framing it as a personal failure. "
        "Be direct and warm."
    ),
    'Addiction': (
        "This person is dealing with addiction. "
        "Be non-judgmental. Acknowledge the struggle without shaming. "
        "Focus on the emotional pain, not the behavior."
    ),
    'Schizophrenia': (
        "This person is dealing with schizophrenia or psychosis. "
        "Be calm, steady, grounding. Don't validate delusions, but "
        "don't dismiss their distress either. Focus on their feelings."
    ),
    'PersonalityDisorder': (
        "This person has a personality disorder and may feel deeply misunderstood. "
        "Lead with acceptance. Make them feel their experience is valid."
    ),
}

DEFAULT_TONE = (
    "Acknowledge what they said and make them feel genuinely heard."
)

def build_prompt(text, condition):
    tone = CONDITION_TONE.get(condition, DEFAULT_TONE)
    
    return f"""You are a compassionate, empathetic person responding to someone who shared something difficult.

{tone}

They wrote:
\"\"\"{text}\"\"\"

Write a SHORT empathetic response (2-4 sentences). STRICT RULES:
- Sound like a genuine caring human, not a therapist, not an AI
- Do NOT start with "I" as the first word
- Do NOT use: "Certainly", "Of course", "As an AI", "It's important to", "I understand that"
- Do NOT give unsolicited advice or solutions
- Do NOT be preachy or clinical
- Match their emotional register — if they're raw, be real; if tentative, be gentle
- Vary your sentence length naturally

Reply ONLY with the response text. No preamble, no quotes around it."""

# ─────────────────────────────────────────────────────────────
# QUALITY FILTER
# Reject generated replies that are clearly AI-sounding or 
# off-target before they pollute the training data
# ─────────────────────────────────────────────────────────────

AI_PHRASES = [
    "as an ai", "i'm an ai", "i am an ai", "i cannot", "i don't have",
    "certainly", "of course!", "it's important to note",
    "i recommend", "you should consider", "please seek",
    "i understand that you", "i'm so sorry to hear",
    "that must be", "i can imagine how",
]

def is_quality_reply(reply, min_words=10, max_words=120):
    if not reply or not isinstance(reply, str):
        return False
    
    reply_stripped = reply.strip()
    word_count = len(reply_stripped.split())
    
    # Too short or too long
    if not (min_words <= word_count <= max_words):
        return False
    
    # Contains AI-sounding phrases
    reply_lower = reply_stripped.lower()
    if any(phrase in reply_lower for phrase in AI_PHRASES):
        return False
    
    # Starts with a quote mark (model wrapped its response in quotes)
    if reply_stripped.startswith('"') or reply_stripped.startswith("'"):
        return False
    
    return True

# ─────────────────────────────────────────────────────────────
# GENERATION LOOP
# ─────────────────────────────────────────────────────────────

def call_ollama(prompt, retries=3, timeout=90):
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.85,
                        "top_p":       0.92,
                        "num_predict": 150,
                        "stop":        ["\n\n", "Human:", "User:", "They wrote:"]
                    }
                },
                timeout=timeout
            )
            if resp.status_code == 200:
                return resp.json().get('response', '').strip()
        except requests.exceptions.Timeout:
            print(f"\n  Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"\n  Error on attempt {attempt + 1}: {e}")
            time.sleep(3)
    return None

skipped  = 0
accepted = 0
start_time = time.time()

for idx, row in tqdm(
    df_remaining.iterrows(),
    total=len(df_remaining),
    desc="Generating replies",
    unit="reply"
):
    text      = str(row['text']).strip()
    condition = str(row['label']).strip()
    
    prompt = build_prompt(text, condition)
    reply  = call_ollama(prompt)
    
    if reply and is_quality_reply(reply):
        results.append({
            "user_text":        text,
            "condition":        condition,
            "assistant_reply":  reply
        })
        accepted += 1
    else:
        skipped += 1
    
    # ── Checkpoint save every N examples ──────────────────────
    # If Ollama crashes or Mac sleeps, you keep everything so far
    if (accepted + skipped) % CHECKPOINT_EVERY == 0:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        elapsed  = time.time() - start_time
        rate     = (accepted + skipped) / elapsed
        remaining = len(df_remaining) - (accepted + skipped)
        eta_mins = (remaining / rate) / 60 if rate > 0 else 0
        
        tqdm.write(
            f"  Checkpoint: {accepted} accepted | "
            f"{skipped} skipped | "
            f"ETA: {eta_mins:.0f} min"
        )

# Final save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────

total_time = (time.time() - start_time) / 60
acceptance_rate = accepted / (accepted + skipped) * 100 if (accepted + skipped) > 0 else 0

print(f"""
{'=' * 60}
STEP 3 COMPLETE
{'=' * 60}
Total processed:   {accepted + skipped}
Accepted replies:  {accepted}  ({acceptance_rate:.1f}%)
Skipped (low quality): {skipped}
Time taken:        {total_time:.1f} minutes
Output:            {OUTPUT_FILE}

Condition breakdown:""")

condition_counts = {}
for item in results:
    c = item['condition']
    condition_counts[c] = condition_counts.get(c, 0) + 1

for condition, count in sorted(condition_counts.items()):
    print(f"  {condition:<25} {count}")

print(f"\nNext step: Step 4 — Build chatbot JSONL and start training")
