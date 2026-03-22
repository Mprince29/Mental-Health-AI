import json
import random
import os

random.seed(42)

print("=" * 60)
print("STEP 4A: BUILD CHATBOT TRAINING DATASET")
print("=" * 60)

# ── Load generated replies ────────────────────────────────────
with open('data_chatbot_raw/all_replies.json') as f:
    data = json.load(f)

print(f"Loaded: {len(data)} reply pairs")

# ── System prompt ─────────────────────────────────────────────
# This becomes the permanent persona baked into the chatbot.
# Keep it short — Phi-3 Mini has a smaller context window.
SYSTEM_PROMPT = (
    "You are a compassionate support companion. "
    "You listen carefully and respond like a caring human friend — "
    "warm, brief, and genuine. You never give unsolicited advice. "
    "You never dismiss emotions. You respond in 2-4 sentences."
)

def to_chat_format(item):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": item['user_text'].strip()
            },
            {
                "role": "assistant",
                "content": item['assistant_reply'].strip()
            }
        ]
    }

records = [to_chat_format(item) for item in data]

# ── Shuffle and split 80/10/10 ────────────────────────────────
random.shuffle(records)
n = len(records)
train = records[:int(n * 0.80)]
valid = records[int(n * 0.80):int(n * 0.90)]
test  = records[int(n * 0.90):]

os.makedirs('data_chatbot', exist_ok=True)

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Written: {path} ({len(data)} rows)")

write_jsonl('data_chatbot/train.jsonl', train)
write_jsonl('data_chatbot/valid.jsonl', valid)
write_jsonl('data_chatbot/test.jsonl',  test)

# ── Preview ───────────────────────────────────────────────────
print("\n── Sample training example ──")
sample = train[0]
for msg in sample['messages']:
    print(f"[{msg['role'].upper()}]: {msg['content'][:120]}")

print(f"""
Summary:
  Train: {len(train)} examples
  Valid: {len(valid)} examples
  Test:  {len(test)} examples
  Total: {len(records)} pairs
""")