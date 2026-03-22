import pandas as pd
import json
import random
import os
import re

random.seed(42)

print("=" * 60)
print("STEP 2: DATA CLEANING AND PREPARATION")
print("=" * 60)

os.makedirs('data_classifier', exist_ok=True)
os.makedirs('data_chatbot_raw', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION A: LOAD AND STANDARDIZE ALL THREE FILES
# Goal: get every file to two columns → [label, text]
# ─────────────────────────────────────────────────────────────

print("\n── Loading files ──")

# File 1: already clean, just rename
df1 = pd.read_csv('mental_health_balanced (1).csv')
df1 = df1.rename(columns={'condition': 'label', 'text': 'text'})
df1 = df1[['label', 'text']]
print(f"File 1 loaded: {len(df1)} rows")

# File 2: drop the useless index column, rename
df2 = pd.read_csv('cleanData.csv')
df2 = df2.rename(columns={'status': 'label', 'statement': 'text'})
df2 = df2[['label', 'text']]
print(f"File 2 loaded: {len(df2)} rows")

# File 3: complex salvage — handled separately below
df3_raw = pd.read_csv('labeled.csv')
df3_raw = df3_raw.rename(columns={'label': 'label', 'text': 'text'})
df3_raw = df3_raw[['label', 'text']]
print(f"File 3 loaded: {len(df3_raw)} rows (needs salvage)")

# ─────────────────────────────────────────────────────────────
# SECTION B: NORMALIZE LABEL NAMES
# Strip whitespace, consistent casing, fix known variations
# ─────────────────────────────────────────────────────────────

print("\n── Normalizing labels ──")

# Master normalization map — catches every variation across files
LABEL_NORMALIZE = {
    # File 2 variations
    'personality disorder':   'PersonalityDisorder',
    'Personality disorder':   'PersonalityDisorder',
    'Personality Disorder':   'PersonalityDisorder',

    # File 1 variations (already clean but normalize casing)
    'EatingDisorder':         'EatingDisorder',
    'Eating Disorder':        'EatingDisorder',

    # Keep everything else as title-cased
}

def normalize_label(label):
    if not isinstance(label, str):
        return None
    label = label.strip()
    # Check explicit map first
    if label in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[label]
    # Otherwise title-case (Depression, Anxiety, Suicidal etc.)
    return label.strip()

df1['label'] = df1['label'].apply(normalize_label)
df2['label'] = df2['label'].apply(normalize_label)

# ─────────────────────────────────────────────────────────────
# SECTION C: SALVAGE labeled.csv
# Two streams: valid labeled rows + keyword rescue of unlabeled
# ─────────────────────────────────────────────────────────────

print("\n── Salvaging labeled.csv ──")

# Map file 3's topic labels → our mental health conditions
# Only keep labels that have genuine mental health relevance
TOPIC_TO_CONDITION = {
    'Mental Health': 'Depression',  # generic MH posts → Depression
    'Death':         'Suicidal',    # death/grief themes → Suicidal
    'Academics':     'Stress',      # academic pressure → Stress
}

# Stream 1: labeled rows with a usable topic
df3_labeled = df3_raw[df3_raw['label'].notna()].copy()
df3_labeled = df3_labeled[df3_labeled['label'].isin(TOPIC_TO_CONDITION)]
df3_labeled['label'] = df3_labeled['label'].map(TOPIC_TO_CONDITION)
print(f"  Stream 1 (remapped labeled rows): {len(df3_labeled)}")

# Stream 2: unlabeled rows — scan for mental health keywords
# Order matters: Suicidal checked before Depression 
# (more specific signals checked first)
KEYWORD_MAP = {
    'Suicidal': [
        'suicid', 'kill myself', 'end my life', "don't want to live",
        'want to die', 'no reason to live', 'better off dead',
        'self harm', 'cutting myself', 'hurt myself',
        'not worth living', 'take my own life'
    ],
    'Depression': [
        'depressed', 'depression', 'hopeless', 'worthless',
        'empty inside', "can't get out of bed", 'no motivation',
        'nothing matters', 'feel numb', 'crying for no reason',
        'feel like a burden', 'pointless', 'no point',
        'exhausted all the time'
    ],
    'Anxiety': [
        'anxiety', 'anxious', 'panic attack', "can't stop worrying",
        'overthinking', 'nervous all the time', 'racing thoughts',
        'scared of everything', 'constant fear', 'social anxiety',
        'heart racing', 'can\'t breathe when'
    ],
    'PTSD': [
        'trauma', 'ptsd', 'flashback', 'nightmare', 'abused',
        "can't forget", 'keeps coming back', 'haunted by',
        'replaying', 'traumatic', 'abuse'
    ],
    'Bipolar': [
        'bipolar', 'manic', 'mania', 'mood swing',
        'extreme highs and lows', 'impulsive episode'
    ],
    'Stress': [
        'overwhelmed', 'burned out', 'burnout', 'stressed out',
        "can't cope", 'too much to handle', 'breaking point',
        'falling apart', "can't handle it", 'drowning in'
    ],
}

def classify_by_keywords(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for condition, keywords in KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return condition
    return None

df3_unlabeled = df3_raw[df3_raw['label'].isna()].copy()
df3_unlabeled['label'] = df3_unlabeled['text'].apply(classify_by_keywords)
df3_rescued = df3_unlabeled[df3_unlabeled['label'].notna()].copy()
print(f"  Stream 2 (keyword-rescued unlabeled): {len(df3_rescued)}")

# Merge both streams
df3_salvaged = pd.concat(
    [df3_labeled[['label', 'text']], df3_rescued[['label', 'text']]],
    ignore_index=True
)
print(f"  Total salvaged from File 3: {len(df3_salvaged)}")
print(f"  Distribution:\n{df3_salvaged['label'].value_counts()}")

# ─────────────────────────────────────────────────────────────
# SECTION D: MERGE ALL THREE SOURCES
# ─────────────────────────────────────────────────────────────

print("\n── Merging all sources ──")

df = pd.concat([df1, df2, df3_salvaged], ignore_index=True)
print(f"Total rows before cleaning: {len(df)}")

# ─────────────────────────────────────────────────────────────
# SECTION E: TEXT CLEANING
# ─────────────────────────────────────────────────────────────

print("\n── Cleaning text ──")

def clean_text(text):
    if not isinstance(text, str):
        return None
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove Reddit artifacts
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove markdown formatting (bold, italic, headers)
    text = re.sub(r'\*+|#+|_{2,}', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text if text else None

df['text'] = df['text'].apply(clean_text)

# ─────────────────────────────────────────────────────────────
# SECTION F: QUALITY FILTERS
# ─────────────────────────────────────────────────────────────

print("\n── Applying quality filters ──")

initial = len(df)

# 1. Drop rows with null text or label
df = df.dropna(subset=['text', 'label'])
print(f"  After null drop:        {len(df)} (removed {initial - len(df)})")

# 2. Drop texts that are too short — less than 8 words is not
#    enough signal for either classification or chatbot training
before = len(df)
df = df[df['text'].str.split().str.len() >= 8]
print(f"  After min-length (8w):  {len(df)} (removed {before - len(df)})")

# 3. Drop texts that are too long — over 300 words exceeds our
#    MLX sequence length and creates memory pressure during training
before = len(df)
df = df[df['text'].str.split().str.len() <= 300]
print(f"  After max-length (300w):{len(df)} (removed {before - len(df)})")

# 4. Remove exact duplicates (same text appearing in multiple files)
before = len(df)
df = df.drop_duplicates(subset='text')
print(f"  After dedup:            {len(df)} (removed {before - len(df)})")

# 5. Only keep known labels (catches any stray normalization misses)
VALID_LABELS = {
    'Depression', 'Anxiety', 'Suicidal', 'Bipolar', 'Stress',
    'Normal', 'PTSD', 'Loneliness', 'ADHD', 'Schizophrenia',
    'OCD', 'EatingDisorder', 'Addiction', 'PersonalityDisorder'
}
before = len(df)
df = df[df['label'].isin(VALID_LABELS)]
print(f"  After valid-label filter:{len(df)} (removed {before - len(df)})")

print(f"\nFinal merged + cleaned dataset: {len(df)} rows")
print(f"\nFull label distribution:")
print(df['label'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# SECTION G: CLASS BALANCING
# 
# Why balance? If Normal has 10,000 rows and Loneliness has 500,
# the model learns to always predict Normal. It would get ~20%
# accuracy just by guessing Normal every time.
#
# Strategy: cap each class at max_per_class
# We pick 2000 — enough data per class, not so much that
# common classes dominate. Rarer classes keep all their rows.
# ─────────────────────────────────────────────────────────────

print("\n── Balancing classes ──")

MAX_PER_CLASS = 2000

balanced = pd.concat([
    group.sample(min(len(group), MAX_PER_CLASS), random_state=42)
    for _, group in df.groupby('label')
], ignore_index=True)

print(f"After balancing (max {MAX_PER_CLASS}/class):")
print(balanced['label'].value_counts().to_string())
print(f"\nTotal balanced dataset: {len(balanced)} rows")

# ─────────────────────────────────────────────────────────────
# SECTION H: SAVE THE MASTER CLEANED CSV
# This is the single source of truth for both Model A and B
# ─────────────────────────────────────────────────────────────

balanced.to_csv('master_cleaned.csv', index=False)
print(f"\nSaved: master_cleaned.csv ({len(balanced)} rows)")

# ─────────────────────────────────────────────────────────────
# SECTION I: BUILD CLASSIFIER JSONL FILES (Model A)
#
# Format: chat — system sets the task, user provides text,
# assistant outputs the label. One word output = easy to parse
# at inference time.
# ─────────────────────────────────────────────────────────────

print("\n── Building classifier dataset (Model A) ──")

CLASSES = sorted(balanced['label'].unique().tolist())

CLASSIFIER_SYSTEM = (
    "You are a mental health text classifier. "
    "Read the given text and classify it into exactly one of "
    f"these categories: {', '.join(CLASSES)}. "
    "Respond with only the category name, nothing else. "
    "No explanation, no punctuation, just the category."
)

def to_classifier_format(row):
    return {
        "messages": [
            {
                "role": "system",
                "content": CLASSIFIER_SYSTEM
            },
            {
                "role": "user",
                "content": f"Classify this text:\n\n{row['text'].strip()}"
            },
            {
                "role": "assistant",
                "content": row['label']
            }
        ]
    }

classifier_records = [to_classifier_format(r) for _, r in balanced.iterrows()]

# Shuffle before splitting — crucial so all classes appear in
# train, valid, and test rather than clustering by class
random.shuffle(classifier_records)

n = len(classifier_records)
train_end = int(n * 0.80)
valid_end = int(n * 0.90)

clf_train = classifier_records[:train_end]
clf_valid = classifier_records[train_end:valid_end]
clf_test  = classifier_records[valid_end:]

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Written: {path} ({len(data)} rows)")

write_jsonl('data_classifier/train.jsonl', clf_train)
write_jsonl('data_classifier/valid.jsonl', clf_valid)
write_jsonl('data_classifier/test.jsonl',  clf_test)

# Save class list — needed at inference time to validate predictions
with open('data_classifier/classes.json', 'w') as f:
    json.dump(CLASSES, f, indent=2)
print(f"  Saved: data_classifier/classes.json → {CLASSES}")

# ─────────────────────────────────────────────────────────────
# SECTION J: BUILD CHATBOT SOURCE FILE (for Step 3 generation)
#
# We DON'T build the chatbot JSONL here because we need 
# synthetic replies first (Ollama generates those in Step 3).
# Here we just save the cleaned distress texts that aren't
# "Normal" — those become the USER side of the chatbot pairs.
# ─────────────────────────────────────────────────────────────

print("\n── Preparing chatbot source texts (Model B) ──")

# Exclude "Normal" — chatbot should respond to distress, not everyday text
chatbot_source = balanced[balanced['label'] != 'Normal'].copy()

# For chatbot training, we want a focused set of the most
# representative texts — not all 20,000+. 
# 3000 is enough for good chatbot fine-tuning, and generation
# takes time locally (Ollama will generate ~3000 replies)
MAX_CHATBOT_EXAMPLES = 3000
per_class = MAX_CHATBOT_EXAMPLES // len(chatbot_source['label'].unique())
chatbot_source = pd.concat([
    group.sample(min(len(group), per_class), random_state=42)
    for _, group in chatbot_source.groupby('label')
], ignore_index=True)

chatbot_source.to_csv('data_chatbot_raw/source_texts.csv', index=False)
print(f"  Saved: data_chatbot_raw/source_texts.csv ({len(chatbot_source)} rows)")
print(f"  Distribution:")
print(chatbot_source['label'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Source data:
  File 1 (mental_health_balanced):  13,000 rows
  File 2 (cleanData):               52,680 rows
  File 3 (labeled — salvaged):      ~100-300 rows

After merge + clean + balance:
  master_cleaned.csv:               {len(balanced)} rows

Model A (Classifier) dataset:
  data_classifier/train.jsonl:      {len(clf_train)} examples
  data_classifier/valid.jsonl:      {len(clf_valid)} examples
  data_classifier/test.jsonl:       {len(clf_test)} examples
  Classes:                          {len(CLASSES)} labels

Model B (Chatbot) source:
  data_chatbot_raw/source_texts.csv:{len(chatbot_source)} texts
  → Synthetic replies generated in Step 3

Next step: Step 3 — Synthetic reply generation via Ollama
""")