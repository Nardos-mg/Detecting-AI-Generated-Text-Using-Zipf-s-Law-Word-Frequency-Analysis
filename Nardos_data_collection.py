# ============================================================
# Nardos Ayele
# Data Collection Script
# Project: Detecting AI-Generated Text Using Zipf's Law
# ============================================================
# This script:
# 1. Downloads human texts from Project Gutenberg
# 2. Generates AI texts using Claude API
# 3. Saves everything to Nardos_data folder as CSV
# ============================================================

import requests
import random
import time
import os
import csv
import re

os.makedirs("Nardos_data", exist_ok=True)

# ============================================================
# PART 1: HUMAN TEXTS FROM PROJECT GUTENBERG
# ============================================================
# These are real book IDs from Project Gutenberg (free classics)

GUTENBERG_IDS = [
    1342,   # Pride and Prejudice - Jane Austen
    11,     # Alice in Wonderland - Lewis Carroll
    1661,   # Sherlock Holmes - Arthur Conan Doyle
    84,     # Frankenstein - Mary Shelley
    98,     # A Tale of Two Cities - Dickens
    2701,   # Moby Dick - Herman Melville
    1952,   # The Yellow Wallpaper
    74,     # Tom Sawyer - Mark Twain
    345,    # Dracula - Bram Stoker
    5200,   # Metamorphosis - Kafka
    1080,   # A Modest Proposal - Swift
    46,     # A Christmas Carol - Dickens
    2600,   # War and Peace - Tolstoy
    1400,   # Great Expectations - Dickens
    76,     # Adventures of Huckleberry Finn
    2554,   # Crime and Punishment
    1260,   # Jane Eyre - Charlotte Bronte
    4300,   # Ulysses - James Joyce
    174,    # The Picture of Dorian Gray
    16,     # Peter Pan
]

def download_gutenberg(book_id):
    """Download a book from Project Gutenberg."""
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
        except:
            continue
    return None

def clean_gutenberg(text):
    """Remove Gutenberg header/footer boilerplate."""
    start_markers = ["*** START OF", "***START OF", "** START OF"]
    end_markers   = ["*** END OF",   "***END OF",   "** END OF"]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            text = text[text.find('\n')+1:]
            break
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

def extract_chunks(text, chunk_size=400, num_chunks=25):
    """Split text into chunks of approximately chunk_size words."""
    words  = text.split()
    chunks = []
    start  = 0
    while start + chunk_size <= len(words) and len(chunks) < num_chunks:
        chunk = ' '.join(words[start:start+chunk_size])
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(chunk.split()) >= 300:
            chunks.append(chunk)
        start += chunk_size
    return chunks

print("=" * 60)
print("  NARDOS AYELE — DATA COLLECTION SCRIPT")
print("=" * 60)
print("\nPART 1: Downloading human texts from Project Gutenberg...")
print("-" * 60)

human_texts = []

for book_id in GUTENBERG_IDS:
    if len(human_texts) >= 500:
        break
    print(f"  Downloading book ID {book_id}...", end=" ")
    raw = download_gutenberg(book_id)
    if raw:
        cleaned = clean_gutenberg(raw)
        chunks  = extract_chunks(cleaned, chunk_size=400, num_chunks=30)
        human_texts.extend(chunks)
        print(f"✓ got {len(chunks)} chunks (total: {len(human_texts)})")
    else:
        print("✗ failed")
    time.sleep(1)  # be polite to Gutenberg servers

# Trim to exactly 500
human_texts = human_texts[:500]
print(f"\n✓ Total human texts collected: {len(human_texts)}")

# ============================================================
# PART 2: AI TEXTS USING CLAUDE API
# ============================================================

print("\nPART 2: Generating AI texts using Claude API...")
print("-" * 60)

# Topics for AI text generation — varied to match human text diversity
TOPICS = [
    "the importance of education in modern society",
    "how technology has changed communication",
    "the effects of climate change on daily life",
    "the role of government in healthcare",
    "artificial intelligence and the future of work",
    "the history of democracy and its challenges",
    "why reading books is important",
    "the impact of social media on young people",
    "how exercise affects mental health",
    "the relationship between science and religion",
    "the causes and effects of poverty",
    "space exploration and its benefits",
    "the importance of biodiversity",
    "how music affects human emotion",
    "the history of money and trade",
    "the ethics of genetic engineering",
    "how cities can become more sustainable",
    "the psychology of decision making",
    "the role of art in society",
    "ocean conservation and its importance",
]

def generate_ai_text(topic):
    """Generate AI text using Claude API."""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": f"Write a detailed paragraph of exactly 400 words about: {topic}. Write in a clear, informative style. Do not use headers or bullet points — just flowing prose."
                }]
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            return None
    except Exception as e:
        return None

ai_texts = []
texts_per_topic = 25  # 20 topics x 25 = 500

for topic in TOPICS:
    if len(ai_texts) >= 500:
        break
    print(f"  Generating texts about: {topic[:45]}...", end=" ")
    topic_texts = []
    attempts = 0
    while len(topic_texts) < texts_per_topic and attempts < texts_per_topic + 5:
        text = generate_ai_text(topic)
        if text and len(text.split()) >= 300:
            topic_texts.append(text)
        attempts += 1
        time.sleep(0.3)
    ai_texts.extend(topic_texts)
    print(f"✓ got {len(topic_texts)} texts (total: {len(ai_texts)})")

# Trim to exactly 500
ai_texts = ai_texts[:500]
print(f"\n✓ Total AI texts generated: {len(ai_texts)}")

# ============================================================
# PART 3: SAVE TO CSV
# ============================================================

print("\nPART 3: Saving data to Nardos_data folder...")
print("-" * 60)

# Save human texts
human_path = "Nardos_data/human_texts.csv"
with open(human_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label', 'source'])
    for i, text in enumerate(human_texts):
        writer.writerow([i+1, text, 'human', 'Project Gutenberg'])
print(f"  ✓ Saved {len(human_texts)} human texts → {human_path}")

# Save AI texts
ai_path = "Nardos_data/ai_texts.csv"
with open(ai_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label', 'source'])
    for i, text in enumerate(ai_texts):
        writer.writerow([i+1, text, 'ai', 'Claude claude-sonnet-4-6'])
print(f"  ✓ Saved {len(ai_texts)} AI texts → {ai_path}")

# Save combined dataset
combined_path = "Nardos_data/combined_dataset.csv"
with open(combined_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label', 'source'])
    for i, text in enumerate(human_texts):
        writer.writerow([i+1, text, 0, 'Project Gutenberg'])
    for i, text in enumerate(ai_texts):
        writer.writerow([len(human_texts)+i+1, text, 1, 'Claude claude-sonnet-4-6'])
print(f"  ✓ Saved combined dataset → {combined_path}")

print(f"\n{'='*60}")
print("  DATA COLLECTION COMPLETE")
print(f"{'='*60}")
print(f"  Human texts : {len(human_texts)}")
print(f"  AI texts    : {len(ai_texts)}")
print(f"  Total       : {len(human_texts) + len(ai_texts)}")
print(f"\n  Files saved in: Nardos_data/")
print(f"    - human_texts.csv")
print(f"    - ai_texts.csv")
print(f"    - combined_dataset.csv")
print(f"\n  Next step: run Nardos_code.py to analyse the data")
print("=" * 60)
