import pandas as pd
import os
import time
import json
from openai import OpenAI

# =========================
# CONFIGURATION
# =========================
INPUT_FILE = "data/processed/cleaned_text.csv"
OUTPUT_FILE = "data/processed/topic_modeling_openrouter_results.csv"

BATCH_SIZE = 50   
MODEL_NAME = "mistralai/mistral-7b-instruct"
TEMPERATURE = 0.2

# =========================
# OpenRouter Client
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is not set")

# =========================
# Load Data
# =========================
df = pd.read_csv(INPUT_FILE)
texts = df.iloc[:, 0].astype(str).tolist()
total_records = len(texts)

print(f"Loaded {total_records} records from {INPUT_FILE}")
print(f"Processing in batches of {BATCH_SIZE}...\n")

num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
results = []

# =========================
# Batch-wise Topic Modeling
# =========================
for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, total_records)
    batch_texts = texts[start_idx:end_idx]

    print(f"Processing batch {batch_idx + 1}/{num_batches}")

    prompt = f"""
You are an AI system for topic modeling.

TASK:
- Group the following sentences into meaningful topics
- Assign a concise topic name to each group
- Provide representative keywords
- Use sentence indices starting from 0 (relative to this batch)

RULES:
- Return ONLY valid JSON
- No explanations
- No markdown

OUTPUT FORMAT:
{{
  "topics": [
    {{
      "topic_name": "",
      "keywords": [],
      "sentence_indices": []
    }}
  ]
}}

Sentences:
{batch_texts}
"""

    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE
    )

    latency = round(time.time() - start_time, 2)

    results.append({
        "batch_id": batch_idx + 1,
        "start_index": start_idx,
        "end_index": end_idx,
        "topic_output_json": response.choices[0].message.content,
        "latency_sec": latency
    })

    print(f"  Completed {end_idx}/{total_records} records (Latency: {latency}s)\n")

    time.sleep(0.5)  # polite rate limiting

# =========================
# Save Results
# =========================
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)

print("Topic modeling complete!")
print(f"Results saved to {OUTPUT_FILE}")
