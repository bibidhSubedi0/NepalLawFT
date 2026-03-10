import os
import json
import time
import random
import argparse
from pathlib import Path

from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
CHUNKS_DIR = Path("data/processed/chunks")
OUTPUT_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

# --- Groq settings ---
GROQ_MODEL       = "llama-3.1-8b-instant"
TEMPERATURE      = 0.7
MAX_TOKENS       = 512
REQUESTS_PER_MIN = 30

# --- Chunk filtering ---
MIN_CHUNK_LEN = 150
MAX_CHUNK_LEN = 1500

# --- Dataset settings ---
QA_PER_CHUNK    = 2
SAVE_EVERY      = 100
TRAIN_RATIO     = 0.9 # 90% train, 10% val
TARGET_NP_RATIO = 0.80

SYSTEM_PROMPT = (
    "You are CivicLens, a legal assistant specialized in Nepal's laws, "
    "constitution, and governance documents. Answer questions accurately, "
    "cite your sources, and respond in the same language as the question. "
    "If you don't know, say so."
)

PROMPT_NP = """तपाईंलाई नेपाली कानुनी दस्तावेजको एक अनुच्छेद दिइएको छ।
            यस अनुच्छेदबाट एउटा प्रश्न र विस्तृत उत्तर नेपालीमा तयार गर्नुहोस्।

            अनुच्छेद:
            {chunk}

            JSON मात्र फर्काउनुहोस् (कुनै अतिरिक्त पाठ नगर्नुहोस्):
            {{"question": "...", "answer": "..."}}"""

PROMPT_EN = """You are given a passage from a Nepali legal or governance document.
            Generate one question and a detailed answer in English based strictly on this passage.

            Passage:
            {chunk}

            Return JSON only (no extra text):
            {{"question": "...", "answer": "..."}}"""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_chunks(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_already_processed(path):
    seen = set()
    if not path.exists():
        return seen
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                seen.add(row["source"] + row["question"])
            except Exception:
                continue
    return seen


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_line(file_handle, row):
    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_split(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            write_jsonl_line(f, row)


# ---------------------------------------------------------------------------
# Chunk filtering
# ---------------------------------------------------------------------------

def is_valid_chunk(chunk_data):
    text     = chunk_data.get("text", "")
    language = chunk_data.get("language", "")
    if not (MIN_CHUNK_LEN <= len(text) <= MAX_CHUNK_LEN):
        return False
    if language not in ("np", "en"):
        return False
    return True


# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------

def strip_markdown_fences(raw):
    """Remove ```json ... ``` wrappers that the model sometimes adds."""
    if raw.startswith("```"):
        parts = raw.split("```")
        return parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return raw


def ask_groq(client, chunk, language):
    prompt = PROMPT_NP.format(chunk=chunk) if language == "np" else PROMPT_EN.format(chunk=chunk)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        raw = strip_markdown_fences(raw)

        result = json.loads(raw)
        if result.get("question") and result.get("answer"):
            return result
        return None

    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"\n  [WARN] Groq error: {e}")
        time.sleep(10)
        return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_for_finetuning(question, answer, source_file):
    """Wrap a Q&A pair in LLaMA 3 chat tokens for fine-tuning."""
    text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{answer} [Source: {source_file}]<|eot_id|>"
    )
    return {"text": text, "question": question, "answer": answer, "source": source_file}


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def balance_by_language(all_np, all_en):
    random.seed(42)
    random.shuffle(all_np)
    random.shuffle(all_en)

    total     = len(all_np) + len(all_en)
    target_np = min(len(all_np), int(total * TARGET_NP_RATIO))
    target_en = min(len(all_en), total - target_np)

    combined = all_np[:target_np] + all_en[:target_en]
    random.shuffle(combined)

    print(f"  Using {target_np} Nepali + {target_en} English = {len(combined)} total")
    return combined


def split_train_val(data):
    idx = int(len(data) * TRAIN_RATIO)
    return data[:idx], data[idx:]


def build_dataset(out_np, out_en):
    print("\nBuilding final dataset with 80% Nepali / 20% English...")

    combined             = balance_by_language(load_jsonl(out_np), load_jsonl(out_en))
    train_data, val_data = split_train_val(combined)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data)]:
        save_split(data, SPLITS_DIR / f"{name}.jsonl")
        print(f"  {name}.jsonl — {len(data)} pairs")

    print(f"\nDataset ready in {SPLITS_DIR}")


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def process_chunk(client, chunk_data, chunk_file, seen_np, seen_en, f_np, f_en, delay):
    text     = chunk_data.get("text", "")
    language = chunk_data.get("language", "np")
    source   = chunk_data.get("source_file", chunk_file.name)

    added_np = added_en = 0

    for _ in range(QA_PER_CHUNK):
        result = ask_groq(client, text, language)
        if not result:
            continue

        key  = source + result["question"]
        seen = seen_np if language == "np" else seen_en
        if key in seen:
            continue

        row = format_for_finetuning(result["question"], result["answer"], source)

        if language == "np":
            write_jsonl_line(f_np, row)
            seen_np.add(key)
            added_np += 1
        else:
            write_jsonl_line(f_en, row)
            seen_en.add(key)
            added_en += 1

        time.sleep(delay)

    return added_np, added_en


def run_generation(client, chunk_files, out_np, out_en):
    seen_np = load_already_processed(out_np)
    seen_en = load_already_processed(out_en)

    if seen_np or seen_en:
        print(f"Resuming — {len(seen_np)} Nepali + {len(seen_en)} English pairs already done")

    delay    = 60.0 / REQUESTS_PER_MIN
    total_np = len(seen_np)
    total_en = len(seen_en)

    f_np = open(out_np, "a", encoding="utf-8")
    f_en = open(out_en, "a", encoding="utf-8")

    try:
        for chunk_file in tqdm(chunk_files, desc="Files"):
            chunks = load_chunks(chunk_file)

            for chunk_data in tqdm(chunks, desc=f"  {chunk_file.name[:40]}", leave=False):
                if not is_valid_chunk(chunk_data):
                    continue

                added_np, added_en = process_chunk(
                    client, chunk_data, chunk_file,
                    seen_np, seen_en, f_np, f_en, delay
                )
                total_np += added_np
                total_en += added_en

                total = total_np + total_en
                if total % SAVE_EVERY == 0 and total > 0:
                    f_np.flush()
                    f_en.flush()
                    print(f"\n  {total_np} Nepali + {total_en} English pairs so far...")
    finally:
        f_np.close()
        f_en.close()

    print(f"\nDone generating — {total_np} Nepali, {total_en} English pairs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(limit=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not chunk_files:
        print(f"No JSONL files found in {CHUNKS_DIR}")
        return

    if limit:
        chunk_files = chunk_files[:limit]
        print(f"Test mode — processing {limit} file(s)")

    print(f"Found {len(chunk_files)} file(s) in {CHUNKS_DIR}")

    out_np = OUTPUT_DIR / "qa_raw_np.jsonl"
    out_en = OUTPUT_DIR / "qa_raw_en.jsonl"

    run_generation(client, chunk_files, out_np, out_en)
    build_dataset(out_np, out_en)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of chunk files to process (for testing)")
    args = parser.parse_args()
    main(limit=args.limit)