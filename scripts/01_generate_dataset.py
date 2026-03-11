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
OUTPUT_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

# --- Groq settings ---
GROQ_MODEL       = "llama-3.1-8b-instant"
TEMPERATURE      = 0.7
MAX_TOKENS       = 512
REQUESTS_PER_MIN = 30  # per key

# --- Chunk filtering ---
MIN_CHUNK_LEN = 150
MAX_CHUNK_LEN = 1500

# --- Dataset settings ---
QA_PER_CHUNK    = 1
SAVE_EVERY      = 100
TRAIN_RATIO     = 0.9
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
# Key rotation
# ---------------------------------------------------------------------------

def load_keys():
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GROQ_API_KEY{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    if not keys:
        raise ValueError("No GROQ_API_KEY1, GROQ_API_KEY2, ... found in .env")
    print(f"Loaded {len(keys)} API key(s) — effective rate limit: {REQUESTS_PER_MIN * len(keys)} req/min")
    return keys


class KeyRotator:
    """Cycles through N Groq API keys. Rate-limited or errored keys go into
    cooldown; the next available key is used. If ALL keys are cooling down,
    waits until the earliest one recovers, then continues."""

    def __init__(self, keys):
        self.keys     = keys
        self.clients  = {k: Groq(api_key=k) for k in keys}
        self.cooldown = {}   # key -> timestamp when usable again
        self._idx     = 0    # round-robin pointer

    def _available(self):
        now = time.time()
        return [k for k in self.keys if self.cooldown.get(k, 0) <= now]

    def get(self):
        """Return (key, client), blocking until a key is available."""
        while True:
            available = self._available()
            if available:
                self._idx = self._idx % len(available)
                key = available[self._idx]
                self._idx += 1
                return key, self.clients[key]
            wait = min(self.cooldown[k] for k in self.keys) - time.time()
            print(f"\n  [WAIT] All {len(self.keys)} keys rate-limited. "
                  f"Resuming in {wait:.1f}s...", flush=True)
            time.sleep(max(wait, 1))

    def mark_rate_limited(self, key, retry_after=60):
        self.cooldown[key] = time.time() + retry_after
        n_avail = len(self._available())
        print(f"\n  [LIMIT] Key ...{key[-6:]} cooling down {retry_after}s. "
              f"{n_avail}/{len(self.keys)} keys still available.", flush=True)

    def mark_error(self, key, cooldown_secs=30):
        self.cooldown[key] = time.time() + cooldown_secs
        print(f"\n  [ERROR] Key ...{key[-6:]} cooling down {cooldown_secs}s.", flush=True)

    def disable(self, key):
        self.cooldown[key] = float("inf")
        print(f"\n  [DISABLED] Key ...{key[-6:]} permanently disabled.", flush=True)


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
    if raw.startswith("```"):
        parts = raw.split("```")
        return parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return raw


def ask_groq(rotator, chunk, language):
    """Call Groq with automatic key rotation on any error."""
    import re
    prompt = PROMPT_NP.format(chunk=chunk) if language == "np" else PROMPT_EN.format(chunk=chunk)

    while True:
        key, client = rotator.get()
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
            return None  # bad JSON shape — skip this chunk

        except json.JSONDecodeError:
            return None  # model returned non-JSON — skip this chunk

        except Exception as e:
            err = str(e).lower()

            if "rate limit" in err or "429" in err:
                retry_after = 60
                m = re.search(r"try again in ([\d.]+)s", err)
                if m:
                    retry_after = float(m.group(1)) + 2
                rotator.mark_rate_limited(key, retry_after=retry_after)
                # loop — next available key will be picked

            elif "401" in err or "invalid api key" in err:
                rotator.disable(key)
                # loop — try next key

            else:
                print(f"\n  [WARN] Key ...{key[-6:]}: {e}")
                rotator.mark_error(key, cooldown_secs=30)
                # loop — try next key


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_for_finetuning(question, answer, source_file):
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

    all_np = load_jsonl(out_np) if out_np.exists() else []
    all_en = load_jsonl(out_en) if out_en.exists() else []

    if not all_np and not all_en:
        print("  No data to split yet.")
        return

    combined             = balance_by_language(all_np, all_en)
    train_data, val_data = split_train_val(combined)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data)]:
        save_split(data, SPLITS_DIR / f"{name}.jsonl")
        print(f"  {name}.jsonl — {len(data)} pairs")

    print(f"\nDataset ready in {SPLITS_DIR}")


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def run_generation(rotator, chunks, out_np, out_en, limit):
    seen_np = load_already_processed(out_np)
    seen_en = load_already_processed(out_en)

    already_done = len(seen_np) + len(seen_en)
    if already_done:
        print(f"Resuming — {len(seen_np)} Nepali + {len(seen_en)} English pairs already done")

    delay     = 60.0 / REQUESTS_PER_MIN / len(rotator.keys)
    total_np  = len(seen_np)
    total_en  = len(seen_en)
    new_total = 0

    f_np = open(out_np, "a", encoding="utf-8")
    f_en = open(out_en, "a", encoding="utf-8")

    try:
        for chunk_data in tqdm(chunks, desc="Chunks"):
            if limit and new_total >= limit:
                print(f"\nReached limit of {limit} new samples.")
                break

            if not is_valid_chunk(chunk_data):
                continue

            text     = chunk_data.get("text", "")
            language = chunk_data.get("language", "np")
            source   = chunk_data.get("source_file", "unknown")

            result = ask_groq(rotator, text, language)
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
                total_np += 1
            else:
                write_jsonl_line(f_en, row)
                seen_en.add(key)
                total_en += 1

            new_total += 1
            time.sleep(delay)

            if new_total % SAVE_EVERY == 0:
                f_np.flush()
                f_en.flush()
                print(f"\n  {total_np} Nepali + {total_en} English pairs so far...")

    finally:
        f_np.close()
        f_en.close()

    print(f"\nDone — {total_np} Nepali, {total_en} English pairs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CHUNKS_DIR = Path("data/processed/chunks")
TOTAL_CORPUS_CHUNKS = 67_000  # approximate total chunks across all files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10_000,
                        help="Target total Q&A pairs across all files (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk sampling (default: 42)")
    parser.add_argument("--split", action="store_true",
                        help="Build train/val split after generation")
    args = parser.parse_args()

    keys    = load_keys()
    rotator = KeyRotator(keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_np = OUTPUT_DIR / "qa_raw_np.jsonl"
    out_en = OUTPUT_DIR / "qa_raw_en.jsonl"

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(f"No .jsonl files found in {CHUNKS_DIR}")

    # --- Count valid chunks per file ---
    print(f"Scanning {len(chunk_files)} file(s) to count valid chunks...")
    file_chunks = {}  # path -> list of valid chunks
    for chunk_file in chunk_files:
        chunks = [c for c in load_chunks(chunk_file) if is_valid_chunk(c)]
        if chunks:
            file_chunks[chunk_file] = chunks

    total_valid = sum(len(c) for c in file_chunks.values())
    print(f"Total valid chunks across all files: {total_valid}")
    print(f"Target samples: {args.target}")
    print(f"{'File':<60} {'Chunks':>8} {'Share':>7} {'Quota':>7}")
    print("-" * 85)

    # --- Calculate per-file quota proportionally ---
    quotas = {}
    for chunk_file, chunks in file_chunks.items():
        share    = len(chunks) / total_valid
        quota    = max(1, round(share * args.target))
        quotas[chunk_file] = quota
        print(f"{chunk_file.name:<60} {len(chunks):>8} {share*100:>6.2f}% {quota:>7}")

    # Adjust rounding drift so sum == target exactly
    total_quota = sum(quotas.values())
    diff = args.target - total_quota
    if diff != 0:
        # Add/remove from the file with the largest quota
        biggest = max(quotas, key=quotas.get)
        quotas[biggest] += diff

    print(f"\nTotal quota: {sum(quotas.values())} pairs")

    # --- Process each file ---
    rng       = random.Random(args.seed)
    total_new = 0

    for i, (chunk_file, chunks) in enumerate(file_chunks.items(), 1):
        quota = quotas[chunk_file]
        print(f"\n[{i}/{len(file_chunks)}] {chunk_file.name}")
        print(f"  {len(chunks)} valid chunks → sampling {quota}")

        # Random sample without replacement (or all if quota >= available)
        sampled = rng.sample(chunks, min(quota, len(chunks)))

        before_np = sum(1 for _ in open(out_np, encoding="utf-8")) if out_np.exists() else 0
        before_en = sum(1 for _ in open(out_en, encoding="utf-8")) if out_en.exists() else 0

        run_generation(rotator, sampled, out_np, out_en, limit=quota)

        after_np = sum(1 for _ in open(out_np, encoding="utf-8")) if out_np.exists() else 0
        after_en = sum(1 for _ in open(out_en, encoding="utf-8")) if out_en.exists() else 0
        generated = (after_np - before_np) + (after_en - before_en)
        total_new += generated
        print(f"  Generated {generated} pairs (running total: {total_new})")

    print(f"\nAll files done. {total_new} new pairs generated total.")
    if args.split:
        build_dataset(out_np, out_en)


if __name__ == "__main__":
    main()