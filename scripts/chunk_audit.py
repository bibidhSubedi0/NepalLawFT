import asyncio
import aiofiles
from pathlib import Path
from collections import defaultdict

CHUNKS_PATH = Path("data/processed/chunks")
LOG_INTERVAL = 1000
CONCURRENCY = 200

sem = asyncio.Semaphore(CONCURRENCY)

def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    # 20% devanagari chars = Nepali. could probably be lower but 20% is safe
    # same logic used in paila ko rag pipeline
    return "np" if devanagari / max(len(text), 1) > 0.2 else "en"

async def analyze(path: Path) -> dict | None:
    try:
        async with sem, aiofiles.open(path, encoding="utf-8") as f:
            c = await f.read()
    except Exception as e:
        print(f"[WARN] {path.name}: {e}")
        return None

    if not c:
        return None

    return {
        "too_short":        len(c.split()) < 100,
        "likely_gibberish": len(set(c)) < 20,
        "lang":             detect_language(c),
    }

async def main():
    files = [p for p in CHUNKS_PATH.iterdir() if p.is_file()]
    print(f"Found {len(files)} files")

    stats: dict[str, int] = defaultdict(int)
    done = 0

    for coro in asyncio.as_completed([analyze(p) for p in files]):
        r = await coro
        if r is None:
            continue
        stats["total"]            += 1
        stats["too_short"]        += r["too_short"]
        stats["likely_gibberish"] += r["likely_gibberish"]
        stats[r["lang"]]          += 1
        done += 1
        if done % LOG_INTERVAL == 0:
            print(f"  {done}/{len(files)}")

    print("Done:", dict(stats))

asyncio.run(main())

'''
Result:
Done: {'total': 1214, 'too_short': 2, 'likely_gibberish': 0, 'np': 1082, 'en': 132}

Will do for now,
aile lai just building the pipeline anyways


'''