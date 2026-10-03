"""Writes prompts/prompts_long.json: 103 prompts per category (3 warmup + 100 measured), each
300-600 target tokens, with the text that really follows each prompt so predictability can
be measured without a model.

The length range was fixed once from the regime the repo makes claims about (serving
prompts in the hundreds of tokens) and is not tuned against any result.

  code        windows of CPython 3.12's standard library, one file per prompt.
  prose       windows of six Project Gutenberg novels, starting at a paragraph.
  repetitive  generated logs, CSV, configs and boilerplate whose fields are low-entropy:
              counters that increment, values that cycle through short lists. Text whose
              continuation is mostly determined by its prefix, unlike the short set's
              templates, which draw a fresh random value for every field.

Both real sources are likely in the models' training data. That raises acceptance for code
and prose alike and is noted in the README's limitations.
"""
import hashlib
import json
import random
import sysconfig
import urllib.request
from pathlib import Path

import yaml

from lm.load import load_tokenizer

PER_CATEGORY = 103
MIN_TOKENS, MAX_TOKENS = 300, 600
CONTINUATION_TOKENS = 64
BOOKS = {  # Project Gutenberg ebook id: title
    1342: "Pride and Prejudice",
    2701: "Moby Dick",
    84: "Frankenstein",
    98: "A Tale of Two Cities",
    1661: "The Adventures of Sherlock Holmes",
    345: "Dracula",
}
CACHE = Path("models/gutenberg")


def window(tokenizer, text: str, start: int, n_tokens: int) -> tuple[str, str] | None:
    """Prompt = the first n_tokens of text[start:], cut at a token boundary in the original
    string (decode(encode(x)) is not always x); continuation = the next 64 tokens."""
    enc = tokenizer(text[start:], return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    if len(offsets) < n_tokens + CONTINUATION_TOKENS:
        return None
    cut = start + offsets[n_tokens - 1][1]
    end = start + offsets[n_tokens + CONTINUATION_TOKENS - 1][1]
    return text[start:cut], text[cut:end]


def code(tokenizer, rng: random.Random) -> list[tuple[str, str]]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.glob("*.py") if not p.name.startswith(("test", "_")))
    rng.shuffle(files)
    out = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if len(lines) < 200:
            continue
        # Skip the licence/docstring header; start on a line boundary.
        line = rng.randrange(len(lines) // 5, len(lines) // 2)
        start = sum(len(x) for x in lines[:line])
        item = window(tokenizer, text, start, rng.randint(MIN_TOKENS, MAX_TOKENS))
        if item:
            out.append(item)
        if len(out) == PER_CATEGORY:
            return out
    raise RuntimeError(f"only {len(out)} stdlib files long enough")


def book(ebook: int) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"pg{ebook}.txt"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt").read())
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    body = text.split("*** START OF", 1)[1].split("\n", 1)[1]
    return body.split("*** END OF", 1)[0]


def prose(tokenizer, rng: random.Random) -> tuple[list[tuple[str, str]], dict[str, str]]:
    out, digests = [], {}
    per_book = -(-PER_CATEGORY // len(BOOKS))
    for ebook, title in BOOKS.items():
        text = book(ebook)
        digests[title] = hashlib.sha256(text.encode()).hexdigest()
        # Paragraph starts past the front matter, spaced so windows do not overlap.
        starts = [i + 2 for i in range(len(text) // 10, len(text) - 20_000) if text.startswith("\n\n", i)
                  and text[i + 2:i + 3].isalpha()]
        picked = sorted(rng.sample(starts, per_book * 4))
        taken, last = 0, -1
        for start in picked:
            if start < last or taken == per_book:
                continue
            item = window(tokenizer, text, start, rng.randint(MIN_TOKENS, MAX_TOKENS))
            if item:
                out.append(item)
                taken += 1
                last = start + len(item[0]) + len(item[1])
    return out[:PER_CATEGORY], digests


def repetitive_text(family: int, rng: random.Random) -> str:
    """~12k characters of one family; low-entropy fields only."""
    hosts = ["10.0.0.11", "10.0.0.12", "10.0.0.13"]
    paths = ["/api/health", "/api/users", "/api/orders", "/static/app.js"]
    names = ["alice", "bob", "carol", "dave", "erin"]
    base = rng.randint(0, 5000)
    rows = []
    for i in range(400):
        n = base + i
        if family == 0:
            rows.append(f"{hosts[i % 3]} - - [02/Oct/2026:09:{(i // 60) % 60:02d}:{i % 60:02d} +0000] "
                        f"\"GET {paths[i % 4]} HTTP/1.1\" 200 {512 * (i % 4 + 1)}\n")
        elif family == 1:
            rows.append(f"{n},SKU-{n:05d},warehouse-{'ABC'[i % 3]},{10 * (i % 3 + 1)},in_stock\n")
        elif family == 2:
            rows.append(f"  service-{i}:\n    image: registry.local/service-{i}:1.0.0\n    replicas: 2\n"
                        f"    port: {8000 + i}\n    restart: always\n")
        elif family == 3:
            rows.append(f"def test_add_{n}():\n    assert add({n}, {n}) == {2 * n}\n\n\n")
        elif family == 4:
            name = names[i % 5]
            rows.append(f"Hi {name},\n\nYour weekly report is ready. You can download it from the dashboard.\n\n"
                        f"Thanks,\nThe Reports Team\n\n---\n\n")
        elif family == 5:
            rows.append(f"<tr><td>{n}</td><td>{names[i % 5]}</td><td>active</td><td>2026-10-02</td></tr>\n")
        else:
            rows.append(f"[INFO] step {n}: loss computed, gradients applied, checkpoint skipped\n")
    return ("services:\n" if family == 2 else "") + "".join(rows)


def repetitive(tokenizer, rng: random.Random) -> list[tuple[str, str]]:
    out = []
    for i in range(PER_CATEGORY):
        text = repetitive_text(i % 7, rng)
        line_starts = [0] + [j + 1 for j, c in enumerate(text[:4000]) if c == "\n"]
        out.append(window(tokenizer, text, rng.choice(line_starts[:20]), rng.randint(MIN_TOKENS, MAX_TOKENS)))
    return out


def main() -> None:
    config = yaml.safe_load(open("config.yaml"))
    tokenizer = load_tokenizer(config["models"]["target"], config["models"]["cache_dir"])
    rng = random.Random(0)
    prose_items, digests = prose(tokenizer, rng)
    items = {"code": code(tokenizer, rng), "prose": prose_items, "repetitive": repetitive(tokenizer, rng)}
    for category, pairs in items.items():
        assert len(pairs) == PER_CATEGORY and len({p for p, _ in pairs}) == PER_CATEGORY, category
    record = {
        "prompts": {c: [p for p, _ in pairs] for c, pairs in items.items()},
        "continuations": {c: [x for _, x in pairs] for c, pairs in items.items()},
        "sources": {"code": f"CPython {sysconfig.get_python_version()} stdlib", "prose_sha256": digests},
    }
    Path(__file__).with_name("prompts_long.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
