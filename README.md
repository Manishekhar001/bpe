# BPE Tokenizer

A from-scratch, byte-level **Byte Pair Encoding (BPE)** tokenizer library for Python — inspired by Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe). Built as a learning-oriented implementation that progresses from a minimal tokenizer to a full GPT-4-compatible one.

## Overview

This project implements BPE tokenization step-by-step across four modules:

| Module | Class | Description |
|--------|-------|-------------|
| `base.py` | `Tokenizer` | Base class with save/load, vocab building, and shared helpers |
| `basic.py` | `BasicTokenizer` | Minimal BPE — no regex splitting, no special tokens |
| `regex.py` | `RegexTokenizer` | Adds regex-based pre-splitting and special token handling |
| `gpt4.py` | `GPT4Tokenizer` | Drop-in GPT-4 tokenizer (loads weights from `tiktoken`) |

### Class Hierarchy

```
Tokenizer (base.py)
 └── BasicTokenizer (basic.py)
 └── RegexTokenizer (regex.py)
      └── GPT4Tokenizer (gpt4.py)
```

## How BPE Works

1. **Encode** text as raw UTF-8 bytes (values 0–255) — this is the base vocabulary.
2. **Find** the most frequent adjacent pair of tokens in the training data.
3. **Merge** that pair into a single new token (assigned the next available id).
4. **Repeat** steps 2–3 until the vocabulary reaches the target size.
5. To **encode** new text, replay the learned merges in the same order.

> **Why bytes instead of characters?** Unicode code points are unbounded and vary in size. Bytes give a fixed, universal alphabet of exactly 256 values — no unknown-token fallback needed.

## Project Structure

```
.
├── pyproject.toml          # Project metadata & dependencies (uv/pip)
├── train.py                # Script to train basic & regex tokenizers
├── src/
│   └── bpe/
│       ├── __init__.py
│       ├── base.py         # Tokenizer base class + helpers
│       ├── basic.py        # BasicTokenizer
│       ├── regex.py        # RegexTokenizer
│       └── gpt4.py         # GPT4Tokenizer (wraps tiktoken)
├── models/                 # Pre-trained model & vocab files
│   ├── basic.model
│   ├── basic.vocab
│   ├── regex.model
│   └── regex.vocab
├── tests/
│   ├── test_tokenizer.py   # pytest test suite
│   └── taylorswift.txt     # Training data (Taylor Swift Wikipedia article)
└── documents/              # Detailed code documentation (Obsidian-style)
```

## Installation

Requires **Python 3.12+**.

```bash
# Clone the repository
git clone <repo-url>
cd bpe

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Dependencies

- `regex` — extended regex library (used for GPT-2/GPT-4 split patterns)
- `tiktoken` — OpenAI's tokenizer library (used only by `GPT4Tokenizer` as a data source)
- `pytest` — testing framework

## Usage

### Basic Tokenizer

```python
from src.bpe.basic import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train("hello world", vocab_size=256 + 10, verbose=True)

ids = tokenizer.encode("hello world")
print(ids)  # [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

text = tokenizer.decode(ids)
print(text)  # "hello world"
```

### Regex Tokenizer

```python
from src.bpe.regex import RegexTokenizer

tokenizer = RegexTokenizer()
tokenizer.train("hello world!!!", vocab_size=256 + 50, verbose=True)

# Register special tokens
tokenizer.register_special_tokens({"<|endoftext|>": 100257})

ids = tokenizer.encode("hello world!!!", allowed_special="all")
text = tokenizer.decode(ids)
```

### GPT-4 Tokenizer (Pretrained)

```python
from src.bpe.gpt4 import GPT4Tokenizer

tokenizer = GPT4Tokenizer()

ids = tokenizer.encode("hello world!!!")
print(ids)

text = tokenizer.decode(ids)
print(text)
```

## Training a Model

Train both `BasicTokenizer` and `RegexTokenizer` on the included Taylor Swift Wikipedia article:

```bash
python train.py
```

This produces four files in `models/`:
- `basic.model` / `basic.vocab` — BasicTokenizer weights
- `regex.model` / `regex.vocab` — RegexTokenizer weights

Training takes ~25 seconds on a typical laptop.

## Save & Load

```python
# Save
tokenizer.save("models/my_tokenizer")
# Creates: my_tokenizer.model + my_tokenizer.vocab

# Load
tokenizer = RegexTokenizer()
tokenizer.load("models/my_tokenizer.model")
```

The `.model` file stores the version, regex pattern, special tokens, and merge rules. The `.vocab` file is a human-readable representation for inspection.

## Testing

```bash
pytest tests/
```

Tests cover:
- **Encode/decode identity** — `decode(encode(x)) == x` for all tokenizers
- **GPT-4 parity** — output matches `tiktoken`'s `cl100k_base` encoding
- **Special tokens** — `ickerView`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`
- **Wikipedia BPE example** — verifies merges against the classic "aaabdaaabac" example
- **Save/load roundtrip** — model persistence preserves encode/decode behavior

## Core Helpers

| Function | Location | Purpose |
|----------|----------|---------|
| `get_stats(ids)` | `base.py` | Count frequencies of adjacent token pairs |
| `merge(ids, pair, idx)` | `base.py` | Replace all occurrences of a pair with a new token id |
| `render_token(t)` | `base.py` | Safely display a token (escaping control characters) |
| `replace_control_characters(s)` | `base.py` | Escape Unicode control characters for display |

## Key Concepts

- **Vocabulary size** = 256 (base bytes) + number of merges. So `vocab_size=512` means 256 merges.
- **Merge priority**: During training, the most frequent pair is merged first. During encoding, the pair with the lowest merge index (earliest learned) is applied first.
- **Regex splitting** (in `RegexTokenizer`): Text is pre-split into chunks before BPE merging. The default pattern matches GPT-4's splitting behavior — words, numbers, whitespace, and punctuation are separated.
- **Byte shuffling** (in `GPT4Tokenizer`): OpenAI's tokenizer permutes the 256 base byte values into a non-sequential order. This class handles the shuffle/unshuffle transparently.

## License

Educational project — see repository for license details.
