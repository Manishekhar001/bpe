"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose = True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256


        # input text processing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..256

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key = stats.get)

            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]


            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
                
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)

        while(len(ids) > 2):
            stats = get_stats(ids)

            pair = min(stats, key = lambda p : self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break # Nothing else if there to merge

            idx = self.merge[pair]
            ids = merge(ids, pair, idx)

        return ids