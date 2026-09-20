"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""

import tiktoken                    # OpenAI's official tokenizer library — used here ONLY as a data source
from .regex import RegexTokenizer  # our own hand-built tokenizer class (regex split + BPE merge logic)


def bpe(mergeable_ranks, token, max_rank):
    """
    Given tiktoken's full vocabulary (mergeable_ranks) and one specific
    finished `token`, figure out the two pieces it was originally merged
    from, by replaying BPE merges on its raw bytes and stopping just
    before we'd reconstruct the token itself (max_rank = that stopping point).
    """
    # break the token into its individual raw bytes to start from scratch
    parts = [bytes([b]) for b in token]

    while True:
        min_idx = None   # index of the best (earliest-ranked) mergeable pair found this pass
        min_rank = None  # that pair's rank

        # scan every pair of CURRENT neighbors in `parts`
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            # would joining this pair produce a token tiktoken already knows about?
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # keep track of whichever valid pair has the LOWEST rank (earliest created)
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # stop if: no mergeable pair exists anymore, OR
        # the best pair found would be (or is past) our own token's rank —
        # meaning the next merge would recreate the token itself, so don't do it
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        assert min_idx is not None  # sanity check — we should have found something if we didn't break

        # perform the merge: splice the two adjacent pieces into one
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]

    # whatever's left are the token's two immediate parent pieces
    return parts


def recover_merges(mergeable_ranks):
    """
    Convert tiktoken's flat {finished_token_bytes: rank} vocabulary into
    minbpe's own {(parent_id_0, parent_id_1): new_token_id} merge format,
    by calling bpe() on every multi-byte token.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # single raw bytes have no parents — they're the base alphabet

        # find this token's two parent pieces (as raw bytes)
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2  # a merge always combines exactly two things

        # convert those parent BYTES into their integer ranks/ids
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]

        # record the recovered rule
        merges[(ix0, ix1)] = rank

    return merges


# the exact regex GPT-4 uses to pre-split text before BPE merging
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-4's real reserved special tokens and their fixed ids
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,    # marks end of a document
    '<|fim_prefix|>': 100258,   # fill-in-the-middle: marks the prefix section
    '<|fim_middle|>': 100259,   # fill-in-the-middle: marks the (to-predict) middle section
    '<|fim_suffix|>': 100260,   # fill-in-the-middle: marks the suffix section
    '<|endofprompt|>': 100276   # marks end of a prompt
}


class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        # set up as a RegexTokenizer, using GPT-4's own splitting pattern
        super().__init__(pattern=GPT4_SPLIT_PATTERN)

        # load OpenAI's real, pretrained GPT-4 / GPT-3.5-turbo tokenizer data
        enc = tiktoken.get_encoding("cl100k_base")

        # reach into tiktoken's internal data: {finished_token_bytes: rank}
        mergeable_ranks = enc._mergeable_ranks

        # reconstruct minbpe-style merges from that flat data
        self.merges = recover_merges(mergeable_ranks)

        # rebuild the vocab (id -> bytes) from the recovered merges
        vocab = {idx: bytes([idx]) for idx in range(256)}  # start with the 256 raw bytes
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]              # concatenate parents' bytes
        self.vocab = vocab

        # --- the byte shuffle quirk ---
        # OpenAI's tokenizer does NOT map raw byte value i -> token id i.
        # The 256 single bytes are permuted into some other (historical,
        # non-sensical) order. We must record that permutation both ways:
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}          # real byte -> tiktoken's id for it
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}          # tiktoken's id -> real byte

        # register GPT-4's real special tokens (sets up special_tokens / inverse_special_tokens)
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # before merging, permute the raw bytes into tiktoken's shuffled numbering,
        # since that's the numbering our recovered `merges` rules were built in
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        # now run the NORMAL RegexTokenizer merge loop (earliest-pair-first) on the shuffled bytes
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # look up each id's bytes and join them — still in SHUFFLED numbering
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # un-shuffle back to the TRUE original byte values
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        # finally decode the correctly-ordered bytes into a real string
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # this is a pretrained tokenizer — it's not meant to be trained on new data
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would need extra logic to also persist byte_shuffle,
    # which the generic base-class .model format doesn't support —
    # rather than complicate that clean class, it's simply disabled here
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        # write-only: dump GPT-4's real vocab to a human-readable file,
        # in the same [parent1][parent2] -> [result] id format base.py uses.
        # Usage: python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .base import render_token

        # rebuild the base 256 bytes, but UN-shuffled this time, so the
        # printed file shows real byte meanings, not tiktoken's internal order
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        # flip merges so we can look up "what pair created this id"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)  # safe, printable version of this token
                if idx in inverted_merges:
                    # this token came from a merge — show its parents too
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # a raw byte or special token — just print it
                    f.write(f"[{s}] {idx}\n")