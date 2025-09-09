import collections
import json
from typing import List, Tuple, Iterator, Iterable
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing
import functools

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(chuck: tuple[int],
                  input_path: str,
                  special_tokens: list[str]) -> collections.Counter:
    """
    Used in the multiprocessing
    Process each chunk of the file and update the vocabulary counter.
    """
    start, end = chuck
    special_tokens_pattern = '|'.join(special_tokens)
    # special_tokens_pattern = "|".join(map(re.escape, special_tokens))
    chunk_counter = collections.Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Removing special tokens before pre-tokenization
        for tokens in re.split(special_tokens_pattern, chunk):
            # Pre-tokenization
            for match in re.finditer(PAT, tokens):
                # Returns the entire substring that matched the regex.
                if match.group():
                    chunk_counter.update([tuple(bytes([b]) for b in match.group().encode('utf-8'))])
    return chunk_counter


def myself_train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):
    """
        Train a byte-level BPE tokenizer.

        Args:
            input_path (str): Path to training text file.
            vocab_size (int): Max size of final vocabulary (including bytes + merges + specials).
            special_tokens (list[str]): Special tokens to include in vocab.

        Returns:
            vocab (dict[int, bytes]): Token ID -> bytes mapping
            merges (list[tuple[bytes, bytes]]): List of merge rules
        """

    # 1. Initialize vocabulary with all byte values
    vocab_dict = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Add special tokens
    for tok in special_tokens:
        vocab_dict[next_id] = tok.encode("utf-8")
        next_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    # 2. Pre-tokenization and use multiprocess
    num_processes = 8
    # vocab_counter is the starting vocabulary for BPE training
    # every word is broken into byte sequences
    vocab_counter : dict[tuple[bytes], int] = collections.Counter()
    boundaries = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(
            functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
            zip(boundaries[:-1], boundaries[1:]),
        )

        for res in results:
            vocab_counter.update(res)

    # Helper: count the frequency of all adjacent symbol pairs
    def get_pair_counts(counter_to_calculate):
        pair_count = collections.defaultdict(int)
        for token, freq in counter_to_calculate.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_count[pair] += freq
        return pair_count

    def select_pair_to_merge(pair_freqs):
        """
        Select the pair with highest frequency.
        Break ties by choosing the lexicographically greatest pair.

        pair_freqs: Counter {(a, b): freq}
        """
        # 1. Get max frequency
        max_freq = max(pair_freqs.values())
        # 2. Filter pairs with max frequency
        candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        # 3. Pick the lexicographically greatest
        selected_pair = max(candidates)

        return selected_pair

    # Helper: merge most frequent pair
    def merge_pair_tokens(counter_to_merge, pair_to_merge):
        new_vocab = collections.Counter()
        first, second = pair_to_merge
        merged_symbol = first + second  # bytes

        for token, freq in counter_to_merge.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == first and token[i + 1] == second:
                    new_token.append(merged_symbol)  # merged as a single element
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_vocab[tuple(new_token)] += freq  # store as tuple!
        return new_vocab


    # 3. BPE loop
    while len(vocab_dict) < vocab_size:
        pair_counts = get_pair_counts(vocab_counter)
        if not pair_counts:
            break  # no more pairs to merge
        #most_freq_pair = max(pair_counts, key=pair_counts.get)
        pair_to_merge = select_pair_to_merge(pair_counts)
        merges.append(pair_to_merge)
        vocab_counter = merge_pair_tokens(vocab_counter, pair_to_merge)

        if pair_to_merge not in vocab_dict.values():
            # !!!NOTICE: pair_to_merge is tuple type, while pair_to_merge[0] is btyes type
            # Given pair_to_merge = (b'a', b'b'),
            # pair_to_merge[0]+pair_to_merge[1] leads to b'a' + b'b'  → b'ab'
            vocab_dict[next_id] = pair_to_merge[0]+pair_to_merge[1]
            next_id += 1

    return vocab_dict, merges


'''
=========================================================================================================
'''

class MySelfTokenizer(object):

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = dict(vocab)  # id -> bytes
        self.id_to_bytes = dict(vocab)
        self.bytes_to_id = {v: k for k, v in vocab.items()}  # bytes -> id

        # merges: list[(b'a', b'b')]
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # handle special tokens
        self.special_tokens = []
        if special_tokens:
            for tok in special_tokens:
                print(tok)
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self.bytes_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = tok_bytes
                    self.id_to_bytes[new_id] = tok_bytes
                    self.bytes_to_id[tok_bytes] = new_id
                self.special_tokens.append(tok)
            self.special_tokens_pattern = "|".join(map(re.escape, sorted(self.special_tokens, key=len, reverse=True)))
            print(self.special_tokens_pattern)
        else:
            self.special_tokens_pattern = None
        self.special_token_bytes = {tok.encode("utf-8") for tok in self.special_tokens}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # load vocab
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            for line in f:
                idx, token = line.strip().split(maxsplit=1)
                vocab[int(idx)] = token  # already bytes

        # load merges
        merges = []
        with open(merges_filepath, "rb") as f:
            for line in f:
                a, b = line.strip().split()
                merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[bytes]:
        # base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # #?[^\s\p{L}\p{N}]+
        #
        # if self.special_tokens_pattern:
        #     # specials first, base pattern second
        #     pattern = f"(?>(?:{self.special_tokens_pattern}))|{base_pattern}"
        # else:
        #     pattern = base_pattern
        #
        # regex_compiled = re.compile(pattern, re.UNICODE)
        #
        # # findall returns the full matched strings
        # tokens = [tok.encode("utf-8") for tok in regex_compiled.findall(text)]
        # print(regex_compiled.findall(text)) if self.special_tokens_pattern else None
        # return tokens
        base_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        base_re = re.compile(base_pattern, re.UNICODE)

        if self.special_tokens_pattern:
            # keep special tokens as separate list elements (capturing group in split)
            split_re = re.compile(f"({self.special_tokens_pattern})", re.UNICODE)
            parts = split_re.split(text)
        else:
            parts = [text]

        tokens: list[bytes] = []
        for part in parts:
            if not part:
                continue
            # if this part is exactly a special token, preserve it intact
            if self.special_tokens_pattern and split_re.fullmatch(part):
                tokens.append(part.encode("utf-8"))
                continue

            # otherwise run the base tokenizer on this fragment
            found = base_re.findall(part)
            # handle the (unlikely here) case where findall returns tuples when there are groups
            if found and isinstance(found[0], tuple):
                found = ["".join(g) for g in found]
            tokens.extend(m.encode("utf-8") for m in found if m)

        # optional debug print:
        # print([t.decode('utf-8') for t in tokens])
        return tokens


    def _apply_bpe(self, token: bytes) -> list[bytes]:
        # turn token into list of single-byte tokens
        word = [bytes([b]) for b in token]

        if not word:
            return []

        while True:
            # find all adjacent pairs
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            if not pairs:
                break

            # find best pair (lowest rank)
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break

            _, best = min(ranked, key=lambda x: x[0])
            # merge all occurrences of best
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        ids = []
        for tok in self._pre_tokenize(text):
            #print(f"Token: {tok!r}")
            if tok in self.special_token_bytes:
                # special token: use directly
                ids.append(self.bytes_to_id[tok])
            else:
                # normal token: run BPE
                for merged in self._apply_bpe(tok):
                    if merged in self.bytes_to_id:
                        ids.append(self.bytes_to_id[merged])
                    else:
                        # Fallback: split into individual bytes
                        for b in merged:
                            byte_token = bytes([b])
                            ids.append(self.bytes_to_id[byte_token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs
        """
        for text in iterable:
            for tok in self._pre_tokenize(text):
                if tok in self.special_token_bytes:
                    yield self.bytes_to_id[tok]
                else:
                    for merged in self._apply_bpe(tok):
                        if merged in self.bytes_to_id:
                            yield self.bytes_to_id[merged]
                        else:
                            # Fallback: break down into raw bytes
                            for b in merged:
                                byte_token = bytes([b])
                                yield self.bytes_to_id[byte_token]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_seq = b"".join(self.id_to_bytes[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")



"""
Train the BPE Tokenizer
"""
if __name__ == '__main__':

    special_tokens = ["<|endoftext|>"]

    vocab, merges = myself_train_bpe(
        input_path="/Users/clarence_deng/PycharmProjects/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=special_tokens
    )

    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Save vocab.json: {token_id: token_string}
    with open("/Users/clarence_deng/PycharmProjects/assignment1-basics/data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2
        )

    # Save merges.txt: one merge rule per line
    with open("/Users/clarence_deng/PycharmProjects/assignment1-basics/data/merges.txt", "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(f"{first.decode('utf-8', errors='replace')} {second.decode('utf-8', errors='replace')}\n")

