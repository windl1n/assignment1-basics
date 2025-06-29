from collections import defaultdict
import os
import re
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def merge(token: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """
    Merge the pair of tokens in the token list and return a new token list.
    from https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_01.py#L533
    """
    new_token = []
    i = 0
    while i < len(token):
        if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
            new_token.append(new_index)  # Use the new index for the merged token
            i += 2  # Skip the next token since it's merged
        else:
            new_token.append(token[i])
            i += 1
    return new_token

def merge_dict(token_counts: dict[list[int], int], pair: tuple[int, int], new_index):
    new_token: dict[list[int], int] = defaultdict(int)
    for token, count in token_counts.items():
        if len(token) == 1:
            new_token[token] += count
        else:
            token = merge(token, pair, new_index)
            new_token[token] += count
    return new_token


def pre_tokenization(corpus: str, vocab: dict[int, bytes], merges: dict[tuple[int, int], int], vocab_size: int = 1000):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    token_counts: dict[list[int], int] = defaultdict(int) # merge this after got a new vocab 
    for token_match in re.finditer(PAT, corpus):
        token_counts[list(map(int, token_match.group().encode("utf-8")))] += 1    # {[, ,]: 1}
    
    while len(vocab) < vocab_size:
        # {(l, o): 1, (o, w): 1} everytime recount this
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for token, count in token_counts.items():
            for b1, b2 in zip(token, token[1:]):
                pair_counts[(b1, b2)] += count                       # (l, o), (o, w)
        pair = max(pair_counts)
        index1, index2 = pair

        new_index = len(vocab)
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        token_counts = merge_dict(token_counts, pair, new_index)
    return vocab, merges
    

def tokenize(file_path: str, num_processes: int = 1):

    merges: dict[tuple[int, int], int] = {} # (old1, old2) -> new
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # Initialize with single byte tokens
    ## Usage
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes. 
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            vocab, merges = pre_tokenization(chunk, vocab, merges, vocab_size=1000)
    print(f"Vocabulary size: {len(vocab)}")
    # Save the vocab and merges to files
    with open("vocab.txt", "w") as vocab_file:
        for index, token in sorted(vocab.items()):
            vocab_file.write(f"{index}\t{token.decode('utf-8')}\n")
    with open("merges.txt", "w") as merges_file:
        for (old1, old2), new_index in sorted(merges.items()):
            merges_file.write(f"{old1} {old2} -> {new_index}\n")

if __name__ == "__main__":
    # Example usage
    file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 4  # Adjust based on your system
    tokenize(file_path, num_processes)