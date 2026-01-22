import os
from typing import BinaryIO
import regex as re
from collections import Counter
from multiprocessing import Pool


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.

    May return fewer chunks if the boundaries end up overlapping.

    为 pretokenize() 找到合适的块边界，确保不会在特殊 token 处拆分。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def process_chunk(input_path: str, special_tokens: list[str], start: int, end: int) -> Counter:
    """
    Process a chunk of the input file from start to end byte offsets.

    Returns a Counter of pre-token occurrences in that chunk.

    为 pretokenize() 处理文件的一个块。
    """

    counts = Counter()

    # 这个函数会在各自的子进程中运行
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start).decode("utf-8", errors="ignore")

        # 先处理特殊token，切分文本
        pattern = "|".join(re.escape(token) for token in special_tokens)
        small_chunks = [chunk for chunk in re.split(pattern, data) if chunk] 

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for small_chunk in small_chunks:
            tokens = re.finditer(PAT, small_chunk)
            for match in tokens:
                token_bytes = match.group(0).encode('utf-8')    # 转换为字节流
                counts[token_bytes] += 1

        return counts # 返回这个块的统计结果


def pretokenize(input_path: str, special_tokens: list[str], num_processes: int=4) -> Counter:
    """
    Pre-tokenize the text with chunks in parallel.
    """

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 使用多进程处理各个大块
    tasks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        tasks.append((input_path, special_tokens, start, end))
    with Pool(processes=len(boundaries) - 1) as pool:
        results = pool.starmap(process_chunk, tasks)

    final_counts = Counter()
    for res in results:
        final_counts.update(res)
    return final_counts