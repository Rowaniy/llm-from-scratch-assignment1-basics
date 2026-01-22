from .pretokenization import pretokenize
from collections import defaultdict
import heapq
import regex as re
import json
from typing import Iterable, Iterator
from dataclasses import dataclass

@dataclass(order=False)
class Mergecompare:
    """包装类，用于定义 BPE 合并比较的堆排序逻辑"""
    freq: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other):

        # 1. 先比较频率
        if self.freq != other.freq:
            return self.freq > other.freq
        
        # 2. 再比较字典序（字典序大的优先）
        return self.pair > other.pair

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE tokenizer.

    Args:
        input_path (str): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special 
                                   tokens do not otherwise affect BPE training.

    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in 
                                 the vocabulary) to bytes (token bytes).

        **merges** : *(list[tuple[bytes, bytes]])*
         
           A list of BPE merges produced from training. Each list item is a tuple of bytes (\<token1\>, \<token2\>),representing that \<token1\> was merged with \<token2\>. The merges should be ordered by order of creation.
    """

    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode('utf-8') for i, token in enumerate(special_tokens)})
    len_vocab = len(vocab)
    merges = []
    
    # 预切分文本，得到预token的计数
    pretoken_counts = pretokenize(input_path, special_tokens, num_processes=12)
    
    # 使用链表节点来表示每个位置
    class Node:
        __slots__ = ['value', 'prev', 'next', 'word_freq']
        def __init__(self, value, word_freq):
            self.value = value
            self.word_freq = word_freq
            self.prev = None
            self.next = None
    
    # pair_to_nodes: 记录每个 pair 的所有左节点
    pair_to_nodes = defaultdict(set)
    
    for token_bytes, count in pretoken_counts.items():
        if len(token_bytes) < 2:
            continue
        
        word_freq = {'count': count}
        
        head = Node(bytes([token_bytes[0]]), word_freq)
        prev_node = head
        for i in range(1, len(token_bytes)):
            curr_node = Node(bytes([token_bytes[i]]), word_freq)
            prev_node.next = curr_node
            curr_node.prev = prev_node
            
            pair = (prev_node.value, curr_node.value)
            pair_to_nodes[pair].add(prev_node)
            prev_node = curr_node
    
    # pair_counts: 每个 pair 的总频率
    pair_counts = {}
    for pair, nodes in pair_to_nodes.items():
        pair_counts[pair] = sum(node.word_freq['count'] for node in nodes)
    
    # 初始化堆
    pair_heap = [Mergecompare(freq, pair) for pair, freq in pair_counts.items()]
    heapq.heapify(pair_heap)

    # 开始合并
    for _ in range(vocab_size - len_vocab):
        best_pair = None
        while pair_heap:
            candidate = heapq.heappop(pair_heap)
            freq = candidate.freq
            pair = candidate.pair

            # 检查该pair是否仍然有效
            if pair in pair_counts and pair_counts[pair] == freq:
                best_pair = pair
                break
        
        if best_pair is None:
            break
        
        # 执行合并
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len_vocab] = merged_token
        len_vocab += 1
        
        # 获取包含此 pair 的所有左节点
        nodes_to_process = list(pair_to_nodes[best_pair])
        
        for node1 in nodes_to_process:
            node2 = node1.next
            if node2 is None:
                continue
            
            word_freq = node1.word_freq['count']
            
            # 更新左侧相邻 pair 的频率及映射关系
            if node1.prev:
                left = node1.prev
                old_left_pair = (left.value, node1.value)
                pair_counts[old_left_pair] -= word_freq
                heapq.heappush(pair_heap, Mergecompare(pair_counts[old_left_pair], old_left_pair))
                
                pair_to_nodes[old_left_pair].discard(left)
                new_left_pair = (left.value, merged_token)
                pair_to_nodes[new_left_pair].add(left)
                if new_left_pair not in pair_counts:
                    pair_counts[new_left_pair] = 0
                pair_counts[new_left_pair] += word_freq
                heapq.heappush(pair_heap, Mergecompare(pair_counts[new_left_pair], new_left_pair))
            
            # 更新右侧相邻 pair 的频率及映射关系
            if node2.next:
                right = node2.next
                old_right_pair = (node2.value, right.value)
                pair_counts[old_right_pair] -= word_freq
                heapq.heappush(pair_heap, Mergecompare(pair_counts[old_right_pair], old_right_pair))
                
                pair_to_nodes[old_right_pair].discard(node2)
                new_right_pair = (merged_token, right.value)
                pair_to_nodes[new_right_pair].add(node1)
                if new_right_pair not in pair_counts:
                    pair_counts[new_right_pair] = 0
                pair_counts[new_right_pair] += word_freq
                heapq.heappush(pair_heap, Mergecompare(pair_counts[new_right_pair], new_right_pair))
            
            # 链表合并：node1、node2合成 merged_token
            node1.value = merged_token
            node1.next = node2.next
            if node2.next:
                node2.next.prev = node1
        
        # 删除被合并 pair 的所有统计
        del pair_counts[best_pair]
        del pair_to_nodes[best_pair]

    return vocab, merges

        
class Tokenizer:
    """
    BPE (Byte-Pair Encoding) Tokenizer
    
    Implements encoding and decoding of text using a trained BPE vocabulary and merges.
    Supports special tokens that can be appended to the vocabulary.
    """
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, mapping token IDs to token bytes.
            merges (list[tuple[bytes, bytes]]): List of BPE merges in order of creation.
                                                Each tuple is (token1, token2) indicating token1 
                                                was merged with token2.
            special_tokens (list[str] | None): Optional list of special tokens to add to vocabulary
                                               if they aren't already present.
        """
        # 保存和扩展vocab
        self.vocab = vocab.copy()
        
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in self.vocab.values():
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
        
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merges = merges
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)  # 处理正则表达式的贪婪匹配
            self.special_pattern = f"({"|".join(re.escape(token) for token in special_tokens)})"     # 切分之后保留 special_tokens
        else:
            self.special_pattern = None
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 建立 bytes -> ID 的反向查找表
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        # 建立合并对的优先级查找表
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}


    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from serialized vocabulary 
        and merges files (in the same format that BPE training code outputs).
        
        Args:
            vocab_filepath (str): Path to the serialized vocabulary file (JSON format).
            merges_filepath (str): Path to the serialized merges file (text format).
            special_tokens (list[str] | None): Optional list of special tokens.
            
        Returns:
            Tokenizer: A new Tokenizer instance constructed from the files.
        """
        
        # 读取vocab文件
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_data = json.load(vf)
            vocab = {int(k): v.encode('latin1') for k, v in vocab_data.items()}
        
        # 读取merges文件
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            for line in mf:
                token1, token2 = line.strip().split()
                merges.append((token1.encode('latin1'), token2.encode('latin1')))
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text (str): The input text string to encode.
            
        Returns:
            list[int]: List of token IDs representing the encoded text.
        """
        # 处理 special_tokens
        if self.special_pattern:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]
        
        result_ids = []
        
        for chunk in chunks:
            if not chunk:
                continue
            
            # 如果是特殊 token，直接转换为 ID
            if chunk in self.special_tokens:
                chunk_bytes = chunk.encode('utf-8')
                if chunk_bytes in self.bytes_to_id:
                    result_ids.append(self.bytes_to_id[chunk_bytes])
                continue
            
            # 预切分
            pre_tokens = re.finditer(self.PAT, chunk)
            
            for pre_token_match in pre_tokens:
                pre_token = pre_token_match.group(0)
                parts = [bytes([b]) for b in pre_token.encode("utf-8")]
                
                while len(parts) >= 2:
                    # 找到当前 parts 中 rank 最小（最先合并）的 pair
                    min_rank = float("inf")
                    min_idx = -1
                    
                    for i in range(len(parts) - 1):
                        pair = (parts[i], parts[i + 1])
                        rank = self.merges_dict.get(pair, float("inf"))
                        
                        if rank < min_rank:
                            min_rank = rank
                            min_idx = i
                    
                    # 如果找不到可合并的 pair，退出循环
                    if min_rank == float("inf"):
                        break
                    
                    # 合并
                    parts[min_idx] = parts[min_idx] + parts[min_idx + 1]
                    parts.pop(min_idx + 1)
                
                # 转换为 token IDs
                for part in parts:
                    if part in self.bytes_to_id:
                        result_ids.append(self.bytes_to_id[part])
        
        return result_ids
                


    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-efficient tokenization 
        of large files that cannot be directly loaded into memory.
        
        Args:
            iterable (Iterable[str]): An iterable of strings (e.g., file handle iterating lines).
            
        Yields:
            int: Token IDs one at a time, lazily generated.
        """
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids
        
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        
        Args:
            ids (list[int]): List of token IDs to decode.
            
        Returns:
            str: The decoded text string.
        """
        decoded_text = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_text.decode('utf-8', errors='ignore')
