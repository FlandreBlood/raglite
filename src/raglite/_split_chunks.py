"""Split a document into semantic chunks."""

import re

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from raglite._typing import FloatMatrix

def split_chunks(
    sentences: list[str],
    sentence_embeddings: FloatMatrix,
    sentence_window_size: int = 3,
    max_size: int = 1440,
) -> tuple[list[str], list[FloatMatrix]]:
    """Split sentences into optimal semantic chunks with corresponding sentence embeddings."""
    grouped_sentences = []
    grouped_embeddings = []
    current_chunk = []
    current_embeddings = []
    current_heading = "Default Heading"  # 修改默认第一个 heading

    for sentence, embedding in zip(sentences, sentence_embeddings):
        # 去除特殊内容 [UNK]
        sentence = sentence.replace("[UNK]", "")
        # 去除句子中的所有中英文空格
        sentence = sentence.replace(" ", "").replace("\u3000", "")
        # 如果是标题
        if sentence.startswith("#"):
            # 如果当前 chunk 不为空，先保存
            if current_chunk:
                grouped_sentences.append((current_heading, "".join(current_chunk)))
                grouped_embeddings.append(np.array(current_embeddings))
                current_chunk = []
                current_embeddings = []
            current_heading = sentence.strip() or "Default Heading"  # 确保标题有效
        else:
            current_chunk.append(sentence)
            current_embeddings.append(embedding)

    # 添加最后一个 chunk
    if current_chunk:
        grouped_sentences.append((current_heading, "".join(current_chunk)))
        grouped_embeddings.append(np.array(current_embeddings))

    # 分割内容，确保不超过 max_size
    final_chunks = []
    final_embeddings = []

    for heading, content in grouped_sentences:
        if len(content) > max_size:
            paragraphs = content.split("\n\n")
            for para in paragraphs:
                if len(para) > max_size:
                    sentences = re.split(r"(?<=[。！？])", para)  # 按句子进一步分割
                    temp_chunk = ""
                    for s in sentences:
                        if len(temp_chunk) + len(s) > max_size:
                            final_chunks.append((heading, temp_chunk))
                            temp_chunk = s
                        else:
                            temp_chunk += s
                    if temp_chunk:
                        final_chunks.append((heading, temp_chunk))
                else:
                    final_chunks.append((heading, para))
        else:
            final_chunks.append((heading, content))

    # 将元组转换回字符串格式
    formatted_chunks = [
        f"{heading}\n\n{content}" if heading else content for heading, content in final_chunks
    ]

    return formatted_chunks, grouped_embeddings