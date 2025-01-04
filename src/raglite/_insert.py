"""Index documents."""

from pathlib import Path

import numpy as np
from sqlalchemy.engine import make_url
from sqlmodel import Session, select
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, ChunkEmbedding, Document, IndexMetadata, create_database_engine
from raglite._embed import embed_sentences, sentence_embedding_type
from raglite._markdown import document_to_markdown
from raglite._split_chunks import split_chunks
from raglite._split_sentences import split_sentences
from raglite._typing import DocumentId, FloatMatrix


def _create_chunk_records(
    document_id: DocumentId,
    chunks: list[str],
    chunk_embeddings: list[FloatMatrix],
    config: RAGLiteConfig,
) -> tuple[list[Chunk], list[list[ChunkEmbedding]]]:
    """从块和它们的嵌入创建块和块嵌入记录。"""
    chunk_records = []
    chunk_embedding_records = []
    
    for i, (chunk, embeddings) in enumerate(zip(chunks, chunk_embeddings)):
        # 解析标题和正文
        if "\n\n" in chunk:
            heading, body = chunk.split("\n\n", 1)
            # 去掉标题中的#号和斜杠
            heading = heading.replace("#", "").replace("\\", "")
        else:
            heading, body = "", chunk
        
        # 创建块记录
        chunk_record = Chunk.from_body(
            document_id=document_id,
            index=i,
            headings=heading,
            body=body,
        )
        chunk_records.append(chunk_record)
        
        # 创建块嵌入记录
        chunk_embedding_record_list = [
            ChunkEmbedding(chunk_id=chunk_record.id, embedding=embedding)
            for embedding in embeddings
        ]
        chunk_embedding_records.append(chunk_embedding_record_list)
    
    return chunk_records, chunk_embedding_records

def insert_document(doc_path: Path, *, config: RAGLiteConfig | None = None) -> None:  # noqa: PLR0915
    """将文档插入数据库并更新索引。"""
    # 如果没有提供配置，则使用默认配置。
    config = config or RAGLiteConfig()
    db_backend = make_url(config.db_url).get_backend_name()
    # 将文档预处理为块和块嵌入。
    with tqdm(total=5, unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("初始化数据库")
        engine = create_database_engine(config)
        pbar.update(1)
        pbar.set_description("转换为Markdown")
        doc = document_to_markdown(doc_path)
        pbar.update(1)
        pbar.set_description("分割句子")
        sentences = split_sentences(doc, max_len=config.chunk_max_size)
        pbar.update(1)
        pbar.set_description("Embedding句子")
        sentence_embeddings = embed_sentences(sentences, config=config)
        pbar.update(1)
        pbar.set_description("分割块")
        chunks, chunk_embeddings = split_chunks(
            sentences=sentences,
            sentence_embeddings=sentence_embeddings,
            sentence_window_size=config.embedder_sentence_window_size,
            max_size=config.chunk_max_size,
        )
        pbar.update(1)
    # 创建并存储块记录。
    with Session(engine) as session:
        # 将文档添加到文档表中。
        document_record = Document.from_path(doc_path)
        if session.get(Document, document_record.id) is None:
            session.add(document_record)
            session.commit()
        # 创建要插入到块表中的块记录。
        chunk_records, chunk_embedding_records = _create_chunk_records(
            document_record.id, chunks, chunk_embeddings, config
        )
        # 存储块和块嵌入记录。
        for chunk_record, chunk_embedding_record_list in tqdm(
            zip(chunk_records, chunk_embedding_records, strict=True),
            desc="插入块",
            total=len(chunk_records),
            unit="块",
            dynamic_ncols=True,
            
        ):
            if session.get(Chunk, chunk_record.id) is not None:
                continue
            session.add(chunk_record)
            session.add_all(chunk_embedding_record_list)
            session.commit()
    # 手动更新SQLite的向量搜索块索引。
    if db_backend == "sqlite":
        from pynndescent import NNDescent

        with Session(engine) as session:
            # 从数据库获取向量搜索块索引，或创建一个新的。
            index_metadata = session.get(IndexMetadata, "default") or IndexMetadata(id="default")
            chunk_ids = index_metadata.metadata_.get("chunk_ids", [])
            chunk_sizes = index_metadata.metadata_.get("chunk_sizes", [])
            # 获取未索引的块。
            unindexed_chunks = list(session.exec(select(Chunk).offset(len(chunk_ids))).all())
            if not unindexed_chunks:
                return
            # 将未索引的块嵌入组装成一个NumPy数组。
            unindexed_chunk_embeddings = [chunk.embedding_matrix for chunk in unindexed_chunks]
            X = np.vstack(unindexed_chunk_embeddings)  # noqa: N806
            # 索引未索引的块。
            with tqdm(
                total=len(unindexed_chunks),
                desc="索引块",
                unit="块",
                dynamic_ncols=True,
            ) as pbar:
                # 拟合或更新ANN索引。
                if len(chunk_ids) == 0:
                    nndescent = NNDescent(X, metric=config.vector_search_index_metric)
                else:
                    nndescent = index_metadata.metadata_["index"]
                    nndescent.update(X)
                # 准备ANN索引，以便能够处理不在训练集中的查询向量。
                nndescent.prepare()
                # 更新索引元数据，并通过重新创建字典将其标记为脏。
                index_metadata.metadata_ = {
                    **index_metadata.metadata_,
                    "index": nndescent,
                    "chunk_ids": chunk_ids + [c.id for c in unindexed_chunks],
                    "chunk_sizes": chunk_sizes + [len(em) for em in unindexed_chunk_embeddings],
                }
                # 存储更新后的向量搜索块索引。
                session.add(index_metadata)
                session.commit()
                pbar.update(len(unindexed_chunks))
