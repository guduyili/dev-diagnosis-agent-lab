"""Qdrant 本地向量库适配层。

这里同时配置 dense embedding 和 BM25 sparse embedding，QdrantVectorStore
再以 HYBRID 模式合并两类召回。该类不负责 parent 文档，它只保存可检索的
child chunk；parent_id 会跟随 child metadata 返回给工具层。
"""

import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels





class VectorDbManager:
    """创建，校验和获取本地的 Qdrant collection。"""
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse


    def __init__(self):
        # QdrantClient(path=...) 使用本地目录持久化；首次创建 collection 前，
        # embedding 对象也会准备模型，可能触发本地模型下载。
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        # 稠密模型
        # dense embedding 模型是 HuggingFaceEmbeddings 的子类，初始化时会下载模型到本地缓存。
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        # 稀疏模型
        # sparse embedding 模型是 FastEmbedSparse 的子类，初始化时会下载模型
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)


    def __dense_vector_size(self):
        """用一个探测文本得到 dense 向量维度 供 collection schema 使用"""
        # 真正执行一次 embedding 不只是读取配置，collection 已存在也会执行
        return len(self.__dense_embeddings.embed_query("test"))


    @staticmethod
    def _collection_vector_size(collection_info):
        # 兼容单向量配置和命名向量字典。字典只取第一项，未按具体向量名核对。
        # 不认识的配置返回 None；下方 existing_size 判断会跳过维度比较。
        vectors_config = collection_info.config.params.vectors
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        if isinstance(vectors_config, dict) and vectors_config:
            first_vector = next(iter(vectors_config.values()))
            if hasattr(first_vector, "size"):
                return first_vector.size

        return None


    def create_collection(self, collection_name):
        """创建 collection，或比较已有 collection 的 dense 向量维度。"""
        # 维度相同是必要条件，不足以证明模型兼容：两个模型都输出 384 维，
        # 其向量空间仍可能不同。这里没有记录模型版本或分块策略供一致性校验。

        expected_size = self.__dense_vector_size()
        if not self.__client.collection_exists(collection_name):
            # sparse_vectors_config 的名字必须与 QdrantVectorStore 初始化时一致。
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=expected_size,distance=qmodels.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ Collection created: {collection_name}")

        else:
            collection_info = self.__client.get_collection(collection_name)
            existing_size = self._collection_vector_size(collection_info)
            if existing_size and existing_size != expected_size:
                raise ValueError(
                    f"Qdrant collection '{collection_name}' has dense vector size "
                    f"{existing_size}, but '{config.DENSE_MODEL}' produces size "
                    f"{expected_size}. Clear and re-index the collection after "
                    "changing embedding models."
                )
            print(f"✓ Collection already exists: {collection_name}")
    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            raise RuntimeError(f"Unable to delete Qdrant collection '{collection_name}'.") from e


    def get_collection(self, collection_name) -> QdrantVectorStore:
        """返回绑定了 dense/sparse embedding 的混合检索对象。"""
        # 这是指向同一 collection 的适配器，不复制数据。HYBRID 的查询融合由
        # langchain_qdrant/Qdrant 实现；此文件没有自定义融合公式或统一概率分数。

        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embeddings,
                sparse_embedding=self.__sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name=config.SPARSE_VECTOR_NAME,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get collection '{collection_name}': {e}")
