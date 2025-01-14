
import networkx as nx
import numpy as np
from pydantic import PrivateAttr
from langchain.schema import Document, BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings

class NetworkXRetriever(BaseRetriever):
    """
    Ретривер на основе графа NetworkX. 
    """
    _graph: nx.Graph = PrivateAttr()
    _embeddings: HuggingFaceEmbeddings = PrivateAttr()
    _k: int = PrivateAttr(default=3)
    _node_embeddings: dict = PrivateAttr(default_factory=dict)

    def __init__(self, graph: nx.Graph, embeddings: HuggingFaceEmbeddings, k=3, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_embeddings", embeddings)
        object.__setattr__(self, "_k", k)
        object.__setattr__(self, "_node_embeddings", {})

        for node_id, data in graph.nodes(data=True):
            self._node_embeddings[node_id] = data["embedding"]

    def get_relevant_documents(self, query: str):
        query_embedding = self._embeddings.embed_query(query)
        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec) + 1e-9

        scores = []
        for node_id, emb in self._node_embeddings.items():
            d_vec = np.array(emb)
            dot = np.dot(q_vec, d_vec)
            norm_d = np.linalg.norm(d_vec) + 1e-9
            cos_sim = dot / (norm_q * norm_d)
            scores.append((node_id, cos_sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[: self._k]

        docs = []
        for node_id, sim in top_nodes:
            content = self._graph.nodes[node_id]["content"]
            docs.append(Document(
                page_content=content,
                metadata={"score": sim, "node_id": node_id}
            ))
        return docs
