import os
import json
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from .schemas import RerankBatchResult

class ResumeRetriever:
    """Multi-stage retriever supporting Small-to-Big expansion and high-precision Re-ranking."""
    def __init__(
        self,
        resumes: Dict[str, Dict[str, Any]],
        embeddings: OpenAIEmbeddings,
        llm: ChatOpenAI,
        cache_path: str = "cache/resume_vectorstore.json"
    ):
        self.resumes = resumes
        self.embeddings = embeddings
        self.llm = llm
        self.cache_path = cache_path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.vectorstore = InMemoryVectorStore(embeddings)
        self.reranker_llm = self.llm.with_structured_output(RerankBatchResult)
        self._build_index()

    def _build_index(self):
        docs = []
        for cid, info in self.resumes.items():
            chunks = self.splitter.split_text(info["text"])
            for idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "candidate_id": cid,
                            "name": info["name"],
                            "role": info["role"],
                            "chunk_index": idx
                        }
                    )
                )

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    cached_data = json.load(f)
                cached_docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in cached_data["docs"]]
                self.vectorstore.add_documents(cached_docs)
                return
            except Exception:
                pass

        self.vectorstore.add_documents(docs)
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            to_cache = {"docs": [{"text": d.page_content, "metadata": d.metadata} for d in docs]}
            with open(self.cache_path, "w") as f:
                json.dump(to_cache, f)
        except Exception:
            pass

    def naive_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def rerank(self, query: str, chunks: List[Document], top_n: int = 4) -> List[Document]:
        chunk_listing = "\n\n".join(
            f"[{i}] ({c.metadata['name']} - {c.metadata['role']}): {c.page_content}"
            for i, c in enumerate(chunks)
        )
        prompt = (
            f"Recruiter Query: {query}\n\n"
            "Score each numbered resume chunk below from 0 to 10 based on whether it directly "
            "provides concrete evidence answering the query. Differentiate exact hands-on experience from passive mentions.\n\n"
            f"{chunk_listing}"
        )
        res = self.reranker_llm.invoke(prompt)
        score_map = {s.index: s.score for s in res.scores}
        ranked = sorted(enumerate(chunks), key=lambda x: score_map.get(x[0], 0), reverse=True)
        return [chunk for _, chunk in ranked[:top_n]]

    def small_to_big_retrieval(self, query: str, k: int = 4, fetch_k: int = 15) -> Tuple[str, List[str]]:
        """Over-fetches candidate chunks, re-ranks them, and traces back to full parent resumes."""
        overfetched = self.vectorstore.similarity_search(query, k=fetch_k)
        top_chunks = self.rerank(query, overfetched, top_n=k)
        seen_ids = list(dict.fromkeys(c.metadata["candidate_id"] for c in top_chunks))
        
        context = "\n\n".join(
            f"--- Candidate: {self.resumes[cid]['name']} ({self.resumes[cid]['role']}) ---\n{self.resumes[cid]['text']}"
            for cid in seen_ids
        )
        return context, seen_ids
