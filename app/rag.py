import os
import glob
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


class RAGPipeline:
    def __init__(self, data_dir: str = "data", chunk_size: int = 800, chunk_overlap: int = 200):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def load_documents(self) -> None:
        patterns = [os.path.join(self.data_dir, "*.txt"), os.path.join(self.data_dir, "*.md")]
        files: List[str] = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

        chunks: List[DocumentChunk] = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            filename = os.path.basename(file_path)
            file_chunks = self._split_into_chunks(text, filename)
            chunks.extend(file_chunks)

        self.chunks = chunks

    def _split_into_chunks(self, text: str, filename: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunk_id = f"{filename}#{index}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                metadata={"filename": filename}
            ))
            if end >= len(text):
                break
            start = end - self.chunk_overlap
            index += 1
        return chunks

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        vectors = [d.embedding for d in response.data]
        return np.array(vectors, dtype="float32")

    def build_index(self) -> None:
        if not self.chunks:
            self.load_documents()

        texts = [c.text for c in self.chunks]
        if not texts:
            self.embeddings = np.empty((0, 0))
            return

        self.embeddings = self._embed_texts(texts)

    def _cosine_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embeddings is None or self.embeddings.size == 0:
            return np.array([])
        # normalize
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        query_norm = np.linalg.norm(query_vec) + 1e-10
        normalized_docs = self.embeddings / doc_norms
        normalized_query = query_vec / query_norm
        sims = normalized_docs @ normalized_query
        return sims

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or self.embeddings.size == 0:
            # no documents loaded
            return []

        query_vec = self._embed_texts([query])[0]
        sims = self._cosine_similarities(query_vec)
        if sims.size == 0:
            return []

        top_indices = np.argsort(-sims)[:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append({
                "id": chunk.id,
                "score": float(sims[idx]),
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
        return results

    def generate_answer(self, question: str, history: List[Dict[str, str]] | None = None, top_k: int = 5) -> Dict[str, Any]:
        if history is None:
            history = []

        retrieved = self.retrieve(question, top_k=top_k)
        context_blocks = []
        for r in retrieved:
            context_blocks.append(f"[{r['metadata'].get('filename')}] {r['text'].strip()}")
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        system_prompt = (
            "You are an AI assistant that answers questions based strictly on the provided context. "
            "If the answer is not in the context, say you don't know or that it isn't specified. "
            "Be concise and clear."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if context_blocks:
            messages.append({
                "role": "system",
                "content": f"Here is the context:\n\n{context_text}"
            })

        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        messages.append({"role": "user", "content": question})

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )

        answer = completion.choices[0].message.content

        sources = [
            {
                "id": r["id"],
                "score": r["score"],
                "metadata": r["metadata"],
            }
        for r in retrieved
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


# Create a global pipeline instance that can be imported in main.py
pipeline = RAGPipeline()
