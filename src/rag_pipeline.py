"""
RAG Core Logic for CrediTrust Complaint Chatbot (Task 3)
Single source of truth — imported by app.py and callable from notebooks.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "complaint_chunks"
DEFAULT_TOP_K = 5
GENERATOR_MODEL = "openai/gpt-oss-120b"

# Resolves to <project_root>/vector_store no matter where this is run from
DEFAULT_VECTOR_STORE_PATH = str(Path(__file__).resolve().parent.parent / "vector_store")

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions \
about customer complaints. Use the following retrieved complaint excerpts to formulate \
your answer. If the context doesn't contain the answer, state that you don't have \
enough information. Do not invent details that are not present in the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, vector_store_path=DEFAULT_VECTOR_STORE_PATH, collection_name=COLLECTION_NAME,
                 embedding_model_name=EMBEDDING_MODEL_NAME, generator_model=GENERATOR_MODEL,
                 top_k=DEFAULT_TOP_K):
        self.top_k = top_k
        self.generator_model = generator_model

        print(f"Loading embedding model '{embedding_model_name}'...")
        self.embedder = SentenceTransformer(embedding_model_name)

        print(f"Connecting to vector store at '{vector_store_path}'...")
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"Connected. Collection contains {self.collection.count()} chunks.")

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN not found in environment. Generator calls will fail.")
        self.hf_client = InferenceClient(model=self.generator_model, token=hf_token)

    def retrieve(self, question, top_k=None, product_filter=None):
        k = top_k or self.top_k
        query_embedding = self.embedder.encode([question]).tolist()
        # "All" (or None/empty) means no filter — matches the dropdown in app.py
        where_clause = None
        if product_filter and product_filter != "All":
            where_clause = {"product_category": product_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_clause,
        )
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
        }

    @staticmethod
    def build_prompt(question, retrieved_chunks):
        context = "\n\n---\n\n".join(retrieved_chunks)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, prompt, max_new_tokens=300):
        try:
            response = self.hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Generator error: {e}]"

    def answer(self, question, top_k=None, product_filter=None):
        retrieved = self.retrieve(question, top_k=top_k, product_filter=product_filter)
        prompt = self.build_prompt(question, retrieved["documents"])
        answer_text = self.generate(prompt)

        sources = []
        for doc, meta, dist in zip(retrieved["documents"], retrieved["metadatas"], retrieved["distances"]):
            sources.append({
                "text": doc,
                "complaint_id": meta.get("complaint_id"),
                "product_category": meta.get("product_category"),
                "issue": meta.get("issue"),
                "company": meta.get("company"),
                "distance": dist,
            })

        return {"answer": answer_text, "sources": sources, "prompt": prompt}
    def generate_stream(self, prompt, max_new_tokens=300):
        """Yields the answer token-by-token instead of returning it all at once."""
        try:
            stream = self.hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"[Generator error: {e}]"

    def answer_stream(self, question, top_k=None, product_filter=None):
        """
        Retrieves sources once, then streams the generated answer.
        Yields (partial_answer_so_far, sources_list) — sources are stable
        across all yields, only the answer text grows.
        """
        retrieved = self.retrieve(question, top_k=top_k, product_filter=product_filter)
        prompt = self.build_prompt(question, retrieved["documents"])

        sources = []
        for doc, meta, dist in zip(retrieved["documents"], retrieved["metadatas"], retrieved["distances"]):
            sources.append({
                "text": doc,
                "complaint_id": meta.get("complaint_id"),
                "product_category": meta.get("product_category"),
                "issue": meta.get("issue"),
                "company": meta.get("company"),
                "distance": dist,
            })

        # Yield sources immediately (with empty answer) so the UI shows them right away
        yield "", sources

        partial = ""
        for token in self.generate_stream(prompt):
            partial += token
            yield partial, sources


if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.answer("Why are people unhappy with Credit Cards?")
    print("\nANSWER:\n", result["answer"])
    print("\nTOP SOURCE:\n", result["sources"][0]["text"][:200])