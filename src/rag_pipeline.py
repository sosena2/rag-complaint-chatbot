"""
RAG Core Logic for CrediTrust Complaint Chatbot (Task 3)
"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

load_dotenv()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "complaint_chunks"
DEFAULT_TOP_K = 5
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions \
about customer complaints. Use the following retrieved complaint excerpts to formulate \
your answer. If the context doesn't contain the answer, state that you don't have \
enough information. Do not invent details that are not present in the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, vector_store_path="../vector_store", collection_name=COLLECTION_NAME,
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
        where_clause = {"product_category": product_filter} if product_filter else None

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
            response = self.hf_client.text_generation(
                prompt, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True
            )
            return response.strip()
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


if __name__ == "__main__":
    rag = RAGPipeline(vector_store_path="../vector_store")
    result = rag.answer("Why are people unhappy with Credit Cards?")
    print("\nANSWER:\n", result["answer"])
    print("\nTOP SOURCE:\n", result["sources"][0]["text"][:200])