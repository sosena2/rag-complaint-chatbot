import os
import gradio as gr
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ── setup ──────────────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")

client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_collection(name="complaint_chunks")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions \
about customer complaints. Use the following retrieved complaint excerpts to formulate \
your answer. If the context doesn't contain the answer, state that you don't have \
enough information. Do not invent details that are not present in the context.

Context:
{context}

Question: {question}

Answer:"""

PRODUCT_OPTIONS = ["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

# ── core logic ─────────────────────────────────────────────────────────────
def retrieve(question, top_k=5, product_filter=None):
    query_embedding = embedder.encode([question]).tolist()
    where = {"product_category": product_filter} if product_filter and product_filter != "All" else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where
    )
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
    }

def build_prompt(question, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)

def generate(prompt, max_new_tokens=300):
    try:
        response = hf_client.text_generation(
            prompt, max_new_tokens=max_new_tokens, temperature=0.3, do_sample=True
        )
        return response.strip()
    except Exception as e:
        return f"[Generator error: {e}]"

def answer_question(question, product_filter="All", top_k=5):
    if not question.strip():
        return "", ""

    retrieved = retrieve(question, top_k=int(top_k), product_filter=product_filter)
    prompt = build_prompt(question, retrieved["documents"])
    answer_text = generate(prompt)

    # format sources
    sources_md = ""
    for i, (doc, meta) in enumerate(zip(retrieved["documents"], retrieved["metadatas"]), 1):
        sources_md += (
            f"**Source {i}** — "
            f"Product: `{meta.get('product_category', 'N/A')}` | "
            f"Issue: `{meta.get('issue', 'N/A')}` | "
            f"Complaint ID: `{meta.get('complaint_id', 'N/A')}`\n\n"
            f"> {doc[:300]}...\n\n---\n\n"
        )

    return answer_text, sources_md

# ── UI ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="CrediTrust Complaint Analyzer", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏦 CrediTrust Complaint Analyzer
    Ask plain-English questions about customer complaints across all product lines.
    The system retrieves real complaint excerpts and generates a synthesized answer.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Why are people unhappy with Credit Cards?",
                lines=2
            )
        with gr.Column(scale=1):
            product_filter = gr.Dropdown(
                choices=PRODUCT_OPTIONS,
                value="All",
                label="Filter by Product"
            )
            top_k_slider = gr.Slider(
                minimum=3, maximum=10, value=5, step=1,
                label="Number of sources to retrieve"
            )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    gr.Markdown("### Answer")
    answer_box = gr.Textbox(
        label="Generated Answer",
        lines=6,
        interactive=False
    )

    gr.Markdown("### Retrieved Sources")
    sources_box = gr.Markdown(value="*Sources will appear here after you ask a question.*")

    # example questions
    gr.Examples(
        examples=[
            ["Why are people unhappy with Credit Cards?", "Credit Card", 5],
            ["What are the most common complaints about money transfers?", "Money Transfer", 5],
            ["Are customers reporting unauthorized charges?", "All", 5],
            ["What issues do customers have with personal loans?", "Personal Loan", 5],
            ["Are there complaints about savings account fees?", "Savings Account", 5],
        ],
        inputs=[question_box, product_filter, top_k_slider],
        label="Example Questions"
    )

    # wire up buttons
    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, product_filter, top_k_slider],
        outputs=[answer_box, sources_box]
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, product_filter, top_k_slider],
        outputs=[answer_box, sources_box]
    )

    clear_btn.click(
        fn=lambda: ("", "*Sources will appear here after you ask a question.*", ""),
        inputs=[],
        outputs=[answer_box, sources_box, question_box]
    )

if __name__ == "__main__":
    demo.launch()