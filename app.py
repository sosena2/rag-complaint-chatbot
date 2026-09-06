import gradio as gr
from src.rag_pipeline import RAGPipeline

# ── setup ──────────────────────────────────────────────────────────────────
rag = RAGPipeline()

PRODUCT_OPTIONS = ["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

# ── helpers ────────────────────────────────────────────────────────────────
def format_sources(sources):
    if not sources:
        return "*Sources will appear here after you ask a question.*"
    sources_md = ""
    for i, s in enumerate(sources, 1):
        sources_md += (
            f"**Source {i}** — "
            f"Product: `{s.get('product_category', 'N/A')}` | "
            f"Issue: `{s.get('issue', 'N/A')}` | "
            f"Complaint ID: `{s.get('complaint_id', 'N/A')}`\n\n"
            f"> {s['text'][:300]}...\n\n---\n\n"
        )
    return sources_md

# ── core logic (now a generator, for streaming) ────────────────────────────
def answer_question(question, product_filter="All", top_k=5):
    if not question.strip():
        yield "", "*Sources will appear here after you ask a question.*"
        return

    for partial_answer, sources in rag.answer_stream(question, top_k=int(top_k), product_filter=product_filter):
        yield partial_answer, format_sources(sources)

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